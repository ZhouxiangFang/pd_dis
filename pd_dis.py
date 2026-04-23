"""
PD Disaggregation via vLLM NixlConnector (two nodes, one GPU each)
===================================================================

This script is launched by srun on BOTH nodes simultaneously.
Each node detects its role from SLURM_PROCID:

  Rank 0 (prefill node)
  ─────────────────────
  • Starts vLLM on GPU 0, port 8100, as KV producer.
  • Binds the NIXL side-channel listener on its own hostname.
  • Blocks until vLLM exits (os.execvp — no Python overhead after launch).

  Rank 1 (decode node)
  ─────────────────────
  • Starts vLLM on GPU 0, port 8200, as KV consumer.
    • Binds NIXL side-channel listener on its own hostname.
  • Waits for both vLLM servers to be healthy.
    • Uses built-in two-phase prefill→decode requests.
    • Sends prompts and prints results.
  • On exit, terminates decode vLLM, proxy, and cancels the SLURM job
    (which also terminates rank 0 / the prefill node).

KV cache transport
──────────────────
NixlConnector uses NIXL + UCX for data transport (not NCCL).
UCX env vars (UCX_TLS, UCX_NET_DEVICES, UCX_MEMTYPE_CACHE) are set in
pd_dis.sh and inherited here via srun.

Side-channel
────────────
VLLM_NIXL_SIDE_CHANNEL_HOST / _PORT: lightweight TCP socket used ONLY for
the initial NIXL handshake (memory-descriptor exchange).  NOT the data path.
Each node binds this to its own hostname on port 5559.

PREFILL_HOST
────────────
Set by pd_dis.sh before srun and exported into the environment of both tasks:
  export PREFILL_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

Prompts
───────
Read from prompts.txt (same directory as this script).
One prompt per line; lines starting with # and blank lines are ignored.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from methods.attn_pruning import PruneConfig, prune_prompt

# ---------------------------------------------------------------------------
# Ports & constants
# ---------------------------------------------------------------------------
PREFILL_PORT           = 8100
DECODE_PORT            = 8200
NIXL_SIDE_CHANNEL_PORT = 5559   # TCP, NIXL handshake only — not the data path

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# NixlConnector KV-transfer config
#   kv_role: prefill="kv_producer", decode="kv_consumer"
#            (important for vLLM>=0.18 NIXL side-channel behavior)
#   kv_buffer_device="cuda" — keep transfer buffer in GPU VRAM (faster)
# ---------------------------------------------------------------------------
def make_kv_config(kv_role: str) -> str:
    return json.dumps({
        "kv_connector": "NixlConnector",
        "kv_role": kv_role,
        "kv_buffer_device": "cuda",
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        print(f"[ERROR] Prompts file not found: {path}", file=sys.stderr)
        sys.exit(1)
    prompts = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not prompts:
        print(f"[ERROR] No prompts found in {path}", file=sys.stderr)
        sys.exit(1)
    return prompts


def wait_for_health(host: str, port: int, label: str,
                    poll: float = 3.0, timeout: float = 1200.0) -> None:
    url = f"http://{host}:{port}/health"
    print(f"[{label}] Waiting for {url} ...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            print(f"[{label}] Ready.")
            return
        except Exception:
            time.sleep(poll)
    print(f"[{label}] ERROR: did not become healthy within {timeout:.0f}s",
          file=sys.stderr)
    sys.exit(1)


def http_post(url: str, body: dict, timeout: float = 600) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_stream_completion(url: str, body: dict, timeout: float = 600) -> dict:
    """
    Stream a /v1/completions request and measure decode-time metrics.

    Returns:
      {
        "text": str,
        "usage": dict,
        "ttft": float | None,           # seconds to first emitted text chunk
        "total_time": float,            # total streaming wall time
      }
    """
    stream_body = {
        **body,
        "stream": True,
        # Best effort: include usage in final chunk when supported.
        "stream_options": {"include_usage": True},
    }

    data = json.dumps(stream_body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    usage: dict = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue

            payload_str = line[5:].strip()
            if not payload_str or payload_str == "[DONE]":
                continue

            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]

            choices = payload.get("choices") or []
            if choices and isinstance(choices[0], dict):
                delta_text = choices[0].get("text") or ""
                if delta_text:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(delta_text)

    total_time = time.perf_counter() - t0
    return {
        "text": "".join(text_parts),
        "usage": usage,
        "ttft": ttft,
        "total_time": total_time,
    }


def extract_kv_transfer_params(response: dict) -> dict | None:
    """Best-effort extraction of kv_transfer_params across response shapes."""
    kv_params = response.get("kv_transfer_params")
    if kv_params:
        return kv_params

    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        kv_params = choices[0].get("kv_transfer_params")
        if kv_params:
            return kv_params
        extra = choices[0].get("extra")
        if isinstance(extra, dict):
            kv_params = extra.get("kv_transfer_params")
            if kv_params:
                return kv_params

    return None


def two_phase_disagg_completion(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    prefill_host: str,
    timeout: float = 600,
) -> dict:
    """
    Built-in fallback for disaggregated serving without toy_proxy_server.py:
      1) prefill on node 0 with do_remote_decode=True
      2) decode on node 1 with returned kv_transfer_params
    """
    _t_prefill_start = time.perf_counter()
    prefill_resp = http_post(
        f"http://{prefill_host}:{PREFILL_PORT}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": temperature,
            "kv_transfer_params": {"do_remote_decode": True},
        },
        timeout=timeout,
    )
    prefill_http_time = time.perf_counter() - _t_prefill_start

    kv_params = extract_kv_transfer_params(prefill_resp)
    if not kv_params:
        # Some builds may complete remote decode directly for this request.
        # In that case, return the prefill response if it already looks like
        # a normal completion payload.
        choices = prefill_resp.get("choices")
        if isinstance(choices, list) and choices:
            return prefill_resp
        raise RuntimeError(
            "Prefill response did not include kv_transfer_params; "
            "cannot continue disaggregated decode. "
            f"Response keys: {sorted(prefill_resp.keys())}"
        )

    decode_stream = http_stream_completion(
        f"http://localhost:{DECODE_PORT}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "kv_transfer_params": kv_params,
        },
        timeout=timeout,
    )

    return {
        "choices": [{"text": decode_stream["text"]}],
        "usage": decode_stream["usage"],
        "metrics": {
            "ttft": decode_stream["ttft"],
            "total_time": decode_stream["total_time"],
            "prefill_http_time": prefill_http_time,
        },
    }


@functools.lru_cache(maxsize=1)
def optional_vllm_serve_flags() -> list[str]:
    """Return optional vLLM serve flags supported by this installed version."""
    try:
        probe = subprocess.run(
            ["vllm", "serve", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        help_text = f"{probe.stdout}\n{probe.stderr}"
    except FileNotFoundError:
        return []

    flags: list[str] = []
    if "--disable-log-requests" in help_text:
        flags.append("--disable-log-requests")
    return flags


def vllm_cmd(args: argparse.Namespace, port: int, kv_config: str) -> list[str]:
    """Build the common vllm serve command (port-agnostic parts)."""
    cmd = [
        "vllm", "serve", args.model,
        "--host",                   "0.0.0.0",
        "--port",                   str(port),
        "--max-model-len",          str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--block-size",             str(args.block_size),
        "--dtype",                  "float16",
        "--enforce-eager",
        "--kv-transfer-config",     kv_config,
    ]
    cmd.extend(optional_vllm_serve_flags())
    return cmd


# ---------------------------------------------------------------------------
# Rank 0 — Prefill node
# ---------------------------------------------------------------------------

def run_prefill(args: argparse.Namespace, my_host: str) -> None:
    """
    Launch vLLM as the prefill (KV producer) instance, then hand over
    to it via os.execvp so this Python process is fully replaced.
    Rank 0 blocks here until the job is cancelled by rank 1.
    """
    kv_config = make_kv_config("kv_producer")
    print(f"[Prefill] host={my_host}  port={PREFILL_PORT}  model={args.model}")
    print(f"[Prefill] NIXL side-channel: {my_host}:{NIXL_SIDE_CHANNEL_PORT}")
    print(f"[Prefill] KV config: {kv_config}")

    # Each node has exactly one GPU — CUDA_VISIBLE_DEVICES=0 is correct.
    os.environ.update({
        "CUDA_VISIBLE_DEVICES":        "0",
        "VLLM_KV_CACHE_LAYOUT":        "HND",
        # Bind the side-channel listener to this node's own hostname so the
        # decode node can reach it across the network.
        "VLLM_NIXL_SIDE_CHANNEL_HOST": my_host,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(NIXL_SIDE_CHANNEL_PORT),
    })

    cmd = vllm_cmd(args, PREFILL_PORT, kv_config)
    os.execvp(cmd[0], cmd)   # replaces this process — nothing below runs


# ---------------------------------------------------------------------------
# Rank 1 — Decode node + proxy
# ---------------------------------------------------------------------------

def run_decode(args: argparse.Namespace, prefill_host: str, my_host: str) -> None:
    """
    Launch vLLM as the decode (KV consumer) instance, then use the
    built-in two-phase prefill→decode flow, process prompts, and clean up.
    """
    kv_config = make_kv_config("kv_consumer")
    print(f"[Decode] port={DECODE_PORT}  model={args.model}")
    print(f"[Decode] host={my_host}")
    print(f"[Decode] Prefill node: {prefill_host}:{PREFILL_PORT}")
    print(f"[Decode] NIXL side-channel bind: {my_host}:{NIXL_SIDE_CHANNEL_PORT}")
    print(f"[Decode] KV config: {kv_config}")
    prune_cfg = PruneConfig(
        method=args.pruning_method,
        keep_ratio=args.pruning_keep_ratio,
        min_tokens=args.pruning_min_tokens,
        seed=args.pruning_seed,
    )
    print(
        "[Decode] Pruning config: "
        f"method={prune_cfg.method}, keep_ratio={prune_cfg.keep_ratio:.2f}, "
        f"min_tokens={prune_cfg.min_tokens}, seed={prune_cfg.seed}"
    )

    decode_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES":        "0",   # one GPU per node
        "VLLM_KV_CACHE_LAYOUT":        "HND",
        # vLLM 0.18.0 starts a listener thread on each instance.
        # Therefore this must be a local bindable host on the decode node.
        "VLLM_NIXL_SIDE_CHANNEL_HOST": my_host,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(NIXL_SIDE_CHANNEL_PORT),
    }

    decode_proc = subprocess.Popen(vllm_cmd(args, DECODE_PORT, kv_config), env=decode_env)
    procs = [decode_proc]

    def cleanup(*_):
        print("\n[Decode] Shutting down ...")
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        slurm_job = os.environ.get("SLURM_JOB_ID")
        if slurm_job:
            # scancel terminates ALL nodes in the job, including rank 0 (prefill).
            print(f"[Decode] Cancelling SLURM job {slurm_job} ...")
            os.system(f"scancel {slurm_job}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT,  cleanup)

    # ── Wait for both vLLM instances to be healthy ────────────────────────
    # Decode health is local; prefill health is checked over the network.
    wait_for_health(prefill_host, PREFILL_PORT, "Prefill")
    wait_for_health("localhost",  DECODE_PORT,  "Decode")

    print("[Flow] Using built-in two-phase prefill→decode flow.")

    # ── Optional warmup ───────────────────────────────────────────────────
    if args.warmup:
        print("[Warmup] Sending warmup request ...")
        try:
            two_phase_disagg_completion(
                model=args.model,
                prompt="warmup",
                max_tokens=args.warmup_max_tokens,
                temperature=0,
                prefill_host=prefill_host,
            )
            print("[Warmup] Done.")
        except Exception as exc:
            print(f"[Warmup] Warning: {exc} (continuing)", file=sys.stderr)

    # ── Process prompts ───────────────────────────────────────────────────
    prompts = load_prompts(args.prompts_file)
    print(f"\n[Decode] Loaded {len(prompts)} prompt(s) from '{args.prompts_file}'")

    results = []
    wall_start = time.perf_counter()

    for idx, prompt in enumerate(prompts, 1):
        print(f"[Decode] Processing prompt {idx}/{len(prompts)} ...")
        prune_result = prune_prompt(prompt, prune_cfg)
        if prune_result.applied:
            reduction = 1.0 - (prune_result.kept_tokens / prune_result.original_tokens)
            print(
                f"[Prune] [{idx}] method={prune_result.method} "
                f"orig_tokens={prune_result.original_tokens} "
                f"kept_tokens={prune_result.kept_tokens} "
                f"reduction={reduction:.2%}"
            )
        else:
            print(
                f"[Prune] [{idx}] method={prune_result.method} "
                f"orig_tokens={prune_result.original_tokens} "
                f"kept_tokens={prune_result.kept_tokens} "
                "reduction=0.00%"
            )

        try:
            response = two_phase_disagg_completion(
                model=args.model,
                prompt=prune_result.text,
                max_tokens=args.max_tokens,
                temperature=0,
                prefill_host=prefill_host,
            )

            usage      = response.get("usage", {})
            output     = response["choices"][0]["text"]
            metrics = response.get("metrics", {})
            total_t = metrics.get("total_time")
            ttft    = metrics.get("ttft")
            prefill_t = metrics.get("prefill_http_time")

            n_compl = usage.get("completion_tokens")
            count_is_estimate = False
            if not n_compl:
                # Fallback when streaming usage is unavailable.
                n_compl = len(output.split())
                count_is_estimate = True

            ms_per_tok = (total_t / n_compl * 1000) if (total_t and n_compl) else None

            results.append({"prompt": prompt, "output": output,
                             "prompt_after_pruning": prune_result.text,
                             "prompt_tokens_before_pruning": prune_result.original_tokens,
                             "prompt_tokens_after_pruning": prune_result.kept_tokens,
                             "usage": usage,
                             "total_time": total_t,
                             "ttft": ttft,
                             "prefill_http_time": prefill_t,
                             "completion_tokens": n_compl,
                             "completion_tokens_estimated": count_is_estimate,
                             "ms_per_token": ms_per_tok})

        except Exception as exc:
            import traceback
            print(f"[Decode] ERROR on prompt {idx}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results.append({"prompt": prompt, "error": str(exc)})

    wall_elapsed = time.perf_counter() - wall_start

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({len(results)} prompts, wall-clock {wall_elapsed:.3f}s)")
    print(f"{'='*60}")

    ok = [r for r in results if "error" not in r]
    mean_ttft_vals = [r["ttft"] for r in ok if r.get("ttft") is not None]
    mean_tpot_vals = [r["ms_per_token"] for r in ok if r.get("ms_per_token") is not None]

    print("\n  Per-prompt metrics:")
    for j, r in enumerate(results, 1):
        if "error" in r:
            print(f"    [{j}] ERROR | prompt='{r['prompt']}' | msg={r['error']}")
            continue

        ttft = r.get("ttft")
        ttft_s = f"{ttft:.3f}s" if ttft is not None else "n/a"
        tpot = r.get("ms_per_token")
        tpot_s = f"{tpot:.1f}ms" if tpot is not None else "n/a"
        c_tok = r.get("completion_tokens")
        est = " (est.)" if r.get("completion_tokens_estimated") else ""
        total_t = r.get("total_time")
        total_s = f"{total_t:.3f}s" if total_t is not None else "n/a"
        prefill_t = r.get("prefill_http_time")
        prefill_s = f"{prefill_t:.3f}s" if prefill_t is not None else "n/a"
        print(
            f"    [{j}] OK    | prompt='{r['prompt']}' | "
            f"completion={c_tok}{est} | ttft={ttft_s} | tpot={tpot_s} | "
            f"total={total_s} | prefill={prefill_s}"
        )

    for j, r in enumerate(results, 1):
        print(f"\n  [{j}] Input : {r['prompt']}")
        if "error" in r:
            print(f"      Error : {r['error']}")
        else:
            u = r["usage"]
            print(f"      Tokens: prompt={u.get('prompt_tokens')}  "
                  f"completion={r.get('completion_tokens')}"
                  f"{' (est.)' if r.get('completion_tokens_estimated') else ''}")
            if r.get("ttft") is not None:
                print(f"      TTFT  : {r['ttft']:.3f}s")
            print(f"      Time  : {r['total_time']:.3f}s", end="")
            if r["ms_per_token"] is not None:
                print(f"  |  {r['ms_per_token']:.1f} ms/output-token")
            else:
                print()

    print(f"\n{'='*60}")
    print("  RESPONSES")
    print(f"{'='*60}")
    for j, r in enumerate(results, 1):
        print(f"\n  [{j}] Input : {r['prompt']}")
        if "error" in r:
            print(f"      Error : {r['error']}")
        else:
            print(f"      Output: {r['output']}")

    print(f"\n{'='*60}")
    print("  AVG METRICS")
    print(f"{'='*60}")
    if mean_ttft_vals:
        print(f"  Avg TTFT           : {sum(mean_ttft_vals)/len(mean_ttft_vals):.3f}s")
    else:
        print("  Avg TTFT           : n/a")
    if mean_tpot_vals:
        print(f"  Avg ms/output-token: {sum(mean_tpot_vals)/len(mean_tpot_vals):.1f} ms")
    else:
        print("  Avg ms/output-token: n/a")

    cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM NixlConnector PD disaggregation — two nodes, one GPU each"
    )
    parser.add_argument("--model",                   default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-tokens",    type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--block-size",    type=int, default=128,
                        help="KV cache block size in tokens "
                             "(larger = fewer transfer round-trips)")
    parser.add_argument("--prompts-file",  type=Path,
                        default=SCRIPT_DIR / "prompts.txt")
    parser.add_argument("--warmup",    action="store_true",  default=True,
                        help="Send a tiny warmup request before prompts (default: on)")
    parser.add_argument("--no-warmup", action="store_false", dest="warmup",
                        help="Disable the warmup request")
    parser.add_argument("--warmup-max-tokens", type=int, default=4)
    parser.add_argument(
        "--pruning-method",
        choices=["none", "attn_proxy", "random"],
        default="none",
        help="Prompt-side pruning mode for method-2 experiments.",
    )
    parser.add_argument(
        "--pruning-keep-ratio",
        type=float,
        default=1.0,
        help="Fraction of tokens kept by the pruner (0.0-1.0).",
    )
    parser.add_argument(
        "--pruning-min-tokens",
        type=int,
        default=0,
        help="Only prune prompts with at least this many tokens "
             "(0 = always prune when method != none).",
    )
    parser.add_argument(
        "--pruning-seed",
        type=int,
        default=42,
        help="Random seed used when pruning method is random.",
    )
    args = parser.parse_args()
    if not (0.0 <= args.pruning_keep_ratio <= 1.0):
        parser.error("--pruning-keep-ratio must be in [0.0, 1.0]")

    # SLURM_PROCID is set by srun on every task: 0 on node 0, 1 on node 1.
    rank = int(os.environ.get("SLURM_PROCID", 0))

    # PREFILL_HOST is exported by pd_dis.sh before srun and inherited by both tasks.
    prefill_host = os.environ.get("PREFILL_HOST", "")
    if not prefill_host:
        print("[ERROR] PREFILL_HOST is not set. "
              "Make sure pd_dis.sh exported it before calling srun.", file=sys.stderr)
        sys.exit(1)

    import socket
    my_host = socket.getfqdn()
    print(f"[Rank {rank}] host={my_host}  prefill_host={prefill_host}  model={args.model}")

    if rank == 0:
        run_prefill(args, my_host)
    else:
        run_decode(args, prefill_host, my_host)


if __name__ == "__main__":
    main()

