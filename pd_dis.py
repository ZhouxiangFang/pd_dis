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

Prompts / Dataset
─────────────────
When --dataset is "none" (default): read from prompts.txt (same directory).
When --dataset is "lveval":  load Infinigence/LVEval (sample --dataset-n-samples
  instances with seed --dataset-seed) and compute F1 accuracy.
When --dataset is "aime25":  load math-ai/aime25 and compute exact-match accuracy.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import Counter
from pathlib import Path

from methods.attn_pruning import PruneConfig, prune_prompt

# ---------------------------------------------------------------------------
# Ports & constants
#
# SLURM may pack concurrent jobs onto the same compute node (an L40S node has
# 4 GPUs; many 1-GPU jobs share a node). Hardcoded ports collide in that
# case. Offsets from SLURM_JOB_ID keep concurrent jobs disjoint; SLURM_PROCID
# keeps the two ranks of a single 1-node 2-GPU job from colliding on the NIXL
# side-channel port.
# ---------------------------------------------------------------------------
_JOB_OFFSET  = int(os.environ.get("SLURM_JOB_ID", "0")) % 1000
_RANK_OFFSET = int(os.environ.get("SLURM_PROCID", "0")) * 10
PREFILL_PORT           = 18000 + _JOB_OFFSET                    # one bind per role
DECODE_PORT            = 19000 + _JOB_OFFSET
NIXL_SIDE_CHANNEL_PORT = 25000 + _JOB_OFFSET + _RANK_OFFSET     # per-rank listener

SCRIPT_DIR = Path(__file__).resolve().parent

LVEVAL_HF_NAME      = "Infinigence/LVEval"
AIME25_HF_NAME      = "math-ai/aime25"
LVEVAL_DEFAULT_SUBSET  = "hotpotwikiqa_mixup"
LVEVAL_LENGTH_LEVELS   = ["16k", "32k", "64k", "128k", "256k"]
LVEVAL_DEFAULT_LEN     = "16k"

# ---------------------------------------------------------------------------
# NixlConnector KV-transfer config
# ---------------------------------------------------------------------------
def make_kv_config(kv_role: str) -> str:
    return json.dumps({
        "kv_connector": "NixlConnector",
        "kv_role": kv_role,
        "kv_buffer_device": "cuda",
    })


# ---------------------------------------------------------------------------
# Helpers — prompts
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


# ---------------------------------------------------------------------------
# Helpers — statistics
# ---------------------------------------------------------------------------

def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _fmt_s(v: float | None, prec: int = 3) -> str:
    return f"{v:.{prec}f}s" if v is not None else "n/a"


def _fmt_ms(v: float | None) -> str:
    return f"{v:.1f}ms" if v is not None else "n/a"


# ---------------------------------------------------------------------------
# Helpers — dataset loading
# ---------------------------------------------------------------------------

def _hf_load_dataset(name: str, *args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[ERROR] 'datasets' package not found.\n"
            "        Install it: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)
    return load_dataset(name, *args, **kwargs)


def load_dataset_items(
    dataset: str,
    subset: str | None,
    n_samples: int,
    seed: int,
    len_level: str = LVEVAL_DEFAULT_LEN,
) -> list[dict]:
    """
    Load evaluation items from a HuggingFace dataset.

    Returns list of dicts:
      {
        "prompt":        str,           # full prompt sent to vLLM
        "prompt_display": str,          # ≤80-char version for logs
        "ground_truth":  str | list,   # reference answer(s)
      }
    """
    rng = random.Random(seed)

    if dataset == "lveval":
        base = subset or LVEVAL_DEFAULT_SUBSET
        cfg  = f"{base}_{len_level}"
        print(f"[Dataset] Loading {LVEVAL_HF_NAME} subset={cfg} ...")
        ds = _hf_load_dataset(LVEVAL_HF_NAME, cfg, split="test")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[:min(n_samples, len(ds))]
        items = []
        for i in selected:
            ex = ds[i]
            context  = ex.get("context", "")
            question = ex.get("input", "")
            prompt = (
                f"{context}\n\n"
                f"Based on the above text, answer the following question briefly.\n"
                f"Question: {question}\nAnswer:"
            )
            gts = ex.get("answers", ex.get("output", []))
            if isinstance(gts, str):
                gts = [gts]
            items.append({
                "prompt":         prompt,
                "prompt_display": question[:80].replace("\n", " ").replace("'", " "),
                "ground_truth":   gts,
            })
        print(f"[Dataset] Loaded {len(items)} items from LVEval/{cfg}.")
        return items

    elif dataset == "aime25":
        print(f"[Dataset] Loading {AIME25_HF_NAME} ...")
        ds = None
        for split in ("train", "test", "validation"):
            try:
                ds = _hf_load_dataset(AIME25_HF_NAME, split=split)
                break
            except Exception:
                continue
        if ds is None:
            ds_dict = _hf_load_dataset(AIME25_HF_NAME)
            ds = ds_dict[next(iter(ds_dict))]

        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[:min(n_samples, len(ds))]
        items = []
        for i in selected:
            ex    = ds[i]
            prob  = ex.get("problem", ex.get("question", ""))
            ans   = str(ex.get("answer", ""))
            prompt = (
                "Solve the following competition math problem. "
                r"Show your work, then put your final integer answer inside \boxed{}."
                "\n\n"
                f"Problem: {prob}"
            )
            items.append({
                "prompt":         prompt,
                "prompt_display": prob[:80].replace("\n", " ").replace("'", " "),
                "ground_truth":   ans,
            })
        print(f"[Dataset] Loaded {len(items)} items from AIME25.")
        return items

    else:
        print(f"[ERROR] Unknown dataset: {dataset}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers — answer extraction and scoring
# ---------------------------------------------------------------------------

def extract_answer_aime(text: str) -> str | None:
    r"""Extract the final integer answer (0-999) from model output."""
    # \boxed{N}
    m = re.search(r'\\boxed\{(\d+)\}', text)
    if m:
        return m.group(1)
    # "the answer is N" / "answer: N" / "answer = N"
    m = re.search(
        r'(?:the\s+answer\s+is|answer\s*[:=])\s*(\d{1,3})\b',
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)
    # last 1–3 digit number (AIME answers are 0–999)
    nums = re.findall(r'\b(\d{1,3})\b', text)
    return nums[-1] if nums else None


def f1_score_tokens(pred: str, ref: str) -> float:
    """Token-level F1 between prediction and reference strings."""
    p_toks = pred.lower().split()
    r_toks = ref.lower().split()
    if not p_toks or not r_toks:
        return 0.0
    common = Counter(p_toks) & Counter(r_toks)
    num = sum(common.values())
    if num == 0:
        return 0.0
    prec = num / len(p_toks)
    rec  = num / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def score_result(
    output: str,
    ground_truth,
    dataset: str,
) -> tuple[float, str | None]:
    """
    Returns (score, extracted_answer).
      aime25: 1.0 if extracted integer matches reference, else 0.0
      lveval:  best token-F1 over all ground truth strings
    """
    if dataset == "aime25":
        extracted = extract_answer_aime(output)
        if extracted is None:
            return 0.0, None
        ref = ground_truth[0] if isinstance(ground_truth, (list, tuple)) else str(ground_truth)
        try:
            score = 1.0 if int(extracted) == int(ref) else 0.0
        except (ValueError, TypeError):
            score = 1.0 if extracted.strip() == ref.strip() else 0.0
        return score, extracted

    elif dataset == "lveval":
        # The prompt ends with "Answer:" so the model's answer is the text that
        # follows. Strip any chain-of-thought that precedes "Answer:" in the
        # output (some models repeat the cue before answering).
        answer_marker = re.search(r'[Aa]nswer\s*:\s*', output)
        if answer_marker:
            extracted = output[answer_marker.end():].strip()
        else:
            extracted = output.strip()
        # Keep only the first sentence/line — LVEval answers are short phrases.
        extracted = re.split(r'[\n.。]', extracted)[0].strip()[:200]
        refs = ground_truth if isinstance(ground_truth, (list, tuple)) else [str(ground_truth)]
        score = max((f1_score_tokens(extracted, r) for r in refs), default=0.0)
        return score, extracted

    return 0.0, None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

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
    Built-in two-phase disaggregated serving:
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
        "--kv-cache-dtype",         args.kv_cache_dtype,
        "--kv-transfer-config",     kv_config,
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    cmd.extend(optional_vllm_serve_flags())
    return cmd


# ---------------------------------------------------------------------------
# Rank 0 — Prefill node
# ---------------------------------------------------------------------------

def run_prefill(args: argparse.Namespace, my_host: str) -> None:
    kv_config = make_kv_config("kv_producer")
    print(f"[Prefill] host={my_host}  port={PREFILL_PORT}  model={args.model}")
    print(f"[Prefill] NIXL side-channel: {my_host}:{NIXL_SIDE_CHANNEL_PORT}")
    print(f"[Prefill] KV config: {kv_config}")

    # GPU selection: 2-node 1-GPU sees only its own GPU as device 0; in
    # 1-node 2-GPU mode SLURM assigns each task its own CUDA_VISIBLE_DEVICES.
    # Don't stomp on SLURM's assignment.
    env_update = {
        "VLLM_KV_CACHE_LAYOUT":        "HND",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": my_host,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(NIXL_SIDE_CHANNEL_PORT),
    }
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env_update["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.update(env_update)

    cmd = vllm_cmd(args, PREFILL_PORT, kv_config)
    os.execvp(cmd[0], cmd)


# ---------------------------------------------------------------------------
# Rank 1 — Decode node + proxy
# ---------------------------------------------------------------------------

def run_decode(args: argparse.Namespace, prefill_host: str, my_host: str) -> None:
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
    _lveval_cfg = f"{args.dataset_subset or LVEVAL_DEFAULT_SUBSET}_{args.dataset_len}"
    print(f"[Decode] Dataset: {args.dataset}"
          + (f"  subset={_lveval_cfg}" if args.dataset == "lveval" else
             (f"  subset={args.dataset_subset}" if args.dataset_subset else "")))

    decode_env = {
        **os.environ,
        "VLLM_KV_CACHE_LAYOUT":        "HND",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": my_host,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(NIXL_SIDE_CHANNEL_PORT),
    }
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        decode_env["CUDA_VISIBLE_DEVICES"] = "0"

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
            print(f"[Decode] Cancelling SLURM job {slurm_job} ...")
            os.system(f"scancel {slurm_job}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT,  cleanup)

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

    # ── Load items (prompt + optional ground truth) ───────────────────────
    if args.dataset == "none":
        raw_prompts = load_prompts(args.prompts_file)
        items = [
            {
                "prompt":         p,
                "prompt_display": p[:80].replace("\n", " ").replace("'", " "),
                "ground_truth":   None,
            }
            for p in raw_prompts
        ]
    else:
        items = load_dataset_items(
            args.dataset,
            args.dataset_subset,
            args.dataset_n_samples,
            args.dataset_seed,
            args.dataset_len,
        )

    print(f"\n[Decode] Loaded {len(items)} item(s)")

    results = []
    wall_start = time.perf_counter()

    for idx, item in enumerate(items, 1):
        prompt         = item["prompt"]
        prompt_display = item.get("prompt_display", prompt[:80].replace("\n", " ").replace("'", " "))
        ground_truth   = item.get("ground_truth")

        print(f"[Decode] Processing item {idx}/{len(items)} ...")
        prune_result = prune_prompt(prompt, prune_cfg)
        reduction_str = (
            f"{1.0 - prune_result.kept_tokens / prune_result.original_tokens:.2%}"
            if prune_result.applied else "0.00%"
        )
        print(
            f"[Prune] [{idx}] method={prune_result.method} "
            f"orig_tokens={prune_result.original_tokens} "
            f"kept_tokens={prune_result.kept_tokens} "
            f"reduction={reduction_str}"
        )

        try:
            response = two_phase_disagg_completion(
                model=args.model,
                prompt=prune_result.text,
                max_tokens=args.max_tokens,
                temperature=0,
                prefill_host=prefill_host,
            )

            usage     = response.get("usage", {})
            output    = response["choices"][0]["text"]
            metrics   = response.get("metrics", {})
            total_t   = metrics.get("total_time")
            ttft      = metrics.get("ttft")
            prefill_t = metrics.get("prefill_http_time")

            n_compl = usage.get("completion_tokens")
            count_is_estimate = False
            if not n_compl:
                n_compl = len(output.split())
                count_is_estimate = True

            ms_per_tok = (total_t / n_compl * 1000) if (total_t and n_compl) else None

            score, extracted = (
                score_result(output, ground_truth, args.dataset)
                if (args.dataset != "none" and ground_truth is not None)
                else (None, None)
            )

            results.append({
                "prompt":                       prompt,
                "prompt_display":               prompt_display,
                "output":                       output,
                "ground_truth":                 ground_truth,
                "score":                        score,
                "extracted_answer":             extracted,
                "prompt_after_pruning":         prune_result.text,
                "prompt_tokens_before_pruning": prune_result.original_tokens,
                "prompt_tokens_after_pruning":  prune_result.kept_tokens,
                "usage":                        usage,
                "total_time":                   total_t,
                "ttft":                         ttft,
                "prefill_http_time":            prefill_t,
                "completion_tokens":            n_compl,
                "completion_tokens_estimated":  count_is_estimate,
                "ms_per_token":                 ms_per_tok,
            })

        except Exception as exc:
            print(f"[Decode] ERROR on item {idx}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results.append({
                "prompt":         prompt,
                "prompt_display": prompt_display,
                "ground_truth":   ground_truth,
                "error":          str(exc),
            })

    wall_elapsed = time.perf_counter() - wall_start

    # ── Aggregate values ──────────────────────────────────────────────────
    ok = [r for r in results if "error" not in r]

    ttft_vals    = [r["ttft"]              for r in ok if r.get("ttft")              is not None]
    tpot_vals    = [r["ms_per_token"]      for r in ok if r.get("ms_per_token")      is not None]
    e2e_vals     = [r["total_time"]        for r in ok if r.get("total_time")        is not None]
    prefill_vals = [r["prefill_http_time"] for r in ok if r.get("prefill_http_time") is not None]
    total_ctok   = sum(r.get("completion_tokens", 0) or 0 for r in ok)
    tok_per_s    = (total_ctok / wall_elapsed) if (wall_elapsed > 0 and total_ctok) else None

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    # ── Accuracy values ───────────────────────────────────────────────────
    n_correct = acc_pct = avg_score = ds_tag = threshold_label = None
    if args.dataset != "none":
        scored = [r for r in ok if r.get("score") is not None]
        n_scored  = len(scored)
        avg_score = _mean([r["score"] for r in scored]) or 0.0
        if args.dataset == "aime25":
            n_correct       = sum(1 for r in scored if r["score"] >= 1.0)
            threshold_label = "exact match"
        else:
            n_correct       = sum(1 for r in scored if r["score"] >= 0.3)
            threshold_label = "F1 >= 0.3"
        acc_pct = (n_correct / n_scored * 100) if n_scored else 0.0
        ds_tag  = (
            f"{(args.dataset_subset or LVEVAL_DEFAULT_SUBSET)}_{args.dataset_len}".replace("-", "_")
            if args.dataset == "lveval"
            else args.dataset
        )

    # ── Build aggregate dict (used for CSV and printing) ──────────────────
    agg = {
        "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
        "model":            args.model,
        "dataset":          args.dataset,
        "dataset_subset":   (
            f"{args.dataset_subset or LVEVAL_DEFAULT_SUBSET}_{args.dataset_len}"
            if args.dataset == "lveval" else (args.dataset_subset or "")
        ),
        "n_samples":        args.dataset_n_samples,
        "dataset_seed":     args.dataset_seed,
        "n_total":          len(results),
        "n_ok":             len(ok),
        "n_err":            len(results) - len(ok),
        "wall_elapsed_s":   round(wall_elapsed, 3),
        "ttft_mean_s":      _mean(ttft_vals),
        "ttft_p50_s":       percentile(ttft_vals, 50),
        "ttft_p95_s":       percentile(ttft_vals, 95),
        "ttft_p99_s":       percentile(ttft_vals, 99),
        "e2e_mean_s":       _mean(e2e_vals),
        "e2e_p50_s":        percentile(e2e_vals, 50),
        "e2e_p95_s":        percentile(e2e_vals, 95),
        "e2e_p99_s":        percentile(e2e_vals, 99),
        "prefill_mean_s":   _mean(prefill_vals),
        "prefill_p50_s":    percentile(prefill_vals, 50),
        "prefill_p95_s":    percentile(prefill_vals, 95),
        "prefill_p99_s":    percentile(prefill_vals, 99),
        "tpot_mean_ms":     _mean(tpot_vals),
        "tok_per_s":        tok_per_s,
        "total_ctok":       total_ctok,
        "acc_pct":          acc_pct,
        "avg_score":        avg_score,
        "n_correct":        n_correct,
    }

    # ── SUMMARY header (parsed by parse_results.py) ───────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({len(results)} prompts, wall-clock {wall_elapsed:.3f}s)")
    print(f"{'='*60}")

    # Per-prompt compact lines (ALL items — parse_results.py reads these)
    print("\n  Per-prompt metrics:")
    for j, r in enumerate(results, 1):
        disp = r.get("prompt_display", r["prompt"])[:80]
        if "error" in r:
            print(f"    [{j}] ERROR | prompt='{disp}' | msg={r['error']}")
            continue
        ttft_v    = r.get("ttft")
        tpot_v    = r.get("ms_per_token")
        total_v   = r.get("total_time")
        prefill_v = r.get("prefill_http_time")
        c_tok     = r.get("completion_tokens")
        est       = " (est.)" if r.get("completion_tokens_estimated") else ""
        print(
            f"    [{j}] OK    | prompt='{disp}' | "
            f"completion={c_tok}{est} | "
            f"ttft={_fmt_s(ttft_v)} | tpot={_fmt_ms(tpot_v)} | "
            f"total={_fmt_s(total_v)} | prefill={_fmt_s(prefill_v)}"
        )

    # Verbose detail — first 5 only
    SHOW = 5
    print(f"\n  Detail (first {min(SHOW, len(results))} of {len(results)}):")
    for j, r in enumerate(results[:SHOW], 1):
        disp = r.get("prompt_display", r["prompt"])[:80]
        print(f"\n  [{j}] Input : {disp}")
        if "error" in r:
            print(f"      Error : {r['error']}")
        else:
            u = r.get("usage", {})
            print(f"      Tokens: prompt={u.get('prompt_tokens')}  "
                  f"completion={r.get('completion_tokens')}"
                  f"{' (est.)' if r.get('completion_tokens_estimated') else ''}")
            print(f"      TTFT  : {_fmt_s(r.get('ttft'))}")
            print(f"      E2E   : {_fmt_s(r.get('total_time'))}  "
                  f"tpot={_fmt_ms(r.get('ms_per_token'))}")
            if args.dataset != "none" and r.get("score") is not None:
                print(f"      Score : {r['score']:.4f}  "
                      f"extracted='{r.get('extracted_answer', '')}'  "
                      f"ref='{r.get('ground_truth', '')}'")
    if len(results) > SHOW:
        print(f"\n  ... ({len(results) - SHOW} more items not shown)")

    # Responses — first 5 only
    print(f"\n{'='*60}")
    print(f"  RESPONSES (first {min(SHOW, len(results))} of {len(results)})")
    print(f"{'='*60}")
    for j, r in enumerate(results[:SHOW], 1):
        disp = r.get("prompt_display", r["prompt"])[:80]
        print(f"\n  [{j}] Input : {disp}")
        if "error" in r:
            print(f"      Error : {r['error']}")
        else:
            out = r.get("output", "")
            print(f"      Output: {out[:500]}" + ("..." if len(out) > 500 else ""))
    if len(results) > SHOW:
        print(f"\n  ... ({len(results) - SHOW} more responses not shown)")

    # ── Aggregate metrics ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AGGREGATE METRICS")
    print(f"{'='*60}")
    print(f"  Errors             : {agg['n_err']} / {agg['n_total']}")

    def _stat_row(label: str, mean, p50, p95, p99, fmt_fn) -> None:
        print(f"  {label:<20} mean={fmt_fn(mean)}  "
              f"p50={fmt_fn(p50)}  p95={fmt_fn(p95)}  p99={fmt_fn(p99)}")

    _stat_row("TTFT",         agg["ttft_mean_s"],    agg["ttft_p50_s"],
              agg["ttft_p95_s"],    agg["ttft_p99_s"],    _fmt_s)
    _stat_row("E2E latency",  agg["e2e_mean_s"],     agg["e2e_p50_s"],
              agg["e2e_p95_s"],     agg["e2e_p99_s"],     _fmt_s)
    _stat_row("Prefill time", agg["prefill_mean_s"], agg["prefill_p50_s"],
              agg["prefill_p95_s"], agg["prefill_p99_s"], _fmt_s)
    print(f"  {'TPOT (ms/tok)':<20} mean={_fmt_ms(agg['tpot_mean_ms'])}")
    tps_str = f"{agg['tok_per_s']:.1f} tok/s" if agg["tok_per_s"] else "n/a"
    print(f"  {'Throughput':<20} {tps_str}"
          f"  (total_ctok={agg['total_ctok']}, wall={wall_elapsed:.1f}s)")

    # ── Accuracy ──────────────────────────────────────────────────────────
    if args.dataset != "none":
        print(f"\n{'='*60}")
        print(f"  ACCURACY [{ds_tag}]  ({threshold_label})")
        print(f"{'='*60}")
        print(f"  Correct   : {n_correct} / {n_scored} ({acc_pct:.2f}%)")
        print(f"  Avg score : {avg_score:.4f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_str     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ds_slug    = args.dataset if args.dataset != "none" else "prompts"
        # Sanitize model name: "Qwen/Qwen3-4B" → "Qwen3-4B"
        model_slug = re.sub(r"[^\w\-.]", "_", args.model.split("/")[-1])
        run_slug   = f"{model_slug}_{ds_slug}_{ts_str}"

        # Per-prompt CSV
        prompt_csv = out_dir / f"per_prompt_{run_slug}.csv"
        pp_fields  = [
            "idx", "prompt_display", "ground_truth", "extracted_answer", "score",
            "prompt_tokens", "completion_tokens", "completion_tokens_estimated",
            "ttft_s", "prefill_s", "e2e_s", "tpot_ms", "error",
        ]
        with prompt_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=pp_fields, extrasaction="ignore")
            w.writeheader()
            for j, r in enumerate(results, 1):
                u = r.get("usage", {})
                w.writerow({
                    "idx":                        j,
                    "prompt_display":             r.get("prompt_display", ""),
                    "ground_truth":               str(r.get("ground_truth", "")),
                    "extracted_answer":           r.get("extracted_answer", ""),
                    "score":                      r.get("score", ""),
                    "prompt_tokens":              u.get("prompt_tokens", ""),
                    "completion_tokens":          r.get("completion_tokens", ""),
                    "completion_tokens_estimated": r.get("completion_tokens_estimated", ""),
                    "ttft_s":                     r.get("ttft", ""),
                    "prefill_s":                  r.get("prefill_http_time", ""),
                    "e2e_s":                      r.get("total_time", ""),
                    "tpot_ms":                    r.get("ms_per_token", ""),
                    "error":                      r.get("error", ""),
                })

        # Summary CSV (one row per run)
        summary_csv = out_dir / f"summary_{run_slug}.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agg.keys()))
            w.writeheader()
            w.writerow(agg)

        print(f"\n[CSV] per-prompt : {prompt_csv}")
        print(f"[CSV] summary    : {summary_csv}")

    cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM NixlConnector PD disaggregation — two nodes, one GPU each"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Qwen3 thinking mode (--enable-reasoning + qwen3 parser)")
    parser.add_argument("--max-tokens",    type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Max sequence length passed to vLLM. Must not exceed the "
                             "model's max_position_embeddings (32,768 for Qwen3-4B).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size",    type=int, default=1024,
                        help="KV cache block size in tokens — also the unit at "
                             "which NixlConnector transfers KV. Smaller = more "
                             "transfer/compute overlap (pipelining knob).")
    parser.add_argument("--kv-cache-dtype",
                        choices=("auto", "float16", "bfloat16",
                                 "fp8", "fp8_e4m3", "fp8_e5m2"),
                        default="auto",
                        help="vLLM KV cache storage dtype. 'auto' = model dtype "
                             "(fp16 here). 'fp8_e5m2' / 'fp8_e4m3' = 2x compression "
                             "on the wire via NIXL (KV IS fp8 in memory → NIXL ships "
                             "fp8 bytes, no extra code needed). Requires GCC module "
                             "loaded for flashinfer JIT (see pd_dis.sh).")
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Pass --enforce-eager to vllm serve. Disables CUDA "
                             "graph capture, which prevents fp8 flashinfer paths "
                             "from blowing past the gpu-memory-utilization budget.")
    parser.add_argument("--prompts-file",  type=Path,
                        default=SCRIPT_DIR / "prompts.txt")
    parser.add_argument("--warmup",    action="store_true",  default=True,
                        help="Send a tiny warmup request before prompts (default: on)")
    parser.add_argument("--no-warmup", action="store_false", dest="warmup",
                        help="Disable the warmup request")
    parser.add_argument("--warmup-max-tokens", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of requests to run in parallel (default: 4)")

    # ── Pruning ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--pruning-method",
        choices=["none", "attn_proxy", "random"],
        default="none",
    )
    parser.add_argument("--pruning-keep-ratio", type=float, default=1.0)
    parser.add_argument("--pruning-min-tokens", type=int,   default=0)
    parser.add_argument("--pruning-seed",       type=int,   default=42)

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        choices=["none", "lveval", "aime25"],
        default="aime25",
        help=(
            "Evaluation dataset. "
            "'lveval' → Infinigence/LVEval (F1 accuracy); "
            "'aime25' → math-ai/aime25 (exact-match accuracy). "
            "Default: none (read from --prompts-file)."
        ),
    )
    parser.add_argument(
        "--dataset-subset",
        default=None,
        help=f"LVEval base subset name without length suffix (default: {LVEVAL_DEFAULT_SUBSET}). "
             "E.g. hotpotwikiqa_mixup, multifieldqa_en_mixup, loogle_SD_mixup, "
             "loogle_CR_mixup, loogle_MIR_mixup, factrecall_en, cmrc_mixup.",
    )
    parser.add_argument(
        "--dataset-len",
        default=LVEVAL_DEFAULT_LEN,
        choices=LVEVAL_LENGTH_LEVELS,
        help=f"LVEval context length level (default: {LVEVAL_DEFAULT_LEN}). "
             "Combined with --dataset-subset to form the HF config name, "
             f"e.g. hotpotwikiqa_mixup_32k. Choices: {', '.join(LVEVAL_LENGTH_LEVELS)}.",
    )
    parser.add_argument(
        "--dataset-n-samples",
        type=int,
        default=100,
        help="Number of dataset instances to sample (default: 100).",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Directory to write per_prompt_*.csv and summary_*.csv "
             f"(default: {SCRIPT_DIR}/results).",
    )

    args = parser.parse_args()
    if not (0.0 <= args.pruning_keep_ratio <= 1.0):
        parser.error("--pruning-keep-ratio must be in [0.0, 1.0]")

    rank = int(os.environ.get("SLURM_PROCID", 0))

    prefill_host = os.environ.get("PREFILL_HOST", "")
    if not prefill_host:
        print("[ERROR] PREFILL_HOST is not set. "
              "Make sure pd_dis.sh exported it before calling srun.", file=sys.stderr)
        sys.exit(1)

    my_host = socket.getfqdn()
    print(f"[Rank {rank}] host={my_host}  prefill_host={prefill_host}  model={args.model}")

    if rank == 0:
        run_prefill(args, my_host)
    else:
        run_decode(args, prefill_host, my_host)


if __name__ == "__main__":
    main()
