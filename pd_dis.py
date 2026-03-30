"""
PD Disaggregation via vLLM  (RDMA / InfiniBand transport)
==========================================================
Rank 0 (prefill node): vllm serve  --kv-role kv_producer
Rank 1 (decode node):  vllm serve  --kv-role kv_consumer  +  disagg proxy

KV cache transport
------------------
P2pNcclConnector drives the KV transfer via ZMQ (control) + NCCL (data).
NCCL uses InfiniBand with GPU Direct RDMA when the env vars in pd_dis.sh are set:

  NCCL_NET=IB               -> force IB transport (no Ethernet fallback)
  NCCL_NET_GDR_LEVEL=5      -> GPU Direct RDMA for all messages (SYS level)
  NCCL_IB_GDR_LEVEL=5       -> GDR on the IB transport layer
  NCCL_NET_GDR_READ=1       -> enable GDR reads (not just writes)

With GDR active the data path is:
  GPU VRAM -> IB HCA (PCIe peer-to-peer) -> remote IB HCA -> GPU VRAM
without staging through host (CPU) memory on either side.

Proxy routing  (P2pNcclConnector request_id convention)
--------------------------------------------------------
The connector locates the remote ZMQ socket via the request_id field:
  Prefill request_id:  <uid>___decode_addr_<decode_host>:<kv_port>
  Decode  request_id:  <uid>___prefill_addr_<prefill_host>:<kv_port>___

The inline proxy sets these fields and fans each request out to both servers
concurrently; the decode server blocks internally until it receives the KV
cache from prefill, then generates tokens and returns the response.

Prompts
-------
Prompts are read from a plain-text file (one prompt per line).
Lines starting with # and blank lines are ignored.
Default file: prompts.txt (same directory as this script).
"""

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
import threading
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer


# ---------------------------------------------------------------------------
# Config (populated from env vars set by pd_dis.sh / SLURM)
# ---------------------------------------------------------------------------

PREFILL_HOST = os.environ.get("PREFILL_HOST", "127.0.0.1")
PREFILL_PORT = int(os.environ.get("PREFILL_PORT", 8100))
DECODE_PORT  = int(os.environ.get("DECODE_PORT",  8200))
PROXY_PORT   = int(os.environ.get("PROXY_PORT",   8000))
KV_PORT      = int(os.environ.get("KV_PORT",      14579))
DECODE_WAIT_TIMEOUT = float(os.environ.get("DECODE_WAIT_TIMEOUT", 120))
DECODE_HTTP_TIMEOUT = float(os.environ.get("DECODE_HTTP_TIMEOUT", 300))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> list:
    """Read prompts from a text file. One prompt per line; # lines are comments."""
    if not os.path.exists(path):
        print(f"[ERROR] Prompts file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        prompts = [line.strip() for line in f
                   if line.strip() and not line.startswith("#")]
    if not prompts:
        print(f"[ERROR] No prompts found in {path}", file=sys.stderr)
        sys.exit(1)
    return prompts


def wait_for_health(url: str, label: str, poll_interval: float = 5.0):
    print(f"[{label}] Waiting for {url} to be ready ...")
    while True:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=3)
            return
        except Exception:
            time.sleep(poll_interval)


def http_post(url: str, body: dict, timeout: float = 600) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def sync_decode(url: str, body: dict, timeout: float = DECODE_HTTP_TIMEOUT):
    """Non-stream decode path to avoid hanging stream chunk iteration."""
    t0 = time.perf_counter()
    response = http_post(url, body, timeout=timeout)
    t1 = time.perf_counter()
    usage = response.get("usage", {})
    n_tokens = usage.get("completion_tokens") or 1
    return response, t0, t1, n_tokens


def vllm_mode_args(mode: str) -> list[str]:
    """Extra vLLM serve args for startup/runtime trade-offs.

    fast:       lowest startup latency (best for short interactive runs)
    balanced:   baseline behavior
    throughput: alias of baseline for now (explicit intent)
    """
    if mode == "fast":
        # Skip torch.compile + CUDA graph capture warmup; first token comes much sooner.
        return ["--enforce-eager"]
    return []


def stream_decode(url: str, body: dict):
    """POST url with stream=True.

    Returns (response_dict, t_first_token, t_last_token, n_tokens) where
    timestamps are from time.perf_counter() at the moment each boundary token
    arrived — giving true TTFT and per-token decode cadence.
    """
    streaming_body = {
        **body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(streaming_body).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t_first: float | None = None
    t_last:  float | None = None
    text_parts: list[str] = []
    last_meta: dict       = {}
    usage: dict           = {}
    line_count = 0
    max_idle = 60  # max seconds without seeing a token
    req_timeout = 120

    try:
        with urllib.request.urlopen(req, timeout=req_timeout) as resp:
            t_last_chunk = time.perf_counter()
            for raw in resp:
                now = time.perf_counter()
                if now - t_last_chunk > max_idle:
                    print(f"[stream_decode] Timeout waiting for tokens ({now - t_last_chunk:.1f}s idle)",
                          file=sys.stderr)
                    break

                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue

                payload = line[5:].strip()
                if payload == "[DONE]":
                    break

                line_count += 1
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if not chunk.get("choices"):
                    continue
                last_meta = chunk
                token_text = chunk["choices"][0].get("text", "")
                if token_text:
                    t_last_chunk = now
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text_parts.append(token_text)
    except Exception as e:
        print(f"[stream_decode] Exception: {e}", file=sys.stderr)
        # Continue with partial results
        pass

    now     = time.perf_counter()
    t_first = t_first or now
    t_last  = t_last  or now

    # Prefer server-reported token count; fall back to chunk count
    n_tokens = usage.get("completion_tokens") or len(text_parts) or 1

    # Some backends may not stream chunks reliably; fall back to regular POST.
    if not text_parts and not usage:
        t_sync0 = time.perf_counter()
        response = http_post(url, body)
        t_sync1 = time.perf_counter()
        usage = response.get("usage", {})
        n_tokens = usage.get("completion_tokens") or 1
        return response, t_sync0, t_sync1, n_tokens

    response = {
        "id":      last_meta.get("id", ""),
        "object":  "text_completion",
        "created": last_meta.get("created", 0),
        "model":   last_meta.get("model", body.get("model", "")),
        "choices": [{
            "text":          "".join(text_parts),
            "index":         0,
            "finish_reason": last_meta.get("choices", [{}])[0].get("finish_reason") if last_meta.get("choices") else None,
            "logprobs":      None,
        }],
        "usage": usage,
    }
    return response, t_first, t_last, n_tokens


def kv_config(role: str, rank: int) -> str:
    return json.dumps({
        "kv_connector":     "P2pNcclConnector",
        "kv_role":          role,   # "kv_producer" or "kv_consumer"
        "kv_rank":          rank,
        "kv_parallel_size": 2,
        "kv_ip":            PREFILL_HOST,
        "kv_port":          KV_PORT,
    })


# ---------------------------------------------------------------------------
# Inline disaggregated proxy
#
# P2pNcclConnector routes via ZMQ addresses embedded in request_id:
#   prefill sees: <uid>___decode_addr_<decode_zmq>
#   decode  sees: <uid>___prefill_addr_<prefill_zmq>___
#
# Both requests are sent concurrently. The decode server blocks until it
# receives the KV cache from prefill, then returns the completion response.
# ---------------------------------------------------------------------------

class DisaggProxyHandler(BaseHTTPRequestHandler):
    prefill_url: str = ""
    decode_url:  str = ""
    prefill_zmq: str = ""   # HOST:KV_PORT
    decode_zmq:  str = ""   # HOST:KV_PORT

    def log_message(self, fmt, *args):
        pass  # suppress per-request access logs

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        uid = str(uuid.uuid4())
        endpoint = f"/v1/completions"
        prompt_snippet = body.get("prompt", "")[:50]

        # Prefill should only build/send KV cache; do not decode/generate tokens on GPU0.
        prefill_body = {
            **body,
            "max_tokens": 1,
            "stream": False,
            "request_id": f"{uid}___decode_addr_{self.decode_zmq}",
        }
        decode_body  = {**body,
                        "request_id": f"{uid}___prefill_addr_{self.prefill_zmq}___"}

        print(f"[Proxy] [uid={uid[:8]}] Starting: '{prompt_snippet[:40]}...'", file=sys.stderr)
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            prefill_fut = pool.submit(http_post,
                                      self.prefill_url + endpoint, prefill_body)
            decode_fut  = pool.submit(sync_decode,
                                      self.decode_url + endpoint,
                                      decode_body,
                                      DECODE_HTTP_TIMEOUT)
            print(f"[Proxy] [uid={uid[:8]}] Waiting for decode...", file=sys.stderr)
            try:
                response, t_first, t_last, n_tokens = decode_fut.result(
                    timeout=DECODE_WAIT_TIMEOUT
                )
            except concurrent.futures.TimeoutError:
                print(
                    f"[Proxy] [uid={uid[:8]}] Decode wait timed out after "
                    f"{DECODE_WAIT_TIMEOUT:.0f}s; returning timeout error.",
                    file=sys.stderr,
                )
                decode_fut.cancel()
                response = {
                    "id": uid,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": body.get("model", ""),
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "finish_reason": "timeout",
                        "logprobs": None,
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "error": f"Decode timed out after {DECODE_WAIT_TIMEOUT:.0f}s",
                }
                t_first = t_last = time.perf_counter()
                n_tokens = 0
            print(f"[Proxy] [uid={uid[:8]}] Decode done, n_tokens={n_tokens}", file=sys.stderr)
            # Do not block the client on prefill finishing full generation.
            # Decode completion already implies KV transfer succeeded.
            try:
                prefill_fut.result(timeout=60)
                print(f"[Proxy] [uid={uid[:8]}] Prefill done", file=sys.stderr)
            except concurrent.futures.TimeoutError:
                print("[Proxy] Prefill response still in-flight after decode completion.",
                      file=sys.stderr)
            except Exception as exc:
                print(f"[Proxy] Prefill request error: {exc}", file=sys.stderr)

        # t_first = time of first decoded token  (TTFT = prefill + KV transfer)
        # t_last  = time of last  decoded token
        response["_prefill_time"]  = round(t_first - t0, 3)
        response["_decode_time"]   = round(t_last  - t_first, 3)
        response["_total_time"]    = round(t_last  - t0, 3)
        response["_decode_tokens"] = n_tokens
        print(f"[Proxy] [uid={uid[:8]}] Sending response: prefill={response['_prefill_time']:.3f}s, "
              f"decode={response['_decode_time']:.3f}s, tokens={n_tokens}", file=sys.stderr)

        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_proxy(prefill_url, decode_url, prefill_zmq, decode_zmq, port):
    DisaggProxyHandler.prefill_url = prefill_url
    DisaggProxyHandler.decode_url  = decode_url
    DisaggProxyHandler.prefill_zmq = prefill_zmq
    DisaggProxyHandler.decode_zmq  = decode_zmq

    server = HTTPServer(("0.0.0.0", port), DisaggProxyHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ---------------------------------------------------------------------------
# Prefill node (rank 0)
# ---------------------------------------------------------------------------

def run_prefill(args):
    cfg = kv_config("kv_producer", rank=0)
    print(f"[Prefill] Initializing GPU and loading model '{args.model}' ...")
    print(f"[Prefill] KV config: {cfg}")

    # os.execvp replaces this Python process with vLLM — no code runs after.
    # Rank 1's wait_for_health(:8100) confirms the GPU is ready.
    cmd = [
        "vllm", "serve", args.model,
        "--host", "0.0.0.0",
        "--port", str(PREFILL_PORT),
        "--max-model-len", str(args.max_model_len),
        "--dtype", "float16",
        "--kv-transfer-config", cfg,
    ]
    cmd += vllm_mode_args(args.startup_mode)
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    os.execvp("vllm", cmd)


# ---------------------------------------------------------------------------
# Decode node (rank 1)
# ---------------------------------------------------------------------------

def run_decode(args):
    cfg = kv_config("kv_consumer", rank=1)
    print(f"[Decode] Initializing GPU and loading model '{args.model}' ...")
    print(f"[Decode] KV config: {cfg}")

    cmd = [
        "vllm", "serve", args.model,
        "--host", "0.0.0.0",
        "--port", str(DECODE_PORT),
        "--max-model-len", str(args.max_model_len),
        "--dtype", "float16",
        "--kv-transfer-config", cfg,
    ]
    cmd += vllm_mode_args(args.startup_mode)
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    decode_proc = subprocess.Popen(cmd)

    # Health checks confirm both GPUs have finished loading the model
    wait_for_health(f"http://{PREFILL_HOST}:{PREFILL_PORT}", "Decode")
    print(f"[Prefill] GPU initialized and ready  ({PREFILL_HOST}:{PREFILL_PORT})")

    wait_for_health(f"http://localhost:{DECODE_PORT}", "Decode")
    print(f"[Decode]  GPU initialized and ready  (localhost:{DECODE_PORT})")

    # ZMQ addresses used by P2pNcclConnector for KV routing
    decode_host = socket.gethostname()
    prefill_zmq = f"{PREFILL_HOST}:{KV_PORT}"
    decode_zmq  = f"{decode_host}:{KV_PORT}"

    # Start inline disagg proxy
    proxy = start_proxy(
        prefill_url = f"http://{PREFILL_HOST}:{PREFILL_PORT}",
        decode_url  = f"http://localhost:{DECODE_PORT}",
        prefill_zmq = prefill_zmq,
        decode_zmq  = decode_zmq,
        port        = PROXY_PORT,
    )
    print(f"[Decode] Disagg proxy listening on port {PROXY_PORT}")
    print(f"[Decode] Prefill ZMQ: {prefill_zmq}  |  Decode ZMQ: {decode_zmq}")

    # Optional warmup: triggers NCCL communicator setup + connector path before
    # processing user prompts so the first real prompt avoids one-time init cost.
    if args.warmup:
        try:
            warmup_prompt = "warmup"
            print("[Decode] Running warmup request ...")
            _ = http_post(
                f"http://localhost:{PROXY_PORT}/v1/completions",
                {
                    "model": args.model,
                    "prompt": warmup_prompt,
                    "max_tokens": args.warmup_max_tokens,
                    "temperature": 0,
                },
            )
            print("[Decode] Warmup complete.")
        except Exception as exc:
            print(f"[Decode] Warmup failed (continuing): {exc}", file=sys.stderr)

    # Load and process prompts
    prompts = load_prompts(args.prompts_file)
    print(f"[Decode] Loaded {len(prompts)} prompt(s) from '{args.prompts_file}'")

    proxy_url  = f"http://localhost:{PROXY_PORT}"
    total_start = time.perf_counter()
    results = []

    try:
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*60}")
            print(f"  Prompt {i}/{len(prompts)}")
            print(f"{'='*60}")
            print(f"  Input : {prompt}")

            try:
                response = http_post(
                    f"{proxy_url}/v1/completions",
                    {"model": args.model, "prompt": prompt,
                     "max_tokens": args.max_tokens, "temperature": 0},
                )
                usage      = response.get("usage", {})
                output     = response["choices"][0]["text"]
                prefill_t  = response.get("_prefill_time", 0.0)
                decode_t   = response.get("_decode_time",  0.0)
                total_t    = response.get("_total_time",   0.0)
                n_decoded  = usage.get("completion_tokens") or 1
                ms_per_tok = (decode_t / n_decoded * 1000) if (n_decoded and decode_t > 0) else None

                print(f"  Output  : {output}")
                print(f"  Tokens  : prompt={usage.get('prompt_tokens')}  "
                      f"completion={usage.get('completion_tokens')}  "
                      f"total={usage.get('total_tokens')}")
                print(f"  Prefill : {prefill_t:.3f}s  (Time to first token, TTFT)")
                if ms_per_tok is None:
                    print(f"  Decode  : {decode_t:.3f}s total")
                else:
                    print(f"  Decode  : {decode_t:.3f}s total  |  "
                          f"{ms_per_tok:.1f} ms/token avg  |  "
                          f"{1000/ms_per_tok:.1f} tok/s")
                print(f"  Total   : {total_t:.3f}s")

                results.append({
                    "prompt": prompt, "output": output,
                    "usage": usage,
                    "prefill_time":  prefill_t,
                    "decode_time":   decode_t,
                    "total_time":    total_t,
                    "ms_per_token":  ms_per_tok,
                })

            except Exception as exc:
                import traceback
                print(f"  ERROR: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                results.append({"prompt": prompt, "error": str(exc)})

        total_elapsed = time.perf_counter() - total_start

        # ── Final summary ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  SUMMARY  ({len(results)} prompt(s), wall-clock {total_elapsed:.3f}s)")
        print(f"{'='*60}")
        for j, r in enumerate(results, 1):
            print(f"\n  [{j}] Input   : {r['prompt']}")
            if "error" in r:
                print(f"       Error   : {r['error']}")
            else:
                print(f"       Output  : {r['output']}")
                u = r["usage"]
                print(f"       Tokens  : prompt={u.get('prompt_tokens')}  "
                      f"completion={u.get('completion_tokens')}  "
                      f"total={u.get('total_tokens')}")
                print(f"       Prefill : {r['prefill_time']:.3f}s  (Time to first token, TTFT)")
                if r["ms_per_token"] is None:
                    print(f"       Decode  : {r['decode_time']:.3f}s total")
                else:
                    print(f"       Decode  : {r['decode_time']:.3f}s total  |  "
                          f"{r['ms_per_token']:.1f} ms/token avg  |  "
                          f"{1000/r['ms_per_token']:.1f} tok/s")
                print(f"       Total   : {r['total_time']:.3f}s")
        print(f"\n{'='*60}")

    finally:
        proxy.shutdown()
        decode_proc.send_signal(signal.SIGTERM)
        decode_proc.wait()

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            print(f"[Decode] Cancelling SLURM job {slurm_job_id} ...")
            os.system(f"scancel {slurm_job_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="vLLM-based prefill-decode disaggregation")
    parser.add_argument("--model",         default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--quantization",  default=None,
                        help="vLLM quantization method (e.g. awq, gptq, fp8)")
    parser.add_argument(
        "--startup-mode",
        choices=["fast", "balanced", "throughput"],
        default="throughput",
        help=(
            "vLLM startup/runtime tradeoff. "
            "fast skips compile/graph-capture warmup for lower startup latency."
        ),
    )
    parser.add_argument("--prompts-file",  default=os.path.join(script_dir, "prompts.txt"),
                        help="Path to a text file with one prompt per line")
    parser.add_argument("--max-tokens",    type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Run a tiny warmup request before processing prompts (default: enabled)")
    parser.add_argument("--no-warmup", action="store_false", dest="warmup",
                        help="Disable the initial warmup request")
    parser.add_argument("--warmup-max-tokens", type=int, default=4,
                        help="Max tokens for warmup request")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    print(f"[Rank {rank}] host={os.uname().nodename}  model={args.model}")

    if rank == 0:
        run_prefill(args)
    else:
        run_decode(args)
