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


def http_post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=500) as resp:
        return json.loads(resp.read())


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

    with urllib.request.urlopen(req, timeout=500) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):                   # trailing usage chunk
                usage = chunk["usage"]
            if not chunk.get("choices"):
                continue
            last_meta  = chunk
            token_text = chunk["choices"][0].get("text", "")
            if token_text:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                text_parts.append(token_text)

    now     = time.perf_counter()
    t_first = t_first or now
    t_last  = t_last  or now

    # Prefer server-reported token count; fall back to chunk count
    n_tokens = usage.get("completion_tokens") or len(text_parts) or 1

    response = {
        "id":      last_meta.get("id", ""),
        "object":  "text_completion",
        "created": last_meta.get("created", 0),
        "model":   last_meta.get("model", body.get("model", "")),
        "choices": [{
            "text":          "".join(text_parts),
            "index":         0,
            "finish_reason": last_meta["choices"][0].get("finish_reason"),
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

        prefill_body = {**body,
                        "request_id": f"{uid}___decode_addr_{self.decode_zmq}"}
        decode_body  = {**body,
                        "request_id": f"{uid}___prefill_addr_{self.prefill_zmq}___"}

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            prefill_fut = pool.submit(http_post,
                                      self.prefill_url + endpoint, prefill_body)
            decode_fut  = pool.submit(stream_decode,
                                      self.decode_url  + endpoint, decode_body)
            try:
                prefill_fut.result()       # wait for KV transfer to complete
            except Exception:
                pass                       # prefill may timeout; decode still works
            response, t_first, t_last, n_tokens = decode_fut.result()

        # t_first = time of first decoded token  (TTFT = prefill + KV transfer)
        # t_last  = time of last  decoded token
        response["_prefill_time"]  = round(t_first - t0, 3)
        response["_decode_time"]   = round(t_last  - t_first, 3)
        response["_total_time"]    = round(t_last  - t0, 3)
        response["_decode_tokens"] = n_tokens

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
                ms_per_tok = decode_t / n_decoded * 1000

                print(f"  Output  : {output}")
                print(f"  Tokens  : prompt={usage.get('prompt_tokens')}  "
                      f"completion={usage.get('completion_tokens')}  "
                      f"total={usage.get('total_tokens')}")
                print(f"  Prefill : {prefill_t:.3f}s  (Time to first token, TTFT)")
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
                print(f"  ERROR: {exc}", file=sys.stderr)
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
    parser.add_argument("--model",         default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--quantization",  default=None,
                        help="vLLM quantization method (e.g. awq, gptq, fp8)")
    parser.add_argument("--prompts-file",  default=os.path.join(script_dir, "prompts.txt"),
                        help="Path to a text file with one prompt per line")
    parser.add_argument("--max-tokens",    type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    print(f"[Rank {rank}] host={os.uname().nodename}  model={args.model}")

    if rank == 0:
        run_prefill(args)
    else:
        run_decode(args)
