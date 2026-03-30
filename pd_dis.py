"""
PD Disaggregation via vLLM  (RDMA / InfiniBand transport)
==========================================================
Rank 0 (prefill node): vllm serve  --kv-role kv_producer
Rank 1 (decode node):  vllm serve  --kv-role kv_consumer  +  disagg proxy

KV cache transport
------------------
PyNcclConnector drives the KV transfer.  NCCL uses InfiniBand with GPU Direct
RDMA (GDR) when the env vars in pd_dis.sh are set:

  NCCL_NET=IB               -> force IB transport (no Ethernet fallback)
  NCCL_NET_GDR_LEVEL=5      -> GPU Direct RDMA for all messages (SYS level)
  NCCL_IB_GDR_LEVEL=5       -> GDR on the IB transport layer
  NCCL_NET_GDR_READ=1       -> enable GDR reads (not just writes)

With GDR active the data path is:
  GPU VRAM -> IB HCA (PCIe peer-to-peer) -> remote IB HCA -> GPU VRAM
without staging through host (CPU) memory on either side.

Proxy routing
-------------
  1. POST to prefill server  -> prefill runs, KV cache pushed via RDMA
  2. POST to decode server   -> decode runs using the received KV cache
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


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

def wait_for_health(url: str, label: str, poll_interval: float = 5.0):
    print(f"[{label}] Waiting for {url} ...")
    while True:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=3)
            print(f"[{label}] {url} is healthy.")
            return
        except Exception:
            time.sleep(poll_interval)


def send_request(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model":       model,
        "prompt":      prompt,
        "max_tokens":  max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def kv_config(role: str, rank: int) -> str:
    return json.dumps({
        "kv_connector":    "PyNcclConnector",
        "kv_role":         role,           # "kv_producer" or "kv_consumer"
        "kv_rank":         rank,
        "kv_parallel_size": 2,
        "kv_ip":           PREFILL_HOST,
        "kv_port":         KV_PORT,
    })


# ---------------------------------------------------------------------------
# Prefill node (rank 0)
# ---------------------------------------------------------------------------

def run_prefill(args):
    print(f"[Prefill] Starting vLLM server on {PREFILL_HOST}:{PREFILL_PORT}")
    cfg = kv_config("kv_producer", rank=0)
    print(f"[Prefill] KV config: {cfg}")

    # Blocking — process stays alive until the job is cancelled by rank 1
    os.execvp("vllm", [
        "vllm", "serve", args.model,
        "--host", "0.0.0.0",
        "--port", str(PREFILL_PORT),
        "--max-model-len", str(args.max_model_len),
        "--dtype", "float16",
        "--disable-log-requests",
        "--kv-transfer-config", cfg,
    ])


# ---------------------------------------------------------------------------
# Decode node (rank 1)
# ---------------------------------------------------------------------------

def run_decode(args):
    print(f"[Decode] Starting vLLM server on port {DECODE_PORT}")
    cfg = kv_config("kv_consumer", rank=1)
    print(f"[Decode] KV config: {cfg}")

    decode_proc = subprocess.Popen([
        "vllm", "serve", args.model,
        "--host", "0.0.0.0",
        "--port", str(DECODE_PORT),
        "--max-model-len", str(args.max_model_len),
        "--dtype", "float16",
        "--disable-log-requests",
        "--kv-transfer-config", cfg,
    ])

    # Wait for both servers to be healthy before starting the proxy
    wait_for_health(f"http://{PREFILL_HOST}:{PREFILL_PORT}", "Decode")
    wait_for_health(f"http://localhost:{DECODE_PORT}", "Decode")

    # Start disaggregated prefill proxy
    # The proxy sends each request to the prefill server first (triggering KV
    # transfer via NCCL), then forwards the same request to the decode server.
    print(f"[Decode] Starting disagg proxy on port {PROXY_PORT} ...")
    proxy_proc = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.disagg_prefill_proxy_server",
        "--host",    "0.0.0.0",
        "--port",    str(PROXY_PORT),
        "--prefill", f"http://{PREFILL_HOST}:{PREFILL_PORT}",
        "--decode",  f"http://localhost:{DECODE_PORT}",
    ])
    time.sleep(5)  # give proxy time to bind

    # Send test request through the proxy
    proxy_url = f"http://localhost:{PROXY_PORT}"
    print(f"[Decode] Sending test request: {args.prompt!r}")
    t_start = time.perf_counter()
    try:
        response = send_request(proxy_url, args.model, args.prompt, args.max_tokens)
        elapsed = time.perf_counter() - t_start

        print("=== Generated Text ===")
        print(response["choices"][0]["text"])
        usage = response.get("usage", {})
        if usage:
            print(f"=== Usage: prompt={usage.get('prompt_tokens')} "
                  f"completion={usage.get('completion_tokens')} "
                  f"total={usage.get('total_tokens')} ===")
        print(f"Total end-to-end latency: {elapsed:.3f}s")

    except Exception as exc:
        print(f"[Decode] ERROR: request failed — {exc}", file=sys.stderr)

    finally:
        # Clean up local processes; scancel releases the prefill node
        for proc in (proxy_proc, decode_proc):
            proc.send_signal(signal.SIGTERM)
        for proc in (proxy_proc, decode_proc):
            proc.wait()

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            print(f"[Decode] Cancelling SLURM job {slurm_job_id} to release prefill node ...")
            os.system(f"scancel {slurm_job_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM-based prefill-decode disaggregation")
    parser.add_argument("--model",          default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt",         default="Explain what prefill-decode disaggregation means in LLM inference:")
    parser.add_argument("--max-tokens",     type=int, default=1024)
    parser.add_argument("--max-model-len",  type=int, default=4096)
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    print(f"[Rank {rank}] host={os.uname().nodename}  model={args.model}")

    if rank == 0:
        run_prefill(args)
    else:
        run_decode(args)
