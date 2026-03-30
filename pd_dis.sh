#!/bin/bash
# =============================================================================
# Inter-Node Prefill-Decode Disaggregation via vLLM NixlConnector
# =============================================================================
#
# Architecture
# ------------
# Node 0 (SLURM_PROCID=0): prefill instance  — KV producer, port 8100
# Node 1 (SLURM_PROCID=1): decode  instance  — KV consumer, port 8200
#                           + inline proxy    — port 8000
#
# KV cache transport
# ------------------
# NixlConnector uses NIXL (NVIDIA Inference Xfer Library) with UCX as the
# default transport backend.  For InfiniBand / GPU-Direct RDMA set:
#   UCX_TLS=rc,ud,cuda_ipc,cuda_copy   (or "all" to let UCX auto-select)
#   UCX_NET_DEVICES=mlx5_0:1           (set to your IB HCA)
# Unlike P2pNcclConnector, NixlConnector does NOT use NCCL env vars.
#
# kv_role is set to "kv_both" on both instances — NixlConnector ignores the
# role; actual P/D routing is handled by the toy_proxy_server.py above.
#
# Side-channel (VLLM_NIXL_SIDE_CHANNEL_*)
# ----------------------------------------
# The side channel is a lightweight TCP socket used only for the initial NIXL
# handshake (exchanging memory descriptors).  It is NOT the data path.
# Both instances must be able to reach the prefill node's side-channel port.
#
# Usage
#   sbatch pd_dis.sh [--model <hf-model-id>] [--max-tokens <n>]
#                    [--max-model-len <n>] [--no-warmup]
#
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --job-name=pd_disagg_nixl
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:lovelace:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=../log/pd_disagg_%j.txt

# check nodes:   sinfo -o "%50N %10P %20G %10T" | grep -E "idle|mixed|allocated" | grep -v "null"
# check account: sacctmgr show associations user=$USER format=User,Account,Partition,QOS

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse optional arguments passed via sbatch
# ---------------------------------------------------------------------------
MODEL="Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS=128
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.8
BLOCK_SIZE=128          # larger block size reduces KV-transfer overhead
WARMUP=true

ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)           MODEL="$2";             shift 2 ;;
        --max-tokens)      MAX_TOKENS="$2";        shift 2 ;;
        --max-model-len)   MAX_MODEL_LEN="$2";     shift 2 ;;
        --no-warmup)       WARMUP=false;            shift   ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Topology — first node in the SLURM allocation is always the prefill node
# ---------------------------------------------------------------------------
PREFILL_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000

# NIXL side-channel port (TCP, used only for the initial NIXL handshake).
# Must be reachable from the decode node to the prefill node.
NIXL_SIDE_CHANNEL_PORT=5559

# Prompts file — same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTS_FILE="${SCRIPT_DIR}/prompts.txt"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load CUDA/12.4.1

CONDA_ENV=/scratch/$USER/comp529/miniconda3/envs/nlp
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"

# Cache HuggingFace models to project scratch space instead of home quota
export HF_HOME=/scratch/$USER/comp529/cache

# ---------------------------------------------------------------------------
# UCX / NIXL transport settings
#
# NixlConnector uses UCX (not NCCL) for data transport.
# UCX_TLS selects the transport layers:
#   rc          — InfiniBand Reliable Connection (RDMA, best for inter-node)
#   ud          — InfiniBand Unreliable Datagram
#   cuda_ipc    — intra-node GPU-to-GPU (NVLink / PCIe peer)
#   cuda_copy   — staged copy through host if peer access unavailable
#   tcp         — TCP/IP fallback
# Set UCX_NET_DEVICES to your IB HCA (e.g. mlx5_0:1) or leave as "all".
# ---------------------------------------------------------------------------
export UCX_TLS=rc,ud,cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=all         # override with e.g. mlx5_0:1 if needed
export UCX_MEMTYPE_CACHE=n         # avoids false-positive GPU-memory detects

# Suppress XALT executable tracking (injects a stale libcrypto into LD_PRELOAD)
export XALT_EXECUTABLE_TRACKING=no
unset LD_PRELOAD

# ---------------------------------------------------------------------------
# Helper: wait for a vLLM /health endpoint
# ---------------------------------------------------------------------------
wait_for_server() {
    local host="$1"
    local port="$2"
    local label="$3"
    echo "[${label}] Waiting for http://${host}:${port}/health ..."
    timeout 1200 bash -c "
        until curl -sf http://${host}:${port}/health > /dev/null 2>&1; do
            sleep 3
        done" && echo "[${label}] Ready." || {
        echo "[${label}] ERROR: server did not start within 1200 s" >&2
        exit 1
    }
}

# ---------------------------------------------------------------------------
# NixlConnector KV-transfer config
# kv_role is "kv_both" on both sides — the proxy determines actual P/D roles.
# kv_buffer_device="cuda" keeps the transfer buffer in GPU VRAM (faster).
# ---------------------------------------------------------------------------
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda"}'

# ---------------------------------------------------------------------------
# Branch on SLURM process rank
# ---------------------------------------------------------------------------
# The batch script runs once on the first node with SLURM_PROCID unset.
# Re-launch via srun so every node gets a task with SLURM_PROCID set correctly.
if [[ -z "${SLURM_PROCID+x}" ]]; then
    exec srun --ntasks="${SLURM_NNODES:-2}" --ntasks-per-node=1 \
         bash "${BASH_SOURCE[0]}" "${ORIG_ARGS[@]}"
fi

RANK=${SLURM_PROCID:-0}
MY_HOST=$(hostname -f)       # use FQDN for reliable inter-node resolution

echo "[Rank ${RANK}] host=${MY_HOST}  model=${MODEL}"

# ===========================================================================
# RANK 0 — Prefill node (KV producer)
# ===========================================================================
if [[ "$RANK" -eq 0 ]]; then

    echo "[Prefill] Starting vLLM prefill instance on port ${PREFILL_PORT} ..."
    echo "[Prefill] NIXL side-channel: ${MY_HOST}:${NIXL_SIDE_CHANNEL_PORT}"
    echo "[Prefill] KV config: ${KV_CONFIG}"

    # VLLM_NIXL_SIDE_CHANNEL_HOST — the IP/hostname the prefill instance
    # binds the side-channel listener on.  The decode node uses this address
    # to initiate the NIXL handshake.
    # VLLM_NIXL_SIDE_CHANNEL_PORT — TCP port for that listener.
    VLLM_KV_CACHE_LAYOUT=HND \
    VLLM_NIXL_SIDE_CHANNEL_HOST="${MY_HOST}" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT}" \
    UCX_TLS="${UCX_TLS}" \
    UCX_NET_DEVICES="${UCX_NET_DEVICES}" \
    exec vllm serve "${MODEL}" \
        --host 0.0.0.0 \
        --port "${PREFILL_PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --block-size "${BLOCK_SIZE}" \
        --dtype float16 \
        --enforce-eager \
        --kv-transfer-config "${KV_CONFIG}"
    # exec replaces this shell — nothing below runs on rank 0.
fi

# ===========================================================================
# RANK 1 — Decode node (KV consumer) + proxy
# ===========================================================================

echo "[Decode] Starting vLLM decode instance on port ${DECODE_PORT} ..."
echo "[Decode] Prefill node: ${PREFILL_HOST}:${PREFILL_PORT}"
echo "[Decode] KV config: ${KV_CONFIG}"

# The decode instance must know where to find the prefill side-channel so
# it can initiate the NIXL handshake.
VLLM_KV_CACHE_LAYOUT=HND \
VLLM_NIXL_SIDE_CHANNEL_HOST="${PREFILL_HOST}" \
VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT}" \
UCX_TLS="${UCX_TLS}" \
UCX_NET_DEVICES="${UCX_NET_DEVICES}" \
vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${DECODE_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --block-size "${BLOCK_SIZE}" \
    --dtype float16 \
    --enforce-eager \
    --kv-transfer-config "${KV_CONFIG}" &

DECODE_PID=$!

# ---------------------------------------------------------------------------
# Wait for both vLLM instances to be healthy before starting the proxy
# ---------------------------------------------------------------------------
wait_for_server "${PREFILL_HOST}" "${PREFILL_PORT}" "Prefill"
wait_for_server "localhost"       "${DECODE_PORT}"  "Decode"

# ---------------------------------------------------------------------------
# Start the vLLM toy_proxy_server
#
# toy_proxy_server.py (from vllm/tests/v1/kv_connector/nixl_integration/)
# is the canonical NixlConnector proxy.  It:
#   1. Receives a request from the client.
#   2. Forwards it to a prefill instance (which builds + sends KV cache).
#   3. Forwards it (with the prefill request_id) to a decode instance.
#   4. The decode instance waits for the KV cache, then generates tokens.
#   5. The proxy returns the decode response to the client.
#
# We assume toy_proxy_server.py is on PYTHONPATH / in the vLLM repo checkout.
# Adjust PROXY_SCRIPT if it lives elsewhere.
# ---------------------------------------------------------------------------
PROXY_SCRIPT="${SCRIPT_DIR}/toy_proxy_server.py"
if [[ ! -f "$PROXY_SCRIPT" ]]; then
    # Fall back to the copy inside the vLLM repo if available
    VLLM_REPO=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)
    PROXY_SCRIPT="${VLLM_REPO}/../tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
fi

if [[ ! -f "$PROXY_SCRIPT" ]]; then
    echo "[Proxy] ERROR: toy_proxy_server.py not found." \
         "Copy it from vllm/tests/v1/kv_connector/nixl_integration/ or set PROXY_SCRIPT." >&2
    kill "$DECODE_PID" 2>/dev/null || true
    exit 1
fi

echo "[Proxy] Starting toy_proxy_server on port ${PROXY_PORT} ..."
python3 "${PROXY_SCRIPT}" \
    --port            "${PROXY_PORT}" \
    --prefiller-hosts "${PREFILL_HOST}" \
    --prefiller-ports "${PREFILL_PORT}" \
    --decoder-hosts   "$(hostname -f)" \
    --decoder-ports   "${DECODE_PORT}" &

PROXY_PID=$!

# Give the proxy a moment to bind its socket
sleep 5

echo "[Proxy] Ready on port ${PROXY_PORT}."

# ---------------------------------------------------------------------------
# Optional warmup — triggers the NIXL handshake before real requests land,
# eliminating the one-time handshake latency on the first user prompt.
# ---------------------------------------------------------------------------
if [[ "$WARMUP" == "true" ]]; then
    echo "[Warmup] Sending warmup request ..."
    curl -sf "http://localhost:${PROXY_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"warmup\",\"max_tokens\":4,\"temperature\":0}" \
        > /dev/null && echo "[Warmup] Done." \
        || echo "[Warmup] Warning: warmup request failed (continuing)." >&2
fi

# ---------------------------------------------------------------------------
# Process prompts from file
# ---------------------------------------------------------------------------
if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "[ERROR] Prompts file not found: ${PROMPTS_FILE}" >&2
    kill "$PROXY_PID" "$DECODE_PID" 2>/dev/null || true
    exit 1
fi

mapfile -t PROMPTS < <(grep -v '^#' "$PROMPTS_FILE" | grep -v '^[[:space:]]*$')

if [[ ${#PROMPTS[@]} -eq 0 ]]; then
    echo "[ERROR] No prompts found in ${PROMPTS_FILE}" >&2
    kill "$PROXY_PID" "$DECODE_PID" 2>/dev/null || true
    exit 1
fi

echo ""
echo "[Decode] Processing ${#PROMPTS[@]} prompt(s) ..."

TOTAL_START=$(date +%s%N)

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    IDX=$((i + 1))

    echo ""
    echo "============================================================"
    echo " Prompt ${IDX}/${#PROMPTS[@]}"
    echo "============================================================"
    echo " Input : ${PROMPT}"

    T0=$(date +%s%N)

    RESPONSE=$(curl -sf "http://localhost:${PROXY_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
              \"model\": \"${MODEL}\",
              \"prompt\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${PROMPT}"),
              \"max_tokens\": ${MAX_TOKENS},
              \"temperature\": 0
            }") || {
        echo " ERROR: request failed." >&2
        continue
    }

    T1=$(date +%s%N)
    ELAPSED_MS=$(( (T1 - T0) / 1000000 ))

    OUTPUT=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null || echo "(parse error)")
    PROMPT_TOK=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null || echo "?")
    COMPL_TOK=$(echo  "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens','?'))" 2>/dev/null || echo "?")

    echo " Output  : ${OUTPUT}"
    echo " Tokens  : prompt=${PROMPT_TOK}  completion=${COMPL_TOK}"
    echo " Elapsed : ${ELAPSED_MS} ms"
done

TOTAL_END=$(date +%s%N)
TOTAL_MS=$(( (TOTAL_END - TOTAL_START) / 1000000 ))

echo ""
echo "============================================================"
echo " All ${#PROMPTS[@]} prompt(s) done in ${TOTAL_MS} ms."
echo "============================================================"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
echo "[Decode] Shutting down ..."
kill "$PROXY_PID"  2>/dev/null || true
kill "$DECODE_PID" 2>/dev/null || true
wait "$DECODE_PID" 2>/dev/null || true

# Cancel the SLURM job (also terminates rank 0 / prefill node)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[Decode] Cancelling SLURM job ${SLURM_JOB_ID} ..."
    scancel "${SLURM_JOB_ID}"
fi
