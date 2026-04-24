#!/bin/bash
# =============================================================================
# Inter-Node Prefill-Decode Disaggregation via vLLM NixlConnector
# =============================================================================
#
# Architecture (two nodes, one GPU each)
# ----------------------------------------
# Node 0  (SLURM_PROCID=0): prefill instance — KV producer, port 8100
# Node 1  (SLURM_PROCID=1): decode  instance — KV consumer, port 8200
#                            + built-in two-phase prefill→decode client flow
#
# srun launches pd_dis.py on BOTH nodes simultaneously.  Each node detects
# its role from SLURM_PROCID and manages only its own vLLM process.
# Prompt requests are coordinated by rank 1, which reaches prefill over
# PREFILL_HOST and then decodes locally.
#
# KV cache transport
# ------------------
# NixlConnector uses NIXL with UCX as the default transport backend.
# For inter-node InfiniBand / GPU-Direct RDMA:
#   UCX_TLS=rc,ud,cuda_ipc,cuda_copy,tcp
#   UCX_NET_DEVICES=mlx5_0:1   (set to your IB HCA; leave "all" to auto-detect)
# NixlConnector does NOT use NCCL env vars.
#
# Side-channel (VLLM_NIXL_SIDE_CHANNEL_*)
# ----------------------------------------
# Lightweight TCP socket used ONLY for the initial NIXL handshake.
# Each node binds its own side-channel host with the configured port.
#
# Usage
#   sbatch pd_dis.sh
#   sbatch pd_dis.sh --model Qwen/Qwen2.5-7B-Instruct --max-tokens 256
#   sbatch pd_dis.sh --dataset aime25 --max-tokens 16384
#   sbatch pd_dis.sh --dataset lveval --dataset-subset hotpotqa_en --max-tokens 512
#   sbatch pd_dis.sh --dataset lveval --dataset-n-samples 100 --dataset-seed 42
#
# Dataset options (forwarded to pd_dis.py):
#   --dataset none|lveval|aime25   (default: none → use --prompts-file)
#   --dataset-subset <name>        LVEval subset, e.g. hotpotqa_en (default)
#   --dataset-n-samples <N>        Instances to sample (default: 100)
#   --dataset-seed <N>             Random seed for sampling (default: 42)
#   --output-dir <path>            Write per_prompt_*.csv + summary_*.csv here
#
# Metrics reported (inline + parseable by eval/parse_results.py):
#   TTFT mean/p50/p95/p99, E2E mean/p50/p95/p99, prefill mean/p95,
#   TPOT ms/tok, throughput tok/s, error rate, accuracy (dataset mode)
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --job-name=pd_disagg_nixl
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=48G
#SBATCH --output=../log/pd_disagg_%j.txt

# check nodes:   sinfo -o "%50N %10P %20G %10T" | grep -E "idle|mixed|allocated" | grep -v "null"
# check account: sacctmgr show associations user=$USER format=User,Account,Partition,QOS

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup  (runs on the submission node before srun)
# ---------------------------------------------------------------------------
module purge
# GCC must load BEFORE CUDA so flashinfer's JIT compile (triggered by
# --kv-cache-dtype fp8_*) can find cc1plus on the compute node.
module load GCC/13.2.0
module load CUDA/12.4.1

# Locate conda env and HF cache. Prefer the current user's scratch if it
# has the env, otherwise fall back to the shared team location under zf28.
_COMP529_USER_ROOT="/scratch/$USER/comp529"
_COMP529_SHARED_ROOT="/scratch/zf28/comp529"
if [[ -d "$_COMP529_USER_ROOT/miniconda3/envs/nlp" ]]; then
    _COMP529_ROOT="$_COMP529_USER_ROOT"
else
    _COMP529_ROOT="$_COMP529_SHARED_ROOT"
fi
CONDA_ENV="$_COMP529_ROOT/miniconda3/envs/nlp"
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME="$_COMP529_ROOT/cache"
# Redirect flashinfer / torch / triton JIT caches off HOME (10 GB quota).
mkdir -p "/scratch/$USER/comp529/cache_local" 2>/dev/null
export XDG_CACHE_HOME="/scratch/$USER/comp529/cache_local"

# Suppress XALT executable tracking (injects a stale libcrypto into LD_PRELOAD)
export XALT_EXECUTABLE_TRACKING=no
unset LD_PRELOAD

# ---------------------------------------------------------------------------
# UCX / NIXL transport settings
# NixlConnector uses UCX — NCCL env vars have NO effect here.
# rc/ud = InfiniBand RDMA transports (best for inter-node KV transfer)
# cuda_ipc/cuda_copy = GPU peer-access fallbacks
# tcp = TCP/IP fallback
# ---------------------------------------------------------------------------
export UCX_TLS=rc,ud,cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=all    # override with e.g. mlx5_0:1 for a specific IB HCA
export UCX_MEMTYPE_CACHE=n    # avoids false-positive GPU-memory detects

# ---------------------------------------------------------------------------
# Topology: first node in the allocation is always the prefill node.
# Export so pd_dis.py can read it on both nodes.
# ---------------------------------------------------------------------------
export PREFILL_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# ---------------------------------------------------------------------------
# srun launches pd_dis.py on every node (SLURM_PROCID=0 on node 0,
# SLURM_PROCID=1 on node 1).  Each node runs independently — rank 0
# starts the prefill vLLM server, rank 1 starts decode + built-in client flow.
#
# IMPORTANT: under sbatch, this shell script may execute from Slurm's spool
# directory (/var/spool/slurmd/job*/). Use SLURM_SUBMIT_DIR so we reference
# files in the original submit location.
# ---------------------------------------------------------------------------
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_PATH="${SCRIPT_DIR}/pd_dis.py"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
	echo "[ERROR] Cannot find ${SCRIPT_PATH}" >&2
	echo "        Submit from the directory containing pd_dis.py, or update SCRIPT_DIR." >&2
	exit 1
fi

srun --chdir "${SCRIPT_DIR}" python3 "${SCRIPT_PATH}" "$@"
