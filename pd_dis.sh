#!/bin/bash
# =============================================================================
# Inter-Node Prefill-Decode Disaggregation via vLLM NixlConnector
# =============================================================================
#
# Architecture (two nodes, one GPU each)
# ----------------------------------------
# Node 0  (SLURM_PROCID=0): prefill instance — KV producer, port 8100
# Node 1  (SLURM_PROCID=1): decode  instance — KV consumer, port 8200
#                            + proxy (pd_dis.py)             port 8000
#
# srun launches pd_dis.py on BOTH nodes simultaneously.  Each node detects
# its role from SLURM_PROCID and manages only its own vLLM process.
# The proxy runs only on the decode node and reaches the prefill node over
# the network via PREFILL_HOST.
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
# Both nodes point at the prefill node's hostname and side-channel port.
#
# Usage
#   sbatch pd_dis.sh
#   sbatch pd_dis.sh --model Qwen/Qwen2.5-7B-Instruct --max-tokens 256
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --job-name=pd_disagg_nixl
#SBATCH --time=01:00:00
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
module load CUDA/12.4.1

CONDA_ENV=/scratch/$USER/comp529/miniconda3/envs/nlp
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME=/scratch/$USER/comp529/cache

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
# starts the prefill vLLM server, rank 1 starts decode + proxy.
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
