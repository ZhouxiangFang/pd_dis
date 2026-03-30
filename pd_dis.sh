#!/bin/bash
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --job-name=pd_disagg_vllm
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:lovelace:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../log/pd_disagg_%j.txt

# check nodes:   sinfo -o "%20N %15P %20G %10T" | grep -v "down"
# check account: sacctmgr show associations user=$USER format=User,Account,Partition,QOS

module purge
module load CUDA/12.4.1

# Topology — prefill node is always the first node in the allocation
export PREFILL_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export PREFILL_PORT=8100
export DECODE_PORT=8200
export PROXY_PORT=8000
export KV_PORT=14579        # PyNcclConnector rendezvous port on the prefill node

# ---------------------------------------------------------------------------
# NCCL / InfiniBand / GPU Direct RDMA settings
#
# PyNcclConnector uses NCCL for KV cache transfer.  NCCL will automatically
# select InfiniBand when the IB verbs stack is present, but the variables
# below force RDMA on and tune GDR (GPU Direct RDMA) so tensor data moves
# directly GPU→IB HCA→GPU without staging through host memory.
#
# NCCL_IB_DISABLE=0        – keep IB enabled (overrides any system default)
# NCCL_NET_GDR_LEVEL=5     – GPU Direct RDMA for all transfers (SYS = 5)
# NCCL_IB_GDR_LEVEL=5      – GDR on the IB transport specifically
# NCCL_NET_GDR_READ=1      – enable GDR *reads* (not just writes)
# NCCL_IB_HCA              – comma-list of HCAs to use; leave unset to let
#                            NCCL auto-detect, or set e.g. mlx5_0,mlx5_1
# NCCL_IB_TC=106           – DSCP traffic class for IB (106 = high-priority)
# NCCL_IB_QPS_PER_CONNECTION=4  – parallel QPs per connection for bandwidth
# ---------------------------------------------------------------------------
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1
export NCCL_IB_TC=106
export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_HCA=mlx5_0,mlx5_1   # uncomment and set to your HCA names if needed

# Route all NCCL traffic over IB (disables TCP/IP fallback to Ethernet)
export NCCL_NET=IB

# Optional: print NCCL transport selection at startup for verification
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

# srun launches one task per node (SLURM_PROCID=0 → prefill, SLURM_PROCID=1 → decode)
srun python pd_dis.py "$@"
