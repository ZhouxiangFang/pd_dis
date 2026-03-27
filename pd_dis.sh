#!/bin/bash
#SBATCH --account=commons
#SBATCH --partition=scavenge
#SBATCH --job-name=pd_disagg
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:lovelace:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/%u/log/pd_disagg_%j.txt


# Load the requested CUDA module
module purge
module load CUDA/12.4.1

# Get the hostname of the master node (Prefill Node)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=2

# Run the python script
# srun will launch one process per node (Rank 0 and Rank 1)
srun python pd_dis.py