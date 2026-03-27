import os
import torch
import torch.distributed as dist
import time

def run_prefill(rank, size):
    """
    Prefill Stage: Processes prompt and produces KV cache[cite: 16].
    """
    print(f"[Rank {rank}] Role: PREFILL NODE")
    
    # Simulate KV cache generation (size depends on sequence length S, layers L, etc.)
    # Example: 1 layer, seq_len 1024, head_dim 128, 32 heads
    # KV bytes ≈ 2 * L * S * H * D * b 
    kv_cache = torch.randn(1024, 32, 128, device='cuda', dtype=torch.float16)
    
    print(f"[Rank {rank}] Prefill compute complete. Starting KV transfer...")
    
    start_time = time.perf_counter()
    # Send KV cache to Rank 1 (Decode Node) [cite: 34, 51]
    dist.send(tensor=kv_cache, dst=1)
    end_time = time.perf_counter()
    
    transfer_time = end_time - start_time
    print(f"[Rank {rank}] KV Transfer Sent in {transfer_time:.4f}s")

def run_decode(rank, size):
    """
    Decode Stage: Receives KV cache and performs autoregressive generation[cite: 18, 34].
    """
    print(f"[Rank {rank}] Role: DECODE NODE")
    
    # Initialize empty buffer to receive KV cache
    kv_buffer = torch.zeros(1024, 32, 128, device='cuda', dtype=torch.float16)
    
    print(f"[Rank {rank}] Waiting for KV cache from Prefill node...")
    
    start_time = time.perf_counter()
    # Receive KV cache from Rank 0 [cite: 21, 51]
    dist.recv(tensor=kv_buffer, src=0)
    end_time = time.perf_counter()
    
    print(f"[Rank {rank}] KV Transfer Received in {end_time - start_time:.4f}s")
    print(f"[Rank {rank}] Starting Decode/Autoregressive generation...")
    # Simulate decode step
    time.sleep(0.5) 
    print(f"[Rank {rank}] Generation complete.")

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Use NCCL for GPU-to-GPU communication (typically uses RDMA/InfiniBand) 
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    # Slurm sets SLURM_PROCID for each task launched via srun
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['SLURM_PROCID'])
    
    if rank == 0:
        init_process(rank, world_size, run_prefill)
    else:
        init_process(rank, world_size, run_decode)