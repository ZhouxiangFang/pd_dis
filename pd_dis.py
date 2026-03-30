import os
import argparse
import torch
import torch.distributed as dist
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TEST_PROMPT = "Explain what prefill-decode disaggregation means in LLM inference:"
MAX_NEW_TOKENS = 50

def run_prefill(rank, size):
    """
    Prefill Stage: Processes prompt and produces KV cache using Qwen2.5-7B-Instruct.
    """
    print(f"[Rank {rank}] Role: PREFILL NODE")
    print(f"[Rank {rank}] Loading model {MODEL_NAME}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print(f"[Rank {rank}] Model loaded. Starting prefill with prompt...")
    print(f"[Rank {rank}] Prompt: {TEST_PROMPT}")
    
    # Tokenize input
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    
    prefill_start = time.perf_counter()
    
    # Run prefill: forward pass to generate KV cache
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values  # KV cache from all layers
    
    prefill_end = time.perf_counter()
    print(f"[Rank {rank}] Prefill compute complete in {prefill_end - prefill_start:.4f}s")
    print(f"[Rank {rank}] Input tokens: {input_ids.shape[1]}")
    
    # Prepare KV cache for transfer
    # past_key_values is a tuple of (key, value) pairs per layer
    # We need to flatten and send them
    kv_list = []
    for layer_kv in past_key_values:
        kv_list.append(layer_kv[0])  # keys
        kv_list.append(layer_kv[1])  # values
    
    print(f"[Rank {rank}] Starting KV transfer ({len(kv_list)} tensors)...")
    
    transfer_start = time.perf_counter()
    # Send metadata first
    metadata = torch.tensor([len(kv_list), input_ids.shape[1]], dtype=torch.long, device='cuda')
    dist.send(tensor=metadata, dst=1)
    
    # Send input_ids
    dist.send(tensor=input_ids, dst=1)
    
    # Send each KV tensor
    for kv_tensor in kv_list:
        dist.send(tensor=kv_tensor.contiguous(), dst=1)
    
    transfer_end = time.perf_counter()
    
    transfer_time = transfer_end - transfer_start
    total_bytes = sum(kv.element_size() * kv.numel() for kv in kv_list)
    print(f"[Rank {rank}] KV Transfer Sent in {transfer_time:.4f}s ({total_bytes / 1e9:.2f} GB)")

def run_decode(rank, size):
    """
    Decode Stage: Receives KV cache and performs autoregressive generation with Qwen2.5-7B-Instruct.
    """
    print(f"[Rank {rank}] Role: DECODE NODE")
    print(f"[Rank {rank}] Loading model {MODEL_NAME}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print(f"[Rank {rank}] Model loaded. Waiting for KV cache from Prefill node...")
    
    transfer_start = time.perf_counter()
    
    # Receive metadata
    metadata = torch.zeros(2, dtype=torch.long, device='cuda')
    dist.recv(tensor=metadata, src=0)
    num_kv_tensors = metadata[0].item()
    seq_len = metadata[1].item()
    
    # Receive input_ids
    input_ids = torch.zeros(1, seq_len, dtype=torch.long, device='cuda')
    dist.recv(tensor=input_ids, src=0)
    
    # Receive KV cache tensors
    # We need to know the shape - for Qwen2.5-7B it has specific architecture
    # Typically: (batch, num_heads, seq_len, head_dim)
    # We'll receive and reconstruct
    kv_list = []
    num_layers = num_kv_tensors // 2
    
    # Get model config to determine shapes
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    
    for _ in range(num_kv_tensors):
        kv_tensor = torch.zeros(1, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        dist.recv(tensor=kv_tensor, src=0)
        kv_list.append(kv_tensor)
    
    # Reconstruct past_key_values tuple
    past_key_values = tuple(
        (kv_list[i*2], kv_list[i*2+1]) for i in range(num_layers)
    )
    
    transfer_end = time.perf_counter()
    print(f"[Rank {rank}] KV Transfer Received in {transfer_end - transfer_start:.4f}s")
    
    print(f"[Rank {rank}] Starting Decode/Autoregressive generation...")
    decode_start = time.perf_counter()
    
    # Generate tokens autoregressively using received KV cache
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # Use past_key_values for faster generation
            outputs = model(
                input_ids=generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            past_key_values = outputs.past_key_values
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    decode_end = time.perf_counter()
    
    # Decode and print generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"[Rank {rank}] Generation complete in {decode_end - decode_start:.4f}s")
    print(f"[Rank {rank}] Generated {generated_ids.shape[1] - input_ids.shape[1]} tokens")
    print(f"[Rank {rank}] Generated text:\n{generated_text}")

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Use NCCL for GPU-to-GPU communication (typically uses RDMA/InfiniBand) 
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def parse_args():
    parser = argparse.ArgumentParser(description="Prefill-decode disaggregation demo")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        default=TEST_PROMPT,
        help="Prompt used for prefill stage",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens generated during decode",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model
    TEST_PROMPT = args.prompt
    MAX_NEW_TOKENS = args.max_new_tokens

    # Slurm sets SLURM_PROCID for each task launched via srun
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['SLURM_PROCID'])
    
    if rank == 0:
        init_process(rank, world_size, run_prefill)
    else:
        init_process(rank, world_size, run_decode)
