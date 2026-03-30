# Prefill-Decode Disaggregation with Qwen2.5-7B-Instruct

The python version is 3.12. 

There might be a assert error for the P2pNcclConnector. Modification for that in vllm lib is needed.

## Default Model
- **Model**: Qwen/Qwen2.5-7B-Instruct (7 billion parameter instruct-tuned model)
- **Location**: Lines 7-8 in `pd_dis.py`

## Implementation

#### Prefill Stage (Rank 0)
1. Loads the Qwen2.5-7B-Instruct model
2. Tokenizes the input prompt
3. Performs forward pass to generate KV cache
4. Transfers KV cache to decode node via NCCL
5. Measures prefill compute time and transfer time

### Decode Stage (Rank 1)
1. Loads the Qwen2.5-7B-Instruct model
2. Receives KV cache from prefill node
3. Performs autoregressive token generation using the received KV cache
4. Generates up to 50 new tokens
5. Outputs the complete generated text

## Requirements
Updated `requirements.txt` includes:
- transformers>=4.37.0
- torch>=2.0.0
- accelerate>=0.20.0
- vllm

## Usage
```bash
sbatch pd_dis.sh
```

The script will:
1. Allocate 2 nodes with 1 GPU each
2. Run prefill on node 0
3. Run decode on node 1
4. Transfer KV cache between nodes
5. Generate text output
