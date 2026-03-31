# Prefill-Decode Disaggregation (vLLM + NixlConnector)

This project runs **2-node disaggregated serving** with one GPU per node:

- **Rank 0 / Node 0**: prefill (`--kv-role kv_producer`) on port `8100`
- **Rank 1 / Node 1**: decode (`--kv-role kv_consumer`) on port `8200`

It uses **vLLM NixlConnector over UCX** (not NCCL connector flow).

## Current behavior (important)

- Uses **built-in two-phase prefill → decode** flow in `pd_dis.py`
- Does **not** depend on `toy_proxy_server.py`
- Prompts are processed first, then output is reported in this order:
	1. `SUMMARY`
	2. per-prompt metrics (including errors)
	3. `RESPONSES` (final generated text only)
	4. `AVG METRICS` at the very end

## Metrics reported

For each prompt:

- completion token count
- **TTFT** (time to first token)
- **time per output token** (ms/output-token)
- total time

Global averages at end:

- average TTFT
- average ms/output-token

## Defaults

From `pd_dis.py`:

- `--model Qwen/Qwen2.5-3B-Instruct`
- `--max-tokens 1024`
- `--max-model-len 4096`
- `--gpu-memory-utilization 0.8`
- `--block-size 128`
- `--warmup` enabled by default

## Requirements

- Python 3.12
- vLLM environment available in your conda env
- Slurm allocation with 2 nodes / 1 GPU per node

UCX/NIXL env setup is handled in `pd_dis.sh`.

## Run

```bash
sbatch pd_dis.sh
```

Pass arguments through `sbatch` to override defaults:

```bash
sbatch pd_dis.sh --model Qwen/Qwen2.5-7B-Instruct --max-tokens 256 --no-warmup
```

## Files

- `pd_dis.sh`: Slurm launcher + environment setup
- `pd_dis.py`: prefill/decode orchestration and metrics reporting
- `prompts.txt`: one prompt per line (`#` comments and blank lines ignored)
