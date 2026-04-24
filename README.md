# Prefill-Decode Disaggregation (vLLM + NixlConnector)

Two-node disaggregated serving, one GPU per node:

- **Node 0** (prefill, `kv_producer`): port `8100`
- **Node 1** (decode, `kv_consumer`): port `8200`

KV cache is transferred between nodes via **vLLM NixlConnector over UCX** (not NCCL).

---

## Quick start

### AIME25 — generation-heavy (long reasoning outputs)

Each problem requires up to 16k output tokens of chain-of-thought reasoning.
The decode node is the bottleneck; the prefill node is mostly idle.

```bash
sbatch pd_dis.sh \
  --dataset aime25 \
  --max-tokens 16384 \
  --concurrency 4
```

AIME25 has exactly 30 problems, so `--dataset-n-samples` is capped at 30 automatically.
`--concurrency 4` submits 4 requests in parallel, keeping the decode GPU batched and busy.

### LVEval — prompt-heavy (long context, short answers)

Each sample has a long document context (~16k–256k tokens) with a short factoid answer.
The prefill node is the bottleneck; decode is fast.

```bash
# Default: hotpotwikiqa_mixup, 16k context, 100 samples
sbatch pd_dis.sh \
  --dataset lveval \
  --dataset-len 16k \
  --dataset-n-samples 50 \
  --max-tokens 512

# 32k context, different task
sbatch pd_dis.sh \
  --dataset lveval \
  --dataset-subset multifieldqa_en_mixup \
  --dataset-len 32k \
  --dataset-n-samples 50 \
  --max-tokens 512
```

**`--dataset-len`** choices: `16k` `32k` `64k` `128k` `256k` (default: `16k`)

> **Model context limit**: Qwen3-4B supports at most **40960 tokens** total (prompt + output).
> This means `--dataset-len 32k` is the largest subset that fits safely (32k prompt + ~512 output tokens).
> Using `64k` or above requires a model with a longer context window (e.g. Qwen3-14B supports 131072).

**`--dataset-subset`** base names (without length suffix):

| Subset | Language | Task type |
|---|---|---|
| `hotpotwikiqa_mixup` | EN | multi-hop QA (default) |
| `multifieldqa_en_mixup` | EN | single-hop QA |
| `loogle_SD_mixup` | EN | short-dep QA |
| `loogle_CR_mixup` | EN | comprehension |
| `loogle_MIR_mixup` | EN | multi-inference |
| `factrecall_en` | EN | fact recall |
| `multifieldqa_zh_mixup` | ZH | single-hop QA |
| `cmrc_mixup` | ZH | reading comprehension |
| `dureader_mixup` | ZH | open-domain QA |
| `lic_mixup` | ZH | in-context |

---

## Key options

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-4B` | HuggingFace model name |
| `--max-tokens` | `8192` | Max output tokens per request |
| `--max-model-len` | `40960` | Max total sequence length — **do not exceed 40960 for Qwen3-4B** |
| `--concurrency` | `4` | Parallel requests in flight |
| `--dataset` | `none` | `none` / `lveval` / `aime25` |
| `--dataset-len` | `16k` | LVEval context length bucket |
| `--dataset-subset` | `hotpotwikiqa_mixup` | LVEval base task name |
| `--dataset-n-samples` | `100` | Number of examples to sample |
| `--dataset-seed` | `42` | Random seed for sampling |
| `--pruning-method` | `none` | Prompt pruning: `none` / `attn_proxy` / `random` |
| `--pruning-keep-ratio` | `1.0` | Fraction of tokens to keep after pruning |
| `--no-warmup` | — | Skip the warmup request |

---

## Output statistics

After all prompts finish, the script prints three blocks:

### Per-prompt metrics
```
[1] OK | prompt='...' | completion=512 | ttft=1.234s | tpot=14.2ms | total=8.512s | prefill=0.210s
```
- **ttft**: time from request start to first output token (measures prefill + KV transfer latency)
- **tpot**: ms per output token (measures decode throughput)
- **total**: wall time for the full request
- **prefill**: HTTP round-trip to the prefill node only

### Aggregate metrics
```
TTFT           mean=1.234s  p50=1.1s  p95=2.3s  p99=3.1s
E2E latency    mean=8.5s    p50=8.1s  p95=12.s  p99=15.s
Prefill time   mean=0.21s   p50=0.20s p95=0.31s p99=0.35s
TPOT (ms/tok)  mean=14.2ms
Throughput     312.4 tok/s  (total_ctok=15620, wall=50.0s)
```

### Accuracy (dataset mode)
- **AIME25**: exact-match on the final integer answer extracted from `\boxed{}`
- **LVEval**: token-level F1 against reference answers (threshold F1 ≥ 0.3 counted as correct)

---

## CSV output

Results are saved to `pd_dis/results/` (override with `--output-dir`):

```
results/
  per_prompt_<model>_<dataset>_<timestamp>.csv   # one row per prompt
  summary_<model>_<dataset>_<timestamp>.csv      # one row for the whole run
```

The summary CSV contains all aggregate metrics (TTFT mean/p50/p95/p99, E2E, prefill, TPOT, throughput, accuracy) in a single row, suitable for comparison across runs.

---

## Files

- `pd_dis.sh` — SLURM launcher, UCX/NIXL environment setup
- `pd_dis.py` — prefill/decode orchestration, dataset loading, metrics reporting
- `prompts.txt` — custom prompts used when `--dataset none` (one per line, `#` for comments)
- `methods/` — prompt pruning implementations (`attn_proxy`, `random`)
- `results/` — CSV output directory (created automatically)
