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
- `eval/` — sweep drivers + result aggregators (see below)
- `results/` — CSV output directory (created automatically)

---

## Compression and pipelining sweeps

`--kv-cache-dtype` and `--block-size` are the two knobs that control the
compression and pipelining optimization axes from the proposal. Each is a
plain passthrough to `vllm serve` and works with the default NixlConnector
path — no LMCache / CacheGen needed.

| Flag | Default | Values | What it does |
|---|---|---|---|
| `--kv-cache-dtype` | `auto` (fp16) | `auto`, `fp8_e5m2`, `fp8_e4m3` | Stores KV cache in fp8. NixlConnector ships fp8 bytes over UCX → ~2× on-wire compression. Costs ~1 ms/output-token of dequant in attention kernels. |
| `--block-size` | `1024` | any power-of-2 in `[16, 2048]` | vLLM KV paging unit AND NIXL transfer granularity. Smaller = more transfer/compute overlap (pipelining) but higher per-transfer overhead. |

> **Note (fp8 path prerequisite):** fp8 triggers flashinfer JIT compilation of
> attention kernels, which requires `cc1plus`. `pd_dis.sh` loads `GCC/13.2.0`
> and `CUDA/12.4.1` before running; the flashinfer cache is redirected to
> `/scratch/$USER/comp529/cache_local` to avoid HOME quota pressure.

### Single-run examples

```bash
# Baseline (fp16 KV, block-size 1024) — same as the vanilla commands above.
sbatch pd_dis.sh --dataset aime25 --max-tokens 16384

# Compression only
sbatch pd_dis.sh --dataset aime25 --max-tokens 16384 --kv-cache-dtype fp8_e5m2

# Pipelining only (smaller blocks = more transfer/compute overlap)
sbatch pd_dis.sh --dataset aime25 --max-tokens 16384 --block-size 128

# Combined
sbatch pd_dis.sh --dataset lveval --dataset-len 16k --max-tokens 512 \
                 --kv-cache-dtype fp8_e5m2 --block-size 128
```

### Ready-to-run sweep scripts

Under `eval/` there are two drivers that submit one sbatch job per cell
and collect the results into one directory. Each cell writes its own
`summary_*.csv` + `per_prompt_*.csv`, plus a `manifest.tsv` for the sweep.

#### `eval/sweep_compression.sh` — fp8_e5m2 vs fp16 on both datasets

```bash
# 4 jobs: (aime25, lveval) × (baseline, fp8_e5m2), 30 samples each
./eval/sweep_compression.sh

# Only one dataset
./eval/sweep_compression.sh aime25
./eval/sweep_compression.sh lveval

# Override sample count / LVEval context bucket
N_SAMPLES=50 ./eval/sweep_compression.sh
LVEVAL_LEN=32k ./eval/sweep_compression.sh lveval
```

#### `eval/sweep_pipelining.sh` — block-size sweep on both datasets

```bash
# 12 jobs: (aime25, lveval) × block_size ∈ {16, 32, 64, 128, 256, 1024}
./eval/sweep_pipelining.sh

# Only AIME25
./eval/sweep_pipelining.sh aime25

# Custom sweep
BLOCK_SIZES="128 256 1024" ./eval/sweep_pipelining.sh lveval
```

Each sweep script writes to `eval/results/compression_<ts>/` or
`eval/results/pipelining_<ts>/`, with one sub-directory per cell and a
`jobs.txt` for bulk cancel:

```bash
# Watch all jobs from a sweep
squeue -u $USER -j $(paste -sd, eval/results/compression_*/jobs.txt)

# Cancel them
scancel $(cat eval/results/compression_<ts>/jobs.txt)
```

#### Aggregate results

```bash
# Combine every summary_*.csv in a sweep directory into one markdown table + CSV
python3 eval/collect_summaries.py eval/results/compression_20260423_153805

# Pick which columns to include in the markdown table
python3 eval/collect_summaries.py eval/results/pipelining_20260423_153805 \
    --columns dataset tag ttft_mean ttft_p99 e2e_mean tpot_mean_ms accuracy
```

Combine compression + pipelining + baseline into one view:

```bash
mkdir -p eval/results/combined_all
for d in eval/results/compression_*/eval/results/pipelining_*/; do
    cp -r "$d"* eval/results/combined_all/
done
python3 eval/collect_summaries.py eval/results/combined_all
```

### Interpreting the trade-offs

Both axes are **trade-offs, not strict wins**:

- **Compression (fp8_e5m2)** — KV storage halves, on-wire transfer halves,
  but attention pays fp8 dequant per-decode-token. Net-positive for TTFT
  on prompt-heavy workloads (transfer-dominated); often net-negative for
  end-to-end throughput on decode-heavy workloads.
- **Pipelining (smaller `--block-size`)** — more chunks = more
  transfer/decode overlap, but more NIXL round-trips. Sweet spot is
  workload-dependent; empirically `block_size=32` looked best for
  long-context prompt-heavy Qwen2.5-3B at 2500-token context, while
  `block_size=256` minimized E2E for 1500-token context.

See [notes/tabs/2026-04-23_pd-dis-compression-pipelining.md](notes/tabs/2026-04-23_pd-dis-compression-pipelining.md)
for the full measurement table on the prior-generation synthetic workloads.
