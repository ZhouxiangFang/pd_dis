# Prefill-Decode Disaggregation (vLLM + NixlConnector)

## Repository layout

| Path | Purpose |
|------|---------|
| `pd_dis.sh` | SLURM wrapper: modules, UCX/NIXL env, `srun` → `pd_dis.py` |
| `pd_dis.py` | Core orchestration, datasets, metrics, CSV export |
| `pd_dis_chat.sh` / `pd_dis_chat.py` | Chat-template path for report-quality benchmarks |
| `methods/` | Pruning backends (`attn_proxy`, `random`, tests) |
| `eval/` | `launch_full_sweep.sh`, `sweep_compression.sh`, `sweep_pipelining.sh`, `aggregate_sweep.py`, `collect_summaries.py`, … |
| `prompts.txt` | Used when `--dataset none` (one prompt per line, `#` comments) |
| `results/` | Default CSV output if `--output-dir` is not set |

This repository implements **two-node** disaggregated LLM serving with **vLLM
NixlConnector** over **UCX** (InfiniBand–friendly; not NCCL). GPU **0** runs
prefill (`kv_producer`, port **8100**); GPU **1** runs decode (`kv_consumer`,
port **8200**) and issues remote prefill plus KV pull.

We study **three** complementary ways to cut or overlap inter-node KV cost.
Each has a clear **baseline** (no pruning, fp16 KV + default block size); the
sections below show how to run cells for all three axes, then shared flags,
artifacts, and repo layout.

| Axis | Idea | Main flags / drivers |
|------|------|----------------------|
| **Pruning** | Ship fewer prompt tokens by shortening text before prefill. | `--pruning-method`, `--pruning-keep-ratio`; **`pd_dis_chat.sh`**, **`eval/launch_full_sweep.sh`** |
| **Compression** | Fewer bytes per token on the wire (fp8 KV). | `--kv-cache-dtype`; **`eval/sweep_compression.sh`** |
| **Pipelining** | Smaller KV blocks so transfer can overlap decode. | `--block-size`; **`eval/sweep_pipelining.sh`** |

---

## Running workloads

Everything is submitted from the repository root with **`sbatch`**. There are
**two** batch entry points; pick one before copying commands.

| Script | Role |
|--------|------|
| **`pd_dis.sh`** | Raw prompt path into `pd_dis.py` — quick runs and the **compression** / **pipelining** sweeps (`sweep_*.sh`). |
| **`pd_dis_chat.sh`** | Wraps **`pd_dis_chat.py`**: Qwen3 **chat template** (including chain-of-thought). Use for **pruning-axis** AIME25 / LVEval numbers that match the report scorer. |

**Pruning** heuristics (`attn_proxy`, etc.) live in `methods/attn_pruning/`; see
`methods/attn_pruning/README.md`.

### Vanilla runs as Baselines (`pd_dis.sh`)

#### AIME25 — generation-heavy (long reasoning outputs)

Each problem can use up to 16k output tokens of chain-of-thought; the decode
node is usually the bottleneck.

```bash
sbatch pd_dis.sh \
  --dataset aime25 \
  --max-tokens 16384 \
  --concurrency 4
```

AIME25 ships **30** problems; `--dataset-n-samples` is capped at 30. The example
uses `--concurrency 4` to keep the decode GPU busy.

#### LVEval — prompt-heavy (long context, short answers)

Long documents with short factoid answers; prefill + KV transfer dominate.

```bash
# Default subset: hotpotwikiqa_mixup @ 16k (here: 50 samples)
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

**`--dataset-len`**: `16k` `32k` `64k` `128k` `256k` (default `16k`).

> **Model context limit (Qwen3-4B):** max **40960** tokens total (prompt +
> output). For this model, **`--dataset-len 32k`** is about the largest LVEval
> bucket that still leaves headroom for output and template overhead. Longer
> buckets need a bigger-context checkpoint (e.g. Qwen3-14B at 131k).

### Pruning axis — `pd_dis_chat.sh` + `launch_full_sweep.sh`

This is **optimization axis (1)**. We compare a **no-pruning baseline** to the
same stack with decode-side **`attn_proxy`** pruning at **ρ = 0.5** and **ρ =
0.3** (three **input policies** per benchmark, all on fp16 KV and default block
size unless you change them).

**What `eval/launch_full_sweep.sh` covers**

| Dimension | Settings |
|-----------|-----------|
| Pruning cells | **baseline** (`--pruning-method none`), **prune_k05** (`attn_proxy`, `--pruning-keep-ratio 0.5`), **prune_k03** (`attn_proxy`, `0.3`) |
| Benchmarks | **AIME25** (`--max-tokens 16384`, `--concurrency 8`), **LVEval 16k** (`hotpotwikiqa_mixup` @ 16k, 50 samples, `--max-tokens 512`, `--concurrency 8`) |
| Repeats | Two SLURM jobs per cell (`r1`, `r2`) |
| Extras | LVEval **32k** baseline-only pair; **concurrency=1** baselines for AIME25 and LVEval-16k |

**Smoke jobs** (from repo root; add `--exclusive` on `sbatch` inside
`eval/launch_full_sweep.sh` if your cluster sees NIXL side-channel collisions):

```bash
sbatch pd_dis_chat.sh --dataset aime25 --max-tokens 16384 --concurrency 8 \
  --pruning-method none --output-dir eval/results/smoke_aime_baseline

sbatch pd_dis_chat.sh --dataset aime25 --max-tokens 16384 --concurrency 8 \
  --pruning-method attn_proxy --pruning-keep-ratio 0.5 \
  --output-dir eval/results/smoke_aime_k05

sbatch pd_dis_chat.sh --dataset lveval --dataset-len 16k --dataset-n-samples 50 \
  --max-tokens 512 --concurrency 8 --pruning-method none \
  --output-dir eval/results/smoke_lv_baseline
```

**Full launcher** — `eval/launch_full_sweep.sh` submits **16** jobs (12-cell
main matrix + 2× LVEval-32k baseline + 2× c=1 controls), one output directory
per cell under `eval/results/<TAG_PREFIX>/`, plus `manifest.tsv`.

```bash
cd /path/to/pd_dis
./eval/launch_full_sweep.sh
TAG_PREFIX=resweep_$(date +%Y%m%d_%H%M%S) ./eval/launch_full_sweep.sh
```

Whole-node jobs: edit `submit_one` in `eval/launch_full_sweep.sh` and add
`--exclusive` to the `sbatch` line. Extra **driver** flags (not `sbatch`
flags) can be appended via `SWEEP_EXTRA_FLAGS` as documented in that script.

---


## Compression and pipelining sweeps

These are optimization axes **(2) compression** and **(3) pipelining** on the
same disaggregated baseline: knobs are passed straight into **`vllm serve`** as
**`--kv-cache-dtype`** (fp8 KV storage) and **`--block-size`** (KV page size =
NIXL transfer chunk size). No LMCache / CacheGen layer. Typical cells compare
**fp16 / default block** against fp8 and/or smaller blocks via
`eval/sweep_compression.sh` and `eval/sweep_pipelining.sh` (see scripts for exact
matrices).

| Flag | Default | Values | Effect |
|------|---------|--------|--------|
| `--kv-cache-dtype` | `auto` (fp16) | `auto`, `fp8_e5m2`, `fp8_e4m3` | Halves KV bytes on the wire for fp8; decode pays dequant in attention. |
| `--block-size` | `1024` | powers of two in `[16, 2048]` | Smaller blocks increase transfer/decode overlap but add per-chunk overhead. |

> **fp8 prerequisite:** flashinfer JIT needs **`cc1plus`**. The batch scripts load
> **GCC/13.2.0** and **CUDA/12.4.1**; JIT caches are steered under
> **`/scratch/$USER/comp529/cache_local`** to avoid HOME quota issues.

### Single-job examples (`pd_dis.sh`)

```bash
sbatch pd_dis.sh --dataset aime25 --max-tokens 16384
sbatch pd_dis.sh --dataset aime25 --max-tokens 16384 --kv-cache-dtype fp8_e5m2
sbatch pd_dis.sh --dataset aime25 --max-tokens 16384 --block-size 128
sbatch pd_dis.sh --dataset lveval --dataset-len 16k --max-tokens 512 \
                 --kv-cache-dtype fp8_e5m2 --block-size 128
```

### Batch sweep scripts

Each sweep writes **`eval/results/<compression|pipelining>_<timestamp>/`** with
one subdirectory per cell, `summary_*.csv`, `per_prompt_*.csv`, and
`jobs.txt` for bulk `scancel`.

**`eval/sweep_compression.sh`** — fp16 vs fp8_e5m2 on AIME25 and LVEval (default
**30** samples per cell, overridable with `N_SAMPLES`).

```bash
./eval/sweep_compression.sh
./eval/sweep_compression.sh aime25
N_SAMPLES=50 ./eval/sweep_compression.sh
LVEVAL_LEN=32k ./eval/sweep_compression.sh lveval
```

**`eval/sweep_pipelining.sh`** — block sizes `{16,32,64,128,256,1024}` by default.

```bash
./eval/sweep_pipelining.sh
./eval/sweep_pipelining.sh aime25
BLOCK_SIZES="128 256 1024" ./eval/sweep_pipelining.sh lveval
```

```bash
squeue -u $USER -j $(paste -sd, eval/results/compression_*/jobs.txt)
scancel $(cat eval/results/compression_<ts>/jobs.txt)
```

### Interpreting results

Both axes are **trade-offs**: fp8 helps TTFT when transfer-bound but can hurt
decode-heavy E2E; smaller blocks improve overlap until handshake overhead wins.
Numbers cited for **Qwen2.5-3B** on **synthetic** prompt lengths live in
[notes/tabs/2026-04-23_pd-dis-compression-pipelining.md](notes/tabs/2026-04-23_pd-dis-compression-pipelining.md)
and are kept for historical comparison with the current **Qwen3-4B** + dataset
runs above.

---

## Command-line options

Flags are parsed by **`pd_dis.py`** and forwarded from both launchers where
applicable.

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-4B` | HuggingFace model id |
| `--max-tokens` | `8192` | Max new tokens per request |
| `--max-model-len` | `40960` | Max total sequence length (**≤40960** for Qwen3-4B) |
| `--concurrency` | `4` | Parallel in-flight requests on the decode driver |
| `--dataset` | `none` | `none` / `lveval` / `aime25` |
| `--dataset-len` | `16k` | LVEval length bucket |
| `--dataset-subset` | `hotpotwikiqa_mixup` | LVEval subset base name |
| `--dataset-n-samples` | `100` | LVEval / cap for AIME |
| `--dataset-seed` | `42` | Sampling seed |
| `--pruning-method` | `none` | `none` / `attn_proxy` / `random` |
| `--pruning-keep-ratio` | `1.0` | Fraction of prompt tokens kept after pruning |
| `--no-warmup` | — | Skip the warmup request |

**Compression** and **pipelining** (axes **2** and **3**) use `--kv-cache-dtype`
and `--block-size`; see [Compression and pipelining sweeps](#compression-and-pipelining-sweeps).

---

## Outputs and artifacts

### Log format (`pd_dis.py`)

After a run finishes you get parseable stdout, including:

**Per-prompt line**

```
[1] OK | prompt='...' | completion=512 | ttft=1.234s | tpot=14.2ms | total=8.512s | prefill=0.210s
```

- **ttft** — time to first output token (prefill + KV transfer + first decode step, as seen from the decode client).
- **tpot** — ms per output token (decode throughput).
- **total** — wall time for the full stream.
- **prefill** — HTTP round-trip to the prefill node for `kv_transfer_params`.

**Aggregate block (example)**

```
TTFT           mean=1.234s  p50=1.1s  p95=2.3s  p99=3.1s
E2E latency    mean=8.5s    p50=8.1s  p95=12.0s p99=15.0s
Prefill time   mean=0.21s  p50=0.20s p95=0.31s p99=0.35s
TPOT (ms/tok)  mean=14.2ms
Throughput     312.4 tok/s  (total_ctok=15620, wall=50.0s)
```

**Accuracy (dataset mode)**

- **AIME25** — exact match on the extracted integer (e.g. `\boxed{}` parsing).
- **LVEval** — best token-level F1 vs references; **F1 ≥ 0.3** counts as correct
  (benchmark convention).

### CSV files

If **`--output-dir` is omitted**, CSVs are written under the repository’s
top-level **`results/`** next to `pd_dis.py`:

```
results/
  per_prompt_<model>_<dataset>_<timestamp>.csv
  summary_<model>_<dataset>_<timestamp>.csv
```

Sweeps and the report matrix always pass **`--output-dir`**, usually under
`eval/results/<tag>/…`, so cells stay grouped.

### Aggregating multi-cell sweeps

Different launchers write the same per-cell `summary_*.csv` layout, but use
different merge helpers:

1. **Pruning axis** (`launch_full_sweep.sh`): merge all cells with
   **`python3 eval/aggregate_sweep.py --results-dir eval/results/<TAG_PREFIX>`**.
   Produces `combined.csv` and `summary.md` beside `manifest.tsv`.

2. **Compression and pipelining axes** (`sweep_compression.sh`,
   `sweep_pipelining.sh`): use **`python3 eval/collect_summaries.py eval/results/<sweep_dir>`**.

To feed **both** compression and pipelining cell directories into one
`collect_summaries` run:

```bash
mkdir -p eval/results/combined_all
shopt -s nullglob
for d in eval/results/compression_*/*/ eval/results/pipelining_*/*/; do
  [[ -d "$d" ]] || continue
  cp -a "$d" eval/results/combined_all/
done
python3 eval/collect_summaries.py eval/results/combined_all
```

(`shopt -s nullglob` avoids errors when a glob matches nothing. Bash 3.x on
macOS may omit `shopt` and run the loop only when those paths exist.)

---

