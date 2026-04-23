# PD-Disagg Evaluation Suite

Benchmark driver for the two-node vLLM NixlConnector prefill-decode pipeline.
Submits a matrix of SLURM jobs, captures per-prompt metrics from the SLURM
log files, and produces CSV + a markdown summary table.

**Baseline has already been run.** Results are in
`results/baseline_20260421_220526/` (see `REPORT.md` in there for full metric
definitions + observations). To compare your optimisation against the
baseline, see [How to add your method](#how-to-add-your-method).

## What this measures

| Metric | How |
|---|---|
| TTFT mean / p50 / p95 / p99 | `ttft=` field in the compact `[i] OK` line (decode-side stream time → first token) |
| E2E latency mean / p50 / p95 / p99 | `total=` field (decode-stream open → last token) |
| ms/output-token (tpot) | `tpot=` field |
| Prefill HTTP time | `prefill=` field (decode node → prefill node POST round-trip for `kv_transfer_params`) |
| Token throughput (tok/s) | Σ completion_tokens / wall_clock, **serial driver only** |
| Error rate | count of `[i] ERROR` lines vs total |
| Per-prompt TTFT vs real prompt_tokens | from the vLLM-logged detail block |
| KV payload bytes (analytical) | `python3 kv_bytes.py` (needs no GPU) |

Full English metric definitions + interpretation of the baseline numbers:
`results/baseline_20260421_220526/REPORT.md`.

## Files

- `gen_prompts.py` — synthesises prompts at a target token length
- `run_sweep.sh` — master driver: generates prompts, submits sbatch jobs, writes `manifest.tsv`
- `parse_results.py` — parses the SLURM logs → `per_run.csv`, `per_prompt.csv`, `summary.md`
- `kv_bytes.py` — analytical KV-cache payload calculator (reads HF config)

## Quick start

```bash
cd pd_dis/eval

# Analytical KV sizes per sequence length (Qwen2.5-3B default)
python3 kv_bytes.py

# Smoke test: 1 repeat per workload = 4 jobs, ~15–25 min wall
./run_sweep.sh 1

# Full sweep: 3 repeats = 12 jobs
TAG=baseline ./run_sweep.sh 3

# Blocks until jobs finish and auto-runs parse_results
WAIT=1 TAG=baseline ./run_sweep.sh 3

# Manual parse if you didn't use WAIT=1
python3 parse_results.py --results-dir results/<TAG>_<timestamp>
```

## Workload matrix

| name | prompt tokens | output tokens | max_model_len | purpose |
|---|---:|---:|---:|---|
| `short` | 256 | 128 | 4096 | sanity check |
| `prompt_heavy` | 1500 | 128 | 4096 | stresses KV transfer size |
| `gen_heavy` | 256 | 512 | 4096 | decode-dominated, KV transfer small |
| `long_ctx` | 2500 | 256 | 4096 | realistic long-context chat |

Each config runs `N_PROMPTS=16` prompts per job. Override via env var.

## How to add your method

The pipeline supports A/B comparison via the `TAG` env var. `parse_results.py`
groups results by `(workload, tag)` so your method and the baseline land in
the same summary table.

### Step 1 — implement your optimisation

Add whatever flag your method needs to `pd_dis.py` (e.g. `--kv-compress int8`
or `--pruning h2o`) and pass it through to the vLLM launch or the KV
transfer path. Keep the logging format in `pd_dis.py:run_decode()` unchanged
— the parser relies on the compact `[i] OK | ... | ttft=...s | tpot=...ms | total=...s | prefill=...s` line.

### Step 2 — run the sweep with your tag

```bash
TAG=quant_int8 ./run_sweep.sh 3 -- --kv-compress int8
TAG=h2o_prune  ./run_sweep.sh 3 -- --pruning h2o --keep-ratio 0.5
TAG=layer_pipe ./run_sweep.sh 3 -- --layer-pipelining 1
```

Arguments after the `--` are forwarded to `pd_dis.sh` and then to `pd_dis.py`.

### Step 3 — parse and compare

```bash
# Parse each run individually
python3 parse_results.py --results-dir results/quant_int8_<timestamp>
python3 parse_results.py --results-dir results/h2o_prune_<timestamp>

# To combine with baseline into one table, concatenate manifests:
mkdir -p results/all_methods
cat results/baseline_*/manifest.tsv \
    results/quant_int8_*/manifest.tsv \
    results/h2o_prune_*/manifest.tsv \
    results/layer_pipe_*/manifest.tsv | \
    awk 'NR==1 || !/^job_id/' > results/all_methods/manifest.tsv

python3 parse_results.py --results-dir results/all_methods
```

The resulting `summary.md` will have one row per `(workload, tag)`.

## Quality measurement (perplexity) — do this AFTER implementing compression or pruning

Compression and pruning need a quality control. The latency sweep above cannot
answer "did the method hurt output quality?" — run this **offline, on a
single node**, independent of the PD-disagg pipeline:

```python
# Pseudo-code — ~30 lines, ~10 min on one L40S per method
from datasets import load_dataset
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model with your KV modification enabled (reuse your vLLM plugin,
#    or use transformers with a monkey-patched attention).
# 2. Run 512 sequences of WikiText-2 through the model.
# 3. Compute mean NLL; exp() it → perplexity.
# 4. Compare to the unmodified baseline (same 512 sequences).
```

Report PPL_baseline vs PPL_{method} side-by-side. Anything under +0.5 PPL is
"no meaningful degradation" for a 3B model.

## Time estimate on NOTS `commons`

| phase | time |
|---|---|
| SLURM queue wait (commons, typical) | 0 – 10 min |
| Model load from shared HF cache | 60 – 120 s |
| NIXL handshake + health-check | 10 – 20 s |
| Warmup | 3 – 8 s |
| 16 prompts × 5–10 s each | 80 – 160 s |
| vLLM shutdown + cleanup | 10 – 20 s |
| **per-job total compute** | **3 – 6 min** |

Full sweep (12 jobs) wall clock: **parallel 10–15 min / serial 60–90 min**,
depending on how the QoS schedules them. Overnight the baseline sweep here
took ~2.5 h because jobs ran nearly serially.

## Environment

Scripts auto-fall-back between:

```
/scratch/$USER/comp529/miniconda3/envs/nlp    (if it exists)
/scratch/zf28/comp529/miniconda3/envs/nlp     (shared fallback — world-readable)
```

with the same logic for `HF_HOME`. No setup needed if you are on NOTS and
are part of the `comp529` team.

## Limitations (still open)

1. **Serial load only.** `tok/s` is an upper bound, not real goodput under
   concurrent request rate. Would need a ~40-line asyncio driver (e.g. a
   stripped-down `benchmarks/benchmark_serving.py` from vLLM upstream).
2. **ITL distribution** is not captured per-token. Only mean `tpot`. To get
   a CDF, collect `perf_counter()` after every delta in
   `pd_dis.py:http_stream_completion()` and emit a list per prompt.
3. **Shared cluster noise.** p99 on NOTS reflects other jobs' IB traffic.
   Run 3+ repeats.
4. **Synthetic prompts.** `gen_prompts.py` repeats a paragraph — do not use
   these outputs for anything about model output quality.
5. **No network-condition sweep.** NOTS compute nodes don't allow `tc
   netem`. For a low-bandwidth point, add
   `export UCX_TLS=tcp` to `pd_dis.sh` just before the existing `UCX_TLS`
   export, and run with a new `TAG`. Compare against the default IB run.
