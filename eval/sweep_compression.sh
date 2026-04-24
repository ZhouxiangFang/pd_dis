#!/bin/bash
# =============================================================================
# Compression sweep — fp8_e5m2 KV cache vs fp16 baseline, on AIME25 + LVEval.
# -----------------------------------------------------------------------------
# What this tests
#   Axis: --kv-cache-dtype ∈ {auto (fp16), fp8_e5m2}
#   Datasets: AIME25 (gen-heavy) and LVEval (prompt-heavy)
#   Model:   Qwen/Qwen3-4B (teammate's default — KV stored in fp8 halves the
#            on-wire NIXL transfer payload for free with no LMCache needed).
#
# Why these two datasets
#   The proposal's break-even hypothesis says compression helps when transfer
#   time dominates, and hurts when compute dominates (per-decode-token fp8
#   dequant adds ~1 ms/token to the kernel). AIME25 is decode-bound (16k out),
#   LVEval is prefill-bound (up to 32k in). So we expect:
#     • AIME25: fp8 probably a wash or slight loss (decode-dominated).
#     • LVEval: fp8 should save transfer time on TTFT but may cost on E2E.
#
# Output
#   One results/ directory per cell. Each pd_dis.py run writes:
#     summary_<model>_<dataset>_<timestamp>.csv
#     per_prompt_<model>_<dataset>_<timestamp>.csv
#   Aggregate across cells by grepping the summary CSVs.
#
# Usage
#   cd /home/yx106/code/pd_dis
#   ./eval/sweep_compression.sh                     # submit all 4 cells
#   ./eval/sweep_compression.sh aime25              # submit only AIME25 cells
#   ./eval/sweep_compression.sh lveval              # submit only LVEval cells
#   N_SAMPLES=30 ./eval/sweep_compression.sh        # override sample count
#   LVEVAL_LEN=32k ./eval/sweep_compression.sh      # LVEval context bucket
#
# Cancel all jobs this sweep submitted:
#   scancel $(cat <results-dir>/jobs.txt)
# =============================================================================
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$EVAL_DIR")"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$EVAL_DIR/results/compression_${TS}"
mkdir -p "$RESULTS_DIR"

# Tunable via env var ------------------------------------------------
TARGET="${1:-all}"                      # all | aime25 | lveval
N_SAMPLES="${N_SAMPLES:-30}"            # per-dataset sample count
LVEVAL_LEN="${LVEVAL_LEN:-16k}"         # LVEval context bucket
LVEVAL_SUBSET="${LVEVAL_SUBSET:-hotpotwikiqa_mixup}"
LVEVAL_MAX_TOKENS="${LVEVAL_MAX_TOKENS:-512}"
AIME_MAX_TOKENS="${AIME_MAX_TOKENS:-16384}"
CONCURRENCY="${CONCURRENCY:-4}"
SEED="${SEED:-42}"
# --------------------------------------------------------------------

echo "[sweep] results=$RESULTS_DIR  target=$TARGET  n_samples=$N_SAMPLES"
JOBS_FILE="$RESULTS_DIR/jobs.txt"
MANIFEST="$RESULTS_DIR/manifest.tsv"
printf "job_id\tdataset\ttag\tkv_cache_dtype\tlogfile\n" > "$MANIFEST"
cd "$ROOT_DIR"

submit_cell () {
  local DATASET=$1 TAG=$2 KVDT=$3 MAX_TOKENS=$4
  local EXTRA_DS_ARGS=""
  if [[ "$DATASET" == "lveval" ]]; then
    EXTRA_DS_ARGS="--dataset-subset $LVEVAL_SUBSET --dataset-len $LVEVAL_LEN"
  fi
  local OUTDIR="$RESULTS_DIR/${DATASET}_${TAG}"
  mkdir -p "$OUTDIR"
  local LOG="$OUTDIR/slurm.log"

  local JOB_ID
  JOB_ID=$(sbatch --parsable \
    --output="$LOG" \
    --job-name="sw_comp_${TAG}_${DATASET}" \
    pd_dis.sh \
    --dataset "$DATASET" \
    --dataset-n-samples "$N_SAMPLES" \
    --dataset-seed "$SEED" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" \
    --kv-cache-dtype "$KVDT" \
    --output-dir "$OUTDIR" \
    $EXTRA_DS_ARGS)
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$JOB_ID" "$DATASET" "$TAG" "$KVDT" "$LOG" >> "$MANIFEST"
  echo "$JOB_ID" >> "$JOBS_FILE"
  echo "[sweep] submitted $JOB_ID  dataset=$DATASET tag=$TAG kv_cache_dtype=$KVDT"
}

# --- cells --------------------------------------------------------------
if [[ "$TARGET" == "all" || "$TARGET" == "aime25" ]]; then
  submit_cell aime25 baseline    auto       "$AIME_MAX_TOKENS"
  submit_cell aime25 fp8_e5m2    fp8_e5m2   "$AIME_MAX_TOKENS"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "lveval" ]]; then
  submit_cell lveval baseline    auto       "$LVEVAL_MAX_TOKENS"
  submit_cell lveval fp8_e5m2    fp8_e5m2   "$LVEVAL_MAX_TOKENS"
fi

cd - >/dev/null
echo
echo "[sweep] all jobs submitted  manifest=$MANIFEST"
echo "[sweep] track with: squeue -u \$USER -j \$(paste -sd, $JOBS_FILE)"
echo "[sweep] cancel all: scancel \$(cat $JOBS_FILE)"
echo "[sweep] collect summaries after jobs finish:"
echo "        for f in $RESULTS_DIR/*/summary_*.csv; do echo \"\$f\"; cat \"\$f\"; done"
