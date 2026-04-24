#!/bin/bash
# =============================================================================
# Pipelining sweep — --block-size sweep on AIME25 + LVEval.
# -----------------------------------------------------------------------------
# What this tests
#   Axis: --block-size ∈ {default values below}
#   vLLM allocates KV in blocks of this many tokens. NixlConnector transfers
#   one block at a time — smaller blocks = more transfer/compute overlap
#   (pipelining). But too small: per-transfer overhead dominates.
#
# Why two datasets
#   AIME25 (decode-heavy): expect minimal pipelining effect (KV small).
#   LVEval (prompt-heavy): expect U-shaped TTFT curve with block size,
#     with a sweet spot somewhere between 32 and 256.
#
# Baseline
#   The default --block-size in pd_dis.py is 1024. That's a single point on
#   the curve; we include it as "baseline" for A/B vs. the sweep cells.
#
# Output
#   See sweep_compression.sh — same layout.
#
# Usage
#   cd /home/yx106/code/pd_dis
#   ./eval/sweep_pipelining.sh                       # all cells, both datasets
#   ./eval/sweep_pipelining.sh aime25                # AIME25 only
#   ./eval/sweep_pipelining.sh lveval                # LVEval only
#   BLOCK_SIZES="128 256 1024" ./eval/sweep_pipelining.sh  # custom sweep
#
# Cancel all jobs this sweep submitted:
#   scancel $(cat <results-dir>/jobs.txt)
# =============================================================================
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$EVAL_DIR")"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$EVAL_DIR/results/pipelining_${TS}"
mkdir -p "$RESULTS_DIR"

# Tunable via env var ------------------------------------------------
TARGET="${1:-all}"                      # all | aime25 | lveval
BLOCK_SIZES="${BLOCK_SIZES:-16 32 64 128 256 1024}"
N_SAMPLES="${N_SAMPLES:-30}"
LVEVAL_LEN="${LVEVAL_LEN:-16k}"
LVEVAL_SUBSET="${LVEVAL_SUBSET:-hotpotwikiqa_mixup}"
LVEVAL_MAX_TOKENS="${LVEVAL_MAX_TOKENS:-512}"
AIME_MAX_TOKENS="${AIME_MAX_TOKENS:-16384}"
CONCURRENCY="${CONCURRENCY:-4}"
SEED="${SEED:-42}"
# --------------------------------------------------------------------

echo "[sweep] results=$RESULTS_DIR  target=$TARGET  block_sizes=$BLOCK_SIZES"
JOBS_FILE="$RESULTS_DIR/jobs.txt"
MANIFEST="$RESULTS_DIR/manifest.tsv"
printf "job_id\tdataset\ttag\tblock_size\tlogfile\n" > "$MANIFEST"
cd "$ROOT_DIR"

submit_cell () {
  local DATASET=$1 BS=$2 MAX_TOKENS=$3
  local TAG="bs${BS}"
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
    --job-name="sw_pipe_${TAG}_${DATASET}" \
    pd_dis.sh \
    --dataset "$DATASET" \
    --dataset-n-samples "$N_SAMPLES" \
    --dataset-seed "$SEED" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" \
    --block-size "$BS" \
    --output-dir "$OUTDIR" \
    $EXTRA_DS_ARGS)
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$JOB_ID" "$DATASET" "$TAG" "$BS" "$LOG" >> "$MANIFEST"
  echo "$JOB_ID" >> "$JOBS_FILE"
  echo "[sweep] submitted $JOB_ID  dataset=$DATASET block_size=$BS"
}

# --- cells --------------------------------------------------------------
for BS in $BLOCK_SIZES; do
  if [[ "$TARGET" == "all" || "$TARGET" == "aime25" ]]; then
    submit_cell aime25 "$BS" "$AIME_MAX_TOKENS"
  fi
  if [[ "$TARGET" == "all" || "$TARGET" == "lveval" ]]; then
    submit_cell lveval "$BS" "$LVEVAL_MAX_TOKENS"
  fi
done

cd - >/dev/null
echo
echo "[sweep] all jobs submitted  manifest=$MANIFEST"
echo "[sweep] track with: squeue -u \$USER -j \$(paste -sd, $JOBS_FILE)"
echo "[sweep] cancel all: scancel \$(cat $JOBS_FILE)"
echo "[sweep] collect summaries after jobs finish:"
echo "        for f in $RESULTS_DIR/*/summary_*.csv; do echo \"\$f\"; cat \"\$f\"; done"
