#!/bin/bash
# =============================================================================
# Full experiment sweep launcher. Submits 16 SLURM jobs for the baseline vs
# pruning report matrix. Safe to re-run — each run writes to a tag+timestamp
# subdirectory; nothing is overwritten.
#
# Uses pd_dis_chat.sh (chat-templated wrapper) so AIME25 scoring is meaningful.
# To use the raw-prompt path instead, replace SBATCH_SCRIPT with pd_dis.sh.
#
# Usage:
#   cd <repo-root>
#   ./eval/launch_full_sweep.sh
#   TAG_PREFIX=resweep_20260424_071611 ./eval/launch_full_sweep.sh
# =============================================================================
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$EVAL_DIR")"
cd "$ROOT_DIR"

SBATCH_SCRIPT=pd_dis_chat.sh
TS="$(date +%Y%m%d_%H%M%S)"
TAG_PREFIX="${TAG_PREFIX:-sweep_${TS}}"
RESULTS_ROOT="eval/results"

# Extra flags appended to EVERY sbatch call (e.g. "--exclusive" if your site
# requires whole-node jobs to avoid NIXL side-channel port collisions).
SWEEP_EXTRA_FLAGS="${SWEEP_EXTRA_FLAGS:-}"
# shellcheck disable=SC2206
EXTRA_ARRAY=( $SWEEP_EXTRA_FLAGS )

SWEEP_DIR="$RESULTS_ROOT/${TAG_PREFIX}"
MANIFEST="$SWEEP_DIR/manifest.tsv"
mkdir -p "$SWEEP_DIR"
printf "job_id\ttag\tbenchmark\trepeat\toutput_dir\n" > "$MANIFEST"
echo "[sweep] results root: $SWEEP_DIR"

submit_one() {
    local tag="$1" out="$2"
    shift 2
    mkdir -p "$out"
    local jid
    jid=$(sbatch --parsable \
          --output="$out/log.txt" \
          --job-name="${tag}" \
          "$SBATCH_SCRIPT" "$@" "${EXTRA_ARRAY[@]}" --output-dir "$out")
    printf "%s\t%s\t%s\t%s\t%s\n" "$jid" "$tag" "${BENCH:-?}" "${REP:-?}" "$out" >> "$MANIFEST"
    echo "[sweep] $jid  $tag  →  $out"
}

# ---------------------------------------------------------------------------
# 1. Main matrix: 3 methods × 2 benchmarks × 2 repeats  =  12 jobs
# ---------------------------------------------------------------------------
for r in r1 r2; do
    REP=$r

    # ---- AIME25 (generation-heavy, 16k output) ----
    BENCH=aime25
    submit_one "baseline_${BENCH}_${r}" \
        "$RESULTS_ROOT/${TAG_PREFIX}/baseline_${BENCH}_${r}" \
        --dataset aime25 --max-tokens 16384 --concurrency 8 \
        --pruning-method none

    submit_one "prune_k05_${BENCH}_${r}" \
        "$RESULTS_ROOT/${TAG_PREFIX}/prune_k05_${BENCH}_${r}" \
        --dataset aime25 --max-tokens 16384 --concurrency 8 \
        --pruning-method attn_proxy --pruning-keep-ratio 0.5

    submit_one "prune_k03_${BENCH}_${r}" \
        "$RESULTS_ROOT/${TAG_PREFIX}/prune_k03_${BENCH}_${r}" \
        --dataset aime25 --max-tokens 16384 --concurrency 8 \
        --pruning-method attn_proxy --pruning-keep-ratio 0.3

    # ---- LVEval 16k (prompt-heavy, 512 output) ----
    BENCH=lv16k
    submit_one "baseline_${BENCH}_${r}" \
        "$RESULTS_ROOT/${TAG_PREFIX}/baseline_${BENCH}_${r}" \
        --dataset lveval --dataset-len 16k --dataset-n-samples 50 \
        --max-tokens 512 --concurrency 8 --pruning-method none

    submit_one "prune_k05_${BENCH}_${r}" \
        "$RESULTS_ROOT/${TAG_PREFIX}/prune_k05_${BENCH}_${r}" \
        --dataset lveval --dataset-len 16k --dataset-n-samples 50 \
        --max-tokens 512 --concurrency 8 \
        --pruning-method attn_proxy --pruning-keep-ratio 0.5

    submit_one "prune_k03_${BENCH}_${r}" \
        "$RESULTS_ROOT/${TAG_PREFIX}/prune_k03_${BENCH}_${r}" \
        --dataset lveval --dataset-len 16k --dataset-n-samples 50 \
        --max-tokens 512 --concurrency 8 \
        --pruning-method attn_proxy --pruning-keep-ratio 0.3
done

# ---------------------------------------------------------------------------
# 2. Length-axis sweep: LVEval 32k baseline only  =  2 jobs
# ---------------------------------------------------------------------------
for r in r1 r2; do
    REP=$r
    BENCH=lv32k
    submit_one "baseline_${BENCH}_${r}" \
        "$RESULTS_ROOT/${TAG_PREFIX}/baseline_${BENCH}_${r}" \
        --dataset lveval --dataset-len 32k --dataset-n-samples 50 \
        --max-tokens 512 --max-model-len 40960 \
        --concurrency 8 --pruning-method none
done

# ---------------------------------------------------------------------------
# 3. Concurrency=1 control: baseline only  =  2 jobs
# ---------------------------------------------------------------------------
REP=r1
BENCH=aime25
submit_one "baseline_c1_${BENCH}" \
    "$RESULTS_ROOT/${TAG_PREFIX}/baseline_c1_${BENCH}" \
    --dataset aime25 --max-tokens 16384 --concurrency 1 \
    --pruning-method none

BENCH=lv16k
submit_one "baseline_c1_${BENCH}" \
    "$RESULTS_ROOT/${TAG_PREFIX}/baseline_c1_${BENCH}" \
    --dataset lveval --dataset-len 16k --dataset-n-samples 50 \
    --max-tokens 512 --concurrency 1 --pruning-method none

# ---------------------------------------------------------------------------
echo
echo "=== submitted $(wc -l < "$MANIFEST" | awk '{print $1-1}') jobs ==="
echo "manifest: $MANIFEST"
echo
echo "Track with:"
echo "  squeue -u \$USER"
echo
echo "When all jobs finish, aggregate summary_*.csv under each subdir:"
echo "  python3 eval/aggregate_sweep.py --results-dir $RESULTS_ROOT/${TAG_PREFIX}"
