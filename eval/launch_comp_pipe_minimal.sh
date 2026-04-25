#!/bin/bash
# =============================================================================
# Compression + Pipelining MINIMAL sweep — 4 cells, 1 repeat each.
#
# Goal: fill the empty {compression, pipelining} sockets in the teammate's
#       pruning sweep matrix so every method has a row in the final RESULTS.md.
#
# Uses pd_dis_chat.sh (chat-templated, Qwen3 thinking on) to match the
# teammate's runs exactly. Output dir layout matches launch_full_sweep.sh,
# so eval/aggregate_sweep.py drops the new rows in alongside pruning.
#
# Cells:
#   fp8_e5m2_aime25_r1   AIME25  --kv-cache-dtype fp8_e5m2  bs=1024 default
#   fp8_e5m2_lv16k_r1    LV16k   --kv-cache-dtype fp8_e5m2  bs=1024 default
#   bs128_aime25_r1      AIME25  --kv-cache-dtype auto      bs=128
#   bs128_lv16k_r1       LV16k   --kv-cache-dtype auto      bs=128
#
# Usage:
#   cd pd_dis
#   ./eval/launch_comp_pipe_minimal.sh
# =============================================================================
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$EVAL_DIR")"
cd "$ROOT_DIR"

SBATCH_SCRIPT=pd_dis_chat.sh
TS="$(date +%Y%m%d_%H%M%S)"
TAG_PREFIX="${TAG_PREFIX:-comp_pipe_${TS}}"
SWEEP_DIR="eval/results/${TAG_PREFIX}"
MANIFEST="$SWEEP_DIR/manifest.tsv"
mkdir -p "$SWEEP_DIR"
printf "job_id\ttag\tbenchmark\trepeat\toutput_dir\n" > "$MANIFEST"
echo "[sweep] results root: $SWEEP_DIR"

submit_one() {
    local tag="$1" bench="$2" rep="$3" out="$4"
    shift 4
    mkdir -p "$out"
    local jid
    jid=$(sbatch --parsable \
          --output="$out/log.txt" \
          --job-name="${tag}" \
          "$SBATCH_SCRIPT" "$@" --output-dir "$out")
    printf "%s\t%s\t%s\t%s\t%s\n" "$jid" "$tag" "$bench" "$rep" "$out" >> "$MANIFEST"
    echo "[sweep] $jid  $tag  →  $out"
}

# --- compression cells (kv-cache-dtype fp8_e5m2, default block-size) -------
submit_one "fp8_e5m2_aime25_r1" "aime25" "r1" \
    "$SWEEP_DIR/fp8_e5m2_aime25_r1" \
    --dataset aime25 --max-tokens 16384 --concurrency 8 \
    --pruning-method none --kv-cache-dtype fp8_e5m2 \
    --gpu-memory-utilization 0.85

submit_one "fp8_e5m2_lv16k_r1" "lv16k" "r1" \
    "$SWEEP_DIR/fp8_e5m2_lv16k_r1" \
    --dataset lveval --dataset-len 16k --dataset-n-samples 50 \
    --max-tokens 512 --concurrency 8 \
    --pruning-method none --kv-cache-dtype fp8_e5m2 \
    --gpu-memory-utilization 0.85

# --- pipelining cells (block-size 128, default kv-cache-dtype) -------------
submit_one "bs128_aime25_r1" "aime25" "r1" \
    "$SWEEP_DIR/bs128_aime25_r1" \
    --dataset aime25 --max-tokens 16384 --concurrency 8 \
    --pruning-method none --block-size 128 \
    --gpu-memory-utilization 0.85

submit_one "bs128_lv16k_r1" "lv16k" "r1" \
    "$SWEEP_DIR/bs128_lv16k_r1" \
    --dataset lveval --dataset-len 16k --dataset-n-samples 50 \
    --max-tokens 512 --concurrency 8 \
    --pruning-method none --block-size 128 \
    --gpu-memory-utilization 0.85

echo
echo "=== submitted $(($(wc -l < "$MANIFEST") - 1)) jobs ==="
echo "manifest: $MANIFEST"
echo
echo "Track with:"
echo "  squeue -u \$USER -j \$(awk 'NR>1 {print \$1}' $MANIFEST | paste -sd,)"
echo
echo "Aggregate after completion:"
echo "  python3 eval/aggregate_sweep.py --results-dir $SWEEP_DIR"
