#!/bin/bash
# =============================================================================
# Evaluation driver for PD disaggregation (NOTS, 2 × L40S).
# -----------------------------------------------------------------------------
# Generates synthetic prompt files, submits one sbatch job per
# (workload × repeat) cell, and saves a manifest.  Each job reuses pd_dis.sh
# as-is — NO changes to pd_dis.py or pd_dis.sh are required.
#
# Metrics this produces (after parse_results.py):
#   • TTFT mean / p50 / p95 / p99
#   • E2E latency mean / p50 / p95 / p99
#   • ms/output-token (tpot)
#   • token throughput (tok/s)   — serial upper bound; concurrency would need
#     a real async driver (documented in README)
#   • error rate / tail latency
#   • per-prompt TTFT vs prompt_tokens (from pd_dis.py's token report)
#   • KV payload bytes (run kv_bytes.py separately — analytical)
#
# Usage:
#   ./run_sweep.sh                 # full sweep, 3 repeats, submit & exit
#   ./run_sweep.sh 1               # smoke: 1 repeat of each workload
#   WAIT=1 ./run_sweep.sh          # poll squeue and auto-run parse_results
#   TAG=baseline ./run_sweep.sh    # annotate manifest so you can compare
#                                  # later re-runs (e.g. TAG=quant_int8)
# =============================================================================
set -euo pipefail

REPEATS="${1:-3}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_ARGS=("$@")
TAG="${TAG:-baseline}"
WAIT="${WAIT:-0}"

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$EVAL_DIR")"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$EVAL_DIR/results/${TAG}_${TS}"
mkdir -p "$RESULTS_DIR"

echo "[sweep] tag=$TAG  repeats=$REPEATS  results=$RESULTS_DIR"

# -----------------------------------------------------------------------------
# Workload matrix:  name   prompt_tok  out_tok   max_model_len
# Keep prompt+output+safety_margin <= max_model_len.
# long_ctx deliberately sized under 4096 so default vLLM config works.
# -----------------------------------------------------------------------------
WORKLOADS=(
  "short          256   128   4096"
  "prompt_heavy  1500   128   4096"
  "gen_heavy      256   512   4096"
  "long_ctx      2500   256   4096"
)
N_PROMPTS="${N_PROMPTS:-16}"

# -----------------------------------------------------------------------------
# Generate prompt files once per workload (shared across repeats).
# -----------------------------------------------------------------------------
for row in "${WORKLOADS[@]}"; do
  read -r NAME PTOK OTOK MAXLEN <<<"$row"
  python3 "$EVAL_DIR/gen_prompts.py" \
    --target-tokens "$PTOK" \
    --n-prompts "$N_PROMPTS" \
    --out "$RESULTS_DIR/prompts_${NAME}.txt" >/dev/null
done

# -----------------------------------------------------------------------------
# Submit jobs.  We cd to ROOT_DIR so SLURM_SUBMIT_DIR aligns with where
# pd_dis.py lives.
# -----------------------------------------------------------------------------
MANIFEST="$RESULTS_DIR/manifest.tsv"
printf "job_id\tworkload\trepeat\tptok\totok\tmaxlen\tlogfile\ttag\n" > "$MANIFEST"

cd "$ROOT_DIR"
ALL_JOBS=""
for row in "${WORKLOADS[@]}"; do
  read -r NAME PTOK OTOK MAXLEN <<<"$row"
  for r in $(seq 1 "$REPEATS"); do
    LOG="$RESULTS_DIR/log_${NAME}_r${r}.txt"
    JOB_ID=$(sbatch --parsable \
      --output="$LOG" \
      --job-name="pd_${TAG}_${NAME}_r${r}" \
      pd_dis.sh \
      --prompts-file "$RESULTS_DIR/prompts_${NAME}.txt" \
      --max-tokens  "$OTOK" \
      --max-model-len "$MAXLEN" \
      "${EXTRA_ARGS[@]}")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$JOB_ID" "$NAME" "$r" "$PTOK" "$OTOK" "$MAXLEN" "$LOG" "$TAG" >> "$MANIFEST"
    echo "[sweep] submitted $JOB_ID  $NAME  repeat=$r"
    ALL_JOBS+="${JOB_ID},"
  done
done
ALL_JOBS="${ALL_JOBS%,}"
cd - >/dev/null

echo
echo "[sweep] all jobs submitted → $MANIFEST"
echo "[sweep] track with: squeue -u \$USER -j $ALL_JOBS"
echo "[sweep] parse after completion:"
echo "        python3 $EVAL_DIR/parse_results.py --results-dir $RESULTS_DIR"

# -----------------------------------------------------------------------------
# Optional: block until all jobs leave the queue, then auto-parse.
# -----------------------------------------------------------------------------
if [[ "$WAIT" == "1" ]]; then
  echo "[sweep] WAIT=1 — polling squeue every 30s ..."
  while true; do
    REMAINING=$(squeue -h -u "$USER" -j "$ALL_JOBS" 2>/dev/null | wc -l || echo 0)
    if [[ "$REMAINING" -eq 0 ]]; then
      echo "[sweep] all jobs finished."
      break
    fi
    echo "[sweep]   $REMAINING job(s) still in queue/running"
    sleep 30
  done
  python3 "$EVAL_DIR/parse_results.py" --results-dir "$RESULTS_DIR"
fi
