"""
Aggregate all summary_*.csv files under a sweep results directory into a
single CSV + a pretty-printed markdown table, so compression and pipelining
sweeps can be compared against the baseline side by side.

Usage
    python3 eval/collect_summaries.py eval/results/compression_20260423_153805
    python3 eval/collect_summaries.py eval/results/pipelining_20260423_153805 \
        --columns dataset tag kv_cache_dtype ttft_mean e2e_mean tpot_mean_ms tok_per_s accuracy
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

DEFAULT_COLUMNS = [
    "dataset", "tag", "ttft_mean", "ttft_p95", "ttft_p99",
    "e2e_mean", "e2e_p95", "prefill_mean", "tpot_mean_ms",
    "tok_per_s", "accuracy", "n_ok", "n_err",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Path for combined CSV (default: <results_dir>/combined.csv)")
    ap.add_argument("--columns", nargs="*", default=None,
                    help="Columns to include in the markdown table")
    args = ap.parse_args()

    if not args.results_dir.exists():
        print(f"[ERROR] {args.results_dir} not found", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for cell_dir in sorted(args.results_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        summaries = sorted(cell_dir.glob("summary_*.csv"))
        if not summaries:
            continue
        summary_path = summaries[-1]  # newest if multiple
        with summary_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse "<dataset>_<tag>" from the subdir name, e.g.
                # "aime25_fp8_e5m2" or "lveval_bs128".
                parts = cell_dir.name.split("_", 1)
                row["dataset"] = parts[0] if parts else cell_dir.name
                row["tag"]     = parts[1] if len(parts) == 2 else ""
                row["_source"] = str(summary_path)
                rows.append(row)

    if not rows:
        print(f"[ERROR] No summary_*.csv found under {args.results_dir}",
              file=sys.stderr)
        sys.exit(1)

    # All keys across all rows, dataset/tag first then the rest alphabetical.
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    front = ["dataset", "tag"]
    ordered_keys = front + sorted(k for k in all_keys if k not in front)

    out_csv = args.out or (args.results_dir / "combined.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ordered_keys})
    print(f"[collect] wrote {out_csv}  ({len(rows)} rows)", file=sys.stderr)

    # Markdown table
    cols = args.columns or [c for c in DEFAULT_COLUMNS if c in ordered_keys]
    header = "| " + " | ".join(cols) + " |"
    align  = "|" + "|".join([":---" if c in ("dataset", "tag") else "---:"
                              for c in cols]) + "|"
    print(header)
    print(align)
    for r in rows:
        cells = [str(r.get(c, "")) for c in cols]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
