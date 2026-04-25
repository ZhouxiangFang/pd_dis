"""Aggregate all summary_*.csv files in a sweep directory into one table.

The new pd_dis.py writes `summary_<model>_<dataset>_<ts>.csv` into each
job's --output-dir. This script walks a sweep results tree, reads every
such file, and emits:
  * combined.csv  — one row per job with all metrics and derived metadata
  * summary.md    — human-readable comparison table grouped by (tag, benchmark)

Usage:
    python3 eval/aggregate_sweep.py --results-dir eval/results/<sweep_tag>
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def _load_summary(csv_path: Path, output_dir: Path) -> dict | None:
    try:
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return None
    if not rows:
        return None
    r = rows[0]
    r["_tag"] = output_dir.name
    r["_output_dir"] = str(output_dir)
    return r


def _fmt(v, prec=3, unit=""):
    if v is None or v == "":
        return "n/a"
    try:
        return f"{float(v):.{prec}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Sweep root containing one subdir per job")
    args = ap.parse_args()

    root = args.results_dir
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    # Each subdir should contain one summary_*.csv written by pd_dis.py
    rows = []
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        summaries = sorted(subdir.glob("summary_*.csv"))
        if not summaries:
            print(f"[skip] no summary csv in {subdir.name}")
            continue
        # If multiple (rerun), keep the newest
        rec = _load_summary(summaries[-1], subdir)
        if rec is not None:
            rows.append(rec)

    if not rows:
        raise SystemExit(f"[parse] no summary csvs under {root}")

    # ---- Write combined CSV ----
    all_keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    combined_path = root / "combined.csv"
    with combined_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- Write markdown summary ----
    md = [
        "# Sweep Aggregate",
        "",
        f"Results root: `{root}`",
        f"Jobs aggregated: {len(rows)}",
        "",
        "| tag | dataset | subset | conc | n_ok | acc_pct | avg_score | "
        "TTFT mean | TTFT p95 | TTFT p99 | E2E mean | prefill mean | "
        "tpot ms/tok | tok/s | KV MB/req | err |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r.get("dataset", ""),
                                         r.get("_tag", ""))):
        md.append("| " + " | ".join([
            r.get("_tag", "?"),
            r.get("dataset", ""),
            r.get("dataset_subset", ""),
            _fmt(r.get("concurrency"), prec=0),
            _fmt(r.get("n_ok"), prec=0),
            _fmt(r.get("acc_pct"), prec=1, unit="%"),
            _fmt(r.get("avg_score"), prec=3),
            _fmt(r.get("ttft_mean_s"), prec=3, unit="s"),
            _fmt(r.get("ttft_p95_s"), prec=3, unit="s"),
            _fmt(r.get("ttft_p99_s"), prec=3, unit="s"),
            _fmt(r.get("e2e_mean_s"), prec=2, unit="s"),
            _fmt(r.get("prefill_mean_s"), prec=3, unit="s"),
            _fmt(r.get("tpot_mean_ms"), prec=1),
            _fmt(r.get("tok_per_s"), prec=1),
            _fmt(r.get("kv_payload_mean_mb_fp16"), prec=2),
            _fmt(r.get("n_err"), prec=0),
        ]) + " |")

    md_path = root / "summary.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"[aggregate] wrote {combined_path}")
    print(f"[aggregate] wrote {md_path}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
