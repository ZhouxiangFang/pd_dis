"""Parse pd_dis.py SLURM log files into per-run CSV and a summary markdown.

Expects a manifest.tsv produced by run_sweep.sh with columns:
  job_id  workload  repeat  ptok  otok  maxlen  logfile  tag

For each log, extracts per-prompt TTFT / tpot / E2E from the
"Per-prompt metrics" block, then computes:

  * mean / p50 / p95 / p99 TTFT
  * mean / p50 / p95 / p99 E2E latency
  * mean ms/output-token (tpot)
  * throughput tok/s over wall-clock
  * error count

Also reports per-prompt observed prompt_tokens (pulled from the detail
section) so you can plot TTFT-vs-prompt-tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Optional

COMPACT = re.compile(
    r"\[\s*(?P<idx>\d+)\s*\]\s+(?P<status>OK|ERROR)\s+\|\s+"
    r"prompt='(?P<prompt>[^']*)'\s+\|\s+"
    r"completion=(?P<comp>\d+)(?:\s+\(est\.\))?\s+\|\s+"
    r"ttft=(?P<ttft>[-\d.]+)s\s+\|\s+"
    r"tpot=(?P<tpot>[-\d.]+)ms\s+\|\s+"
    r"total=(?P<total>[-\d.]+)s"
    r"(?:\s+\|\s+prefill=(?P<prefill>[-\d.]+)s)?"
)

DETAIL_TOKENS = re.compile(
    r"\[\s*(?P<idx>\d+)\s*\]\s+Input\s*:\s*.*?Tokens:\s*prompt=(?P<ptok>\d+)\s+completion=(?P<ctok>\d+)",
    re.DOTALL,
)

SUMMARY = re.compile(
    r"SUMMARY\s+\((?P<n>\d+)\s+prompts,\s+wall-clock\s+(?P<wall>[\d.]+)s"
)


def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def parse_log(path):
    text = path.read_text(errors="replace")
    prompts = {}
    for m in COMPACT.finditer(text):
        d = m.groupdict()
        prompts[int(d["idx"])] = {
            "idx": int(d["idx"]),
            "status": d["status"],
            "completion_tokens": int(d["comp"]),
            "ttft": float(d["ttft"]),
            "tpot_ms": float(d["tpot"]),
            "e2e_s": float(d["total"]),
            "prefill_s": float(d["prefill"]) if d.get("prefill") else None,
            "prompt_tokens": None,
        }
    for m in DETAIL_TOKENS.finditer(text):
        d = m.groupdict()
        i = int(d["idx"])
        if i in prompts:
            prompts[i]["prompt_tokens"] = int(d["ptok"])

    sm = SUMMARY.search(text)
    wall = float(sm.group("wall")) if sm else None

    # Errors get no compact line (the ERROR case); scan for error markers.
    err_lines = [ln for ln in text.splitlines()
                 if "[Decode] ERROR on prompt" in ln or "| ERROR " in ln]

    return {
        "prompts": list(prompts.values()),
        "wall_clock": wall,
        "errors_in_log": len(err_lines),
    }


def agg(rows):
    ok = [r for r in rows if r["status"] == "OK"]
    ttfts = [r["ttft"] for r in ok]
    e2es = [r["e2e_s"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok]
    comps = [r["completion_tokens"] for r in ok]
    prefills = [r["prefill_s"] for r in ok if r.get("prefill_s") is not None]
    return {
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_err": len(rows) - len(ok),
        "ttft_mean": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "ttft_p99": percentile(ttfts, 99),
        "e2e_mean": statistics.mean(e2es) if e2es else None,
        "e2e_p50": percentile(e2es, 50),
        "e2e_p95": percentile(e2es, 95),
        "e2e_p99": percentile(e2es, 99),
        "tpot_mean_ms": statistics.mean(tpots) if tpots else None,
        "prefill_mean_s": statistics.mean(prefills) if prefills else None,
        "prefill_p95_s": percentile(prefills, 95),
        "total_completion_tokens": sum(comps),
    }


def fmt(x, unit="", prec=3) -> str:
    if x is None:
        return "n/a"
    if unit == "ms":
        return f"{x:.1f}ms"
    if unit == "s":
        return f"{x:.{prec}f}s"
    if unit == "tps":
        return f"{x:.1f}"
    return f"{x:.{prec}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = args.results_dir / "manifest.tsv"
    if not manifest.exists():
        raise SystemExit(f"[parse] manifest not found: {manifest}")

    rows = manifest.read_text().strip().splitlines()
    header = rows[0].split("\t")
    runs_out = []
    raw_rows_out = []  # one row per prompt, for per-prompt scatter plots

    for line in rows[1:]:
        cells = line.split("\t")
        rec = dict(zip(header, cells))
        lp = Path(rec["logfile"])
        if not lp.exists():
            print(f"[skip] missing log: {lp}")
            continue
        parsed = parse_log(lp)
        a = agg(parsed["prompts"])
        wall = parsed["wall_clock"]
        tps = (a["total_completion_tokens"] / wall) if (wall and wall > 0) else None

        runs_out.append({
            **rec,
            "wall_clock": wall,
            "tok_per_s": tps,
            **a,
        })

        for p in parsed["prompts"]:
            raw_rows_out.append({
                **{k: rec[k] for k in ("job_id", "workload", "repeat", "ptok",
                                       "otok", "tag")},
                **p,
            })

    if not runs_out:
        raise SystemExit("[parse] no runs parsed; did jobs complete?")

    # ---- Write per-run CSV ------------------------------------------------
    run_csv = args.results_dir / "per_run.csv"
    fields = list(runs_out[0].keys())
    with run_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(runs_out)

    # ---- Per-prompt CSV (for scatter / regression plots) -----------------
    prompt_csv = args.results_dir / "per_prompt.csv"
    if raw_rows_out:
        with prompt_csv.open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(raw_rows_out[0].keys()))
            w.writeheader()
            w.writerows(raw_rows_out)

    # ---- Workload×tag aggregation -----------------------------------------
    groups = {}
    for r in runs_out:
        key = (r["workload"], r.get("tag", ""))
        groups.setdefault(key, []).append(r)

    def avg(rs, k):
        vs = [r[k] for r in rs if r.get(k) is not None]
        return sum(vs) / len(vs) if vs else None

    md = ["# PD-disagg Eval Summary", ""]
    md.append(f"Results dir: `{args.results_dir}`")
    md.append(f"Runs parsed: {len(runs_out)}  |  Prompts parsed: {len(raw_rows_out)}")
    md.append("")
    md.append("| workload | tag | ptok | otok | n_ok | prefill mean | TTFT mean | TTFT p95 | TTFT p99 | E2E mean | E2E p95 | tpot | tok/s | err |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (wl, tag), rs in sorted(groups.items()):
        md.append(
            "| " + " | ".join([
                wl, tag or "-",
                str(rs[0]["ptok"]), str(rs[0]["otok"]),
                str(sum(r["n_ok"] for r in rs)),
                fmt(avg(rs, "prefill_mean_s"), "s"),
                fmt(avg(rs, "ttft_mean"), "s"),
                fmt(avg(rs, "ttft_p95"), "s"),
                fmt(avg(rs, "ttft_p99"), "s"),
                fmt(avg(rs, "e2e_mean"), "s"),
                fmt(avg(rs, "e2e_p95"), "s"),
                fmt(avg(rs, "tpot_mean_ms"), "ms"),
                fmt(avg(rs, "tok_per_s"), "tps"),
                str(sum(r["n_err"] for r in rs)),
            ]) + " |"
        )

    md_path = args.results_dir / "summary.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"[parse] wrote {run_csv}")
    print(f"[parse] wrote {prompt_csv}")
    print(f"[parse] wrote {md_path}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
