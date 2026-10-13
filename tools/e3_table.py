#!/usr/bin/env python3
"""Print the LaTeX rows of the paper's table of the live experiments E2 and E3, and the numbers that the text cites.

Inputs are the JSON outputs of live/e2_report.py (one per experiment and model), of live/e3_report.py (pooled) and
of tools/e3_numbers.py is recomputed here from the results files.

Usage: python3 tools/e3_table.py --questions data/QRData/benchmark/QRData.json \\
           --row "E2, gpt-oss:120b" runs/e2/results.jsonl /tmp/e2_report.json \\
           --row "E3, gpt-oss:120b" runs/e3/gpt-oss-120b.jsonl /tmp/e3_gpt-oss-120b.json \\
           --row "E3, gemma4:31b" runs/e3/gemma4-31b.jsonl /tmp/e3_gemma4-31b.json --pooled /tmp/e3_pooled.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live"))
import e2_report as r2  # noqa: E402

ARM_NAME = {"N": "No checker", "R": "Checker", "E": "Names in error"}


def pts(x):
    return f"{100 * x:+.2f}".replace("+", "$+$").replace("-", "$-$") if abs(x) < 0.1 else f"{100 * x:+.1f}"


def ci(c):
    return f"{100 * c[0]:.2f} to {100 * c[1]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--row", nargs=3, action="append", metavar=("LABEL", "RESULTS", "REPORT"), required=True)
    ap.add_argument("--pooled")
    a = ap.parse_args()
    qsrc = json.load(open(a.questions))
    gold = {f"qr-{i}": (q["answer"], (q.get("meta_data") or {}).get("question_type")) for i, q in enumerate(qsrc)}
    print("% panel A: one row per experiment, model and arm")
    for label, results, report in a.row:
        rep = json.load(open(report))
        rows = [json.loads(line) for line in open(results)]
        for arm in ("N", "R", "E"):
            xs = [r for r in rows if r["arm"] == arm]
            ms = [r2.run_measures(r, gold) for r in xs]
            dre = sum(m["kinds"]["file"] + m["kinds"]["key_checker_block"] for m in ms)
            b = rep["by_arm"][arm]
            blocks = str(b["blocks"]) if arm == "R" else "--"
            first = label if arm == "N" else ""
            print(f"{first} & {ARM_NAME[arm]} & {100 * b['accuracy']:.1f}\\% & {b['steps_mean']:.2f} & "
                  f"{b['out_tokens_mean']:,.0f} & {b['runs_with_dra']:,} & {dre:,} & {blocks} \\\\")
        print("\\midrule")
    print("% panel B: differences in accuracy, points (95% CI); p from the permutation test")
    for label, results, report in a.row:
        rep = json.load(open(report))
        h1, h2, h3 = rep["H1 checker"], rep["H2 names in the error"], rep["H3 checker against names in the error"]
        h5 = rep["H5 checker"]["correct"]
        print(f"{label} & {100 * h1['diff']:.2f} ({ci(h1['ci'])}; p={h1['perm_p']:.3f}) & "
              f"{100 * h2['diff']:.2f} ({ci(h2['ci'])}; p={h2['perm_p']:.3f}) & "
              f"{100 * h3['diff']:.2f} ({ci(h3['ci'])}) & {100 * h5['diff']:.1f} ({ci(h5['ci'])}) \\\\")
    if a.pooled:
        p = json.load(open(a.pooled))["pooled"]
        h1, h2, h3 = p["H1 checker (primary)"], p["H2 names in the error"], p["H3 checker against names"]
        h5 = p["H5 checker"]["correct"]
        print(f"E3, pooled & {100 * h1['diff']:.2f} ({ci(h1['ci'])}; p={h1['perm_p']:.4f}; {h1['verdict']}) & "
              f"{100 * h2['diff']:.2f} ({ci(h2['ci'])}; p={h2['perm_p']:.3f}) & "
              f"{100 * h3['diff']:.2f} ({ci(h3['ci'])}) & {100 * h5['diff']:.1f} ({ci(h5['ci'])}) \\\\")


if __name__ == "__main__":
    main()
