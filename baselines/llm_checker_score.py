#!/usr/bin/env python3
"""Score baselines/llm_checker.py output: cause-matched detection and false blocks (Wilson CIs).
Usage: python3 baselines/llm_checker_score.py runs/llm_check/qr_gpt-oss.jsonl [...]"""
import json
import math
import statistics
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), (c + h) / d)


for f in sys.argv[1:]:
    rows = [json.loads(line) for line in open(f)]
    fails = [r for r in rows if not r["ok"] and r["error_type"] in ("KeyError", "FileNotFoundError")]
    ok = [r for r in rows if r["ok"]]
    c, fb = sum(r["cause_match"] for r in fails), sum(r["block"] for r in ok)
    cw, fw = wilson(c, len(fails)), wilson(fb, len(ok))
    print(f"{f}: cells {len(rows)}; caught {c}/{len(fails)} ({c / max(1, len(fails)):.1%}, CI {cw[0]:.1%}-{cw[1]:.1%}); "
          f"false blocks {fb}/{len(ok)} ({fb / max(1, len(ok)):.2%}, CI {fw[0]:.2%}-{fw[1]:.2%}); "
          f"cost ${sum(r['cost_usd'] for r in rows):.2f}; median {statistics.median(r['seconds'] for r in rows):.2f} s/cell")
