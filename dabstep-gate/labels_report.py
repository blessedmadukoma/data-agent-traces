#!/usr/bin/env python3
"""Report on the labelled audit sample.

Prints: class shares overall and per submitter with Wilson 95% intervals
and a cluster bootstrap by submitter; Cohen's kappa against the second
labeller; decision rules K1 and K5; and, with --gate-dir, the gate's
cause-matched recall on labelled program-data failures.

Usage:
  python3 labels_report.py --labels $RUN/audit/labels_v2.csv \
      [--second $RUN/audit/labels_v2_second.csv] \
      [--gate-dir . --data $DATA --audit $RUN/audit/audit_failed_v2.jsonl]
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def boot(rows, pred, reps=2000, seed=1):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["submitter"]].append(r)
    keys = list(by)
    rnd = random.Random(seed)
    vals = []
    for _ in range(reps):
        s = [r for k in (rnd.choice(keys) for _ in keys) for r in by[k]]
        vals.append(sum(pred(r) for r in s) / len(s))
    vals.sort()
    return vals[int(.025 * reps)], vals[int(.975 * reps)]


def kappa(a, b):
    n = len(a)
    cats = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def yes(v):
    return str(v).strip().lower() in ("true", "yes", "y", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--second")
    ap.add_argument("--gate-dir")
    ap.add_argument("--data")
    ap.add_argument("--audit")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(
        open(a.labels)) if r["failure_class"].strip()]
    n = len(rows)
    print(
        f"labelled failed cells: {n} from {len({r['submitter'] for r in rows})} submitters")
    c = collections.Counter(r["failure_class"].strip().lower() for r in rows)
    for k, v in c.most_common():
        lo, hi = wilson(v, n)
        print(f"  {k:14s} {v:4d}  {v/n:6.1%}  Wilson {lo:.1%}-{hi:.1%}")

    def exact(r): return r["failure_class"].strip().lower() in (
        "program", "source") and yes(r["visible_before_execution"])
    k1 = sum(map(exact, rows))
    lo, hi = boot(rows, exact)
    print(
        f"\nexact, visible program/source preconditions: {k1}/{n} = {k1/n:.1%} (cluster bootstrap {lo:.1%}-{hi:.1%})")
    print("K1 (stop if < 10%):",
          "FAILS" if hi < .10 else "passes" if lo >= .10 else "undecided: extend the sample")
    # K5 is about apparent data failures: cells whose error looks like a data problem.
    DATA = {"KeyError", "KeyError(wrapped)", "index-error(wrapped)",
            "FileNotFoundError", "ValueError:usecols"}
    app = [r for r in rows if r.get("error_type") in DATA]

    def k5(r): return r["failure_class"].strip().lower() in (
        "ordinary", "ordinary code", "executor")
    if app:
        v5 = sum(map(k5, app))
        lo5, hi5 = boot(app, k5)
        print(f"apparent data failures (KeyError, FileNotFound, usecols): {len(app)}; labelled ordinary or executor: "
              f"{v5/len(app):.1%} (bootstrap {lo5:.1%}-{hi5:.1%}); K5 (stop if most): "
              + ("FAILS" if lo5 > .5 else "passes" if hi5 < .5 else "undecided"))
    print("\nper submitter (exact visible share):")
    for s, g in sorted(collections.defaultdict(list, {s: [r for r in rows if r['submitter'] == s] for s in {r['submitter'] for r in rows}}).items()):
        k = sum(map(exact, g))
        l, h = wilson(k, len(g))
        print(f"  {s:15s} {k:3d}/{len(g):3d} = {k/len(g):5.1%}  ({l:.0%}-{h:.0%})")
    sub = collections.Counter(r["program_subclass"].strip(
    ) for r in rows if r["failure_class"].strip().lower() == "program")
    if sub:
        print("program subclasses:", dict(sub))

    if a.second:
        sec = {r["audit_id"]: r for r in csv.DictReader(
            open(a.second)) if r["failure_class"].strip()}
        common = [r for r in rows if r["audit_id"] in sec]
        if common:
            for col in ("failure_class", "visible_before_execution"):
                x = [r[col].strip().lower() for r in common]
                y = [sec[r["audit_id"]][col].strip().lower() for r in common]
                print(
                    f"\nCohen's kappa for {col}: {kappa(x, y):.2f} on {len(common)} cells (revise codebook if < 0.6)")

    if a.gate_dir:
        import gate_adapter as ga
        decide = ga.load_decide(a.gate_dir)
        schema = ga.context_schema(a.data)
        cells = {str(json.loads(l)["audit_id"]): json.loads(l)
                 for l in open(a.audit)}
        prog = [r for r in rows if exact(r)]
        hit = cause = 0
        for r in prog:
            cell = cells[r["audit_id"]]
            o, why = ga.safe_decide(decide, cell["code"] or "", schema)
            if o == "block":
                hit += 1
                key = cell.get("error_key")
                cause += bool(key) and key in why
        if prog:
            l, h = wilson(cause, len(prog))
            print(f"\ngate on labelled exact program/source failures: blocks {hit}/{len(prog)}; "
                  f"cause-matched {cause}/{len(prog)} = {cause/len(prog):.0%} (Wilson {l:.0%}-{h:.0%})")


if __name__ == "__main__":
    main()
