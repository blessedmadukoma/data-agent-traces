#!/usr/bin/env python3
"""Run-to-run variation in the live experiment (preprint, Section 3.3).

Prints accuracy by arm and seed, and, for each seed and prompt, the accuracy of the arms
with and without the checker on the questions where the checker arm never blocked. On those
questions the checker did nothing, so their difference shows how far accuracy moves between
conditions that should not differ.

Usage: uv run live/e1_noise.py --results recorded/e1/results.jsonl --questions data/QRData/benchmark/QRData.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--questions", required=True)
    a = ap.parse_args()
    qs = json.load(open(a.questions))
    gold = {f"qr-{i}": (q["answer"], (q.get("meta_data") or {}).get("question_type")) for i, q in enumerate(qs)}
    runs = collections.defaultdict(dict)
    for line in open(a.results):
        r = json.loads(line)
        runs[(r["task"], r["seed"])][r["arm"]] = r

    def ok(r):
        return e1_report.correct(r.get("final_answer"), *gold[r["task"]])

    seeds = sorted({s for _, s in runs})
    for arm in ("P0G0", "P0G1", "P1G0", "P1G1"):
        parts = []
        for s in seeds:
            xs = [v[arm] for (t, sd), v in runs.items() if sd == s and arm in v]
            if xs:
                parts.append(f"seed {s}: {sum(map(ok, xs)) / len(xs):.3f} (n {len(xs)})")
        if parts:
            print(f"{arm}: " + "; ".join(parts))
    for s in seeds:
        for p in ("P0", "P1"):
            pairs = [(v[p + "G0"], v[p + "G1"]) for (t, sd), v in runs.items() if sd == s and p + "G0" in v
                     and p + "G1" in v and not any(c.get("blocked") for c in v[p + "G1"]["cells"])]
            if pairs:
                a0 = sum(ok(x) for x, _ in pairs) / len(pairs)
                a1 = sum(ok(y) for _, y in pairs) / len(pairs)
                print(f"seed {s}, {p}, questions where the checker never blocked: {len(pairs)}; accuracy without the "
                      f"checker {a0:.3f}, with it {a1:.3f}, difference {100 * (a1 - a0):+.1f} points")


if __name__ == "__main__":
    main()
