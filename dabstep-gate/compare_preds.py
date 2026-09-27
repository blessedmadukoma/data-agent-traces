#!/usr/bin/env python3
"""Compare two gate runs cell by cell (one gate change at a time).

Usage:
  python3 compare_preds.py dabstep OLD.jsonl NEW.jsonl [--show 10]
  python3 compare_preds.py dab "OLD_DIR/gate_*.jsonl" "NEW_DIR/gate_*.jsonl" [--show 10]

Reports detections (cause-matched blocks) in each run, detections lost and
gained, blocks on cells without an error in each run, and every new block on
a cell without an error (these must be audited before a change is kept).
A change is harmless only if it loses no detection and adds no block on a
cell without an error that the audit confirms as false.
"""
import argparse
import collections
import glob
import json


def load(mode, spec):
    out = {}
    files = sorted(glob.glob(spec)) if mode == "dab" else [spec]
    for p in files:
        if p.endswith(".json"):
            continue
        for line in open(p):
            r = json.loads(line)
            if mode == "dabstep":
                k = (r["submission_file"], str(r["task_id"]), r.get("trace_line"), r["cell_index"])
                out[k] = {"failed": r["failed"], "outcome": r["outcome"], "cause": r["cause_match"],
                          "missing": r.get("missing"), "reason": r.get("reason")}
            else:
                k = (r["trace"], r["i"])
                out[k] = {"failed": r["failed"], "outcome": r.get("outcome"), "cause": bool(r.get("cause")),
                          "missing": r.get("missing"), "reason": r.get("reason")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dabstep", "dab"])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    A, B = load(a.mode, a.old), load(a.mode, a.new)
    keys = set(A) & set(B)
    det_a = {k for k in keys if A[k]["cause"]}
    det_b = {k for k in keys if B[k]["cause"]}
    okb_a = {k for k in keys if not A[k]["failed"] and A[k]["outcome"] == "block"}
    okb_b = {k for k in keys if not B[k]["failed"] and B[k]["outcome"] == "block"}
    trans = collections.Counter((A[k]["outcome"], B[k]["outcome"]) for k in keys if A[k]["outcome"] != B[k]["outcome"])
    print(f"cells compared {len(keys)} (only in old {len(set(A) - set(B))}, only in new {len(set(B) - set(A))})")
    print(f"detections: old {len(det_a)}, new {len(det_b)}; gained {len(det_b - det_a)}, lost {len(det_a - det_b)}")
    print(f"blocks on cells without an error: old {len(okb_a)}, new {len(okb_b)}; new ones {len(okb_b - okb_a)}")
    print("outcome changes: " + ", ".join(f"{x}->{y} {n}" for (x, y), n in trans.most_common()))
    for k in sorted(okb_b - okb_a)[: a.show]:
        print("  NEW BLOCK ON ERROR-FREE CELL:", k, B[k]["missing"], str(B[k]["reason"])[:120])
    for k in sorted(det_a - det_b)[: a.show]:
        print("  LOST DETECTION:", k, A[k]["missing"])


if __name__ == "__main__":
    main()
