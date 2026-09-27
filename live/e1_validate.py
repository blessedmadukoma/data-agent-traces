#!/usr/bin/env python3
"""E1 block validation (plan 2026-09-24j, section 4): replay every G1 block offline.

For each blocked cell of a G1 run, a fresh sandbox runs the earlier cells of that run
that were executed (blocked cells are skipped, as in the live run), then the blocked
cell. Classes:
  confirmed      the cell raises KeyError naming the receipt's missing name, or
                 FileNotFoundError on the receipt's path;
  other_error    the cell raises another exception first (the gate's claim is not
                 contradicted, but the error seen first is different);
  no_error       the cell runs without error: a false block;
  not_replayed   an earlier cell was lost (time limit) in the replay.

Usage: /home/claude/iaenv/bin/python live/e1_validate.py --results runs/e1/results.jsonl \
    --tables /home/claude/qrdata_tables/data --out runs/e1/validation.jsonl
"""
import argparse
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "kramabench"))
import e1_run  # noqa: E402
import kb_gate  # noqa: E402


def validate(rec, lake, timeout):
    out = []
    cells = [c for c in rec["cells"] if not c.get("no_code")]
    for i, c in enumerate(cells):
        if not c.get("blocked"):
            continue
        rc = c["receipt"]
        ex = e1_run.LiveExecutor(lake, timeout=timeout)
        cls = None
        try:
            for p in cells[:i]:
                if p.get("blocked"):
                    continue
                r = ex.run(p["code"])
                if r.get("lost"):
                    cls = "not_replayed"
                    break
            if cls is None:
                r = ex.run(c["code"])
                keys, path = kb_gate.error_names(r.get("error"))
                if r.get("ok"):
                    cls = "no_error"
                elif rc.get("kind") == "file" and path is not None and kb_gate.same_path(rc.get("source") or "", path):
                    cls = "confirmed"
                elif rc.get("kind") in ("column", "key") and rc.get("missing") in keys:
                    cls = "confirmed"
                elif r.get("lost"):
                    cls = "not_replayed"
                else:
                    cls = "other_error"
                err = (r.get("error") or "").strip().splitlines()[-1:] if r.get("error") else []
            else:
                err = []
        finally:
            ex.close()
        out.append({"task": rec["task"], "arm": rec["arm"], "step": c["step"], "class": cls,
                    "receipt_reason": rc.get("reason"), "error": err[0][:200] if err else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    lake = os.path.abspath(a.tables)
    recs = [r for r in map(json.loads, open(a.results)) if r["arm"].endswith("G1")]
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor

    def job(r):
        rows = validate(r, lake, a.timeout)
        with lock, open(a.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    open(a.out, "w").close()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(job, [r for r in recs if any(c.get("blocked") for c in r["cells"])]))
    import collections
    rows = [json.loads(line) for line in open(a.out)]
    print(dict(collections.Counter((r["arm"], r["class"]) for r in rows)))


if __name__ == "__main__":
    main()
