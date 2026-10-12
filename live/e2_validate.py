#!/usr/bin/env python3
"""E2 block check (pre-registration section 4): execute every step that arm R blocked, without the checker.

For each blocked step of an R run, a fresh sandbox (the E2 Docker sandbox) runs the earlier steps of that
run that were executed (blocked steps are skipped, as in the live run), then the blocked step. Classes, as in
the first live experiment (live/e1_validate.py):
  confirmed      the step raises KeyError naming the receipt's missing name, or a file error on its path;
  other_error    the step raises another error first;
  no_error       the step runs without an error: a false block;
  not_replayed   an earlier step was lost (time limit) in the replay.

Usage: python3 live/e2_validate.py --results runs/e2/results.jsonl --tables data/QRData/benchmark/data \\
           --out runs/e2/validation.jsonl
"""
import argparse
import collections
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2_run  # noqa: E402
kb_gate = e2_run.kb_gate


def validate(rec, lake, timeout):
    out = []
    cells = [c for c in rec["cells"] if not c.get("no_code")]
    for i, c in enumerate(cells):
        if not c.get("blocked"):
            continue
        rc = c["receipt"] or {}
        ex = e2_run.DockerExecutor(lake, timeout=timeout)
        cls, err = None, []
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
        finally:
            ex.close()
        out.append({"task": rec["task"], "seed": rec["seed"], "step": c["step"], "class": cls,
                    "receipt_kind": rc.get("kind"), "receipt_reason": rc.get("reason"),
                    "error": err[0][:200] if err else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    lake = os.path.abspath(a.tables)
    recs = [r for r in map(json.loads, open(a.results))
            if r["arm"] == "R" and any(c.get("blocked") for c in r["cells"])]
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor

    def job(r):
        rows = validate(r, lake, a.timeout)
        with lock, open(a.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    open(a.out, "w").close()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(job, recs))
    rows = [json.loads(line) for line in open(a.out)]
    print(dict(collections.Counter(r["class"] for r in rows)))


if __name__ == "__main__":
    main()
