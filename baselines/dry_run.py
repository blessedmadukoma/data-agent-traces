#!/usr/bin/env python3
"""Baseline: replay recorded traces on truncated copies of the data.

A "dry run" is the obvious alternative to a static gate: run each cell on a
small copy of the data first and block it if it raises KeyError or
FileNotFoundError. This script replays every recorded trace of a run, cell by
cell, in a fresh sandbox per trace, with the data directory replaced by a
truncated copy:

  h0    every CSV cut to its first line (the raw header bytes);
  s100  every CSV cut to its first 101 raw lines.

Other files are copied unchanged. Each cell runs with a time limit; after a
lost executor, the rest of the trace is not run.

Output: one line per recorded cell with the recorded outcome and the dry-run
outcome. Score with baselines/dry_run_score.py.

Usage (sandbox Python of the recorded run):
  python baselines/dry_run.py --results runs/qr/results_gpt-oss.jsonl \
      --lake $QRDATA_TABLES --variant h0 --out runs/dry/qr_gpt-oss_h0.jsonl --workers 2
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kramabench"))
import kb_run  # noqa: E402

LINES = {"h0": 1, "s100": 101}


def truncated_lake(lake, variant, root):
    """Copy the lake with every CSV cut to its first n raw lines. Returns the copy's path."""
    n = LINES[variant]
    dst = os.path.join(root, f"{os.path.basename(os.path.normpath(lake))}_{variant}")
    done = os.path.join(dst, ".complete")
    if os.path.exists(done):
        return dst
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for d, _, files in os.walk(lake, followlinks=True):
        out = os.path.join(dst, os.path.relpath(d, lake))
        os.makedirs(out, exist_ok=True)
        for f in files:
            src, tgt = os.path.join(d, f), os.path.join(out, f)
            if f.lower().endswith(".csv"):
                with open(src, "rb") as a, open(tgt, "wb") as b:
                    for i, line in enumerate(a):
                        if i >= n:
                            break
                        b.write(line)
            else:
                shutil.copy2(src, tgt)
    open(done, "w").close()
    return dst


def replay(rec, lake, timeout):
    ex = kb_run.KBExecutor(lake, timeout=timeout)
    out, lost = [], False
    try:
        for c in rec.get("cells") or []:
            if c.get("no_code"):
                continue
            row = {"task": rec["task"], "step": c["step"], "rec_ok": bool(c.get("ok")),
                   "rec_error_type": c.get("error_type"), "rec_error": (c.get("error") or "")[-1500:]}
            if lost:
                row.update(dry="not_run")
                out.append(row)
                continue
            before = ex.ls()
            t0 = time.perf_counter()
            r = ex.run(c["code"])
            after = ex.ls()
            wrote = None
            if before and after:
                wrote = len(set(after.get("files") or []) - set(before.get("files") or []))
            row.update(dry="ran", dry_ok=bool(r.get("ok")), dry_error_type=r.get("error_type"),
                       dry_error=(r.get("error") or "")[-1500:], seconds=round(time.perf_counter() - t0, 3),
                       new_files=wrote)
            out.append(row)
            if r.get("lost"):
                lost = True
                try:
                    ex.close()
                except Exception:  # noqa: BLE001
                    pass
        return out
    finally:
        try:
            ex.close()
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--lake", help="one data directory for every trace")
    ap.add_argument("--kb", help="kb_gate-style root: the lake of a trace is <kb>/data/<domain>/input")
    ap.add_argument("--variant", choices=sorted(LINES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--lakes-root", default="dry_lakes")
    a = ap.parse_args()
    lakes = {}

    def lake_of(r):
        src = os.path.abspath(a.lake) if a.lake else os.path.join(a.kb, "data", r["domain"], "input")
        if src not in lakes:
            root = a.lakes_root if a.lake else os.path.join(a.lakes_root, r["domain"])
            lakes[src] = truncated_lake(src, a.variant, root)
        return lakes[src]
    recs = [json.loads(line) for line in open(a.results)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["task"] for line in open(a.out)}
    todo = [r for r in recs if r["task"] not in done]
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor

    def job(r):
        try:
            rows = replay(r, lake_of(r), a.timeout)
        except Exception as e:  # noqa: BLE001  not written: the next run retries it
            print(f"{r['task']}: error {type(e).__name__}: {e}", flush=True)
            return
        with lock:
            with open(a.out, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        print(f"{r['task']}: {len(rows)} cells", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(job, todo))
    print("done")


if __name__ == "__main__":
    main()
