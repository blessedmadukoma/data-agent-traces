#!/usr/bin/env python3
"""Score dry runs (baselines/dry_run.py) and compare with gate predictions on the same cells.

A dry-run block is a cell that raises KeyError or FileNotFoundError in the replay on
truncated data. It is cause-matched to a recorded failure as in kb_gate.py (the same
missing key, or the same missing path). A false block is a dry-run block on a cell
that ran without error in the recording.

Usage: python3 baselines/dry_run_score.py --dry runs/dry/qr_gpt-oss_h0.jsonl \
    --gate runs/qr/gate_v7_gpt-oss.jsonl [--gate runs/qr/gate_v8_gpt-oss.jsonl]
"""
import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kramabench"))
import kb_gate  # noqa: E402

DATA_ERRORS = ("KeyError", "FileNotFoundError")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), (c + h) / d)


def matches(rec_err, dry_err, et):
    k1, p1 = kb_gate.error_names(rec_err)
    k2, p2 = kb_gate.error_names(dry_err)
    if et == "KeyError":
        return bool(k1 & k2)
    return p1 is not None and p2 is not None and kb_gate.same_path(p1, p2)


def score(rows):
    s = {"cells": len(rows), "not_run": sum(r["dry"] == "not_run" for r in rows)}
    ran = [r for r in rows if r["dry"] == "ran"]
    fails = [r for r in ran if not r["rec_ok"] and r["rec_error_type"] in DATA_ERRORS]
    s["data_failures"] = sum(1 for r in rows if not r["rec_ok"] and r["rec_error_type"] in DATA_ERRORS)
    s["caught"] = sum(1 for r in fails if r["dry_error_type"] == r["rec_error_type"]
                      and matches(r["rec_error"], r["dry_error"], r["rec_error_type"]))
    ok = [r for r in ran if r["rec_ok"]]
    s["ok_cells_ran"] = len(ok)
    s["false_blocks"] = sum(1 for r in ok if r["dry_error_type"] in DATA_ERRORS)
    s["false_blocks_by_type"] = {t: sum(1 for r in ok if r["dry_error_type"] == t) for t in DATA_ERRORS}
    s["other_errors_on_ok"] = sum(1 for r in ok if not r["dry_ok"] and r["dry_error_type"] not in DATA_ERRORS)
    s["new_files_cells"] = sum(1 for r in ran if r.get("new_files"))
    secs = sorted(r["seconds"] for r in ran if r.get("seconds") is not None)
    if secs:
        s["seconds_median"] = statistics.median(secs)
        s["seconds_p95"] = secs[int(0.95 * len(secs))]
    s["caught_ci"] = wilson(s["caught"], s["data_failures"])
    s["false_block_rate_ci"] = wilson(s["false_blocks"], s["ok_cells_ran"])
    return s


def gate_score(path, keys):
    g = {(r["task"], r["step"]): r for r in map(json.loads, open(path))}
    rows = [g[k] for k in keys if k in g]
    fails = [r for r in rows if not r["ok"] and r["error_type"] in DATA_ERRORS]
    ok = [r for r in rows if r["ok"]]
    return {"cells": len(rows), "data_failures": len(fails), "caught": sum(1 for r in fails if r.get("cause_match")),
            "ok_cells": len(ok), "blocks_on_ok": sum(1 for r in ok if r["outcome"] == "block")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", required=True)
    ap.add_argument("--gate", action="append", default=[])
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.dry)]
    out = {"dry_run": score(rows)}
    keys = [(r["task"], r["step"]) for r in rows]
    for g in a.gate:
        out[os.path.basename(g)] = gate_score(g, keys)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
