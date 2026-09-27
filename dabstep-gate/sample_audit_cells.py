#!/usr/bin/env python3
"""Draw the stratified audit sample and write labelling sheets.

Failed cells: N per submitter, spread across that submitter's submissions,
at most one failed cell per trace. Controls: cells without an error that
contain a data access. Selection order is SHA-256(salt + file + task + cell),
so the same inputs and salt always give the same sample.

Outputs in --out-dir:
  audit_failed_v2.jsonl       selected failed cells (full code and output)
  audit_controls_v2.jsonl     selected control cells
  labels_v2.csv               labelling sheet, first labeller (blank label columns)
  labels_v2_second.csv        30 random rows for the second labeller (no first-labeller columns)

Usage:
  python3 sample_audit_cells.py --manifest $RUN/manifest.jsonl \
      --cells $RUN/parsed_*_cells.jsonl --out-dir $RUN/audit \
      --per-submitter 20 --controls 300 --salt dabstep-audit-v2
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re

DATA_ACCESS = re.compile(
    r"read_csv|read_json|json\.load|\[['\"][A-Za-z_]+['\"]\]|\.loc\[|\.query\(")
LABEL_COLS = ["failure_class", "program_subclass", "exact_precondition", "visible_before_execution",
              "requirement_location", "evidence", "confidence"]


def h(salt, r):
    return hashlib.sha256(f"{salt}|{r['submission_file']}|{r['task_id']}|{r['cell_index']}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-submitter", type=int, default=20)
    ap.add_argument("--controls", type=int, default=300)
    ap.add_argument("--second", type=int, default=30)
    ap.add_argument("--salt", required=True)
    a = ap.parse_args()
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    inc = {f for f, m in man.items() if m["included"]}
    failed, ok = collections.defaultdict(list), collections.defaultdict(list)
    for path in a.cells:
        for line in open(path):
            r = json.loads(line)
            if r["submission_file"] not in inc:
                continue
            r["submitter"] = man[r["submission_file"]]["submitter"]
            (failed if r["failed"] else ok)[r["submitter"]].append(r)
    os.makedirs(a.out_dir, exist_ok=True)

    sel = []
    for sub in sorted(failed):
        by_file = collections.defaultdict(list)
        for r in sorted(failed[sub], key=lambda r: h(a.salt, r)):
            by_file[r["submission_file"]].append(r)
        used_traces, picked = set(), []
        queues = [by_file[f] for f in sorted(by_file)]
        while len(picked) < a.per_submitter and any(queues):
            for q in queues:
                while q:
                    r = q.pop(0)
                    t = (r["submission_file"], r["task_id"])
                    if t not in used_traces:
                        used_traces.add(t)
                        picked.append(r)
                        break
                if len(picked) >= a.per_submitter:
                    break
        sel += picked
        print(
            f"failed sample: {sub:15s} {len(picked):3d} cells from {len(by_file)} submissions")

    ctrl_pool = [r for s in ok for r in ok[s]
                 if DATA_ACCESS.search(r.get("code") or "")]
    per = max(1, a.controls // max(1, len(ok)))
    ctrl = []
    for s in sorted(ok):
        pool = sorted([r for r in ctrl_pool if r["submitter"]
                      == s], key=lambda r: h(a.salt, r))
        ctrl += pool[:per]
    taken = {(r["submission_file"], r["task_id"], r["cell_index"])
             for r in ctrl}
    extra = sorted([r for r in ctrl_pool if (r["submission_file"], r["task_id"], r["cell_index"]) not in taken],
                   key=lambda r: h(a.salt, r))
    ctrl += extra[:max(0, a.controls - len(ctrl))]

    def dump(rows, name):
        with open(os.path.join(a.out_dir, name), "w") as w:
            for i, r in enumerate(rows, 1):
                w.write(json.dumps(dict(audit_id=i, **r)) + "\n")
    dump(sel, "audit_failed_v2.jsonl")
    dump(ctrl, "audit_controls_v2.jsonl")

    base = ["audit_id", "submitter", "submission_file",
            "task_id", "cell_index", "error_type", "error_key"]
    with open(os.path.join(a.out_dir, "labels_v2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base + ["labeller"] + LABEL_COLS)
        for i, r in enumerate(sel, 1):
            w.writerow([i] + [r.get(k)
                       for k in base[1:]] + [""] * (1 + len(LABEL_COLS)))
    rnd = random.Random(a.salt)
    second = sorted(rnd.sample(
        range(1, len(sel) + 1), min(a.second, len(sel))))
    with open(os.path.join(a.out_dir, "labels_v2_second.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base + ["labeller"] + LABEL_COLS)
        for i in second:
            r = sel[i - 1]
            w.writerow([i] + [r.get(k)
                       for k in base[1:]] + [""] * (1 + len(LABEL_COLS)))
    print(
        f"failed cells: {len(sel)}; controls: {len(ctrl)}; second-labeller rows: {len(second)}")
    print(f"open {a.out_dir}/labels_v2.csv and label each audit_id using audit_failed_v2.jsonl for the code and output.")


if __name__ == "__main__":
    main()
