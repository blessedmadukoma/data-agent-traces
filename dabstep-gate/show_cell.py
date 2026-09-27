#!/usr/bin/env python3
"""Print everything you need to label one audit cell.

Shows the task question and guidelines, every earlier cell in the same
trace (code and output, shortened), and the selected cell in full.

Usage:
  python3 show_cell.py --data $DATA --audit $RUN/audit/audit_failed_v2.jsonl \
      --cells $RUN/parsed_*_cells.jsonl --id 17
"""
import argparse
import json
import os
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--short", type=int, default=600,
                    help="characters shown for earlier and later cells")
    ap.add_argument("--after", type=int, default=0,
                    help="also show this many later cells of the same trace")
    a = ap.parse_args()
    target = next(json.loads(l)
                  for l in open(a.audit) if json.loads(l)["audit_id"] == a.id)
    tasks = {str(json.loads(l)["task_id"]): json.loads(
        l) for l in open(os.path.join(a.data, "data/tasks/all.jsonl"))}
    t = tasks.get(str(target["task_id"]), {})
    print(
        f"=== audit {a.id}: {target['submission_file']} | task {target['task_id']} ({t.get('level')}) | cell {target['cell_index']}")
    print("QUESTION:", t.get("question"))
    print("GUIDELINES:", t.get("guidelines"))
    earlier, later = [], []
    for p in a.cells:
        for l in open(p):
            r = json.loads(l)
            if (r["submission_file"], str(r["task_id"])) == (target["submission_file"], str(target["task_id"])):
                if r["cell_index"] < target["cell_index"]:
                    earlier.append(r)
                elif target["cell_index"] < r["cell_index"] <= target["cell_index"] + a.after:
                    later.append(r)
    for r in sorted(earlier, key=lambda r: r["cell_index"]):
        print(f"\n--- earlier cell {r['cell_index']} (failed={r['failed']})")
        print(r["code"][:a.short])
        print(">>>", ANSI.sub("", r["output"] or "")[:a.short])
    print(f"\n=== SELECTED CELL {target['cell_index']} (failed={target['failed']}, "
          f"error={target.get('error_type')}, key={target.get('error_key')})")
    print(target["code"])
    print(">>>")
    print(ANSI.sub("", target["output"] or ""))
    for r in sorted(later, key=lambda r: r["cell_index"]):
        print(
            f"\n--- later cell {r['cell_index']} (failed={r['failed']}, error={r.get('error_type')}, key={r.get('error_key')})")
        print(r["code"][:a.short])
        print(">>>", ANSI.sub("", r["output"] or "")[:a.short])


if __name__ == "__main__":
    main()
