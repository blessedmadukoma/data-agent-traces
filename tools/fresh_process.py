"""For each DABstep submission, the share of cells that read a name from an earlier cell and fail with NameError.

A high share means the harness ran every cell in a fresh process, so names did not persist.

Usage: uv run python tools/fresh_process.py --manifest runs/dabstep/manifest.jsonl --cells runs/dabstep/parsed_*_cells.jsonl
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dabstep-gate"))
import gate_adapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    a = ap.parse_args()
    manifest = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    traces = collections.defaultdict(list)
    for path in a.cells:
        for line in open(path):
            cell = json.loads(line)
            m = manifest.get(cell["submission_file"])
            if m and m["included"]:
                traces[(cell["submission_file"], str(cell["task_id"]), cell.get("trace_line"))].append(cell)
    result = gate_adapter.stateful_submissions(traces)
    rows = sorted(result.items(), key=lambda item: -(item[1][1] / item[1][2] if item[1][2] else 0))
    print(f"{'submission':70} {'fail':>6} {'cells':>6} {'share':>6}  harness")
    for name, (stateful, failed, cells) in rows:
        share = failed / cells if cells else 0
        kind = "keeps state" if stateful else "fresh process"
        print(f"{name[:70]:70} {failed:6} {cells:6} {share:6.0%}  {kind}")


if __name__ == "__main__":
    main()
