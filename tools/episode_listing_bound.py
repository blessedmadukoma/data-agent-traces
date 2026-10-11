#!/usr/bin/env python3
"""Sensitivity of the DABstep error-episode shares to the inspection rule (preprint v3.13, Section 3.4).

An error episode ends at the first later working step that the rule of recovery_episodes.is_inspection
labels as computation. A hand check found that the rule labels some file listings as computation, so
episodes can end too soon. This script prints the share of episodes that end within two steps under
three rules:
  base        recovery_episodes.is_inspection, as in recovery_report.txt (61%);
  extended    the base rule with more listing calls (os.walk, scandir, iterdir, rglob, ls, path
              checks and file sizes), as in the receipt-replay sensitivity of preprint_v3_numbers.py;
  aggressive  the base rule, and every step that calls a file-listing function counts as inspection,
              even when it also reads data and computes. This overcounts inspection, so its share is a
              lower bound for this kind of error in the rule.

Usage (after `make dabstep`):
  uv run python tools/episode_listing_bound.py --manifest runs/dabstep/manifest.jsonl \
      --cells runs/dabstep/parsed_*_cells.jsonl
"""
import argparse
import ast
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dabstep-gate"))
import recovery_episodes as RE  # noqa: E402

EXTRA = {"walk", "iterdir", "scandir", "rglob", "exists", "isfile", "isdir", "getsize", "stat", "ls"}
LISTING = {"listdir", "glob", "walk", "iterdir", "scandir", "rglob"}


def lists_files(code):
    tree = RE._parse(code or "")
    return tree is not None and any(
        isinstance(n, ast.Call) and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) in LISTING
        for n in ast.walk(tree))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    a = ap.parse_args()
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    traces = collections.defaultdict(list)
    for path in a.cells:
        for line in open(path):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if m and m["included"]:
                traces[(r["submission_file"], str(r["task_id"]), r.get("trace_line"))].append(r)
    for cells in traces.values():
        cells.sort(key=lambda r: r["cell_index"])

    base_look, base_plain, base_rule = set(RE.LOOK_CALLS), set(RE.PLAIN), RE.is_inspection
    for name in ("base", "extended", "aggressive"):
        RE.LOOK_CALLS.clear()
        RE.LOOK_CALLS |= base_look | (EXTRA if name == "extended" else set())
        RE.PLAIN.clear()
        RE.PLAIN |= base_plain | (EXTRA if name == "extended" else set())
        RE.is_inspection = (lambda code: base_rule(code) or lists_files(code)) if name == "aggressive" else base_rule
        eps = [e for cells in traces.values() for e in RE.episodes(cells)]
        s = RE.summary(eps)
        print(f"{name:10s} episodes {s['episodes']}; next step ends the episode {s['fix_next']:.1%}; "
              f"within two steps {s['fix_next'] + s['fix_2']:.1%}; three or more {s['fix_3plus']:.1%}; "
              f"no later working step {s['never_ok']:.1%}")
    RE.is_inspection = base_rule


if __name__ == "__main__":
    main()
