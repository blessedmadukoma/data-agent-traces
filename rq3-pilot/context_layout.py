#!/usr/bin/env python3
"""Where each DABstep submission kept its context files, from its own traces.

The RQ3 sandbox must give the model the workspace that the original run had.
A recorded trace does not list the workspace, so this module uses the replay
evidence of gate_adapter.FileEvidence at the level of the submission:

  * a directory D holds the context files if some cell of the submission read
    a context file in D, in code that always runs, and did not fail;
  * a context file is missing from D if some cell of the submission failed
    with FileNotFoundError on it.

Assumption: the harness put all seven context files together, so every
context file without evidence of absence is placed in D. `check` counts the
directories for which the submission's own traces contradict this.

Usage: python3 rq3-pilot/context_layout.py check --gate-dir dabstep-gate --data dab \
    --manifest $RUN/manifest.jsonl --cells $RUN/parsed_*_cells.jsonl
"""
import argparse
import collections
import json
import os
import sys

CONTEXT = ("payments.csv", "payments-readme.md", "acquirer_countries.csv", "fees.json",
           "merchant_category_codes.csv", "merchant_data.json", "manual.md")


def layouts(ev, context=CONTEXT):
    """{submission_file: {dir: {"ok": set(base), "missing": set(base)}}} from a FileEvidence."""
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: {"ok": set(), "missing": set()}))
    for fn, paths in ev.ok.items():
        for p in paths:
            b = os.path.basename(p)
            if b in context:
                out[fn][os.path.dirname(p)]["ok"].add(b)
    for fn, paths in ev.missing.items():
        for p in paths:
            b = os.path.basename(p)
            if b in context and os.path.dirname(p) in out[fn]:
                out[fn][os.path.dirname(p)]["missing"].add(b)
    return out


def placements(layout, context=CONTEXT):
    """{path: base name} for the sandbox: every context file in every directory that
    held context files, except a file that the submission failed to find there and
    never read there. (Where the submission both read and failed to find the same
    file, its workspace changed between traces; the file is placed, and the caller
    removes it if the episode's own trace failed to find it.)"""
    return {os.path.join(d, b) if d else b: b
            for d, v in layout.items() for b in context if b in v["ok"] or b not in v["missing"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check"])
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.gate_dir))
    import gate_adapter as ga  # noqa: E402
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    traces = collections.defaultdict(list)
    for p in a.cells:
        for line in open(p):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if m and m["included"]:
                traces[(r["submission_file"], str(r["task_id"]), r.get("trace_line"))].append(r)
    rp = ga.Replay(a.gate_dir, a.data, traces, man, carry=True, files=True, tolerant="auto")
    L = layouts(rp.ev)
    n_dirs = n_bad = 0
    for fn in sorted(L):
        for d, v in sorted(L[fn].items()):
            n_dirs += 1
            if v["missing"]:
                n_bad += 1
                print(f"CONTRADICTION {man[fn]['submitter']} | {fn[:50]} | {d!r} ok={sorted(v['ok'])} "
                      f"missing={sorted(v['missing'])}")
    print(f"submissions {len(L)}; directories with context files {n_dirs}; "
          f"with a context file recorded missing {n_bad}")
    per = collections.Counter()
    for fn in L:
        for d, v in L[fn].items():
            per[len(v["ok"])] += 1
    print("directories by number of context files read OK:", dict(sorted(per.items())))


if __name__ == "__main__":
    main()
