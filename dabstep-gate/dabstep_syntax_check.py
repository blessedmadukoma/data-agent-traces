#!/usr/bin/env python3
"""Check DABstep for harness-caused SyntaxErrors (the DataAgentBench defect class).

Usage: python3 dabstep_syntax_check.py --manifest "$RUN/manifest.jsonl" --cells "$RUN"/parsed_*_cells.jsonl [--show 5]

A failed cell whose recorded error is a SyntaxError, but whose recorded code
compiles, did not run as recorded: the harness changed the code, extracted
different code, or ran it under an older Python. Python 3.12 accepts f-strings
that 3.11 rejects (PEP 701), so each such cell is also compiled with the
3.11 grammar rules approximated by feature_version=(3, 11). The count is an
upper bound on harness-caused SyntaxErrors; a sample must be read.
"""
import argparse
import ast
import collections
import json
import warnings

warnings.filterwarnings("ignore")


def compiles(code, fv=None):
    try:
        if fv:
            ast.parse(code, feature_version=fv)
        else:
            compile(code, "<cell>", "exec", dont_inherit=True)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    C = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    shown = 0
    for p in a.cells:
        for line in open(p):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if not (m and m["included"]) or not r["failed"] or r.get("error_type") != "SyntaxError":
                continue
            s = m["submitter"]
            code = (r.get("code") or "").strip()
            ok312 = compiles(code)
            ok311 = ok312 and compiles(code, (3, 11))
            for c in (C, per[s]):
                c["syntax_failed"] += 1
                c["compiles_3.12"] += ok312
                c["compiles_3.11"] += ok311
            if ok311 and shown < a.show:
                shown += 1
                print(f"===== {r['submission_file']} task {r['task_id']} cell {r['cell_index']}")
                print(code[:600])
                print("--- recorded error:", (r.get("output") or "")[:300].replace("\n", " | "))
    print(f"SyntaxError cells {C['syntax_failed']}; recorded code compiles under 3.12: {C['compiles_3.12']}; "
          f"also with the 3.11 grammar: {C['compiles_3.11']}")
    for s, c in sorted(per.items()):
        print(f"  {s:18s} {c['syntax_failed']:6d} {c['compiles_3.12']:6d} {c['compiles_3.11']:6d}")


if __name__ == "__main__":
    main()
