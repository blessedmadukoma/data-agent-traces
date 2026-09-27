#!/usr/bin/env python3
"""Extract code cells from DeepAnalyze-style XML traces (<Code> ... </Code> <Execute> ... </Execute>).

Replacement for the first-audit parser. A cell failed when its <Execute>
output contains a Python traceback or starts with "[Error]". Error type and
missing key come from the last traceback line.

Usage:
  python3 extract_xml_trace_cells.py <submissions_dir> <out.jsonl> [--manifest manifest.jsonl]

Without --manifest every file is read. With --manifest only included files
with trace_format "xml" are read.
"""
from dabstep_corpus_recount import classify, EXE
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CELL = re.compile(r"<Code>(.*?)</Code>\s*<Execute>(.*?)</Execute>", re.S)
LAST = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_.]*(?:Error|Exception)): ?([^\n]{0,300})$", re.M)
FENCE = re.compile(r"^\s*```(?:python|py)?\s*\n|\n\s*```\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions_dir")
    ap.add_argument("out")
    ap.add_argument("--manifest")
    a = ap.parse_args()
    files = sorted(os.listdir(a.submissions_dir))
    if a.manifest:
        man = [json.loads(l) for l in open(a.manifest)]
        keep = {m["submission_file"] for m in man if m.get(
            "included") and m.get("trace_format") == "xml"}
        files = [f for f in files if f in keep]
    total = failed = 0
    with open(a.out, "w") as w:
        for fn in files:
            p = os.path.join(a.submissions_dir, fn)
            if os.path.getsize(p) < 200:
                continue
            n = f = 0
            for line_no, line in enumerate(open(p), 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                t = r.get("reasoning_trace") or ""
                if not isinstance(t, str) or "<Code>" not in t:
                    continue
                for i, (code, out) in enumerate(CELL.findall(t)):
                    code = FENCE.sub("", code.strip("\n"))
                    bad = ("Traceback (most recent call last)" in out) or out.strip(
                    ).startswith("[Error]")
                    et = key = None
                    if bad:
                        last = LAST.findall(out)
                        et, key = classify(
                            last[-1][0] + ": " + last[-1][1]) if last else ("other", None)
                    w.write(json.dumps(dict(submission_file=fn, task_id=str(r.get("task_id")), trace_line=line_no,
                                            cell_index=i, code=code, output=out, failed=bad, error_type=et,
                                            error_key=key, executor_rule=et if et in EXE else None)) + "\n")
                    n += 1
                    f += bad
            if n:
                print(f"{n:7d} cells, {f:6d} failed  {fn}")
            total += n
            failed += f
    print(f"total cells: {total}; failed: {failed}")


if __name__ == "__main__":
    main()
