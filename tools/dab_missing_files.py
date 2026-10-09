"""Classify the DAB reads of missing files that the checker blocks (preprint, Section 3.3).

A name that only a stored result would have (tmp*, temp*, *result*, query_db*, func_*,
list_db*, *response*, data_<8 random characters>.json) counts as a file that the model invented
to hold an earlier result. The script prints every other blocked read with the code around it,
for reading by hand. Our reading of the 29 such reads (9 October 2026): 27 are also files that
the model invented to hold an earlier result (for example "Read the full result from the file",
/tmp/<random>.json); 2 name a file after a database (indextrade_database.json,
clinical_database.json).

Usage (after make dab): uv run python tools/dab_missing_files.py --run runs/dab
"""
import argparse
import collections
import glob
import json
import os
import re

STORED = re.compile(r"(^|/)(tmp|temp)|result|query_db|query_result|func_|list_db|response|^data_[a-z0-9]{8}\.json$",
                    re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/dab")
    a = ap.parse_args()
    blocked = []
    for path in glob.glob(os.path.join(a.run, "gate_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("outcome") == "block" and r.get("cause") and r.get("group") == "py:data-file":
                blocked.append(r)
    by_model = collections.Counter(r["trace"].split("/")[0] for r in blocked)
    rest = [r for r in blocked if not STORED.search((r.get("missing") or [""])[0])]
    print(f"blocked reads of missing files: {len(blocked)} {dict(by_model)}")
    print(f"names of stored results: {len(blocked) - len(rest)}; to read by hand: {len(rest)}")
    want = {(r["trace"], r["i"]) for r in rest}
    code = {}
    for path in glob.glob(os.path.join(a.run, "cells_*.jsonl")):
        for line in open(path):
            t = json.loads(line)
            for c in t["calls"]:
                if (t["trace"], c["i"]) in want:
                    code[(t["trace"], c["i"])] = c.get("code") or ""
    for r in rest:
        name = r["missing"][0]
        text = code.get((r["trace"], r["i"]), "")
        at = text.find(os.path.basename(name.rstrip("/")))
        print(f"== {r['trace']} call {r['i']}: {name}")
        print("   " + text[max(0, at - 200):at + 80].replace("\n", " | "))


if __name__ == "__main__":
    main()
