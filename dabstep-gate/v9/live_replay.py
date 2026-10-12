#!/usr/bin/env python3
"""Decide every step of the live QRData experiment again, with release 3 (version 8) and release 4 (version 9).

The live experiment ran frozen release 3 inside the agent's loop (live/e1_run.py). This script
takes the recorded runs of the two arms with the checker, and for every step of each run, in
order, asks each release for its decision with the same state that the live run carried: a
blocked step counts as failed, a step that ran counts as failed if it raised an error.

The live run answered whether a file exists from the sandbox. The recording does not keep the
sandbox, so this script answers from the evidence that the run recorded:
  * a path that a receipt of the run named as missing, or that a FileNotFoundError of the run
    named, does not exist;
  * a path under input/ (the benchmark's data folder) exists if the file is in the data folder;
  * a path that an earlier step of the run wrote exists;
  * any other path is unknown.
With this evidence, release 3 should make the recorded decision on every blocked step; the
script reports how often it does, and then compares release 4 with it.

Usage: python3 live_replay.py --results recorded/e1/results.jsonl --tables data/QRData/benchmark/data \\
           --kb runs/rescore/qr_kb
"""
import argparse
import collections
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FNF = re.compile(r"No such file or directory: '([^']+)'|File (\S+) does not exist")


def load_gate(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--kb", required=True, help="unused; kept for the same call as tools/rescore.sh")
    a = ap.parse_args()
    root = os.path.join(HERE, "..", "..")
    v8 = load_gate(os.path.join(root, "dabstep-gate", "gate.py"), "gate_v8")
    v9 = load_gate(os.path.join(root, "dabstep-gate", "v9", "gate.py"), "gate_v9")
    sys.modules["gate"] = v8                    # kb_gate.domain_schema reads csv_columns from module "gate"
    sys.path.insert(0, os.path.join(root, "kramabench"))
    import kb_gate                               # noqa: E402
    lake = os.path.abspath(a.tables)
    schema, _, _ = kb_gate.domain_schema(lake)
    where, files = {}, set()
    for d, _, fs in os.walk(lake):
        for f in fs:
            rel = os.path.relpath(os.path.join(d, f), lake)
            where.setdefault(f, []).append(os.path.join("input", rel))
            files.add(os.path.normpath(os.path.join("input", rel)))

    def locate(base):
        return where.get(base, [])

    stats = collections.Counter()
    differ = []
    for line in open(a.results):
        r = json.loads(line)
        if not r["arm"].endswith("G1"):
            continue
        missing = set()
        for c in r["cells"]:
            if c.get("blocked"):
                missing.add(os.path.normpath(c["receipt"]["source"]))
            for m in FNF.finditer(c.get("error") or ""):
                missing.add(os.path.normpath(next(g for g in m.groups() if g)))
        written = set()

        def exists(path):
            p = os.path.normpath(str(path))
            if p in missing:
                return False
            if p in files or p in written:
                return True
            return None
        states = {"v8": v8.GateState(), "v9": v9.GateState()}
        for c in r["cells"]:
            if c.get("no_code"):
                continue
            dec = {}
            for tag, g in (("v8", v8), ("v9", v9)):
                try:
                    res = g.decide(c["code"], schema, states[tag], exists=exists, locate=locate)
                except Exception as e:                      # noqa: BLE001  a crash lets the step run, as live
                    res = {"outcome": "gate_error", "reason": str(e)}
                dec[tag] = res
            blocked = bool(c.get("blocked"))
            stats["steps"] += 1
            stats["blocked"] += blocked
            o8, o9 = dec["v8"]["outcome"], dec["v9"]["outcome"]
            if blocked:
                stats["v8_blocks_blocked"] += o8 == "block"
                stats["v9_blocks_blocked"] += o9 == "block"
                stats["v9_same_missing"] += o9 == "block" and dec["v9"].get("missing") == c["receipt"]["missing"]
            else:
                stats["v8_blocks_ran"] += o8 == "block"
                stats["v9_blocks_ran"] += o9 == "block"
            if o8 != o9:
                stats["changed"] += 1
                differ.append({"task": r["task"], "arm": r["arm"], "seed": r["seed"], "step": c["step"],
                               "blocked": blocked, "v8": o8, "v9": o9, "v9_reason": dec["v9"].get("reason")})
            failed = blocked or not c.get("ok")
            for tag in states:
                states[tag].commit(failed=failed)
            for w in dec["v8"].get("written", []):
                written.add(os.path.normpath(w))
    print(f"steps of the runs with the checker: {stats['steps']}; blocked live: {stats['blocked']}")
    print(f"release 3 with the recorded evidence blocks {stats['v8_blocks_blocked']} of the {stats['blocked']} "
          f"blocked steps and {stats['v8_blocks_ran']} of the steps that ran")
    print(f"release 4 blocks {stats['v9_blocks_blocked']} of the {stats['blocked']} blocked steps "
          f"({stats['v9_same_missing']} with the same missing name) and {stats['v9_blocks_ran']} of the steps that ran")
    print(f"decisions that differ between the releases: {stats['changed']}")
    for d in differ[:20]:
        print("  ", json.dumps(d)[:300])


if __name__ == "__main__":
    main()
