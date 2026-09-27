#!/usr/bin/env python3
"""False-block audit.

Mode 1, sample: run your gate on every cell without a recorded error,
report the non-error block rate per submitter, and write a sheet with a
deterministic sample of blocked cells for you to classify.

  python3 sample_false_block_audit.py sample --gate-dir . --data $DATA \
      --manifest $RUN/manifest.jsonl --cells $RUN/parsed_*_cells.jsonl \
      --n 60 --salt fb-audit-v1 --out $RUN/false_block_audit.csv [--carry] [--files] [--tolerant auto]

Use the same --carry, --files and --tolerant options as evaluate_gate.py, so
the audit checks the same gate.

Fill the column fb_class in the CSV with one of:
  not_executed, created_earlier, path_mismatch, workspace_file, gate_defect   (false blocks)
  harness_semantics                                                           (false block: the executor
                                                                              does not raise where Python does)
  error_swallowed, unrecorded_error                                           (the gate was right)

Mode 2, summarise: read the filled sheet and compute the confirmed false-block rate.

  python3 sample_false_block_audit.py summarise --sheet $RUN/false_block_audit.csv
"""
import gate_adapter as ga
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FALSE = {"not_executed", "created_earlier",
         "path_mismatch", "workspace_file", "harness_semantics", "gate_defect"}
TRUE = {"error_swallowed", "unrecorded_error"}  # the gate was right


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def sample(a):
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    traces = collections.defaultdict(list)
    for p in a.cells:
        for line in open(p):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if m and m["included"]:
                traces[(r["submission_file"], str(r["task_id"]), r.get("trace_line"))].append(r)
    rp = ga.Replay(a.gate_dir, a.data, traces, man, carry=a.carry, files=a.files, tolerant=a.tolerant)
    per = collections.defaultdict(collections.Counter)
    blocked = []
    for tkey, cells in traces.items():
        for r, res, o, _ in rp.run(tkey, cells):
            if r["failed"]:
                continue   # the audit covers cells without a recorded error
            s = man[tkey[0]]["submitter"]
            per[s]["cells"] += 1
            per[s][o] += 1
            if o == "block":
                r["gate_reason"] = ga.reason(res)
                r["submitter"] = s
                blocked.append(r)
    tot = collections.Counter()
    print("submitter        non-error cells   block   unknown   run   gate_error")
    for s, c in sorted(per.items()):
        tot.update(c)
        print(
            f"{s:15s} {c['cells']:15d} {c['block']:7d} {c['unknown']:9d} {c['run']:5d} {c['gate_error']:12d}")
    lo, hi = wilson(tot["block"], tot["cells"])
    rate = tot["block"] / max(tot["cells"], 1)
    print(
        f"non-error block rate: {tot['block']}/{tot['cells']} = {rate:.2%} (Wilson 95% CI {lo:.2%}-{hi:.2%})")

    def key(r): return hashlib.sha256(
        f"{a.salt}|{r['submission_file']}|{r['task_id']}|{r['cell_index']}".encode()).hexdigest()
    pick = sorted(blocked, key=key)[:a.n]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "submitter", "submission_file",
                   "task_id", "cell_index", "gate_reason", "fb_class", "notes"])
        for i, r in enumerate(pick, 1):
            w.writerow([i, r["submitter"], r["submission_file"],
                       r["task_id"], r["cell_index"], r["gate_reason"], "", ""])
    with open(a.out + ".cells.jsonl", "w") as f:
        for i, r in enumerate(pick, 1):
            f.write(json.dumps(dict(audit_id=i, **r)) + "\n")
    with open(a.out + ".meta.json", "w") as f:
        json.dump(dict(blocks=tot["block"],
                  cells=tot["cells"], salt=a.salt), f)
    print(
        f"wrote {len(pick)} blocked cells to {a.out} (code in {a.out}.cells.jsonl)")


def summarise(a):
    rows = list(csv.DictReader(open(a.sheet)))
    meta = json.load(open(a.sheet + ".meta.json"))
    c = collections.Counter(r["fb_class"].strip() for r in rows)
    done = [r for r in rows if r["fb_class"].strip()]
    k = sum(r["fb_class"].strip() in FALSE for r in done)
    bad = sorted({r["fb_class"].strip() for r in done} - FALSE - TRUE)
    if bad:
        print("unknown class names (fix the sheet):", bad)
        return
    print("classes:", dict(c))
    if not done:
        print("no rows classified yet")
        return
    share = k / len(done)
    lo, hi = wilson(k, len(done))
    rate = meta["blocks"] / meta["cells"]
    print(
        f"confirmed false blocks in the audit: {k}/{len(done)} = {share:.0%} (95% CI {lo:.0%}-{hi:.0%})")
    print(f"estimated confirmed false-block rate: {rate * share:.2%} "
          f"(range {rate * lo:.2%}-{rate * hi:.2%}); decision rule K4 limit is 1%")
    print("K4:", "FAILS (above 1%)" if rate * lo >
          0.01 else "passes (below 1%)" if rate * hi < 0.01 else "undecided: audit more cells")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    s = sp.add_parser("sample")
    s.add_argument("--gate-dir", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--cells", nargs="+", required=True)
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--salt", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--carry", action="store_true",
                   help="carry tracked DataFrames across cells, as evaluate_gate.py --carry")
    s.add_argument("--files", action="store_true", help="as evaluate_gate.py --files")
    s.add_argument("--tolerant", choices=["none", "auto", "all"], default="none",
                   help="as evaluate_gate.py --tolerant")
    m = sp.add_parser("summarise")
    m.add_argument("--sheet", required=True)
    a = ap.parse_args()
    sample(a) if a.mode == "sample" else summarise(a)


if __name__ == "__main__":
    main()
