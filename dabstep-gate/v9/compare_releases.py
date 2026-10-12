#!/usr/bin/env python3
"""Compare the decisions of two checker releases on the same recorded steps.

Each input is a prediction file written by one of the evaluation scripts:
  DABstep   dabstep-gate/evaluate_gate.py   key (submission_file, task_id, trace_line, cell_index)
  DAB       dataagentbench-replay/dab_gate_eval.py   key (trace, i)
  our runs  kramabench/kb_gate.py and infiagent/mount_exists.py   key (task, step)

For every step it compares the outcome (block, run, unknown) of release A with
that of release B and prints, per file and in total: the steps compared, the
decisions that changed, the blocks that B makes and A does not, the blocks of
working steps of each release, and the blocks with a cause match (the step
failed on the name or file that the block names) of each release.

Usage:
  python3 compare_releases.py --pair NAME A.jsonl B.jsonl [--pair ...] [--json out.json]
"""
import argparse
import collections
import json


def key_of(r):
    if "submission_file" in r:
        return (r["submission_file"], r["task_id"], r["trace_line"], r["cell_index"])
    if "trace" in r:
        return (r["trace"], r["i"])
    return (r["task"], r["step"])


def working(r):
    if "ok" in r:
        return bool(r["ok"])
    return not r.get("failed")


def cause(r):
    return bool(r.get("cause_match") or r.get("cause"))


def load(path):
    out = collections.OrderedDict()
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            k = key_of(r)
            if k in out:
                raise SystemExit(f"{path}: duplicate key {k}")
            out[k] = r
    return out


def compare(a_path, b_path):
    a, b = load(a_path), load(b_path)
    if set(a) != set(b):
        raise SystemExit(f"{a_path} and {b_path} hold different steps ({len(a)} and {len(b)})")
    res = {"steps": len(a), "changed": 0, "transitions": collections.Counter(), "new_blocks": [],
           "a_blocks": 0, "b_blocks": 0, "a_working_blocks": 0, "b_working_blocks": 0,
           "a_cause_blocks": 0, "b_cause_blocks": 0, "lost_cause_blocks": []}
    for k, ra in a.items():
        rb = b[k]
        oa, ob = ra["outcome"], rb["outcome"]
        res["a_blocks"] += oa == "block"
        res["b_blocks"] += ob == "block"
        res["a_working_blocks"] += oa == "block" and working(ra)
        res["b_working_blocks"] += ob == "block" and working(rb)
        res["a_cause_blocks"] += oa == "block" and cause(ra)
        res["b_cause_blocks"] += ob == "block" and cause(rb)
        if oa != ob:
            res["changed"] += 1
            res["transitions"][f"{oa} -> {ob}"] += 1
            if ob == "block":
                res["new_blocks"].append(list(k))
            if oa == "block" and cause(ra):
                res["lost_cause_blocks"].append({"key": list(k), "b_outcome": ob, "b_reason": rb.get("reason")})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "A", "B"), required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    total = collections.Counter()
    out = {}
    for name, pa, pb in a.pair:
        r = compare(pa, pb)
        out[name] = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in r.items()}
        for k in ("steps", "changed", "a_blocks", "b_blocks", "a_working_blocks", "b_working_blocks",
                  "a_cause_blocks", "b_cause_blocks"):
            total[k] += r[k]
        total["new_blocks"] += len(r["new_blocks"])
        total["lost_cause_blocks"] += len(r["lost_cause_blocks"])
        print(f"{name}: {r['steps']} steps; {r['changed']} decisions changed {dict(r['transitions'])}; "
              f"blocks {r['a_blocks']} -> {r['b_blocks']}; blocks of working steps {r['a_working_blocks']} -> "
              f"{r['b_working_blocks']}; cause-matched blocks {r['a_cause_blocks']} -> {r['b_cause_blocks']}; "
              f"new blocks {len(r['new_blocks'])}")
    print(f"TOTAL: {total['steps']} steps; {total['changed']} decisions changed; blocks {total['a_blocks']} -> "
          f"{total['b_blocks']}; blocks of working steps {total['a_working_blocks']} -> {total['b_working_blocks']}; "
          f"cause-matched blocks {total['a_cause_blocks']} -> {total['b_cause_blocks']}; "
          f"new blocks {total['new_blocks']}; cause-matched blocks lost {total['lost_cause_blocks']}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"total": dict(total), "files": out}, fh, indent=1)


if __name__ == "__main__":
    main()
