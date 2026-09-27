#!/usr/bin/env python3
"""Choose receipt-replay episodes from recorded DABstep traces (prefix continuation).

An episode is a trace cut at the first cell that the gate blocks with a
cause match (the gate names the missing column, key or file of the
recorded error). The episode keeps the earlier cells, the blocked cell, the
gate's receipt for it, and the paths at which the trace reads context
files, so that the sandbox can rebuild the state.

Usage:
  uv run rq3-pilot/select_prefixes.py --gate-dir dabstep-gate --data data/dabstep --manifest runs/dabstep/manifest.jsonl \
      --cells runs/dabstep/parsed_*_cells.jsonl --out runs/receipts/episodes.jsonl --n 10 --seed 1 [--max-share 0.3]

Sampling: a seeded shuffle, with at most --max-share of the sample from one
submitter, and at most one episode per (submission, task). --max-prefix
skips traces with more earlier cells than that (token cost).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import context_layout as cl  # noqa: E402


def sandbox_paths(layout, trace_missing):
    """Where the sandbox puts context files for one episode: every context file in
    every directory where the submission kept them (context_layout.placements),
    except a file that this trace, up to the blocked cell, failed to find there."""
    gone = {os.path.normpath(p) for p in trace_missing}
    return {p: b for p, b in cl.placements(layout).items() if os.path.normpath(p) not in gone}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-share", type=float, default=0.3)
    ap.add_argument("--max-prefix", type=int, default=12)
    ap.add_argument("--exclude", nargs="*", default=[], help="episode files whose (submission, task) pairs are left out")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.gate_dir))
    import gate_adapter as ga  # noqa: E402

    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    tasks = {str(t["task_id"]): t for t in map(json.loads, open(os.path.join(a.data, "data/tasks/all.jsonl")))}
    for t in map(json.loads, open(os.path.join(a.data, "data/tasks/dev.jsonl"))):
        tasks.setdefault(str(t["task_id"]), t)["answer"] = t.get("answer")
    traces = collections.defaultdict(list)
    for p in a.cells:
        for line in open(p):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if m and m["included"]:
                traces[(r["submission_file"], str(r["task_id"]), r.get("trace_line"))].append(r)
    rp = ga.Replay(a.gate_dir, a.data, traces, man, carry=True, files=True, tolerant="auto")
    context = set(rp.schema)
    layouts = cl.layouts(rp.ev)
    cands = []
    for tkey, cells in traces.items():
        fn, tid, tl = tkey
        if tid not in tasks:
            continue
        seen, paths, gone = [], {}, set()
        for r, res, o, _ in rp.run(tkey, cells):
            if r["failed"] and r.get("error_type") == "FileNotFoundError" and ga.fnf_path(r.get("output")):
                gone.add(ga.fnf_path(r.get("output")))
            for p, _, _ in (res.get("files") or []):
                base = os.path.basename(p)
                if base in context:
                    paths[p] = base
            viol = res.get("violations") or []
            missing = {v.get("missing") for v in viol}
            fpath = ga.fnf_path(r.get("output")) if r.get("error_type") == "FileNotFoundError" else None
            cause = o == "block" and r["failed"] and (
                (r.get("error_key") and r["error_key"] in missing)
                or (r.get("error_type") == "FileNotFoundError" and any(
                    v.get("kind") == "file" and (fpath is None or ga.same_path(v["source"], fpath)) for v in viol)))
            if cause:
                if len(seen) <= a.max_prefix:
                    receipt = {k: res.get(k) for k in ("location", "kind", "source", "missing", "reason",
                                                         "available", "found_in", "near_matches")}
                    t = tasks[tid]
                    cands.append({
                        "episode": hashlib.sha1(f"{fn}|{tid}|{tl}|{r['cell_index']}".encode()).hexdigest()[:12],
                        "submission_file": fn, "submitter": man[fn]["submitter"],
                        "trace_format": man[fn].get("trace_format"), "stateful": rp.stateful[fn][0],
                        "task_id": tid, "trace_line": tl, "level": t.get("level"),
                        "question": t["question"], "guidelines": t.get("guidelines"), "answer": t.get("answer") or None,
                        "prefix": [{"cell_index": c["cell_index"], "code": c["code"], "failed": c["failed"],
                                    "error_type": c.get("error_type"), "output": (c.get("output") or "")[:3000]}
                                   for c in seen],
                        "blocked": {"cell_index": r["cell_index"], "code": r["code"],
                                    "error_type": r.get("error_type"), "error_key": r.get("error_key"),
                                    "output": (r.get("output") or "")[:3000]},
                        "receipt": receipt, "context_paths": paths,
                        "sandbox_paths": sandbox_paths(layouts[fn], gone)})
                break
            seen.append(r)
    print(f"candidate episodes: {len(cands)} from {len({c['submitter'] for c in cands})} submitters")
    rnd = random.Random(a.seed)
    rnd.shuffle(cands)
    cap = max(1, int(a.max_share * a.n))
    per, pick, used = collections.Counter(), [], set()
    for ex in a.exclude:
        for line in open(ex):
            e = json.loads(line)
            used.add((e["submission_file"], e["task_id"]))
    for c in cands:
        k = (c["submission_file"], c["task_id"])
        if per[c["submitter"]] >= cap or k in used:
            continue
        pick.append(c)
        per[c["submitter"]] += 1
        used.add(k)
        if len(pick) == a.n:
            break
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for c in pick:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(pick)} episodes to {a.out}; by submitter {dict(per)}; "
          f"by kind {dict(collections.Counter(c['receipt']['kind'] for c in pick))}")
    with open(a.out + ".candidates.json", "w") as f:
        json.dump({"n_candidates": len(cands),
                   "by_submitter": collections.Counter(c["submitter"] for c in cands),
                   "by_kind": collections.Counter(c["receipt"]["kind"] for c in cands),
                   "seed": a.seed, "n": a.n, "max_share": a.max_share, "max_prefix": a.max_prefix}, f, indent=1)


if __name__ == "__main__":
    main()
