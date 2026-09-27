#!/usr/bin/env python3
"""Held-out test: frozen gate v6 on the cells that the agent wrote on KramaBench.

The run (kb_run.py) executed every cell and recorded the workspace before each
cell, so the replay knows the files exactly:
  * input/ is the task's data lake (read-only): a path exists if the file exists;
  * the work directory and /tmp: a path exists if the recorded listing has it;
  * relative paths resolve against /work until a cell calls os.chdir, after which
    they are unknown; hidden files and other absolute paths are unknown.
Schema (fixed in the plan): first row of each CSV by csv.reader (UTF-8), union of keys
of JSON files that hold a list of objects; base names that occur twice are left out.

Usage: python3 kramabench/kb_gate.py --gate-dir dabstep-gate --kb /home/claude/krama/src \
    --results runs/kb/results_gpt-oss.jsonl --out runs/kb/gate_gpt-oss.jsonl [--show 40]
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dataagentbench-replay"))
import dab_lib as dl  # noqa: E402


def domain_schema(lake):
    by_base = collections.defaultdict(list)
    for d, _, files in os.walk(lake):
        for f in files:
            by_base[f].append(os.path.join(d, f))
    schema, ambiguous, skipped = {}, [], []
    for base, paths in by_base.items():
        if not base.endswith((".csv", ".json")):
            continue
        if len(paths) > 1:
            ambiguous.append(base)
            continue
        p = paths[0]
        cols_of = getattr(sys.modules.get("gate"), "csv_columns", None)   # gate v7, rule 1
        try:
            if base.endswith(".csv") and cols_of is not None:
                cols = cols_of(p)
                if cols is None:
                    skipped.append(base)
                else:
                    schema[base] = cols
            elif base.endswith(".csv"):
                with open(p, newline="", encoding="utf-8") as fh:
                    row = next(csv.reader(fh), None)
                if row is not None:
                    schema[base] = set(row)
            else:
                with open(p, encoding="utf-8") as fh:
                    recs = json.load(fh)
                if isinstance(recs, list) and recs and all(isinstance(r, dict) for r in recs):
                    schema[base] = set().union(*recs)
        except (UnicodeDecodeError, OSError, csv.Error, ValueError):
            skipped.append(base)
    return schema, ambiguous, skipped


def make_exists(lake, workspace, cwd_known):
    ws = set(workspace or [])
    dirs = set()
    for p in ws:
        d = os.path.dirname(p)
        while d and d not in dirs and d != "/":
            dirs.add(d)
            d = os.path.dirname(d)

    def exists(path):
        path = str(path)
        if path.startswith("~"):
            return None
        if not os.path.isabs(path):
            if not cwd_known:
                return None
            path = os.path.join("/work", path)
        n = os.path.normpath(path)
        if any(part.startswith(".") and part not in (".", "..") for part in n.split("/")):
            return None
        if n == "/work/input" or n.startswith("/work/input/"):
            return os.path.exists(os.path.join(lake, n[len("/work/input"):].lstrip("/")))
        if n in ("/work", "/tmp"):
            return True
        if n.startswith(("/work/", "/tmp/")):
            if workspace is None:
                return None
            return n in ws or n in dirs
        return None
    return exists


def error_names(err):
    keys, path = set(), None
    for line in (err or "").splitlines():
        m = re.match(r"^(?:\w+\.)*KeyError:\s?(.*)$", line.strip())
        if m:
            keys |= set(dl.key_list(m.group(1)))
        m = re.search(r"No such file or directory: '([^']+)'", line)
        if m:
            path = m.group(1)
    return keys, path


def same_path(a, b):
    a = os.path.normpath(a if os.path.isabs(a) else os.path.join("/work", a))
    b = os.path.normpath(b if os.path.isabs(b) else os.path.join("/work", b))
    return a == b


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--kb", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.gate_dir))
    from gate import GateState, decide  # noqa: E402
    schemas = {}
    C = collections.Counter()
    by_kind = collections.Counter()
    unknown_reasons = collections.Counter()
    ms, audit, other_blocks, misses = [], [], [], []
    with open(a.out, "w") as out:
        for line in open(a.results):
            r = json.loads(line)
            dom = r["domain"]
            lake = os.path.join(a.kb, "data", dom, "input")
            if dom not in schemas:
                schemas[dom] = domain_schema(lake)
                s, amb, sk = schemas[dom]
                print(f"schema {dom}: {len(s)} files; ambiguous base names {len(amb)}; unreadable {len(sk)}")
            schema = schemas[dom][0]
            st, cwd_known = GateState(), True
            C["tasks"] += 1
            for c in r.get("cells") or []:
                if c.get("no_code"):
                    continue
                code = c["code"]
                if "chdir" in code:
                    cwd_known = False            # this cell may change the working directory
                ex = make_exists(lake, c.get("workspace") if c.get("workspace_known") and not c.get("workspace_cut")
                                 else None, cwd_known)
                t0 = time.perf_counter()
                try:
                    res = decide(code, schema, st, ex)
                    o = res["outcome"]
                except Exception as e:  # a gate crash is a defect, not a block
                    res, o = {"outcome": "gate_error", "reason": f"{type(e).__name__}: {e}"}, "gate_error"
                ms.append((time.perf_counter() - t0) * 1000)
                ok = bool(c.get("ok"))
                C["cells"] += 1
                C["o:" + o] += 1
                if o == "unknown":
                    unknown_reasons[re.sub(r"line \d+: ", "", str(res.get("reason")))[:70]] += 1
                rec = {"task": r["task"], "domain": dom, "step": c["step"], "ok": ok, "error_type": c.get("error_type"),
                       "outcome": o, "reason": str(res.get("reason"))[:300], "kind": res.get("kind"),
                       "missing": res.get("missing"), "source": res.get("source")}
                if ok:
                    C["ok"] += 1
                    if o == "block":
                        C["ok_block"] += 1
                        audit.append((r["task"], c["step"], res.get("reason"), code, c.get("stdout")))
                else:
                    C["failed"] += 1
                    keys, path = error_names(c.get("error"))
                    viol = res.get("violations") or []
                    missing = {v.get("missing") for v in viol}
                    cause = o == "block" and bool((keys & missing) or (path is not None and any(
                        v.get("kind") == "file" and same_path(v["source"], path) for v in viol)))
                    rec["cause_match"] = cause
                    et = c.get("error_type")
                    if et in ("KeyError", "FileNotFoundError"):
                        C["data_fail"] += 1
                        C["data_fail:" + et] += 1
                        C["caught"] += cause
                        C["caught:" + et] += cause
                        if cause:
                            by_kind[res.get("kind")] += 1
                        else:
                            misses.append((r["task"], c["step"], et, o, str(res.get("reason"))[:160],
                                           (c.get("error") or "").strip().splitlines()[-1][:160] if c.get("error") else ""))
                    elif o == "block":
                        C["block_on_other_failure"] += 1
                        other_blocks.append((r["task"], c["step"], et, res.get("reason")))
                out.write(json.dumps(rec) + "\n")
                st.commit(failed=not ok)
    n = max(1, C["cells"])
    print(f"tasks {C['tasks']}; model cells {C['cells']}: run {C['o:run'] / n:.1%}, unknown {C['o:unknown'] / n:.1%}, "
          f"block {C['o:block'] / n:.1%}, gate errors {C['o:gate_error']}")
    print(f"failed cells {C['failed']}; KeyError {C['data_fail:KeyError']}, FileNotFoundError "
          f"{C['data_fail:FileNotFoundError']}")
    k, m = C["caught"], C["data_fail"]
    w = wilson(k, m)
    print(f"caught with a cause match: {k} / {m} ({k / max(1, m):.1%}, 95% CI {w[0]:.1%}-{w[1]:.1%}); "
          f"KeyError {C['caught:KeyError']}, FileNotFoundError {C['caught:FileNotFoundError']}; by kind {dict(by_kind)}")
    print(f"blocks on other failures: {C['block_on_other_failure']}")
    print(f"cells without an error: {C['ok']}; blocked {C['ok_block']}")
    if ms:
        ms.sort()
        print(f"latency per cell: median {statistics.median(ms):.2f} ms, p95 {ms[int(0.95 * len(ms))]:.2f} ms")
    print("top unknown reasons:")
    for rsn, cnt in unknown_reasons.most_common(12):
        print(f"  {cnt:5d} {rsn}")
    for x in audit[: a.show]:
        print("=== BLOCK ON ERROR-FREE CELL", x[0], "step", x[1], "|", str(x[2])[:200])
        print(x[3][:1500])
        print("--- output:", (x[4] or "")[:400])
    for x in other_blocks[: a.show]:
        print("=== BLOCK ON OTHER FAILURE", x)
    print("misses (first 30):")
    for x in misses[:30]:
        print("  ", x)


if __name__ == "__main__":
    main()
