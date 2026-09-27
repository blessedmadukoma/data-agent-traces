#!/usr/bin/env python3
"""DataAgentBench logs -> one JSON line per run, with every tool call classified.

Usage:
  python3 extract_dab_cells.py --logs dataagentbench/logs-pr6 --model gpt-5-mini --out runs/dab/cells_gpt-5-mini.jsonl

Per run: model, dataset, query, run, number of unreadable log lines, the
tool calls in order, and the stored result variables. Per call: tool,
db_type, engine, failed, error text, error group (dab_lib.classify), the
SQL or MongoDB query, and for execute_python the code, the wrapper
reproduction (own, status, decoded code), whether the failure is
harness-caused, and the names of the variables that exist before the call.

Variables: every successful call stores its result as var_<id>. The log
does not record the id, so names come from the env of later Python calls:
the k-th successful call owns the k-th env key. Each run records whether
the env of every Python call has as many keys as there were successful
calls before it, and whether inline values and file names agree with the
producing call (alignment check).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dab_lib as dl  # noqa: E402


def build_run(model, dataset, query, run, path):
    raw, bad = dl.read_calls(path)
    calls, succ = [], []
    longest = []
    env_ok = True
    for i, d in enumerate(raw):
        tool = d.get("tool_name")
        res = d.get("result") or {}
        ok = bool(res.get("success"))
        prev = res.get("preview", "")
        va = d.get("val_args") if isinstance(d.get("val_args"), dict) else {}
        args = d.get("args") if isinstance(d.get("args"), dict) else {}
        c = {"i": i, "tool": tool, "failed": not ok}
        if tool == "query_db":
            c["db_type"] = va.get("db_type")
            c["query"] = args.get("query")
            if c["db_type"] == "mongo":
                ea = va.get("exec_args") or {}
                try:
                    q = json.loads(args.get("query") or "")
                except ValueError:
                    q = None
                c["mongo"] = {"collection": ea.get("collection"), "filter": ea.get("filter"),
                              "projection": ea.get("projection"), "limit": ea.get("limit"),
                              "limit_given": isinstance(q, dict) and "limit" in q}
        if not ok:
            err = dl.result_text(prev)
            c["error"] = err[:4000]
            c["group"], c["detail"] = dl.classify(tool, c.get("db_type"), err)
            if tool == "query_db":
                c["engine"] = dl.engine(err) or c.get("db_type")
                c["db_hint"] = bool(dl.SQL_HINT.search(err))
        else:
            succ.append(i)
            if tool == "query_db":
                c["n_records"] = None
                v = dl.result_value(prev)
                if isinstance(v, list):
                    c["n_records"] = len(v)
                c["preview_cut"] = len(prev) >= dl.PREVIEW and v is None
                k = dl.first_record_keys(prev)
                c["keys"] = sorted(k) if k is not None else None
        if tool == "execute_python":
            code = args.get("code") or ""
            own, status, dec = dl.wrapper(code)
            c.update(code=code, own_compiles=own, wrapper=status,
                     decoded=dec if status == "changed" else None)
            c["harness_syntax"] = (not ok) and dl.harness_syntax(own, status, c.get("error"))
            env = va.get("env") if isinstance(va.get("env"), dict) else {}
            names = list(env)
            c["env"] = names
            if len(names) != len(succ) - (1 if ok else 0) - 0:
                # env is built before the call; succ already includes this call if it succeeded
                env_ok = False
            if len(names) > len(longest):
                longest = names
            c["_env_values"] = env
        calls.append(c)

    # names of successful calls, from the longest env
    vars_ = {}
    align = {"env_counts": env_ok, "value_mismatch": 0, "checked": 0}
    values = {}
    for c in calls:
        for n, v in (c.pop("_env_values", None) or {}).items():
            values.setdefault(n, v)
    for k, name in enumerate(longest):
        if k >= len(succ):
            align["env_counts"] = False
            break
        pc = calls[succ[k]]
        v = values.get(name)
        ent = {"producer": pc["i"], "tool": pc["tool"]}
        if isinstance(v, str) and v.startswith("file_storage/") and v.endswith(".json"):
            ent["path"] = v
            ent["keys"] = pc.get("keys") if pc["tool"] == "query_db" else None
            align["checked"] += 1
            if os.path.basename(v) != name.removeprefix("var_") + ".json":
                align["value_mismatch"] += 1
        else:
            ent["value"] = v
            prev = (raw[pc["i"]].get("result") or {}).get("preview", "")
            if isinstance(prev, str) and len(prev) < dl.PREVIEW:
                align["checked"] += 1
                if json.dumps(v) != prev:
                    align["value_mismatch"] += 1
        vars_[name] = ent
    return {"trace": f"{model}/{dataset}/{query}/{run}", "model": model, "dataset": dataset,
            "query": query, "run": run, "bad_lines": bad, "calls": calls, "vars": vars_, "align": align}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--model", required=True, help="model name as in dab_lib.run_files, e.g. gpt-5-mini")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = 0
    with open(a.out, "w") as out:
        for model, ds, q, run, path in sorted(dl.run_files(a.logs)):
            if model != a.model:
                continue
            out.write(json.dumps(build_run(model, ds, q, run, path)) + "\n")
            n += 1
    print(f"{a.model}: {n} runs -> {a.out}")


if __name__ == "__main__":
    main()
