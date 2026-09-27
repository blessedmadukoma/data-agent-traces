#!/usr/bin/env python3
"""Why does the gate miss a data-reference failure?

For each failed cell with a data-reference error that the gate does not
block with a cause match, find how the missing name is used and where the
object comes from. Heuristic, per cell; the result ranks the next gate
extensions.

  python3 miss_analysis.py --manifest "$RUN/manifest.jsonl" --cells "$RUN"/parsed_*_cells.jsonl \
      --pred "$RUN/gate_predictions.jsonl" [--examples 3]
"""
import argparse
import ast
import collections
import json
import warnings

DATA = {"KeyError", "KeyError(wrapped)", "index-error(wrapped)",
        "FileNotFoundError", "ValueError:usecols"}
READERS = {"read_csv", "read_json", "read_excel", "read_parquet"}
JSONLOAD = {"load", "loads", "read_json_metadata_file", "read_csv_metadata_file", "safe_read_file"}
COLUMN_ARGS = {"groupby", "sort_values", "merge", "drop", "set_index", "pivot", "pivot_table",
               "value_counts", "drop_duplicates", "dropna", "agg", "rename", "explode", "melt",
               "nlargest", "nsmallest", "get", "filter", "join"}


def parse(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def value_kind(v):
    """Kind of object produced by an assigned expression."""
    if isinstance(v, ast.Call):
        f = v.func
        if isinstance(f, ast.Attribute) and f.attr in READERS:
            return "frame from a reader"
        if isinstance(f, (ast.Attribute, ast.Name)) and (getattr(f, "attr", None) in JSONLOAD
                                                         or getattr(f, "id", None) in JSONLOAD):
            return "JSON or text from a file"
        if isinstance(f, ast.Attribute) and f.attr in {"DataFrame", "concat", "merge", "json_normalize"}:
            return "frame built in code"
        if isinstance(f, ast.Attribute):
            return "result of a method call"
        return "result of a function call"
    if isinstance(v, (ast.Subscript, ast.Attribute)):
        return "subscript or attribute of another object"
    if isinstance(v, (ast.Dict, ast.List, ast.ListComp, ast.DictComp)):
        return "literal or comprehension"
    return "other"


def origin(name, trees):
    """Last binding of `name` in the cells so far (latest cell first)."""
    for t in reversed(trees):
        found = None
        for n in ast.walk(t):
            if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in n.targets):
                found = value_kind(n.value)
            elif isinstance(n, (ast.For, ast.comprehension)) and any(
                    isinstance(x, ast.Name) and x.id == name for x in ast.walk(n.target)):
                it = n.iter
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in {"iterrows", "itertuples"}:
                    found = "row of a DataFrame loop"
                else:
                    found = "loop variable"
            elif isinstance(n, ast.arg) and n.arg == name:
                found = "function argument"
        if found:
            return found
    return "not bound in the trace"


def classify(r, trees, pred):
    et, key = r.get("error_type"), r.get("error_key")
    if et == "FileNotFoundError":
        return "missing file", None
    if et == "ValueError:usecols":
        return "usecols", None
    if (pred.get("reason") or "").startswith("invalid Python"):
        return "invalid Python", None
    t = trees[-1]
    if t is None:
        return "invalid Python", None
    if not key:
        return "error text without a key", None
    subs = [n for n in ast.walk(t) if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
            and n.slice.value == key]
    if subs:
        base = subs[0].value
        if isinstance(base, ast.Name):
            return "x['key'], x = " + origin(base.id, [x for x in trees if x is not None]), base.id
        return "expr['key'] (derived expression)", None
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in COLUMN_ARGS:
            lits = [c.value for c in ast.walk(n) if isinstance(c, ast.Constant)]
            if key in lits:
                return f"key as a method argument (.{n.func.attr})", None
    for n in ast.walk(t):
        if isinstance(n, ast.Constant) and n.value == key:
            return "key in another position", None
    return "key not written in this cell", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args()
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    pred = {}
    for line in open(a.pred):
        p = json.loads(line)
        pred[(p["submission_file"], str(p["task_id"]), p.get("trace_line"), p["cell_index"])] = p
    traces = collections.defaultdict(list)
    for path in a.cells:
        for line in open(path):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if m and m["included"]:
                traces[(r["submission_file"], str(r["task_id"]), r.get("trace_line"))].append(r)
    c, per, ex = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(list)
    total = caught = 0
    for (fn, tid, tl), cells in traces.items():
        cells.sort(key=lambda r: r["cell_index"])
        trees = []
        for r in cells:
            trees.append(parse(r["code"] or ""))
            if not (r["failed"] and r.get("error_type") in DATA):
                continue
            total += 1
            p = pred[(fn, tid, tl, r["cell_index"])]
            if p["cause_match"]:
                caught += 1
                continue
            k, _ = classify(r, trees, p)
            k = f"{k} [gate: {p['outcome']}]" if p["outcome"] != "unknown" else k
            c[k] += 1
            per[man[fn]["submitter"]][k] += 1
            if len(ex[k]) < a.examples:
                ex[k].append((fn, tid, r["cell_index"], r.get("error_key")))
    print(f"data-reference failures: {total}; caught with a cause match: {caught}; missed: {total - caught}")
    for k, n in c.most_common():
        subs = len([s for s in per if per[s][k]])
        print(f"{n:6d} {n / (total - caught):6.1%}  {k}   ({subs} submitters)")
        for e in ex[k]:
            print("         e.g.", e)


if __name__ == "__main__":
    main()
