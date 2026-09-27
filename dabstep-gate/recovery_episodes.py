#!/usr/bin/env python3
"""Recovery episodes after a data-reference error, from recorded traces.

An episode starts at a failed cell with a data-reference error (missing
column, key or file) whose previous cell did not fail. For each episode:

  k_success  cells until the first later cell without an error (1 = next cell)
  inspect    that first successful cell only looks at the data (prints
             .columns, .keys(), .dtypes, .info(), .head(), ...), which is what a
             receipt already tells the agent (heuristic, see is_inspection)
  k_fix      cells until the first later cell without an error that is not
             inspection-only
  repeat     a failing cell in between fails on the same missing name

A receipt can save at most k_fix - 1 cells per episode (the agent's next cell
is the fix). This is an upper bound: it assumes the receipt always leads to
the fix. If most episodes already have k_fix = 1, RQ3 has little room.

  python3 recovery_episodes.py --manifest "$RUN/manifest.jsonl" --cells "$RUN"/parsed_*_cells.jsonl \
      [--pred "$RUN/gate_predictions.jsonl"] [--examples 5] [--out "$RUN/recovery_episodes.csv"]

With --pred, results are also given for the episodes whose first failing
cell the gate blocks with a cause match (the cases where a receipt exists).
"""
import argparse
import ast
import collections
import csv
import json
import random
import statistics
import warnings

DATA = {"KeyError", "KeyError(wrapped)", "index-error(wrapped)",
        "FileNotFoundError", "ValueError:usecols"}
LOOK_ATTRS = {"columns", "dtypes", "shape", "index"}
LOOK_CALLS = {"keys", "info", "head", "tail", "sample", "describe", "unique", "listdir",
              "read_csv_metadata_file", "read_json_metadata_file", "glob"}
PLAIN = LOOK_ATTRS | LOOK_CALLS | {"tolist", "to_list", "list", "len", "type", "sorted", "set", "str"}
QUIET = {"print", "display", "len", "list", "type", "sorted", "set", "str", "repr"}


def _parse(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def _is_load(v):
    """A statement value that only loads data (reader, json.load, open().read())."""
    return isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr in {
        "read_csv", "read_json", "load", "loads", "read", "read_excel", "read_parquet"}


def _plain(v):
    """An expression that only reads structure: df.columns.tolist(), list(d.keys())."""
    calls = [n for n in ast.walk(v) if isinstance(n, ast.Call)]
    attrs = [n.attr for n in ast.walk(v) if isinstance(n, ast.Attribute)]
    return bool(calls or attrs) and all(
        (getattr(c.func, "attr", None) or getattr(c.func, "id", None)) in PLAIN for c in calls) \
        and all(a in PLAIN for a in attrs)


def gave_up(code):
    """final_answer with a constant, for example final_answer('Not Applicable')."""
    t = _parse(code)
    return t is not None and any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "final_answer"
                                 and n.args and isinstance(n.args[0], ast.Constant) for n in ast.walk(t)) \
        and not any(isinstance(n, ast.Call) and getattr(n.func, "id", None) != "final_answer"
                    for n in ast.walk(t))


def is_inspection(code):
    """True if the cell looks at the data's structure and does no other work.

    Work = a statement that is not an import, a print/display call, a data
    load, or a with/for/if whose body is only such statements."""
    t = _parse(code)
    if t is None:
        return False
    looks = any((isinstance(n, ast.Attribute) and n.attr in LOOK_ATTRS)
                or (isinstance(n, ast.Call) and (getattr(n.func, "attr", None) in LOOK_CALLS
                                                 or getattr(n.func, "id", None) in LOOK_CALLS))
                for n in ast.walk(t))
    if not looks:
        return False

    def quiet(s):
        if isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass)):
            return True
        if isinstance(s, ast.Expr):
            v = s.value
            if isinstance(v, ast.Constant):
                return True
            if isinstance(v, ast.Call) and getattr(v.func, "id", None) in QUIET:
                return True
            return isinstance(v, (ast.Name, ast.Attribute, ast.Subscript)) or _plain(v)
        if isinstance(s, ast.Assign):
            return _is_load(s.value) or isinstance(s.value, ast.Constant) or _plain(s.value)
        if isinstance(s, (ast.With, ast.For, ast.If, ast.Try)):
            body = s.body + getattr(s, "orelse", []) + [b for h in getattr(s, "handlers", []) for b in h.body]
            return all(quiet(b) for b in body)
        return False
    return all(quiet(s) for s in t.body)


def episodes(cells):
    out = []
    for i, r in enumerate(cells):
        if not (r["failed"] and r.get("error_type") in DATA):
            continue
        if i > 0 and cells[i - 1]["failed"]:
            continue
        e = dict(start=r, k_success=None, k_fix=None, inspect=None, repeat=False, gave_up=False)
        for j in range(i + 1, len(cells)):
            c = cells[j]
            if c["failed"]:
                if r.get("error_key") and c.get("error_key") == r["error_key"]:
                    e["repeat"] = True
                continue
            if e["k_success"] is None:
                e["k_success"] = j - i
                e["inspect"] = is_inspection(c["code"] or "")
                e["first_ok"] = c
            if not is_inspection(c["code"] or ""):
                e["k_fix"] = j - i
                e["gave_up"] = gave_up(c["code"] or "")
                break
        out.append(e)
    return out


def summary(eps):
    n = len(eps)
    rec = [e for e in eps if e["k_success"] is not None]
    fix = [e["k_fix"] for e in eps if e["k_fix"] is not None]
    if not n:
        return None
    return dict(episodes=n,
                never_ok=1 - len(rec) / n,
                next_ok=sum(e["k_success"] == 1 for e in rec) / n,
                median_k_success=statistics.median(e["k_success"] for e in rec) if rec else float("nan"),
                first_ok_inspection=sum(e["inspect"] for e in rec) / len(rec) if rec else float("nan"),
                fix_next=sum(k == 1 for k in fix) / n,
                mean_room=statistics.mean(k - 1 for k in fix) if fix else float("nan"),
                fix_2=sum(k == 2 for k in fix) / n, fix_3plus=sum(k >= 3 for k in fix) / n,
                gave_up=sum(e["gave_up"] for e in eps) / n,
                repeat=sum(e["repeat"] for e in eps) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--pred")
    ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    caught = set()
    if a.pred:
        for line in open(a.pred):
            p = json.loads(line)
            if p["cause_match"]:
                caught.add((p["submission_file"], str(p["task_id"]), p.get("trace_line"), p["cell_index"]))
    traces = collections.defaultdict(list)
    for path in a.cells:
        for line in open(path):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if m and m["included"]:
                traces[(r["submission_file"], str(r["task_id"]), r.get("trace_line"))].append(r)
    per = collections.defaultdict(list)
    for (fn, tid, tl), cells in traces.items():
        cells.sort(key=lambda r: r["cell_index"])
        for e in episodes(cells):
            e["caught"] = (fn, tid, tl, e["start"]["cell_index"]) in caught
            e["fn"], e["tid"] = fn, tid
            per[man[fn]["submitter"]].append(e)
    head = (f"{'submitter':15s} {'episodes':>8s} {'never ok':>8s} {'next ok':>8s} {'med k':>6s} "
            f"{'1st ok = look':>13s} {'fix next':>8s} {'fix 2':>6s} {'fix 3+':>6s} {'room':>6s} {'gave up':>7s}")
    rows = []

    def show(label, eps):
        s = summary(eps)
        if s:
            print(f"{label:15s} {s['episodes']:8d} {s['never_ok']:8.0%} {s['next_ok']:8.0%} {s['median_k_success']:6.1f} "
                  f"{s['first_ok_inspection']:13.0%} {s['fix_next']:8.0%} {s['fix_2']:6.0%} {s['fix_3plus']:6.0%} "
                  f"{s['mean_room']:6.2f} {s['gave_up']:7.0%}")
            rows.append(dict(group=label, **s))
    print("All data-reference episodes")
    print(head)
    alleps = []
    for s in sorted(per):
        show(s, per[s])
        alleps += per[s]
    show("ALL (pooled)", alleps)
    saved = sum(e["k_fix"] - 1 for e in alleps if e["k_fix"] is not None)
    n_cells = sum(len(cells) for cells in traces.values())
    print(f"ceiling: if every receipt made the next cell the fix, {saved} cells would be saved, "
          f"{saved / n_cells:.1%} of all {n_cells} cells")
    if a.pred:
        print("\nEpisodes whose first failing cell the gate blocks with a cause match (a receipt exists)")
        print(head)
        for s in sorted(per):
            show(s, [e for e in per[s] if e["caught"]])
        show("ALL (pooled)", [e for e in alleps if e["caught"]])
    print("\nroom = mean(k_fix - 1): the most cells per episode that a receipt could save.")
    print("gave up = the first working cell after the error is final_answer(<constant>).")
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if a.examples:
        random.seed(7)
        for label, want in (("inspection-only", True), ("not inspection", False)):
            pool = [e for e in alleps if e["inspect"] is want]
            print(f"\n##### {a.examples} random first successful cells classed {label}")
            for e in random.sample(pool, min(a.examples, len(pool))):
                print(f"--- {e['fn'][:50]} task {e['tid']} cell {e['first_ok']['cell_index']}")
                print((e["first_ok"]["code"] or "")[:500])


if __name__ == "__main__":
    main()
