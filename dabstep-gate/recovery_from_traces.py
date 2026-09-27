#!/usr/bin/env python3
"""Recovery after errors, from recorded traces.

For each failed cell: how many failing cells follow before a cell runs
without error ("retry chain"), whether the next cell runs, whether the
next cell repeats the same missing key, and whether the trace never
recovers. Results are grouped by error group (data-reference, executor,
other) and by submitter. The script then estimates the association
between a data-reference error in a trace and a correct final answer,
with task and submission fixed effects, clustered by submitter, plus a
leave-one-submitter-out check.

Usage:
  python3 recovery_from_traces.py --data $DATA --manifest $RUN/manifest.jsonl \
      --cells $RUN/parsed_*_cells.jsonl --out $RUN/recovery.csv
"""
import argparse
import collections
import csv
import json
import math
import os
import statistics
import warnings
warnings.filterwarnings("ignore")

DATA = {"KeyError", "KeyError(wrapped)", "index-error(wrapped)",
        "FileNotFoundError", "ValueError:usecols"}
EXE = {"executor:forbidden", "executor:budget", "ModuleNotFoundError"}


def group(e):
    return "data-reference" if e in DATA else "executor" if e in EXE else "other"


def fit(d, col, cl="submitter"):
    import pandas as pd
    import statsmodels.api as sm
    d = d[d.groupby("task")["y"].transform(
        "nunique") > 1].reset_index(drop=True)
    X = pd.get_dummies(d[["sub", "task"]].astype(str),
                       drop_first=True).astype(float)
    X.insert(0, col, d[col].astype(float))
    X = sm.add_constant(X)
    m = sm.Logit(d["y"], X).fit(disp=0, method="newton", maxiter=100, cov_type="cluster",
                                cov_kwds={"groups": pd.factorize(d[cl])[0]})
    b, se = m.params[col], m.bse[col]
    return math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    traces = collections.defaultdict(list)
    for p in a.cells:
        for line in open(p):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if m and m["included"]:
                traces[(r["submission_file"], str(r["task_id"]), r.get("trace_line"))].append(r)
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    for (fn, _, _), cs in traces.items():
        cs.sort(key=lambda r: r["cell_index"])
        s = man[fn]["submitter"]
        for i, r in enumerate(cs):
            if not r["failed"]:
                continue
            g = group(r.get("error_type"))
            j = i + 1
            while j < len(cs) and cs[j]["failed"]:
                j += 1
            st = stats[(s, g)]
            st["next_ok"].append(i + 1 < len(cs) and not cs[i + 1]["failed"])
            st["never"].append(j >= len(cs))
            if j < len(cs):
                st["chain"].append(j - i - 1)
            if g == "data-reference" and r.get("error_key") and i + 1 < len(cs):
                st["same_key"].append(
                    cs[i + 1].get("error_key") == r["error_key"])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["submitter", "group", "errors", "next_cell_ok",
                   "never_recovers", "mean_chain", "median_chain", "same_key_next"])
        print(
            f"{'submitter':15s} {'group':15s} errors  next_ok  never  mean_chain  same_key")
        for (s, g), st in sorted(stats.items()):
            n = len(st["next_ok"])
            row = [s, g, n, sum(st["next_ok"]) / n, sum(st["never"]) / n,
                   statistics.mean(
                       st["chain"]) if st["chain"] else float("nan"),
                   statistics.median(
                       st["chain"]) if st["chain"] else float("nan"),
                   (sum(st["same_key"]) / len(st["same_key"])) if st["same_key"] else float("nan")]
            w.writerow(row)
            print(
                f"{s:15s} {g:15s} {n:6d} {row[3]:7.0%} {row[4]:6.0%} {row[5]:10.2f} {row[7]:9.0%}")
    rows = []
    for (fn, tid, _), cs in traces.items():
        sp = os.path.join(a.data, "data/task_scores", fn)
        rows.append(dict(sub=fn, task=tid, submitter=man[fn]["submitter"],
                         d=int(any(r["failed"] and r.get("error_type") in DATA for r in cs))))
    sc = {}
    for fn in {r["sub"] for r in rows}:
        for l in open(os.path.join(a.data, "data/task_scores", fn)):
            if l.strip():
                x = json.loads(l)
                sc[(fn, str(x["task_id"]))] = int(bool(x["score"]))
    import pandas as pd
    df = pd.DataFrame(rows)
    df["y"] = [sc.get((s, t)) for s, t in zip(df["sub"], df["task"])]
    df = df.dropna(subset=["y"])
    o, lo, hi, n = fit(df, "d")
    print(f"\nOR(correct | data-reference error in trace), task + submission FE, clustered by submitter: "
          f"{o:.2f} [{lo:.2f}, {hi:.2f}], n={n}, submitters={df['submitter'].nunique()}")
    for s in sorted(df["submitter"].unique()):
        o, lo, hi, n = fit(df[df["submitter"] != s].reset_index(
            drop=True), "d", cl="sub")
        print(f"  without {s:15s} {o:.2f} [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
