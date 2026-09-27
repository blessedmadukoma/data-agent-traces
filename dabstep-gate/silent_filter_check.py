#!/usr/bin/env python3
"""Find silent program-data violations: literal filters that can never match.

A cell such as  df[df['card_scheme'] == 'Visa']  runs without error but
returns no rows, because 'Visa' is not a value of card_scheme in any
DABstep source. The agent receives no traceback, so it cannot see the
problem. This is an exact precondition: the literal must be in the
column's value set.

For each parsed cell, the script finds equality and isin filters with
string literals on low-cardinality text columns, checks the literals
against the value sets of the context files, and marks a trace as
exposed when a cell that ran without error contains an impossible
literal. It then estimates the association with a correct final answer
(public task_scores), with task and submission fixed effects.

Usage:
  python3 silent_filter_check.py --data <dabstep-data> --manifest manifest.jsonl --cells parsed_*_cells.jsonl

Cell files must have: submission_file, task_id, code, failed.
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import warnings
warnings.filterwarnings("ignore")

TEXT_COLS = ["merchant", "card_scheme", "aci", "device_type", "shopper_interaction", "account_type",
             "ip_country", "issuing_country", "acquirer_country", "acquirer", "capture_delay",
             "monthly_volume", "monthly_fraud_level", "country_code", "description"]
LIT = r"['\"]([^'\"\n]{1,80})['\"]"


def value_sets(data):
    c = os.path.join(data, "data/context/")
    vals = collections.defaultdict(set)
    with open(c + "payments.csv") as f:
        for row in csv.DictReader(f):
            for k in ("merchant", "card_scheme", "aci", "device_type", "shopper_interaction",
                      "ip_country", "issuing_country", "acquirer_country"):
                vals[k].add(row[k])
    for fn in ("merchant_data.json", "fees.json"):
        for m in json.load(open(c + fn)):
            for k, v in m.items():
                for x in (v if isinstance(v, list) else [v]):
                    if x is not None:
                        vals[k].add(str(x))
    for fn in ("acquirer_countries.csv", "merchant_category_codes.csv"):
        for row in csv.DictReader(open(c + fn)):
            for k, v in row.items():
                if k:
                    vals[k].add(v)
    return {k: vals[k] for k in TEXT_COLS if k in vals}


def filters(code):
    """Yield (column, literal) pairs from equality, isin and query filters."""
    cols = "|".join(TEXT_COLS)
    for m in re.finditer(r"\[\s*['\"](%s)['\"]\s*\]\s*==\s*%s" % (cols, LIT), code):
        yield m.group(1), m.group(2)
    for m in re.finditer(r"\.(%s)\s*==\s*%s" % (cols, LIT), code):
        yield m.group(1), m.group(2)
    for m in re.finditer(r"\[\s*['\"](%s)['\"]\s*\]\.isin\(\s*\[([^\]]*)\]" % cols, code):
        for lit in re.findall(LIT, m.group(2)):
            yield m.group(1), lit
    for m in re.finditer(r"query\(\s*(['\"])(.*?)\1\s*\)", code):
        for q in re.finditer(r"\b(%s)\s*==\s*\\?['\"`]([^'\"`]{1,80})\\?['\"`]" % cols, m.group(2)):
            yield q.group(1), q.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--manifest", help="skip files that the manifest excludes")
    a = ap.parse_args()
    if a.manifest:
        a.exclude += [m["submission_file"]
                      for m in map(json.loads, open(a.manifest)) if not m["included"]]
    vs = value_sets(a.data)
    trace = collections.defaultdict(
        lambda: dict(silent=0, any_filter=0, data_err=0))
    lits = collections.Counter()
    bad = collections.Counter()
    badcol = collections.Counter()
    for path in a.cells:
        for line in open(path):
            r = json.loads(line)
            if r["submission_file"] in a.exclude:
                continue
            key = (r["submission_file"], str(r["task_id"]))
            t = trace[key]
            if r.get("failed") and r.get("error_type") in ("KeyError", "KeyError(wrapped)", "index-error(wrapped)",
                                                           "FileNotFoundError", "ValueError:usecols"):
                t["data_err"] = 1
            for col, lit in filters(r.get("code") or ""):
                lits["all"] += 1
                t["any_filter"] = 1
                if lit not in vs[col]:
                    lits["impossible"] += 1
                    badcol[col] += 1
                    bad[(col, lit)] += 1
                    if not r.get("failed"):
                        t["silent"] = 1
    print(f"literal filters: {lits['all']}; impossible literals: {lits['impossible']} "
          f"({lits['impossible'] / max(lits['all'], 1):.1%})")
    print("impossible literals by column:", badcol.most_common(8))
    print("most frequent impossible literals:", bad.most_common(15))

    rows = []
    for (sub, task), t in trace.items():
        sp = os.path.join(a.data, "data/task_scores", sub)
        rows.append(dict(sub=sub, task=task, **t))
    scores = {}
    for sub in {r["sub"] for r in rows}:
        sp = os.path.join(a.data, "data/task_scores", sub)
        if os.path.exists(sp):
            for l in open(sp):
                if l.strip():
                    x = json.loads(l)
                    scores[(sub, str(x["task_id"]))] = bool(x["score"])
    import pandas as pd
    import statsmodels.api as sm
    df = pd.DataFrame(rows)
    df["y"] = [scores.get((s, t)) for s, t in zip(df["sub"], df["task"])]
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)
    org = {s: (json.loads(open(os.path.join(a.data, "data/submissions", s)).readline()).get("organisation") or "")
           for s in df["sub"].unique()}
    df["submitter"] = df["sub"].map(lambda s: org[s].split(
        "user")[-1].strip() if "user" in org[s] else s)
    print(
        f"\ntraces: {len(df)}; submissions: {df['sub'].nunique()}; submitters: {df['submitter'].nunique()}")
    print(
        f"traces with a silent impossible filter: {df['silent'].sum()} ({df['silent'].mean():.1%})")
    for s, g in df.groupby("submitter"):
        print(f"  {s:15s} silent {g['silent'].mean():5.1%}  acc(silent) {g[g.silent == 1]['y'].mean():.2f}  "
              f"acc(no silent) {g[g.silent == 0]['y'].mean():.2f}  n={len(g)}")
    d = df[df.groupby("task")["y"].transform(
        "nunique") > 1].reset_index(drop=True)
    X = pd.get_dummies(d[["sub", "task"]].astype(str),
                       drop_first=True).astype(float)
    for cols in (["silent"], ["silent", "data_err"]):
        Z = X.copy()
        for i, c in enumerate(cols):
            Z.insert(i, c, d[c].astype(float))
        Z = sm.add_constant(Z)
        for cl in ("sub", "submitter"):
            m = sm.Logit(d["y"], Z).fit(disp=0, method="newton", maxiter=100, cov_type="cluster",
                                        cov_kwds={"groups": pd.factorize(d[cl])[0]})
            out = "; ".join(f"{c}: OR {math.exp(m.params[c]):.2f} [{math.exp(m.params[c]-1.96*m.bse[c]):.2f}, "
                            f"{math.exp(m.params[c]+1.96*m.bse[c]):.2f}]" for c in cols)
            print(f"FE task+submission, clustered by {cl}, n={len(d)}: {out}")


if __name__ == "__main__":
    main()
