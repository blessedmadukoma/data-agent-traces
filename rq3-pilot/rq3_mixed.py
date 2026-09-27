#!/usr/bin/env python3
"""A4, post hoc: RQ3 saving with a random intercept per
submitter. Same pairs, exclusions and outcome as rq3_full_report.py.

Model: D_e = mu + u_submitter(e) + eps_e, with D_e the episode-level mean of
d = y(traceback) - y(receipt). Fitted by REML (statsmodels MixedLM). mu is the
submitter-balanced estimate; its Wald 95% CI is reported, with the between-submitter
standard deviation.

Usage: python3 rq3-pilot/rq3_mixed.py --episodes runs/rq3-full/episodes.jsonl \
    --results runs/rq3-full/results_gpt-oss.jsonl
"""
import argparse
import collections
import json
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq3_full_report as fr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    import numpy as np
    import statsmodels.formula.api as smf
    import pandas as pd
    eps = {e["episode"]: e for e in map(json.loads, open(a.episodes))}
    by = collections.defaultdict(dict)
    for r in map(json.loads, open(a.results)):
        by[(r["episode"], r["seed"])][r["arm"]] = r
    acc = collections.defaultdict(list)
    for (ep, seed), arms in by.items():
        t, rc = arms.get("traceback"), arms.get("receipt")
        if not t or not rc or t.get("status") != "done" or rc.get("status") != "done":
            continue
        if not t.get("blocked_names_missing"):
            continue
        acc[ep].append(fr.y(t) - fr.y(rc))
    df = pd.DataFrame([{"episode": ep, "D": statistics.mean(v), "submitter": eps[ep]["submitter"],
                        "kind": eps[ep]["receipt"]["kind"]} for ep, v in acc.items()])
    m = smf.mixedlm("D ~ 1", df, groups=df["submitter"]).fit(reml=True)
    mu, se = float(m.fe_params["Intercept"]), float(m.bse_fe["Intercept"])
    sd_u = float(np.sqrt(m.cov_re.iloc[0, 0])) if m.cov_re.shape[0] else float("nan")
    out = {"episodes": len(df), "submitters": int(df["submitter"].nunique()), "mu": mu,
           "ci": [mu - 1.96 * se, mu + 1.96 * se], "sd_submitter": sd_u, "sd_residual": float(np.sqrt(m.scale)),
           "raw_mean": float(df["D"].mean())}
    mk = smf.mixedlm("D ~ C(kind, Treatment('column'))", df, groups=df["submitter"]).fit(reml=True)
    out["kind_effects"] = {k: [float(v), float(mk.bse_fe[k])] for k, v in mk.fe_params.items()}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
