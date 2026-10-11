#!/usr/bin/env python3
"""Numbers added in preprint v3.16 (October 2026).

Prints:
  1. The audit estimate with the first stratum weighted by error kind (post-stratification, not
     pre-registered): corrected count, share of the failed steps that the harness did not cause, and a
     95% bootstrap interval (10,000 samples, seed 20261011). The pre-registered estimate is in
     audit/results_FINAL.json.
  2. The exact one-sided 95% upper bound for 0 non-invented files among the 19 sampled DAB
     missing-file reads.
  3. A bootstrap over the 64 DAB tasks of the rank of gpt-5.1 by failed calls per run, with and without
     the harness-caused failures (10,000 samples, seed 20261011).
  4. Requests to the model per run in the DAB rerun, without the runs whose count was not recorded.

Usage (from the repository root, after `make dabstep` and `make dab`):
  uv run python tools/v3_16_numbers.py --dabstep-cells runs/dabstep/parsed_*_cells.jsonl \
      --dab-cells runs/dab/cells_*.jsonl
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SEED, B = 20261011, 10000
DABSTEP_POS = {"index-error(wrapped)", "FileNotFoundError", "KeyError", "ValueError:usecols", "KeyError(wrapped)"}
DAB_POS = {"sql:missing-column", "sql:missing-table", "mongo:missing-collection", "py:data-key", "py:data-file"}


def group_sizes(dabstep_cells, dab_cells):
    """Size of each error kind in the first stratum of the audit, from the parsed cells."""
    g = {"DABstep": collections.Counter(), "DAB": collections.Counter()}
    for path in dabstep_cells:
        for line in open(path):
            r = json.loads(line)
            if r.get("failed") and str(r.get("error_type")) in DABSTEP_POS:
                g["DABstep"][str(r.get("error_type"))] += 1
    for path in dab_cells:
        for line in open(path):
            for x in json.loads(line)["calls"]:
                if x["failed"] and not x.get("harness_syntax") and x["group"] in DAB_POS:
                    g["DAB"][x["group"]] += 1
    return g


def post_stratified(sizes):
    ids = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(ROOT, "audit/sample_ids.csv")))}
    labels = {int(k): v["label"] for k, v in json.load(open(os.path.join(ROOT, "audit/labels_coder1_final.json"))).items()}
    strata = json.load(open(os.path.join(ROOT, "audit/strata.json")))
    remaining = {"DABstep": 19466, "DAB": 27984}   # failed steps that the rules do not attribute to the harness
    rng = np.random.default_rng(SEED)
    print("== audit, first stratum weighted by error kind (not pre-registered)")
    for corpus in ("DABstep", "DAB"):
        pos, neg = collections.defaultdict(list), collections.defaultdict(list)
        for i, r in ids.items():
            if r["corpus"] == corpus and labels[i] in (1, 2, 3):
                (pos[r["rule_group"]] if r["stratum"] == "pos" else neg[r["stratum"]]).append(labels[i] == 1)
        N = strata[corpus]["population"]
        assert sum(sizes[corpus].values()) == N["pos"], (corpus, sum(sizes[corpus].values()), N["pos"])

        def count(p, n):
            return (sum(sizes[corpus][k] * np.mean(v) for k, v in p.items())
                    + sum(N[h] * np.mean(v) for h, v in n.items()))
        est = count(pos, neg)
        boot = [count({k: rng.choice(v, len(v)) for k, v in pos.items()},
                      {h: rng.choice(v, len(v)) for h, v in neg.items()}) for _ in range(B)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        d = remaining[corpus]
        kinds = {k: f"{sum(v)}/{len(v)} of {sizes[corpus][k]}" for k, v in sorted(pos.items())}
        print(f"  {corpus}: {est:.0f} (95% CI {lo:.0f} to {hi:.0f}); {est / d:.1%} ({lo / d:.1%} to {hi / d:.1%}); "
              f"first stratum by kind (real/sampled of stratum size): {kinds}")


def rank_bootstrap(dab_cells):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in dab_cells:
        for line in open(path):
            r = json.loads(line)
            failed = sum(1 for x in r["calls"] if x["failed"])
            harness = sum(1 for x in r["calls"] if x["failed"] and x.get("harness_syntax"))
            by[r["model"]][(r["dataset"], r["query"])].append((failed, failed - harness))
    models = sorted(by)
    tasks = sorted(set.intersection(*[set(by[m]) for m in models]))
    W = np.array([[len(by[m][t]) for t in tasks] for m in models])
    A = np.array([[np.mean([a for a, _ in by[m][t]]) for t in tasks] for m in models])
    C = np.array([[np.mean([c for _, c in by[m][t]]) for t in tasks] for m in models])
    rng = np.random.default_rng(SEED)
    i = models.index("gpt-5.1")
    with_h, without_h, falls, falls2 = collections.Counter(), collections.Counter(), 0, 0
    for _ in range(B):
        idx = rng.integers(0, len(tasks), len(tasks))
        w = W[:, idx]
        ka = list(np.argsort((A[:, idx] * w).sum(1) / w.sum(1))).index(i) + 1
        kc = list(np.argsort((C[:, idx] * w).sum(1) / w.sum(1))).index(i) + 1
        with_h[ka] += 1
        without_h[kc] += 1
        falls += kc > ka
        falls2 += kc - ka >= 2
    print(f"== rank of gpt-5.1 by failed calls per run (1 = fewest), bootstrap over {len(tasks)} DAB tasks")
    print(f"  with harness-caused failures: {dict(sorted(with_h.items()))}")
    print(f"  without them: {dict(sorted(without_h.items()))}")
    print(f"  rank falls: {falls / B:.1%}; falls by two or more places: {falls2 / B:.1%}")


def rerun_requests():
    final = {}
    for line in open(os.path.join(ROOT, "dab-rerun/runs.jsonl")):
        r = json.loads(line)
        final[(r["arm"], r["dataset"], r["query"], r["block"])] = r
    infra = ("llm_response_failed", "agent_run_failed")
    kept = {k: r for k, r in final.items() if not (r["terminate_reason"] or "").startswith(infra)}
    print("== DAB rerun: requests to the model per run")
    means = {}
    for arm in "AB":
        calls = [r["llm_calls"] for k, r in kept.items() if k[0] == arm]
        known = [c for c in calls if c is not None]
        means[arm] = np.mean(known)
        print(f"  {arm}: {means[arm]:.2f} over {len(known)} runs ({len(calls) - len(known)} without a recorded count)")
    print(f"  A / B = {means['A'] / means['B']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dabstep-cells", nargs="+", required=True)
    ap.add_argument("--dab-cells", nargs="+", required=True)
    a = ap.parse_args()
    post_stratified(group_sizes(a.dabstep_cells, a.dab_cells))
    print(f"== 0 of 19 sampled missing-file reads not invented: one-sided 95% upper bound {1 - 0.05 ** (1 / 19):.1%}")
    rank_bootstrap(a.dab_cells)
    rerun_requests()


if __name__ == "__main__":
    main()
