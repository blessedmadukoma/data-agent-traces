#!/usr/bin/env python3
"""Summarise an RQ3 run: paired outcomes, validity checks and cost.

Usage: python3 rq3-pilot/rq3_report.py $RQ3_RUN/results.jsonl [--plan-episodes 200 --plan-seeds 3]

Pairs: the traceback and receipt arms of one (episode, seed). A pair is
excluded if either arm did not finish, or if the blocked cell did not fail
in the sandbox with the missing name in its error (the replay does not
reproduce the recorded failure). Primary outcome: cells to the first cell
that runs without error and does not only inspect the data; a pair where an
arm never recovers within the step limit counts that arm as steps + 1.
"""
import argparse
import collections
import json
import math
import statistics


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--plan-episodes", type=int, default=200)
    ap.add_argument("--plan-seeds", type=int, default=3)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.results)]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["episode"], r["seed"])][r["arm"]] = r
    C = collections.Counter()
    pairs = []
    for key, arms in by.items():
        t, rc = arms.get("traceback"), arms.get("receipt")
        if not t or not rc:
            C["incomplete"] += 1
            continue
        if t.get("status") != "done" or rc.get("status") != "done":
            C["not_done"] += 1
            continue
        if not t.get("blocked_names_missing"):
            C["replay_does_not_reproduce"] += 1
            continue
        pairs.append((t, rc))
    steps = max((r["params"]["steps"] for r in rows), default=6)

    def y(r):
        return r["steps_to_fix"] if r["steps_to_fix"] is not None else steps + 1

    print(f"episode-arms: {len(rows)}; pairs used: {len(pairs)}; excluded: {dict(C)}")
    if pairs:
        for arm, idx in (("traceback", 0), ("receipt", 1)):
            rs = [p[idx] for p in pairs]
            print(f"  {arm:9s} steps to fix: mean {statistics.mean(map(y, rs)):.2f}, median {statistics.median(map(y, rs))}; "
                  f"never recovered {sum(r['never_recovered'] for r in rs)}/{len(rs)}; "
                  f"constant answer {sum(r['constant_answer'] for r in rs)}/{len(rs)}; "
                  f"first step repeats the missing name {sum(r['first_step_repeats_missing'] for r in rs)}/{len(rs)}")
        def y2(r):
            v = r.get("steps_to_fix_nonconstant")
            return v if v is not None else steps + 1
        for arm, idx in (("traceback", 0), ("receipt", 1)):
            rs = [p[idx] for p in pairs]
            print(f"  {arm:9s} steps to fix, a constant final answer not counted as a fix: "
                  f"mean {statistics.mean(map(y2, rs)):.2f}")
        d = [y(t) - y(rc) for t, rc in pairs]
        wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
        print(f"  paired difference (traceback - receipt): mean {statistics.mean(d):.2f}; receipt fewer steps in {wins}, "
              f"more in {losses}, equal in {len(d) - wins - losses}; sign test p = {sign_test(wins, losses):.3f}")
        d2 = [y2(t) - y2(rc) for t, rc in pairs]
        w2, l2 = sum(x > 0 for x in d2), sum(x < 0 for x in d2)
        print(f"  same, constant answers not counted: receipt fewer steps in {w2}, more in {l2}; "
              f"sign test p = {sign_test(w2, l2):.3f}")
        import random
        rnd = random.Random(0)
        boots = sorted(statistics.mean(rnd.choice(d) for _ in d) for _ in range(10000))
        print(f"  bootstrap 95% CI of the mean difference: [{boots[250]:.2f}, {boots[9749]:.2f}]")
        try:
            from scipy.stats import wilcoxon
            nz = [x for x in d if x != 0]
            if nz:
                print(f"  Wilcoxon signed-rank (non-zero differences, n = {len(nz)}): p = {wilcoxon(nz).pvalue:.3f}")
        except ImportError:
            pass
        mism = [t.get("prefix_mismatch", 0) for t, _ in pairs]
        print(f"  replay: earlier cells whose outcome differs from the recording, per episode: mean {statistics.mean(mism):.2f}, "
              f"max {max(mism)}")
    cost = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    toks = [(r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"], r["usage"]["calls"]) for r in rows if r.get("usage")]
    if toks:
        pin = statistics.mean(t[0] for t in toks)
        pout = statistics.mean(t[1] for t in toks)
        calls = statistics.mean(t[2] for t in toks)
        print(f"tokens per episode-arm: input {pin:,.0f}, output {pout:,.0f}, model calls {calls:.1f}")
    if cost:
        per = statistics.mean(cost)
        n = a.plan_episodes * a.plan_seeds * 2
        print(f"cost: total ${sum(cost):.2f}; per episode-arm ${per:.4f}; plan {a.plan_episodes} episodes x "
              f"{a.plan_seeds} seeds x 2 arms = {n} episode-arms, about ${per * n:.2f}")
    served = collections.Counter(r.get("served_model") for r in rows)
    print(f"served model ids: {dict(served)}")


if __name__ == "__main__":
    main()
