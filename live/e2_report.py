#!/usr/bin/env python3
"""E2 analysis, as pre-registered in live/e2_preregistration.md, section 5.

Question = unit of inference. For each question, arm and measure, values are averaged over the
replicates (seeds) that both compared arms finished; a difference between arms is the mean over
questions of the per-question differences.

  H1  accuracy R - N (primary); H2 E - N; H3 R - E: 95% CI by percentile bootstrap over questions
      (10,000 resamples, seed 20261012); two-sided sign-flip permutation test on the per-question
      differences (100,000 permutations); McNemar and Newcombe on (question, seed) pairs for continuity
      with E1. Decision rules of section 5.
  H4  steps (model replies) and output tokens: the same CI, and the exact sign test.
  H5  runs that reach a data-reference attempt (DRA): share per arm; accuracy, steps and output tokens
      among them; differences by bootstrap over questions.
  H6  live false blocks of release 4: shadow blocks in N and E on steps that ran without an error;
      exact one-sided 95% upper bounds per step and per run; block check of R (--validation).
  H7  executed missing-name errors by kind; DRAs per 100 runs; whether the step after the first DRA
      ran without a missing-name error.

Only complete seeds are used (all their jobs recorded, or the runner has moved on to the next seed);
missing jobs are reported.

Usage: python3 live/e2_report.py --results runs/e2/results.jsonl \\
           --questions data/QRData/benchmark/QRData.json [--validation runs/e2/validation.jsonl] [--json out.json]
"""
import argparse
import collections
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_report import correct, newcombe_paired, sign_test  # noqa: E402

ARMS = ("N", "R", "E")
NAME_ERRORS = ("KeyError", "FileNotFoundError", "IsADirectoryError", "NotADirectoryError")
FILE_ERRORS = ("FileNotFoundError", "IsADirectoryError", "NotADirectoryError")
SEED = 20261012
FA_DEF = re.compile(r"^\s*(def\s+final_answer\b|final_answer\s*=)", re.M)


def upper95(k, n):
    """Exact one-sided 95% upper bound of a binomial proportion (Clopper-Pearson)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > 0.05 else (lo, mid)
    return hi


def run_measures(r, gold):
    cells = [c for c in r["cells"] if not c.get("no_code")]
    dra_idx = None
    kinds = collections.Counter()
    false_shadow = 0
    for i, c in enumerate(cells):
        blocked = bool(c.get("blocked"))
        gate_block = c.get("gate") == "block"
        name_err = (not blocked) and c.get("ok") is False and c.get("error_type") in NAME_ERRORS
        if (gate_block or name_err) and dra_idx is None:
            dra_idx = i
        if name_err:
            if c["error_type"] in FILE_ERRORS:
                kinds["file"] += 1
            elif gate_block:
                kinds["key_checker_block"] += 1
            else:
                kinds["key_other"] += 1
        if r["arm"] != "R" and gate_block and c.get("ok") is True:
            false_shadow += 1
    after_ok = None
    if dra_idx is not None and dra_idx + 1 < len(cells):
        nxt = cells[dra_idx + 1]
        after_ok = not (nxt.get("blocked") or (nxt.get("ok") is False and nxt.get("error_type") in NAME_ERRORS))
    fin_ok = any(c.get("called") and c.get("ok") for c in cells)
    task = gold[r["task"]]
    return {"correct": correct(r.get("final_answer"), task[0], task[1]),
            "steps": r["usage"]["calls"], "out_tokens": r["usage"]["completion_tokens"],
            "in_tokens": r["usage"]["prompt_tokens"], "cost": r.get("cost_usd") or 0,
            "dra": dra_idx is not None, "kinds": kinds, "after_ok": after_ok,
            "blocks": sum(1 for c in cells if c.get("blocked")),
            "shadow_blocks": sum(1 for c in cells if r["arm"] != "R" and c.get("gate") == "block"),
            "false_shadow": false_shadow, "shadow_steps_ran": sum(1 for c in cells if r["arm"] != "R" and c.get("ok") is True),
            "executor_lost": r.get("status") == "executor_lost",
            "lost_answer": fin_ok and r.get("final_answer") is None,
            "defines_final_answer": any(FA_DEF.search(c.get("code") or "") for c in cells),
            "answered": r.get("final_answer") is not None}


def boot_ci(per_q, n=10000, seed=SEED):
    x = np.asarray(per_q, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def perm_p(per_q, n=100000, seed=SEED):
    x = np.asarray(per_q, dtype=float)
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n, 10000):
        k = min(10000, n - start)
        signs = rng.choice((-1.0, 1.0), size=(k, len(x)))
        hits += int((np.abs((signs * x).mean(axis=1)) >= obs - 1e-12).sum())
    return (hits + 1) / (n + 1)


def compare(by, qs, x, y, label, out):
    """y minus x (treatment minus control), paired on (question, seed)."""
    dq, steps_q, tok_q, pairs = [], [], [], []
    for t in qs:
        if x not in by[t] or y not in by[t]:
            continue
        seeds = sorted(set(by[t][x]) & set(by[t][y]))
        if not seeds:
            continue
        dq.append(np.mean([by[t][y][s]["correct"] - by[t][x][s]["correct"] for s in seeds]))
        steps_q.append(np.mean([by[t][x][s]["steps"] - by[t][y][s]["steps"] for s in seeds]))
        tok_q.append(np.mean([by[t][x][s]["out_tokens"] - by[t][y][s]["out_tokens"] for s in seeds]))
        pairs += [(by[t][x][s]["correct"], by[t][y][s]["correct"]) for s in seeds]
    diff = float(np.mean(dq))
    lo, hi = boot_ci(dq)
    p = perm_p(dq)
    a = sum(1 for u, v in pairs if u and v)
    b = sum(1 for u, v in pairs if v and not u)
    c = sum(1 for u, v in pairs if u and not v)
    d = len(pairs) - a - b - c
    nd, nlo, nhi = newcombe_paired(a, b, c, d)
    if lo > 0 or hi < 0:
        verdict = "changed accuracy" if p < 0.05 else "interval excludes 0 but p >= 0.05: no change claimed"
    else:
        verdict = "no change was detected"
    res = {"questions": len(dq), "pairs": len(pairs), "acc_control": sum(u for u, _ in pairs) / len(pairs),
           "acc_treatment": sum(v for _, v in pairs) / len(pairs), "diff": diff, "ci": [lo, hi], "perm_p": p,
           "verdict": verdict, "not_lower_than_minus_2_points": lo > -0.02,
           "mcnemar": {"treatment_only": b, "control_only": c, "p": sign_test(b, c)},
           "newcombe": [nd, nlo, nhi],
           "steps_saved": float(np.mean(steps_q)), "steps_saved_ci": list(boot_ci(steps_q)),
           "steps_sign": [int(sum(v > 0 for v in steps_q)), int(sum(v < 0 for v in steps_q)),
                          sign_test(int(sum(v > 0 for v in steps_q)), int(sum(v < 0 for v in steps_q)))],
           "out_tokens_saved": float(np.mean(tok_q)), "out_tokens_saved_ci": list(boot_ci(tok_q)),
           "tokens_sign": [int(sum(v > 0 for v in tok_q)), int(sum(v < 0 for v in tok_q)),
                           sign_test(int(sum(v > 0 for v in tok_q)), int(sum(v < 0 for v in tok_q)))]}
    print(f"{label} ({y} - {x}): {len(dq)} questions, {len(pairs)} pairs; accuracy {res['acc_control']:.4f} -> "
          f"{res['acc_treatment']:.4f}; diff {100 * diff:+.2f} points [{100 * lo:+.2f}, {100 * hi:+.2f}], "
          f"permutation p {p:.4f}; {verdict}; lower bound above -2 points: {res['not_lower_than_minus_2_points']}; "
          f"McNemar {b}/{c} p {res['mcnemar']['p']:.4f}; Newcombe {100 * nd:+.2f} [{100 * nlo:+.2f}, {100 * nhi:+.2f}]")
    print(f"   steps saved ({x} - {y}) {res['steps_saved']:.3f} {[round(v, 3) for v in res['steps_saved_ci']]}, "
          f"sign {res['steps_sign'][0]}/{res['steps_sign'][1]} p {res['steps_sign'][2]:.4f}; output tokens saved "
          f"{res['out_tokens_saved']:.0f} {[round(v) for v in res['out_tokens_saved_ci']]}, sign p {res['tokens_sign'][2]:.4f}")
    out[label] = res


def dra_compare(by, qs, x, y, label, out, n=10000):
    """H5: accuracy, steps and output tokens among runs with a DRA, y minus x; bootstrap over questions."""
    def pooled(sel, arm, key):
        num = den = 0
        for t in sel:
            for v in by[t].get(arm, {}).values():
                if v["dra"]:
                    num += v[key]
                    den += 1
        return num / den if den else float("nan")
    res = {}
    rng = np.random.default_rng(SEED)
    for key in ("correct", "steps", "out_tokens"):
        point = pooled(qs, y, key) - pooled(qs, x, key)
        bs = []
        for _ in range(n):
            sel = [qs[i] for i in rng.integers(0, len(qs), size=len(qs))]
            v = pooled(sel, y, key) - pooled(sel, x, key)
            if v == v:
                bs.append(v)
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else [float("nan")] * 2
        res[key] = {"diff": point, "ci": ci}
    c = res["correct"]
    print(f"{label} ({y} - {x}) among runs with a DRA: accuracy {100 * c['diff']:+.2f} points "
          f"[{100 * c['ci'][0]:+.2f}, {100 * c['ci'][1]:+.2f}]; steps {res['steps']['diff']:+.3f} "
          f"{[round(v, 3) for v in res['steps']['ci']]}; output tokens {res['out_tokens']['diff']:+.0f}")
    out[label] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--validation")
    ap.add_argument("--json")
    ap.add_argument("--jobs-per-seed", type=int, default=1233)
    a = ap.parse_args()
    qsrc = json.load(open(a.questions))
    gold = {f"qr-{i}": (q["answer"], (q.get("meta_data") or {}).get("question_type")) for i, q in enumerate(qsrc)}
    rows = [json.loads(line) for line in open(a.results)]
    per_seed = collections.Counter(r["seed"] for r in rows)
    seeds_seen = sorted(per_seed)
    complete = [s for s in seeds_seen if per_seed[s] >= a.jobs_per_seed or any(t > s for t in seeds_seen)]
    out = {"records": len(rows), "jobs_per_seed": dict(per_seed), "complete_seeds": complete,
           "missing_jobs": {s: a.jobs_per_seed - per_seed[s] for s in complete if per_seed[s] < a.jobs_per_seed}}
    print(f"records {len(rows)}; complete seeds {complete}; missing jobs in them {out['missing_jobs']}")
    rows = [r for r in rows if r["seed"] in complete]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["task"]].setdefault(r["arm"], {})[r["seed"]] = run_measures(r, gold)
    qs = sorted(by)

    out["by_arm"] = {}
    for arm in ARMS:
        xs = [v for t in qs for v in by[t].get(arm, {}).values()]
        if not xs:
            continue
        k = collections.Counter()
        for v in xs:
            k.update(v["kinds"])
        s = {"runs": len(xs), "accuracy": sum(v["correct"] for v in xs) / len(xs),
             "answered": sum(v["answered"] for v in xs), "executor_lost": sum(v["executor_lost"] for v in xs),
             "lost_answers": sum(v["lost_answer"] for v in xs),
             "runs_defining_final_answer": sum(v["defines_final_answer"] for v in xs),
             "steps_mean": float(np.mean([v["steps"] for v in xs])),
             "out_tokens_mean": float(np.mean([v["out_tokens"] for v in xs])),
             "in_tokens_mean": float(np.mean([v["in_tokens"] for v in xs])), "cost": sum(v["cost"] for v in xs),
             "runs_with_dra": sum(v["dra"] for v in xs), "dra_share": sum(v["dra"] for v in xs) / len(xs),
             "accuracy_with_dra": (sum(v["correct"] for v in xs if v["dra"]) / max(1, sum(v["dra"] for v in xs))),
             "accuracy_without_dra": (sum(v["correct"] for v in xs if not v["dra"]) /
                                      max(1, sum(not v["dra"] for v in xs))),
             "executed_name_errors": dict(k), "executed_name_errors_per_100_runs": 100 * sum(k.values()) / len(xs),
             "blocks": sum(v["blocks"] for v in xs), "shadow_blocks": sum(v["shadow_blocks"] for v in xs),
             "next_step_ok_after_first_dra": [sum(1 for v in xs if v["after_ok"]),
                                             sum(1 for v in xs if v["after_ok"] is not None)]}
        out["by_arm"][arm] = s
        print(f"{arm}: runs {s['runs']}; accuracy {s['accuracy']:.4f}; answered {s['answered']}; executor lost "
              f"{s['executor_lost']}; lost answers {s['lost_answers']} (runs defining final_answer "
              f"{s['runs_defining_final_answer']}); steps {s['steps_mean']:.2f}; output tokens {s['out_tokens_mean']:.0f}; "
              f"cost ${s['cost']:.2f}; runs with a DRA {s['runs_with_dra']} ({100 * s['dra_share']:.1f}%); executed "
              f"name errors {dict(k)} ({s['executed_name_errors_per_100_runs']:.1f} per 100 runs); blocks {s['blocks']}; "
              f"shadow blocks {s['shadow_blocks']}; next step ok after first DRA {s['next_step_ok_after_first_dra']}")

    print("-- H1 to H4")
    compare(by, qs, "N", "R", "H1 checker", out)
    compare(by, qs, "N", "E", "H2 names in the error", out)
    compare(by, qs, "E", "R", "H3 checker against names in the error", out)
    print("-- H5")
    dra_compare(by, qs, "N", "R", "H5 checker", out)
    dra_compare(by, qs, "N", "E", "H5 names in the error", out)
    dra_compare(by, qs, "E", "R", "H5 checker against names in the error", out)
    print("-- H6")
    fs = sum(v["false_shadow"] for t in qs for arm in ("N", "E") for v in by[t].get(arm, {}).values())
    steps_ran = sum(v["shadow_steps_ran"] for t in qs for arm in ("N", "E") for v in by[t].get(arm, {}).values())
    runs_ne = [v for t in qs for arm in ("N", "E") for v in by[t].get(arm, {}).values()]
    runs_fs = sum(v["false_shadow"] > 0 for v in runs_ne)
    out["H6"] = {"false_shadow_blocks": fs, "steps_that_ran": steps_ran, "upper95_per_step": upper95(fs, steps_ran),
                 "runs_with_false_block": runs_fs, "runs": len(runs_ne),
                 "upper95_per_run": upper95(runs_fs, len(runs_ne))}
    print(f"H6 live false blocks (shadow, N and E): {fs} of {steps_ran} steps that ran (95% upper bound "
          f"{100 * out['H6']['upper95_per_step']:.3f}%); runs {runs_fs} of {len(runs_ne)} (upper bound "
          f"{100 * out['H6']['upper95_per_run']:.3f}%)")
    if a.validation:
        v = [json.loads(line) for line in open(a.validation)]
        cls = collections.Counter(x["class"] for x in v)
        n_rep = sum(cls[c] for c in cls if c != "not_replayed")
        out["block_check"] = {"classes": dict(cls), "upper95_no_error": upper95(cls.get("no_error", 0), n_rep)}
        print(f"block check of R: {dict(cls)}; 95% upper bound on false blocks {100 * out['block_check']['upper95_no_error']:.3f}%")
    print("-- identity check (H5): share with a DRA x DRA difference against the all-run difference")
    for x, y, lab in (("N", "R", "H1 checker"), ("N", "E", "H2 names in the error")):
        sh = np.mean([out["by_arm"][x]["dra_share"], out["by_arm"][y]["dra_share"]])
        lab5 = "H5 checker" if lab.startswith("H1") else "H5 names in the error"
        print(f"   {lab}: {100 * out[lab]['diff']:+.2f} points; share {100 * sh:.1f}% x {100 * out[lab5]['correct']['diff']:+.2f}"
              f" = {sh * 100 * out[lab5]['correct']['diff']:+.2f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
