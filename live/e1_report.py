#!/usr/bin/env python3
"""E1 analysis. Question = unit, paired between arms.

H1 (gate, within P0): cells(P0G0) - cells(P0G1) > 0 (exact sign test on non-zero
differences, p < 0.05, bootstrap 95% CI of the mean above 0), and accuracy not lower by
more than 2 points (lower bound of the Newcombe paired 95% CI of acc(G1) - acc(G0) above
-0.02). H3: the same within P1. H2 (prompting, within G0): executed missing-name failures
per question, P0 against P1.

Accuracy follows QRData's eval.py: numbers within +-3% of the gold value; multiple choice
by prefix of the lower-cased answer.

Usage: python3 live/e1_report.py --results runs/e1/results.jsonl --questions .../QRData.json \
    [--validation runs/e1/validation.jsonl] [--json out.json]
"""
import argparse
import collections
import json
import math
import random
import re
import statistics

DATA_ERRORS = ("KeyError", "FileNotFoundError")


def first_number(s):
    m = re.search(r"-?\d+\.?\d*%?", s)
    return m.group() if m else None


def correct(pred, gold, qtype):
    """QRData eval.py calc_acc, for one answer."""
    if pred is None:
        return False
    p, g = str(pred).lower(), str(gold).lower()
    if qtype == "numerical":
        gf = float(g[:-1]) / 100 if g.endswith("%") else float(g)
        x = first_number(p)
        if x is None:
            return False
        try:
            pf = float(x[:-1]) / 100 if x.endswith("%") else float(x)
        except ValueError:
            return False
        lo, hi = min(gf * 0.97, gf * 1.03), max(gf * 0.97, gf * 1.03)
        return lo < pf < hi
    return g == p[:len(g)]


def sign_test(w, l):
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_mean(xs, n=10000, seed=0):
    rng = random.Random(seed)
    m = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def newcombe_paired(a, b, c, d):
    """95% CI of p1 - p2 for paired proportions (Newcombe 1998, method 10).
    a: both right; b: arm 1 right only; c: arm 2 right only; d: both wrong."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    num = a * d - b * c
    phi = 0.0 if den == 0 else (max(num - n / 2, 0) if num > 0 else num) / den
    diff = p1 - p2
    lo = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return diff, lo, hi


def per_arm(r, gold):
    cells = [c for c in r["cells"] if not c.get("no_code")]
    return {"cells": len(cells),
            "correct": correct(r.get("final_answer"), gold[r["task"]][0], gold[r["task"]][1]),
            "answered": r.get("final_answer") is not None,
            "tokens": r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"],
            "cost": r.get("cost_usd") or 0,
            "name_failures": sum(1 for c in cells if c.get("ok") is False and c.get("error_type") in DATA_ERRORS),
            "blocked": sum(1 for c in cells if c.get("blocked")),
            "done": r.get("status") == "done"}


def compare(by, q, x, y, label, out):
    """x = control arm, y = treatment arm. by[task][arm] = {seed: per_arm dict}. Cells and tokens:
    question-level means over the seeds that both arms finished. Accuracy: (question, seed) pairs."""
    qd, td, accp = [], [], []
    for t in q:
        if x not in by[t] or y not in by[t]:
            continue
        seeds = sorted(s for s in by[t][x] if s in by[t][y] and by[t][x][s]["done"] and by[t][y][s]["done"])
        if not seeds:
            continue
        qd.append(statistics.mean(by[t][x][s]["cells"] - by[t][y][s]["cells"] for s in seeds))
        td.append(statistics.mean(by[t][x][s]["tokens"] - by[t][y][s]["tokens"] for s in seeds))
        accp += [(by[t][x][s], by[t][y][s]) for s in seeds]
    w, l = sum(v > 0 for v in qd), sum(v < 0 for v in qd)
    lo, hi = boot_mean(qd)
    a_ = sum(1 for a, b in accp if a["correct"] and b["correct"])
    b_ = sum(1 for a, b in accp if b["correct"] and not a["correct"])      # treatment right only
    c_ = sum(1 for a, b in accp if a["correct"] and not b["correct"])      # control right only
    dd = len(accp) - a_ - b_ - c_
    diff, alo, ahi = newcombe_paired(a_, b_, c_, dd)
    tlo, thi = boot_mean(td)
    res = {"questions": len(qd), "pairs": len(accp), "cells_saved_mean": statistics.mean(qd), "cells_ci": [lo, hi],
           "fewer": w, "more": l, "equal": len(qd) - w - l, "sign_p": sign_test(w, l),
           "acc_control": sum(a["correct"] for a, _ in accp) / len(accp),
           "acc_treatment": sum(b["correct"] for _, b in accp) / len(accp),
           "acc_diff": diff, "acc_diff_ci": [alo, ahi], "discordant": [b_, c_], "mcnemar_p": sign_test(b_, c_),
           "tokens_saved_mean": statistics.mean(td), "tokens_ci": [tlo, thi]}
    res["benefit_claim"] = bool(lo > 0 and res["sign_p"] < 0.05 and alo > -0.02)
    print(f"{label}: questions {len(qd)}, pairs {len(accp)}; cells saved {res['cells_saved_mean']:.3f} [{lo:.3f}, {hi:.3f}], "
          f"fewer/more/equal {w}/{l}/{res['equal']}, sign p {res['sign_p']:.4f}; accuracy {res['acc_control']:.3f} -> "
          f"{res['acc_treatment']:.3f}, diff {diff:+.3f} [{alo:+.3f}, {ahi:+.3f}], McNemar p {res['mcnemar_p']:.3f}; "
          f"tokens saved {res['tokens_saved_mean']:.0f} [{tlo:.0f}, {thi:.0f}]; benefit claim {res['benefit_claim']}")
    out[label] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--validation")
    ap.add_argument("--json")
    ap.add_argument("--seeds", help="comma-separated seeds to use (default: all)")
    a = ap.parse_args()
    qs = json.load(open(a.questions))
    gold = {f"qr-{i}": (q["answer"], (q.get("meta_data") or {}).get("question_type")) for i, q in enumerate(qs)}
    by = collections.defaultdict(dict)
    rows = [json.loads(line) for line in open(a.results)]
    if a.seeds:
        keep = {int(x) for x in a.seeds.split(",")}
        rows = [r for r in rows if r["seed"] in keep]
    for r in rows:
        by[r["task"]].setdefault(r["arm"], {})[r["seed"]] = per_arm(r, gold)
    q = sorted(by)
    out = {"records": len(rows), "questions": len(q), "by_arm": {}}
    for arm in ("P0G0", "P0G1", "P1G0", "P1G1"):
        xs = [v for t in q if arm in by[t] for v in by[t][arm].values()]
        if not xs:
            continue
        s = {"n": len(xs), "not_done": sum(not x["done"] for x in xs),
             "accuracy": sum(x["correct"] for x in xs) / len(xs), "answered": sum(x["answered"] for x in xs),
             "cells_mean": statistics.mean(x["cells"] for x in xs),
             "tokens_mean": statistics.mean(x["tokens"] for x in xs), "cost": sum(x["cost"] for x in xs),
             "name_failures": sum(x["name_failures"] for x in xs),
             "questions_with_name_failure": sum(x["name_failures"] > 0 for x in xs),
             "blocked": sum(x["blocked"] for x in xs), "questions_with_block": sum(x["blocked"] > 0 for x in xs)}
        out["by_arm"][arm] = s
        print(f"{arm}: n {s['n']} (not done {s['not_done']}); accuracy {s['accuracy']:.3f}; cells {s['cells_mean']:.2f}; "
              f"tokens {s['tokens_mean']:.0f}; executed name failures {s['name_failures']} in "
              f"{s['questions_with_name_failure']} questions; blocked {s['blocked']} in {s['questions_with_block']}; "
              f"cost ${s['cost']:.2f}")
    compare(by, q, "P0G0", "P0G1", "H1 gate within P0", out)
    compare(by, q, "P1G0", "P1G1", "H3 gate within P1", out)
    compare(by, q, "P0G0", "P1G0", "prompt within G0", out)
    pairs = [(by[t]["P0G0"][sd], by[t]["P1G0"][sd]) for t in q if "P0G0" in by[t] and "P1G0" in by[t]
             for sd in by[t]["P0G0"] if sd in by[t]["P1G0"]]
    if pairs:
        f0, f1 = sum(a["name_failures"] for a, _ in pairs), sum(b["name_failures"] for _, b in pairs)
        w = sum(a["name_failures"] > b["name_failures"] for a, b in pairs)
        l = sum(a["name_failures"] < b["name_failures"] for a, b in pairs)
        out["H2"] = {"pairs": len(pairs), "failures_P0": f0, "failures_P1": f1, "fewer": w, "more": l,
                     "sign_p": sign_test(w, l),
                     "questions_with_failure_P0": sum(a["name_failures"] > 0 for a, _ in pairs),
                     "questions_with_failure_P1": sum(b["name_failures"] > 0 for _, b in pairs)}
        print(f"H2 prompting within G0: executed name failures P0 {f0} -> P1 {f1} (question-runs with one: "
              f"{out['H2']['questions_with_failure_P0']} -> {out['H2']['questions_with_failure_P1']}); "
              f"fewer/more {w}/{l}, sign p {sign_test(w, l):.4f}")
    if a.validation:
        v = [json.loads(line) for line in open(a.validation)]
        out["validation"] = dict(collections.Counter(x["class"] for x in v))
        print("block validation:", out["validation"])
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
