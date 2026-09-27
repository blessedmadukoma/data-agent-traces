#!/usr/bin/env python3
"""RQ3 full study: the pre-registered analysis of the receipt replay.

Usage: python3 rq3-pilot/rq3_full_report.py --episodes runs/rq3-full/episodes.jsonl \
    --results runs/rq3-full/results_gpt-oss.jsonl [--transcripts runs/rq3-full/transcripts_gpt-oss] [--json out.json]

Pair = (episode, seed). A pair is dropped if either arm did not finish, or if in the
traceback arm the blocked cell did not fail with the missing name in its error.
y = cells to the first cell that runs without error and does not only inspect the
data (7 if never). d = y(traceback) - y(receipt). D = mean d over an episode's pairs.
"""
import argparse
import collections
import json
import math
import os
import random
import re
import statistics

STEPS = 6


def sign_test(w, l):
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon(xs):
    try:
        from scipy.stats import wilcoxon as w
        nz = [x for x in xs if x != 0]
        return w(nz).pvalue if nz else float("nan")
    except ImportError:
        return float("nan")


def boot(values_by_unit, stat, n=10000, seed=0):
    rnd = random.Random(seed)
    units = list(values_by_unit)
    out = sorted(stat([values_by_unit[rnd.choice(units)] for _ in units]) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


def y(r, nonconstant=False):
    v = r.get("steps_to_fix_nonconstant") if nonconstant else r.get("steps_to_fix")
    return v if v is not None else STEPS + 1


def summarise(name, D, extra=""):
    """D: {unit: difference}. Mean, bootstrap CI, sign test, Wilcoxon."""
    vals = list(D.values())
    if not vals:
        print(f"  {name}: no units")
        return None
    w, l = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
    lo, hi = boot(D, statistics.mean)
    p, pw = sign_test(w, l), wilcoxon(vals)
    print(f"  {name}: n = {len(vals)}; mean saving {statistics.mean(vals):.3f} [{lo:.3f}, {hi:.3f}]; "
          f"receipt fewer {w}, more {l}, equal {len(vals) - w - l}; sign p = {p:.4f}; Wilcoxon p = {pw:.4f}{extra}")
    return {"n": len(vals), "mean": statistics.mean(vals), "ci": [lo, hi], "fewer": w, "more": l,
            "equal": len(vals) - w - l, "sign_p": p, "wilcoxon_p": pw}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--transcripts")
    ap.add_argument("--json")
    a = ap.parse_args()
    eps = {e["episode"]: e for e in map(json.loads, open(a.episodes))}
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
        pairs.append((key, t, rc))
    seeds = collections.Counter(k[1] for k, _, _ in pairs)
    print(f"episode-arms {len(rows)}; pairs {len(by)}; used {len(pairs)} (by seed {dict(seeds)}); excluded {dict(C)}")
    out = {"excluded": dict(C), "pairs_used": len(pairs), "by_seed": dict(seeds)}

    def episode_D(ps, nonconstant=False):
        acc = collections.defaultdict(list)
        for (ep, seed), t, rc in ps:
            acc[ep].append(y(t, nonconstant) - y(rc, nonconstant))
        return {ep: statistics.mean(v) for ep, v in acc.items()}

    for arm, idx in (("traceback", 1), ("receipt", 2)):
        ys = [y(p[idx]) for p in pairs]
        print(f"  {arm:9s} cells to fix: mean {statistics.mean(ys):.3f}; never recovered "
              f"{sum(p[idx]['never_recovered'] for p in pairs)}; constant answer {sum(p[idx]['constant_answer'] for p in pairs)}; "
              f"first cell repeats the missing name {sum(p[idx]['first_step_repeats_missing'] for p in pairs)}")
    print("PRIMARY (episode = unit)")
    out["primary"] = summarise("all episodes", episode_D(pairs))
    print("SECONDARY: by receipt kind (Holm over the three kinds)")
    kinds = {}
    for kind in ("column", "file", "key"):
        kinds[kind] = summarise(kind, episode_D([p for p in pairs if eps[p[0][0]]["receipt"]["kind"] == kind]))
    adj = holm([kinds[k]["sign_p"] if kinds[k] else 1.0 for k in ("column", "file", "key")])
    print("  Holm-adjusted sign p: " + ", ".join(f"{k} {p:.4f}" for k, p in zip(("column", "file", "key"), adj)))
    out["kinds"], out["kinds_holm"] = kinds, dict(zip(("column", "file", "key"), adj))
    Dc = episode_D([p for p in pairs if eps[p[0][0]]["receipt"]["kind"] == "column"])
    Do = episode_D([p for p in pairs if eps[p[0][0]]["receipt"]["kind"] != "column"])
    rnd = random.Random(0)
    diffs = []
    kc, ko = list(Dc), list(Do)
    for _ in range(10000):
        diffs.append(statistics.mean(Dc[rnd.choice(kc)] for _ in kc) - statistics.mean(Do[rnd.choice(ko)] for _ in ko))
    diffs.sort()
    dd = statistics.mean(Dc.values()) - statistics.mean(Do.values())
    print(f"  column minus (file and key): {dd:.3f} [{diffs[250]:.3f}, {diffs[9749]:.3f}]")
    out["column_minus_other"] = [dd, diffs[250], diffs[9749]]
    print("SENSITIVITY")
    out["seed1"] = summarise("(a) seed 1 only", episode_D([p for p in pairs if p[0][1] == 1]))
    out["pairs_unit"] = summarise("(b) pairs as the unit", {k: y(t) - y(rc) for k, t, rc in pairs})
    out["no_yiliu"] = summarise("(c) without yiliu051016",
                                episode_D([p for p in pairs if eps[p[0][0]]["submitter"] != "yiliu051016"]))
    D = episode_D(pairs)
    per_sub = collections.defaultdict(dict)
    for ep, v in D.items():
        per_sub[eps[ep]["submitter"]][ep] = v
    means = {s: statistics.mean(v.values()) for s, v in per_sub.items()}
    rnd = random.Random(0)
    bs = []
    for _ in range(10000):
        m = []
        for s, v in per_sub.items():
            ks = list(v)
            m.append(statistics.mean(v[rnd.choice(ks)] for _ in ks))
        bs.append(statistics.mean(m))
    bs.sort()
    print(f"  (d) mean of per-submitter means: {statistics.mean(means.values()):.3f} [{bs[250]:.3f}, {bs[9749]:.3f}]; "
          + ", ".join(f"{s} {m:.2f} (n={len(per_sub[s])})" for s, m in sorted(means.items(), key=lambda x: -len(per_sub[x[0]]))))
    out["submitter_balanced"] = [statistics.mean(means.values()), bs[250], bs[9749]]
    out["per_submitter"] = {s: [m, len(per_sub[s])] for s, m in means.items()}
    out["nonconstant"] = summarise("(e) constant answers not counted as a fix", episode_D(pairs, nonconstant=True))
    print("SECONDARY OUTCOMES (pairs)")
    for arm, idx in (("traceback", 1), ("receipt", 2)):
        tin = statistics.mean(p[idx]["usage"]["prompt_tokens"] for p in pairs)
        tout = statistics.mean(p[idx]["usage"]["completion_tokens"] for p in pairs)
        cost = statistics.mean(p[idx].get("cost_usd") or 0 for p in pairs)
        print(f"  {arm:9s} tokens in {tin:,.0f}, out {tout:,.0f}; cost ${cost:.4f} per arm")
    mism = [p[1].get("prefix_mismatch", 0) for p in pairs]
    print(f"  replay: earlier cells whose outcome differs from the recording: mean {statistics.mean(mism):.2f}, max {max(mism)}")
    if a.transcripts:
        fnf = collections.Counter()
        for (ep, seed), t, rc in pairs:
            e = eps[ep]
            dirs = {os.path.dirname(os.path.normpath(p if os.path.isabs(p) else os.path.join("/work", p)))
                    for p in (e.get("sandbox_paths") or e.get("context_paths") or {})} | {"/work/data/context"}
            for arm, r in (("traceback", t), ("receipt", rc)):
                tp = os.path.join(a.transcripts, f"{ep}_{arm}_{seed}.json")
                if not os.path.exists(tp):
                    continue
                msgs = json.load(open(tp))
                k = 2 + 2 * len(e["prefix"]) + 2
                for s in r.get("steps") or []:
                    if k >= len(msgs):
                        break
                    obs = msgs[k + 1]["content"] if k + 1 < len(msgs) else ""
                    k += 2
                    if s.get("error_type") != "FileNotFoundError":
                        continue
                    m = re.search(r"No such file or directory: '([^']+)'", obs)
                    if not m:
                        fnf[(arm, "no path")] += 1
                        continue
                    pth = os.path.normpath(m.group(1) if os.path.isabs(m.group(1)) else os.path.join("/work", m.group(1)))
                    ctx = os.path.basename(pth) in {"payments.csv", "payments-readme.md", "acquirer_countries.csv",
                                                    "fees.json", "merchant_category_codes.csv", "merchant_data.json",
                                                    "manual.md"}
                    fnf[(arm, ("layout" if os.path.dirname(pth) in dirs else "other_dir") if ctx else "not_context")] += 1
        print(f"  FileNotFoundError in continuation cells by class: {dict(sorted(fnf.items()))}")
        out["fnf"] = {f"{k[0]}:{k[1]}": v for k, v in fnf.items()}
    cost = sum(r.get("cost_usd") or 0 for r in rows)
    print(f"cost: ${cost:.2f} for {len(rows)} episode-arms; served {dict(collections.Counter(r.get('served_model') for r in rows))}")
    out["cost"] = cost
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
