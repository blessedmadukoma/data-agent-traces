"""Pre-registered analysis of the DAB harness rerun (H1-H5). Writes analysis.json and analysis.md.

Pre-registration: claude/2026-10-06e-preregistration-dab-harness-rerun.md.
"""
import ast, collections, json, random, re, sys, warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("/home/claude/dabx")
OUT = Path("/home/claude/dabx-runs")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "failure_analysis"))
from rerun import run_dir, is_infra, DATASETS, MONGO_DATASETS
SEED, NBOOT = 20261006, 10000
SYNTAX = ("SyntaxError", "IndentationError", "TabError")
PRICE_IN, PRICE_OUT = 0.15, 0.60


# ---------- harness model (same rule as the paper's dab_harness_count.py) ----------
def compiles(src):
    try:
        compile(src, "<code>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def decode_arm_a(code):
    wrapped = f'code = """{code}"""\n\nenv_args = {{}}\n\nexec(code, env_args)\n'
    try:
        tree = ast.parse(wrapped)
    except (SyntaxError, ValueError):
        return None
    first = tree.body[0] if len(tree.body) == 3 else None
    if (isinstance(first, ast.Assign) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return first.value.value
    return None


def string_constants(src):
    try:
        return [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    except Exception:
        return None


# ---------- data ----------
def load_runs():
    final = {}
    for line in open(OUT / "runs.jsonl"):
        r = json.loads(line)
        final[(r["arm"], r["dataset"], r["query"], r["block"])] = r
    kept, excluded = {}, collections.Counter()
    for k, r in final.items():
        tr = r["terminate_reason"] or ""
        if is_infra(tr):
            excluded[(r["arm"], "infrastructure")] += 1
        elif "budget cap" in tr:
            excluded[(r["arm"], "budget")] += 1
        else:
            kept[k] = r
    return kept, excluded


def tool_calls(key, r):
    p = run_dir(*key, r["attempt"]) / "tool_calls.jsonl"
    calls = []
    if p.exists():
        for line in open(p):
            try:
                calls.append(json.loads(line))
            except Exception:
                pass
    return calls


def call_info(arm, c):
    res = c.get("result") or {}
    ok = res.get("success") is True
    info = {"tool": c.get("tool_name"), "ok": ok, "syntax": False, "harness": False, "silent": False}
    if not ok:
        info["syntax"] = any(s in str(res.get("preview", "")) + str(res.get("error", "")) for s in SYNTAX)
    if c.get("tool_name") != "execute_python":
        return info
    code = (c.get("val_args") or {}).get("code") if isinstance(c.get("val_args"), dict) else None
    if not isinstance(code, str):
        return info
    if arm == "A":
        dec = decode_arm_a(code)
        if not ok and info["syntax"] and compiles(code) and not (dec is not None and compiles(dec)):
            info["harness"] = True
        if ok and compiles(code) and dec is not None and compiles(dec) and string_constants(code) != string_constants(dec):
            info["silent"] = True
    else:  # arms B and C pass the code unchanged; a syntax error on code that compiles would be an anomaly
        if not ok and info["syntax"] and compiles(code):
            info["harness"] = True
    return info


# ---------- statistics ----------
def pass_rates(runs, arm, datasets):
    pr = {}
    for ds in datasets:
        for q in range(1, DATASETS[ds] + 1):
            v = [r["is_valid"] for (a, d, qq, k), r in runs.items() if a == arm and d == ds and qq == q]
            if v:
                pr[(ds, q)] = float(np.mean(v))
    return pr


def dab_pass1(pr, datasets, queries=None):
    per_ds = []
    for ds in datasets:
        qs = [q for (d, q) in (queries or pr.keys()) if d == ds and (d, q) in pr]
        if qs:
            per_ds.append(np.mean([pr[(ds, q)] for q in qs]))
    return float(np.mean(per_ds))


def boot_diff(pr_x, pr_y, datasets, seed=SEED, n=NBOOT):
    """Pass@1(x) - Pass@1(y), DAB definition; queries resampled within dataset, paired."""
    rng = random.Random(seed)
    common = [k for k in pr_x if k in pr_y]
    by_ds = collections.defaultdict(list)
    for k in common:
        by_ds[k[0]].append(k)
    est = dab_pass1({k: pr_x[k] for k in common}, datasets) - dab_pass1({k: pr_y[k] for k in common}, datasets)
    vals = []
    for _ in range(n):
        x, y = [], []
        for ds in datasets:
            ks = by_ds.get(ds)
            if not ks:
                continue
            s = [rng.choice(ks) for _ in ks]
            x.append(np.mean([pr_x[k] for k in s]))
            y.append(np.mean([pr_y[k] for k in s]))
        vals.append(np.mean(x) - np.mean(y))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    diffs = [pr_x[k] - pr_y[k] for k in common]
    nz = [d for d in diffs if d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else float("nan")
    return {"estimate": est, "ci95": [float(lo), float(hi)], "wilcoxon_p": p, "n_queries": len(common),
            "n_nonzero": len(nz), "equal_weight_mean_diff": float(np.mean(diffs))}


def boot_metric(per_query_x, per_query_y, seed=SEED, n=NBOOT):
    """Difference in the mean of a per-query metric (x - y), queries resampled, paired."""
    rng = random.Random(seed)
    common = [k for k in per_query_x if k in per_query_y]
    d = np.array([per_query_x[k] - per_query_y[k] for k in common])
    vals = [np.mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(n)]
    nz = [v for v in d if v != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else float("nan")
    return {"estimate": float(d.mean()), "ci95": [float(x) for x in np.percentile(vals, [2.5, 97.5])],
            "wilcoxon_p": p, "n_queries": len(common)}


def parse_judge(text):
    labels = {}
    for i in range(1, 5):
        m = re.search(rf"FM{i}\s*[:]?\s*(yes|no)", text or "", re.I)
        labels[f"FM{i}"] = (m.group(1).lower() == "yes") if m else None
    return labels


def main():
    runs, excluded = load_runs()
    all_ds = list(DATASETS)
    res = {"excluded": {f"{a}:{why}": n for (a, why), n in excluded.items()}}

    # H1 and H3 from the tool calls
    per_run = {}
    for key, r in runs.items():
        infos = [call_info(key[0], c) for c in tool_calls(key, r)]
        py = [i for i in infos if i["tool"] == "execute_python"]
        failed = [i for i in infos if not i["ok"]]
        per_run[key] = {"python_calls": len(py), "harness": sum(i["harness"] for i in py),
                        "silent": sum(i["silent"] for i in py), "failed": len(failed),
                        "failed_syntax": sum(i["syntax"] for i in failed)}
    h1 = {}
    for arm in ("A", "B", "C"):
        pr = [v for k, v in per_run.items() if k[0] == arm]
        if not pr:
            continue
        pc = sum(v["python_calls"] for v in pr)
        h1[arm] = {"runs": len(pr), "python_calls": pc, "harness_caused": sum(v["harness"] for v in pr),
                   "harness_per_python_call": sum(v["harness"] for v in pr) / max(pc, 1),
                   "runs_with_harness_caused": sum(v["harness"] > 0 for v in pr),
                   "silent_changes": sum(v["silent"] for v in pr)}
    res["H1"] = h1

    # H2 primary
    prA, prB = pass_rates(runs, "A", all_ds), pass_rates(runs, "B", all_ds)
    res["pass1"] = {"A": dab_pass1(prA, all_ds), "B": dab_pass1(prB, all_ds)}
    res["H2_B_minus_A"] = boot_diff(prB, prA, all_ds)

    # H3
    def per_query(arm, field, denom=None):
        acc = collections.defaultdict(list)
        for (a, ds, q, k), v in per_run.items():
            if a != arm:
                continue
            if denom:
                if v[denom] > 0:
                    acc[(ds, q)].append(v[field] / v[denom])
            else:
                acc[(ds, q)].append(v[field])
        return {k: float(np.mean(v)) for k, v in acc.items()}
    res["H3_failed_calls_B_minus_A"] = boot_metric(per_query("B", "failed"), per_query("A", "failed"))
    res["H3_syntax_share_B_minus_A"] = boot_metric(per_query("B", "failed_syntax", "failed"),
                                                   per_query("A", "failed_syntax", "failed"))

    # H4 judge
    jp = OUT / "judge.jsonl"
    if jp.exists():
        judged = {}
        for line in open(jp):
            j = json.loads(line)
            if j.get("judge"):
                judged[(j["arm"], j["dataset"], j["query"], j["block"])] = (parse_judge(j["judge"]), j["terminate_reason"])
        h4 = {}
        for scope in ("all_failed", "return_answer_only"):
            share = {}
            for arm in ("A", "B"):
                rows = [(k, lab) for k, (lab, tr) in judged.items() if k[0] == arm and lab["FM4"] is not None
                        and (scope == "all_failed" or tr == "return_answer")]
                share[arm] = {"n": len(rows), "fm4": sum(l["FM4"] for _, l in rows),
                              "share_fm4": (sum(l["FM4"] for _, l in rows) / len(rows)) if rows else None}
            # per-query FM4 share, A - B, queries resampled
            pq = {}
            for arm in ("A", "B"):
                acc = collections.defaultdict(list)
                for (a, ds, q, k), (lab, tr) in judged.items():
                    if a == arm and lab["FM4"] is not None and (scope == "all_failed" or tr == "return_answer"):
                        acc[(ds, q)].append(lab["FM4"])
                pq[arm] = {k: float(np.mean(v)) for k, v in acc.items()}
            share["A_minus_B_per_query"] = boot_metric(pq["A"], pq["B"]) if pq["A"] and pq["B"] else None
            h4[scope] = share
        res["H4"] = h4

    # H5 exploratory
    prC = pass_rates(runs, "C", list(MONGO_DATASETS))
    prBm = {k: v for k, v in prB.items() if k[0] in MONGO_DATASETS}
    if prC:
        res["H5_C_minus_B_mongo"] = boot_diff(prC, prBm, list(MONGO_DATASETS))
        notes = limit_after = 0
        for key, r in runs.items():
            if key[0] != "C":
                continue
            fa = run_dir(*key, r["attempt"]) / "final_agent.json"
            if not fa.exists():
                continue
            msgs = json.load(open(fa))["messages"]
            for i, m in enumerate(msgs):
                if m["role"] == "tool" and "Note: this query set no" in str(m.get("content")):
                    notes += 1
                    later = [c for mm in msgs[i + 1:] if mm["role"] == "assistant"
                             for c in (mm.get("tool_calls") or []) if c["function"]["name"] == "query_db"]
                    if any('\\"limit\\"' in c["function"]["arguments"] or '"limit"' in c["function"]["arguments"] for c in later):
                        limit_after += 1
        res["H5_notes"] = {"notes_shown": notes, "followed_by_query_with_limit": limit_after}

    # cost and tokens
    cost = {}
    for arm in ("A", "B", "C"):
        rr = [r for k, r in runs.items() if k[0] == arm]
        if rr:
            pt, ct = sum(r["prompt_tokens"] for r in rr), sum(r["completion_tokens"] for r in rr)
            cost[arm] = {"runs": len(rr), "prompt_tokens": pt, "completion_tokens": ct,
                         "usd": pt / 1e6 * PRICE_IN + ct / 1e6 * PRICE_OUT,
                         "terminate": dict(collections.Counter(r["terminate_reason"] for r in rr))}
    res["runs_by_arm"] = cost
    res["ledger_usd"] = json.load(open(OUT / "ledger.json"))["cost_usd"]
    json.dump(res, open(OUT / "analysis.json", "w"), indent=2, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
