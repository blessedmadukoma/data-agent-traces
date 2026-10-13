#!/usr/bin/env python3
"""E3 analysis (live/e3_preregistration.md, section 5): per model and pooled over the models.

Per model, this runs the E2 analysis (live/e2_report.py) on that model's records. Pooled, the paired unit is
(question, model, seed): for each question, a difference is the mean over models and seeds; the bootstrap and the
permutation test resample questions, as in E2. Only seeds that are complete for every model are pooled.

Usage: python3 live/e3_report.py --questions data/QRData/benchmark/QRData.json \\
           --results runs/e3/gpt-oss-120b.jsonl runs/e3/gemma4-31b.jsonl [--validation ...] [--json out.json]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_report as r2  # noqa: E402


def load(path, jobs_per_seed):
    rows = [json.loads(line) for line in open(path)]
    per_seed = collections.Counter(r["seed"] for r in rows)
    seen = sorted(per_seed)
    complete = [s for s in seen if per_seed[s] >= jobs_per_seed or any(t > s for t in seen)]
    return rows, complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--validation", nargs="*", default=[])
    ap.add_argument("--json")
    ap.add_argument("--jobs-per-seed", type=int, default=1233)
    a = ap.parse_args()
    qsrc = json.load(open(a.questions))
    gold = {f"qr-{i}": (q["answer"], (q.get("meta_data") or {}).get("question_type")) for i, q in enumerate(qsrc)}
    loaded = {p: load(p, a.jobs_per_seed) for p in a.results}
    common = sorted(set.intersection(*[set(c) for _, c in loaded.values()]))
    out = {"complete_seeds_per_file": {p: c for p, (_, c) in loaded.items()}, "pooled_seeds": common, "per_model": {}}
    by = collections.defaultdict(dict)
    for p, (rows, complete) in loaded.items():
        model = rows[0]["model"] if rows else p
        sub = collections.defaultdict(dict)
        for r in rows:
            if r["seed"] not in complete:
                continue
            m = r2.run_measures(r, gold)
            sub[r["task"]].setdefault(r["arm"], {})[r["seed"]] = m
            if r["seed"] in common:
                by[r["task"]].setdefault(r["arm"], {})[(model, r["seed"])] = m
        qs = sorted(sub)
        res = {}
        print(f"==== {model}: complete seeds {complete}")
        for arm in r2.ARMS:
            xs = [v for t in qs for v in sub[t].get(arm, {}).values()]
            if xs:
                k = collections.Counter()
                for v in xs:
                    k.update(v["kinds"])
                res[arm] = {"runs": len(xs), "accuracy": sum(v["correct"] for v in xs) / len(xs),
                            "dra_share": sum(v["dra"] for v in xs) / len(xs), "kinds": dict(k),
                            "blocks": sum(v["blocks"] for v in xs), "lost_answers": sum(v["lost_answer"] for v in xs),
                            "executor_lost": sum(v["executor_lost"] for v in xs),
                            "steps_mean": sum(v["steps"] for v in xs) / len(xs),
                            "out_tokens_mean": sum(v["out_tokens"] for v in xs) / len(xs),
                            "next_step_ok_after_first_dra": [sum(1 for v in xs if v["after_ok"]),
                                                             sum(1 for v in xs if v["after_ok"] is not None)]}
                print(f"{arm}: {json.dumps(res[arm])}")
        r2.compare(sub, qs, "N", "R", "H1 checker", res)
        r2.compare(sub, qs, "N", "E", "H2 names in the error", res)
        r2.compare(sub, qs, "E", "R", "H3 checker against names", res)
        r2.dra_compare(sub, qs, "N", "R", "H5 checker", res)
        r2.dra_compare(sub, qs, "N", "E", "H5 names in the error", res)
        r2.dra_compare(sub, qs, "E", "R", "H5 checker against names", res)
        fs = sum(v["false_shadow"] for t in qs for arm in ("N", "E") for v in sub[t].get(arm, {}).values())
        ran = sum(v["shadow_steps_ran"] for t in qs for arm in ("N", "E") for v in sub[t].get(arm, {}).values())
        res["H6"] = {"false_shadow_blocks": fs, "steps_that_ran": ran, "upper95_per_step": r2.upper95(fs, ran)}
        print(f"H6: {fs} false shadow blocks of {ran} steps that ran (upper bound {100 * res['H6']['upper95_per_step']:.3f}%)")
        out["per_model"][model] = res
    print(f"==== pooled over models, seeds {common}")
    qs = sorted(by)
    pooled = {}
    r2.compare(by, qs, "N", "R", "H1 checker (primary)", pooled)
    r2.compare(by, qs, "N", "E", "H2 names in the error", pooled)
    r2.compare(by, qs, "E", "R", "H3 checker against names", pooled)
    r2.dra_compare(by, qs, "N", "R", "H5 checker", pooled)
    r2.dra_compare(by, qs, "N", "E", "H5 names in the error", pooled)
    out["pooled"] = pooled
    for v in a.validation:
        rows = [json.loads(line) for line in open(v)]
        cls = collections.Counter(x["class"] for x in rows)
        out.setdefault("block_check", {})[v] = dict(cls)
        print(f"block check {v}: {dict(cls)}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
