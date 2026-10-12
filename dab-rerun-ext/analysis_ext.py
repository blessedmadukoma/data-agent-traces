"""Analysis of the extension (pre-registration section 4).

1. The frozen analysis of the first rerun (dab-rerun/analysis.py), unchanged, on blocks 5 to 9: H1, H2
   (replication), H3 and H4a. Only its paths change: OUT is the extension's folder, and `rerun` is the
   extension's runner. Writes OUT/analysis.json.
2. Pooled over blocks 0 to 9 (arms A and B): H2, with the same functions (boot_diff), and failed tool calls
   per run (H3), from the per-run records of both reruns (dab-rerun/runs.jsonl and OUT/runs.jsonl).
   Writes OUT/analysis_pooled.json.

Usage: python3 dab-rerun-ext/analysis_ext.py   (needs numpy and scipy)
"""
import collections, importlib.util, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ext = load("rerun", HERE / "rerun.py")
ext.MONGO_DATASETS = ("yelp", "agnews")          # analysis.py imports it; arm C has no runs here
sys.modules["rerun"] = ext
an = load("analysis_frozen", REPO / "dab-rerun" / "analysis.py")
an.OUT = ext.OUT
an.PRICE_IN, an.PRICE_OUT = 0.15, 0.60


def final_runs(path):
    final = {}
    for line in open(path):
        r = json.loads(line)
        final[(r["arm"], r["dataset"], r["query"], r["block"])] = r
    kept, excluded = {}, collections.Counter()
    for k, r in final.items():
        tr = r["terminate_reason"] or ""
        if k[0] not in ("A", "B"):
            continue
        if ext.is_infra(tr):
            excluded[(k[0], "infrastructure")] += 1
        elif "budget cap" in tr:
            excluded[(k[0], "budget")] += 1
        else:
            kept[k] = r
    return kept, excluded


def main():
    print("== Extension, blocks 5 to 9 (frozen analysis)")
    an.main()
    first, ex1 = final_runs(REPO / "dab-rerun" / "runs.jsonl")
    second, ex2 = final_runs(ext.OUT / "runs.jsonl")
    blocks2 = collections.Counter(k[3] for k in second)
    complete = sorted(b for b, n in blocks2.items() if n >= 100)   # 108 jobs per block, minus exclusions
    second = {k: v for k, v in second.items() if k[3] in complete}
    runs = {**first, **second}
    all_ds = list(ext.DATASETS)
    prA, prB = an.pass_rates(runs, "A", all_ds), an.pass_rates(runs, "B", all_ds)
    res = {"blocks_first": sorted({k[3] for k in first}), "blocks_extension": complete,
           "runs": {arm: sum(1 for k in runs if k[0] == arm) for arm in ("A", "B")},
           "excluded": {f"{a}:{w}": n for (a, w), n in (ex1 + ex2).items()},
           "pass1": {"A": an.dab_pass1(prA, all_ds), "B": an.dab_pass1(prB, all_ds)},
           "H2_B_minus_A": an.boot_diff(prB, prA, all_ds)}

    def per_query(arm):
        acc = collections.defaultdict(list)
        for (a, ds, q, k), r in runs.items():
            if a == arm and r.get("failed_tool_calls") is not None:
                acc[(ds, q)].append(r["failed_tool_calls"])
        return {k: sum(v) / len(v) for k, v in acc.items()}
    res["H3_failed_calls_B_minus_A"] = an.boot_metric(per_query("B"), per_query("A"))
    json.dump(res, open(ext.OUT / "analysis_pooled.json", "w"), indent=2, default=str)
    print("== Pooled, blocks 0 to 9")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
