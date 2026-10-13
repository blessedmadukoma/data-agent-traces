"""Analysis of the gpt-5-mini DAB rerun (pre-registration: dab-rerun-mini/preregistration.md, section 5).

1. The frozen analysis of the first rerun (dab-rerun/analysis.py), unchanged, on the complete blocks of this rerun:
   H1 (harness-caused syntax errors per Python call), Pass@1 B - A, failed calls per run B - A, cost.
   Only its paths and prices change. Writes OUT/analysis.json.
2. Added here (section 5, H3 and H5): the subtraction check, B's failed calls per run against A's failed calls
   minus A's harness-caused calls (per task, tasks resampled), and requests to the model per run, B - A.
   Writes OUT/analysis_mini.json.

Usage: OPENAI_API_KEY is not needed. python3 dab-rerun-mini/analysis_mini.py   (needs numpy and scipy)
"""
import collections, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mini = load("rerun", HERE / "rerun.py")
mini.MONGO_DATASETS = ("yelp", "agnews")       # analysis.py imports it; this rerun has no arm C
sys.modules["rerun"] = mini
an = load("analysis_frozen", REPO / "dab-rerun" / "analysis.py")
an.OUT = mini.OUT
an.PRICE_IN, an.PRICE_OUT = 0.25, 2.00

_load_all = an.load_runs


def load_complete():
    """Only blocks in which at least 100 of the 108 jobs have a kept run (pre-registration section 4)."""
    kept, excluded = _load_all()
    n = collections.Counter(k[3] for k in kept)
    complete = sorted(b for b, c in n.items() if c >= 100)
    load_complete.blocks = complete
    return {k: v for k, v in kept.items() if k[3] in complete}, excluded


an.load_runs = load_complete


def main():
    print("== frozen analysis on the complete blocks")
    an.main()
    runs, _ = load_complete()
    per_run = {}
    for key, r in runs.items():
        infos = [an.call_info(key[0], c) for c in an.tool_calls(key, r)]
        per_run[key] = {"failed": sum(not i["ok"] for i in infos),
                        "harness": sum(i["harness"] for i in infos if i["tool"] == "execute_python"),
                        "requests": r.get("llm_calls") or 0}

    def per_task(arm, f):
        acc = collections.defaultdict(list)
        for (a, ds, q, k), v in per_run.items():
            if a == arm:
                acc[(ds, q)].append(f(v))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    out = {"complete_blocks": load_complete.blocks}
    b_failed = per_task("B", lambda v: v["failed"])
    a_sub = per_task("A", lambda v: v["failed"] - v["harness"])
    out["H3_B_failed_minus_A_subtraction"] = an.boot_metric(b_failed, a_sub)
    out["A_failed_per_run"] = float(np.mean(list(per_task("A", lambda v: v["failed"]).values())))
    out["A_harness_per_run"] = float(np.mean(list(per_task("A", lambda v: v["harness"]).values())))
    out["A_subtraction_per_run"] = float(np.mean(list(a_sub.values())))
    out["B_failed_per_run"] = float(np.mean(list(b_failed.values())))
    out["H5_requests_B_minus_A"] = an.boot_metric(per_task("B", lambda v: v["requests"]),
                                                  per_task("A", lambda v: v["requests"]))
    out["runs_with_harness_caused_A"] = sum(1 for k, v in per_run.items() if k[0] == "A" and v["harness"] > 0)
    out["runs_A"] = sum(1 for k in per_run if k[0] == "A")
    json.dump(out, open(mini.OUT / "analysis_mini.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
