#!/usr/bin/env python3
"""Build the recorded runs as a harness with a protected final_answer would have recorded them.

Input: the recorded runs (recorded/) and the replays of live/fa_replay.py (one file per tag).
For each replayed run in which the harness now records an answer that the recording lacks,
the run ends at that cell and the answer is scored; the later cells are removed. All other
runs are copied unchanged. The dry-run and LLM-checker records are cut to the same cells.
The output folder has the layout of recorded/, so tools/rescore.sh scores it unchanged:

  uv run live/fa_rescore.py --rec recorded --replays runs/fa_replay --out runs/fa_recorded
  REC=runs/fa_recorded OUT=runs/rescore_fa sh tools/rescore.sh

It also prints, per run file, how many runs the effect touched and how well the replay
reproduced the recorded outcome of each cell before the answer.
"""
import argparse
import ast
import collections
import json
import os
import shutil
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)   # from parsing the models' code

RUNS = {"live_gpt": "e1/results.jsonl", "live_ds": "e1/results_deepseek41.jsonl",
        "ia_gpt": "ia/results_gpt-oss.jsonl", "ia_ds": "ia/results_deepseek41.jsonl",
        "qr_gpt": "qr/results_gpt-oss.jsonl", "db_gpt": "db/results_gpt-oss.jsonl"}
BASELINE_TAG = {"ia_gpt-oss": "ia_gpt", "ia_deepseek41": "ia_ds", "qr_gpt-oss": "qr_gpt", "db_gpt-oss": "db_gpt"}


def defines_final_answer(code):
    try:
        t = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "final_answer"
               or isinstance(n, ast.Name) and n.id == "final_answer" and isinstance(n.ctx, ast.Store)
               for n in ast.walk(t))


def counterfactual(recs, replays):
    out, st = [], collections.Counter()
    for i, r in enumerate(recs):
        st["runs"] += 1
        arm = r.get("arm")
        if arm:
            st[f"runs {arm}"] += 1
        cells = [c for c in r["cells"] if "code" in c and not c.get("no_code") and not c.get("blocked")]
        if any(defines_final_answer(c["code"]) for c in cells):
            st["model_defined_final_answer"] += 1
            if arm:
                st[f"defined {arm}"] += 1
        if r.get("final_answer") is None:
            st["no_recorded_answer"] += 1
        o = replays.get(i)
        r2 = dict(r)
        if o is not None:
            st["replayed"] += 1
            stop = o["answer_cell"] if o["answer"] is not None else None
            diverged = False
            for c in o["cells"]:
                if stop is not None and c["cell"] >= stop:
                    break
                rc = r["cells"][c["cell"]]
                same = (bool(rc.get("ok")) == c["ok"] and (rc.get("error_type") or None) == (c["error_type"] or None))
                st["cells_compared"] += 1
                st["cells_same_outcome"] += same
                diverged = diverged or not same
            if diverged and o["answer"] is not None:
                st["answer_not_used_replay_diverged"] += 1   # the model saw other outputs: keep the recording
            elif o["answer"] is not None and (r.get("final_answer") is None or str(o["answer"]) != str(r["final_answer"])):
                removed = [c for c in r["cells"][stop + 1:] if "code" in c and not c.get("no_code")]
                r2["cells"] = r["cells"][:stop + 1]
                r2["final_answer"] = o["answer"]
                st["answer_recovered" if r.get("final_answer") is None else "answer_changed"] += 1
                if arm and r.get("final_answer") is None:
                    st[f"recovered {arm}"] += 1
                st["cells_removed"] += len(removed)
        out.append(r2)
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--replays", required=True, help="folder with <tag>.jsonl from live/fa_replay.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    shutil.copytree(a.rec, a.out)
    kept = {}
    for tag, rel in RUNS.items():
        recs = [json.loads(line) for line in open(os.path.join(a.rec, rel))]
        rp = {}
        path = os.path.join(a.replays, f"{tag}.jsonl")
        if os.path.exists(path):
            for line in open(path):
                o = json.loads(line)
                rp[int(o["key"].split(":")[1])] = o
        cf, st = counterfactual(recs, rp)
        with open(os.path.join(a.out, rel), "w") as f:
            for r in cf:
                f.write(json.dumps(r) + "\n")
        kept[tag] = {(r["task"], c["step"]) for r in cf for c in r["cells"]}
        same = f"{st['cells_same_outcome']}/{st['cells_compared']}"
        print(f"{tag}: runs {st['runs']}, no recorded answer {st['no_recorded_answer']}, model defined or assigned "
              f"final_answer {st['model_defined_final_answer']}, replayed {st['replayed']}, answer recovered "
              f"{st['answer_recovered']}, answer changed {st['answer_changed']}, not used because an earlier cell diverged "
              f"{st['answer_not_used_replay_diverged']}, cells removed {st['cells_removed']}; "
              f"replay reproduced the recorded outcome of {same} earlier cells")
        arms = sorted(k.split()[1] for k in st if k.startswith("runs "))
        for arm in arms:
            print(f"  {arm}: runs {st['runs ' + arm]}, model defined or assigned final_answer {st['defined ' + arm]}, "
                  f"answer recovered {st['recovered ' + arm]}")
    for sub in ("dry", "llm_check"):
        for fn in sorted(os.listdir(os.path.join(a.rec, sub))):
            stem = fn[:-6] if sub == "llm_check" else fn.rsplit("_", 1)[0]
            tag = BASELINE_TAG[stem]
            rows = [json.loads(line) for line in open(os.path.join(a.rec, sub, fn))]
            with open(os.path.join(a.out, sub, fn), "w") as f:
                for row in rows:
                    if (row["task"], row["step"]) in kept[tag]:
                        f.write(json.dumps(row) + "\n")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
