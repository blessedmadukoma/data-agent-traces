"""H4b: draw the 40 failed runs for the human check of DAB's judge, and write the items for the private page.

Sample (pre-registration section 4): failed runs of blocks 5 to 9 that end with an answer (terminate reason
return_answer) and that the judge labelled, 20 per arm, drawn at random with seed 20261012, then shown in one
random order. Each item holds the query, the ground-truth answer, the agent's final answer and the trace in
DAB's judge format (failure_analysis/get_trace.format_trace). To keep the coder blind to the arm, the one clause
of key rule 7 that differs between the arms is replaced in the system message by a neutral clause; the rest of
the system message (schemas and hints, which the coder needs) stays.

Writes OUT/judge_check/items.json (for the page; holds DAB data, so it stays private) and
OUT/judge_check/key_SEALED.json (item -> arm, dataset, query, block, and the judge's labels; not for the coder).

Usage: /home/claude/dab-venv/bin/python dab-rerun-ext/judge_check_items.py
"""
import importlib.util, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rerun", HERE / "rerun.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
sys.path.insert(0, str(ext.ROOT))
sys.path.insert(0, str(ext.ROOT / "failure_analysis"))
from get_trace import format_trace  # noqa: E402

CLAUSES = ("your code will be quoted by triple double-quotes and passed as a string to `exec(...)`",
           "your code will be passed unchanged as a string to `exec(...)`")
NEUTRAL = "your code will be passed as a string to `exec(...)`"


def parse(text):
    import re
    out = {}
    for i in range(1, 5):
        m = re.search(rf"FM{i}\s*[:]?\s*(yes|no)", text or "", re.I)
        out[f"FM{i}"] = (m.group(1).lower() == "yes") if m else None
    return out


def main():
    judged = {}
    for line in open(ext.OUT / "judge.jsonl"):
        j = json.loads(line)
        if j.get("judge"):
            judged[(j["arm"], j["dataset"], j["query"], j["block"])] = j
    final = {}
    for line in open(ext.OUT / "runs.jsonl"):
        r = json.loads(line)
        final[(r["arm"], r["dataset"], r["query"], r["block"])] = r
    pool = {"A": [], "B": []}
    for k, r in sorted(final.items()):
        if (r["terminate_reason"] == "return_answer" and not r["is_valid"] and k in judged
                and parse(judged[k]["judge"])["FM4"] is not None):
            pool[k[0]].append(k)
    rng = random.Random(20261012)
    chosen = rng.sample(pool["A"], 20) + rng.sample(pool["B"], 20)
    rng.shuffle(chosen)
    items, key = [], {}
    for i, k in enumerate(chosen, 1):
        arm, ds, q, b = k
        r = final[k]
        rd = ext.run_dir(arm, ds, q, b, r["attempt"])
        msgs = json.load(open(rd / "final_agent.json"))["messages"]
        for m in msgs:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                for c in CLAUSES:
                    m["content"] = m["content"].replace(c, NEUTRAL)
        qdir = ext.ROOT / f"query_{ds}" / f"query{q}"
        query = json.load(open(qdir / "query.json", encoding="utf-8"))
        items.append({"id": i, "dataset": ds, "query": query if isinstance(query, str) else json.dumps(query),
                      "ground_truth": (qdir / "ground_truth.csv").read_text(encoding="utf-8").strip(),
                      "final_answer": r.get("final_result"), "trace": format_trace(msgs)})
        key[i] = {"arm": arm, "dataset": ds, "query": q, "block": b, "judge": parse(judged[k]["judge"]),
                  "judge_text": judged[k]["judge"]}
    out = ext.OUT / "judge_check"
    out.mkdir(exist_ok=True)
    json.dump(items, open(out / "items.json", "w"))
    json.dump(key, open(out / "key_SEALED.json", "w"), indent=1)
    assert not any(c in it["trace"] for it in items for c in CLAUSES)
    print(f"pool: A {len(pool['A'])}, B {len(pool['B'])}; wrote {len(items)} items")


if __name__ == "__main__":
    main()
