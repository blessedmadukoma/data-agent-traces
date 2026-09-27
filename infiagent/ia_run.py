#!/usr/bin/env python3
"""Run a code agent on InfiAgent-DABench (dev set) and record every cell: second
held-out test of the gate (gate v7, frozen before this run).

The harness is kb_run.py (KramaBench) with three changes: the tasks are the 257
dev questions; the prompt gives the question, its constraints, the answer format
and the question's data file; and the sandbox Python has scikit-learn,
statsmodels and matplotlib, which the questions need. The data directory
da-dev-tables is mounted read-only at /work/input. Every cell runs; the gate is
not used during the run. The workspace is listed before each cell.

kb_gate.py scores the cells unchanged: --kb points at a directory whose
data/infiagent/input is (a link to) da-dev-tables, and every record has
domain "infiagent".

Usage (with the InfiAgent sandbox environment):
  python infiagent/ia_run.py --tables .../da-dev-tables \
      --questions .../da-dev-questions.jsonl --out runs/ia/results_gpt-oss.jsonl \
      --transcripts runs/ia/transcripts_gpt-oss --model gpt-oss:120b --workers 2 --budget-usd 2
"""
import argparse
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kramabench"))
import kb_run  # noqa: E402

kb_run.SYSTEM = (
    "You are a data analysis agent. You answer one question using the data files in the directory input/ "
    "(read-only). You work in a Python session: names persist between cells. Available packages: pandas, "
    "numpy, scipy, scikit-learn, statsmodels, matplotlib. You may write files in the current directory. In each "
    "reply, think briefly, then write exactly one Python code block in ```python ... ```. Its printed output "
    "comes back to you. When you know the answer, call final_answer(answer) in a code block, with the answer "
    "as a string in the required format.")


def load_tasks(path):
    out = []
    for line in open(path):
        q = json.loads(line)
        query = (f"{q['question']}\n\nConstraints: {q['constraints']}\n\nAnswer format: {q['format']}\n\n"
                 f"Data file: input/{q['file_name']}")
        out.append({"domain": "infiagent", "id": f"ia-{q['id']}", "query": query, "file_name": q["file_name"],
                    "level": q.get("level"), "concepts": q.get("concepts")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--cell-timeout", type=int, default=120)
    ap.add_argument("--sandbox", choices=["bwrap", "local"], default="bwrap")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=0)
    a = ap.parse_args()
    # kb_run.run_task finds the data at <kb>/data/<domain>/input
    root = os.path.join(os.path.dirname(os.path.abspath(a.out)), "ia_kb")
    os.makedirs(os.path.join(root, "data", "infiagent"), exist_ok=True)
    link = os.path.join(root, "data", "infiagent", "input")
    if not os.path.islink(link):
        os.symlink(os.path.abspath(a.tables), link)
    a.kb = root
    model = kb_run.ScriptedModel() if a.model == "dry" else kb_run.rr.Ollama(a.model, a.temperature, a.max_tokens)
    tasks = load_tasks(a.questions)
    if a.limit:
        tasks = tasks[: a.limit]
    done, spent = set(), [0.0]
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["task"], r["seed"]))
            spent[0] += r.get("cost_usd") or 0
    todo = [t for t in tasks if (t["id"], a.seed) not in done]
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor

    def job(t):
        if a.budget_usd and spent[0] >= a.budget_usd:
            return
        try:
            rec = kb_run.run_task(t, a.seed, model, a)
        except Exception as e:  # not written: the next run retries it
            print(f"{t['id']}: error {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        rec.update(file_name=t["file_name"], level=t["level"])
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            spent[0] += rec.get("cost_usd") or 0
        nfail = sum(1 for c in rec["cells"] if c.get("ok") is False)
        print(f"{t['id']}: {rec['status']}, cells {len(rec['cells'])}, failed {nfail}, "
              f"answer {str(rec['final_answer'])[:40]!r}, cost {rec.get('cost_usd')}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(job, todo))
    print(f"done; recorded cost ${spent[0]:.2f}")


if __name__ == "__main__":
    main()
