#!/usr/bin/env python3
"""Run a code agent on QRData and record every cell: third held-out test of the
frozen gate v7.

The harness is infiagent/ia_run.py (the kb_run.py pattern) with the 411 QRData
questions. The user message gives the data description, the question, the
answer options for a multiple-choice question, and the question's data files.
The data directory (data.zip unpacked) is mounted read-only at /work/input.
Every cell runs; the gate is not used during the run.

kb_gate.py scores the cells unchanged: --kb points at the directory whose
data/qrdata/input is a link to the tables; every record has domain "qrdata".

Usage:
  python qrdata/qr_run.py --tables $QRDATA_TABLES \
      --questions .../benchmark/QRData.json --out runs/qr/results_gpt-oss.jsonl \
      --transcripts runs/qr/transcripts_gpt-oss --model gpt-oss:120b --workers 2 --budget-usd 1.8
"""
import argparse
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "infiagent"))
import ia_run  # noqa: E402  (sets kb_run.SYSTEM to the InfiAgent-DABench system prompt)

kb_run = ia_run.kb_run


def load_tasks(path):
    out = []
    for i, q in enumerate(json.load(open(path))):
        md = q.get("meta_data") or {}
        parts = [f"Data description: {q['data_description']}", f"Question: {q['question']}"]
        if md.get("question_type") == "multiple_choice" and md.get("multiple_choices"):
            parts.append("Options: " + "; ".join(str(c) for c in md["multiple_choices"])
                         + "\nAnswer with the text of one option.")
        parts.append("Data files: " + ", ".join(f"input/{f}" for f in q["data_files"]))
        out.append({"domain": "qrdata", "id": f"qr-{i}", "query": "\n\n".join(parts),
                    "data_files": q["data_files"], "question_type": md.get("question_type")})
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
    root = os.path.join(os.path.dirname(os.path.abspath(a.out)), "qr_kb")
    os.makedirs(os.path.join(root, "data", "qrdata"), exist_ok=True)
    link = os.path.join(root, "data", "qrdata", "input")
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
        rec.update(data_files=t["data_files"], question_type=t["question_type"])
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
