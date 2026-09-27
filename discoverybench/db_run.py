#!/usr/bin/env python3
"""E2: run a code agent on the real DiscoveryBench tasks and record every cell: held-out
test of the frozen gate v8.

Harness: the kb_run.py pattern, as in qrdata/qr_run.py. Each task's sandbox mounts only
the data files of its dataset folder (not the metadata files, which hold the gold
hypotheses) read-only at input/. The prompt gives the dataset descriptions, the query,
and the data file names. Every cell runs; the gate is not used during the run.

Each dataset folder is its own kb_gate "domain" (<split>__<dataset>), because a base
name can occur in two folders (nls_raw.csv). The script builds <out dir>/db_kb/data/
<domain>/input with copies of the data files, so kb_gate.py scores the cells unchanged.

Usage:
  python discoverybench/db_run.py --real .../discoverybench/real \
      --out runs/db/results_gpt-oss.jsonl --transcripts runs/db/transcripts_gpt-oss \
      --model gpt-oss:120b --workers 2 --budget-usd 1.5
"""
import argparse
import glob
import json
import os
import shutil
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "infiagent"))
import ia_run  # noqa: E402  (sets the system prompt used for InfiAgent-DABench and QRData)

kb_run = ia_run.kb_run


def load_tasks(real, kb_root):
    out = []
    for split in ("train", "test"):
        for f in sorted(glob.glob(os.path.join(real, split, "*", "metadata_*.json")),
                        key=lambda p: (os.path.dirname(p), int(p.rsplit("_", 1)[1].split(".")[0]))):
            d = json.load(open(f))
            folder = os.path.dirname(f)
            dom = f"{split}__{os.path.basename(folder)}"
            lake = os.path.join(kb_root, "data", dom, "input")
            os.makedirs(lake, exist_ok=True)
            names = [ds["name"] for ds in d["datasets"]]
            for n in names:
                tgt = os.path.join(lake, n)
                if not os.path.exists(tgt):
                    os.makedirs(os.path.dirname(tgt), exist_ok=True)
                    shutil.copy2(os.path.join(folder, n), tgt)
            desc = "\n".join(f"- input/{ds['name']}: {ds.get('description', '')}" for ds in d["datasets"])
            mid = os.path.basename(f).rsplit(".", 1)[0]
            for grp in d["queries"]:
                for q in grp:
                    query = (f"Datasets:\n{desc}\n\nQuery: {q['question']}\n\nAnswer with a hypothesis that answers "
                             f"the query, based on your analysis.\n\nData files: "
                             + ", ".join(f"input/{n}" for n in names))
                    out.append({"domain": dom, "id": f"db-{dom}-{mid}-q{q['qid']}", "query": query,
                                "data_files": names, "question_type": q.get("question_type")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
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
    a.kb = os.path.join(os.path.dirname(os.path.abspath(a.out)), "db_kb")
    tasks = load_tasks(os.path.abspath(a.real), a.kb)
    if a.limit:
        tasks = tasks[: a.limit]
    model = kb_run.ScriptedModel() if a.model == "dry" else kb_run.rr.Ollama(a.model, a.temperature, a.max_tokens)
    done, spent = set(), [0.0]
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["task"], r["seed"]))
            spent[0] += r.get("cost_usd") or 0
    todo = [t for t in tasks if (t["id"], a.seed) not in done]
    print(f"{len(tasks)} tasks, {len(todo)} to run; python roots {kb_run.sandbox._python_roots()}", flush=True)
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor

    def job(t):
        if a.budget_usd and spent[0] >= a.budget_usd:
            return
        try:
            rec = kb_run.run_task(t, a.seed, model, a)
        except Exception as e:  # noqa: BLE001  not written: the next run retries it
            print(f"{t['id']}: error {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        rec.update(data_files=t["data_files"], question_type=t["question_type"],
                   python_roots=kb_run.sandbox._python_roots())
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            spent[0] += rec.get("cost_usd") or 0
        nfail = sum(1 for c in rec["cells"] if c.get("ok") is False)
        print(f"{t['id']}: {rec['status']}, cells {len(rec['cells'])}, failed {nfail}, cost {rec.get('cost_usd')}",
              flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(job, todo))
    print(f"done; recorded cost ${spent[0]:.2f}")


if __name__ == "__main__":
    main()
