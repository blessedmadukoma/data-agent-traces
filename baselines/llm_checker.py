#!/usr/bin/env python3
"""Baseline M2.3: an LLM as the pre-execution checker (CRANE-LLM style, without runtime state).

For each recorded code cell of a run, the model sees what the gate sees: the working
directory, where the data files are, the question's data files with their columns
(pandas order), the earlier cells of the trace (code only, last 6, each cut to 1,500
characters) and the cell. It answers whether the cell will raise KeyError or
FileNotFoundError because it uses a column, key or file that does not exist.

Decision: a cell is blocked if the reply contains "VERDICT: BLOCK". A block on a failed
cell is cause-matched if the named item ("MISSING: ...") is the recorded missing key, or
the recorded missing path (same path or same base name). A block on a cell without an
error is a false block. Cells are checked in a random order (seed 11) until the budget
stop, so a partial run is a random sample.

Usage: python3 baselines/llm_checker.py --results runs/qr/results_gpt-oss.jsonl \
    --tables /home/claude/qrdata_tables/data --out runs/llm_check/qr_gpt-oss.jsonl \
    --model gpt-oss:120b --workers 2 --budget-usd 1.8
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("rq3-pilot", "kramabench"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import kb_gate  # noqa: E402
import rq3_run as rr  # noqa: E402

PROMPT = """You check Python code cells before they run. A data agent works in a Python session (names persist between cells). The working directory is /work. The data files are in the directory input/ (read-only); no other data directory exists.

The question's data files and their columns:
{schema}

Earlier cells in this session (code only; they ran in this order):
{prefix}

The cell to check:
```python
{code}
```

Will this cell raise KeyError or FileNotFoundError because it uses a column, dictionary key, file or directory that does not exist? Answer with "VERDICT: BLOCK" and on the next line "MISSING: <the missing name or path>", or with "VERDICT: RUN" if it will not raise such an error or you are not sure."""


def schema_text(lake, files):
    import pandas as pd
    out = []
    for f in files:
        try:
            cols = [str(c) for c in pd.read_csv(os.path.join(lake, f), nrows=0).columns]
            out.append(f"- input/{f}: " + ", ".join(cols[:300]))
        except Exception:  # noqa: BLE001
            out.append(f"- input/{f}: (cannot be read with default arguments)")
    return "\n".join(out)


def parse(text):
    block = bool(re.search(r"VERDICT:\s*BLOCK", text or "", re.I))
    m = re.search(r"MISSING:\s*(.+)", text or "")
    missing = m.group(1).strip().strip("`'\" ") if m else None
    return block, missing


def cause(missing, rec_error, et):
    if not missing:
        return False
    keys, path = kb_gate.error_names(rec_error)
    if et == "KeyError":
        return missing in keys or any(missing.strip("[]'\" ") == k for k in keys)
    if et == "FileNotFoundError" and path:
        return kb_gate.same_path(missing, path) or os.path.basename(missing) == os.path.basename(path)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tables", help="data directory (one lake for every task)")
    ap.add_argument("--kb", help="kb_gate-style root: the lake of a task is <kb>/data/<domain>/input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget-usd", type=float, default=1.8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model = rr.Ollama(a.model, 0.0, 2048)
    jobs = []
    for r in map(json.loads, open(a.results)):
        cells = [c for c in r["cells"] if not c.get("no_code")]
        lake = a.tables or os.path.join(a.kb, "data", r["domain"], "input")
        files = r.get("data_files") or ([r["file_name"]] if r.get("file_name") else [])
        sch = schema_text(lake, files)
        for i, c in enumerate(cells):
            prefix = "\n\n".join(f"# cell {p['step']}\n{p['code'][:1500]}" for p in cells[max(0, i - 6):i]) or "(none)"
            jobs.append({"task": r["task"], "step": c["step"], "ok": bool(c.get("ok")),
                         "error_type": c.get("error_type"), "error": (c.get("error") or "")[-1500:],
                         "prompt": PROMPT.format(schema=sch, prefix=prefix, code=c["code"][:6000])})
    random.Random(11).shuffle(jobs)
    if a.limit:
        jobs = jobs[: a.limit]
    done, spent = set(), [0.0]
    if os.path.exists(a.out):
        for line in open(a.out):
            x = json.loads(line)
            done.add((x["task"], x["step"]))
            spent[0] += x.get("cost_usd") or 0
    todo = [j for j in jobs if (j["task"], j["step"]) not in done]
    print(f"{len(jobs)} cells, {len(todo)} to check", flush=True)
    lock = threading.Lock()
    pr = rr.PRICES[a.model]
    from concurrent.futures import ThreadPoolExecutor

    def job(j):
        if spent[0] >= a.budget_usd:
            return
        t0 = time.perf_counter()
        try:
            text, u = model.chat([{"role": "user", "content": j["prompt"]}], 1)
        except Exception as e:  # noqa: BLE001  not written: the next run retries it
            print(f"{j['task']} {j['step']}: error {type(e).__name__}", flush=True)
            return
        block, missing = parse(text)
        cost = (u["prompt_tokens"] * pr[0] + u["completion_tokens"] * pr[1]) / 1e6
        row = {"task": j["task"], "step": j["step"], "ok": j["ok"], "error_type": j["error_type"],
               "block": block, "missing": missing,
               "cause_match": (not j["ok"]) and block and cause(missing, j["error"], j["error_type"]),
               "reply": (text or "")[-400:], "seconds": round(time.perf_counter() - t0, 2),
               "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"], "cost_usd": cost,
               "served_model": u.get("model")}
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            spent[0] += cost

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(job, todo))
    print(f"done; recorded cost ${spent[0]:.2f}")


if __name__ == "__main__":
    main()
