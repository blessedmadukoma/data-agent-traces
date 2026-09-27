#!/usr/bin/env python3
"""Run a code agent on KramaBench tasks and record every cell (held-out gate test).

Each task runs in a bubblewrap sandbox (no network, private /tmp) with the
domain's data lake mounted read-only at /work/input. The model writes one
Python cell per reply; every cell runs (the gate is not used during the run).
Before each cell the harness records the files in the work directory and in
/tmp, so that a file-existence check can be replayed exactly afterwards.

Usage (with the KramaBench virtual environment, which has openpyxl and scipy):
  python kramabench/kb_run.py --kb $KRAMABENCH \
      --out runs/kb/results.jsonl --transcripts runs/kb/transcripts --model gpt-oss:120b \
      --workers 2 [--limit-per-domain 1] [--budget-usd 5]
"""
import argparse
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rq3-pilot"))
import rq3_run as rr  # noqa: E402
import sandbox  # noqa: E402

DOMAINS = ("archeology", "astronomy", "biomedical", "environment", "legal", "wildfire")
SYSTEM = (
    "You are a data analysis agent. You answer one question using the data files in the directory input/ "
    "(read-only). You work in a Python session: names persist between cells. Available packages: pandas, "
    "numpy, scipy, openpyxl. You may write files in the current directory. In each reply, think briefly, "
    "then write exactly one Python code block in ```python ... ```. Its printed output comes back to you. "
    "When you know the answer, call final_answer(answer) in a code block.")


def lake_listing(root, per_dir=20, limit=6000):
    rows, n = [], 0
    for d, dirs, files in sorted(os.walk(root)):
        dirs.sort()
        files = sorted(f for f in files if not f.startswith("."))
        if not files:
            continue
        n += len(files)
        rel = os.path.relpath(d, root)
        rel = "input/" if rel == "." else f"input/{rel}/"
        shown = ", ".join(files[:per_dir]) + (f", ... ({len(files) - per_dir} more)" if len(files) > per_dir else "")
        rows.append(f"- {rel} ({len(files)} files): {shown}")
    text = "\n".join(rows)
    if len(text) > limit:
        text = text[:limit] + "\n- ... (listing cut; use os.listdir to see more)"
    return n, text


class KBExecutor(sandbox.Executor):
    """sandbox.Executor with the data lake mounted read-only at /work/input and a runner
    that can list the workspace."""

    def __init__(self, lake, mode="bwrap", timeout=120):
        self.timeout, self.mode, self.lake = timeout, mode, os.path.abspath(lake)
        self.work = tempfile.mkdtemp(prefix="kb-", dir=os.environ.get("RQ3_SCRATCH"))
        os.makedirs(os.path.join(self.work, "input"))
        shutil.copy2(os.path.join(HERE, "kb_runner.py"), os.path.join(self.work, ".runner.py"))
        self.binds = []
        if mode == "local":        # tests only: a symlink stands in for the read-only mount
            os.rmdir(os.path.join(self.work, "input"))
            os.symlink(self.lake, os.path.join(self.work, "input"))
        self.proc = self._start()

    def _start(self):
        if self.mode == "local":
            return super()._start()
        py = os.path.abspath(sys.executable)
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for d in ("/usr", "/bin", "/lib", "/lib64", "/sbin", "/etc"):
            if os.path.exists(d):
                cmd += ["--ro-bind", d, d]
        for r in sandbox._python_roots():
            cmd += ["--ro-bind", r, r]
        cmd += ["--bind", self.work, "/work", "--ro-bind", self.lake, "/work/input", "--chdir", "/work",
                "--setenv", "HOME", "/work", "--setenv", "MPLBACKEND", "Agg", py, "-u", "/work/.runner.py"]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err(),
                                text=True, bufsize=1)

    def ls(self):
        if self.proc.poll() is not None:
            return None
        self.proc.stdin.write(json.dumps({"ls": True}) + "\n")
        self.proc.stdin.flush()
        r, _, _ = select.select([self.proc.stdout], [], [], 30)
        if not r:
            return None
        line = self.proc.stdout.readline()
        return json.loads(line) if line else None


def load_tasks(kb):
    out = []
    for d in DOMAINS:
        for t in json.load(open(os.path.join(kb, "workload", f"{d}.json"))):
            out.append({"domain": d, "id": t["id"], "query": t["query"], "answer": t.get("answer"),
                        "answer_type": t.get("answer_type"), "data_sources": t.get("data_sources")})
    return out


def run_task(t, seed, model, a):
    lake = os.path.join(a.kb, "data", t["domain"], "input")
    ex = KBExecutor(lake, mode=a.sandbox, timeout=a.cell_timeout)
    rec = {"task": t["id"], "domain": t["domain"], "seed": seed, "model": getattr(model, "model", None),
           "params": {"temperature": a.temperature, "max_tokens": a.max_tokens, "steps": a.steps,
                      "cell_timeout": a.cell_timeout}}
    try:
        n, listing = lake_listing(lake)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {t['query']}\n\nFiles in input/ ({n} files):\n{listing}"}]
        cells, usage = [], {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        for k in range(1, a.steps + 1):
            text, u = model.chat(msgs, seed)
            usage["prompt_tokens"] += u["prompt_tokens"]
            usage["completion_tokens"] += u["completion_tokens"]
            usage["calls"] += 1
            rec["served_model"] = u.get("model")
            msgs.append({"role": "assistant", "content": text})
            code = rr.extract_code(text)
            if code is None:
                cells.append({"step": k, "no_code": True})
                msgs.append({"role": "user", "content": "Please reply with exactly one Python code block."})
                continue
            ws = ex.ls()
            res = ex.run(code)
            cells.append({"step": k, "code": code, "ok": bool(res.get("ok")), "error_type": res.get("error_type"),
                          "error": (res.get("error") or "")[-6000:], "stdout": rr.cut(res.get("stdout"), 3000),
                          "final_answer": res.get("final_answer"), "seconds": res.get("seconds"),
                          "workspace": (ws or {}).get("files"), "workspace_cut": (ws or {}).get("cut"),
                          "workspace_known": ws is not None, "lost": bool(res.get("lost"))})
            if res.get("lost"):
                rec["status"] = "executor_lost"
                break
            if res.get("final_answer") is not None and res.get("ok"):
                break
            msgs.append({"role": "user", "content": rr.observation(res)})
        rec.update(cells=cells, usage=usage)
        rec.setdefault("status", "done")
        fin = next((c for c in cells if c.get("final_answer") is not None and c.get("ok")), None)
        rec["final_answer"] = fin["final_answer"] if fin else None
        pr = rr.PRICES.get(rec["model"])
        rec["cost_usd"] = (usage["prompt_tokens"] * pr[0] + usage["completion_tokens"] * pr[1]) / 1e6 if pr else None
        if a.transcripts:
            os.makedirs(a.transcripts, exist_ok=True)
            with open(os.path.join(a.transcripts, f"{t['id']}_{seed}.json"), "w") as f:
                json.dump(msgs, f, indent=1)
        return rec
    finally:
        ex.close()


class ScriptedModel:
    """For tests: lists the lake, reads the first CSV it finds, then answers."""
    model = "dry"

    def chat(self, messages, seed):
        n = sum(1 for m in messages if m["role"] == "assistant")
        steps = ["```python\nimport os\nfor d, _, fs in os.walk('input'):\n    print(d, fs[:3])\n```",
                 "```python\nimport glob, pandas as pd\nf = sorted(glob.glob('input/**/*.csv', recursive=True))[0]\n"
                 "df = pd.read_csv(f)\nprint(f, df.columns.tolist()[:5])\n```",
                 "```python\ndf['__no_such_column__']\n```",
                 "```python\nopen('input/__missing__.csv').read()\n```",
                 "```python\ndf.to_csv('out.csv')\nfinal_answer(len(df))\n```"]
        return steps[min(n, len(steps) - 1)], {"prompt_tokens": 100, "completion_tokens": 20, "model": "dry",
                                               "finish_reason": "stop"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
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
    ap.add_argument("--limit-per-domain", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=0)
    a = ap.parse_args()
    model = ScriptedModel() if a.model == "dry" else rr.Ollama(a.model, a.temperature, a.max_tokens)
    tasks = load_tasks(a.kb)
    if a.limit_per_domain:
        seen = {}
        tasks = [t for t in tasks if seen.setdefault(t["domain"], []).append(t) or
                 len(seen[t["domain"]]) <= a.limit_per_domain]
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
            rec = run_task(t, a.seed, model, a)
        except Exception as e:  # not written: the next run retries it
            print(f"{t['id']}: error {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
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
