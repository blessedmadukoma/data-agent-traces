#!/usr/bin/env python3
"""Replay recorded runs with final_answer protected (preprint, Section 4.4).

Our harness defines final_answer in the same namespace as the model's variables. When the
model defines its own final_answer, for example `def final_answer(a): print(a)`, its calls
go to that function, the harness records no answer, and the run goes on until the cell limit.

This script replays the recorded cells of every run that has no recorded answer, or whose
code defines, assigns or deletes final_answer. The cells run in order in the sandbox of our
runs (data read-only at /work/input, no network, 120 s per cell), with live/fa_runner.py,
which keeps final_answer safe. The replay stops at the first cell in which the harness
records an answer that is not None. A cell that timed out in the recording is not run again;
it counts as a timeout. Up to that cell, the model received the same outputs
as in the recording, so the recorded cells are what the model would have written.

Usage (one line per results file; --data is the folder that the run mounted at input/):
  uv run live/fa_replay.py --tag live_gpt --results recorded/e1/results.jsonl \
      --data data/QRData/benchmark/data --out runs/fa_replay/live_gpt.jsonl
  DiscoveryBench and KramaBench mount one folder per domain: pass --data-per-domain with a
  folder that holds <domain>/input (tools/make_kb.py writes it under <kb>/data).
"""
import argparse
import ast
import json
import os
import shutil
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)   # from parsing the models' code

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kramabench"))
sys.path.insert(0, os.path.join(HERE, "..", "rq3-pilot"))
import kb_run  # noqa: E402


class FAExecutor(kb_run.KBExecutor):
    """KBExecutor with live/fa_runner.py as the REPL server."""

    def __init__(self, lake, mode="bwrap", timeout=120):
        self.timeout, self.mode, self.lake = timeout, mode, os.path.abspath(lake)
        self.work = tempfile.mkdtemp(prefix="fa-", dir=os.environ.get("RQ3_SCRATCH"))
        os.makedirs(os.path.join(self.work, "input"))
        shutil.copy2(os.path.join(HERE, "fa_runner.py"), os.path.join(self.work, ".runner.py"))
        self.binds = []
        if mode == "local":        # tests only: a symlink stands in for the read-only mount
            os.rmdir(os.path.join(self.work, "input"))
            os.symlink(self.lake, os.path.join(self.work, "input"))
        self.proc = self._start()


def touches(code):
    """True if the code defines, assigns, imports as or deletes the name final_answer."""
    try:
        t = ast.parse(code)
    except SyntaxError:
        return False
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == "final_answer":
            return True
        if isinstance(n, ast.Name) and n.id == "final_answer" and isinstance(n.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(n, ast.alias) and n.asname == "final_answer":
            return True
    return False


def ran_cells(r):
    """Indices of the cells that ran: they have code, and the checker did not block them."""
    return [k for k, c in enumerate(r["cells"]) if "code" in c and not c.get("no_code") and not c.get("blocked")]


def replay(r, lake, mode, timeout):
    idx = ran_cells(r)
    ex = FAExecutor(lake, mode=mode, timeout=timeout)
    out = {"cells": [], "answer": None, "answer_cell": None, "lost": False}
    try:
        for k in idx:
            if r["cells"][k].get("lost"):     # the cell timed out in the recording: count it as lost
                out["cells"].append({"cell": k, "ok": False, "error_type": "TimeoutError", "called": False,
                                     "final_answer": None})
                out["lost"] = out["lost_from_recording"] = True
                break
            res = ex.run(r["cells"][k]["code"])
            out["cells"].append({"cell": k, "ok": bool(res.get("ok")), "error_type": res.get("error_type"),
                                 "called": bool(res.get("called")), "final_answer": res.get("final_answer")})
            if res.get("lost"):
                out["lost"] = True
                break
            if res.get("final_answer") is not None and res.get("ok"):
                out["answer"], out["answer_cell"] = res["final_answer"], k
                break
    finally:
        ex.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="a name for the run, used in the output keys")
    ap.add_argument("--results", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="the folder mounted at input/ for every task")
    g.add_argument("--data-per-domain", help="a folder with <domain>/input for each task's domain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sandbox", choices=["bwrap", "local"], default="bwrap")
    ap.add_argument("--cell-timeout", type=int, default=120)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["key"] for line in open(a.out)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = 0
    with open(a.out, "a") as f:
        for i, line in enumerate(open(a.results)):
            r = json.loads(line)
            idx = ran_cells(r)
            if not (r.get("final_answer") is None or any(touches(r["cells"][k]["code"]) for k in idx)):
                continue
            key = f"{a.tag}:{i}"
            if key in done:
                continue
            lake = a.data or os.path.join(a.data_per_domain, r["domain"], "input")
            t0 = time.time()
            o = replay(r, lake, a.sandbox, a.cell_timeout)
            o.update(key=key, task=r["task"], arm=r.get("arm"), seed=r.get("seed"), wall=round(time.time() - t0, 1))
            f.write(json.dumps(o, default=str) + "\n")
            f.flush()
            n += 1
    print(f"{a.tag}: replayed {n} runs, output {a.out}")


if __name__ == "__main__":
    main()
