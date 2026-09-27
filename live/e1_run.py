#!/usr/bin/env python3
"""E1: live 2 x 2 experiment on QRData (plan 2026-09-24j, section 4).

Factors:
  prompt  P0: the QRData prompt of qrdata/qr_run.py;
          P1: P0 plus the column names of each of the question's data files, in pandas order
              (pd.read_csv(path, nrows=0).columns), which is what the gate knows;
  gate    G0: every cell runs;
          G1: frozen gate v8 checks each cell first. exists() is answered inside the live
              sandbox (live_runner.py), locate() from the data directory. A blocked cell does
              not run, and the model receives the receipt, as in the RQ3 receipt arm.
              GateState is carried across cells.

All (question, arm) jobs run in one random order (seed 7), so the arms share a time
window. Settings as in the QRData run: up to 15 cells, 120 s per cell, temperature 1.0.

Usage:
  /home/claude/iaenv/bin/python live/e1_run.py --tables /home/claude/qrdata_tables/data \
      --questions .../QRData.json --out runs/e1/results.jsonl --transcripts runs/e1/transcripts \
      --model gpt-oss:120b --workers 3 --budget-usd 6.5
"""
import argparse
import json
import os
import random
import select
import shutil
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for d in ("kramabench", "rq3-pilot", "qrdata", "dabstep-gate"):
    sys.path.insert(0, os.path.join(ROOT, d))
from gate import GateState, decide  # noqa: E402  (gate v8, frozen before this run)
import kb_gate  # noqa: E402
import kb_run  # noqa: E402
import qr_run  # noqa: E402  (sets the InfiAgent/QRData system prompt through ia_run)

rr = kb_run.rr
ARMS = ("P0G0", "P0G1", "P1G0", "P1G1")
BLOCKED = ("Observation:\nThis cell was not run. A check before execution found that it uses data that does "
           "not exist.\n")


class LiveExecutor(kb_run.KBExecutor):
    """KBExecutor with live_runner.py, which also answers {"exists": [paths]}."""

    def __init__(self, lake, mode="bwrap", timeout=120):
        self.timeout, self.mode, self.lake = timeout, mode, os.path.abspath(lake)
        self.work = tempfile.mkdtemp(prefix="e1-", dir=os.environ.get("RQ3_SCRATCH"))
        os.makedirs(os.path.join(self.work, "input"))
        shutil.copy2(os.path.join(HERE, "live_runner.py"), os.path.join(self.work, ".runner.py"))
        self.binds = []
        if mode == "local":
            os.rmdir(os.path.join(self.work, "input"))
            os.symlink(self.lake, os.path.join(self.work, "input"))
        self.proc = self._start()

    def exists(self, path):
        if self.proc.poll() is not None:
            return None
        self.proc.stdin.write(json.dumps({"exists": [str(path)]}) + "\n")
        self.proc.stdin.flush()
        r, _, _ = select.select([self.proc.stdout], [], [], 30)
        if not r:
            return None
        line = self.proc.stdout.readline()
        try:
            return bool(json.loads(line)["exists"][0])
        except (ValueError, KeyError, IndexError, TypeError):
            return None


def receipt_text(rc):
    """The RQ3 receipt text, with file-specific wording for found_in (gate v8, R3)."""
    kind = rc.get("kind") or "name"
    lines = [f"- {rc.get('location')}: {rc.get('reason')}."]
    av = rc.get("available") or []
    if kind == "file":
        if rc.get("found_in"):
            lines.append("- A file with this name exists at: " + ", ".join(rc["found_in"][:10]))
        if av:
            what = "Data directories" if str(rc.get("reason", "")).startswith("directory") else "Data files"
            shown = ", ".join(map(str, av[:60])) + (f", ... ({len(av)} in total)" if len(av) > 60 else "")
            lines.append(f"- {what}: {shown}")
        if rc.get("near_matches"):
            lines.append("- Close matches: " + ", ".join(map(str, rc["near_matches"])))
        return "\n".join(lines)
    return rr.receipt_text(rc)


def columns_text(lake, files, cap=300):
    import pandas as pd
    out = []
    for f in files:
        try:
            cols = [str(c) for c in pd.read_csv(os.path.join(lake, f), nrows=0).columns]
            shown = ", ".join(cols[:cap]) + (f", ... ({len(cols)} in total)" if len(cols) > cap else "")
            out.append(f"Columns of input/{f}: {shown}")
        except Exception:  # noqa: BLE001
            out.append(f"Columns of input/{f}: (pandas cannot read this file with default arguments)")
    return "\n".join(out)


def run_task(t, arm, seed, model, a, schema, locate):
    lake = os.path.join(a.kb, "data", "qrdata", "input")
    gate_on, schema_prompt = arm.endswith("G1"), arm.startswith("P1")
    ex = LiveExecutor(lake, timeout=a.cell_timeout)
    rec = {"task": t["id"], "arm": arm, "seed": seed, "model": getattr(model, "model", None),
           "params": {"temperature": a.temperature, "max_tokens": a.max_tokens, "steps": a.steps,
                      "cell_timeout": a.cell_timeout}}
    try:
        n, listing = kb_run.lake_listing(lake)
        query = t["query"]
        if schema_prompt:
            query += "\n\n" + columns_text(lake, t["data_files"])
        msgs = [{"role": "system", "content": kb_run.SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nFiles in input/ ({n} files):\n{listing}"}]
        cells, usage = [], {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        st = GateState()
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
            if gate_on:
                try:
                    g = decide(code, schema, st, exists=ex.exists, locate=locate)
                except Exception as e:  # noqa: BLE001  a gate crash lets the cell run
                    g = {"outcome": "gate_error", "reason": f"{type(e).__name__}: {e}"}
                if g["outcome"] == "block":
                    rc = {x: g.get(x) for x in ("location", "kind", "source", "missing", "reason", "available",
                                                "found_in", "near_matches")}
                    cells.append({"step": k, "code": code, "blocked": True, "receipt": rc, "ok": None})
                    st.commit(failed=True)
                    msgs.append({"role": "user", "content": BLOCKED + receipt_text(rc) + "\nContinue."})
                    continue
                gate_outcome = g["outcome"]
            else:
                gate_outcome = None
            res = ex.run(code)
            if gate_on:
                st.commit(failed=not res.get("ok"))
            cells.append({"step": k, "code": code, "blocked": False, "gate": gate_outcome, "ok": bool(res.get("ok")),
                          "error_type": res.get("error_type"), "error": (res.get("error") or "")[-6000:],
                          "stdout": rr.cut(res.get("stdout"), 3000), "final_answer": res.get("final_answer"),
                          "seconds": res.get("seconds"), "lost": bool(res.get("lost"))})
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
            with open(os.path.join(a.transcripts, f"{t['id']}_{arm}_{seed}.json"), "w") as f:
                json.dump(msgs, f, indent=1)
        return rec
    finally:
        ex.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--cell-timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=0)
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(a.out)), "qr_kb")
    os.makedirs(os.path.join(root, "data", "qrdata"), exist_ok=True)
    link = os.path.join(root, "data", "qrdata", "input")
    if not os.path.islink(link):
        os.symlink(os.path.abspath(a.tables), link)
    a.kb = root
    lake = os.path.abspath(a.tables)
    schema, ambiguous, skipped = kb_gate.domain_schema(lake)
    where = {}
    for d, _, files in os.walk(lake):
        for f in files:
            where.setdefault(f, []).append(os.path.join("input", os.path.relpath(os.path.join(d, f), lake)))

    def locate(base):
        return where.get(base, [])

    model = kb_run.ScriptedModel() if a.model == "dry" else rr.Ollama(a.model, a.temperature, a.max_tokens)
    tasks = qr_run.load_tasks(a.questions)
    if a.limit:
        tasks = tasks[: a.limit]
    arms = a.arms.split(",")
    jobs = [(t, arm) for t in tasks for arm in arms]
    random.Random(7).shuffle(jobs)
    done, spent = set(), [0.0]
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["task"], r["arm"], r["seed"]))
            spent[0] += r.get("cost_usd") or 0
    todo = [(t, arm) for t, arm in jobs if (t["id"], arm, a.seed) not in done]
    print(f"schema {len(schema)} files (ambiguous {len(ambiguous)}, unreadable {len(skipped)}); "
          f"{len(todo)} jobs to run; gate {sys.modules['gate'].__file__}", flush=True)
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor

    def job(ta):
        t, arm = ta
        if a.budget_usd and spent[0] >= a.budget_usd:
            return
        try:
            rec = run_task(t, arm, a.seed, model, a, schema, locate)
        except Exception as e:  # noqa: BLE001  not written: the next run retries it
            print(f"{t['id']} {arm}: error {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        rec.update(data_files=t["data_files"], question_type=t["question_type"])
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            spent[0] += rec.get("cost_usd") or 0
        nb = sum(1 for c in rec["cells"] if c.get("blocked"))
        print(f"{t['id']} {arm}: {rec['status']}, cells {len(rec['cells'])}, blocked {nb}, "
              f"answer {str(rec['final_answer'])[:30]!r}, cost {rec.get('cost_usd')}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(job, todo))
    print(f"done; recorded cost ${spent[0]:.2f}")


if __name__ == "__main__":
    main()
