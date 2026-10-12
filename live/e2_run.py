#!/usr/bin/env python3
"""E2: second live experiment on QRData (pre-registration: live/e2_preregistration.md).

Arms (all with the prompt that gives the exact data paths, and with final_answer protected):
  N  no checker: every step runs and the model receives the executor's output or error;
  R  checker release 4 (dabstep-gate/v9) checks each step first; a blocked step does not run and the
     model receives the receipt (as in the first live experiment);
  E  no checker, but when a step fails with a missing-name error, the error message is followed by the
     names that exist: the paths of the question's data files and the columns of each.
In arms N and E the checker also decides every step in shadow mode (the step runs whatever it decides),
so a shadow block on a step that then runs without an error is a false block of release 4.

Every (question, arm, seed) job runs in one random order, so the arms share the same time window.
The sandbox is a Docker container without network, with the data mounted read-only at /work/input.

Usage:
  OLLAMA_API_KEY=... python3 live/e2_run.py --tables data/QRData/benchmark/data \\
      --questions data/QRData/benchmark/QRData.json --out runs/e2/results.jsonl \\
      --seeds 1,2,3,4,5,6,7,8,9,10 --workers 3 --budget-usd 40
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
GATE_PATH = os.path.join(ROOT, "dabstep-gate", "v9", "gate.py")
_spec = importlib.util.spec_from_file_location("gate", GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
sys.modules["gate"] = gate                      # kb_gate.domain_schema reads gate.csv_columns
for d in ("kramabench", "rq3-pilot", "qrdata"):
    sys.path.insert(0, os.path.join(ROOT, d))
import kb_gate  # noqa: E402
import qr_run  # noqa: E402  (load_tasks; sets the QRData system prompt in kb_run)

kb_run = qr_run.kb_run
rr = kb_run.rr                                    # extract_code, observation, receipt_text, cut
ARMS = ("N", "R", "E")
# The QRData system prompt of the first experiment, plus one sentence that gives the exact paths.
SYSTEM = kb_run.SYSTEM + (" The data files are in the folder input/ of the working directory: read them with "
                          "exactly the paths given with the question, for example pd.read_csv('input/name.csv').")
BLOCKED = ("Observation:\nThis cell was not run. A check before execution found that it uses data that does "
           "not exist.\n")
MISSING_NAME_ERRORS = ("KeyError", "FileNotFoundError", "IsADirectoryError", "NotADirectoryError")
PRICES = {"gpt-oss:120b": (0.15, 0.014, 0.60)}     # USD per 1M tokens: input, cached input, output
SANDBOX_PY = os.environ.get("E2_SANDBOX_PY", "/home/claude/work/envs/sandbox/bin/python")
IMAGE = os.environ.get("E2_IMAGE", "mirror.gcr.io/library/python:3.12-slim")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class DockerExecutor:
    """One Python session per run, in a container without network. The host's /usr and /etc and the
    sandbox environment are mounted read-only (as the bubblewrap sandbox of the first experiment did)."""

    def __init__(self, lake, timeout=120):
        self.timeout, self.lake = timeout, os.path.abspath(lake)
        self.work = tempfile.mkdtemp(prefix="e2-", dir=os.environ.get("E2_SCRATCH"))
        os.makedirs(os.path.join(self.work, "input"))
        shutil.copy2(os.path.join(HERE, "e2_runner.py"), os.path.join(self.work, ".runner.py"))
        self.name = "e2-" + uuid.uuid4().hex[:12]
        env_root = os.path.dirname(os.path.dirname(SANDBOX_PY))
        cmd = ["docker", "run", "-i", "--rm", "--name", self.name, "--network", "none", "--read-only",
               "--tmpfs", "/tmp", "--memory", "2g", "--cpus", "1",
               "-v", "/usr:/usr:ro", "-v", "/etc:/etc:ro", "-v", f"{env_root}:{env_root}:ro",
               "-v", f"{self.work}:/work", "-v", f"{self.lake}:/work/input:ro", "-w", "/work",
               "-e", "HOME=/work", "-e", "MPLBACKEND=Agg", IMAGE, SANDBOX_PY, "-u", "/work/.runner.py"]
        self.err = open(os.path.join(tempfile.gettempdir(), self.name + ".err"), "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                     text=True, bufsize=1)

    def _ask(self, req, timeout):
        if self.proc.poll() is not None:
            return None
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        t0 = time.time()
        while time.time() - t0 < timeout:
            r, _, _ = select.select([self.proc.stdout], [], [], 0.5)
            if r:
                line = self.proc.stdout.readline()
                return json.loads(line) if line else None
        return "timeout"

    def run(self, code):
        res = self._ask({"code": code}, self.timeout)
        if res == "timeout":
            self.close()
            return {"ok": False, "error_type": "TimeoutError", "stdout": "", "final_answer": None, "lost": True,
                    "error": f"TimeoutError: the cell did not finish within {self.timeout} s"}
        if res is None:
            return {"ok": False, "error_type": "ExecutorLost", "stdout": "", "final_answer": None, "lost": True,
                    "error": "the executor process ended"}
        return res

    def exists(self, path):
        res = self._ask({"exists": [str(path)]}, 30)
        try:
            return bool(res["exists"][0])
        except (TypeError, KeyError, IndexError):
            return None

    def close(self):
        subprocess.run(["docker", "kill", self.name], capture_output=True)
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.err.close()
        shutil.rmtree(self.work, ignore_errors=True)


class Model:
    """Chat completions on Ollama Cloud; records input, cached input and output tokens and the latency."""

    def __init__(self, model, temperature, max_tokens):
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.key = os.environ.get("OLLAMA_API_KEY")
        if not self.key:
            raise SystemExit("set OLLAMA_API_KEY")

    def chat(self, messages, seed):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                           "max_tokens": self.max_tokens, "seed": seed}).encode()
        for attempt in range(6):
            req = urllib.request.Request("https://ollama.com/v1/chat/completions", data=body, method="POST",
                                         headers={"Authorization": "Bearer " + self.key,
                                                  "Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.load(r)
                u = d.get("usage") or {}
                cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
                return (d["choices"][0]["message"].get("content") or "",
                        {"prompt_tokens": u.get("prompt_tokens", 0), "cached_tokens": cached,
                         "completion_tokens": u.get("completion_tokens", 0), "model": d.get("model"),
                         "finish_reason": d["choices"][0].get("finish_reason"), "seconds": round(time.time() - t0, 2)})
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt < 5:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise


def cost(model, usage):
    pin, pcached, pout = PRICES[model]
    fresh = usage["prompt_tokens"] - usage["cached_tokens"]
    return (fresh * pin + usage["cached_tokens"] * pcached + usage["completion_tokens"] * pout) / 1e6


def names_text(lake, files, cap=300):
    """Arm E: the names that exist, appended to a missing-name error."""
    import pandas as pd
    out = ["Note from the environment: the data files of this question are " +
           ", ".join(f"input/{f}" for f in files) + "."]
    for f in files:
        try:
            cols = [str(c) for c in pd.read_csv(os.path.join(lake, f), nrows=0).columns]
            shown = ", ".join(cols[:cap]) + (f", ... ({len(cols)} in total)" if len(cols) > cap else "")
            out.append(f"Columns of input/{f}: {shown}")
        except Exception:  # noqa: BLE001
            out.append(f"Columns of input/{f}: (pandas cannot read this file with default arguments)")
    return "\n".join(out)


def receipt_text(rc):
    """The receipt of the first live experiment (live/e1_run.receipt_text)."""
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


def run_task(t, arm, seed, model, a, schema, locate):
    ex = DockerExecutor(a.lake, timeout=a.cell_timeout)
    rec = {"task": t["id"], "arm": arm, "seed": seed, "model": model.model,
           "params": {"temperature": a.temperature, "max_tokens": a.max_tokens, "steps": a.steps,
                      "cell_timeout": a.cell_timeout}}
    usage = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "calls": 0, "model_seconds": 0.0}
    t0 = time.time()
    try:
        n, listing = kb_run.lake_listing(a.lake)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {t['query']}\n\nFiles in input/ ({n} files):\n{listing}"}]
        cells, st = [], gate.GateState()
        for k in range(1, a.steps + 1):
            text, u = model.chat(msgs, seed)
            for key in ("prompt_tokens", "cached_tokens", "completion_tokens"):
                usage[key] += u[key]
            usage["calls"] += 1
            usage["model_seconds"] += u["seconds"]
            rec["served_model"] = u.get("model")
            msgs.append({"role": "assistant", "content": text})
            code = rr.extract_code(text)
            if code is None:
                cells.append({"step": k, "no_code": True})
                msgs.append({"role": "user", "content": "Please reply with exactly one Python code block."})
                continue
            try:
                g = gate.decide(code, schema, st, exists=ex.exists, locate=locate)
            except Exception as e:  # noqa: BLE001  a checker crash lets the step run
                g = {"outcome": "gate_error", "reason": f"{type(e).__name__}: {e}"}
            rc = {x: g.get(x) for x in ("location", "kind", "source", "missing", "reason", "available",
                                        "found_in", "near_matches")} if g["outcome"] == "block" else None
            if arm == "R" and g["outcome"] == "block":
                cells.append({"step": k, "code": code, "blocked": True, "gate": "block", "receipt": rc, "ok": None})
                st.commit(failed=True)
                msgs.append({"role": "user", "content": BLOCKED + receipt_text(rc) + "\nContinue."})
                continue
            res = ex.run(code)
            st.commit(failed=not res.get("ok"))
            obs = rr.observation(res)
            enriched = False
            if arm == "E" and not res.get("ok") and res.get("error_type") in MISSING_NAME_ERRORS:
                obs += "\n" + names_text(a.lake, t["data_files"])
                enriched = True
            cells.append({"step": k, "code": code, "blocked": False, "gate": g["outcome"],
                          "shadow_receipt": rc if arm != "R" else None, "ok": bool(res.get("ok")),
                          "error_type": res.get("error_type"), "error": (res.get("error") or "")[-6000:],
                          "stdout": rr.cut(res.get("stdout"), 3000), "final_answer": res.get("final_answer"),
                          "called": bool(res.get("called")), "seconds": res.get("seconds"),
                          "lost": bool(res.get("lost")), "enriched": enriched})
            if res.get("lost"):                # a timeout or a crash ends the session, as in the first experiment
                rec["status"] = "executor_lost"
                break
            if res.get("called") and res.get("ok"):
                break
            msgs.append({"role": "user", "content": obs})
        rec.update(cells=cells, usage=usage)
        rec.setdefault("status", "done")
        fin = next((c for c in cells if c.get("called") and c.get("ok")), None)
        rec["final_answer"] = fin["final_answer"] if fin else None
        rec["cost_usd"] = cost(model.model, usage)
        rec["wall_seconds"] = round(time.time() - t0, 1)
        if a.transcripts:
            os.makedirs(a.transcripts, exist_ok=True)
            with open(os.path.join(a.transcripts, f"{t['id']}_{arm}_{seed}.json"), "w") as f:
                json.dump(msgs, f)
        return rec
    finally:
        ex.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts")
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--cell-timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (pilot)")
    ap.add_argument("--order-seed", type=int, default=20261012)
    ap.add_argument("--budget-usd", type=float, default=0)
    a = ap.parse_args()
    a.lake = os.path.abspath(a.tables)
    schema, ambiguous, skipped = kb_gate.domain_schema(a.lake)
    where = {}
    for d, _, files in os.walk(a.lake):
        for f in files:
            where.setdefault(f, []).append(os.path.join("input", os.path.relpath(os.path.join(d, f), a.lake)))

    def locate(base):
        return where.get(base, [])

    model = Model(a.model, a.temperature, a.max_tokens)
    tasks = qr_run.load_tasks(a.questions)
    if a.limit:
        tasks = tasks[: a.limit]
    seeds = [int(s) for s in a.seeds.split(",")]
    # Seeds run as blocks, in the order given: block s holds every (question, arm) job of seed s, in an order
    # shuffled with order_seed + s, so the arms share each block's time window and a stop between blocks
    # leaves complete seeds.
    jobs = []
    for s in seeds:
        block = [(t, arm, s) for t in tasks for arm in a.arms.split(",")]
        random.Random(a.order_seed + s).shuffle(block)
        jobs += block
    done, spent = set(), [0.0]
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["task"], r["arm"], r["seed"]))
            spent[0] += r.get("cost_usd") or 0
    todo = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]
    print(f"gate {GATE_PATH} sha256 {sha256(GATE_PATH)[:16]}; runner sha256 "
          f"{sha256(os.path.join(HERE, 'e2_runner.py'))[:16]}; schema {len(schema)} files; {len(todo)} jobs to run; "
          f"spent so far ${spent[0]:.2f}", flush=True)
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor

    def job(j):
        t, arm, s = j
        if a.budget_usd and spent[0] >= a.budget_usd:
            return
        try:
            rec = run_task(t, arm, s, model, a, schema, locate)
        except Exception as e:  # noqa: BLE001  not written: the next start retries it
            print(f"{t['id']} {arm} {s}: error {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        rec.update(data_files=t["data_files"], question_type=t["question_type"])
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            spent[0] += rec.get("cost_usd") or 0
        nb = sum(1 for c in rec["cells"] if c.get("blocked"))
        print(f"{t['id']} {arm} s{s}: {rec['status']}, cells {len(rec['cells'])}, blocked {nb}, "
              f"answer {str(rec['final_answer'])[:30]!r}, ${rec['cost_usd']:.4f}, {rec['wall_seconds']} s; "
              f"total ${spent[0]:.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(job, todo))
    print(f"done; recorded cost ${spent[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
