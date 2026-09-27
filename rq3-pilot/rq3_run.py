#!/usr/bin/env python3
"""RQ3 pilot: continue recorded DABstep traces with one fixed model, twice.

For each episode (select_prefixes.py) and seed, the harness rebuilds the
state by running the earlier cells in a sandbox (sandbox.py), shows the
model the task, the earlier cells and their outputs, and the blocked cell,
and then gives one of two feedbacks:

  traceback  the blocked cell runs and fails; the model sees the traceback.
  receipt    the blocked cell does not run; the model sees the gate's receipt.

The model then writes up to --steps more cells. The gate is not used after
the blocked cell (--gate-after off), so the two arms differ only in the
feedback at that one point. Both arms use the same seed.

Usage (dry run with a scripted model, no API):
  python3 rq3-pilot/rq3_run.py --episodes $RQ3_RUN/episodes_cost10.jsonl --data dab \
      --out $RQ3_RUN/results_dry.jsonl --model dry --seeds 1 --limit 2
Real run (Ollama Cloud, key in OLLAMA_API_KEY or in a .env file):
  python3 rq3-pilot/rq3_run.py ... --model gpt-oss:120b --seeds 1 --max-seconds 160

The run is resumable: results already in --out are skipped, and
--max-seconds stops before starting a new episode-arm after that time.
"""
import argparse
import ast
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dabstep-gate"))
import sandbox  # noqa: E402
from recovery_episodes import gave_up, is_inspection  # noqa: E402

# USD per 1M tokens (input, output), Ollama pricing page, 23 September 2026.
PRICES = {"gpt-oss:120b": (0.15, 0.60), "gpt-oss:20b": (0.07, 0.30), "deepseek-v4-flash:0731": (0.44, 1.32), "deepseek-v4.1-flash": (0.30, 1.20),
          "glm-5.3": (1.40, 4.40), "kimi-k2.6": (0.95, 4.00)}
SYSTEM = (
    "You are a data analysis agent. You answer one question about the data files in the directory data/context/ "
    "(payments.csv, payments-readme.md, acquirer_countries.csv, fees.json, merchant_category_codes.csv, "
    "merchant_data.json, manual.md). You work in a Python session: names persist between cells. In each reply, "
    "think briefly, then write exactly one Python code block in ```python ... ```. Its printed output comes back "
    "to you. When you know the answer, call final_answer(answer) in a code block. Follow the answer guidelines "
    "exactly.")
OBS = 2500


# ---------------------------------------------------------------- messages
def task_message(ep):
    return (f"Question: {ep['question']}\nGuidelines: {ep.get('guidelines') or ''}\n"
            "The files are in data/context/. Earlier cells may use other paths to the same files; those paths "
            "also work.")


def cut(text, n=OBS):
    text = text or ""
    return text if len(text) <= n else text[: n // 2] + "\n...[cut]...\n" + text[-n // 2:]


def observation(res):
    parts = [res.get("stdout") or ""]
    if not res.get("ok"):
        tb = (res.get("error") or "").strip().splitlines()
        parts.append("\n".join(tb[-25:]))
    return "Observation:\n" + cut("\n".join(p for p in parts if p).strip() or "(no output)")


def receipt_text(rc):
    kind = rc.get("kind") or "name"
    what = {"column": "columns", "key": "keys", "file": "files"}.get(kind, "names")
    lines = [f"- {rc.get('location')}: {rc.get('reason')}."]
    av = rc.get("available") or []
    if av:
        shown = ", ".join(map(str, av[:60])) + (f", ... ({len(av)} in total)" if len(av) > 60 else "")
        src = "the workspace" if kind == "file" else rc.get("source")
        lines.append(f"- Available {what} in {src}: {shown}")
    if rc.get("near_matches"):
        lines.append("- Close matches: " + ", ".join(map(str, rc["near_matches"])))
    if rc.get("found_in"):
        lines.append(f"- Other files that have '{rc.get('missing')}': " + ", ".join(rc["found_in"]))
    return "\n".join(lines)


def feedback(arm, res, rc):
    if arm == "traceback":
        return observation(res) + "\nContinue."
    return ("Observation:\nThis cell was not run. A check before execution found that it uses data that does "
            "not exist.\n" + receipt_text(rc) + "\nContinue.")


CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_code(text):
    """The first fenced code block; else the whole reply if it is valid Python with a call
    (gpt-oss sometimes replies with a bare final_answer(...) line)."""
    m = CODE.search(text or "")
    if m:
        return m.group(1).strip()
    t = (text or "").strip()
    if t and len(t) < 2000:
        try:
            tree = ast.parse(t)
        except (SyntaxError, ValueError):
            return None
        if any(isinstance(n, ast.Call) for n in ast.walk(tree)):
            return t
    return None


def fence(code):
    return f"```python\n{code}\n```"


def mentions(code, name):
    """True if the code still uses the missing name as a string literal."""
    try:
        t = ast.parse(code)
    except (SyntaxError, ValueError):
        return name in (code or "")
    return any(isinstance(n, ast.Constant) and n.value == name for n in ast.walk(t))


# ----------------------------------------------------------------- models
def api_key():
    k = os.environ.get("OLLAMA_API_KEY")
    if k:
        return k
    for p in (os.path.join(HERE, "..", ".env"), ".env"):
        if os.path.exists(p):
            for line in open(p):
                if line.strip().startswith("OLLAMA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return None


class Ollama:
    """OpenAI-compatible chat completions on Ollama Cloud (https://ollama.com/v1)."""

    def __init__(self, model, temperature, max_tokens, base="https://ollama.com/v1"):
        self.model, self.temperature, self.max_tokens, self.base = model, temperature, max_tokens, base
        self.key = api_key()
        if not self.key:
            raise SystemExit("No API key: set OLLAMA_API_KEY or add OLLAMA_API_KEY=... to .env (gitignored).")

    def chat(self, messages, seed):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                           "max_tokens": self.max_tokens, "seed": seed}).encode()
        for attempt in range(4):
            req = urllib.request.Request(self.base + "/chat/completions", data=body, method="POST",
                                         headers={"Authorization": "Bearer " + self.key,
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.load(r)
                u = d.get("usage") or {}
                return (d["choices"][0]["message"].get("content") or "",
                        {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                         "model": d.get("model"), "created": d.get("created"),
                         "finish_reason": d["choices"][0].get("finish_reason")})
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise


class Dry:
    """A scripted model for tests: inspects the source once, then gives a constant answer."""
    model = "dry"

    def chat(self, messages, seed):
        n = sum(1 for m in messages if m["role"] == "assistant" and m["content"].startswith("Look at the columns."))
        text = ("Look at the columns.\n```python\nimport pandas as pd\nprint(pd.read_csv('data/context/payments.csv')"
                ".columns.tolist())\n```") if n == 0 else "Done.\n```python\nfinal_answer('Not Applicable')\n```"
        return text, {"prompt_tokens": sum(len(m["content"]) for m in messages) // 4, "completion_tokens": 30,
                      "model": "dry", "created": None, "finish_reason": "stop"}


# ---------------------------------------------------------------- episodes
def extra_paths(ep):
    """Where the sandbox puts context files, besides data/context/.

    Episodes selected after commit 95a0b9e carry sandbox_paths: every context
    file in every directory where the submission kept them, except files that
    the trace failed to find (context_layout.py). Older episodes carry only
    context_paths, the files that the prefix itself read; a model that read
    another file from the same directory got an error that the original run
    would not have had."""
    return dict(ep.get("sandbox_paths") or ep.get("context_paths") or {})


def run_episode(ep, arm, seed, model, a):
    ex = sandbox.Executor(os.path.join(a.data, "data/context"), extra_paths(ep), mode=a.sandbox, timeout=a.cell_timeout)
    rec = {"episode": ep["episode"], "arm": arm, "seed": seed, "model": getattr(model, "model", None),
           "submitter": ep["submitter"], "task_id": ep["task_id"], "kind": ep["receipt"].get("kind"),
           "missing": ep["receipt"].get("missing"), "params": {"temperature": a.temperature, "max_tokens": a.max_tokens,
                                                               "steps": a.steps, "gate_after": "off"}}
    try:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_message(ep)}]
        mismatch = 0
        for c in ep["prefix"]:
            res = ex.run(c["code"] or "")
            mismatch += bool(res.get("ok")) != (not c["failed"])
            if res.get("lost"):
                rec.update(status="executor_lost_in_prefix")
                return rec
            msgs += [{"role": "assistant", "content": fence(c["code"] or "")},
                     {"role": "user", "content": observation(res)}]
        rec["prefix_cells"], rec["prefix_mismatch"] = len(ep["prefix"]), mismatch
        blocked = ep["blocked"]["code"] or ""
        msgs.append({"role": "assistant", "content": fence(blocked)})
        if arm == "traceback":
            res = ex.run(blocked)
            rec["blocked_fails"] = not res.get("ok")
            rec["blocked_names_missing"] = (not res.get("ok")) and (str(ep["receipt"].get("missing")) in
                                                                    ((res.get("error") or "") + (res.get("stdout") or "")))
        else:
            res = None
        msgs.append({"role": "user", "content": feedback(arm, res, ep["receipt"])})
        steps, usage = [], {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        for k in range(1, a.steps + 1):
            text, u = model.chat([{kk: vv for kk, vv in m.items() if not kk.startswith("_")} for m in msgs], seed)
            usage["prompt_tokens"] += u["prompt_tokens"]
            usage["completion_tokens"] += u["completion_tokens"]
            usage["calls"] += 1
            rec["served_model"] = u.get("model")
            msgs.append({"role": "assistant", "content": text, "_new": True})
            code = extract_code(text)
            st = {"step": k, "finish_reason": u.get("finish_reason")}
            if code is None:
                st["no_code"] = True
                steps.append(st)
                msgs.append({"role": "user", "content": "Please reply with exactly one Python code block."})
                continue
            res = ex.run(code)
            st.update(code=code, ok=bool(res.get("ok")), error_type=res.get("error_type"),
                      inspection=is_inspection(code), constant_answer=gave_up(code),
                      repeats_missing=mentions(code, str(ep["receipt"].get("missing"))),
                      final_answer=res.get("final_answer"))
            steps.append(st)
            if res.get("lost"):
                rec["status"] = "executor_lost"
                break
            if res.get("final_answer") is not None and res.get("ok"):
                break
            msgs.append({"role": "user", "content": observation(res)})
        rec.update(steps=steps, usage=usage)
        rec.setdefault("status", "done")
        fix = next((s["step"] for s in steps if s.get("ok") and not s.get("inspection")), None)
        rec["steps_to_fix"] = fix          # as k_fix in recovery_episodes.py: a constant final answer counts
        rec["never_recovered"] = fix is None
        rec["steps_to_fix_nonconstant"] = next((s["step"] for s in steps if s.get("ok") and not s.get("inspection")
                                                and not s.get("constant_answer")), None)
        fin = next((s for s in steps if s.get("final_answer") is not None and s.get("ok")), None)
        rec["final_answer"] = fin["final_answer"] if fin else None
        rec["constant_answer"] = bool(fin and fin.get("constant_answer"))
        rec["first_step_repeats_missing"] = bool(steps and steps[0].get("repeats_missing"))
        pr = PRICES.get(rec["model"])
        rec["cost_usd"] = (usage["prompt_tokens"] * pr[0] + usage["completion_tokens"] * pr[1]) / 1e6 if pr else None
        if a.transcripts:
            os.makedirs(a.transcripts, exist_ok=True)
            with open(os.path.join(a.transcripts, f"{ep['episode']}_{arm}_{seed}.json"), "w") as f:
                json.dump([{k: v for k, v in m.items() if not k.startswith("_")} for m in msgs], f, indent=1)
        return rec
    finally:
        ex.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="an Ollama Cloud model id, or 'dry'")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--cell-timeout", type=int, default=120)
    ap.add_argument("--sandbox", choices=["bwrap", "local"], default="bwrap")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop before a new episode-arm after this time")
    ap.add_argument("--transcripts", help="directory for one JSON transcript per episode-arm-seed")
    ap.add_argument("--workers", type=int, default=1, help="episode-arms run at the same time (Ollama Pro allows 3 requests)")
    ap.add_argument("--seed-major", action="store_true",
                    help="run every episode with seed 1 before any with seed 2, so that a stopped run keeps whole seeds")
    ap.add_argument("--budget-usd", type=float, default=0,
                    help="stop starting episode-arms when the cost recorded in --out reaches this amount")
    a = ap.parse_args()
    t0 = time.time()
    model = Dry() if a.model == "dry" else Ollama(a.model, a.temperature, a.max_tokens)
    eps = [json.loads(line) for line in open(a.episodes)]
    if a.limit:
        eps = eps[: a.limit]
    done, spent = set(), [0.0]
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["episode"], r["arm"], r["seed"]))
            spent[0] += r.get("cost_usd") or 0
    if a.seed_major:
        order = [(ep, arm, s) for s in range(1, a.seeds + 1) for ep in eps for arm in ("traceback", "receipt")]
    else:
        order = [(ep, arm, s) for ep in eps for s in range(1, a.seeds + 1) for arm in ("traceback", "receipt")]
    todo = [x for x in order if (x[0]["episode"], x[1], x[2]) not in done]
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    count = [0]

    def job(item):
        ep, arm, seed = item
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            return
        if a.budget_usd and spent[0] >= a.budget_usd:
            return
        try:
            rec = run_episode(ep, arm, seed, model, a)
        except Exception as e:  # an API or harness failure: not written, so the next run retries it
            print(f"{ep['episode']} {arm:9s} seed {seed}: error {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        with lock:
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            count[0] += 1
            spent[0] += rec.get("cost_usd") or 0
        print(f"{ep['episode']} {arm:9s} seed {seed}: {rec.get('status')}, steps to fix {rec.get('steps_to_fix')}, "
              f"cost {rec.get('cost_usd')}", flush=True)

    with open(a.out, "a") as out, ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(job, todo))
    n = count[0]
    left = len(todo) - n
    print(f"ran {n}; {left} episode-arms left; recorded cost ${spent[0]:.2f}" + ("; run again to continue" if left else ""))


if __name__ == "__main__":
    main()
