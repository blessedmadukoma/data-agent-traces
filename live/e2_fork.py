#!/usr/bin/env python3
"""E2 with shared prefixes (amendment 1 of live/e2_preregistration.md).

The three arms N, R and E behave identically until a step where they differ: a step that the checker blocks
(R) or that raises a missing-name error (E, and the receipt in R). Arms whose message histories are identical
therefore share one model call per step. Each (question, seed) job starts with one branch that holds all three
arms. At every step, each arm in the branch handles the step as in live/e2_run.py, with its own sandbox and its
own checker state; arms whose next message, end state and sandbox result are identical stay together, and the
branch splits where they differ. After a split, each branch samples its own replies. Every arm's trajectory is
therefore a run of that arm, and a model reply is shared only between arms that sent identical messages.

Each arm still has its own Docker sandbox, which runs every step of that arm, so its state is its own.

Output: one record per (question, arm, seed), in the format of live/e2_run.py, plus:
  shared_with   for each step, the arms that shared that model call;
  calls_billed  the model calls that this job made (once per branch), in the arm record that lists them first;
  job_cost_usd  the job's billed cost (in each record of the job).
cost_usd and usage are the arm's trajectory: every call on that arm's path, shared or not.

Usage: OLLAMA_API_KEY=... python3 live/e2_fork.py --tables data/QRData/benchmark/data \\
           --questions data/QRData/benchmark/QRData.json --out runs/e2/results.jsonl --transcripts runs/e2/tr \\
           --seeds 1,...,10 --workers 4 --budget-usd 11
"""
import argparse
import copy
import json
import os
import random
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2_run as base  # noqa: E402  (loads checker release 4 and the E1 modules)

gate, kb_gate, kb_run, rr, qr_run = base.gate, base.kb_gate, base.kb_run, base.rr, base.qr_run
ARMS = base.ARMS


class Arm:
    """One arm's record: its cells, token use and end state."""

    def __init__(self, name):
        self.name = name
        self.cells = []
        self.usage = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "calls": 0, "model_seconds": 0.0}
        self.cost = 0.0
        self.status = None          # None while running; "done" or "executor_lost" at the end
        self.final = None
        self.shared = []
        self.msgs = None
        self.served = None


class Branch:
    """Arms whose message histories are identical. They share one sandbox, one checker state and the list of
    code that the sandbox has executed (to rebuild the sandbox when the branch splits)."""

    def __init__(self, names, msgs, ex, st, executed):
        self.names, self.msgs, self.ex, self.st, self.executed = names, msgs, ex, st, executed


def cell_from(res, k, code, g, rc, arm_name, enriched):
    return {"step": k, "code": code, "blocked": False, "gate": g["outcome"],
            "shadow_receipt": rc if arm_name != "R" else None, "ok": bool(res.get("ok")),
            "error_type": res.get("error_type"), "error": (res.get("error") or "")[-6000:],
            "stdout": rr.cut(res.get("stdout"), 3000), "final_answer": res.get("final_answer"),
            "called": bool(res.get("called")), "seconds": res.get("seconds"),
            "lost": bool(res.get("lost")), "enriched": enriched}


def rebuild(a, executed, stats):
    """A fresh sandbox that has executed the same code, in order. Counts steps whose outcome differs."""
    ex = base.DockerExecutor(a.lake, timeout=a.cell_timeout)
    stats["replays"] += 1
    for code, ok, etype in executed:
        res = ex.run(code)
        if res.get("lost"):
            stats["replay_lost"] += 1
            return ex, False
        if bool(res.get("ok")) != ok or (not ok and res.get("error_type") != etype):
            stats["replay_mismatch"] += 1
    return ex, True


def run_group(t, seed, model, a, schema, locate):
    arms = {n: Arm(n) for n in a.arms}
    billed = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "calls": 0}
    stats = {"replays": 0, "replay_mismatch": 0, "replay_lost": 0}
    sandboxes = []
    t0 = time.time()
    try:
        n, listing = kb_run.lake_listing(a.lake)
        start = [{"role": "system", "content": base.SYSTEM},
                 {"role": "user", "content": f"Question: {t['query']}\n\nFiles in input/ ({n} files):\n{listing}"}]
        ex0 = base.DockerExecutor(a.lake, timeout=a.cell_timeout)
        sandboxes.append(ex0)
        branches = [Branch(list(a.arms), start, ex0, gate.GateState(), [])]
        k = 0
        while branches and k < a.steps:
            k += 1
            nxt = []
            for br in branches:
                text, u = model.chat(br.msgs, seed)
                for key in ("prompt_tokens", "cached_tokens", "completion_tokens"):
                    billed[key] += u[key]
                billed["calls"] += 1
                c = base.cost(model.model, u)
                msgs = br.msgs + [{"role": "assistant", "content": text}]
                for nm in br.names:
                    arm = arms[nm]
                    for key in ("prompt_tokens", "cached_tokens", "completion_tokens"):
                        arm.usage[key] += u[key]
                    arm.usage["calls"] += 1
                    arm.usage["model_seconds"] += u["seconds"]
                    arm.cost += c
                    arm.shared.append(sorted(br.names))
                    arm.served = u.get("model")
                    arm.msgs = msgs
                code = rr.extract_code(text)
                if code is None:
                    for nm in br.names:
                        arms[nm].cells.append({"step": k, "no_code": True})
                    nxt.append(Branch(br.names, msgs + [{"role": "user", "content": "Please reply with exactly one Python code block."}],
                                      br.ex, br.st, br.executed))
                    continue
                try:
                    g = gate.decide(code, schema, br.st, exists=br.ex.exists, locate=locate)
                except Exception as e:  # noqa: BLE001  a checker crash lets the step run
                    g = {"outcome": "gate_error", "reason": f"{type(e).__name__}: {e}"}
                rc = {x: g.get(x) for x in ("location", "kind", "source", "missing", "reason", "available",
                                            "found_in", "near_matches")} if g["outcome"] == "block" else None
                blocked = [nm for nm in br.names if nm == "R" and g["outcome"] == "block"]
                runs = [nm for nm in br.names if nm not in blocked]
                s0 = copy.deepcopy(br.st)                     # checker state before this step
                executed_before = list(br.executed)
                res = br.ex.run(code) if runs else None
                groups = []                                   # (names, reply, executed the step?)
                if blocked:
                    arms["R"].cells.append({"step": k, "code": code, "blocked": True, "gate": "block",
                                            "receipt": rc, "ok": None})
                    groups.append((["R"], base.BLOCKED + base.receipt_text(rc) + "\nContinue.", False))
                if runs:
                    obs = rr.observation(res)
                    name_err = (not res.get("ok")) and res.get("error_type") in base.MISSING_NAME_ERRORS
                    ended = bool(res.get("lost")) or bool(res.get("called") and res.get("ok"))
                    by_reply = {}
                    for nm in runs:
                        enriched = nm == "E" and name_err
                        arms[nm].cells.append(cell_from(res, k, code, g, rc, nm, enriched))
                        if res.get("lost"):
                            arms[nm].status = "executor_lost"
                        elif res.get("called") and res.get("ok"):
                            arms[nm].status = "done"
                            arms[nm].final = res.get("final_answer")
                        if ended:
                            continue
                        reply = obs + ("\n" + base.names_text(a.lake, t["data_files"]) if enriched else "")
                        by_reply.setdefault(reply, []).append(nm)
                    for reply, members in by_reply.items():
                        groups.append((members, reply, True))
                original_taken = False
                for members, reply, did_run in groups:
                    st = copy.deepcopy(s0)
                    if did_run:
                        st.commit(failed=not res.get("ok"))
                        executed = executed_before + [(code, bool(res.get("ok")), res.get("error_type"))]
                    else:
                        st.commit(failed=True)                # a blocked step counts as failed
                        executed = executed_before
                    # the original sandbox holds this step if it ran; it goes to the first group whose state matches
                    if not original_taken and (did_run or not runs):
                        ex, original_taken = br.ex, True
                    else:
                        ex, ok = rebuild(a, executed, stats)
                        sandboxes.append(ex)
                        if not ok:
                            for nm in members:
                                arms[nm].status = "executor_lost"
                            continue
                    nxt.append(Branch(members, msgs + [{"role": "user", "content": reply}], ex, st, executed))
            branches = nxt
        wall = round(time.time() - t0, 1)
        if a.transcripts:
            os.makedirs(a.transcripts, exist_ok=True)
            for nm, arm in arms.items():
                with open(os.path.join(a.transcripts, f"{t['id']}_{nm}_{seed}.json"), "w") as f:
                    json.dump(arm.msgs or start, f)
        recs = []
        job_cost = base.cost(model.model, billed)
        for i, nm in enumerate(a.arms):
            arm = arms[nm]
            recs.append({"task": t["id"], "arm": nm, "seed": seed, "model": model.model,
                         "served_model": arm.served, "design": "shared_prefix_v2",
                         "params": {"temperature": a.temperature, "max_tokens": a.max_tokens, "steps": a.steps,
                                    "cell_timeout": a.cell_timeout},
                         "cells": arm.cells, "usage": arm.usage, "status": arm.status or "done",
                         "final_answer": arm.final, "cost_usd": arm.cost, "shared_with": arm.shared,
                         "calls_billed": billed["calls"] if i == 0 else None, "job_cost_usd": job_cost,
                         "sandbox_rebuilds": stats if i == 0 else None, "wall_seconds": wall})
        return recs, job_cost
    finally:
        for ex in sandboxes:
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
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (pilot)")
    ap.add_argument("--order-seed", type=int, default=20261012)
    ap.add_argument("--budget-usd", type=float, default=0)
    a = ap.parse_args()
    a.arms = a.arms.split(",")
    a.lake = os.path.abspath(a.tables)
    schema, _, _ = kb_gate.domain_schema(a.lake)
    where = {}
    for d, _, files in os.walk(a.lake):
        for f in files:
            where.setdefault(f, []).append(os.path.join("input", os.path.relpath(os.path.join(d, f), a.lake)))

    def locate(base_name):
        return where.get(base_name, [])

    model = base.Model(a.model, a.temperature, a.max_tokens)
    tasks = qr_run.load_tasks(a.questions)
    if a.limit:
        tasks = tasks[: a.limit]
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = []
    for s in seeds:                                   # one block per seed, shuffled with order_seed + s
        block = [(t, s) for t in tasks]
        random.Random(a.order_seed + s).shuffle(block)
        jobs += block
    done, spent = set(), [0.0]
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if (r["task"], r["seed"]) not in done:
                done.add((r["task"], r["seed"]))
                spent[0] += r.get("job_cost_usd") or 0
    todo = [j for j in jobs if (j[0]["id"], j[1]) not in done]
    print(f"gate {base.GATE_PATH} sha256 {base.sha256(base.GATE_PATH)[:16]}; runner sha256 "
          f"{base.sha256(os.path.join(HERE, 'e2_runner.py'))[:16]}; schema {len(schema)} files; {len(todo)} jobs "
          f"(question, seed) to run; spent so far ${spent[0]:.2f}", flush=True)
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor

    def job(j):
        t, s = j
        if a.budget_usd and spent[0] >= a.budget_usd:
            return
        try:
            recs, jc = run_group(t, s, model, a, schema, locate)
        except Exception as e:  # noqa: BLE001  not written: the next start retries it
            print(f"{t['id']} s{s}: error {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        for r in recs:
            r.update(data_files=t["data_files"], question_type=t["question_type"])
        with lock:
            with open(a.out, "a") as f:
                for r in recs:
                    f.write(json.dumps(r, default=str) + "\n")
            spent[0] += jc
        branches = len({tuple(x) for r in recs for x in r["shared_with"]})
        print(f"{t['id']} s{s}: calls {recs[0]['calls_billed']}, steps " +
              "/".join(str(len(r['cells'])) for r in recs) + f", blocks {sum(1 for r in recs for c in r['cells'] if c.get('blocked'))}, "
              f"branch sets {branches}, ${jc:.4f}, {recs[0]['wall_seconds']} s; total ${spent[0]:.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(job, todo))
    print(f"done; recorded cost ${spent[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
