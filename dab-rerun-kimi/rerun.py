"""Second controlled DataAgentBench rerun, with kimi-k2.6: released harness (A) against the code passed
unchanged (B). Pre-registration: dab-rerun-kimi/preregistration.md.

The first rerun (dab-rerun/rerun.py, gpt-oss:120b) is unchanged. Differences from it: the model; arms A
and B only; prices for the ledger; no wait at block boundaries after block 0 (amendment 4 of the first
rerun, now part of the design); OUT and PY paths.

Usage:
  python rerun.py run      # start or resume all jobs (3 workers)
  python rerun.py status   # print progress and cost
Resumable: a job with a final_agent.json whose run did not fail for infrastructure reasons is skipped.
"""
import json, os, queue, random, shutil, subprocess, sys, time, fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("DAB_ROOT", str(Path.home() / "dabx")))
OUT = Path(os.environ.get("DAB_OUT", str(Path.home() / "dabx-runs-kimi")))
PY = os.environ.get("DAB_PY", str(Path.home() / "dab-venv/bin/python"))
MODEL = "ollama/kimi-k2.6"
PRICES = {"DAB_PRICE_IN": "0.95", "DAB_PRICE_CACHED": "0.16", "DAB_PRICE_OUT": "4.00"}  # USD per 1M tokens
SEED = 20261012
BLOCKS = int(os.environ.get("DAB_BLOCKS", "5"))   # runs per query and arm (one per block)
SOFT_CAP = float(os.environ.get("DAB_SOFT_CAP", "38.0"))
HARD_CAP = float(os.environ.get("DAB_HARD_CAP", "40.0"))
WALL_CAP_S = 2 * 3600           # per-run wall-clock cap
MAX_INFRA_RETRIES = 2
WORKERS = int(os.environ.get("DAB_WORKERS", "3"))

DATASETS = {  # the 54 leaderboard queries (12 datasets)
    "DEPS_DEV_V1": 2, "GITHUB_REPOS": 4, "PANCANCER_ATLAS": 3, "PATENTS": 3, "agnews": 4,
    "bookreview": 3, "crmarenapro": 13, "googlelocal": 4, "music_brainz_20k": 3,
    "stockindex": 3, "stockmarket": 5, "yelp": 7,
}
MONGO_DATASETS = ("yelp", "agnews")  # arm C runs only where MongoDB results can be cut


def jobs_for_block(k):
    js = [(arm, ds, q, k) for arm in ("A", "B") for ds, n in DATASETS.items() for q in range(1, n + 1)]
    random.Random(SEED + k).shuffle(js)
    return js


def run_dir(arm, ds, q, k, attempt=0):
    name = f"{arm}_run_{k}" + (f"_retry{attempt}" if attempt else "")
    return ROOT / f"query_{ds}" / f"query{q}" / "logs" / "data_agent" / name


def is_infra(reason):
    r = reason or ""
    return r.startswith("llm_response_failed") or (r.startswith("agent_run_failed") and "budget cap" not in r)


def summarise(arm, ds, q, k, d, attempt, wall):
    sys.path.insert(0, str(ROOT))
    from common_scaffold.validate.validate import validate
    qdir = ROOT / f"query_{ds}" / f"query{q}"
    val = validate(qdir, d.get("final_result") or "", d.get("terminate_reason"))
    usage = [json.loads(l) for l in open(d["_usage_path"])] if os.path.exists(d["_usage_path"]) else []
    tool_calls, failed = 0, 0
    tc_path = Path(d["tool_log_path"])
    if tc_path.exists():
        for line in open(tc_path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            tool_calls += 1
            if (r.get("result") or {}).get("success") is False:
                failed += 1
    return {
        "arm": arm, "dataset": ds, "query": q, "block": k, "attempt": attempt,
        "is_valid": bool(val["is_valid"]), "terminate_reason": d.get("terminate_reason"),
        "llm_calls": d.get("llm_call_count"), "tool_calls": tool_calls, "failed_tool_calls": failed,
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage),
        "completion_tokens": sum(u["completion_tokens"] for u in usage),
        "cached_tokens": sum(u.get("cached_tokens", 0) for u in usage),
        "cost_usd": round(sum(u.get("cost_usd", 0) for u in usage), 5),
        "wall_s": round(wall, 1), "final_result": (d.get("final_result") or "")[:500],
    }


def append_summary(rec):
    with open(OUT / "runs.jsonl.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(OUT / "runs.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def done_keys():
    keys = set()
    if (OUT / "runs.jsonl").exists():
        for line in open(OUT / "runs.jsonl"):
            r = json.loads(line)
            if not is_infra(r["terminate_reason"]) or r["attempt"] >= MAX_INFRA_RETRIES:
                keys.add((r["arm"], r["dataset"], r["query"], r["block"]))
    return keys


def kill_run_containers(rd):
    """After a wall-clock kill, stop the executor container that mounted this run's work folder."""
    ids = subprocess.run(["docker", "ps", "-q", "--filter", "ancestor=python-data:3.12"],
                         capture_output=True, text=True).stdout.split()
    for cid in ids:
        mounts = subprocess.run(["docker", "inspect", "--format", "{{range .Mounts}}{{.Source}} {{end}}", cid],
                                capture_output=True, text=True).stdout
        if str(rd) in mounts:
            subprocess.run(["docker", "kill", cid], capture_output=True)


def ledger_cost():
    p = OUT / "ledger.json"
    return json.load(open(p))["cost_usd"] if p.exists() else 0.0


def run_job(job, worker):
    arm, ds, q, k = job
    for attempt in range(MAX_INFRA_RETRIES + 1):
        if ledger_cost() >= SOFT_CAP:
            return "budget"
        rd = run_dir(arm, ds, q, k, attempt)
        if rd.exists():
            shutil.rmtree(rd)  # a partial run from an interrupted session
        usage_path = OUT / "usage" / f"{arm}_{ds}_{q}_{k}_{attempt}.jsonl"
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        if usage_path.exists():
            usage_path.unlink()
        env = dict(os.environ)
        env.update(PRICES)
        env.update({
            "DAB_ARM": arm, "DAB_LEDGER": str(OUT / "ledger.json"), "DAB_RUN_USAGE": str(usage_path),
            "DAB_HARD_CAP_USD": str(HARD_CAP),
            "PG_HOST": "127.0.0.1", "PG_PORT": str(5432 + worker), "PGPORT": str(5432 + worker),
            "MONGO_URI": f"mongodb://127.0.0.1:{27017 + worker}/",
        })
        cmd = [PY, "run_agent.py", "--dataset", ds, "--query_id", str(q), "--llm", MODEL,
               "--iterations", "100", "--use_hints", "--root_name", rd.name]
        t0 = time.time()
        log = open(OUT / "logs" / f"{arm}_{ds}_{q}_{k}_{attempt}.out", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=WALL_CAP_S)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, 9)
            except Exception:
                pass
            proc.wait()
            kill_run_containers(rd)
        wall = time.time() - t0
        fa = rd / "final_agent.json"
        if timed_out or not fa.exists():
            d = {"final_result": "", "terminate_reason": "wall_clock_cap" if timed_out else "agent_run_failed (no final_agent.json)",
                 "llm_call_count": None, "tool_log_path": str(rd / "tool_calls.jsonl")}
        else:
            d = json.load(open(fa))
        d["_usage_path"] = str(usage_path)
        rec = summarise(arm, ds, q, k, d, attempt, wall)
        rec["worker"] = worker
        append_summary(rec)
        if not is_infra(rec["terminate_reason"]):
            return "ok"
    return "infra_failed"


def run_all():
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if (OUT / "STOP_ASK_USER").exists():
        print("STOP_ASK_USER present: waiting for Blessed's decision"); return
    plan = json.load(open(OUT / "plan.json")) if (OUT / "plan.json").exists() else {"blocks": BLOCKS}
    k = 0
    while k < plan["blocks"]:
        if k == 0:
            todo = [j for j in jobs_for_block(0) if j not in done_keys()]
        else:   # blocks 1 onward run in block order without waiting at the boundaries
            todo = [j for kk in range(1, plan["blocks"]) for j in jobs_for_block(kk) if j not in done_keys()]
        slots = queue.Queue()
        for w in range(1, WORKERS + 1):
            slots.put(w)  # each slot owns one PostgreSQL and one MongoDB instance

        def with_slot(job):
            w = slots.get()
            try:
                return run_job(job, w)
            finally:
                slots.put(w)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for f in [pool.submit(with_slot, job) for job in todo]:
                f.result()
        if k == 0 and not (OUT / "plan.json").exists():
            plan = project_budget()
        k = 1 if k == 0 else plan["blocks"]
        if ledger_cost() >= SOFT_CAP:
            break
    status()


def project_budget():
    runs = [json.loads(l) for l in open(OUT / "runs.jsonl")]
    block0 = [r for r in runs if r["block"] == 0]
    cost0 = ledger_cost()
    per_block = cost0  # block 0 is the first full block, so its cost is the cost per block
    blocks = BLOCKS
    while blocks > 0 and per_block * blocks > SOFT_CAP:
        blocks -= 1
    plan = {"blocks": blocks, "cost_block0": cost0, "projected_total": per_block * blocks,
            "decided_at": time.time()}
    if blocks < 3 and not os.environ.get("DAB_PILOT"):
        # Pre-registered rule: fewer than 3 blocks fit, so stop after block 0 and ask Blessed.
        (OUT / "STOP_ASK_USER").write_text("Projected cost allows fewer than 3 runs per query; stop and ask.\n")
        plan["blocks_affordable"] = blocks
        plan["blocks"] = 1
    json.dump(plan, open(OUT / "plan.json", "w"), indent=2)
    return plan


def status():
    runs = [json.loads(l) for l in open(OUT / "runs.jsonl")] if (OUT / "runs.jsonl").exists() else []
    final = {}
    for r in runs:
        final[(r["arm"], r["dataset"], r["query"], r["block"])] = r
    by_arm = {}
    for (arm, *_), r in final.items():
        a = by_arm.setdefault(arm, {"runs": 0, "valid": 0, "infra": 0})
        a["runs"] += 1
        a["valid"] += int(r["is_valid"])
        a["infra"] += int(is_infra(r["terminate_reason"]))
    total = sum(len(jobs_for_block(k)) for k in range(BLOCKS))
    import statistics as _st
    costs = [r.get("cost_usd") for r in final.values() if r.get("cost_usd") is not None]
    if costs:
        print(f"mean cost per run ${_st.mean(costs):.4f}; median wall {(_st.median([r['wall_s'] for r in final.values()])):.0f} s")
    print(json.dumps({"done": len(final), "planned_max": total, "cost_usd": round(ledger_cost(), 3),
                      "by_arm": by_arm}, indent=1))


def pilot():
    """Cost and set-up check before the freeze: the given jobs only, written to OUT/pilot (never analysed)."""
    global OUT
    OUT = OUT / "pilot"
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [tuple(j.split(":")) for j in sys.argv[2:]]          # arm:dataset:query
    jobs = [(a, d, int(q), 0) for a, d, q in jobs]
    slots = queue.Queue()
    for w in range(1, WORKERS + 1):
        slots.put(w)

    def with_slot(job):
        w = slots.get()
        try:
            return run_job(job, w)
        finally:
            slots.put(w)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        print([f.result() for f in [pool.submit(with_slot, j) for j in jobs]])
    status()


if __name__ == "__main__":
    {"run": run_all, "status": status, "pilot": pilot}[sys.argv[1]]()
