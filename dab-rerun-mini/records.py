"""Per-run call statistics of the gpt-5-mini DAB rerun, and a check of every reported number (not pre-registered).

The frozen analysis (analysis_mini.py) reads the run folders that DAB's harness writes, which contain DAB's data and are
not redistributed. This script lets a reader check the reported numbers without them:

  python3 dab-rerun-mini/records.py export   # needs the run folders; writes dab-rerun-mini/calls.jsonl
  python3 dab-rerun-mini/records.py          # needs only runs.jsonl and calls.jsonl; prints every number

calls.jsonl has one line per run attempt: arm, dataset, query, block, attempt, and counts of tool calls, Python calls,
failed calls, failed calls with a syntax error, Python calls with a syntax error that the embedding introduced
("harness", the frozen rule of dab-rerun/analysis.py), and Python calls whose code contains the escape sequence
backslash-n or a doubled backslash before n or t. It contains no code, answers or data.

The check uses the frozen statistics of dab-rerun/analysis.py (pass_rates, dab_pass1, boot_diff, boot_metric) and
compares its results with analysis.json and analysis_mini.json. The splits of the tasks (by whether the released harness broke a
step, and by whether the code contained the escape sequence backslash-n) are post hoc.
"""
import collections, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mini = load("rerun", HERE / "rerun.py")
mini.MONGO_DATASETS = ("yelp", "agnews")
sys.modules["rerun"] = mini
an = load("analysis_frozen", REPO / "dab-rerun" / "analysis.py")


def read_runs():
    """Last attempt per job, as the frozen load_runs; infrastructure failures excluded; complete blocks only."""
    final = {}
    for line in open(HERE / "runs.jsonl"):
        r = json.loads(line)
        final[(r["arm"], r["dataset"], r["query"], r["block"])] = r
    kept = {k: r for k, r in final.items() if not an.is_infra(r["terminate_reason"] or "")
            and "budget cap" not in (r["terminate_reason"] or "")}
    n = collections.Counter(k[3] for k in kept)
    complete = sorted(b for b, c in n.items() if c >= 100)
    return {k: r for k, r in kept.items() if k[3] in complete}, complete


def code_of(c):
    v = c.get("val_args")
    return v.get("code") if isinstance(v, dict) and isinstance(v.get("code"), str) else None


def export():
    out = open(HERE / "calls.jsonl", "w")
    n = 0
    for line in open(HERE / "runs.jsonl"):
        r = json.loads(line)
        key = (r["arm"], r["dataset"], r["query"], r["block"])
        calls = an.tool_calls(key, r)
        infos = [(an.call_info(r["arm"], c), code_of(c)) for c in calls]
        py = [(i, code) for i, code in infos if i["tool"] == "execute_python"]
        rec = {"arm": r["arm"], "dataset": r["dataset"], "query": r["query"], "block": r["block"],
               "attempt": r["attempt"], "tool_calls": len(infos), "python_calls": len(py),
               "failed": sum(not i["ok"] for i, _ in infos),
               "failed_syntax": sum(i["syntax"] for i, _ in infos if not i["ok"]),
               "harness": sum(i["harness"] for i, _ in py),
               "python_with_escape_n": sum(1 for _, code in py if code and "\\n" in code.replace("\\\\n", "")),
               "python_with_doubled_escape": sum(1 for _, code in py if code and ("\\\\n" in code or "\\\\t" in code))}
        out.write(json.dumps(rec) + "\n")
        n += 1
    out.close()
    print(f"wrote {n} lines to calls.jsonl")


def check():
    runs, complete = read_runs()
    calls = {}
    for line in open(HERE / "calls.jsonl"):
        c = json.loads(line)
        calls[(c["arm"], c["dataset"], c["query"], c["block"], c["attempt"])] = c
    per_run = {k: calls[k + (r["attempt"],)] for k, r in runs.items()}
    frozen = json.load(open(HERE / "analysis.json"))
    added = json.load(open(HERE / "analysis_mini.json"))
    res, ok = {}, True

    def same(name, mine, ref, tol=1e-9):
        nonlocal ok
        good = abs(mine - ref) <= tol
        ok &= good
        print(f"{'ok      ' if good else 'MISMATCH'} {name}: {mine:.6g} (published {ref:.6g})")

    print("complete blocks", complete)
    for arm in ("A", "B"):
        pr = [v for k, v in per_run.items() if k[0] == arm]
        pc = sum(v["python_calls"] for v in pr)
        h = sum(v["harness"] for v in pr)
        same(f"{arm} runs", len(pr), frozen["H1"][arm]["runs"])
        same(f"{arm} Python calls", pc, frozen["H1"][arm]["python_calls"])
        same(f"{arm} harness-caused syntax errors", h, frozen["H1"][arm]["harness_caused"])
        same(f"{arm} runs with one", sum(v["harness"] > 0 for v in pr), frozen["H1"][arm]["runs_with_harness_caused"])
        print(f"         {arm} Python calls with \\n: {sum(v['python_with_escape_n'] for v in pr)}, with a doubled escape: "
              f"{sum(v['python_with_doubled_escape'] for v in pr)}")

    ds_all = list(mini.DATASETS)
    prA, prB = an.pass_rates(runs, "A", ds_all), an.pass_rates(runs, "B", ds_all)
    same("Pass@1 A", an.dab_pass1(prA, ds_all), frozen["pass1"]["A"])
    same("Pass@1 B", an.dab_pass1(prB, ds_all), frozen["pass1"]["B"])
    d = an.boot_diff(prB, prA, ds_all)
    same("Pass@1 B - A", d["estimate"], frozen["H2_B_minus_A"]["estimate"])
    same("  CI low", d["ci95"][0], frozen["H2_B_minus_A"]["ci95"][0])
    same("  CI high", d["ci95"][1], frozen["H2_B_minus_A"]["ci95"][1])
    same("  Wilcoxon p", d["wilcoxon_p"], frozen["H2_B_minus_A"]["wilcoxon_p"])

    def per_task(arm, f):
        acc = collections.defaultdict(list)
        for (a, ds, q, k), v in per_run.items():
            if a == arm:
                acc[(ds, q)].append(f(v, runs[(a, ds, q, k)]))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    m = an.boot_metric(per_task("B", lambda v, r: v["failed"]), per_task("A", lambda v, r: v["failed"]))
    same("failed calls per run B - A", m["estimate"], frozen["H3_failed_calls_B_minus_A"]["estimate"])
    same("  CI low", m["ci95"][0], frozen["H3_failed_calls_B_minus_A"]["ci95"][0])
    same("  CI high", m["ci95"][1], frozen["H3_failed_calls_B_minus_A"]["ci95"][1])
    same("  Wilcoxon p", m["wilcoxon_p"], frozen["H3_failed_calls_B_minus_A"]["wilcoxon_p"])
    s = an.boot_metric(per_task("B", lambda v, r: v["failed"]),
                       per_task("A", lambda v, r: v["failed"] - v["harness"]))
    same("subtraction check", s["estimate"], added["H3_B_failed_minus_A_subtraction"]["estimate"])
    same("  CI low", s["ci95"][0], added["H3_B_failed_minus_A_subtraction"]["ci95"][0])
    same("  CI high", s["ci95"][1], added["H3_B_failed_minus_A_subtraction"]["ci95"][1])
    rq = an.boot_metric(per_task("B", lambda v, r: r.get("llm_calls") or 0),
                        per_task("A", lambda v, r: r.get("llm_calls") or 0))
    same("requests per run B - A", rq["estimate"], added["H5_requests_B_minus_A"]["estimate"])
    for arm in ("A", "B"):
        print(f"         {arm} failed calls per run (mean over tasks): "
              f"{np.mean(list(per_task(arm, lambda v, r: v['failed']).values())):.3f}; requests per run: "
              f"{np.mean(list(per_task(arm, lambda v, r: r.get('llm_calls') or 0).values())):.3f}")

    print("\npost hoc: Pass@1 B - A by whether the released harness broke a step in any of the task's A runs")
    hit = {(ds, q) for (a, ds, q, k), v in per_run.items() if a == "A" and v["harness"] > 0}
    esc = {(ds, q) for (a, ds, q, k), v in per_run.items() if v["python_with_escape_n"] > 0}
    esc_b = {(ds, q) for (a, ds, q, k), v in per_run.items() if a == "B" and v["python_with_escape_n"] > 0}
    none = set(prA) - hit
    print("  (both splits below depend on outcomes; neither shows what caused the change)")
    for name, group in (("broke a step", hit), ("broke none", none),
                        ("broke none, B's code has \\n", none & esc_b), ("broke none, B's code has no \\n", none - esc_b),
                        ("either arm's code has \\n", esc), ("neither arm's code has \\n", set(prA) - esc)):
        g = an.boot_diff({k: prB[k] for k in group}, {k: prA[k] for k in group}, ds_all)
        print(f"  {name}: {len(group)} tasks; DAB Pass@1 B - A {g['estimate']:+.4f} "
              f"(equal weight per task {g['equal_weight_mean_diff']:+.4f}); Wilcoxon p {g['wilcoxon_p']:.3f}")
    for arm in ("A", "B"):
        pr = [v for k, v in per_run.items() if k[0] == arm]
        print(f"  {arm}: Python calls per run {np.mean([v['python_calls'] for v in pr]):.2f}; "
              f"runs with a Python call {np.mean([v['python_calls'] > 0 for v in pr]):.3f}")
    print("\nall reported numbers reproduced" if ok else "\nSOME NUMBERS DIFFER")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["export"]:
        export()
    else:
        sys.exit(0 if check() else 1)
