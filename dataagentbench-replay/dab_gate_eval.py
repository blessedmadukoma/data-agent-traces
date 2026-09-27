#!/usr/bin/env python3
"""RQ2 on DataAgentBench: run the gate on every execute_python call and report.

Usage:
  python3 dab_gate_eval.py --gate-dir dabstep-gate --cells $DAB_RUN/cells_gpt-5-mini.jsonl \
      --pred $DAB_RUN/gate_gpt-5-mini.jsonl [--lint]

For each Python call, the gate sees the code that the harness actually ran
(the decoded code, dab_lib.wrapper) and the variables that existed before
the call:
  * inline results are tracked objects: a list of records, or one record,
    with exact keys (the whole value is in the log);
  * stored results are file paths; their record keys are exact for query_db
    results (every record has every column);
  * locals()['var_x'] and globals()['var_x'] are read as the variable itself;
  * x = var_x keeps tracking both names (gate alias mode);
  * no Python name is carried from one call to the next (new process);
  * files: dab_lib.Workspace (stored results, files written earlier in the run).
Calls whose code the wrapper turns into a SyntaxError never ran; they are
reported apart and excluded from every rate.

Cause match: the gate blocks a failed call and names the key in its KeyError,
or the path in its FileNotFoundError.
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dab_lib as dl  # noqa: E402


def same_path(a, b):
    a, b = os.path.normpath(a), os.path.normpath(b)
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lint", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.gate_dir))
    from gate import GateState, decide  # noqa: E402
    lint_flags = None
    if a.lint:
        from evaluate_gate import lint_flags  # noqa: E402

    C = collections.Counter()
    kinds = collections.Counter()
    ms = []
    with open(a.pred, "w") as out:
        for line in open(a.cells):
            r = json.loads(line)
            by_producer = {v["producer"]: (n, v) for n, v in r["vars"].items()}
            stored_keys = {}
            ws = dl.Workspace()
            for c in r["calls"]:
                if c["tool"] == "execute_python":
                    C["python"] += 1
                    rec = {"trace": r["trace"], "i": c["i"], "failed": c["failed"], "group": c.get("group")}
                    if c["wrapper"] in ("outer_syntax", "outer_structure", "inner_syntax"):
                        C["not_run"] += 1
                        rec["outcome"] = "not_run"
                        out.write(json.dumps(rec) + "\n")
                        ws.after_call([], dl.write_calls(c["code"]), ran=False)
                        continue
                    code = c["decoded"] if c["wrapper"] == "changed" else c["code"].strip()
                    env = {}
                    for n in c["env"]:
                        v = r["vars"].get(n)
                        if v is not None:
                            env[n] = v["path"] if "path" in v else v.get("value")
                    code_g, nrew = dl.rewrite_env_access(code, set(env))
                    schema, st, info = dl.env_state(env, stored_keys, GateState)
                    t0 = time.perf_counter()
                    try:
                        res = decide(code_g, schema, st, exists=ws.exists, alias=True)
                        o = res["outcome"]
                    except Exception as e:  # a gate crash is a gate defect, not a block
                        res, o = {"reason": f"{type(e).__name__}: {e}"[:300]}, "gate_error"
                    ms.append((time.perf_counter() - t0) * 1000)
                    C["ran"] += 1
                    C["rewritten"] += nrew > 0
                    C["o:" + o] += 1
                    viol = res.get("violations") or []
                    missing = {v.get("missing") for v in viol}
                    g = c.get("group")
                    cause = False
                    if o == "block" and c["failed"]:
                        if g in ("py:data-key", "interface:var-key"):
                            cause = any(k in missing for k in (c.get("detail") or []))
                        elif g in ("py:data-file", "interface:var-file"):
                            p = c.get("detail")
                            cause = any(v.get("kind") == "file" and (p is None or same_path(v["source"], p))
                                        for v in viol)
                    if c["failed"]:
                        C["failed"] += 1
                        C["failed_block"] += o == "block"
                        C["cause"] += cause
                        if g in ("py:data-key", "py:data-file"):
                            C["data_fail"] += 1
                            C["data_cause"] += cause
                            if cause:
                                kinds[res.get("kind")] += 1
                                if lint_flags is not None and not lint_flags(code_g, {dl.ident(n) for n in env}):
                                    C["data_cause_not_lint"] += 1
                        if g in ("interface:var-key", "interface:var-file"):
                            C["iface_fail"] += 1
                            C["iface_cause"] += cause
                    else:
                        C["ok"] += 1
                        C["ok_block"] += o == "block"
                    rec.update(outcome=o, cause=cause, missing=sorted(x for x in missing if x),
                               kind=res.get("kind"), reason=str(res.get("reason"))[:300],
                               env_access_rewritten=nrew > 0)
                    if o == "block" and not c["failed"]:
                        rec["code"] = code_g
                        rec["receipt"] = {k: res.get(k) for k in ("location", "kind", "source", "missing",
                                                                  "available", "near_matches")}
                    out.write(json.dumps(rec, default=str) + "\n")
                    ws.after_call(res.get("written") or [], dl.write_calls(code), ran=not c["failed"])
                # the result of this call is stored after it returns
                if not c["failed"] and c["i"] in by_producer:
                    n, v = by_producer[c["i"]]
                    if "path" in v:
                        ws.stored.add(os.path.normpath(v["path"]))
                        if v.get("keys") is not None:
                            stored_keys[os.path.basename(v["path"])] = set(v["keys"])
    t = C
    print(f"{os.path.basename(a.cells)}: python calls {t['python']}; not run (wrapper SyntaxError) {t['not_run']}; "
          f"ran {t['ran']} (env access rewritten in {t['rewritten']})")
    print(f"  failed {t['failed']}: blocked {t['failed_block']}, cause-matched {t['cause']}")
    print(f"  data-reference failures (py:data-key + py:data-file) {t['data_fail']}: caught with a cause match "
          f"{t['data_cause']} ({t['data_cause'] / max(1, t['data_fail']):.1%}); by kind {dict(kinds)}"
          + (f"; not flagged by pyflakes {t['data_cause_not_lint']}" if a.lint else ""))
    print(f"  interface misuse (var-key, var-file) {t['iface_fail']}: caught {t['iface_cause']}")
    print(f"  calls without an error {t['ok']}: blocked {t['ok_block']} (every one goes to the census audit)")
    print("  outcomes over calls that ran: " + ", ".join(
        f"{k[2:]} {v / max(1, t['ran']):.1%}" for k, v in sorted(t.items()) if k.startswith("o:")))
    if ms:
        ms.sort()
        print(f"  gate latency per call: median {statistics.median(ms):.2f} ms, p95 {ms[int(0.95 * len(ms))]:.2f} ms")
    json.dump({"counts": dict(t), "kinds": dict(kinds),
               "latency_ms": ms[:: max(1, len(ms) // 2000)]}, open(a.pred + ".summary.json", "w"))


if __name__ == "__main__":
    main()
