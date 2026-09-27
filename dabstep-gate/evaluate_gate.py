#!/usr/bin/env python3
"""Run the gate on every parsed cell and report its effect.

Per submitter and pooled:
  * failed cells: block rate, and cause-matched block rate (the gate names
    the same missing column or key as the recorded error);
  * data-reference failures: share caught with a cause match (a recall proxy
    that needs no labels);
  * cells without an error: block rate (upper bound on false blocks);
  * run / unknown shares, and gate latency per cell (median and 95th percentile);
  * with --lint: how many cause-matched blocks a syntax and undefined-name
    check (pyflakes) does NOT already flag, i.e. incremental coverage.

The lint check knows the names assigned in earlier cells of the same trace,
so variables from earlier cells are not reported as undefined.

Writes one prediction per cell to --pred (JSONL), separate from any labels.

Usage:
  python3 evaluate_gate.py --gate-dir . --data "$DATA" --manifest "$RUN/manifest.jsonl" \
      --cells "$RUN"/parsed_*_cells.jsonl --pred "$RUN/gate_predictions.jsonl" [--lint] [--carry]

With --files, the gate also checks that every file it can resolve exists,
using gate_adapter.FileEvidence as a stand-in for the workspace listing.

With --carry, the gate keeps the DataFrames it tracks from one cell to the
next cell of the same trace. A trace is one line of one submission file.
Submissions whose harness runs every cell in a fresh process (detected from
NameErrors, see gate_adapter.stateful_submissions) get no carried state.
"""
import gate_adapter as ga
import argparse
import ast
import builtins
import collections
import json
import os
import statistics
import sys
import warnings

# invalid escapes in agent code
warnings.filterwarnings("ignore", category=SyntaxWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = {"KeyError", "KeyError(wrapped)", "index-error(wrapped)",
        "FileNotFoundError", "ValueError:usecols"}
TOOLS = {"final_answer", "read_csv_metadata_file",
         "read_json_metadata_file", "safe_read_file", "display"}


def assigned_names(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
    return out


def lint_flags(code, known):
    """True if pyflakes reports a syntax error or an undefined name in this cell."""
    from pyflakes import checker, messages
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return True
    w = checker.Checker(tree, filename="<cell>",
                        builtins=set(dir(builtins)) | known | TOOLS)
    return any(isinstance(m, (messages.UndefinedName, messages.UndefinedLocal)) for m in w.messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gate-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lint", action="store_true")
    ap.add_argument("--carry", action="store_true",
                    help="carry tracked DataFrames across the cells of a trace (gate.GateState)")
    ap.add_argument("--files", action="store_true",
                    help="check that read files exist, with the replay evidence model (gate_adapter.FileEvidence)")
    ap.add_argument("--tolerant", choices=["none", "auto", "all"], default="none",
                    help="executors that replace a missing dict key silently (gate_adapter.tolerant_submissions)")
    a = ap.parse_args()
    man = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    traces = collections.defaultdict(list)
    for p in a.cells:
        for line in open(p):
            r = json.loads(line)
            m = man.get(r["submission_file"])
            if m and m["included"]:
                traces[(r["submission_file"], str(r["task_id"]), r.get("trace_line"))].append(r)
    rp = ga.Replay(a.gate_dir, a.data, traces, man, carry=a.carry, files=a.files, tolerant=a.tolerant)
    if a.carry:
        off = sorted(fn for fn, (ok, _, _) in rp.stateful.items() if not ok)
        print(f"--carry: {len(off)} submissions run every cell in a fresh process; no state is carried for them:")
        for fn in off:
            k, n = rp.stateful[fn][1:]
            print(f"  {fn} ({k}/{n} cells that use an earlier name fail with NameError)")
    if rp.tolerant:
        print(f"--tolerant {a.tolerant}: {len(rp.tolerant)} submissions may replace a missing dict key silently "
              f"(smolagents 1.0-1.4); dict-key checks there block only names without a close match")
    F = collections.Counter()
    C = collections.defaultdict(collections.Counter)
    ms = []
    with open(a.pred, "w") as out:
        for (fn, tid, tl), cells in traces.items():
            s = man[fn]["submitter"]
            known = set()
            for r, res, o, dt in rp.run((fn, tid, tl), cells):
                ms.append(dt)
                F["substitutions"] += len(res.get("substitutions") or []) if not r["failed"] else 0
                missing = {v.get("missing") for v in (
                    res.get("violations") or [])} if isinstance(res, dict) else set()
                viol = (res.get("violations") or []) if isinstance(res, dict) else []
                fpath = ga.fnf_path(r.get("output")) if r.get("error_type") == "FileNotFoundError" else None
                file_cause = r.get("error_type") == "FileNotFoundError" and any(
                    v.get("kind") == "file" and (fpath is None or ga.same_path(v["source"], fpath)) for v in viol)
                cause = bool(o == "block" and ((ga.error_keys(r) & missing)
                                               or (r.get("error_type") == "ValueError:usecols" and "usecols" in str(res))
                                               or file_cause))
                if r["failed"] and r.get("error_type") == "FileNotFoundError":
                    F["fnf"] += 1
                    F["resolved"] += any(not cond and (fpath is None or ga.same_path(p, fpath))
                                         for p, cond, _ in (res.get("files") or []))
                    F["caught"] += cause
                lint = lint_flags(r["code"] or "", known) if a.lint else None
                known |= assigned_names(r["code"] or "")
                c = C[s]
                if r["failed"]:
                    c["failed"] += 1
                    c["failed_" + o] += 1
                    c["cause"] += cause
                    if r.get("error_type") in DATA:
                        c["data_fail"] += 1
                        c["data_cause"] += cause
                    if cause and a.lint and not lint:
                        c["cause_not_lint"] += 1
                    if a.lint and lint:
                        c["failed_lint"] += 1
                else:
                    c["ok"] += 1
                    c["ok_" + o] += 1
                out.write(json.dumps(dict(submission_file=fn, task_id=tid, trace_line=r.get("trace_line"), cell_index=r["cell_index"], failed=r["failed"],
                                          error_type=r.get("error_type"), error_key=r.get("error_key"), outcome=o,
                                          reason=(res.get("reason") if isinstance(
                                              res, dict) else str(res)),
                                          kind=res.get("kind") if isinstance(res, dict) else None,
                                          missing=sorted(x for x in missing if x), cause_match=cause,
                                          lint_flag=lint, ms=round(dt, 3))) + "\n")
    tot = collections.Counter()
    head = f"{'submitter':15s} {'failed':>7s} {'block':>6s} {'cause':>6s} {'data-fail':>9s} {'caught':>7s} {'ok':>7s} {'ok-block':>8s} {'unknown':>8s}"
    print(head)
    for s, c in sorted(C.items()):
        tot.update(c)
        unk = (c["failed_unknown"] + c["ok_unknown"]) / \
            max(1, c["failed"] + c["ok"])
        print(f"{s:15s} {c['failed']:7d} {c['failed_block']:6d} {c['cause']:6d} {c['data_fail']:9d} "
              f"{c['data_cause'] / max(1, c['data_fail']):7.1%} {c['ok']:7d} {c['ok_block'] / max(1, c['ok']):8.2%} {unk:8.1%}")
    t = tot
    print(f"\nfailed cells: {t['failed']}; blocked {t['failed_block']} ({t['failed_block']/max(1, t['failed']):.2%}); "
          f"cause-matched {t['cause']} ({t['cause']/max(1, t['failed']):.2%})")
    print(f"data-reference failures: {t['data_fail']}; caught with a cause match: {t['data_cause']} "
          f"({t['data_cause']/max(1, t['data_fail']):.1%})")
    print(f"cells without an error: {t['ok']}; blocked {t['ok_block']} ({t['ok_block']/max(1, t['ok']):.2%}) "
          f"-- upper bound on false blocks; audit a sample with sample_false_block_audit.py")
    allc = t["failed"] + t["ok"]
    print(f"outcomes over all cells: run {(t['failed_run']+t['ok_run'])/allc:.1%}, unknown {(t['failed_unknown']+t['ok_unknown'])/allc:.1%}, "
          f"block {(t['failed_block']+t['ok_block'])/allc:.1%}")
    if F["substitutions"]:
        print(f"dict-key reads that a tolerant executor replaced silently, in cells without an error: {F['substitutions']}")
    if F["fnf"]:
        print(f"missing-file failures: {F['fnf']}; path resolved before execution in code that always runs: "
              f"{F['resolved']} ({F['resolved']/F['fnf']:.1%}, what a live os.path.exists check can block); "
              f"blocked with the replay evidence model: {F['caught']} ({F['caught']/F['fnf']:.1%})")
    ms.sort()
    print(
        f"gate latency per cell: median {statistics.median(ms):.2f} ms, p95 {ms[int(.95*len(ms))]:.2f} ms, n={len(ms)}")
    if a.lint:
        print(f"lint flags {t['failed_lint']} of {t['failed']} failed cells; cause-matched gate blocks not flagged by lint "
              f"(incremental coverage): {t['cause_not_lint']} of {t['cause']}")
    print(f"predictions written to {a.pred}")


if __name__ == "__main__":
    main()
