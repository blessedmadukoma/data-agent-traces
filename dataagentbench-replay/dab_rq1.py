#!/usr/bin/env python3
"""RQ1 on DataAgentBench: how many failed tool calls fail on data that does not exist.

Usage: python3 dab_rq1.py $DAB_RUN/cells_*.jsonl [--json out.json]

Counts per model and pooled:
  * tool calls by tool, failed calls, runs, log alignment checks;
  * harness-caused failures: SyntaxErrors that the code wrapper causes
    (dab_lib.harness_syntax);
  * interface misuse: the agent treats a result variable name as a key, a
    file or a Python name (dab_lib.INTERFACE);
  * data-reference failures (dab_lib.DATA_REF): SQL column or table that
    does not exist, MongoDB collection that does not exist, Python KeyError
    on a data key, FileNotFoundError on a path that is not a variable name;
  * how often the database message already names candidates;
  * shares with the primary denominator (failed calls minus harness-caused
    failures) and the secondary denominator (all failed calls);
  * MongoDB queries without a limit, which return at most 5 documents.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dab_lib as dl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    M = collections.defaultdict(collections.Counter)
    for p in a.cells:
        for line in open(p):
            r = json.loads(line)
            for key in (r["model"], "ALL"):
                c = M[key]
                c["runs"] += 1
                c["bad_lines"] += r["bad_lines"]
                c["align_env_bad"] += not r["align"]["env_counts"]
                c["align_checked"] += r["align"]["checked"]
                c["align_mismatch"] += r["align"]["value_mismatch"]
                for x in r["calls"]:
                    c["calls"] += 1
                    c["calls:" + str(x["tool"])] += 1
                    if x["tool"] == "query_db" and x.get("db_type") == "mongo":
                        mg = x.get("mongo") or {}
                        c["mongo"] += 1
                        if not x["failed"] and not mg.get("limit_given"):
                            c["mongo_ok_nolimit"] += 1
                            c["mongo_ok_nolimit_5"] += x.get("n_records") == 5
                            c["mongo_ok_nolimit_cut"] += bool(x.get("preview_cut"))
                    if not x["failed"]:
                        continue
                    c["failed"] += 1
                    g = x["group"]
                    if x.get("harness_syntax"):
                        c["harness"] += 1
                        continue
                    c["g:" + g] += 1
                    if g in dl.INTERFACE:
                        c["interface"] += 1
                    if g in dl.DATA_REF:
                        c["data_ref"] += 1
                        if g.startswith("sql:"):
                            c["sql_ref"] += 1
                            c[f"sql_ref:{x.get('engine')}"] += 1
                            c["sql_ref_hint"] += bool(x.get("db_hint"))
                        elif g.startswith("mongo:"):
                            c["mongo_ref"] += 1
                        else:
                            c["py_ref"] += 1
    order = [m for m in sorted(M) if m != "ALL"] + ["ALL"]
    print(f"{'model':18s} {'runs':>5s} {'calls':>7s} {'failed':>7s} {'harness':>7s} {'iface':>6s} "
          f"{'sql':>5s} {'hint':>5s} {'mongo':>5s} {'py':>5s} {'data':>6s} {'primary':>8s} {'second.':>8s}")
    for m in order:
        c = M[m]
        prim = c["failed"] - c["harness"]
        print(f"{m:18s} {c['runs']:5d} {c['calls']:7d} {c['failed']:7d} {c['harness']:7d} {c['interface']:6d} "
              f"{c['sql_ref']:5d} {c['sql_ref_hint']:5d} {c['mongo_ref']:5d} {c['py_ref']:5d} {c['data_ref']:6d} "
              f"{c['data_ref'] / max(1, prim):8.1%} {c['data_ref'] / max(1, c['failed']):8.1%}")
    c = M["ALL"]
    print("\ntool calls: " + ", ".join(f"{k[6:]} {v}" for k, v in sorted(c.items()) if k.startswith("calls:")))
    print(f"execute_python calls failing on the code wrapper: {c['harness']} of {c['calls:execute_python']} "
          f"({c['harness'] / max(1, c['calls:execute_python']):.1%}); of all failed calls {c['harness'] / max(1, c['failed']):.1%}")
    print("data-reference failures by group: " + ", ".join(
        f"{g} {c['g:' + g]}" for g in sorted(dl.DATA_REF)))
    print("SQL data-reference failures by engine: " + ", ".join(
        f"{k[8:]} {v}" for k, v in sorted(c.items()) if k.startswith("sql_ref:")) +
        f"; database message names candidates: {c['sql_ref_hint']} ({c['sql_ref_hint'] / max(1, c['sql_ref']):.1%})")
    print("interface misuse: " + ", ".join(f"{g} {c['g:' + g]}" for g in sorted(dl.INTERFACE)))
    print(f"MongoDB: {c['mongo']} queries; successful without limit {c['mongo_ok_nolimit']}, of which exactly 5 "
          f"records {c['mongo_ok_nolimit_5']} and preview cut {c['mongo_ok_nolimit_cut']}")
    print(f"log checks: unreadable lines {c['bad_lines']}; runs with env/result count mismatch {c['align_env_bad']}; "
          f"variable values checked {c['align_checked']}, mismatched {c['align_mismatch']}")
    top = collections.Counter({k[2:]: v for k, v in c.items() if k.startswith("g:")})
    print("\nall failure groups (harness-caused excluded):")
    for g, n in top.most_common():
        print(f"  {g:28s} {n:6d}")
    if a.json:
        json.dump({m: dict(M[m]) for m in order}, open(a.json, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
