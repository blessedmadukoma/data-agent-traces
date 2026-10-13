#!/usr/bin/env python3
"""Descriptive numbers of the live experiments E2 and E3 that the paper cites and that live/e2_report.py does not print.

For each results file and arm: runs; runs with at least one executed data-reference error (a step that fails on a file,
or with a KeyError that the checker blocks or would block); these errors by kind (file, column or key); the share of
runs whose first data-reference attempt is followed by a step that does not repeat it; and, for file errors, the share
of paths under /mnt/data or another absolute path. Uses the per-run measures of live/e2_report.py.

Usage: python3 tools/e3_numbers.py --questions data/QRData/benchmark/QRData.json runs/e2/results.jsonl runs/e3/*.jsonl
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live"))
import e2_report as r2  # noqa: E402

PATH = re.compile(r"No such file or directory: '([^']*)'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("results", nargs="+")
    a = ap.parse_args()
    qsrc = json.load(open(a.questions))
    gold = {f"qr-{i}": (q["answer"], (q.get("meta_data") or {}).get("question_type")) for i, q in enumerate(qsrc)}
    out = {}
    for path in a.results:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        model = rows[0]["model"] if rows else path
        res = {}
        for arm in ("N", "R", "E"):
            xs = [r for r in rows if r["arm"] == arm]
            if not xs:
                continue
            ms = [r2.run_measures(r, gold) for r in xs]
            dre_runs = sum(1 for m in ms if m["kinds"]["file"] + m["kinds"]["key_checker_block"] > 0)
            kinds = collections.Counter()
            for m in ms:
                kinds.update(m["kinds"])
            paths = collections.Counter()
            for r in xs:
                for c in r["cells"]:
                    if not c.get("blocked") and c.get("ok") is False and c.get("error_type") in r2.FILE_ERRORS:
                        p = PATH.search(c.get("error") or c.get("stderr") or c.get("traceback") or "")
                        p = p.group(1) if p else ""
                        paths["/mnt/data" if p.startswith("/mnt/data") else "other absolute" if p.startswith("/")
                              else "relative" if p else "unparsed"] += 1
            after = [m["after_ok"] for m in ms if m["after_ok"] is not None]
            res[arm] = {"runs": len(xs), "runs_with_executed_dre": dre_runs,
                        "share_runs_with_executed_dre": dre_runs / len(xs),
                        "executed_dre_steps": kinds["file"] + kinds["key_checker_block"],
                        "kinds": dict(kinds), "file_error_paths": dict(paths),
                        "runs_with_dra": sum(m["dra"] for m in ms),
                        "next_step_ok_after_first_dra": [sum(after), len(after)],
                        "blocks": sum(m["blocks"] for m in ms),
                        "accuracy": sum(m["correct"] for m in ms) / len(ms),
                        "answered": sum(m["answered"] for m in ms),
                        "executor_lost": sum(m["executor_lost"] for m in ms),
                        "lost_answers": sum(m["lost_answer"] for m in ms),
                        "runs_defining_final_answer": sum(m["defines_final_answer"] for m in ms),
                        "cost_usd": sum(m["cost"] for m in ms)}
        # Distinct steps that ran in arms N and E (shadow decisions): N's steps plus E's steps that did not share
        # N's model call (shared_with), because shared steps are the same step counted twice.
        byjob = collections.defaultdict(dict)
        for r in rows:
            byjob[(r["task"], r["seed"])][r["arm"]] = r
        n_ran = e_unshared = 0
        for arms in byjob.values():
            if "N" not in arms or "E" not in arms:
                continue
            n_ran += sum(1 for c in arms["N"]["cells"] if not c.get("no_code") and c.get("ok") is True)
            sw = arms["E"].get("shared_with") or []
            for i, c in enumerate(arms["E"]["cells"]):
                if not c.get("no_code") and c.get("ok") is True and "N" not in (sw[i] if i < len(sw) else []):
                    e_unshared += 1
        res["distinct_shadow_steps_ran"] = n_ran + e_unshared
        jobs = {(r["task"], r["seed"]): r.get("job_cost_usd") or 0 for r in rows}
        res["jobs"] = len(jobs)
        res["job_cost_usd"] = sum(jobs.values())
        out[f"{model} ({path})"] = res
        print(f"== {model}: {path}")
        print(json.dumps(res, indent=1))
    return out


if __name__ == "__main__":
    main()
