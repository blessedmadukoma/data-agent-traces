"""Count the DABstep failures that the harness caused by dropping variables between steps.

A harness that runs every step in a fresh process drops the names that earlier steps
defined (preprint, Section 3.1, effect 2). We count a failed step as harness-caused when
  - its submission runs every step in a fresh process (gate_adapter.stateful_submissions),
  - it fails with NameError, and
  - the undefined name in the error message was defined by an earlier step of the same
    trace that ran without error.
One stricter and two looser variants are printed for comparison:
  - "certain": as above, and the earlier working step binds the name at its top level
    (assignment, import, def or class in the module body), so the name certainly existed
    when that step ended;
  - "named": the undefined name was defined by any earlier step of the trace;
  - "reads": the step reads any name defined by an earlier step (the rule of
    tools/fresh_process.py).
The script then gives the share of data-reference errors over the failed steps that remain.
The corrected count (4,005; 95% CI 3,658 to 4,340) comes from the hand audit
(audit/results_FINAL.json); removing NameErrors does not change it, because the audit
found no data-reference error among the sampled failures outside the two other strata.

Usage: uv run python tools/harness_nameerror.py --manifest runs/dabstep/manifest.jsonl \
           --cells runs/dabstep/parsed_*_cells.jsonl
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dabstep-gate"))
import gate_adapter  # noqa: E402

UNDEFINED = [re.compile(p) for p in (
    r"NameError: name '([^']+)' is not defined",
    r"The variable `([^`]+)` is not defined",
    r"name '([^']+)' is not defined",
)]
CORRECTED = (4005.19, 3657.74, 4339.54)   # audit: corrected count and 95% bootstrap interval
RULE_COUNT = 4429


def undefined_name(text):
    for p in UNDEFINED:
        m = p.search(text or "")
        if m:
            return m.group(1)
    return None


def as_bool(x):
    return x is True or str(x).lower() == "true"


def top_level_names(code):
    """Names bound directly in the module body of one step."""
    import ast
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                out |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    out |= {n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    a = ap.parse_args()
    manifest = {m["submission_file"]: m for m in map(json.loads, open(a.manifest))}
    traces = collections.defaultdict(list)
    failed_all = 0
    types = collections.Counter()
    for path in a.cells:
        for line in open(path):
            cell = json.loads(line)
            if as_bool(cell.get("failed")):
                failed_all += 1
                types[cell.get("error_type")] += 1
            m = manifest.get(cell["submission_file"])
            if m and m["included"]:
                traces[(cell["submission_file"], str(cell["task_id"]), cell.get("trace_line"))].append(cell)
    stateful = gate_adapter.stateful_submissions(traces)
    fresh = sorted(fn for fn, (ok, _, _) in stateful.items() if not ok)
    skip = set(dir(__builtins__)) | {"final_answer", "display"}
    count = collections.Counter()
    per_sub = collections.defaultdict(collections.Counter)
    no_name = 0
    for (fn, _, _), cells in traces.items():
        if fn not in fresh:
            continue
        before_any, before_ok, before_top = set(), set(), set()
        for c in sorted(cells, key=lambda r: int(r["cell_index"])):
            bound, read = gate_adapter._names(c.get("code") or "")
            failed = as_bool(c.get("failed"))
            if failed and c.get("error_type") == "NameError":
                name = undefined_name(c.get("output"))
                if name is None:
                    no_name += 1
                if name and name in before_ok:
                    count["strict"] += 1
                    per_sub[fn]["strict"] += 1
                if name and name in before_top:
                    count["certain"] += 1
                if name and name in before_any:
                    count["named"] += 1
                if read is not None and (read - (bound or set()) - skip) & before_any:
                    count["reads"] += 1
                per_sub[fn]["nameerror"] += 1
            if bound:
                before_any |= bound
                if not failed:
                    before_ok |= bound
                    before_top |= top_level_names(c.get("code") or "")
    print(f"failed steps: {failed_all}; NameError: {types['NameError']}; "
          f"sandbox-forbidden calls: {types['executor:forbidden']}")
    print(f"fresh-process submissions: {len(fresh)}")
    for fn in fresh:
        c = per_sub[fn]
        print(f"  {fn[:68]:68} NameError {c['nameerror']:5}  harness-caused (strict) {c['strict']:5}")
    print(f"NameErrors in fresh-process submissions without a parsable name: {no_name}")
    print("harness-caused NameErrors: "
          f"certain {count['certain']}, strict {count['strict']}, named {count['named']}, reads {count['reads']}")
    for variant in ("certain", "strict", "named", "reads"):
        rest = failed_all - count[variant]
        lo, hi = CORRECTED[1] / rest, CORRECTED[2] / rest
        print(f"[{variant}] remaining failed steps {rest}: corrected share {CORRECTED[0] / rest:.1%} "
              f"(95% CI {lo:.1%} to {hi:.1%}); rule share {RULE_COUNT / rest:.1%}; "
              f"NameError left {types['NameError'] - count[variant]} "
              f"({(types['NameError'] - count[variant]) / rest:.1%}); "
              f"forbidden {types['executor:forbidden'] / rest:.1%}; "
              f"SyntaxError {types['SyntaxError'] / rest:.1%}")
    print(f"[all failed steps] corrected share {CORRECTED[0] / failed_all:.1%} "
          f"(95% CI {CORRECTED[1] / failed_all:.1%} to {CORRECTED[2] / failed_all:.1%})")


if __name__ == "__main__":
    main()
