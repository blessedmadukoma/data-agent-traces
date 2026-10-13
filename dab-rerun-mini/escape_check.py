"""Post hoc check for the gpt-5-mini DAB rerun (not pre-registered).

For each kept run of the complete blocks (0 to 4), counts the Python calls whose code contains an escape
sequence written as backslash-n (\\n) and those that contain a doubled one (\\\\n or \\\\t). It needs the
run folders that DAB's harness writes (not redistributed, because they contain DAB's data).

Usage: python3 dab-rerun-mini/escape_check.py [DAB_ROOT] [RUNS_JSONL]
       defaults: $DAB_ROOT or ~/dabx, and dab-rerun-mini/runs.jsonl
"""
import collections, json, os, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DAB_ROOT", str(Path.home() / "dabx")))
runs_path = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).with_name("runs.jsonl"))

runs = [json.loads(line) for line in open(runs_path)]
kept = [r for r in runs if r["block"] <= 4
        and not (r["terminate_reason"] or "").startswith(("llm_response_failed", "agent_run_failed"))]

stats = {"A": collections.Counter(), "B": collections.Counter()}
for r in kept:
    name = f"mini{r['arm']}_run_{r['block']}" + (f"_retry{r['attempt']}" if r["attempt"] else "")
    calls = root / f"query_{r['dataset']}" / f"query{r['query']}" / "logs" / "data_agent" / name / "tool_calls.jsonl"
    if not calls.exists():
        stats[r["arm"]]["run folder missing"] += 1
        continue
    for line in open(calls):
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            continue
        args = call.get("arguments") or call.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        code = args.get("code") if isinstance(args, dict) else None
        if not code:
            continue
        stats[r["arm"]]["python calls"] += 1
        if "\\\\n" in code or "\\\\t" in code:
            stats[r["arm"]]["doubled escape sequence"] += 1
        if "\\n" in code.replace("\\\\n", ""):
            stats[r["arm"]]["escape sequence \\n"] += 1

for arm in ("A", "B"):
    print(arm, dict(stats[arm]))
