"""Count DataAgentBench Python calls that fail only because of the harness code wrapper.

DAB's execute_python tool puts the agent's code inside a triple-quoted string literal and then
runs `exec(code, env)`. Python decodes escape sequences in that string first. A call counts as
harness-caused when it failed with a syntax or indentation error, the agent's code
compiles on its own, and the decoded code does not. The script only parses and
compiles code; it never runs agent code. Use Python 3.12, as in DAB's executor image.

Run it inside the folder that holds the results-* directories:
  uv run python dab_harness_count.py
  uv run python dab_harness_count.py --paper-matched --bookreview-rerun ../dab-bookreview

--paper-matched keeps the 5 models and 12 datasets of the DAB paper, takes the three
bookreview queries from the re-run in main-branch commit 1b2d3b95e, and leaves out
pancancer queries 2 and 3, which have no logs that match the published answers.
"""
import argparse
import ast
import collections
import glob
import json
import os
import sys
from multiprocessing import Pool

EXTRA_DATASETS = {"query_civic_unstructured", "query_paper_unstructured"}
MODEL_NOT_IN_PAPER = "results-gpt5.1"
PANCANCER_LEFT_OUT = {("query_PANCANCER_ATLAS", "query2"), ("query_PANCANCER_ATLAS", "query3")}


def as_dict(value):
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return json.loads(value)
        except Exception:
            return None


def compiles(source):
    try:
        compile(source, "<code>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def decode(code):
    """Return the string that the harness passes to exec(), or None if it does not parse."""
    wrapped = f'code = """{code}"""\n\nenv_args = {{}}\n\nexec(code, env_args)\n'
    try:
        tree = ast.parse(wrapped)
    except (SyntaxError, ValueError):
        return None
    first = tree.body[0] if len(tree.body) == 3 else None
    if (isinstance(first, ast.Assign) and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name) and first.targets[0].id == "code"
            and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)):
        return first.value.value
    return None


def string_constants(source):
    try:
        tree = ast.parse(source)
    except Exception:
        return None
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def count_file(path):
    counts = collections.Counter(runs=1)
    for line in open(path):
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            counts["unreadable_lines"] += 1
            continue
        result = as_dict(call.get("result"))
        ok = result is not None and result.get("success") in (True, "True", "true")
        counts["calls"] += 1
        counts["failed"] += not ok
        if call.get("tool_name") != "execute_python":
            continue
        counts["python_calls"] += 1
        code = (as_dict(call.get("args")) or {}).get("code")
        if not isinstance(code, str):
            continue
        agent_ok = compiles(code)
        decoded = decode(code)
        decoded_ok = decoded is not None and compiles(decoded)
        if ok:
            if agent_ok and decoded_ok and string_constants(code) != string_constants(decoded):
                counts["ran_with_changed_string"] += 1
            continue
        error = str((result or {}).get("preview", "")) + str((result or {}).get("error", ""))
        if any(name in error for name in ("SyntaxError", "IndentationError", "TabError")):
            counts["python_syntax_failures"] += 1
            if agent_ok and not decoded_ok:
                counts["harness_caused"] += 1
    return counts


def select_files(paper_matched, bookreview_rerun):
    files = []
    for path in sorted(glob.glob("results-*/**/tool_calls.jsonl", recursive=True)):
        model, dataset, query = path.split(os.sep)[:3]
        if paper_matched:
            if model == MODEL_NOT_IN_PAPER or dataset in EXTRA_DATASETS:
                continue
            if (dataset, query) in PANCANCER_LEFT_OUT or dataset == "query_bookreview":
                continue
        files.append(path)
    if paper_matched:
        pattern = os.path.join(bookreview_rerun, "results-*/query_bookreview/**/tool_calls.jsonl")
        rerun = sorted(glob.glob(pattern, recursive=True))
        rerun = [p for p in rerun if f"{os.sep}{MODEL_NOT_IN_PAPER}{os.sep}" not in p]
        if not rerun:
            sys.exit(f"No bookreview re-run logs found under {bookreview_rerun}")
        files += rerun
    return files


def model_of(path):
    return next(part for part in path.split(os.sep) if part.startswith("results-"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-matched", action="store_true")
    ap.add_argument("--bookreview-rerun", help="folder with the bookreview logs of commit 1b2d3b95e")
    a = ap.parse_args()
    if sys.version_info[:2] != (3, 12):
        print(f"Warning: Python {sys.version.split()[0]}; the published counts use Python 3.12.")
    if a.paper_matched and not a.bookreview_rerun:
        sys.exit("--paper-matched needs --bookreview-rerun")

    files = select_files(a.paper_matched, a.bookreview_rerun)
    by_model = collections.defaultdict(collections.Counter)
    with Pool() as pool:
        for path, counts in zip(files, pool.map(count_file, files, chunksize=50)):
            by_model[model_of(path)].update(counts)
    total = collections.Counter()
    for counts in by_model.values():
        total.update(counts)

    label = "paper-matched corpus" if a.paper_matched else "all released logs"
    print(f"DAB harness count, {label}")
    for model in sorted(by_model):
        c = by_model[model]
        print(f"  {model:28} runs {c['runs']:6}  python calls {c['python_calls']:6}  harness-caused {c['harness_caused']:6}")
    share = total["harness_caused"] / max(total["python_calls"], 1)
    share_failed = total["harness_caused"] / max(total["failed"], 1)
    print(f"runs {total['runs']}; tool calls {total['calls']}; failed {total['failed']}")
    print(f"harness-caused syntax errors: {total['harness_caused']} of {total['python_calls']} Python calls "
          f"({share:.1%}); {share_failed:.1%} of failed calls")
    print(f"successful calls whose string literals changed: {total['ran_with_changed_string']}")


if __name__ == "__main__":
    main()
