#!/usr/bin/env python3
"""Extract code cells from smolagents-style DABstep traces.

One cell = one python_interpreter tool call in one agent step, with the
step's own observation and error. Earlier steps that are repeated inside
later model inputs are not counted again.

Handles three trace encodings seen at commit c8fb51b3:
  * Python repr of memory steps: ActionStep(..., tool_calls=[ToolCall(name='python_interpreter', arguments=...)], error=..., observations=...)
  * Python repr of step dicts:   {'tool_calls': [ToolCall(...)] or [{'name': 'python_interpreter', 'arguments': ...}], 'error': ..., 'observations': ...}
  * JSON list of step dicts:     {"tool_calls": [{"function": {"name": "python_interpreter", "arguments": ...}}], "error": ..., "observations": ...}

Usage:
  python3 extract_smolagents_trace_cells.py <submissions_dir> <out.jsonl> [--manifest manifest.jsonl]

Without --manifest, every file in the directory is tried; files without
smolagents tool calls produce no cells. With --manifest, only files with
"included": true and "trace_format": "smolagents" are read.
"""
from dabstep_corpus_recount import classify, norm, EXE  # shared error classifier
import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANCHORS = [
    re.compile(r"ToolCall\(name='python_interpreter', arguments="),
    re.compile(
        r"'tool_calls': \[\{'name': 'python_interpreter', 'arguments': "),
]
QUOTES = "'\""


def read_pystr(s, i):
    """Read one Python string literal that starts at s[i]."""
    j = i
    while j < len(s) and s[j] not in QUOTES:
        if s[j] not in "rbuRBU":
            return None
        j += 1
    try:
        tok = next(tokenize.generate_tokens(
            io.StringIO(s[i:i + 2_000_000]).readline))
        if tok.type == tokenize.STRING:
            return ast.literal_eval(tok.string)
    except (tokenize.TokenError, SyntaxError, ValueError, StopIteration):
        return None
    return None


def field_after(s, start, end, names):
    """Find the first field in names within s[start:end] and read its value."""
    best = None
    for n in names:
        k = s.find(n, start, end)
        if k >= 0 and (best is None or k < best[0]):
            best = (k, n)
    if best is None:
        return None, False
    v = best[0] + len(best[1])
    rest = s[v:v + 200]
    if rest.startswith("None"):
        return None, True
    m = re.search(r"'message': ", rest[:120])
    if m:
        return read_pystr(s, v + m.end()), True
    for off, ch in enumerate(rest[:120]):
        if ch in QUOTES:
            return read_pystr(s, v + off), True
    return None, True


def cells_from_repr(t):
    hits = sorted((m.end(), a.pattern) for a in ANCHORS for m in a.finditer(t))
    out = []
    for n, (pos, _) in enumerate(hits):
        nxt = hits[n + 1][0] if n + 1 < len(hits) else len(t)
        code = read_pystr(t, pos)
        if code is None:
            continue
        err, _ = field_after(t, pos, nxt, ["error=", "'error': "])
        obs, _ = field_after(
            t, pos, nxt, ["observations=", "'observations': "])
        out.append((code, obs or "", err))
    return out


def cells_from_json(t):
    try:
        steps = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(steps, list):
        return None
    out = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        for tc in st.get("tool_calls") or []:
            f = tc.get("function", tc) if isinstance(tc, dict) else {}
            if f.get("name") != "python_interpreter":
                continue
            code = f.get("arguments")
            if isinstance(code, dict):
                code = code.get("code") or json.dumps(code)
            err = st.get("error")
            if isinstance(err, dict):
                err = err.get("message") or json.dumps(err)
            out.append((code or "", st.get("observations") or "", err))
    return out


LAST = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_.]*(?:Error|Exception)): ?([^\n]{0,300})$", re.M)


def error_fields(err):
    if not err:
        return None, None, None
    err = norm(err)  # removes ANSI colour codes (Jupyter-style tracebacks)
    m = re.search(r"(?:due to|because of the following error)\s*:?\s*", err)
    if m:
        c, k = classify(err[m.end():])
    elif "Traceback (most recent call last)" in err and LAST.findall(err):
        last = LAST.findall(err)[-1]
        c, k = classify(last[0] + ": " + last[1])
    else:
        c, k = classify(err)
    return c, k, (c if c in EXE else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions_dir")
    ap.add_argument("out")
    ap.add_argument("--manifest")
    a = ap.parse_args()
    files = sorted(os.listdir(a.submissions_dir))
    if a.manifest:
        man = [json.loads(l) for l in open(a.manifest)]
        keep = {m["submission_file"] for m in man if m.get(
            "included") and m.get("trace_format") == "smolagents"}
        files = [f for f in files if f in keep]
    n_cells = n_fail = 0
    per_file = {}
    with open(a.out, "w") as w:
        for fn in files:
            p = os.path.join(a.submissions_dir, fn)
            if os.path.getsize(p) < 200:
                continue
            fc = 0
            for line_no, line in enumerate(open(p), 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                t = r.get("reasoning_trace") or ""
                if not isinstance(t, str):
                    t = json.dumps(t)
                cells = cells_from_json(t)
                if cells is None:
                    cells = cells_from_repr(t)
                for i, (code, obs, err) in enumerate(cells):
                    c, k, rule = error_fields(err)
                    rec = dict(submission_file=fn, task_id=str(r.get("task_id")), trace_line=line_no,
                               cell_index=i, code=code, output=obs if not err else err,
                               failed=bool(err), error_type=c, error_key=k, executor_rule=rule)
                    w.write(json.dumps(rec) + "\n")
                    fc += 1
                    n_fail += bool(err)
            if fc:
                per_file[fn] = fc
            n_cells += fc
    for fn, c in per_file.items():
        print(f"{c:7d}  {fn}")
    print(
        f"files with cells: {len(per_file)}; cells: {n_cells}; failed: {n_fail}")


if __name__ == "__main__":
    main()
