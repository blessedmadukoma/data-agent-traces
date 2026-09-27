#!/usr/bin/env python3
"""Fallback cell extractor for trace formats without a dedicated parser.

A cell is a fenced code block (```py or ```python) plus the text after it,
up to the next fenced block. The cell failed if that text contains a
Python traceback, a smolagents "Code execution failed" message, an E2B
ExecutionError, or "exit=1".

Two layouts get their own rules:
  * multi-round Markdown traces, where results follow as
    "### 代码块 k 执行结果 (exit=e)": block k is matched to result k, and a
    block without a result is skipped (it never ran);
  * smolagents console logs (ByteDance), where the result follows the
    "Executing this code" panel; a bare quoted key there is a KeyError.

Check 10 traces per submission by hand before you use
its cells. Record the result in the manifest.

Usage:
  python3 extract_generic_trace_cells.py <submissions_dir> <out.jsonl> --manifest manifest.jsonl
Only files with "included": true and a trace_format other than xml or
smolagents are read, plus files listed with --also.
"""
from dabstep_corpus_recount import classify, ANSI, EXE
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FENCE = re.compile(r"```(?:py|python)[ \t]*\n(.*?)```", re.S)
FAIL = re.compile(
    r"Traceback \(most recent call last\)|Code execution failed|ExecutionError\(name=|exit=1\b")
LAST = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_.]*(?:Error|Exception)): ?([^\n]{0,300})$", re.M)


ROUND = re.compile(r"## Model round \d+")
RESULT = re.compile(r"### 代码块 (\d+) 执行结果 \(exit=(-?\d+)\)")
BORDER = re.compile(r"^\s*─{20,}\s*$", re.M)


def _error_fields(text):
    """Error type and key from a failure message or traceback."""
    k = re.search(r"(?:due to|because of the following error)\s*:?\s*", text)
    if k:
        return classify(text[k.end():k.end() + 3000])
    e2b = re.search(
        r"ExecutionError\(name='([A-Za-z]+)', value=\\?\"?'?([^'\"]{0,80})", text)
    if e2b:
        return classify(f"{e2b.group(1)}: '{e2b.group(2)}'")
    last = LAST.findall(text)
    if last:
        return classify(last[-1][0] + ": " + last[-1][1])
    bare = re.match(r"\s*'([^'\n]{1,80})'\s*$", text)
    if bare:  # smolagents console prints str(KeyError) as the bare quoted key
        return "KeyError", bare.group(1)
    return "other", None


def _cells_numbered_results(t):
    """Multi-round Markdown traces: code blocks in a round, then '### 代码块 k 执行结果 (exit=e)' sections."""
    out = []
    starts = [m.start() for m in ROUND.finditer(t)] or [0]
    for i, s in enumerate(starts):
        seg = t[s:starts[i + 1] if i + 1 < len(starts) else len(t)]
        res = list(RESULT.finditer(seg))
        first_res = res[0].start() if res else len(seg)
        codes = [m.group(1) for m in FENCE.finditer(seg[:first_res])]
        results = {}
        for j, m in enumerate(res):
            end = res[j + 1].start() if j + 1 < len(res) else len(seg)
            results[int(m.group(1))] = (int(m.group(2)), seg[m.end():end])
        for k, code in enumerate(codes, 1):
            if k not in results:
                continue  # no execution record: the block was not run
            exit_code, text = results[k]
            failed = exit_code != 0 or "Traceback (most recent call last)" in text
            et, key = _error_fields(text) if failed else (None, None)
            out.append((code, text[:5000], failed, et, key))
    return out


def _cells_console(t):
    """smolagents console logs (ByteDance): result is printed after the 'Executing this code' panel."""
    out = []
    for m in FENCE.finditer(t):
        after = t[m.end():m.end() + 20000]
        p = after.find("Executing this code")
        if p < 0 or p > 400:
            continue  # not an executed block
        b = BORDER.search(after, p)
        if not b:
            continue
        stop = after.find("[Step", b.end())
        result = after[b.end():stop if stop >= 0 else len(after)].strip()
        ok = result.startswith("Execution logs:") or result.startswith(
            "Out") or result == ""
        failed = (
            not ok) or "Code execution failed" in result or "Traceback (most recent call last)" in result
        et, key = _error_fields(result) if failed else (None, None)
        code = "\n".join(l.rstrip() for l in m.group(1).split("\n"))  # the console pads lines with spaces
        out.append((code, result[:5000], failed, et, key))
    return out


def _unescape(t):
    """Undo one level of escaping in one pass, so an escaped backslash stays a backslash."""
    table = {"n": "\n", "t": "\t", '"': '"', "'": "'", "\\": "\\"}
    return re.sub(r"\\(.)", lambda m: table.get(m.group(1), m.group(0)), t, flags=re.S)


def trace_text(t):
    """Plain text of a trace, with escape sequences inside code kept as written.

    norm() replaces every two-character sequence backslash-n with a newline.
    That also changes code such as print("\\n"), which then fails to parse.
    Here: a trace stored as a JSON document is decoded with json.loads; a
    trace that already has real line breaks only loses its ANSI codes; any
    other trace is unescaped once."""
    s = t.strip()
    if s[:1] in "{[":
        try:
            obj = json.loads(s)
        except ValueError:
            obj = None
        if isinstance(obj, (dict, list)):
            parts = []

            def walk(o):
                if isinstance(o, str):
                    parts.append(o)
                elif isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(obj)
            return ANSI.sub("", "\n".join(parts))
    t = ANSI.sub("", t)
    if t.count("\n") >= t.count("\\n"):
        return t
    return _unescape(t)


def cells(t):
    t = trace_text(t)
    if RESULT.search(t):
        return _cells_numbered_results(t)
    if "Executing this code" in t:
        return _cells_console(t)
    ms = list(FENCE.finditer(t))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + \
            1 < len(ms) else min(len(t), m.end() + 20000)
        after = t[m.end():end]
        failed = bool(FAIL.search(after))
        et, key = _error_fields(after) if failed else (None, None)
        out.append((m.group(1), after[:5000], failed, et, key))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions_dir")
    ap.add_argument("out")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--also", nargs="*", default=[])
    a = ap.parse_args()
    man = [json.loads(l) for l in open(a.manifest)]
    files = [m["submission_file"] for m in man
             if m.get("included") and m.get("trace_format") not in ("xml", "smolagents")] + a.also
    total = 0
    with open(a.out, "w") as w:
        for fn in files:
            n = f = 0
            for line_no, line in enumerate(open(os.path.join(a.submissions_dir, fn)), 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                t = r.get("reasoning_trace") or ""
                t = t if isinstance(t, str) else json.dumps(t)
                for i, (code, out, failed, et, key) in enumerate(cells(t)):
                    w.write(json.dumps(dict(submission_file=fn, task_id=str(r.get("task_id")), trace_line=line_no,
                                            cell_index=i, code=code, output=out, failed=failed, error_type=et,
                                            error_key=key, executor_rule=et if et in EXE else None)) + "\n")
                    n += 1
                    f += failed
            total += n
            print(f"{n:7d} cells, {f:6d} failed  {fn}")
    print(f"total cells: {total}")


if __name__ == "__main__":
    main()
