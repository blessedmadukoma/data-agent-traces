"""Trace-only fidelity checks: flag recorded steps that the recorded code cannot explain.

Each check reads only the trace (the code of each step, whether it failed, and its error or result). It needs no
access to the harness, so anyone who reads a published trace can run it.

  T1 impossible syntax error: the step failed with SyntaxError, IndentationError or TabError, but the recorded code
     compiles. The code that ran must differ from the code in the trace.
  T2 lost variable: the step failed with NameError on a name that an earlier, working step of the same run
     assigned at top level. Variables did not persist, although the code was written as if they would.
  T3 possibly cut result: a MongoDB query that the model wrote without "limit" returned exactly the default number of
     documents (DAB: 5), with no marker. The result may have been cut.

Input: DAB-style run folders (tool_calls.jsonl). Usage:
  python tracecheck.py RUN_FOLDER [RUN_FOLDER ...]   -> JSON lines, one per flagged step, and a summary on stderr
"""
import ast, json, re, sys
from pathlib import Path

SYNTAX = ("SyntaxError", "IndentationError", "TabError")
NAME_RE = re.compile(r"NameError: name '([^']+)' is not defined")
MONGO_DEFAULT_LIMIT = 5


def compiles(src):
    try:
        compile(src, "<code>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def assigned_names(src):
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return set()
    out = set()
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in node.names)
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    out.add(n.id)
    return out


def error_text(res):
    return str(res.get("error", "")) + "\n" + str(res.get("preview", ""))


def check_python(i, c, res, ok, defined, folder, flags):
    args = c.get("val_args") if isinstance(c.get("val_args"), dict) else {}
    code = args.get("code")
    if not isinstance(code, str):
        return
    err = "" if ok else error_text(res)
    if not ok and any(s in err for s in SYNTAX) and compiles(code):
        flags.append({"run": str(folder), "step": i, "check": "T1"})
    m = None if ok else NAME_RE.search(err)
    if m and m.group(1) in defined:
        flags.append({"run": str(folder), "step": i, "check": "T2", "name": m.group(1)})
    if ok:
        defined |= assigned_names(code)


def check_mongo(i, c, res, folder, flags):
    raw = c.get("args") if isinstance(c.get("args"), dict) else {}
    q = raw.get("query")
    try:
        qj = json.loads(q) if isinstance(q, str) else (q if isinstance(q, dict) else None)
    except Exception:
        qj = None
    if not (isinstance(qj, dict) and "collection" in qj and "limit" not in qj):
        return
    try:
        n_docs = len(json.loads(res.get("preview") or ""))
    except Exception:
        n_docs = None
    if n_docs == MONGO_DEFAULT_LIMIT:
        flags.append({"run": str(folder), "step": i, "check": "T3", "query": qj})


def check_run(folder):
    p = Path(folder) / "tool_calls.jsonl"
    flags, defined = [], set()
    if not p.exists():
        return flags
    for i, line in enumerate(open(p)):
        try:
            c = json.loads(line)
        except Exception:
            continue
        res = c.get("result") or {}
        ok = res.get("success") is True
        if c.get("tool_name") == "execute_python":
            check_python(i, c, res, ok, defined, folder, flags)
        elif c.get("tool_name") == "query_db" and ok:
            check_mongo(i, c, res, folder, flags)
    return flags


if __name__ == "__main__":
    total = {}
    for f in sys.argv[1:]:
        for fl in check_run(f):
            total[fl["check"]] = total.get(fl["check"], 0) + 1
            print(json.dumps(fl))
    print(json.dumps({"runs": len(sys.argv) - 1, "flags": total}), file=sys.stderr)
