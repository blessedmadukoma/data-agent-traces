"""REPL server inside the live-gate sandbox (started by live/e1_run.LiveExecutor).

As kramabench/kb_runner.py, plus one request: {"exists": [paths]} returns
os.path.exists for each path, resolved in the session's current working
directory. The gate uses it as its exists callback, so the check sees the
same file system as the cell will (plan 2026-09-24j, E1).
"""
import contextlib
import io
import json
import os
import sys
import time
import traceback

LIMIT = 20000
_proto = sys.stdout
_globals = {"__name__": "__main__"}


class _FinalAnswer(BaseException):
    pass


def final_answer(answer):
    _globals["__final_answer__"] = answer
    raise _FinalAnswer()


_globals["final_answer"] = final_answer


def listing():
    out = []
    for root in ("/work", "/tmp"):
        for d, dirs, files in os.walk(root):
            if d == "/work":
                dirs[:] = [x for x in dirs if x != "input"]
            dirs[:] = [x for x in dirs if not x.startswith(".")]
            for f in files:
                if not f.startswith("."):
                    out.append(os.path.join(d, f))
            if len(out) > 5000:
                return sorted(out), True
    return sorted(out), False


def run(code):
    buf = io.StringIO()
    res = {"ok": True, "error": None, "error_type": None, "final_answer": None}
    _globals.pop("__final_answer__", None)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(code, "<cell>", "exec"), _globals)
        except _FinalAnswer:
            pass
        except SystemExit:
            pass
        except BaseException as e:  # noqa: BLE001
            res["ok"] = False
            res["error_type"] = type(e).__name__
            res["error"] = traceback.format_exc()[-LIMIT:]
    if "__final_answer__" in _globals:
        try:
            json.dumps(_globals["__final_answer__"])
            res["final_answer"] = _globals["__final_answer__"]
        except (TypeError, ValueError):
            res["final_answer"] = repr(_globals["__final_answer__"])
    out = buf.getvalue()
    res["stdout"] = out if len(out) <= LIMIT else out[:LIMIT // 2] + "\n...[output cut]...\n" + out[-LIMIT // 2:]
    res["seconds"] = round(time.perf_counter() - t0, 3)
    return res


def main():
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if req.get("ls"):
            files, cut = listing()
            res = {"files": files, "cut": cut}
        elif "exists" in req:
            res = {"exists": [os.path.exists(p) for p in req["exists"]]}
        else:
            res = run(req.get("code", ""))
        _proto.write(json.dumps(res, default=str) + "\n")
        _proto.flush()


if __name__ == "__main__":
    main()
