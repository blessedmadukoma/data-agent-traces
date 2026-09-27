"""REPL server inside the KramaBench sandbox (started by kb_run.KBExecutor).

As rq3-pilot/runner.py, plus one request: {"ls": true} returns the files under
the work directory (without input/, the read-only data lake) and under /tmp,
so that the harness can record the workspace before each cell.
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
        else:
            res = run(req.get("code", ""))
        _proto.write(json.dumps(res, default=str) + "\n")
        _proto.flush()


if __name__ == "__main__":
    main()
