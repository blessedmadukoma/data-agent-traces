"""REPL server that runs inside the sandbox (started by sandbox.Executor).

Protocol: one JSON object per line on stdin, {"code": "..."}; one JSON object
per line on the original stdout, {"ok", "stdout", "error", "error_type",
"final_answer", "seconds"}. Names persist between cells, as in a notebook
or a stateful agent harness. final_answer(x) records x and ends the cell.
"""
import contextlib
import io
import json
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
        except BaseException as e:  # noqa: BLE001 - every error goes back to the agent
            res["ok"] = False
            res["error_type"] = type(e).__name__
            tb = traceback.format_exc()
            res["error"] = tb[-LIMIT:]
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
        res = run(req.get("code", ""))
        _proto.write(json.dumps(res, default=str) + "\n")
        _proto.flush()


if __name__ == "__main__":
    main()
