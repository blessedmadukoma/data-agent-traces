"""REPL server for live/fa_replay.py. It runs cells as kramabench/kb_runner.py and
live/live_runner.py do, with one change: the model's code cannot replace the harness
function final_answer.

Before a cell runs, its code is rewritten: a definition, assignment, import alias or
parameter named final_answer becomes _model_final_answer; `del final_answer` deletes only
_model_final_answer; a read of final_answer that is not a call returns _model_final_answer
if the model made one, and the harness function otherwise. A call final_answer(x) therefore
always reaches the harness function, which records x and ends the cell.

Protocol: one JSON object per line on stdin, {"code": "..."}; one JSON object per line on
stdout, {"ok", "error_type", "error", "called", "final_answer", "stdout", "seconds"}.
"""
import ast
import builtins
import contextlib
import io
import json
import sys
import time
import traceback

_proto = sys.stdout
_globals = {"__name__": "__main__"}
NEW = "_model_final_answer"


class _FinalAnswer(BaseException):
    pass


def final_answer(answer):
    _globals["__final_answer__"] = answer
    raise _FinalAnswer()


class Protect(ast.NodeTransformer):
    def visit_Call(self, node):
        # keep the function position of final_answer(...) as it is; rename inside the arguments
        if isinstance(node.func, ast.Name) and node.func.id == "final_answer":
            node.args = [self.visit(a) for a in node.args]
            node.keywords = [self.visit(k) for k in node.keywords]
            return node
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "final_answer":
            if isinstance(node.ctx, ast.Load):
                # the model's own value if it made one, else the harness function
                return ast.copy_location(ast.Call(func=ast.Name(id="__fa_value__", ctx=ast.Load()),
                                                  args=[], keywords=[]), node)
            node.id = NEW
        return node

    def visit_Delete(self, node):
        # del final_answer removes only the model's own binding, never the harness function
        new = []
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "final_answer":
                new.append(ast.Name(id=NEW, ctx=ast.Del()))
                continue
            new.append(self.visit(t))
        keep = [t for t in new if not (isinstance(t, ast.Name) and t.id == NEW)]
        dropped = len(new) != len(keep)
        if dropped:
            guard = ast.parse(f"{NEW!r} in globals() and globals().pop({NEW!r})").body[0]
            if not keep:
                return ast.copy_location(guard, node)
            node.targets = keep
            return [node, ast.copy_location(guard, node)]
        node.targets = new
        return node

    def visit_FunctionDef(self, node):
        if node.name == "final_answer":
            node.name = NEW
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        if node.name == "final_answer":
            node.name = NEW
        return self.generic_visit(node)

    def visit_alias(self, node):
        if node.asname == "final_answer":
            node.asname = NEW
        return node

    def visit_arg(self, node):
        if node.arg == "final_answer":
            node.arg = NEW
        return node


def __fa_value__():
    return _globals.get(NEW, final_answer)


def transform(code):
    tree = ast.parse(code)
    tree = Protect().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, "<cell>", "exec")


def run(code):
    buf = io.StringIO()
    res = {"ok": True, "error": None, "error_type": None, "final_answer": None, "called": False}
    _globals.pop("__final_answer__", None)
    _globals["final_answer"] = final_answer
    builtins.final_answer = final_answer
    builtins.__fa_value__ = __fa_value__
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            try:
                obj = transform(code)
            except SyntaxError:
                obj = compile(code, "<cell>", "exec")  # raises the same SyntaxError as the original
            exec(obj, _globals)
        except _FinalAnswer:
            res["called"] = True
        except SystemExit:
            pass
        except BaseException as e:  # noqa: BLE001
            res["ok"] = False
            res["error_type"] = type(e).__name__
            res["error"] = traceback.format_exc()[-3000:]
    if "__final_answer__" in _globals:
        res["called"] = True
        try:
            json.dumps(_globals["__final_answer__"])
            res["final_answer"] = _globals["__final_answer__"]
        except (TypeError, ValueError):
            res["final_answer"] = repr(_globals["__final_answer__"])
    res["stdout"] = buf.getvalue()[-1500:]
    res["seconds"] = round(time.perf_counter() - t0, 3)
    return res


def main():
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        _proto.write(json.dumps(run(req.get("code", "")), default=str) + "\n")
        _proto.flush()


if __name__ == "__main__":
    main()
