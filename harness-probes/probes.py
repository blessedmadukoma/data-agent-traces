"""Harness probes: run the same short programs through several code-execution harnesses and record what each
returns, to find places where a harness changes what runs or what the model receives without a record.

A trace records the code that the model wrote. Each probe is a sequence of steps (programs) in one session.
The reference is plain Python with one namespace for the whole session. A harness result that differs from the
reference is reported with the harness's own output, so that a reader can judge whether the difference is
documented design (for example, no variables between steps) or an unrecorded change.

Usage: <venv>/bin/python probes.py ADAPTER   (prints one JSON object)
Adapters: reference, dab_released, smolagents_old, smolagents_new, langchain_repl, autogen_local
"""
import contextlib, io, json, subprocess, sys, tempfile, traceback

PROBES = {
    # A string literal with an escape sequence, as in print("\n".join(rows)).
    "escape_sequence": ['rows = ["a", "b"]\nprint("\\n".join(rows))'],
    # Triple double quotes inside the model's code.
    "triple_quotes": ['s = """x"""\nprint(s)'],
    # A variable defined in one step and used in the next.
    "variable_between_steps": ["x = 41", "print(x + 1)"],
    # A dictionary key that does not exist, next to a similar one.
    "missing_key": ['d = {"merchant_name": "A"}\nprint(d["merchant"])'],
    # A variable name that does not exist, next to a similar one (same step).
    "missing_name": ["total_amounts = 5\nprint(total_amount)"],
    # A long output: how much reaches the model, and with what marker.
    "long_output": ['print("x" * 100000)'],
}


def short(s, n=160):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"...[{len(s)} chars]"


def summarise(probe, step_results):
    """step_results: list of (ok, text). Returns a compact record."""
    out = {"steps": [{"ok": ok, "text": short(t)} for ok, t in step_results]}
    if probe == "long_output":
        ok, t = step_results[-1]
        out["returned_chars"] = len(t)
        out["x_chars"] = t.count("x")
        out["marker"] = short(t.replace("x", "")[:200], 200)
    return out


# ---------------- adapters ----------------
def run_reference():
    res = {}
    for name, steps in PROBES.items():
        ns, rs = {}, []
        for code in steps:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, "<step>", "exec"), ns)
                rs.append((True, buf.getvalue()))
            except Exception as e:
                rs.append((False, f"{type(e).__name__}: {e}"))
        res[name] = summarise(name, rs)
    return res


def run_dab_released():
    """DAB's execute_python as released (ExecTool.py at 6ad9d68): the code is pasted into a triple-quoted string
    literal and passed to exec(), in a fresh Python process in DAB's executor image. Only DAB's own variables
    (earlier tool results) are passed in env_args; here there are none."""
    res = {}
    for name, steps in PROBES.items():
        rs = []
        for code in steps:
            exec_str = f'''code = """{code}"""\n\nenv_args = {{}}\n\nexec(code, env_args)\n'''
            p = subprocess.run(["docker", "run", "--rm", "-i", "--network", "none", "python-data:3.12", "python", "-"],
                               input=exec_str, capture_output=True, text=True, timeout=120)
            rs.append((p.returncode == 0, p.stdout if p.returncode == 0 else (p.stdout + p.stderr).strip().splitlines()[-1]))
        res[name] = summarise(name, rs)
    return res


def run_smolagents_old():
    """smolagents 1.4.1: LocalPythonInterpreter, as CodeAgent creates it (one state for the whole run)."""
    from smolagents.local_python_executor import LocalPythonInterpreter
    res = {}
    for name, steps in PROBES.items():
        ex = LocalPythonInterpreter(additional_authorized_imports=[], tools={})
        rs = []
        for code in steps:
            try:
                output, logs, _ = ex(code, {})
                rs.append((True, logs))
            except Exception as e:
                rs.append((False, f"{type(e).__name__}: {e}"))
        res[name] = summarise(name, rs)
    return res


def run_smolagents_new():
    """Current smolagents: LocalPythonExecutor (one state for the whole run)."""
    from smolagents.local_python_executor import LocalPythonExecutor
    res = {}
    for name, steps in PROBES.items():
        ex = LocalPythonExecutor(additional_authorized_imports=[])
        try:
            ex.send_tools({})
        except Exception:
            pass
        rs = []
        for code in steps:
            try:
                out = ex(code)
                rs.append((True, out.logs))
            except Exception as e:
                rs.append((False, f"{type(e).__name__}: {e}"))
        res[name] = summarise(name, rs)
    return res


def run_langchain_repl():
    """LangChain's PythonREPL (langchain-experimental), the backend of its Python REPL tool."""
    from langchain_experimental.utilities import PythonREPL
    res = {}
    for name, steps in PROBES.items():
        repl, rs = PythonREPL(), []
        for code in steps:
            t = repl.run(code)
            rs.append((not (t.startswith(("NameError", "KeyError", "SyntaxError")) or "Error(" in t[:40]), t))
        res[name] = summarise(name, rs)
    return res


def run_autogen_local():
    """AutoGen's LocalCommandLineCodeExecutor (autogen-ext 0.7.5): each code block is written to a file and run."""
    import asyncio
    from autogen_core import CancellationToken
    from autogen_core.code_executor import CodeBlock
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    res = {}
    for name, steps in PROBES.items():
        with tempfile.TemporaryDirectory() as d:
            ex, rs = LocalCommandLineCodeExecutor(work_dir=d), []
            for code in steps:
                r = asyncio.run(ex.execute_code_blocks([CodeBlock(code=code, language="python")], CancellationToken()))
                rs.append((r.exit_code == 0, r.output if r.exit_code == 0 else r.output.strip().splitlines()[-1]))
        res[name] = summarise(name, rs)
    return res


if __name__ == "__main__":
    adapter = sys.argv[1]
    try:
        print(json.dumps({"adapter": adapter, "results": globals()["run_" + adapter]()}))
    except Exception:
        print(json.dumps({"adapter": adapter, "error": traceback.format_exc()[-800:]}))
