import json, importlib.metadata as m
import smolagents.local_python_executor as lpe
def run(code):
    if hasattr(lpe, "LocalPythonExecutor"):
        ex = lpe.LocalPythonExecutor(additional_authorized_imports=[])
        try: ex.send_tools({})
        except Exception: pass
        try:
            o = ex(code); return "ok " + (o.logs if hasattr(o, "logs") else o[1]).strip()
        except Exception as e: return "ERR " + type(e).__name__
    ex = lpe.LocalPythonInterpreter(additional_authorized_imports=[], tools={})
    try: return "ok " + ex(code, {})[1].strip()
    except Exception as e: return "ERR " + type(e).__name__
print(json.dumps({"version": m.version("smolagents"),
  "missing_key": run('d = {"merchant_name": "A"}\nprint(d["merchant"])'),
  "missing_name": run("total_amounts = 5\nprint(total_amount)")}))
