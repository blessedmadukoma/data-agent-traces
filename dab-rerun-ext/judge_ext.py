"""H4a on the extension: the frozen judge of the first rerun (dab-rerun/judge.py), unchanged, on the failed runs of
blocks 5 to 9. Only its paths change: OUT is the extension's folder and `rerun` is the extension's runner. The
judge's calls go through the extension's ledger and cap. Writes OUT/judge.jsonl.

budget.BudgetExhausted: the frozen judge catches this name. The rebuilt budget.py raises RuntimeError with the same
message, so the wrapper binds the name to RuntimeError; nothing else changes.

Usage: OLLAMA_API_KEY=... /home/claude/dab-venv/bin/python dab-rerun-ext/judge_ext.py
"""
import importlib.util, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ext = load("rerun", HERE / "rerun.py")
sys.modules["rerun"] = ext
os.environ["DAB_LEDGER"] = str(ext.OUT / "ledger.json")
os.environ["DAB_HARD_CAP_USD"] = os.environ.get("DAB_HARD_CAP_USD", str(ext.HARD_CAP))
for k, v in ext.PRICES.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, str(ext.ROOT))
from common_scaffold import budget  # noqa: E402
if not hasattr(budget, "BudgetExhausted"):
    budget.BudgetExhausted = RuntimeError
judge = load("judge_frozen", REPO / "dab-rerun" / "judge.py")
judge.OUT = ext.OUT
judge.RESULTS = ext.OUT / "judge.jsonl"

if __name__ == "__main__":
    judge.main()
