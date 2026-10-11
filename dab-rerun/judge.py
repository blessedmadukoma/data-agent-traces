"""H4 of the pre-registration: DAB's own failure-analysis judge on failed runs of the rerun.

Same prompt files (failure_analysis/instruction.txt, failure_modes.txt), same trace format
(failure_analysis/get_trace.format_trace), same temperature (1.0) and same parser
(failure_analysis/parse_llm_judge.py) as DAB. Judge model: gpt-oss:120b on Ollama Cloud (decision D2),
because DAB's judge (gpt-5 on Azure) is not available to us.

Pre-registered scope: every failed run. DAB's own pipeline judges only failed runs that ended with
return_answer; the analysis reports both. Runs that ended without a tool call keep their final text
as an assistant message; format_trace needs a tool_calls key, so that message is rendered as text.
Judge calls go through the same cost ledger and cap as the agent runs.
"""
import json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path("/home/claude/dabx")
OUT = Path("/home/claude/dabx-runs")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DAB_LEDGER", str(OUT / "ledger.json"))
from openai import OpenAI
from common_scaffold import budget
from failure_analysis.get_prompt import get_prompt
from failure_analysis.get_trace import format_trace
from rerun import run_dir, is_infra

client = OpenAI(api_key=os.environ["OLLAMA_API_KEY"], base_url="https://ollama.com/v1")
RESULTS = OUT / "judge.jsonl"


def final_runs():
    final = {}
    for line in open(OUT / "runs.jsonl"):
        r = json.loads(line)
        final[(r["arm"], r["dataset"], r["query"], r["block"])] = r
    return final


def trace_text(messages):
    msgs = []
    for m in messages:
        if m["role"] == "assistant" and not m.get("tool_calls"):
            msgs.append({"role": "assistant", "tool_calls": []})
            msgs.append({"role": "user", "content": "[final text from the agent, no tool call]\n" + str(m.get("content"))})
        else:
            msgs.append(m)
    return format_trace(msgs)


def judge_one(key, r):
    arm, ds, q, k = key
    rd = run_dir(arm, ds, q, k, r["attempt"])
    fa = json.load(open(rd / "final_agent.json"))
    qdir = ROOT / f"query_{ds}" / f"query{q}"
    gt = (qdir / "ground_truth.csv").read_text(encoding="utf-8").strip()
    query = str(json.load(open(qdir / "query.json", encoding="utf-8")))
    prompt = get_prompt(trace_text(fa["messages"]), gt, query)
    out = {"arm": arm, "dataset": ds, "query": q, "block": k, "terminate_reason": r["terminate_reason"]}
    for attempt in range(3):
        try:
            budget.check()
            resp = client.chat.completions.create(model="gpt-oss:120b", temperature=1.0,
                                                  messages=[{"role": "user", "content": prompt}], timeout=600)
            budget.record(resp)
            out["judge"] = resp.choices[0].message.content
            break
        except budget.BudgetExhausted as e:
            out["error"] = str(e)
            break
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            time.sleep(20)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(out) + "\n")


def main():
    done = set()
    if RESULTS.exists():
        for line in open(RESULTS):
            j = json.loads(line)
            if j.get("judge"):
                done.add((j["arm"], j["dataset"], j["query"], j["block"]))
    todo = [(k, r) for k, r in final_runs().items()
            if not r["is_valid"] and not is_infra(r["terminate_reason"])
            and "budget cap" not in (r["terminate_reason"] or "")
            and r["terminate_reason"] != "wall_clock_cap" and k not in done]
    print(f"judging {len(todo)} failed runs")
    with ThreadPoolExecutor(max_workers=int(os.environ.get("JUDGE_WORKERS", "3"))) as pool:
        list(pool.map(lambda kr: judge_one(*kr), todo))


if __name__ == "__main__":
    main()
