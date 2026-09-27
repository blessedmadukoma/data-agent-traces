"""Count receipt-replay arms that end with a constant answer, such as final_answer("Not Applicable").

An arm counts if any of its cells calls final_answer with a literal instead of a computed value.
Only finished arms are counted, and a pair counts when both of its arms finished.

Usage: uv run rq3-pilot/constant_answers.py runs/receipts/results_gpt-oss.jsonl
"""
import collections
import json
import os
import sys

from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dabstep-gate"))
from recovery_episodes import gave_up  # noqa: E402


def constant(arm):
    return any(gave_up(step.get("code") or "") for step in arm.get("steps") or [])


def main():
    if len(sys.argv) != 2:
        sys.exit("Give one results file, for example runs/receipts/results_gpt-oss.jsonl")
    rows = [json.loads(line) for line in open(sys.argv[1])]
    finished = [r for r in rows if r.get("status") == "done"]
    by_arm = collections.Counter(r["arm"] for r in finished if constant(r))
    print(f"finished arms with a constant answer: receipt {by_arm['receipt']}, traceback {by_arm['traceback']}")

    arms = {(r["episode"], r["seed"], r["arm"]): r for r in finished}
    only_receipt = only_traceback = 0
    for episode, seed, arm in arms:
        if arm != "receipt" or (episode, seed, "traceback") not in arms:
            continue
        receipt = constant(arms[(episode, seed, "receipt")])
        traceback = constant(arms[(episode, seed, "traceback")])
        only_receipt += receipt and not traceback
        only_traceback += traceback and not receipt
    p = binomtest(only_receipt, only_receipt + only_traceback).pvalue
    print(f"pairs where only the receipt arm did so: {only_receipt}; only the traceback arm: {only_traceback}; "
          f"sign test p = {p:.3f}")


if __name__ == "__main__":
    main()
