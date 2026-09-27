#!/usr/bin/env python3
"""Answer accuracy of an InfiAgent-DABench run (context for the gate test, not a gate measure).

Uses the rules of the benchmark's eval_closed_form.py: answers are extracted with
@name[value]; a sub-answer is correct if the strings are equal or the numbers
differ by less than 1e-6. No reformat step (the benchmark uses GPT-3.5 to
reformat free-form responses; the prompt here asks for the format directly).

Usage: python3 infiagent/ia_score.py runs/ia/results_gpt-oss.jsonl .../da-dev-labels.jsonl
"""
import json
import re
import sys


def extract(s):
    m = re.findall(r"@(\w+)\[(.*?)\]", s or "")
    return dict(m)


def equal(a, b):
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def main():
    res = [json.loads(line) for line in open(sys.argv[1])]
    labels = {f"ia-{r['id']}": {k: v for k, v in r["common_answers"]} for r in map(json.loads, open(sys.argv[2]))}
    q_ok = sub_ok = sub_n = answered = 0
    for r in res:
        lab = labels[r["task"]]
        got = extract(str(r.get("final_answer") or ""))
        answered += bool(got)
        c = [equal(got.get(k), v) for k, v in lab.items()]
        q_ok += all(c)
        sub_ok += sum(c)
        sub_n += len(c)
    n = len(res)
    print(f"questions {n}; with an @name[value] answer {answered}; all sub-answers correct {q_ok} ({q_ok / max(1, n):.1%}); "
          f"sub-answers correct {sub_ok} / {sub_n} ({sub_ok / max(1, sub_n):.1%})")


if __name__ == "__main__":
    main()
