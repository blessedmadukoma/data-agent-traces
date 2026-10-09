"""Count the runs that reach the 15-reply limit without an answer (preprint, Section 2.1).

Reads the recorded runs as a harness with a protected final_answer would have recorded them
(the output folder of live/fa_rescore.py) and counts, per run file and per arm, the runs that
end without an answer after 15 replies. A reply without code is one of the 15.

Usage:
  uv run live/fa_rescore.py --rec recorded --replays results/fa-replay --out runs/fa_recorded
  uv run python tools/run_limit.py --rec runs/fa_recorded
"""
import argparse
import collections
import json
import os

RUNS = {"live_gpt": "e1/results.jsonl", "live_ds": "e1/results_deepseek41.jsonl",
        "ia_gpt": "ia/results_gpt-oss.jsonl", "ia_ds": "ia/results_deepseek41.jsonl",
        "qr_gpt": "qr/results_gpt-oss.jsonl", "db_gpt": "db/results_gpt-oss.jsonl"}
LIMIT = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="runs/fa_recorded")
    a = ap.parse_args()
    for tag, rel in RUNS.items():
        path = os.path.join(a.rec, rel)
        if not os.path.exists(path):
            continue
        runs = collections.Counter()
        capped = collections.Counter()
        for line in open(path):
            r = json.loads(line)
            arm = r.get("arm") or "-"
            runs[arm] += 1
            if r.get("final_answer") is None and len(r.get("cells") or []) >= LIMIT:
                capped[arm] += 1
        total, cap = sum(runs.values()), sum(capped.values())
        print(f"{tag:9} runs {total:5}  reached {LIMIT} replies without an answer {cap:4} ({cap / total:.1%})")
        if len(runs) > 1:
            for arm in sorted(runs):
                print(f"          {arm}: {capped[arm]} of {runs[arm]} ({capped[arm] / runs[arm]:.1%})")


if __name__ == "__main__":
    main()
