#!/usr/bin/env python3
"""E3: third live experiment on QRData, where data-reference errors are common (pre-registration:
live/e3_preregistration.md).

E3 is E2 (live/e2_fork.py: arms N, R and E with shared prefixes, one sandbox per branch, final_answer protected)
with two changes:
  * the system prompt is the plain QRData prompt of the first live experiment (E1): it gives the question's data
    paths but not the sentence that tells the model to use exactly those paths, so models invent paths and miss
    columns far more often;
  * more than one model: --model picks it, and the prices below cover the models of the pilot.

Usage: OLLAMA_API_KEY=... python3 live/e3_fork.py --model gpt-oss:120b --tables data/QRData/benchmark/data \\
           --questions data/QRData/benchmark/QRData.json --out runs/e3/gpt-oss-120b.jsonl \\
           --transcripts runs/e3/tr/gpt-oss-120b --seeds 1,...,10 --workers 6 --budget-usd 15
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2_run as base  # noqa: E402
import e2_fork  # noqa: E402

PLAIN = base.kb_run.SYSTEM          # the QRData system prompt of E1, without E2's sentence on the data paths
assert base.SYSTEM.startswith(PLAIN) and base.SYSTEM != PLAIN
base.SYSTEM = PLAIN
# USD per 1M tokens (input, cached input, output), Ollama Cloud pricing page, 12 October 2026.
base.PRICES.update({
    "gpt-oss:120b": (0.15, 0.014, 0.60),
    "gpt-oss:20b": (0.07, 0.035, 0.30),
    "gemma4:31b": (0.14, 0.05, 0.40),
    "glm-5.3-flash": (0.15, 0.03, 0.50),
    "nemotron-3-nano:30b": (0.06, 0.06, 0.24),
})

_load = base.qr_run.load_tasks


def _load_subset(path):
    """E3_TASK_IDS (comma-separated ids such as qr-17) restricts the questions; used only for the pilot."""
    tasks = _load(path)
    ids = os.environ.get("E3_TASK_IDS")
    return [t for t in tasks if t["id"] in set(ids.split(","))] if ids else tasks


base.qr_run.load_tasks = _load_subset

if __name__ == "__main__":
    e2_fork.main()
