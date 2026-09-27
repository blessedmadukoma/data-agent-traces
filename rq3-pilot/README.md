# Receipt replay: receipt or traceback

Does the checker's receipt help an agent recover in fewer cells than the traceback? These scripts continue recorded DABstep traces with one model, twice. The folder name comes from research question 3 (RQ3) of the study.

1. `select_prefixes.py` cuts each trace at the first cell that the checker blocks with a cause match. It keeps the earlier cells, the receipt and the paths of the context files.
2. `rq3_run.py` rebuilds the state by running the earlier cells in a sandbox. It shows the model the task and the trace, and then gives one of two kinds of feedback for the blocked cell: `traceback` (the cell runs and fails) or `receipt` (the cell does not run). The model then writes up to 6 more cells. Both arms use the same seed. The checker is not used after the blocked cell, so the arms differ only in that one piece of feedback.
3. `rq3_report.py` (cost check) and `rq3_full_report.py` (full study) pair the arms and report the cells to the first working cell that does more than inspect the data, episodes that never recover, constant answers, replay fidelity, tokens and cost. `rq3_mixed.py` fits a random intercept per submitter.

`context_layout.py` puts the context files where each submission kept them, from the evidence in its own traces. `runner.py` is the Python process inside the sandbox.

## Sandbox

The executed code comes from other people's traces and from the model, so `sandbox.py` runs it with bubblewrap: no network, a private `/tmp`, read-only system and Python folders, and one writable work folder. The repository is not visible inside the sandbox. `--sandbox local` runs a plain subprocess and is only for the unit tests.

## Model access

Ollama Cloud, OpenAI-compatible endpoint `https://ollama.com/v1`. Put the key in `.env` at the repository root as `OLLAMA_API_KEY=...`, or export it. `.env` is in `.gitignore`. Prices per million tokens are in `rq3_run.PRICES` (Ollama pricing page, 23 September 2026).

## Commands

From the repository root, after `make dabstep`:

```bash
make receipts-select        # 10 episodes, no model calls
make receipts-cost-check    # about USD 0.07; repeat until it prints "0 episode-arms left"
make receipts-report
```

The commands of the full study are in [docs/REPRODUCE.md](../docs/REPRODUCE.md), section B4.
