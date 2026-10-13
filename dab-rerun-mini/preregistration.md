# Pre-registration: controlled DAB rerun with gpt-5-mini

**Status:** written on 13 October 2026, after a cost pilot of 6 runs (section 8) and before the first run of this
rerun; frozen by the SHA-256 hashes in `FREEZE.txt`. Any later change goes to `AMENDMENTS.md`. The pilot runs are
excluded.
**Paper:** "What Agent Traces Do Not Record", preprint v3.21, Section 3.1 (the controlled rerun).
**Budget:** USD 10 of OpenAI API credit (Blessed, 13 October 2026).

## 1. Why

The two reruns so far used gpt-oss:120b, which is not one of DAB's models; the embedding broke Python calls in 9% of
its runs. A blind reviewer of v3.21 asked (a) whether the paper's "without the harness" counts, made by subtracting
the harness-caused failed calls from DAB's logs, match what a model does when the code is passed unchanged, and
(b) for a rerun with DAB's own models. gpt-5-mini is one of DAB's six models and the one the embedding hit most in
DAB's logs: 1,624 of its 2,709 failed calls (60%) in 3,187 runs, so 0.85 failed calls per run, and 0.34 after the
subtraction.

## 2. Design

- **Arms:** A, DAB's harness as released (commit `6ad9d68`); B, the same harness with the model's code passed to
  `exec()` unchanged and the one clause of the prompt that describes the embedding changed. Both exactly as in
  `dab-rerun/` and `dab-rerun-ext/`.
- **Model:** `gpt-5-mini-2025-08-07` through OpenAI's API, with DAB's request (model, messages, tools; no temperature,
  no seed, default reasoning). DAB called "gpt-5-mini" through Azure; its date is not recorded. The only code change
  is a branch in `common_scaffold/DataAgent.py` for the prefix `openai/` that creates `OpenAI(api_key=...)`
  ([rerun patch: model access]). The prompt is DAB's prompt for GPT models.
- **Tasks and settings:** DAB's 54 tasks, DAB's hints (`--use_hints`), at most 100 requests and 2 hours per run,
  DAB's validators, DAB's Pass@1.
- **Runs:** blocks of 108 jobs (54 tasks × 2 arms), block *k* shuffled with seed 20261013 + *k*, blocks 0 to 9 in
  order (`rerun.py`, `DAB_BLOCKS=10`), 3 workers, each with its own PostgreSQL and MongoDB.

## 3. Budget and stopping

- No new job starts once the recorded cost reaches USD 8.50; no new model request is sent once it reaches USD 9.50.
- The analysis uses complete blocks only (a block is complete when at least 100 of its 108 jobs have a kept run).
- Stopping depends only on cost, never on results.

## 4. Exclusions

As in the earlier reruns: a run that failed for an infrastructure reason (the model request failed, or the agent
process ended without a result) is retried up to 2 times; a run that reached the 2-hour cap counts as wrong.

## 5. Hypotheses and analysis

The frozen analysis of the first rerun (`dab-rerun/analysis.py`) runs unchanged on the complete blocks, through
`analysis_mini.py`, which adds H3 and H5.

- **H1, primary: passing the code unchanged removes the harness-caused failed calls.** Failed tool calls per run,
  B − A: mean over tasks of the per-task difference; 95% bootstrap interval over tasks (10,000 resamples); two-sided
  Wilcoxon signed-rank test on the non-zero per-task differences.
- **H2: harness-caused syntax errors.** The share of A's Python calls that fail with a syntax error that the
  embedding introduced (the compile test of the paper), against DAB's logs for gpt-5-mini; and in B (expected: 0).
- **H3: the subtraction check.** Per task, B's failed calls per run minus (A's failed calls per run − A's
  harness-caused calls per run), with the method of H1. A value near zero means that subtracting the harness-caused
  calls approximates what the model does without the effect; a negative value means the subtraction overstates the
  failures without the effect.
- **H4: accuracy.** Pass@1, B − A, DAB's definition, bootstrap within datasets (as in the earlier reruns).
  Secondary: with at most 10 runs per task, this rerun is not powered to detect a small change.
- **H5: requests to the model per run,** B − A, with the method of H1.
- **No looking:** nobody looks at results by arm until the runs end or stop.

## 6. Code

`rerun.py` (the runner of `dab-rerun-ext/`, with the model, prices, output folder, run-folder names, seed, number of
blocks and caps changed), `analysis_mini.py`, the frozen `dab-rerun/analysis.py`, and the patched
`common_scaffold/DataAgent.py`. Hashes are in `FREEZE.txt`.

## 7. Prices

gpt-5-mini: USD 0.25 per million input tokens, 0.025 cached, 2.00 output (OpenAI's model page, 13 October 2026).

## 8. Pilot (excluded)

Six runs, block 99, written to `pilot/`: A on yelp 1, bookreview 1, stockmarket 1 and agnews 1; B on crmarenapro 1
and googlelocal 1. All six ended with an answer; 27 requests; recorded cost USD 0.037 (about USD 0.006 per run).
At that rate, 10 blocks (1,080 runs) would cost about USD 7, so the USD 8.50 cap may end the rerun before block 9.
