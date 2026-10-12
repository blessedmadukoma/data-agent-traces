# Pre-registration: extension of the controlled DAB rerun with gpt-oss:120b (blocks 5 to 9)

**Status:** written on 12 October 2026 before the first run of the extension, and frozen by the SHA-256 hashes in `FREEZE.txt` beside this file. Any later change is reported in `AMENDMENTS.md`. Approved by Blessed on 12 October 2026 ("Go with option 2").
**Paper:** "What Agent Traces Do Not Record" (preprint v3.19), Section 3.1, controlled rerun.
**First rerun:** `dab-rerun/preregistration.md` (frozen 6 October 2026), blocks 0 to 4. This extension repeats its arms A and B, unchanged, in five more blocks.

## 1. Why an extension, and why not kimi

The first rerun found that the released harness broke 26.3% of gpt-oss:120b's Python calls (1.34 failures per run), but it could not detect an effect on accuracy: Pass@1 difference −1.5 points, 95% CI −7.6 to +4.1. Reviewers asked whether accuracy changes for a model that the harness affects strongly.

We first planned a rerun with kimi-k2.6, because kimi-k2-thinking had the highest rate in DAB's logs (2.24 failures per run) and Ollama Cloud no longer serves it. In two pilots on 12 October 2026 (12 runs, 3 of them stopped early to free Ollama's request slots; `dab-rerun-kimi/`), kimi-k2.6 did most of its work in SQL: in its 9 runs with the released harness it made 19 Python calls, and the harness caused 1 failure (about 0.1 per run). A rerun with kimi-k2.6 would therefore test a model that the harness hardly affects. gpt-oss:120b is the most strongly affected model that we can run, so more of its runs answer the reviewers' question best. The kimi pilots are excluded from every analysis.

## 2. Design

As the first rerun, arms A and B only:

- **Arm A:** the released harness. **Arm B:** `repr()` of the code, so `exec()` receives exactly what the model wrote, with the one clause of key rule 7 changed (decision D1 of the first rerun).
- **Runs:** blocks 5 to 9; block *k* holds the 108 jobs (54 queries × 2 arms) in an order shuffled with seed 20261006 + *k*. Blocks run in order, without waiting at block boundaries. Total: 540 runs.
- **Unchanged:** DAB commit `6ad9d68`; the 54 leaderboard queries; DAB's baseline agent with hints; at most 100 LLM calls per run; Python timeout 600 s; `gpt-oss:120b` on Ollama Cloud with no temperature and no seed; the patched DAB files (their SHA-256 hashes equal those in `dab-rerun/FREEZE.txt`, except `budget.py`, below); PostgreSQL 17 and MongoDB 8.2.12 per worker; the 2-hour wall-clock cap; up to 2 retries of an infrastructure failure.

## 3. Differences from the first rerun (none changes what the agent receives)

1. **No arm C.** It was exploratory (11 queries).
2. **Ledger:** `common_scaffold/budget.py` also records cached input tokens and prices them at USD 0.014 per 1M (Ollama Cloud, 12 October 2026). The first rerun priced all input at USD 0.15.
3. **Executor image:** rebuilt after the workspace was reset, from the same Dockerfile with pandas 2.3.0, pyarrow 22.0.0 and numpy 2.2.6; its image ID differs (`sha256:596528a6…`). The agent's own environment uses `psycopg2-binary` 2.9.9 instead of `psycopg2` 2.9.9.
4. **Workers:** 3, each with its own PostgreSQL and MongoDB instance. Ollama runs at most 3 requests at once for this account, and the second live experiment (E2) shares them.

## 4. Analysis

The analysis code of the first rerun (`dab-rerun/analysis.py`) is used with the paths and blocks as parameters.

- **Primary (H2, pooled):** Pass@1(B) − Pass@1(A) on blocks 0 to 9 together (10 runs per query and arm), with the method of the first rerun: DAB's Pass@1 (per-query pass rate, averaged over the queries of each dataset, then over the 12 datasets); 95% CI by bootstrap over queries stratified by dataset (10,000 resamples, seed 20261006); Wilcoxon signed-rank test on the 54 per-query differences. Decision rule as in the first rerun: "the embedding changed accuracy" only if the CI excludes 0 and p < 0.05; otherwise "no change was detected", and the CI is the range not ruled out. Arm C of the first rerun is not used.
- **Replication:** H2 on blocks 5 to 9 alone.
- **H1, H3 and H4a on blocks 5 to 9**, as in the first rerun (the traces of blocks 0 to 4 were lost when the workspace was reset; their results are in `dab-rerun/analysis.json`). Failed tool calls per run (H3) are also pooled over blocks 0 to 9, because `dab-rerun/runs.jsonl` records them per run.
- **H4b, a human check of the judge:** 40 failed runs of blocks 5 to 9 that end with an answer, 20 per arm, drawn at random (seed 20261012). A person who did not run the experiments labels each run with DAB's instruction and failure modes FM1 to FM4, blind to the arm and to the judge's label, on a private page that shows the query, the ground-truth answer and the trace in DAB's judge format without the system prompt. We report agreement with the judge on FM4 and on each mode (Cohen's kappa), per arm and pooled.
- **Power:** with 10 runs per query and arm instead of 5, the CI of H2 is expected to narrow from about ±6 to about ±4 points.

## 5. Budget, stops and blinding

- **Budget:** the runner starts no new run at or above USD 20 and stops a run before a model call at or above USD 22; the judge's calls count in the same ledger. Expected cost: about USD 10 to 14.
- **Stops:** the runs may stop only for time or cost. If they stop early, only complete blocks are analysed, and we report the planned and achieved numbers.
- **Blinding:** nobody looks at accuracy, failures or judge labels by arm before the last block ends.

## 6. Release

As amended for the first rerun: statistics, scores, per-run summaries, short quotes and code; no full traces.
