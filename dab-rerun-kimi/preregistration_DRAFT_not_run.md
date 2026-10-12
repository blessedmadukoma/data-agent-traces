# Pre-registration: second controlled DAB rerun, with a model that the harness affects strongly

**Status:** written on 12 October 2026 before the first run, and frozen by the SHA-256 hashes in `FREEZE.txt` beside this file. Any later change is reported in `AMENDMENTS.md`. The 6 pilot runs of section 9 are excluded from every analysis.
**Paper:** "What Agent Traces Do Not Record" (preprint v3.19), Section 3.1, controlled rerun.
**First rerun:** `dab-rerun/preregistration.md` (gpt-oss:120b, frozen 6 October 2026). This rerun keeps its design, measures and analysis, except where section 2 says otherwise.

## 1. Why a second rerun

In the first rerun, the released harness caused 1.34 failed calls per run for gpt-oss:120b, and accuracy did not change detectably (Pass@1 difference −1.5 points, 95% CI −7.6 to +4.1). In DAB's released logs, the harness caused 2.24 failed calls per run for kimi-k2-thinking (7,130 in 3,188 runs), the highest rate of the six models, and kimi-k2-thinking made 44% of the remaining failed calls. A reader can therefore ask whether the null result holds for a model that the harness affects more. Ollama Cloud no longer serves kimi-k2-thinking; it serves kimi-k2.6, a later model of the same family. We use it, and H1 measures whether the harness affects it as strongly.

## 2. Differences from the first rerun

| Item | First rerun | This rerun |
|---|---|---|
| Model | `gpt-oss:120b` | `kimi-k2.6` on Ollama Cloud (`ollama/kimi-k2.6`), same endpoint and request settings (no temperature, no seed, as in DAB) |
| Arms | A, B, C | **A and B only.** Arm C (the note on cut MongoDB results) was exploratory, with 11 queries; it is not repeated. |
| Runs | 5 per query and arm | **Planned: 5 per query and arm** (540 runs), in blocks; the budget rule of section 4 may lower the number of blocks. |
| Block order | seed 20261006 + *k* | seed 20261012 + *k*; after block 0, blocks run without waiting at block boundaries (amendment 4 of the first rerun) |
| Ledger | input and output tokens | input, cached input and output tokens (`common_scaffold/budget.py` reads `prompt_tokens_details.cached_tokens`) |
| Prices | USD 0.15 / 0.60 per 1M | USD 0.95 input, 0.16 cached input, 4.00 output per 1M tokens (Ollama Cloud, 12 October 2026) |
| Judge check | none | a human check of the judge's labels (section 5, H4b) |

Everything else is as in the first rerun: DAB commit `6ad9d68`, the 54 leaderboard queries, DAB's baseline agent with hints, at most 100 LLM calls per run, Python timeout 600 s, the executor image `python-data:3.12` (pandas 2.3.0, pyarrow 22.0.0, numpy 2.2.6), PostgreSQL 17 and MongoDB 8.2 per worker, the harness patches in `dab-rerun/harness-patches.diff` (arm B passes `repr()` of the code and changes one clause of key rule 7), a 2-hour wall-clock cap per run, and up to 2 retries of an infrastructure failure.

## 3. Hypotheses and analysis (as in the first rerun)

- **H1, manipulation check.** Arm A has harness-caused failures (a failed `execute_python` call whose code compiles under Python 3.12 on its own and whose decoded string does not); arm B has none. Report per Python call and per run, and compare with kimi-k2-thinking in DAB's logs (2.24 per run).
- **H2, primary.** Δ = Pass@1(B) − Pass@1(A), two-sided; 95% CI by bootstrap over queries stratified by dataset (10,000 resamples, seed 20261012); Wilcoxon signed-rank test on the 54 per-query differences. "The embedding changed accuracy" only if the CI excludes 0 and p < 0.05; otherwise "no change was detected", with the CI as the range not ruled out.
- **H3.** Failed tool calls per run, and the share of failed calls recorded as syntax errors, B − A.
- **H4a.** Among failed runs, the share that DAB's judge labels FM4 (implementation error), B − A. Judge: DAB's failure-analysis prompt with `gpt-oss:120b` (DAB used gpt-5, which we cannot call), on every failed run; we also report the runs that DAB's own pipeline judges (failed runs that end with an answer).
- **Pooled with the first rerun (secondary):** H2 and H3 for both models together, with the model as a stratum.

## 4. Budget and number of blocks

- **Budget:** the USD 50 for this rerun and E2 together is split as USD 39 for this rerun and USD 11 for E2. The runner starts no new run when the ledger is at or above USD 36 and stops a run before a model call at or above USD 38; the judge may spend up to USD 39 in total. Blessed may raise the cap before block 1 starts; any raise is recorded as an amendment, with the cost of block 0 as its only input.
- **Number of blocks:** block 0 (108 runs, every query once in each arm) runs first. From its cost, the runner projects the cost per block, and the number of blocks is the largest number of complete blocks, at most 5, that fits under the cap. The number depends only on cost and time; nobody looks at accuracy, failures or judge labels by arm before the last block ends.
- **Time:** if the runs must stop for time, only complete blocks are analysed, and we report the planned and achieved numbers.
- **Precision:** with fewer blocks, the intervals of H2 to H4 are wider. H1 rests on every Python call and stays precise with one block.

## 5. The human check of the judge (H4b)

- **Sample:** after the judge has labelled every failed run, 40 failed runs, 20 per arm, drawn at random (seed 20261012) among failed runs that end with an answer.
- **Coder:** a person who did not run the experiments, blind to the arm and to the judge's label. The page shows the query, the ground-truth answer and the trace in DAB's judge format, without the system prompt (key rule 7 would reveal the arm). The coder applies DAB's own instruction and failure modes FM1 to FM4.
- **Report:** agreement between the coder and the judge on FM4 (yes or no) and on each mode, with Cohen's kappa, per arm and pooled. Disagreements are listed by run.
- **Release:** the traces contain DAB's data, so the labelling page stays private; we release the labels and run identifiers only.

## 6. Exclusions and retries

As in the first rerun: `llm_response_failed` and `agent_run_failed` (except a budget stop) are infrastructure failures, retried at most twice, then excluded and counted per arm; the 2-hour cap counts as a wrong answer; a run that ends without a tool call is validated as usual.

## 7. Workers

3 workers, each with its own PostgreSQL and MongoDB instance; more workers may be added for throughput, as in amendment 6 of the first rerun. Ollama Cloud queues requests beyond the plan's concurrency, so the workers share it with E2.

## 8. Code (hashes in `FREEZE.txt`)

`rerun.py` (this folder), the patched DAB files listed in `dab-rerun/FREEZE.txt` (with `common_scaffold/budget.py` in its new version), `dab-services.sh`, and the executor image. The analysis reuses `dab-rerun/analysis.py` and the judge `dab-rerun/judge.py`, with paths and the model name as parameters; any other change is a deviation.

## 9. Pilot runs (excluded)

On 12 October 2026, 6 runs (yelp 1, stockmarket 1 and crmarenapro 1, each in A and B) checked the set-up and the cost per run. They are written to `pilot/` and excluded.

## 10. Release

As amended for the first rerun: statistics, scores, per-run summaries, short quotes and code; no full traces.
