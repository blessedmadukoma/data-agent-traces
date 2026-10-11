# Pre-registration: does DataAgentBench's harness change what its traces say? A controlled rerun

**Status:** approved by Blessed on 6 October 2026, 20:35 (Kigali time), with decisions D1 to D4 as recommended. Frozen before the first run (hashes in `FREEZE.txt` beside this file). Three smoke runs (section 9) checked the set-up and are excluded from every analysis.
**Paper:** "What Agent Traces Hide" (v2), research question RQ1, second part: does the harness change what a reader of the trace concludes?
**Freeze:** after approval, this file and the code in section 8 are frozen by their SHA-256 hashes before the first run. Any later change is reported as a deviation.

## 1. Question

DataAgentBench (DAB) runs the model's Python code by placing it inside a triple-quoted string literal (`ExecTool.py`, line 208 at commit `6ad9d68`, 2 October 2026). Python decodes escape sequences in that literal before `exec()` runs it, so valid code such as `print("\n".join(rows))` fails with a syntax error, and a regular expression such as `'\\d'` silently becomes `'\d'`. DAB also returns only five documents for a MongoDB query that sets no limit, without telling the model.

In the released DAB logs, these effects cause more failed calls than all data-reference errors together (paper, Section 3.1). This experiment asks whether they also change the conclusions that a reader draws from the traces:

1. the accuracy of the agent (Pass@1);
2. the failures that the trace records;
3. the failure labels that DAB's own failure-analysis judge assigns.

**What the model is told.** DAB's system prompt, key rule 7, already says: "your code will be quoted by triple double-quotes and passed as a string to `exec(...)` … So you must ensure your code is compatible with this execution method. For exampe, do not use triple double-quotes in your code". The prompt therefore discloses the embedding. It does not say that escape sequences are decoded. This rule was also in the version that produced the logs in the paper (`refs/pull/6/head`).

## 2. Arms

All arms use the same model, prompt, data, executor image, settings and run order. They differ only as follows.

| Arm | How the code reaches `exec()` | Key rule 7 | MongoDB result cut by the default limit | Queries |
|---|---|---|---|---|
| **A, released** | inside `"""…"""`, as released | as released | cut to 5, no note | all 54 |
| **B, code unchanged** | `repr()` of the code, so `exec()` receives exactly what the model wrote | one clause changed: "will be quoted by triple double-quotes and passed as a string to `exec(...)`" becomes "will be passed unchanged as a string to `exec(...)`"; the rest of the rule stays | cut to 5, no note | all 54 |
| **C, code unchanged and cut marked** | as B | as B | cut to 5, and the tool message ends with: "Note: this query set no "limit", so only the first 5 of N matching documents were returned. Add a "limit" field to the query to return more." | the 11 queries on MongoDB datasets (yelp 7, agnews 4) |

The default limit stays at 5 in arm C, so the model receives the same records; only the note is added.

## 3. Fixed settings

- **Benchmark:** DAB main branch, commit `6ad9d68` (2 October 2026); the 54 leaderboard queries over 12 datasets (DEPS_DEV_V1 2, GITHUB_REPOS 4, PANCANCER_ATLAS 3, PATENTS 3, agnews 4, bookreview 3, crmarenapro 13, googlelocal 4, music_brainz_20k 3, stockindex 3, stockmarket 5, yelp 7). The data files listed in DAB's `dataset_manifest.tsv` were verified against its SHA-256 hashes; the other files come from the repository at the same commit.
- **Agent:** DAB's own baseline agent (`run_agent.py`), with dataset hints (`--use_hints`, as on the leaderboard), at most 100 LLM calls per run, Python timeout 600 s.
- **Model:** `gpt-oss:120b` on Ollama Cloud, through its OpenAI-compatible endpoint. As in DAB, the request sets no temperature and no seed. The model is fixed; it will not be changed after the runs start.
- **Executor:** Docker image `python-data:3.12` built from DAB's Dockerfile (python:3.12-slim), with pandas 2.3.0, pyarrow 22.0.0 and numpy 2.2.6 (the versions pinned in DAB's `environment.yaml`), image ID `sha256:3887654f08b0…`.
- **Databases:** PostgreSQL 17 and MongoDB 8.2.12. Each of the 3 parallel workers has its own PostgreSQL and MongoDB instance, because DAB loads and drops databases under fixed names at the start and end of every run.
- **Runs:** 5 per query and arm, in 5 blocks. Block *k* holds run *k* of every query in every arm (119 runs), in an order shuffled with seed 20261006 + *k*, so that the arms are interleaved in time. Total: 595 runs (270 A, 270 B, 55 C).

## 4. Measures

- **Correct answer:** DAB's own `validate.py` for the query, applied to the run's final answer. **Pass@1** as DAB defines it: the pass rate of each query over its runs, averaged over the queries of each dataset, then averaged over the 12 datasets.
- **Harness-caused failure (manipulation check):** a failed `execute_python` call where the model's code compiles under Python 3.12 on its own and the decoded code (the string that `exec()` receives in arm A) does not. This is the rule used in the paper. A **silent change** is a call where both compile and the decoded code differs from the model's code.
- **Recorded failures:** tool calls whose result is a failure, per run, and their error types.
- **DAB judge label:** DAB's failure-analysis prompt (`failure_analysis/`, failure modes FM1 to FM4), applied to every failed run, with `gpt-oss:120b` as the judge (DAB used `gpt-5`, which we cannot call). FM4 includes "improper escaping" and "incorrect regular expressions" as the agent's implementation errors.
- **Cost:** tokens per run and in total, from the API's usage counts.

## 5. Hypotheses and analysis

**H1, manipulation check.** Arm A has at least one harness-caused failure; arms B and C have none. Report the rate per Python call and per run for each arm, and compare arm A's rate with the six models in the DAB logs.

**H2, primary: the embedding changes accuracy.** Effect: Δ = Pass@1(B) − Pass@1(A), two-sided.
- 95% confidence interval: bootstrap over queries, stratified by dataset, 10,000 resamples, seed 20261006.
- Test: Wilcoxon signed-rank test on the 54 per-query differences in pass rate.
- **Decision rule:** we write "the embedding changed accuracy" only if the interval excludes 0 and p < 0.05. Otherwise we write "no change in accuracy was detected" and report the interval as the range of effects that the data cannot rule out.
- Also reported: the mean over the 54 queries with equal weight.

**H3, secondary: the embedding changes the recorded failures.** Failed tool calls per run, and the share of failed calls recorded as syntax errors, compared between A and B with the same bootstrap and test on per-query means.

**H4, secondary: the embedding changes DAB's own diagnosis.** Among failed runs, the share labelled FM4 by the judge, A against B: difference with a 95% bootstrap interval over queries. Prediction: higher in A.

**H5, exploratory: marking cut results.** On the 11 MongoDB queries, Pass@1(C) − Pass@1(B), with the same method. We also count how often the model adds a limit after a note. With 11 queries, this test has low power, and we label it exploratory.

**Not claimed in any case:** that DAB's published scores are wrong for the models in the DAB paper. This experiment uses one other model.

**If arm A shows few harness-caused failures** for this model, we still run every block and report the result as it is. A small effect for this model is a result, because the logged rates differ more than tenfold between models.

## 6. Exclusions, retries and limits

- **Infrastructure failure:** a run that ends with `llm_response_failed` (the API failed after DAB's own 3 attempts) or `agent_run_failed` (an error outside the model's control, such as a database load or Docker error). Such a run is repeated, at most 2 more times. If it still fails, it is excluded, and we report the count per arm.
- **Wall-clock cap:** 2 hours per run. A run that reaches it counts as a wrong answer (as when DAB reaches its step limit), and we report the count per arm.
- **Model ends without a tool call:** this is DAB's own behaviour (`no_tool_call`). The text becomes the final answer and is validated as usual.
- **Smoke runs:** the 3 runs of section 9 are excluded.

## 7. Budget rule

- **Hard cap: USD 50**, from the API's token counts at gpt-oss:120b prices ($0.15 input, $0.60 output per million tokens, checked 6 October 2026). The runner stops every run before a call that would start above the cap, and starts no new run above USD 48.
- **After block 0** (119 runs), the runner projects the cost of all 5 blocks from the cost of block 0. If the projection exceeds USD 48, it lowers the number of blocks for all arms equally to the largest number that fits. If fewer than 3 blocks fit, it stops after block 0, and Claude asks Blessed before going on.
- **Expected cost:** USD 5 to 15 for 595 runs (three smoke runs cost USD 0.019 together). Expected time: 4 to 8 hours with 3 parallel workers.

## 8. Changes to DAB's code (the full diff is in `harness-patches.diff`)

1. **Model access:** an `ollama/` branch that sends the same request to Ollama's OpenAI-compatible endpoint.
2. **Budget:** a token ledger and the cap in section 7. This does not change the agent's messages.
3. **Arms B and C:** the `repr()` line in `ExecTool.py` and the clause in key rule 7.
4. **Arm C:** count the matching documents when the default limit applied, and add the note.
5. **Infrastructure:** `mongorestore --uri`, so that each worker restores into its own MongoDB instance (the original command always used localhost:27017).
6. **Executor image:** pandas, pyarrow and numpy installed from wheels downloaded on the host, at DAB's pinned versions, because the build network cannot verify TLS through the workspace proxy.

## 9. Smoke runs (excluded)

On 6 October 2026, before this draft, three runs checked the set-up:
- bookreview query 1, arm A;
- bookreview query 1, arm B;
- yelp query 1, arm C.

All three reached an answer. Arm B received the changed rule 7, and arm C showed the note ("the first 5 of 8 matching documents"). Their total cost was USD 0.019. In the 6 Python calls of these runs, the decoding changed no code.

## 10. Outputs and release

- **Per run:** answer, correctness, terminate reason, tool calls, token counts, and the judge label for failed runs.
- **Released:** the scores, the per-run summaries, the patches, the runner and the analysis code.
- **Not released yet:** the full traces contain query results from DAB's datasets, so we ask the DAB authors before we release them.

## 11. Decisions (all approved as recommended)

- **D1.** Change the one clause of key rule 7 in arms B and C, as in section 2. **Recommended:** yes. Without it, arm B's prompt would describe an embedding that no longer happens.
- **D2.** Use `gpt-oss:120b` as the judge model for H4. **Recommended:** yes.
- **D3.** Include arm C (55 runs). **Recommended:** yes.
- **D4.** Set a 2-hour wall-clock cap per run. **Recommended:** yes.
