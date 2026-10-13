# Controlled DAB rerun with gpt-5-mini

Pre-registration: [preregistration.md](preregistration.md), frozen by the hashes in [FREEZE.txt](FREEZE.txt) before the first run.

## What ran

- **Arms:**
  - A: DAB's harness as released (commit `6ad9d68`).
  - B: the same harness, with the model's code passed to `exec()` unchanged and the one prompt clause that describes the embedding changed.
- **Model:** `gpt-5-mini-2025-08-07` through OpenAI's API, with DAB's request.
- **Tasks:** DAB's 54 tasks with DAB's hints, at most 100 requests and 2 hours per run.
- **Workers:** 3.

## How it ended

- **Budget cap.** The recorded cost reached the USD 8.50 cap after 589 kept runs (ledger: USD 8.504). Under section 3 of the pre-registration, no job may start after that.
- **Runner stop.** At 16:55 UTC on 13 October 2026, the cloud machine shut down while three jobs were in flight. One of them left an empty `final_agent.json`, and `rerun.py` stopped with a `JSONDecodeError` when it read that file. Because the cost had already reached the cap, a restart would have started no job.
- **Data analysed.** As planned, the analysis uses complete blocks only: blocks 0 to 4. That is 5 runs per task and arm: 270 runs in A, and 267 in B after 3 infrastructure failures. Block 5 (49 runs) is incomplete and excluded.
- **Disk clean-up.** After the analysis, the large query results that DAB's harness caches in each run folder (`exec_tool_work_dir/file_storage`) were deleted to free disk space. No analysis reads them.

## Results

To check every number below without DAB's data: `python3 dab-rerun-mini/records.py` (needs numpy and scipy).

`python3 dab-rerun-mini/analysis_mini.py` writes [analysis.json](analysis.json) (the frozen analysis of `dab-rerun/`) and [analysis_mini.json](analysis_mini.json). Differences are B − A: the mean over tasks of the per-task difference, with a 95% bootstrap interval over tasks. For Pass@1, the interval resamples tasks within each dataset.

| Measure | A, released | B, code unchanged | B − A (95% CI) |
| --- | --- | --- | --- |
| Runs | 270 | 267 | |
| Python calls with a syntax error from the embedding | 26.8% (146 of 544) | 0% (0 of 406) | |
| Failed calls per run | 0.97 | 0.51 | −0.46 (−0.96 to +0.01); Wilcoxon p = 0.044 |
| Subtraction check: B minus (A − harness-caused) | | | +0.08 (−0.27 to +0.43); p = 0.13 |
| Requests to the model per run | 7.35 | 6.58 | −0.77 (−1.81 to +0.08); p = 0.18 |
| Pass@1 | 44.7% | 35.1% | −9.6 points (−14.2 to −5.1); p = 0.016 |

**Reading:**

- With the released harness, the embedding broke 26.8% of gpt-5-mini's Python calls, in 39 of 270 runs. With the code unchanged, it broke none.
- B's failed calls per run minus A's count after subtracting the harness-caused calls is +0.08 (CI −0.27 to +0.43). For gpt-5-mini, one of DAB's six models, the rerun is therefore consistent with the subtraction that the paper applies to DAB's logs, but too imprecise to show that the two are equal.
- Accuracy fell with the code unchanged. The pre-registration makes accuracy a secondary outcome and predicts no direction for it, and we do not know why it fell.
- The pre-registration states no decision rule. After these runs, the paper applies the rule that the first rerun pre-registered for accuracy (95% interval excludes zero and p < 0.05). By that rule, the fall in failed calls (H1) is not detected and the fall in Pass@1 is.
- Post hoc (`records.py`), we looked for a cause, without result. Split by whether the released harness broke a step in at least one of a task's 5 A runs, Pass@1 B − A is −9.8 points in 19 tasks (Wilcoxon p = 0.039) and −7.1 in the other 35 (p = 0.10). But 4 of these 35 tasks have `\n` in B's code (−30 points), and the other 31 show −3.9 (p = 0.31). Split instead by whether either arm's code contains `\n`, it is −19.2 in 22 tasks (p = 0.001) and −3.6 in 32 (p = 0.49). Both splits depend on outcomes, so neither shows whether the removed errors explain the fall. Harness B also changes one clause of the prompt ("will be quoted by triple double-quotes and passed as a string to `exec(...)`" becomes "will be passed unchanged as a string to `exec(...)`"); we did not test whether that clause explains it.
- Post hoc, no B code contains a doubled `\\n` or `\\t` (A's code does in 3 Python calls). `\n` occurs in fewer of B's Python calls than of A's (25 of 406, against 110 of 544). B made fewer Python calls per run (1.52 against 2.01) but used Python in more runs (81% against 77%).
- Means per run are over tasks. In B, one PATENTS task has 2 runs, because 3 of its runs failed for infrastructure reasons after their retries.

## Files

| File | Contents |
| --- | --- |
| `rerun.py` | The runner. |
| `DataAgent_patched.py` | DAB's agent with the model-access branch. |
| `analysis_mini.py` | The analysis. |
| `records.py` | Checks the numbers above from `runs.jsonl` and `calls.jsonl` alone, with the frozen statistics of `dab-rerun/analysis.py`, and prints the post hoc splits. `records.py export` rebuilds `calls.jsonl` from the run folders. |
| `calls.jsonl` | Per-run call statistics: tool calls, Python calls, failed calls, harness-caused syntax errors, escape sequences. No code, answers or data. |
| `escape_check.py` | The first post hoc check of escape sequences (needs the run folders; `records.py` supersedes it). |
| `runs.jsonl` | One record per run, without the answers or any DAB data. |
| `analysis.json`, `analysis_mini.json` | The analysis outputs. |
