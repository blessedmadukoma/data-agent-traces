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
- A post hoc look (`escape_check.py`, which needs the local run folders) found no doubled escape sequences in B's code. B's Python calls contain escape sequences less often than A's (25 of 406, against 110 of 544), and B made fewer Python calls per run.
- Means per run are over tasks. In B, one PATENTS task has 2 runs, because 3 of its runs failed for infrastructure reasons after their retries.

## Files

| File | Contents |
| --- | --- |
| `rerun.py` | The runner. |
| `DataAgent_patched.py` | DAB's agent with the model-access branch. |
| `analysis_mini.py` | The analysis. |
| `escape_check.py` | The post hoc check of escape sequences. |
| `runs.jsonl` | One record per run, without the answers or any DAB data. |
| `analysis.json`, `analysis_mini.json` | The analysis outputs. |
