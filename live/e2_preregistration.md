# Pre-registration: E2, a second live experiment on QRData with checker release 4

**Status:** written on 12 October 2026 before the first run of this experiment, and frozen by the SHA-256 hashes in `e2_FREEZE.txt` beside this file. Any later change is reported as a deviation in `e2_AMENDMENTS.md`. The 24 pilot runs of section 9 are excluded from every analysis.
**Paper:** "What Agent Traces Do Not Record" (preprint v3.19), Section 4.3, live experiment.

## 1. Why a second live experiment

The first live experiment (E1, 24 September 2026) has three faults that its results cannot repair:

1. **Our harness lost answers.** It defined `final_answer` in the model's namespace without saying so; in 29% of the gpt-oss:120b runs the model redefined that name and the harness recorded no answer. The recovered analysis is post hoc.
2. **It tested mainly missing files.** Its prompt gave each question's data paths (`input/name.csv`) but did not say that they are relative to the working directory. Of the 567 executed data-reference errors in the plain-prompt arm without the checker, 510 read missing files, mostly invented paths such as `/mnt/data/input/arctic.csv`. One sentence that tells the model to use exactly the given paths removed every file error in the pilot (section 9), so most of these errors need no checker.
3. **It had no power analysis.** Its 95% intervals for accuracy differences spanned about ±3.3 to ±4.2 points.

E2 removes the three faults: `final_answer` cannot be replaced, the prompt gives the exact paths, and the sample size is set for intervals of about ±1.5 points. E2 also adds the baseline that E1 did not test: an executor whose error message lists the names that exist.

## 2. Arms

All arms use the same model, prompt, sandbox, data, settings and run order. They differ only in what happens around a step.

| Arm | Before a step runs | After a step fails with a missing-name error |
|---|---|---|
| **N, no checker** | The checker decides in shadow mode; the step always runs. | The model receives the executor's error message. |
| **R, checker** | Checker release 4 (`dabstep-gate/v9/gate.py`) decides. On *block*, the step does not run and the model receives the receipt of E1 (the missing item, the names that exist, other files that contain the missing name). | The model receives the executor's error message. |
| **E, names in the error** | The checker decides in shadow mode; the step always runs. | The error message is followed by a note that lists the question's data files and the columns of each (`names_text` in `e2_run.py`). |

A **missing-name error** is a step that raises `KeyError`, `FileNotFoundError`, `IsADirectoryError` or `NotADirectoryError`. In arms N and E, the checker's decision is recorded but has no effect.

## 3. Fixed settings

- **Benchmark:** QRData, all 411 questions (`data/QRData/benchmark/QRData.json`), with its data files mounted read-only at `input/`.
- **Prompt:** the QRData system prompt of E1, plus one sentence: "The data files are in the folder input/ of the working directory: read them with exactly the paths given with the question, for example pd.read_csv('input/name.csv')." The user message gives the question, the paths of its data files and the list of files in `input/` (the first 20 names), as in E1.
- **Model:** `gpt-oss:120b` on Ollama Cloud (OpenAI-compatible endpoint), temperature 1.0, at most 4,096 output tokens per reply, the replicate number sent as the request's `seed`. This is the E1 model, so E2 tests the same agent with the three faults removed. The model will not be changed after the runs start.
- **Agent loop:** at most 15 replies per run; one Python code block per reply; 120 s per step. A run ends when a step calls `final_answer` and runs without an error. A step that times out or crashes the executor ends the run without an answer, as in E1.
- **`final_answer` protected:** the runner (`live/e2_runner.py`) renames any definition of or assignment to `final_answer` in the model's code before it runs, so the harness's function always records the answer.
- **Sandbox:** one Docker container per run (`python:3.12-slim`), no network, read-only root, 2 GB memory, 1 CPU; Python 3.12.3 with pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0 and matplotlib 3.11.2. The checker asks this sandbox whether a file exists.
- **Checker:** release 4, `dabstep-gate/v9/gate.py`, SHA-256 `8a555f9feaa5c7f55773786f0ed0be8b7431268e7a2e0128095321cf1c488fe1`. Schema: the columns of each CSV file in the data folder (`kb_gate.domain_schema`). A checker crash lets the step run.
- **Replicates and order:** 10 replicates (seeds 1 to 10) of each question in each arm: 12,330 runs. Seeds run as blocks in the order 1 to 10. Block *s* holds the 1,233 (question, arm) jobs of seed *s* in an order shuffled with seed 20261012 + *s*, so the three arms share each block's time window.
- **Workers:** 3 to 6 parallel workers, each with its own sandbox per run.

## 4. Measures

- **Correct:** QRData's rule, as in E1 (`live/e1_report.correct`): a numerical answer within 3% of the gold answer; a multiple-choice answer by its prefix. A run without an answer is wrong.
- **Steps:** replies of the model in the run. **Tokens:** input, cached input and output tokens from the API.
- **Data-reference attempt (DRA):** a step on which the checker decides *block* (live in R, shadow in N and E), or which runs and raises a missing-name error. The definition is the same in all arms.
- **Executed missing-name error:** a step that runs and raises a missing-name error, split into files, columns or keys of a data file, and other keys, by the error message and the checker's reason.
- **Live false block of release 4:** in N and E, a step on which the checker decides *block* in shadow mode and which then runs without an error.
- **Block check:** each step blocked in R is executed again without the checker in a fresh sandbox, after the earlier executed steps of its run, as in E1. Classes: confirmed (raises the receipt's missing name), other error first, no error (a false block), not replayed.
- **Lost answers:** runs where a step called `final_answer` and ran without an error but the run recorded no answer. Expected: none.

## 5. Hypotheses and analysis

The unit of inference is the question. For each question, arm and measure, we average over the replicates; a difference between arms is the mean over questions of the per-question differences.

**H1, primary: the checker changes accuracy.** Δ_RN = accuracy(R) − accuracy(N), two-sided.
- 95% CI: percentile bootstrap over questions (all replicates of a question move together), 10,000 resamples, seed 20261012.
- Test: sign-flip permutation test on the 411 per-question differences, 100,000 permutations, two-sided.
- **Decision rules:** "the checker changed accuracy" only if the interval excludes 0 and p < 0.05. "The checker did not lower accuracy by more than 2 points" if the lower bound is above −2 (the E1 margin). Otherwise "no change was detected", and the interval is the range of effects that the data do not rule out.
- For continuity with E1, we also report McNemar's exact test and Newcombe's paired interval on (question, seed) pairs.

**H2: names in the error message change accuracy.** Δ_EN, with the method of H1. **H3: the checker against names in the error message.** Δ_RE, with the method of H1. H2 and H3 are secondary; we report them without correction for multiple tests and say so.

**H4: steps and tokens.** Steps and output tokens, R − N, E − N and R − E: bootstrap CI as in H1 and the exact sign test on the per-question differences.

**H5: runs that reach a data-reference attempt.** Until its first DRA, a run receives the same messages in every arm, because the arms differ only at a block (R) or after a missing-name error (E), and both are DRAs. So the share of runs that reach a DRA has the same expected value in every arm (we report it per arm as a check), and

  Δ (all runs) = share of runs with a DRA × Δ (runs with a DRA).

We report accuracy, steps and tokens among runs with a DRA, per arm, and the three differences with a bootstrap over questions (questions without such a run in an arm contribute nothing to that arm). This analysis says how large the effect is where the arms can differ; H1 says how large it is for the agent as a whole.

**H6: live false blocks of release 4.** The count of shadow blocks in N and E on steps that then ran without an error, with an exact 95% upper bound per step and per run. The block check of R, with the same bound for "no error".

**H7, descriptive: which errors remain when the prompt gives the paths.** Executed missing-name errors per arm and per 100 runs, by kind (files, columns or keys of a data file, other keys); DRAs per 100 runs; and, after the first DRA, whether the next step uses an existing name.

**Not claimed in any case:** that the effects hold for other models, benchmarks or prompts. E2 uses one model and one benchmark.

**If DRAs are rare** (for example, in fewer than 5% of runs), we still run all planned replicates and report the result. A rare error bounds what any check of it can change; H5 then gives the effect where it occurs.

## 6. Power

In E1, 23.3% of the (question, seed) pairs of the two plain-prompt arms were discordant (one arm correct, the other not). With 10 replicates per question, the standard error of a mean difference is about √(0.233 / 4,110) = 0.75 points if the effect does not vary between questions, so the 95% interval spans about ±1.5 points. Variation of the effect between questions widens it; we report the achieved width. H5 has fewer runs: if 10% of runs reach a DRA, about 411 runs per arm, and its intervals span about ±7 points.

## 7. Exclusions, stops and budget

- **API failure:** a run whose model call fails after 6 attempts is not recorded, and the next start of the runner repeats it. A (question, arm, seed) job that is still missing at the end is reported, and its question–seed pair is left out of the paired comparisons of that arm.
- **Executor lost** (time limit or crash): the run ends without an answer and counts as wrong. We report the count per arm.
- **Stops:** we do not look at accuracy, steps or tokens by arm until all planned seeds are done or the run is stopped. The run may be stopped only for time or cost, never because of results. If it stops early, the analysis uses only the seeds whose block is complete, and we report the planned and achieved numbers.
- **Budget:** hard cap USD 11 at gpt-oss:120b prices (USD 0.15 input, 0.014 cached input, 0.60 output per million tokens); no new run starts at or above the cap. Expected cost: USD 8 to 10 (the pilot cost USD 0.00066 per run).

## 8. Code (hashes in `e2_FREEZE.txt`)

`live/e2_run.py` (driver), `live/e2_runner.py` (sandbox process), `dabstep-gate/v9/gate.py` (checker), and the E1 modules it imports for the prompt, code extraction and receipts (`kramabench/kb_run.py`, `kramabench/kb_gate.py`, `qrdata/qr_run.py`, `rq3-pilot/rq3_run.py`). The analysis script `live/e2_report.py` will implement section 5 without changes to it; if it needs a change, the change is listed as a deviation.

## 9. Pilot runs (excluded)

On 12 October 2026, 24 runs (the first 8 questions, each arm, seed 1) checked the set-up: USD 0.016 in total, 4.25 replies and 7.6 s per run on average. One step raised a `KeyError` (a missing column), one step was blocked in R (a missing column), one shadow block fell on a step that raised an error, and no run lost its answer. After the pilot, the job order was changed from one shuffle of all seeds to one shuffle per seed (section 3), so that a stop leaves complete seeds.

## 10. Outputs and release

Per run: answer, correctness, steps with code, checker decisions and receipts, errors, tokens and cost. QRData's data are public, so the full results and transcripts are released with the analysis code.
