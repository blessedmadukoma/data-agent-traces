# Pre-registration: E3, a live experiment where data-reference errors are common

**Status:** written on 12 October 2026 after the pilot of `runs/e3pilot/` chose the second model (section 9) and before the first run of E3; frozen by the SHA-256 hashes in `e3_FREEZE.txt`. Any later change goes to `e3_AMENDMENTS.md`. The pilot runs are excluded.
**Paper:** "What Agent Traces Do Not Record", preprint v3.20, Section 3.4. Approved by Blessed on 12 October 2026 ("go ahead and run").

## 1. Why E3

E2 (`e2_preregistration.md`) found no change in accuracy from the checker (−0.27 points, 95% CI −0.54 to 0.00) or from an error message that lists the existing names (−0.10 points). Its prompt gave the data paths, however, so only 3.1% of runs ran a data-reference error. Two blind reviewers of v3.20 asked for a test where these errors are common, with more than one model. They also noted that E2 cannot test the post hoc loss of 4.0 points that E1 found with the plain prompt, where 42% of gpt-oss:120b runs ran such an error.

E3 therefore repeats E2 with E1's plain prompt and two models.

## 2. Arms, prompt and runner

- **Arms:** N (no checker), R (checker release 4, which blocks and returns the receipt) and E (no checker; a missing-name error is followed by the question's data files and their columns). They are defined exactly as in E2, section 2.
- **Prompt:** the QRData system prompt of E1, which gives the question's data paths without E2's sentence "read them with exactly the paths given …". Everything else is as in E2: the user message, at most 15 replies, 120 s per step, temperature 1.0, at most 4,096 output tokens, `final_answer` protected, and the same Docker sandbox and packages.
- **Runner:** `live/e3_fork.py`, which is `live/e2_fork.py` (E2's runner after its amendment 2: shared prefixes, one sandbox per branch) with the plain prompt and the model as a parameter.
- **Checker:** release 4, `dabstep-gate/v9/gate.py`, SHA-256 `8a555f9f…`, as in E2.

## 3. Models and runs

- **Models:** `gpt-oss:120b` (the E1 and E2 model), and `gemma4:31b`, chosen by the pilot's pre-set rule (section 9). Both run on Ollama Cloud.
- **Runs:** 10 replicates (seeds 1 to 10) of each of the 411 questions, for each model; each replicate gives one run per arm. That is 8,220 question–seed jobs and 24,660 arm-runs.
- **Order:** seeds run as blocks, one shuffle per seed (seed 20261012 + *s*), as in E2. The two models run at the same time, in separate processes that share the 3 concurrent requests.

## 4. Measures

As in E2, section 4: correctness (QRData's rule); steps; output tokens; data-reference attempts (DRA: a step that the checker blocks or would block, or that raises a missing-name error); executed missing-name errors by kind; live false blocks (shadow blocks in N and E on steps that run without error); the block check of R; lost answers.

## 5. Hypotheses and analysis

The question is the unit, and differences are paired on (question, seed). The script is `live/e2_report.py`, run once per model, with the pooled analysis in `live/e3_report.py`.

- **H1, primary: the checker changes accuracy when errors are common.** Δ = accuracy(R) − accuracy(N), pooled over the two models: for each question, the mean over models and seeds of R − N.
  - 95% CI: bootstrap over questions (10,000 resamples, seed 20261012).
  - Test: sign-flip permutation test on the 411 per-question means (100,000 permutations).
  - Decision rules as in E2: "changed accuracy" only if the CI excludes 0 and p < 0.05; "did not lower accuracy by more than 2 points" if the lower bound is above −2.
- **H1a and H1b, secondary:** H1 for each model on its own. H1a (gpt-oss:120b) repeats the comparison of E1 (post hoc −4.0 points) with the answer protected and with power.
- **H2:** E − N. **H3:** R − E. Both pooled and per model, with the method of H1.
- **H4:** steps and output tokens (bootstrap CI and exact sign test), pooled and per model.
- **H5:** runs with a DRA. For each model: the share per arm (a check that the shares are equal) and the differences in accuracy, steps and tokens among these runs (bootstrap over questions). This analysis is central in E3, because most changes happen in these runs.
- **H6:** live false blocks (shadow, N and E) and the block check of R, with exact 95% upper bounds.
- **H7, descriptive:** the kind of each executed missing-name error (file, data column or key, other key); and, after the first DRA, whether the next step repeats the error. Reported per model and arm.
- **Multiplicity:** one primary test (H1). The other tests are reported with their intervals and are labelled secondary.

## 6. Power

In E1, with this prompt, 42% of gpt-oss:120b runs ran a data-reference error. In E2's shared-prefix design, runs without a DRA are identical across arms, so the variance of a difference comes from the runs with a DRA. Suppose 40% of runs have a DRA, and the arms disagree on correctness in 40% of those. Then each model's 4,110 pairs give an SE of about √(0.4 × 0.4 / 4,110) ≈ 0.62 points (CI about ±1.2), and the pooled 8,220 pairs about ±0.9 points. Among runs with a DRA (about 1,640 per model and arm), the interval is about ±3.4 points per model. A loss of 4 points, as in E1, would be detected.

## 7. Exclusions, stops and budget

- **Exclusions and retries:** as in E2, section 7. An API failure is retried at the next start; a lost executor counts as wrong.
- **Stops:** only for time or cost, never because of results. If a stop is needed, only seeds that are complete for both models are analysed.
- **No looking:** nobody looks at accuracy, steps or tokens by arm until the runs end or stop.
- **Budget:** hard cap USD 10 per model (each process stops starting jobs at its cap). Expected: USD 5 to 12 in total.

## 8. Code

`live/e3_fork.py`, `live/e2_fork.py`, `live/e2_run.py`, `live/e2_runner.py`, `dabstep-gate/v9/gate.py` and the E1 modules that they import; `live/e2_report.py` and `live/e3_report.py` (analysis), `live/e2_validate.py` (block check). Hashes are in `e3_FREEZE.txt`.

## 9. Pilot that chose the second model (excluded)

Plan and rule: `runs/e3pilot/PLAN.md`, written before the pilot. Each model ran arm N only, on 40 questions drawn at random (seed 20261012), with seed 1. The table counts runs with at least one executed data-reference error.

| Model | Runs | Runs with a data-reference error | Errors run (file / column) | Answered | Cost (USD) |
|---|---|---|---|---|---|
| gpt-oss:120b (reference) | 40 | 20 (50.0%) | 33 / 2 | 90% | 0.060 |
| **gemma4:31b** | 40 | **5 (12.5%)** | 0 / 5 | 100% | 0.020 |
| nemotron-3-nano:30b | 40 | 3 (7.5%) | 1 / 2 | 85% | 0.040 |
| glm-5.3-flash | 38 of 40 | 2 (5.3%) | 0 / 2 | 82% | 0.265 |
| gpt-oss:20b | 6 of 40 | 0 | 0 / 0 | 100% | 0.001 |

- **Choice by the rule:** gemma4:31b.
- **glm-5.3-flash** was stopped after 38 runs, because its last 2 runs could no longer change the choice: at most 4 of 40 (10%).
- **gpt-oss:20b** could not complete the pilot. The service returned HTTP 500 errors and took about 80 s per reply, so a 10-seed run was not feasible. It was stopped after 6 runs, for that reason alone.
- **Why the two models fit together:** they make different errors. With this prompt, gpt-oss:120b mostly invents file paths; gemma4:31b, when it errs, names columns that do not exist. E3 therefore tests the checker in both cases.
