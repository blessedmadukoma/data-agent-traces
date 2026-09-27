# Reproduce the preprint

This file lists the numbers in the preprint, the command that gives each one, and the line of output to look for. Part A and section B0 need no model calls. Sections B1 to B5 run models on Ollama Cloud and cost money.

All commands run from the repository root. Paths such as `runs/dabstep` are the defaults of the `Makefile`; change them on the command line if you want.

## Before you start

```bash
make setup          # installs Python 3.12 packages with uv
make test           # 127 checker tests, 20 DAB tests and the receipt-replay tests
make check-frozen   # every line must end with OK
```

# Part A: public traces (free)

## A1. DABstep

```bash
make dabstep-data   # 5.6 GB; ends with "OK: data/dabstep at c8fb51b3..., 2246 submission files"
make dabstep        # writes runs/dabstep/checker_report.txt
make dabstep-extra  # writes the other reports in runs/dabstep/
make dabstep-recount
```

`make dabstep-data` stops with an error if a file is still a Git LFS pointer or if `payments.csv` has the wrong SHA-256. If it stops, run it again. Set `HF_TOKEN` to a Hugging Face read token for a faster download.

| Preprint | Value | Report and line |
| --- | --- | --- |
| Table 1 | 74,484 cells, 48,219 working | `checker_report.txt`: `cells without an error: 48219` and `n=74484` |
| Table 1 | 26 submission files, 9 submitters | `silent_filter_report.txt`: `submissions: 26; submitters: 9` |
| Table 1 | 11,226 traces | distinct `(submission_file, trace_line)` pairs in `parsed_*_cells.jsonl` |
| Finding 1 | 4,429 of 26,265 failed cells (16.9%) | `checker_report.txt`: `failed cells: 26265` and `data-reference failures: 4429` |
| Finding 1 | 6.0% to 26.2% by submitter | `checker_report.txt`: column `data-fail` divided by column `failed` (ConfuseG 6 of 100, yiliu051016 1,940 of 7,418) |
| Finding 1 | 3,401 key and index errors: 58.5% in no file, 11.9% only in another file, 11.0% case mismatch | `recount_report.txt`: `missing-key classes (n=3401)`; "only in another file" is the sum of the five `exists-only-in` lines |
| Finding 1 | odds ratio 0.73 (0.49 to 1.08), 10,100 traces, 9 clusters | `outcome_report.txt`: `OR(correct ...): 0.73 [0.49, 1.08], n=10100, submitters=9` |
| Finding 2 | 2,122 of 4,429 stopped (47.9%), 4 of 48,219 working cells blocked | `checker_report.txt`: `caught with a cause match: 2122 (47.9%)` and `blocked 4` |
| Finding 2 | 2,459 recovery episodes; next cell fixes 36%; never recovered 6% | `recovery_report.txt`: row `ALL (pooled)` of the first table, columns `fix next` and `never ok` |
| Finding 2 | within two cells 61% | `recovery_report.txt`: `fix next` plus `fix 2` (36% + 25%) |
| Finding 2 | a perfect receipt would save 3.9% of all cells | `recovery_report.txt`: `ceiling: ... 2916 cells would be saved, 3.9% of all 74484 cells` |
| Finding 3 | 6 submissions without state; 80% to 96% of cells that use an earlier name fail; 0% to 5% in the others | `fresh_process.txt` (also the first lines of `checker_report.txt`) |
| Finding 3 | `NameError`: 8,153 cells, 31% of failures | `error_types.txt` |
| Finding 3 | 23 silently replaced keys | `checker_report.txt`: `replaced silently, in cells without an error: 23` |
| Finding 3 | 18 more false blocks without the tolerant profile | the command below: `cells without an error: 48219; blocked 22` (22 = 4 + 18) |
| What this means | 1.5% of traces have a filter on a value that is not in the column | `silent_filter_report.txt`: `traces with a silent impossible filter: 163 (1.5%)` |

Command for the check without the tolerant profile. It is the command of `make dabstep` without `--tolerant auto`, and without `--lint`, which does not change any block:

```bash
uv run dabstep-gate/evaluate_gate.py --gate-dir dabstep-gate --data data/dabstep \
  --manifest runs/dabstep/manifest.jsonl --cells runs/dabstep/parsed_*_cells.jsonl \
  --pred runs/dabstep/gate_predictions_strict.jsonl --carry --files
```

## A2. DataAgentBench

```bash
make dab-data       # 2.2 GB of logs, 4 MongoDB dumps and the bookreview re-run
make dab            # about 20 minutes; writes runs/dab/*.txt
```

The DAB logs have no licence file. The authors allowed us to publish statistics from them. Do not publish the logs or records derived from them without asking the authors.

| Preprint | Value | Report and line |
| --- | --- | --- |
| Table 1 | 19,130 runs, 198,606 calls, 154,329 working | `rq1_report.txt`: row `ALL` (154,329 = 198,606 calls minus 44,277 failed) |
| Table 2 | failed calls, harness-caused, share by model | `rq1_report.txt`: columns `failed`, `harness`, `primary` |
| Finding 1 | 4,751 of 27,984 (17.0%); 10.7% of all failed calls | `rq1_report.txt`: row `ALL`, columns `data`, `primary`, `second.` |
| Finding 1 | 3,532 SQL and 1,219 Python references | `rq1_report.txt`: columns `sql` and `py` |
| Finding 3 | 16,293 of 75,366 Python calls (21.6%) | `harness_report.txt` |
| Finding 3 | 4,976 calls with a changed string literal | `harness_report.txt`: last line |
| Finding 3 | paper-matched: 12,958 runs; 8,940 of 52,940 (16.9%); 31.3% of failed calls | `harness_report_paper_matched.txt` |
| Finding 3 | 5,418 of 6,306 unlimited MongoDB queries (85.9%) | `mongo_report.txt`: `"nolimit_checked": 6306`, `"nolimit_cut": 5418` |
| Introduction | harness errors outnumber program-data failures (16,293 against 4,751) | `rq1_report.txt`: row `ALL`, columns `harness` and `data` |

`dab_harness_count.py` compiles code with the Python that runs it. The published counts use Python 3.12, as in DAB's executor image, and the script warns if you use another version.

The MongoDB count stops after a time budget and keeps a cache, so `make dab` runs it until the report no longer says `INCOMPLETE`.

# Part B: runs with models

## B0. Score our recorded runs again (free)

`recorded-runs-2026-09.tar.gz` holds the outputs of our model runs (see `recorded/README.md` after you unpack it). With them you can check Tables 3 and 4, the baselines and the receipt replay without calling a model:

```bash
make bench-data     # InfiAgent-DABench, QRData and DiscoveryBench at the pinned commits, about 540 MB
make rescore        # unpacks the recorded runs, scores them, and prints a summary; about 2 minutes
```

The last lines of `make rescore` should read:

```
Table 3 (working cells, blocked; failures caught with the version of the row):
  ia_gpt-oss_v7: cells without an error: 2057; blocked 0 | caught with a cause match: 6 / 23
  qr_gpt-oss_v7: cells without an error: 3054; blocked 0 | caught with a cause match: 13 / 282
  ia_deepseek41_v7: cells without an error: 1517; blocked 0 | caught with a cause match: 8 / 9
  db_gpt-oss_v8: cells without an error: 1889; blocked 0 | caught with a cause match: 18 / 114
Table 3, live file check:
  ia_gpt-oss_v7_live: caught with a cause match: 12 / 23
  qr_gpt-oss_v7_live: caught with a cause match: 143 / 282
  ia_deepseek41_v7_live: caught with a cause match: 8 / 9
  db_gpt-oss_v8_live: caught with a cause match: 86 / 114
v6 on the v7 runs (blocks on working cells):
  ia_gpt-oss_v6: cells without an error: 2057; blocked 12
  qr_gpt-oss_v6: cells without an error: 3054; blocked 0
  ia_deepseek41_v6: cells without an error: 1517; blocked 8
...
Receipt replay:   all episodes: n = 907; mean saving 0.332 [0.257, 0.406]; ...
```

| Preprint | Value | File in `runs/rescore/` |
| --- | --- | --- |
| Table 3 | the four later rows | summary above; one report per run and version |
| Finding 2 | frozen v7 blocked none of 6,628 working cells; v6 would have made 20 false blocks | summary above (2,057 + 3,054 + 1,517 cells; 12 + 0 + 8 blocks) |
| Finding 2 | frozen v8 blocked none of 8,517 working cells | `*_v8.txt` of the four runs |
| Finding 2 | dry run: 425 of 428 caught (99.3%), 11 of 8,517 wrongly stopped | `*_dry_s100.json`: add `caught`, `data_failures` and `false_blocks` over the four runs |
| Finding 2 | LLM checker: 244 of 428 caught (57.0%), 134 wrongly stopped | `llm_checker.txt`: add the four lines |
| Table 4 | accuracy, cells, tokens and failures by arm; H1 saving 0.58 cells, p = 0.056 | `e1_gpt-oss.txt` |
| Finding 2 | 588 live blocks: 580 confirmed, 8 with another error first | `e1_gpt-oss.txt`: `block validation` |
| Finding 1 | deepseek-v4.1-flash: 7 failures in 5 of 411 runs without the checker; accuracy 56.0% in both arms | `e1_deepseek41.txt`: line `P0G0` (failures) and lines `P0G0`, `P0G1` (accuracy) |
| Finding 2 | receipt saves 0.33 cells (0.26 to 0.41); 0.17 without the largest submitter | `receipts_gpt-oss.txt`: `PRIMARY` and `(c)` |
| Finding 2 | the receipt arm more often ends with a constant answer (309 against 245 pairs) | `constant_answers_gpt-oss.txt` |
| Table 1 | cells and working cells of our runs: KramaBench 1,150 and 953; InfiAgent-DABench 3,648 and 3,574; QRData 3,434 and 3,054; DiscoveryBench 2,113 and 1,889 | `model cells` and `cells without an error` in the `*_v8.txt` reports (InfiAgent-DABench: add both models) and in `results/model-runs/kramabench_gpt-oss_v6.txt` |

The KramaBench run is not in the archive, because the KramaBench repository has no licence file. Its report is `results/model-runs/kramabench_gpt-oss_v6.txt` (953 working cells, 4 blocked, 8 of 47 failures caught).

The live file check for the recorded runs uses the Python folders that their sandbox mounted (`/home/claude/iaenv` and `/usr`). `tools/rescore.sh` passes them to `infiagent/mount_exists.py`. For your own runs, pass your own folders (see B2).

## B1. Setup for new runs (paid)

New runs execute code that a model writes. Run them only on Linux, inside the bubblewrap sandbox. The sandbox has no network, a private `/tmp`, read-only system and Python folders, and one writable work folder. The repository is not visible inside it.

1. Install bubblewrap: `sudo apt install bubblewrap` (Debian or Ubuntu). Check with `bwrap --version`.
2. Create the Python that runs inside the sandbox. The agent's code runs with this Python, so it needs the data packages:

   ```bash
   uv venv sandbox-env --python 3.12
   uv pip install --python sandbox-env/bin/python -r requirements/sandbox.txt
   ```

   The KramaBench run used a smaller set (`requirements/sandbox-kramabench.txt`). Create a second environment from it if you want the same failures for missing packages.
3. Create an Ollama Cloud API key (https://ollama.com/settings/keys). Copy `.env.example` to `.env` and put the key there. `.env` is in `.gitignore`. Never commit it.
4. Get the benchmark data: `make bench-data`, or `sh tools/get_benchmarks.sh --kramabench` to add KramaBench (1.6 GB more). We did not record the QRData and DiscoveryBench commits at run time. The pinned commits were the latest on the run dates (their commit dates are February 2025 and June 2025).

Every run script has `--budget-usd`. The script adds up the cost already recorded in `--out` and stops starting new tasks when the total reaches the budget. So when you add a second seed to the same `--out` file, raise the budget. Runs are resumable: finished tasks in `--out` are skipped. Prices per million tokens are in `rq3-pilot/rq3_run.py`, `PRICES` (Ollama Cloud, 23 and 24 September 2026). Settings of the runs in B2 to B4: temperature 1.0, 4,096 output tokens per call, up to 15 cells, 120 s per cell, and seed 1 unless a command says otherwise.

Models on Ollama Cloud can change. `deepseek-v4.1-flash` has no dated tag, and `deepseek-v4-flash:0731` was retired on 25 September 2026. Your model outputs will differ from ours. The checker results on a given recording are exact.

## B2. Held-out tests of the checker (Table 3)

Each run records every cell and the files in the work folder before each cell. The checker is not used during the run. Scoring replays the frozen checker on the recorded cells.

```bash
PY=sandbox-env/bin/python

# KramaBench, gpt-oss:120b, about USD 1.05 (use the KramaBench environment for exactness)
$PY kramabench/kb_run.py --kb data/Kramabench --out runs/kb/results_gpt-oss.jsonl \
  --transcripts runs/kb/transcripts --model gpt-oss:120b --workers 2 --budget-usd 5

# InfiAgent-DABench, gpt-oss:120b (about USD 0.78), then deepseek-v4.1-flash
IA=data/InfiAgent/examples/DA-Agent/data
$PY infiagent/ia_run.py --tables $IA/da-dev-tables --questions $IA/da-dev-questions.jsonl \
  --out runs/ia/results_gpt-oss.jsonl --transcripts runs/ia/transcripts_gpt-oss \
  --model gpt-oss:120b --workers 2 --budget-usd 2
$PY infiagent/ia_run.py --tables $IA/da-dev-tables --questions $IA/da-dev-questions.jsonl \
  --out runs/ia/results_deepseek41.jsonl --transcripts runs/ia/transcripts_deepseek41 \
  --model deepseek-v4.1-flash --workers 2 --budget-usd 1.8

# QRData, gpt-oss:120b, stop at USD 1.8
$PY qrdata/qr_run.py --tables data/QRData/benchmark/data --questions data/QRData/benchmark/QRData.json \
  --out runs/qr/results_gpt-oss.jsonl --transcripts runs/qr/transcripts_gpt-oss \
  --model gpt-oss:120b --workers 2 --budget-usd 1.8

# DiscoveryBench, gpt-oss:120b, about USD 1.19
$PY discoverybench/db_run.py --real data/discoverybench/discoverybench/real \
  --out runs/db/results_gpt-oss.jsonl --transcripts runs/db/transcripts_gpt-oss \
  --model gpt-oss:120b --workers 2 --budget-usd 1.5
```

Score each run with the version that was frozen before it: v6 for KramaBench, v7 for InfiAgent-DABench and QRData, v8 (`dabstep-gate`) for DiscoveryBench. The `--kb` folder is the task data for KramaBench, and the folder that the run script created next to `--out` for the others (`ia_kb`, `qr_kb`, `db_kb`).

```bash
uv run kramabench/kb_gate.py --gate-dir dabstep-gate/v6 --kb data/Kramabench \
  --results runs/kb/results_gpt-oss.jsonl --out runs/kb/gate_v6.jsonl
uv run kramabench/kb_gate.py --gate-dir dabstep-gate/v7 --kb runs/ia/ia_kb \
  --results runs/ia/results_gpt-oss.jsonl --out runs/ia/gate_v7_gpt-oss.jsonl
```

For the live file check, use `infiagent/ia_sandbox_exists.py` (v7 rows) with the same arguments as `kb_gate.py`, or `infiagent/mount_exists.py` (v8 row) with the Python folders that your sandbox mounted:

```bash
ROOTS=$($PY -c "import sys; sys.path.insert(0, 'rq3-pilot'); import sandbox; print(' '.join(sandbox._python_roots()))")
uv run infiagent/mount_exists.py --roots $ROOTS -- --gate-dir dabstep-gate --kb runs/db/db_kb \
  --results runs/db/results_gpt-oss.jsonl --out runs/db/gate_v8_live.jsonl
```

`tools/rescore.sh` shows the full set of scoring commands.

## B3. Baselines (Finding 2)

The baselines use the recorded cells of the four later runs (InfiAgent-DABench with both models, QRData, DiscoveryBench).

```bash
# The checker v8 predictions that the dry-run score compares with
uv run kramabench/kb_gate.py --gate-dir dabstep-gate --kb runs/qr/qr_kb \
  --results runs/qr/results_gpt-oss.jsonl --out runs/qr/gate_v8_gpt-oss.jsonl

# Dry run on 101-line copies of the data (no model calls)
$PY baselines/dry_run.py --results runs/qr/results_gpt-oss.jsonl --lake data/QRData/benchmark/data \
  --variant s100 --out runs/dry/qr_gpt-oss_s100.jsonl --workers 2
uv run baselines/dry_run_score.py --dry runs/dry/qr_gpt-oss_s100.jsonl --gate runs/qr/gate_v8_gpt-oss.jsonl

# LLM checker, about USD 2.22 for all four runs
uv run baselines/llm_checker.py --results runs/qr/results_gpt-oss.jsonl --tables data/QRData/benchmark/data \
  --out runs/llm_check/qr_gpt-oss.jsonl --model gpt-oss:120b --workers 2 --budget-usd 1.8
uv run baselines/llm_checker_score.py runs/llm_check/*.jsonl
```

For InfiAgent-DABench use `--lake $IA/da-dev-tables` (dry run) or `--tables $IA/da-dev-tables` (LLM checker). For DiscoveryBench use `--kb runs/db/db_kb` in both. The dry run is not frozen (see `frozen/README.md`).

## B4. Live experiment on QRData (Table 4)

411 questions, 4 arms: `P0` plain prompt or `P1` with column names, crossed with `G0` no checker or `G1` checker v8 in the loop. gpt-oss:120b ran all 4 arms with seeds 1 and 2 (3,288 runs, USD 10.51). deepseek-v4.1-flash ran `P0G0` and `P0G1` with seed 1 (822 runs; USD 9.73 with 28 partial runs in the other arms, which we stopped when the cost was three times the estimate).

```bash
Q=data/QRData/benchmark/QRData.json; T=data/QRData/benchmark/data
$PY live/e1_run.py --tables $T --questions $Q --out runs/e1/results.jsonl --transcripts runs/e1/transcripts \
  --model gpt-oss:120b --seed 1 --workers 3 --budget-usd 6.5
$PY live/e1_run.py --tables $T --questions $Q --out runs/e1/results.jsonl --transcripts runs/e1/transcripts \
  --model gpt-oss:120b --seed 2 --workers 3 --budget-usd 11
$PY live/e1_validate.py --results runs/e1/results.jsonl --tables $T --out runs/e1/validation.jsonl
uv run live/e1_report.py --results runs/e1/results.jsonl --questions $Q --validation runs/e1/validation.jsonl

# deepseek-v4.1-flash, two arms
$PY live/e1_run.py --tables $T --questions $Q --out runs/e1/results_deepseek41.jsonl \
  --transcripts runs/e1/transcripts_deepseek41 --model deepseek-v4.1-flash --seed 1 \
  --arms P0G0,P0G1 --workers 3 --budget-usd 10
$PY live/e1_validate.py --results runs/e1/results_deepseek41.jsonl --tables $T --out runs/e1/validation_deepseek41.jsonl
uv run live/e1_report.py --results runs/e1/results_deepseek41.jsonl --questions $Q \
  --validation runs/e1/validation_deepseek41.jsonl
```

The preprint's estimate of run-to-run noise (5 to 9 points of accuracy at temperature 1.0) comes from comparing arms on the questions where the checker never blocked (9 points in seed 1). No script in this release prints it.

## B5. Receipt replay on recorded DABstep traces (Finding 2)

The replay cuts a recorded DABstep trace at the first cell that checker v6 blocks, rebuilds the state in the sandbox, and lets a model continue twice: once with the traceback and once with the receipt. It needs Part A1 first. It uses the repository's own Python environment inside the sandbox.

```bash
make receipts-select          # the 1,000 episodes of the study; no model calls
make receipts-cost-check      # the first 10 episodes, one seed, about USD 0.07; repeat until "0 episode-arms left"
make receipts-report
```

`make receipts-select` must print the SHA-256 `bcdb3e6bdba9f7b1af55625212b21b822c1c1393b411ff92df95e7881bc8bb2e`, the hash of the episode file of our study. It uses checker v6 with the current DABstep adapter, and it leaves out the 210 (submission, task) pairs of our pilot and cost check (`rq3-pilot/excluded_pairs.jsonl`). It asks for up to 5,000 episodes, so it takes all 1,000 candidates that are left.

The full study used all 1,000 episodes, 2 seeds and gpt-oss:120b (USD 12.86):

```bash
uv run rq3-pilot/rq3_run.py --episodes runs/receipts/episodes.jsonl --data data/dabstep \
  --out runs/receipts/results_gpt-oss.jsonl --transcripts runs/receipts/transcripts_gpt-oss \
  --model gpt-oss:120b --seeds 2 --seed-major --workers 3 --budget-usd 16
uv run rq3-pilot/rq3_full_report.py --episodes runs/receipts/episodes.jsonl \
  --results runs/receipts/results_gpt-oss.jsonl
uv run rq3-pilot/rq3_mixed.py --episodes runs/receipts/episodes.jsonl \
  --results runs/receipts/results_gpt-oss.jsonl
uv run rq3-pilot/constant_answers.py runs/receipts/results_gpt-oss.jsonl
```

The two deepseek runs used the first 250 episodes (`recorded/receipts/episodes_deepseek.jsonl`) with one seed. Their reports are in `results/model-runs/receipts_deepseek*.txt`.

## What you cannot reproduce exactly

- Model outputs. Models on Ollama Cloud can change, and sampling at temperature 1.0 is random even with a fixed seed.
- The hand audits of blocks on working cells. We read each block against the raw trace. The reports list every block, so you can repeat the audit.
