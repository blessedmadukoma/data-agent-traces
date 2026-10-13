# Extension of the controlled DAB rerun (blocks 5 to 9)

Five more runs per task of arms A (DAB's harness as released) and B (the model's code passed to `exec()`
unchanged), with gpt-oss:120b, pre-registered on 12 October 2026. The primary analysis pools these runs with
blocks 0 to 4 of the first rerun (`../dab-rerun/`).

| File | What it is |
|---|---|
| `preregistration.md`, `FREEZE.txt` | The plan and the SHA-256 hashes frozen before the first run. |
| `AMENDMENTS.md` | One operational change (five workers instead of three). |
| `rerun.py` | The runner (blocks 5 to 9, arms A and B). |
| `judge_ext.py` | DAB's failure-analysis judge (the frozen `../dab-rerun/judge.py`, unchanged) on the failed runs. |
| `analysis_ext.py` | The frozen analysis of the first rerun on blocks 5 to 9, and the pooled analysis of blocks 0 to 9. |
| `judge_check_items.py` | Draws the 40 failed runs for the blind human check of the judge (the page itself holds DAB data and stays private). |
| `runs.jsonl` | One line per run attempt: arm, dataset, query, block, attempt, correctness, end reason, model calls, tool calls, failed tool calls, tokens, cost, wall time. No answers, code or data. |
| `judge_labels.jsonl` | The judge's FM1 to FM4 labels per failed run (no judge text, which quotes the traces). |
| `analysis.json`, `analysis_pooled.json` | The results that preprint v3.20 reports. |

Results (pooled, 1,073 of 1,080 runs): Pass@1 31.4% (A) and 30.4% (B), difference −1.0 points
(95% CI −5.7 to +3.4, Wilcoxon p = 0.38); failed calls per run −1.51 (−2.84 to −0.53, p = 0.004).
In blocks 5 to 9, the embedding broke 272 of 1,298 Python calls (21.0%) in 23 of 269 runs.

Reproduce from the records: `python3 dab-rerun-ext/analysis_ext.py` needs the run folders for H1, H3 and H4
(not released, because they contain DAB's data); the pooled H2 and H3 use only `runs.jsonl` of both reruns.

## H4b: human check of DAB's judge (October 2026)

A person who did not run the experiments labelled the 40 failed runs drawn by `judge_check_items.py` (20 per harness)
with DAB's failure modes, blind to the harness and to the judge's labels, on a private page.
`judge_check_human_labels.json` holds his modes and his "unsure" flag per item; his written summaries quote DAB data and
stay private. `judge_check_analysis.py` compares them with the judge's labels in the sealed key (not released, because
the judge's text quotes DAB data) and writes `judge_check_results.json`.

| Mode | Agreement | Cohen's kappa (95% CI) | Judge yes | Human yes |
|---|---|---|---|---|
| FM1 | 40/40 | n/a (neither used it) | 0 | 0 |
| FM2 | 30/40 | 0.45 (0.14 to 0.72) | 14 | 14 |
| FM3 | 27/40 | 0.27 (-0.02 to 0.55) | 16 | 9 |
| FM4 | 29/40 | 0.41 (0.12 to 0.67) | 10 | 17 |

Per harness, FM4 kappa is 0.40 (A) and 0.39 (B). He marked 5 of the 40 runs as unsure.

