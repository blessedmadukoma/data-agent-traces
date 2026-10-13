# Hand audit of the data-reference classification

This folder holds the audit in Section 2.2 of the preprint: 400 failed steps, 200 from DABstep and 200 from DataAgentBench (DAB), sampled in three strata per corpus and labelled by two coders.

| File | What it is |
|---|---|
| `PLAN.md` | The analysis plan, frozen before any label was given. |
| `CODEBOOK.md` | The codebook that both coders used, frozen at the same time. |
| `FREEZE.txt` | SHA-256 hashes of the frozen files, with the time of the freeze. |
| `sample_ids.csv` | The 400 sampled steps: item id, corpus, stratum, rule group and step identifier. |
| `strata.json` | The size of each stratum and the number of failed steps per corpus. |
| `labels_coder1_final.json` | Coder 1 (Blessed Madukoma), after adjudication. `raw_label` gives the label before adjudication where it changed (28 items). |
| `labels_coder2.jsonl` | Coder 2, an LLM (Claude), labelled blind to the strata and to coder 1. |
| `results_FINAL.json` | Precision, recall, corrected counts, 95% bootstrap intervals and agreement, per corpus. |

## Check the files against the freeze

```bash
cd audit && sha256sum CODEBOOK.md PLAN.md strata.json ../tools/audit_dre_sample.py
```

Each hash must equal the line for that file in `FREEZE.txt`. `labels_coder2.jsonl` was sealed before coder 1 finished; its SHA-256 begins with `94a317ece2c28d07`.

We do not publish `sample_full.jsonl`, `sample_blind.json`, `schemas.json` or `dab_queries.json`, which `FREEZE.txt` also lists. They contain the code, errors and questions of the sampled steps, which come from the DAB logs and the benchmark data. The DAB authors allowed us to publish statistics and short quotes from their logs, not the logs.

## Reproduce the results

From the labels (no data needed):

```bash
uv run python tools/audit_dre_analysis.py --sample audit/sample_ids.csv --strata audit/strata.json \
    --coder1 audit/labels_coder1_final.json --coder2 audit/labels_coder2.jsonl --json /tmp/results.json
```

The output equals `results_FINAL.json`, including the bootstrap intervals (seed 20261007) and the agreement before adjudication.

From the data: after `make dabstep` and `make dab` (Part A of `docs/REPRODUCE.md`), link or copy the data so that `data/dabstep`, `runs/dabstep` and `runs/dab` exist under one root, and run:

```bash
uv run python tools/audit_dre_sample.py --root . --out runs/audit_dre
```

`runs/audit_dre/strata.json` then has the hash in `FREEZE.txt`, and the item ids, strata and step identifiers equal `sample_ids.csv`. `sample_full.jsonl` differs from the frozen file unless `runs/audit_dre/dab_queries.json` (the DAB question texts) is present, because the script copies each question into the item.

## Codebook clarification made during adjudication

**C1.** A file that the model invented to hold an earlier tool result (for example `/tmp/tmp5u4d7u3j.json` or `rl44`) is a misuse of the tool, label 3, not a missing data file. The codebook named the harness's own result variables and `file_storage/` under label 3, but not such invented files. This clarification changed 20 DAB labels from 1 to 3 and lowered DAB's precision from 0.90 to 0.72. The preprint reports it as a decision made during adjudication.

## Step identifiers

- DABstep: `submission file|trace line|cell index`, as in `runs/dabstep/parsed_*_cells.jsonl`.
- DAB: `model/dataset/query/run|call index`, as in `runs/dab/cells_*.jsonl`.

## Second human coder (October 2026)

A second person, who took no other part in the study, labelled all 400 items with the frozen codebook and clarification
C1, blind to the strata, the rules' decisions, coder 1 and the LLM coder. The items were shown in the order of
`human2/order.json` (`human2/make_order.py`); `human2/part1_composition_SEALED.json` says which part-1 items were the
random sample and which were the earlier disagreements, and was kept out of the repository until the labels were in.
He coded from the files of the private page and entered his answers afterwards, because the page did not record his
first session; his final set is `human2/labels_human2.json` (prepared 13 October 2026, 10:51 UTC). Two earlier entries
of his answers are kept privately; they agree with the final set on 384 of 400 and 87 of 87 items.

| | DABstep | DAB |
|---|---|---|
| Cohen's kappa (four labels) with coder 1's final labels, all 200 items | 0.970 | 0.886 |
| the same, random part-1 items only (n = 100) | 0.960 | 0.875 |
| with coder 1 before adjudication (all 200) | 1.000 | 0.684 |
| with the LLM coder (all 200) | 0.950 | 0.928 |
| Disagreements with coder 1's final labels | 3 | 11 |
| Precision from his labels (95% bootstrap CI) | 0.899 (0.838 to 0.950) | 0.810 (0.730 to 0.890) |
| Recall from his labels | 0.962 | 0.963 |
| Corrected share of all failed steps | 15.8% (14.6 to 16.9) | 14.3% (12.8 to 15.8) |

Twelve of the 14 disagreements are keys or indexes of objects that the program built from the data (DAB `py:data-key`,
DABstep wrapped index errors), which he labelled 1 and coder 1 labelled 3; the other two he labelled unclear. DAB's low
kappa with coder 1 before adjudication is expected: those labels predate clarification C1, which he applied. The
paper keeps coder 1's final labels for its estimates and reports his as a check. Reproduce with:

```bash
python3 tools/audit_human2_analysis.py --human2 audit/human2/labels_human2.json --json audit/human2/results_human2.json
```
