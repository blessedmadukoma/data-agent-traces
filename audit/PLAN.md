# Analysis plan: hand audit of the data-reference classification

Version 1, 7 October 2026. Written and frozen (FREEZE.txt) before any label was given.

## Purpose

Paper v2 counts data-reference errors with rules on error messages (Section 2.2): 4,429 of 26,265 failed DABstep cells (16.9%) and 4,751 of 27,984 failed DAB calls that the harness did not cause (17.0%). The audit measures how often these rules are right (precision) and how many data-reference errors they miss (recall).

## Sample

Script `tools/audit_dre_sample.py`, salt `dre-audit-2026-10-07`. Per corpus, three strata:

| Stratum | DABstep population | DAB population | Sample per corpus |
|---|---:|---:|---:|
| pos: classified as data-reference errors | 4,429 | 4,751 | 100 (simple random) |
| neg-sus: other failures whose error type can hide a missing name | 2,577 | 3,626 | 50 |
| neg-rest: all other failures | 19,259 | 19,607 | 50 |

neg-sus is DABstep `other`, `ValueError`, `AttributeError`, `IndexError`, `ParserError`, `UndefinedVariableError`, `IndexingError`, `MergeError`, `PermissionError`; DAB `py:other`, `py:ValueError`, `py:AttributeError`, `py:IndexError`, `sql:other`, `mongo:other`, `py:OSError`, `py:DatabaseError`, `py:OperationalError`. The DAB population excludes the 16,293 failed calls that the harness caused, as the paper's primary share does.

## Coding

- **Coder 1:** Blessed Madukoma, on a page that shows each item without its stratum or classifier label, in an order shuffled within each corpus. Codebook: CODEBOOK.md.
- **Coder 2:** an independent LLM instance (Claude), given the same blinded items and codebook, before coder 1's labels are seen by anyone. It is not a human coder, and the paper will say so.
- Agreement: Cohen's kappa on the four labels and on the binary label (1 against 2 or 3), before adjudication.
- Adjudication: coder 1 reads each disagreement again and gives the final label. We report how many labels changed. The results use the final labels.

## Estimates (per corpus, final labels)

- Precision = items labelled 1 in pos / items in pos labelled 1, 2 or 3. Unclear items are left out; we also give the value with unclear items counted as not 1.
- Missed errors FN = sum over the two negative strata of N_h × (items labelled 1 / items labelled 1, 2 or 3).
- TP = N_pos × precision. Recall = TP / (TP + FN).
- Corrected count = TP + FN; corrected share = corrected count / all failed steps (26,265 DABstep; 27,984 DAB).
- 95% intervals: bootstrap that resamples items within each stratum, 10,000 replicates, percentile intervals. Also Wilson intervals for precision.
- Label 2 (missing value) is reported on its own, because it marks the boundary of the paper's definition.

## Decision rule

- If precision and recall are both at least 0.90 in both corpora (point estimates), the paper keeps the rule-based counts and reports the audit beside them.
- Otherwise the paper reports the corrected counts with their intervals as the main figures, gives the rule-based counts as uncorrected, and changes every sentence that depends on them (abstract, contribution 2, Section 3.2 and its table of DAB models).
- The claim "about one failed step in six" stays only if the corrected share's interval includes values between 0.14 and 0.20 in both corpora; otherwise it is rewritten to the corrected values.
