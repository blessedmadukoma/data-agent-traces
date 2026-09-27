# DABstep traces and the checker

The checker and the scripts for the DABstep submissions at dataset commit `c8fb51b3`. Run the pipeline with `make dabstep` and `make dabstep-extra` from the repository root.

## The checker

| File | What it does |
| --- | --- |
| `gate.py` | The checker, version 8. `decide(code, schema, state, exists)` returns `run`, `unknown` or `block`, and a receipt for a block. It never runs the code. |
| `gate_adapter.py` | Schemas of the DABstep context files, the evidence model for file existence in a replay, and the executor profiles (fresh process, tolerant key lookup). |
| `v6/gate.py`, `v7/gate.py` | Earlier frozen versions, used for the held-out tests. |
| `test_gate.py` | 127 unit tests: `uv run python -m unittest test_gate.py` in this folder. |

## Pipeline

| Step | Script | Output in `runs/dabstep/` |
| --- | --- | --- |
| List every submission file and decide which to use | `build_manifest.py` | `manifest.jsonl` |
| Parse the three trace formats into cells | `extract_xml_trace_cells.py`, `extract_smolagents_trace_cells.py`, `extract_generic_trace_cells.py` | `parsed_*_cells.jsonl` |
| Run the checker on every cell | `evaluate_gate.py` | `gate_predictions.jsonl`, `checker_report.txt` |
| Recovery after a data-reference error | `recovery_episodes.py` | `recovery_report.txt` |
| Filters on values that are not in the column | `silent_filter_check.py` | `silent_filter_report.txt` |
| Syntax errors in code that compiles | `dabstep_syntax_check.py` | `syntax_check_report.txt` |
| Recovery by error group and the outcome regression | `recovery_from_traces.py` | `recovery.csv`, `outcome_report.txt` |
| Independent heuristic recount of the corpus | `dabstep_corpus_recount.py` | `recount_report.txt` |

## Audit and analysis helpers

| Script | Use |
| --- | --- |
| `sample_false_block_audit.py` | Draw a sample of blocks on working cells for a hand audit, and score the filled sheet. |
| `sample_audit_cells.py`, `show_cell.py`, `labels_report.py` | Draw a labelled sample of failed cells, show one cell with its context, and report the labels. |
| `miss_analysis.py` | Classify the failures that the checker misses. |
| `compare_preds.py` | Compare two checker runs cell by cell before a change is kept. |

We used only the public per-task scores. Do not reconstruct hidden answers from the submissions.
