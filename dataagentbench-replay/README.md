# DataAgentBench replay

Scripts for the baseline logs of DataAgentBench (DAB) [1]: branch `refs/pull/6/head` at commit `0a1a9c0c`, 6 models, 19,130 runs. Run them with `make dab-data` and `make dab` from the repository root.

| Step | Script | Output in `runs/dab/` |
| --- | --- | --- |
| One record per run, every call classified | `extract_dab_cells.py` | `cells_<model>.jsonl` |
| Program-data failures and harness-caused failures by model | `dab_rq1.py` | `rq1.json`, `rq1_report.txt` |
| Checker on every Python call | `dab_gate_eval.py`, `dab_gate_summary.py` | `gate_<model>.jsonl`, `checker_report.txt` |
| Syntax errors caused by the code wrapper, all logs and paper-matched | `dab_harness_count.py` | `harness_report.txt`, `harness_report_paper_matched.txt` |
| MongoDB queries cut by the default limit of 5 | `dab_mongo.py` | `mongo.json`, `mongo_report.txt` |

`dab_lib.py` models the harness at DAB commit `0290945`: stored results, the code wrapper in `ExecTool._exec`, a new process for each Python call, and the MongoDB default limit. `test_dab.py` holds the unit tests.

The checker runs with `alias=True` here, so `x = var_x` keeps both names tracked. This option is off for DABstep; the DABstep results are the same with it on.

The DAB repository has no licence file. The authors allowed us to publish statistics from the logs. Do not publish the logs or records derived from them without asking the authors.

[1] R. Ma *et al.*, "Can AI agents answer your data questions? A benchmark for data agents," in *Proc. EMNLP*, 2026, arXiv:2603.20576. https://github.com/ucbepic/DataAgentBench
