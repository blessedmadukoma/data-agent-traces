# Results

Reports printed by the commands in [docs/REPRODUCE.md](../docs/REPRODUCE.md), on our machines. Compare your own output with these files. Absolute paths in the reports were shortened to `runs/...`.

| Folder | Made by | Covers |
| --- | --- | --- |
| `dabstep/` | `make dabstep`, `make dabstep-extra`, `make dabstep-recount` | DABstep prevalence, checker, recovery, silent filters, error types, outcome regression, independent recount |
| `dab/` | `make dab` | DataAgentBench prevalence by model, checker, harness counts, MongoDB limit |
| `model-runs/` | `make rescore`; the KramaBench report and the two deepseek receipt reports come from the original runs | held-out tests (`<run>_<version>.txt`, live file check `<run>_<version>_live.txt`), baselines (`*_dry_s100.json`, `llm_checker.txt`), live QRData experiment (`e1_*.txt`), receipt replay (`receipts_*.txt`) |

`dabstep/checker_report_no_tolerant_profile.txt` is the checker run without the smolagents profile for tolerant key lookup. It shows the 18 extra blocks on working cells that the profile removes (22 instead of 4).
