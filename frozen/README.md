# Frozen files

Each `.sha256` file lists the files of one frozen checker version with their SHA-256 hashes. We recorded the hashes before the workload of that version was opened. Check them with:

```bash
make check-frozen
```

| File | Version | In the preprint | Workloads it was tested on unchanged |
| --- | --- | --- | --- |
| `v6.sha256` | v6 (`dabstep-gate/v6/gate.py`) | frozen release 1 | KramaBench, receipt replay on DABstep |
| `v7.sha256` | v7 (`dabstep-gate/v7/gate.py`, scored with `kramabench/kb_gate.py`) | frozen release 2 | InfiAgent-DABench with gpt-oss:120b and deepseek-v4.1-flash, QRData |
| `v8.sha256` | v8 (`dabstep-gate/gate.py` and the scripts of the live experiment and the LLM baseline) | frozen release 3 | DiscoveryBench, live QRData experiment, baselines |

## Two differences from the freeze records

1. **v7 adapter.** The v7 freeze also listed `dabstep-gate/gate_adapter.py` with hash `aa7ffe50534656fb78839895346f33430a97e9c02e4e99b251679fb5c93485c3`. The adapter holds the DABstep evidence model and is not used on the held-out workloads, because `kramabench/kb_gate.py` loads only `gate.py` from the version folder. This release has only the v8 adapter.
2. **Dry-run baseline.** The v8 freeze listed `baselines/dry_run.py` with hash `d9068699acc52fb41dccd08c0c36341e562c7d22853f3447bb0a108417fa8bc8`. The file changed after the freeze, and we did not keep the frozen copy. For this release we also changed its docstring and the default of `--lakes-root` from an absolute path on our machine to `dry_lakes`. The dry run is a baseline, not part of the checker, so this does not affect any checker result. It can affect the dry-run figures (99.3% detected, 11 false blocks), so treat those as not frozen.

## Paths in frozen files

We did not edit frozen files, so some of their docstrings show paths from our machine, for example `/home/claude/iaenv/bin/python` or `/home/claude/qrdata_tables/data`. Use your own paths instead. [docs/REPRODUCE.md](../docs/REPRODUCE.md) gives the commands with neutral paths.
