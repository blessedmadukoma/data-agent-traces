# Release 4 of the checker (version 9)

Version 9 corrects gaps in the model of Python and pandas of version 8 (frozen release 3, `dabstep-gate/gate.py`). A reader sent three programs that run without an error and that version 8 blocks; a differential test against pandas 1.5.3, 2.2.3 and 3.0.5 found 78 more. Version 8 is unchanged. Version 9 was written after all evaluations, so it was not frozen before a benchmark: `frozen/v9.sha256` records its files but is not a freeze. The docstring of `gate.py` lists every change (C1 to C5, N1 to N4, A1 to A3, F1 to F4, R1, R2, O1, O2). Every change makes the checker block less, never more.

| File | What it does |
| --- | --- |
| `gate.py` | The checker, version 9. Same `decide()` arguments and result as version 8. |
| `test_gate_v9.py` | 135 tests: the 127 tests of version 8 (two replaced on purpose), the 81 programs that version 8 wrongly blocks, programs that must still be blocked, 400 generated programs and state tests. Run `python3 -m unittest test_gate_v9.py` in this folder. |
| `differential_test.py` | Generates programs over a small data set, runs each in a fresh process with one or more Python interpreters (one per pandas version), and counts the steps that a checker blocks although they ran without an error. |
| `compare_releases.py` | Compares two releases step by step on the prediction files of the evaluation scripts. |
| `live_replay.py` | Decides every step of the live QRData experiment again with both releases. |

Run everything with `make release4` from the repository root (needs `make dabstep`, `make dab` and `make rescore` first). The differential test:

```bash
cd dabstep-gate/v9
python3 differential_test.py --n 10000 --seed 7 --python python3 \
  --python /path/to/pandas-2.2.3/bin/python --python /path/to/pandas-1.5.3/bin/python \
  --gate-dir .. --gate-dir . --out differential_seed7.json
```

Results are in `results/release4/`.
