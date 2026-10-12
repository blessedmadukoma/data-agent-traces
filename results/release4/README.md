# Release 4 of the checker (version 9): results

Made by `make release4` (`tools/release4.sh`) and the differential test, on our machine. Version 8 (frozen release 3) is unchanged; release 4 is a correction, not a frozen release (see `frozen/README.md`).

| File | What it shows | Preprint v3.19 |
| --- | --- | --- |
| `compare.txt`, `compare.json` | Release 3 against release 4 on every recorded step: 159,045 steps (DABstep 74,484; DAB Python calls 75,366; our four later evaluations 9,195), also with the live file check. 228 decisions change; release 4 adds no block; none of the 18 blocks it removes was on a working step. | Section 3.3, "Gaps that the evaluations did not expose" |
| `dabstep_v9_report.txt` | DABstep with release 4: 2,121 cause-matched blocks (release 3: 2,122), 4 blocks of working steps (the same 4 real failures), 22 replaced keys listed (release 3: 23). | Section 3.3 |
| `dab_v9_report.txt` | DAB with release 4: 119 Python references to columns or keys stopped (release 3: 124), 379 missing files (unchanged), no block of 38,350 working calls. | Section 3.3 |
| `*_v9.txt`, `*_v9_live.txt` | Our four later evaluations with release 4, from the trace evidence and with the live file check: the same decisions as release 3 on every step. | Section 3.3 |
| `live_replay.txt` | Every step of the live experiment's runs with the checker (12,637 steps), decided again: release 4 makes the same decision as release 3 on every step, including the 588 blocks. | Section 3.3 |
| `differential_seed7.txt`, `.json` | 10,000 generated programs (12,315 steps), seed 7, run in pandas 3.0.5, 2.2.3 and 1.5.3. Release 3 wrongly blocks 3,029, 3,130 and 3,193 steps that ran without an error (and crashes on 55); release 4 blocks none. The generator was written to exercise the gaps, so these rates do not estimate how often agents write such code. | Section 3.3 |

The 81 programs in `dabstep-gate/v9/test_gate_v9.py` (`RUN_CASES` and `VERSION_CASES`: 3 from a reader and 78 that the differential test found) all run without an error in at least one of the three pandas versions; release 3 blocks all 81 and release 4 none.

Timing (same machine, idle, the same 20,000 DABstep steps): median 0.6 ms per step for release 3 and 1.1 ms for release 4.
