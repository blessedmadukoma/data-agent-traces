# Answers lost by our harness

Our harness defines `final_answer` in the same namespace as the model's variables. When the model defines its own `final_answer`, for example `def final_answer(a): print(a)`, or assigns a value to that name, the harness records no answer, and the run usually goes on to the 15-cell limit. The preprint describes this in Section 3.4.

`live/fa_replay.py` replayed the recorded cells of every run without a recorded answer, and of every run whose code defines or assigns `final_answer`, with `live/fa_runner.py`, which keeps `final_answer` safe. The replay stops at the first recorded answer. `live/fa_rescore.py` then ends each such run at that answer. It keeps the recording when an earlier cell had another outcome in the replay.

| File | What it holds |
| --- | --- |
| `<tag>.jsonl` | one replay per run: `key` (`<tag>:<line in the results file>`), the outcome of each replayed cell (`cell` is its index in the recorded run), `answer` and `answer_cell` |
| `summary.txt` | output of `live/fa_rescore.py`: runs with a model-defined `final_answer`, answers recovered, replay fidelity |
| `rescore_summary.txt` | `tools/rescore.sh` on the recovered runs: held-out tests and receipt replay |
| `e1_gpt-oss_recovered.txt`, `e1_deepseek41_recovered.txt` | `live/e1_report.py` on the recovered live experiment |
| `e1_noise_recorded.txt`, `e1_noise_recovered.txt` | `live/e1_noise.py`: accuracy by seed, and the arms compared on questions where the checker never blocked |
| `kramabench_gpt-oss_v6_recovered.txt` | frozen release 1 on the recovered KramaBench run |

Tags: `live_gpt` and `live_ds` (live QRData experiment, gpt-oss:120b and deepseek-v4.1-flash), `ia_gpt` and `ia_ds` (InfiAgent-DABench), `qr_gpt` (QRData), `db_gpt` (DiscoveryBench). The KramaBench replay is not included, for the same reason as the KramaBench run.

We made these replays on 27 September 2026 with `live/fa_runner.py` and a driver that ran it in a Linux namespace without network access, with the data mounted read-only at `/work/input`, instead of bubblewrap. Like `live/fa_replay.py`, the driver did not run again the cells that timed out in the recording. `live/fa_replay.py` does the same replay in the bubblewrap sandbox of the original runs; in a test on the first five replays of the live experiment (with `--sandbox local`), it gave the same answers and cell outcomes.
