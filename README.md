# data-agent-traces

Code for the preprint *What Agent Traces Hide: Data-Reference Errors and Harness Effects in LLM Data Agents* (B. Madukoma, 2026).

A data agent answers a question by writing Python or SQL, running it and reading the output. The code names columns, keys and files. This repository measures what happens when a name is wrong. It has three parts:

1. Parsers that turn public agent traces from DABstep and DataAgentBench (DAB) into one record per code cell.
2. A static checker (called the gate in the code) that reads a cell before it runs and blocks it only when it can prove that a named column, key or file does not exist.
3. The scripts of every experiment in the preprint, including the runs with models on Ollama Cloud.

The repository contains no benchmark data. The commands below download each dataset at a pinned commit.

## Main results and how to check them

| Result in the preprint | Command | Cost |
| --- | --- | --- |
| DABstep: 16.9% of failed cells (4,429 of 26,265) are data-reference errors | `make dabstep` | free, about 10 minutes |
| DABstep: the checker stops 2,122 of them (47.9%) and blocks 4 of 48,219 working cells | `make dabstep` | free |
| DABstep: 2,459 recovery episodes, next cell fixes 36%, never recovered 6% | `make dabstep-extra` | free |
| DAB: 17.0% of failed calls (after harness errors) are data-reference errors | `make dab` | free, about 20 minutes |
| DAB: the harness turns 16,293 of 75,366 Python calls (21.6%) into syntax errors | `make dab` | free |
| DAB, paper-matched corpus: 8,940 of 52,940 Python calls (16.9%) | `make dab` | free |
| DAB: 5,418 of 6,306 unlimited MongoDB queries (85.9%) are cut to 5 documents | `make dab` | free |
| Held-out tests, baselines, live QRData experiment and receipt replay, from our recorded runs | `make bench-data rescore` | free, about 3 minutes |
| Our harness lost the answer in 27% of the live-experiment runs; the same results with those answers recovered | `make bench-data fa-rescore` | free, about 3 minutes |
| The same experiments with new model runs | see [docs/REPRODUCE.md](docs/REPRODUCE.md), B1 to B5 | USD 1 to 13 each |

[docs/REPRODUCE.md](docs/REPRODUCE.md) lists every number in the preprint with its command and the expected output. [results/](results/) holds the reports that these commands printed on our machines.

## Quick start

You need Linux or macOS, Python 3.12, [uv](https://docs.astral.sh/uv/), git, `make`, `unzip` and about 10 GB of free disk space.

```bash
git clone https://github.com/blessedmadukoma/data-agent-traces.git
cd data-agent-traces
make setup          # install the Python environment
make test           # unit tests, about 1 minute
make check-frozen   # check the SHA-256 hashes of the frozen checker files
make dabstep-data   # download DABstep at commit c8fb51b3 (5.6 GB)
make dabstep        # parse the traces and run the checker on every cell
make dabstep-extra  # recovery, silent filters, syntax errors, outcome regression
make dab-data       # download the DAB logs and MongoDB dumps (2.2 GB)
make dab            # the DAB replay
make bench-data     # InfiAgent-DABench, QRData and DiscoveryBench (540 MB)
make rescore        # score our recorded model runs again, without model calls
make fa-rescore     # the same, with the answers that our harness lost recovered
```

Each target prints its report and also writes it to `runs/`. Run `make help` for the full list. You can change any path on the command line, for example `make dabstep DATA=/data/dabstep`.

The experiments with models need Linux with [bubblewrap](https://github.com/containers/bubblewrap), because the code that the models write runs in a sandbox without network access. They also need an Ollama Cloud API key. See [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Layout

| Folder | What it holds |
| --- | --- |
| `dabstep-gate/` | DABstep manifest and trace parsers, the checker (`gate.py`, `gate_adapter.py`), its evaluation and the recovery analyses. `v6/` and `v7/` hold the earlier frozen checker versions. |
| `dataagentbench-replay/` | DAB log parser, prevalence, checker evaluation, harness count and MongoDB count. |
| `kramabench/`, `infiagent/`, `qrdata/`, `discoverybench/` | Runs of our own agent on these benchmarks, and the scoring of the checker on the recorded cells. |
| `live/` | The live 2 x 2 experiment on QRData (column names in the prompt, checker on or off), and the replay that recovers the answers our harness lost. |
| `baselines/` | Two alternative checkers: a dry run on truncated data and an LLM checker. |
| `rq3-pilot/` | Replay of recorded DABstep episodes with a model: receipt against traceback. |
| `tools/` | Download and small helper scripts. |
| `frozen/` | SHA-256 hashes of each frozen checker version. |
| `results/` | Reports printed by the commands. |
| `recorded-runs-2026-09.tar.gz` | Outputs of our model runs, 23 and 24 September 2026 (10 MB). `make rescore` unpacks it into `recorded/`. |

## Frozen checker versions

We changed the checker only between workloads. Before a new workload was opened, we froze the checker by SHA-256 and then ran it unchanged on that workload. Versions 1 to 5 were development versions, built and tested only on DABstep and DAB. Three versions were frozen, and the preprint calls them frozen releases 1, 2 and 3:

| Code version | In the preprint | Frozen on | Used for | Hash file |
| --- | --- | --- | --- | --- |
| v6 | frozen release 1 | 23 September 2026 | KramaBench, receipt replay | `frozen/v6.sha256` |
| v7 | frozen release 2 | 24 September 2026 | InfiAgent-DABench (two models), QRData | `frozen/v7.sha256` |
| v8 | frozen release 3 | 24 September 2026, 10:21 UTC | DiscoveryBench, live QRData experiment, baselines | `frozen/v8.sha256` |

`make check-frozen` checks all three. Frozen files are kept byte for byte, so a few of their docstrings still show paths from our machine, such as `/home/claude/iaenv`. Replace these with your own paths when you run the commands. [frozen/README.md](frozen/README.md) lists the two files that are not byte-identical to their freeze and explains why.

## Data and licences

| Data | Source | Licence |
| --- | --- | --- |
| DABstep | Hugging Face `adyen/DABstep`, commit `c8fb51b3b898e1755f3b5a00c8dc5f25a52ff678` | CC BY 4.0 |
| DataAgentBench logs | GitHub `ucbepic/DataAgentBench`, `refs/pull/6/head` at `0a1a9c0cbb8dd8475c61a88142a778160acccfe9`, and bookreview re-run at `1b2d3b95e8a20aa66aee5ba8a2dd0d9b3e686f3e` | no licence file; the authors allowed us to publish statistics from the logs |
| KramaBench | GitHub `mitdbg/Kramabench`, commit `b2e0d77540263f8b6119f977fe79a8c4386b5a02` | no licence file at that commit |
| InfiAgent-DABench | GitHub `InfiAgent/InfiAgent`, commit `3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c` | data CC BY-NC 4.0, code Apache 2.0 |
| QRData | GitHub `xxxiaol/QRData`, commit `de450af45ff7101b328bb064c6b475f73414a7ed` | CC BY-NC 4.0 |
| DiscoveryBench | GitHub `allenai/discoverybench`, commit `c31fcf011e070f021a5f5b906896d0821f6880e8` | ODC-By |

We do not redistribute any of these datasets. We used only the public DABstep task scores and did not reconstruct hidden answers.

## Citation

If you use this code, please cite the preprint. [CITATION.cff](CITATION.cff) has the details. The archived release has a DOI on Zenodo.

## Licence

The code is released under the MIT licence. See [LICENSE](LICENSE).

## Use of AI tools

Claude (Anthropic) assisted with running the experiments and writing the analysis code.
