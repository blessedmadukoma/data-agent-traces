# kimi-k2.6 rerun of DataAgentBench: pilots only, not run

On 12 October 2026 we planned a second controlled rerun with kimi-k2.6 (`preregistration_DRAFT_not_run.md`, drafted but never
frozen) and ran two pilots: 12 runs, 3 of them stopped early to free Ollama's request slots, USD 0.76 in total.
kimi-k2.6 did most of its work in SQL: in its 9 runs with the released harness it made 19 Python calls, and the
harness caused 1 failure (about 0.1 per run; gpt-oss:120b had 1.34 in the first rerun). Blessed therefore chose
to extend the gpt-oss:120b rerun instead (`dab-rerun-ext/`). The pilot runs are excluded from every analysis.

`rerun.py` is the runner used for the pilots (`python rerun.py pilot ARM:DATASET:QUERY ...`).
