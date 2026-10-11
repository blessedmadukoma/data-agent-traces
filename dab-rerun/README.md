# Controlled rerun of DataAgentBench's harness

This folder holds the pre-registered rerun in Section 3.1 of the preprint. DataAgentBench (DAB), commit `6ad9d68`, ran 54 queries 5 times per arm with `gpt-oss:120b`:

- **arm A:** the released harness;
- **arm B:** the model's code passed unchanged;
- **arm C:** as arm B, and a cut MongoDB result is marked (11 queries).

| File | What it is |
|---|---|
| `preregistration.md` | The plan, frozen before the first run. |
| `FREEZE.txt` | SHA-256 hashes of the frozen files, with the DAB commit, the executor image and the model. |
| `AMENDMENTS.md` | Six changes after the freeze. Five are operational, and one concerns what is released. None changes the design. |
| `harness-patches.diff` | Every change to DAB's code: model access, budget ledger, the `repr()` line, the changed clause of key rule 7, the note on cut results and `mongorestore --uri`. |
| `rerun.py` | The runner as frozen. |
| `analysis.py` | The pre-registered analysis (H1 to H5). |
| `judge.py` | DAB's own failure-analysis judge on the failed runs (H4). |
| `runs.jsonl` | One line per run attempt (613 lines, retries included). The fields give the arm, dataset, query, block, attempt, correctness, end reason, model calls, tool calls, failed tool calls, tokens, wall time and worker. |
| `analysis.json` | The results that the preprint reports. |

## Check the files against the freeze

```bash
cd dab-rerun && sha256sum preregistration.md harness-patches.diff rerun.py
```

Each hash must equal the line for that file in `FREEZE.txt`. The other files that `FREEZE.txt` lists (`executor.Dockerfile`, `dab-services.sh` and the patched DAB files) are not here. The patched DAB files are DAB's code with `harness-patches.diff` applied, and the other two were set-up scripts for our workspace.

## What is not released, and why

- **The traces and the final answers.** They contain query results from DAB's datasets. The DAB authors allowed us to publish statistics and short quotes from their logs (amendment 1). `runs.jsonl` therefore leaves out the `final_result` field that the runner wrote.
- **A row for each tool call.** The call-level counts that the preprint uses are in `analysis.json`, per arm. `H1` gives Python calls, harness-caused failures, runs with at least one such failure and silent changes. `H3` gives failed calls per run and the share recorded as syntax errors.
- **The runner used from block 1 onward.** It changed only the scheduling (amendment 4), and it was not kept.

## Reproduce the numbers in the preprint

From the released files (no model calls):

```bash
python3 - <<'EOF'
import json, collections, numpy as np
final = {}
for line in open("dab-rerun/runs.jsonl"):
    r = json.loads(line)
    final[(r["arm"], r["dataset"], r["query"], r["block"])] = r
bad = lambda t: (t or "").startswith("llm_response_failed") or (t or "").startswith("agent_run_failed")
kept = {k: r for k, r in final.items() if not bad(r["terminate_reason"])}
for arm in "AB":
    per_ds = collections.defaultdict(lambda: collections.defaultdict(list))
    for (a, d, q, b), r in kept.items():
        if a == arm:
            per_ds[d][q].append(r["is_valid"])
    print(arm, round(float(np.mean([np.mean([np.mean(v) for v in qs.values()]) for qs in per_ds.values()])), 4))
EOF
```

This prints Pass@1 0.3099 for arm A and 0.2951 for arm B, as in `analysis.json`. The other results (H1, H3 and H4) need the traces. To make them again, apply `harness-patches.diff` to DAB at commit `6ad9d68`, set `OLLAMA_API_KEY`, and run `rerun.py run`, then `judge.py` and `analysis.py`. The paths at the top of each script point to our workspace; change them to yours. A new run will not give the same traces, because the model samples at its default temperature and the request sets no seed.

| Preprint | Value | In `analysis.json` |
|---|---|---|
| 3.1, Table 3 | 26.3% of arm A's Python calls (360 of 1,367) are harness-caused; 0 in arms B and C | `H1` |
| 3.1 | 26 of 268 arm A runs had at least one; 149 more calls ran with a changed string | `H1.A.runs_with_harness_caused`, `H1.A.silent_changes` |
| 3.1, Table 3 | Pass@1 31.0% and 29.5%; B − A −1.5 points (95% CI −7.6 to +4.1), Wilcoxon p = 0.62 | `pass1`, `H2_B_minus_A` (also from `runs.jsonl`, above) |
| 3.1 | 1.86 fewer failed calls per run (CI 0.63 to 3.48) | `H3_failed_calls_B_minus_A` |
| 3.1 | 5 runs excluded (2 in A, 3 in B); 2 runs in B reached the 2-hour cap | `excluded`, `runs_by_arm.B.terminate` |
| 3.1 | share of failed calls recorded as syntax errors −11.7 points (CI −21.8 to −3.2), p = 0.027 | `H3_syntax_share_B_minus_A` |
| 3.1, Table 3 | judge: FM4 in 39% and 34% of failed runs; B − A per task −5.3 points (CI −14.4 to +3.3); on failed runs that end with an answer, +10.3 (CI −2.2 to +23.5) | `H4.all_failed`, `H4.return_answer_only` (both given as A − B) |
| Table 3 | cost per run USD 0.0256 and 0.0183 | `runs_by_arm`: `usd` divided by `runs` |
| Table 3 | failed calls per run 2.77 and 0.99; requests to the model per run 11.7 and 8.8 | means of `failed_tool_calls` and `llm_calls` over the kept runs in `runs.jsonl`; the 2 arm-B runs that reached the 2-hour cap have no `llm_calls` and count as 0 (without them, 8.9) |
| 3.1 | Arm C: 21 notes, followed by a query with a limit 20 times | `H5_notes` |
