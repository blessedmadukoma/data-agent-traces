# Amendments to the E2 pre-registration (live/e2_preregistration.md)

1. **12 October 2026, 09:36 UTC (11:36 Kigali time), after 216 runs of seed 1. Shared prefixes, for time.**
   - **Why.** Ollama Cloud runs at most 3 requests at once for this account, across all models: a 20-token probe
     waited about 4 s in the queue while the runs were going. At the measured rate, the 10 planned seeds would have
     taken about 20 hours, and Blessed decided not to buy a plan with more concurrent requests.
   - **Change.** The three arms now share a model call while their message histories are identical
     (`live/e2_fork.py`). Until a step where the arms differ (a block in R, or a missing-name error, which changes
     the message in E), all three arms send the same messages, so one reply serves all of them. At such a step,
     each arm handles the step as before, with its own sandbox and its own checker state, and arms whose next
     message differs continue as separate branches with their own replies. Every arm's trajectory is therefore a
     run of that arm, as before; only the coupling between arms changes. Runs without a data-reference attempt are
     now identical across arms, so the differences between arms have less variance, and the identity in H5
     (difference over all runs = share of runs with a DRA × difference among them) holds exactly.
   - **Unchanged.** Arms, prompt, model and settings, checker, sandbox, measures, hypotheses, tests, decision
     rules, seeds (10), block order per seed (one shuffle per seed with 20261012 + s, now over questions) and the
     budget cap. The power estimate of section 6 is now conservative.
   - **The 216 runs of the original design** are set aside in `runs/e2_independent_pilot/` and excluded. No
     analysis by arm was run on them. The only check made on them was whether the request's `seed` makes the
     arms' first steps identical; it does not (identical in 2 of 40 question–seed groups), so the original design
     did not share replies.
   - **New output fields:** `shared_with` (the arms that shared each model call), `calls_billed` and
     `job_cost_usd`. `cost_usd` and `usage` remain the arm's own trajectory.
   - **Test before the change** (8 questions, seed 1, excluded): `runs/e2fork_test/`.

2. **12 October 2026, 10:23 UTC (12:23 Kigali time), after 279 of 4,110 jobs (seed 1). One sandbox per branch, for CPU.**
   - **Why.** The workspace has 2 CPUs. Under amendment 1, each arm had its own sandbox, so every shared step ran
     three times, and three containers started per job; with the DAB extension's long SQL queries on the same
     machine (DEPS_DEV_V1 query 1, as in amendment 6 of the first rerun), the load average reached 7 to 8 and the
     sandboxes, not the model, set the pace (about 66 s per job, of which 13 s were model calls).
   - **Change** (`live/e2_fork.py`; the version of amendment 1 is kept as `live/e2_fork_v1.py`). A branch of arms
     with identical histories now shares one sandbox and one checker state. When a branch splits, the first group
     whose state matches the sandbox (it ran the step, or nothing ran) keeps it; every other group gets a new
     sandbox that executes the branch's earlier executed steps again, in order (and the current step, if that
     group ran it). Each group gets its own copy of the checker state, committed as before (a blocked step
     counts as failed). The job record counts these rebuilds and the rebuilt steps whose outcome (success or error
     type) differs from the original (`sandbox_rebuilds`); a rebuild whose step times out ends that group's arms
     as `executor_lost`.
   - **Unchanged.** What each arm's model receives and how each step is handled; everything that amendment 1 left
     unchanged. The 279 jobs recorded under amendment 1 stay in the data (`design: shared_prefix`); later jobs carry
     `design: shared_prefix_v2`. Jobs in progress at the change were not recorded and run again from the start.
   - **Test before the change** (8 questions × 2 seeds, excluded): `runs/e2fork_test2/`. Six jobs split; 12 rebuilds;
     no rebuilt step differed from the original.

3. **12 October 2026, 14:46 to 15:56 UTC. Operational interruption, no change to the design.** The workspace
   stopped at 14:46 UTC, after 3,441 of 4,110 jobs (seeds 1 to 8 complete). When it restarted at 15:55, the Docker
   daemon was started again and the runner resumed; the jobs that were in progress at 14:46 were not recorded and
   ran again from the start, as section 7 provides. The DAB extension had finished before the stop.

4. **12 October 2026, after the last run. Correction to the analysis code (no change to the plan).** In
   `live/e2_report.py`, the exact sign tests for steps and tokens received NumPy integers, and `2 ** n` overflowed for
   more than 62 questions, which returned p = 1. The counts are now passed as Python integers. No other number
   changes. The block check (`live/e2_validate.py`) and the report were run once all 4,110 jobs were done.
