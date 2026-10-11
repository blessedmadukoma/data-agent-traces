# Amendments to the frozen pre-registration (DAB harness rerun)

Each entry: date, what changed, why, and whether it can affect any result.

1. **6 October 2026, about 21:00 Kigali time (after the run started). Section 10, release plan.**
   Changed from "full traces: ask the DAB authors before release" to "no full traces released; only statistics,
   scores, per-run summaries, short quotes and code". Reason: Blessed decided to stay within the permission that
   Ruiying Ma gave by email (about 26 September 2026): "Please feel free to publish the statistics with quotes from
   the logs." Effect on results: none; this concerns release only.

2. **6 October 2026, 20:41 to 20:49 Kigali time. Operational interruption, no change to the design.**
   The workspace stopped between conversation turns, which killed the runner and the 3 runs in progress
   (after 14 completed runs). On restart, the runner removed the 3 incomplete run folders and started those runs
   again from the beginning, as section 6 provides for infrastructure failures. Their partial token use stays in
   the cost ledger. To keep the workspace alive, the run is now supervised from an active session, with
   `ensure_running.sh` restarting the services and the runner if they stop. Effect on results: none expected;
   restarts are logged in `restarts.log`.

3. **6 October 2026, block 0. Memory limit of the workspace (operational, no change to the design).**
   Runs of PATENTS query 3 in both arms ended without `final_agent.json` because the operating system killed the
   agent process for lack of memory (kernel log: "Memory cgroup out of memory: Killed process … python",
   about 5 GB resident). The model's query reads a large part of the 5.4 GB `patent_publication.db` into memory;
   the workspace has about 7 GB for all processes. Under section 6 these are infrastructure failures: each was
   retried twice and is excluded if it still fails. The analysis reports the excluded runs per arm and query.

4. **6 October 2026, about 21:40 Kigali time. Runner scheduling (operational, no change to the design).**
   The frozen runner waited for every run of a block to finish before it started the next block. One long run
   then left two of the three workers idle (block 0 waited more than 20 minutes for one DEPS_DEV_V1 run). After
   block 0 and the budget projection, the runner was replaced by one that runs blocks 1 onward in the same block
   order without waiting at block boundaries. Runs of block 1 that the old runner had started were stopped and
   redone from the beginning. The order of jobs, the arms, the runs per query and the budget rule are unchanged.
   The frozen runner is kept as `rerun.frozen.py`.

5. **6 October 2026, 21:45 Kigali time. Budget projection from 118 of the 119 block-0 runs.**
   One block-0 run (DEPS_DEV_V1 query 1, arm A) had made no model call for 30 minutes while its agent process
   used a full CPU, which suggests a long local query; it would have held every other worker idle until the
   2-hour cap. The projection used the 118 finished runs (cost USD 2.80, so about USD 14 for 5 blocks, far below
   USD 48), so the decision (5 blocks) does not depend on the last run. The runner was then switched as in
   amendment 4, which stopped that run; it is redone from the beginning like any interrupted run.

6. **6 October 2026, about 22:40 Kigali time. Five workers instead of three (operational, no change to the design).**
   Some runs of DEPS_DEV_V1 query 1, in both arms, send a SQL query with a correlated subquery over a large SQLite
   table; the query runs inside the agent process for a long time at full CPU (seen with py-spy: `read_sql_query`).
   DAB's harness has no time limit for SQL, so such runs end at the pre-registered 2-hour cap. With 3 workers, two
   of them were held by such runs and throughput fell to about one run per minute. Two more workers, each with its
   own PostgreSQL and MongoDB instance, were added. Ollama queues requests beyond its 3 concurrent ones. The runs in
   progress were stopped and are redone from the beginning. Nothing else changed.
