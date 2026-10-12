# Amendments to the frozen pre-registration (DAB rerun extension, blocks 5 to 9)

1. **12 October 2026, 10:45 UTC (12:45 Kigali time), after 97 of 540 runs. Five workers instead of three
   (operational, no change to the design).** As in amendment 6 of the first rerun: two runs of DEPS_DEV_V1 query 1
   (A and B of block 5) sent a SQL query with a correlated subquery over a large SQLite table that runs inside the
   agent process at full CPU, and held two of the three workers for 30 and 57 minutes, so one worker ran all other
   jobs. Two more workers, each with its own PostgreSQL and MongoDB instance (`dab-services.sh` with `WORKERS=5`),
   were added. The runner and the three runs in progress (those two and PANCANCER_ATLAS query 2) were stopped; the
   three runs are redone from the beginning, as for any interrupted run. Nothing else changed.
