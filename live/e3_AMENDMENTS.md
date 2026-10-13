# Amendments to the E3 pre-registration (live/e3_preregistration.md)

1. **12 October 2026, 19:23 UTC (21:23 Kigali time), after 405 of 4,110 gpt-oss:120b jobs and 1,541 of 4,110
   gemma4:31b jobs. Budget cap and workers of gpt-oss:120b, for cost and time.**
   - **Why.** Seed 1 of gpt-oss:120b cost USD 1.16 for 405 jobs (the runner's cost log, which shows no result by
     arm), so the 10 seeds will cost about USD 11.8. The cap of USD 10 (section 7) would stop the process during
     seed 9, and section 7 would then drop seeds 9 and 10 for both models. Blessed approved the cost of the full
     run ("we have enough money"). gemma4:31b will cost about USD 2.7, so the total is about USD 14.5.
   - **Change.**
     - The cap for gpt-oss:120b goes from USD 10 to USD 15. The cap for gemma4:31b stays at USD 10.
     - When gemma4:31b has finished, the gpt-oss:120b process restarts with 6 workers instead of 4, so that it
       uses the request slots that gemma4:31b no longer uses. Jobs are independent and their order in each seed
       does not change. Jobs in progress at the restart are not recorded and run again from the start, as
       section 7 provides.
   - **Unchanged.** Everything else: arms, prompt, models, settings, checker, sandbox, measures, hypotheses,
     tests, decision rules, seeds and the order of jobs.
   - **Decided on** cost and throughput only. Nobody has looked at accuracy, steps or tokens by arm.

2. **12 October 2026, 21:57 to 22:10 UTC. Operational, no change to the design.**
   - gemma4:31b finished at 21:57 UTC with 4,102 of its 4,110 jobs. The other 8 had failed with a `JSONDecodeError`
     (a reply that was not valid JSON) and had not been recorded. A second start at 21:58 ran them again, as section
     7 provides; all 8 completed. gemma4:31b cost USD 2.67 in total.
   - The gpt-oss:120b process restarted with 6 workers at 21:57 UTC (amendment 1). Its jobs in progress at that
     moment were not recorded and run again from the start.
   - The block check of gemma4:31b (`live/e2_validate.py`) ran while gpt-oss:120b was still running, to save time.
     It replays only arm R's blocked steps and reports their classes; it reports no accuracy, steps or tokens by arm.
   - The block check of gpt-oss:120b also ran in 16 parts during the run, each on the result lines recorded since
     the previous part (`runs/e3/validation_parts/PARTS.txt`); their concatenation, `runs/e3/validation_gpt-oss-120b.jsonl`,
     covers all 2,311 blocked steps once. Part 3 was started twice by a numbering error; the second run replaced the
     first and gave the same classes (115 confirmed, 4 other error).

3. **13 October 2026, after the analysis. Correction to the reported bound (no change to the plan).** The H6 bound of
   `live/e2_report.py` counts the steps that ran in arms N and E. Under the shared-prefix design, a step that N and E
   share is one step, counted twice. The paper reports the bound on distinct steps (N's steps plus E's steps that did
   not share N's model call): 31,839 for gpt-oss:120b and 11,590 for gemma4:31b, and 22,507 for E2
   (`tools/e3_numbers.py`, `distinct_shadow_steps_ran`). No false block was found in either count.
