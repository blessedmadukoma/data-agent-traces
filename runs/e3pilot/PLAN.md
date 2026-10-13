# E3 pilot: choice of the second model (written before the pilot ran, 12 October 2026)

- **Purpose:** pick a second model that makes data-reference errors often with the plain QRData prompt of E1.
- **Candidates:** gpt-oss:20b, gemma4:31b, glm-5.3-flash, nemotron-3-nano:30b (all on Ollama Cloud, all priced
  below USD 0.6 per 1M output tokens). Reference: gpt-oss:120b, the first model of E3.
- **Runs:** 40 questions drawn at random from the 411 (Python `random.Random(20261012).sample`), arm N only (no
  checker), seed 1, plain prompt, the E2 runner (`live/e3_fork.py`).
- **Measure:** the share of runs with at least one executed data-reference error (a step that fails with
  `FileNotFoundError`, `IsADirectoryError` or `NotADirectoryError`, or with `KeyError` where the checker, deciding
  in shadow mode, would block); the share of runs that give an answer; accuracy (recorded, not used to choose).
- **Rule:** choose the candidate with the highest share of runs with such an error, among candidates that give an
  answer in at least 50% of runs. A tie goes to the cheaper model.
- **Use:** the pilot runs are excluded from E3. No result of the checker or of arm E is seen before the choice.
