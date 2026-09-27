"""Unit tests for the receipt replay. Run in this folder: uv run python -m unittest test_rq3.py

The tests that run cells need the DABstep context files. They look in ../data/dabstep,
or in the folder named by the environment variable DABSTEP_DATA.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import context_layout as cl  # noqa: E402
import rq3_run as rr  # noqa: E402
import sandbox  # noqa: E402
import select_prefixes as sp  # noqa: E402

DATA = os.environ.get("DABSTEP_DATA", os.path.join(HERE, "..", "data", "dabstep"))
CTX = os.path.join(DATA, "data", "context")
RC = {"location": "line 1", "kind": "column", "source": "payments.csv", "missing": "month",
      "reason": "payments.csv has no column 'month'", "available": ["year", "day_of_year"],
      "found_in": [], "near_matches": ["year"]}


class Messages(unittest.TestCase):
    def test_receipt_text(self):
        t = rr.receipt_text(RC)
        self.assertIn("payments.csv has no column 'month'", t)
        self.assertIn("Available columns in payments.csv: year, day_of_year", t)
        self.assertIn("Close matches: year", t)

    def test_feedback_arms_end_the_same_way(self):
        a = rr.feedback("traceback", {"ok": False, "stdout": "", "error": "Traceback...\nKeyError: 'month'"}, RC)
        b = rr.feedback("receipt", None, RC)
        self.assertTrue(a.endswith("\nContinue.") and b.endswith("\nContinue."))
        self.assertIn("KeyError: 'month'", a)
        self.assertIn("was not run", b)

    def test_extract_code(self):
        self.assertEqual(rr.extract_code("x\n```python\nprint(1)\n```\n```python\nprint(2)\n```"), "print(1)")
        self.assertIsNone(rr.extract_code("no code"))
        self.assertIsNone(rr.extract_code("The answer is 42."))
        self.assertEqual(rr.extract_code("final_answer(691.2)"), "final_answer(691.2)")

    def test_mentions(self):
        self.assertTrue(rr.mentions("df['month']", "month"))
        self.assertFalse(rr.mentions("df['year']  # month", "month"))


class Layout(unittest.TestCase):
    L = {"/data/ctx": {"ok": {"payments.csv"}, "missing": {"fees.json"}},
         "": {"ok": {"fees.json", "manual.md"}, "missing": {"manual.md"}}}

    def test_all_context_files_in_a_used_directory(self):
        p = cl.placements(self.L)
        self.assertEqual(p["/data/ctx/merchant_data.json"], "merchant_data.json")
        self.assertIn("/data/ctx/payments.csv", p)
        self.assertNotIn("/data/ctx/fees.json", p)        # recorded missing, never read there
        self.assertIn("manual.md", p)                     # read and missing: the workspace changed
        self.assertEqual(len(p), 6 + 7)

    def test_trace_evidence_removes_a_file(self):
        p = sp.sandbox_paths(self.L, {"./manual.md", "/data/ctx/payments.csv"})
        self.assertNotIn("manual.md", p)
        self.assertNotIn("/data/ctx/payments.csv", p)
        self.assertIn("/data/ctx/manual.md", p)

    def test_extra_paths_prefers_sandbox_paths(self):
        self.assertEqual(rr.extra_paths({"context_paths": {"a/p.csv": "p.csv"},
                                         "sandbox_paths": {"a/f.json": "fees.json"}}), {"a/f.json": "fees.json"})
        self.assertEqual(rr.extra_paths({"context_paths": {"a/p.csv": "p.csv"}}), {"a/p.csv": "p.csv"})


@unittest.skipUnless(os.path.isdir(CTX), "DABstep context files not present")
class DryEpisode(unittest.TestCase):
    def test_local_sandbox_and_dry_model(self):
        ep = {"episode": "t1", "submitter": "s", "task_id": "1", "question": "q", "guidelines": "g",
              "receipt": RC, "context_paths": {},
              "prefix": [{"code": "import pandas as pd\np = pd.read_csv('data/context/payments.csv')", "failed": False}],
              "blocked": {"code": "p['month']"}}

        class A:
            data = DATA
            sandbox, cell_timeout, temperature, max_tokens, steps, transcripts = "local", 60, 1.0, 100, 4, None
        for arm in ("traceback", "receipt"):
            r = rr.run_episode(ep, arm, 1, rr.Dry(), A)
            self.assertEqual(r["status"], "done")
            self.assertEqual(r["prefix_mismatch"], 0)
            self.assertEqual(r["steps_to_fix"], 2)
            self.assertIsNone(r["steps_to_fix_nonconstant"])
            self.assertTrue(r["constant_answer"])
        self.assertTrue(rr.run_episode(ep, "traceback", 1, rr.Dry(), A)["blocked_names_missing"])

    def test_other_context_file_in_a_used_directory(self):
        """A model that reads another context file from a directory the trace used
        finds it (the pilot sandbox placed only the files the prefix read)."""
        paths = cl.placements({"inputs/ctx": {"ok": {"payments.csv"}, "missing": set()}})
        ex = sandbox.Executor(CTX, paths, mode="local", timeout=30)
        try:
            r = ex.run("import json\nprint(len(json.load(open('inputs/ctx/fees.json'))))")
            self.assertTrue(r["ok"], r.get("error"))
        finally:
            ex.close()

    def test_timeout_is_reported(self):
        ex = sandbox.Executor(CTX, mode="local", timeout=2)
        try:
            r = ex.run("while True:\n    pass")
            self.assertEqual(r["error_type"], "TimeoutError")
            self.assertTrue(r["lost"])
        finally:
            ex.close()


if __name__ == "__main__":
    unittest.main()
