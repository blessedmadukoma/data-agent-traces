"""Tests for live/fa_runner.py: the model's code cannot replace the harness final_answer.
Run: cd live && uv run python -m unittest test_fa_runner.py"""
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def run_cells(cells):
    p = subprocess.run([sys.executable, os.path.join(HERE, "fa_runner.py")], text=True, capture_output=True,
                       input="".join(json.dumps({"code": c}) + "\n" for c in cells), timeout=60)
    return [json.loads(line) for line in p.stdout.splitlines()]


class TestProtectedFinalAnswer(unittest.TestCase):
    def test_own_function_does_not_replace_the_harness(self):
        r = run_cells(["def final_answer(a):\n    print(a)\n", "final_answer('0.20')\nprint('after')"])
        self.assertTrue(r[1]["ok"])
        self.assertTrue(r[1]["called"])
        self.assertEqual(r[1]["final_answer"], "0.20")
        self.assertEqual(r[1]["stdout"], "")

    def test_assignment_keeps_its_value_for_reads(self):
        r = run_cells(["final_answer = 5\nprint(final_answer)", "final_answer(str(final_answer + 1))"])
        self.assertEqual(r[0]["stdout"], "5\n")
        self.assertEqual(r[1]["final_answer"], "6")

    def test_del_removes_only_the_model_binding(self):
        r = run_cells(["final_answer = 'x'", "del final_answer\nprint(callable(final_answer))", "del final_answer",
                       "final_answer('C')"])
        self.assertEqual(r[1]["stdout"], "True\n")
        self.assertTrue(r[2]["ok"])
        self.assertEqual(r[3]["final_answer"], "C")

    def test_state_persists_and_errors_are_reported(self):
        r = run_cells(["x = 3", "print(x + 1)", "d = {}\nd['missing']"])
        self.assertEqual(r[1]["stdout"], "4\n")
        self.assertFalse(r[2]["ok"])
        self.assertEqual(r[2]["error_type"], "KeyError")

    def test_syntax_error_is_the_original_one(self):
        r = run_cells(["print('a'"])
        self.assertFalse(r[0]["ok"])
        self.assertEqual(r[0]["error_type"], "SyntaxError")


if __name__ == "__main__":
    unittest.main()
