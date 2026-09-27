"""Unit tests for the DataAgentBench replay.  Run: python3 -m unittest test_dab.py"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dabstep-gate"))
import dab_lib as dl  # noqa: E402
from gate import GateState, decide  # noqa: E402


class Wrapper(unittest.TestCase):
    def test_newline_escape_breaks_valid_code(self):
        own, status, dec = dl.wrapper('rows = ["a", "b"]\nprint("\\n".join(rows))')
        self.assertTrue(own)
        self.assertEqual(status, "inner_syntax")
        self.assertIsNone(dec)
        self.assertTrue(dl.harness_syntax(own, status, "Execution failed with exit code 1\nSyntaxError: unterminated string literal"))

    def test_triple_quote_ends_literal(self):
        own, status, _ = dl.wrapper('s = """x"""\nprint(s)')
        self.assertTrue(own)
        self.assertIn(status, ("outer_syntax", "outer_structure", "inner_syntax"))

    def test_plain_code_is_unchanged(self):
        self.assertEqual(dl.wrapper("x = 1\nprint(x)")[1:], ("same", "x = 1\nprint(x)"))

    def test_changed_but_valid(self):
        own, status, dec = dl.wrapper("import re\np = re.compile('\\\\d+')")
        self.assertEqual(status, "changed")
        self.assertIn("'\\d+'", dec)

    def test_agent_syntax_error_is_not_harness(self):
        own, status, _ = dl.wrapper("x = (1,")
        self.assertFalse(own)
        self.assertFalse(dl.harness_syntax(own, status, "SyntaxError: '(' was never closed"))


class Classify(unittest.TestCase):
    def test_sql(self):
        e = ('DuckDB query execution error (BinderException): Binder Error: Referenced column "ProjectName" '
             'not found in FROM clause!\nCandidate bindings: "Project_Information"')
        self.assertEqual(dl.classify("query_db", "duckdb", e)[0], "sql:missing-column")
        self.assertTrue(dl.SQL_HINT.search(e))
        e = 'Postgres query exectution error (ProgrammingError): (psycopg2.errors.UndefinedTable) relation "OrderItem" does not exist'
        self.assertEqual(dl.classify("query_db", "postgres", e)[0], "sql:missing-table")
        e = "SQLite query execution error (DatabaseError): Execution failed on sql 'SELECT x FROM t': no such column: x"
        self.assertEqual(dl.classify("query_db", "sqlite", e)[0], "sql:missing-column")
        self.assertEqual(dl.classify("query_db", "mongo", "Collection does not exist: docs")[0],
                         "mongo:missing-collection")

    def test_python(self):
        f = "Execution failed with exit code 1\n"
        self.assertEqual(dl.classify("execute_python", None, f + "KeyError: 'System'"), ("py:data-key", ["System"]))
        self.assertEqual(dl.classify("execute_python", None, f + "KeyError: 'var_function-call-123'")[0],
                         "interface:var-key")
        g, p = dl.classify("execute_python", None,
                           f + "FileNotFoundError: [Errno 2] No such file or directory: 'var_function-call-9.json'")
        self.assertEqual((g, p), ("interface:var-file", "var_function-call-9.json"))
        self.assertEqual(dl.classify("execute_python", None,
                                     f + "FileNotFoundError: [Errno 2] No such file or directory: 'out.json'")[0],
                         "py:data-file")
        self.assertEqual(dl.classify("execute_python", None, f + "NameError: name 'var_function' is not defined")[0],
                         "interface:var-name")
        self.assertEqual(dl.classify("execute_python", None, "Execution failed with exit code 124\nTimeout")[0],
                         "py:timeout")

    def test_pandas_key_lists(self):
        self.assertEqual(dl.key_list("\"None of [Index(['a', 'b'], dtype='object')] are in the [columns]\""),
                         ["a", "b"])
        self.assertEqual(dl.key_list("\"['c'] not in index\""), ["c"])


class EnvAccess(unittest.TestCase):
    N = {"var_function-call-1", "var_call_A"}

    def test_module_level_locals(self):
        code, n = dl.rewrite_env_access("x = locals()['var_function-call-1']\ny = globals().get('var_call_A')", self.N)
        self.assertEqual(n, 2)
        self.assertIn("x = _dabvar_var_function_call_1", code)
        self.assertIn("y = var_call_A", code)

    def test_locals_inside_function_is_kept(self):
        code, n = dl.rewrite_env_access("def f():\n    return locals()['var_call_A']", self.N)
        self.assertEqual(n, 0)

    def test_unknown_name_is_kept(self):
        self.assertEqual(dl.rewrite_env_access("x = locals()['var_other']", self.N)[1], 0)


class Replay(unittest.TestCase):
    ENV = {"var_function-call-1": [{"a": 1, "b": 2}, {"a": 3}],
           "var_call_A": "file_storage/call_A.json",
           "var_call_D": {"count": 3}}

    def gate(self, code, stored=None, ws=None):
        code, _ = dl.rewrite_env_access(code, set(self.ENV))
        schema, st, _ = dl.env_state(self.ENV, stored or {}, GateState)
        return decide(code, schema, st, exists=(ws or dl.Workspace()).exists, alias=True)

    def test_inline_records_through_locals_and_alias(self):
        r = self.gate("x = locals()['var_function-call-1']\nfor row in x:\n    row['c']")
        self.assertEqual((r["outcome"], r["missing"]), ("block", "c"))
        self.assertEqual(self.gate("x = locals()['var_function-call-1']\nfor row in x:\n    row['b']")["outcome"],
                         "run")

    def test_frame_from_inline_records(self):
        r = self.gate("import pandas as pd\ndf = pd.DataFrame(locals()['var_function-call-1'])\ndf['z']")
        self.assertEqual((r["outcome"], r["missing"]), ("block", "z"))

    def test_single_record(self):
        self.assertEqual(self.gate("var_call_D['total']")["outcome"], "block")

    def test_stored_result_with_exact_keys(self):
        ws = dl.Workspace()
        ws.stored.add("file_storage/call_A.json")
        code = "import json\nwith open(var_call_A) as f:\n    recs = json.load(f)\nfor r in recs:\n    r['missing']"
        self.assertEqual(self.gate(code, {"call_A.json": {"k"}}, ws)["outcome"], "block")
        self.assertNotEqual(self.gate(code, {}, ws)["outcome"], "block")      # keys not known

    def test_missing_workspace_file(self):
        r = self.gate("import json\nd = json.load(open('var_function-call-9.json'))")
        self.assertEqual((r["outcome"], r["kind"]), ("block", "file"))

    def test_loop_over_frames_is_not_a_false_block(self):
        code = ("import pandas as pd\na = pd.DataFrame(locals()['var_function-call-1'])\nb = pd.DataFrame([])\n"
                "for df in (a, b):\n    df['n'] = 1\na['n']")
        self.assertNotEqual(self.gate(code)["outcome"], "block")


class Workspace(unittest.TestCase):
    def test_model(self):
        ws = dl.Workspace()
        self.assertFalse(ws.exists("out.csv"))
        self.assertFalse(ws.exists("/tmp/out.csv"))                    # a new container: /tmp starts empty
        self.assertIsNone(ws.exists("/home/user/out.csv"))
        self.assertFalse(ws.exists("/workspace/out.csv"))
        ws.after_call(["/tmp/a.json"], 1, ran=True)
        self.assertTrue(ws.exists("/tmp/a.json"))
        self.assertIsNone(dl.Workspace(empty_tmp=False).exists("/tmp/out.csv"))
        ws.after_call(["out.csv"], 1, ran=True)
        self.assertTrue(ws.exists("./out.csv"))
        ws.after_call([], 1, ran=True)            # a write the gate could not resolve
        self.assertIsNone(ws.exists("other.csv"))

    def test_write_calls(self):
        self.assertEqual(dl.write_calls("open('a', 'w')\nopen('b')\ndf.to_csv(p)\nos.system('x')"), 3)

    def test_first_record_keys_from_cut_preview(self):
        self.assertEqual(dl.first_record_keys('[{"a": 1, "b": "x"}, {"a": 2, "b"'), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
