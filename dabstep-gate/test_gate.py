"""Unit tests for gate.py.  Run: python3 -m unittest test_gate.py"""
import os
import unittest

from gate import GateState, decide

S = {"payments.csv": {"merchant", "year", "eur_amount", "day_of_year"},
     "merchant_data.json": {"merchant", "account_type"},
     "fees.json": {"ID", "rate"}}
H = "import pandas as pd\np = pd.read_csv('payments.csv')\n"


def out(code, state=None):
    return decide(code, S, state)["outcome"]


# When the gate stops tracking a frame, a later read of it is not checked:
# the outcome is "run" or "unknown", never "block".


class Basics(unittest.TestCase):
    def test_missing_column_blocks(self):
        r = decide(H + "p['date']", S)
        self.assertEqual(r["outcome"], "block")
        self.assertEqual(r["missing"], "date")
        self.assertEqual(r["location"], "line 3")

    def test_present_column_runs(self):
        self.assertEqual(out(H + "p['merchant']"), "run")

    def test_list_of_columns(self):
        self.assertEqual(out(H + "p[['merchant', 'date']]"), "block")

    def test_receipt_names_other_source(self):
        r = decide(H + "p['account_type']", S)
        self.assertEqual(r["found_in"], ["merchant_data.json"])

    def test_near_match(self):
        r = decide(H + "p['eur_amt']", S)
        self.assertIn("eur_amount", r["near_matches"])

    def test_path_forms(self):
        for path in ["'/data/context/payments.csv'", "DIR + 'payments.csv'",
                     "os.path.join(DIR, 'payments.csv')", "f'{DIR}payments.csv'"]:
            code = f"import os\nimport pandas as pd\nDIR = '/d/'\np = pd.read_csv({path})\np['date']"
            self.assertEqual(out(code), "block", path)

    def test_dynamic_path_unknown(self):
        self.assertEqual(
            out("p = pd.read_csv(get_path())\np['date']"), "unknown")

    def test_workspace_file_unknown(self):
        self.assertEqual(out("p = pd.read_csv('other.csv')\np['x']"), "unknown")

    def test_usecols(self):
        self.assertEqual(
            out("p = pd.read_csv('payments.csv', usecols=['merchant', 'date'])"), "block")
        self.assertEqual(
            out("p = pd.read_csv('payments.csv', usecols=['merchant'])\np['year']"), "block")

    def test_names_and_header_none(self):
        self.assertEqual(
            out("p = pd.read_csv('payments.csv', names=['a', 'b'])\np['a']"), "run")
        self.assertEqual(
            out("p = pd.read_csv('payments.csv', header=None)\np[0]"), "unknown")

    def test_invalid_python(self):
        self.assertEqual(out("p = pd.read_csv('payments.csv'\n"), "unknown")

    def test_no_read(self):
        self.assertEqual(out("x = 1"), "unknown")


class Tracking(unittest.TestCase):
    def test_created_column(self):
        self.assertEqual(out(H + "p['date'] = 1\np['date']"), "run")

    def test_reassignment_stops_tracking(self):
        self.assertNotEqual(out(H + "p = p.merge(q)\np['date']"), "block")

    def test_inplace_stops_tracking(self):
        self.assertNotEqual(
            out(H + "p.rename(columns={'year': 'date'}, inplace=True)\np['date']"), "block")
        self.assertNotEqual(out(H + "x = p.pop('year')\np['date']"), "block")

    def test_loc_assignment_stops_tracking(self):
        self.assertNotEqual(out(H + "p.loc[:, 'date'] = 1\np['date']"), "block")

    def test_attribute_assignment_stops_tracking(self):
        self.assertNotEqual(
            out(H + "p.columns = [c.lower() for c in p.columns]\np['date']"), "block")

    def test_alias_stops_tracking(self):
        self.assertNotEqual(out(H + "q = p\nq['date'] = 1\np['date']"), "block")
        self.assertNotEqual(out(H + "d = {'p': p}\np['date']"), "block")

    def test_user_function_stops_tracking(self):
        code = H + "def add(df):\n    df['date'] = 1\nadd(p)\np['date']"
        self.assertNotEqual(out(code), "block")
        self.assertEqual(out(H + "print(len(p))\np['date']"), "block")

    def test_function_body_changes_global(self):
        code = H + "def add():\n    p['date'] = 1\np['date']"
        self.assertNotEqual(out(code), "block")

    def test_exec_stops_tracking(self):
        self.assertNotEqual(out(H + "exec('p[\"date\"] = 1')\np['date']"), "block")

    def test_read_in_assignment_target(self):
        self.assertEqual(out(H + "p.loc[p['date'] > 0, 'year'] = 1"), "block")

    def test_augmented_assignment_reads(self):
        self.assertEqual(out(H + "p['date'] += 1"), "block")


class ControlFlow(unittest.TestCase):
    def test_if_body_is_conditional(self):
        self.assertEqual(out(H + "if x:\n    p['date']"), "unknown")

    def test_try_body_is_conditional(self):
        self.assertEqual(out(H + "try:\n    p['date']\nexcept KeyError:\n    pass"), "unknown")

    def test_short_circuit_and(self):   # audit 5 and 6
        code = H + "if 'date' in p.columns and (p['date'] == 1).any():\n    y = 1"
        self.assertEqual(out(code), "unknown")

    def test_first_operand_always_runs(self):
        self.assertEqual(out(H + "if p['date'].any() and x:\n    y = 1"), "block")

    def test_conditional_expression(self):
        self.assertEqual(out(H + "y = p['date'] if 'date' in p.columns else None"), "unknown")

    def test_comprehension(self):
        self.assertEqual(out(H + "y = [p['date'] for _ in range(3)]"), "unknown")
        self.assertEqual(out(H + "y = [v for v in p['date']]"), "block")

    def test_code_after_final_answer(self):   # audit 7
        self.assertEqual(out(H + "final_answer(1)\np['date']"), "run")

    def test_code_after_conditional_final_answer(self):   # audit 3
        code = H + "if 'date' not in p.columns:\n    final_answer('NA')\np['date']"
        self.assertEqual(out(code), "unknown")

    def test_code_after_conditional_raise_still_checked(self):
        code = H + "if 'date' not in p.columns:\n    raise ValueError('no date')\np['date']"
        self.assertEqual(out(code), "block")

    def test_code_after_raise(self):
        self.assertEqual(out(H + "raise ValueError()\np['date']"), "run")

    def test_sys_exit(self):
        self.assertEqual(out(H + "import sys\nsys.exit(0)\np['date']"), "run")

    def test_with_body_runs(self):
        self.assertEqual(out(H + "with ctx():\n    p['date']"), "block")

    def test_read_inside_final_answer_is_checked(self):
        self.assertEqual(out(H + "final_answer(p['date'])"), "block")


class State(unittest.TestCase):
    def run_cells(self, cells):
        st, res = GateState(), []
        for code, failed in cells:
            o = out(code, st)
            res.append(o)
            st.commit(failed=failed or o == "block")
        return res

    def test_frame_from_earlier_cell(self):
        self.assertEqual(self.run_cells([(H, False), ("p['date']", False)]), ["run", "block"])

    def test_without_state_unknown(self):
        self.assertEqual(out("p['date']"), "unknown")

    def test_column_created_earlier(self):
        r = self.run_cells([(H + "p['date'] = 1", False), ("p['date']", False)])
        self.assertEqual(r, ["run", "run"])

    def test_failed_cell_drops_changed_names(self):
        r = self.run_cells([(H, False), ("p['date'] = 1\nundefined()", True), ("p['date']", False)])
        self.assertEqual(r[2], "unknown")

    def test_failed_cell_keeps_unchanged_names(self):
        r = self.run_cells([(H, False), ("print(p.shape)\n1/0", True), ("p['date']", False)])
        self.assertEqual(r[2], "block")

    def test_blocked_cell_does_not_add_columns(self):
        r = self.run_cells([(H, False), ("p['year']\np['date'] = p['dat']", False), ("p['date']", False)])
        self.assertEqual(r, ["run", "block", "unknown"])

    def test_constant_from_earlier_cell(self):
        r = self.run_cells([("P = '/d/payments.csv'", False),
                            ("import pandas as pd\np = pd.read_csv(P)\np['date']", False)])
        self.assertEqual(r[1], "block")

    def test_invalid_cell_keeps_state(self):
        r = self.run_cells([(H, False), ("p[", True), ("p['date']", False)])
        self.assertEqual(r[2], "block")

    def test_state_is_not_shared_between_snapshots(self):
        st = GateState()
        decide(H, S, st)
        st.commit(False)
        before = dict(st.frames)
        decide("p['new'] = 1", S, st)      # decided but not committed
        self.assertEqual(st.frames, before)


J = "import json\n"


class Records(unittest.TestCase):
    """JSON records and csv rows (version 4)."""

    def test_loop_over_json_records(self):
        code = J + "with open('/d/fees.json') as f:\n    fees = json.load(f)\nfor fee in fees:\n    x = fee['id']"
        r = decide(code, S)
        self.assertEqual(r["outcome"], "block")
        self.assertEqual((r["kind"], r["missing"]), ("key", "id"))
        self.assertIn("ID", r["near_matches"] + r["available"])

    def test_loop_present_key_runs(self):
        code = J + "fees = json.load(open('fees.json'))\nfor fee in fees:\n    x = fee['ID']"
        self.assertEqual(out(code), "run")

    def test_tool_text_not_followed(self):
        code = J + "fees = json.loads(read_file('fees.json'))\nfor fee in fees:\n    fee['id']"
        self.assertNotEqual(out(code), "block")

    def test_tolerant_keys(self):
        code = J + "fees = json.load(open('fees.json'))\nfees[0]['rates']"
        r = decide(code, S, tolerant_keys=True)
        self.assertNotEqual(r["outcome"], "block")
        self.assertEqual(r["substitutions"][0]["used"], "rate")
        code = J + "fees = json.load(open('fees.json'))\nfees[0]['zzz']"
        self.assertEqual(decide(code, S, tolerant_keys=True)["outcome"], "block")
        self.assertEqual(decide(H + "p['merchant_name']", S, tolerant_keys=True)["outcome"], "block")

    def test_loads_of_text(self):
        for src in ["open('fees.json').read()", "t"]:
            pre = "t = open('fees.json').read()\n" if src == "t" else ""
            code = J + pre + f"fees = json.loads({src})\nfor fee in fees:\n    fee['id']"
            self.assertEqual(out(code), "block", src)

    def test_first_iteration_until_continue(self):
        code = J + "fees = json.load(open('fees.json'))\nfor fee in fees:\n    if fee['rate'] > 1:\n        continue\n    fee['id']"
        self.assertEqual(out(code), "unknown")

    def test_break_first(self):
        code = J + "fees = json.load(open('fees.json'))\nfor fee in fees:\n    break\n    fee['id']"
        self.assertNotEqual(out(code), "block")

    def test_inner_loop_break_does_not_skip_outer(self):
        code = J + "fees = json.load(open('fees.json'))\nfor fee in fees:\n    for c in 'ab':\n        break\n    fee['id']"
        self.assertEqual(out(code), "block")

    def test_loop_inside_if_is_conditional(self):
        code = J + "fees = json.load(open('fees.json'))\nif x:\n    for fee in fees:\n        fee['id']"
        self.assertEqual(out(code), "unknown")

    def test_comprehension_filter(self):
        code = J + "fees = json.load(open('fees.json'))\ny = [f for f in fees if f['id'] == 787]"
        self.assertEqual(out(code), "block")
        code = J + "fees = json.load(open('fees.json'))\ny = next((f for f in fees if f['id'] == 787), None)"
        self.assertEqual(out(code), "block")

    def test_generator_not_consumed(self):
        code = J + "fees = json.load(open('fees.json'))\ng = (f for f in fees if f['id'] == 787)"
        self.assertEqual(out(code), "unknown")

    def test_comprehension_element_after_filter_is_conditional(self):
        code = J + "fees = json.load(open('fees.json'))\ny = [f['id'] for f in fees if f['rate'] > 99]"
        self.assertEqual(out(code), "unknown")

    def test_index_zero(self):
        code = J + "fees = json.load(open('fees.json'))\nfees[0]['id']"
        self.assertEqual(out(code), "block")

    def test_added_key(self):
        code = J + "fees = json.load(open('fees.json'))\nfor fee in fees:\n    fee['id'] = fee['ID']\nfees[0]['id']"
        self.assertEqual(out(code), "run")
        code = J + "fees = json.load(open('fees.json'))\nfees[0]['id'] = 1\nfor fee in fees:\n    fee['id']"
        self.assertEqual(out(code), "run")

    def test_append_stops_tracking(self):
        code = J + "fees = json.load(open('fees.json'))\nfees.append({'id': 1})\nfees[0]['id']"
        self.assertNotEqual(out(code), "block")

    def test_get_is_not_checked(self):
        code = J + "fees = json.load(open('fees.json'))\nfor fee in fees:\n    fee.get('id')"
        self.assertNotEqual(out(code), "block")

    def test_dictreader_rows(self):
        code = "import csv\nwith open('payments.csv') as f:\n    for row in csv.DictReader(f):\n        row['date']"
        self.assertEqual(out(code), "block")
        code = "import csv\nwith open('payments.csv') as f:\n    rows = list(csv.DictReader(f))\nrows[0]['date']"
        self.assertEqual(out(code), "block")

    def test_dictreader_read_once(self):
        code = "import csv\nf = open('payments.csv')\nr = csv.DictReader(f)\nfor row in r:\n    pass\nfor row in r:\n    row['date']"
        self.assertEqual(out(code), "unknown")

    def test_iterrows_and_to_dict(self):
        self.assertEqual(out(H + "for i, row in p.iterrows():\n    row['date']"), "block")
        self.assertEqual(out(H + "for rec in p.to_dict('records'):\n    rec['date']"), "block")
        self.assertEqual(out(H + "q = pd.read_csv('payments.csv', nrows=0)\nfor i, row in q.iterrows():\n    row['date']"),
                         "unknown")

    def test_chunks(self):
        code = "import pandas as pd\nfor c in pd.read_csv('payments.csv', chunksize=100):\n    c['date']"
        self.assertEqual(out(code), "block")

    def test_dataframe_from_records(self):
        code = J + "import pandas as pd\nfees = json.load(open('fees.json'))\ndf = pd.DataFrame(fees)\ndf['id']"
        self.assertEqual(out(code), "block")

    def test_records_across_cells(self):
        st = GateState()
        out(J + "fees = json.load(open('fees.json'))", st)
        st.commit(False)
        self.assertEqual(out("for fee in fees:\n    fee['id']", st), "block")

    def test_path_from_join(self):
        code = J + "import os\nP = os.path.join('/d', 'fees.json')\nfees = json.load(open(P))\nfees[0]['id']"
        self.assertEqual(out(code), "block")


def gone(path):
    return {"payments.csv": True, "fees.json": True}.get(path, False)


class Files(unittest.TestCase):
    """File existence with an exists() callback (version 4)."""

    def test_missing_file_blocks(self):
        r = decide("import pandas as pd\nd = pd.read_csv('transactions.csv')", S, exists=gone)
        self.assertEqual((r["outcome"], r["kind"], r["missing"]), ("block", "file", "transactions.csv"))

    def test_existing_file_runs(self):
        self.assertEqual(decide(H, S, exists=gone)["outcome"], "run")

    def test_no_callback_no_check(self):
        self.assertEqual(out("import pandas as pd\nd = pd.read_csv('transactions.csv')"), "unknown")

    def test_unknown_existence(self):
        r = decide("import pandas as pd\nd = pd.read_csv('x.csv')", S, exists=lambda p: None)
        self.assertEqual(r["outcome"], "unknown")

    def test_open_and_json(self):
        self.assertEqual(decide("import json\nd = json.load(open('data.json'))", S, exists=gone)["outcome"], "block")
        self.assertEqual(decide("with open('data.txt') as f:\n    t = f.read()", S, exists=gone)["outcome"], "block")

    def test_written_earlier_in_cell(self):
        code = H + "p.to_csv('out.csv')\nq = pd.read_csv('out.csv')"
        self.assertEqual(decide(code, S, exists=gone)["outcome"], "unknown")
        code = "with open('out.txt', 'w') as f:\n    f.write('x')\nt = open('out.txt').read()"
        self.assertNotEqual(decide(code, S, exists=gone)["outcome"], "block")

    def test_conditional_read_is_possible(self):
        code = "import pandas as pd\ntry:\n    d = pd.read_csv('t.csv')\nexcept FileNotFoundError:\n    d = None"
        self.assertEqual(decide(code, S, exists=gone)["outcome"], "unknown")
        code = "import os, pandas as pd\nif os.path.exists('t.csv'):\n    d = pd.read_csv('t.csv')"
        self.assertEqual(decide(code, S, exists=gone)["outcome"], "unknown")

    def test_chdir_stops_relative_checks(self):
        code = "import os, pandas as pd\nos.chdir('/x')\nd = pd.read_csv('t.csv')"
        only_x = lambda p: p == "/x"  # noqa: E731  (gate v8: os.chdir itself needs /x)
        self.assertNotEqual(decide(code, S, exists=only_x)["outcome"], "block")

    def test_files_are_listed(self):
        r = decide(H, S)
        self.assertEqual(r["files"], [("payments.csv", False, 2)])


class Aliases(unittest.TestCase):
    """alias=True (DataAgentBench): x = y keeps tracking both names."""
    R = "import json\nrecs = json.load(open('merchant_data.json'))\n"

    def test_off_by_default(self):
        self.assertNotEqual(decide(H + "q = p\nq['date']", S)["outcome"], "block")

    def test_alias_of_frame_is_checked(self):
        r = decide(H + "q = p\nq['date']", S, alias=True)
        self.assertEqual((r["outcome"], r["missing"]), ("block", "date"))
        self.assertEqual(decide(H + "q = p\np['date']", S, alias=True)["outcome"], "block")

    def test_alias_of_records_is_checked(self):
        r = decide(self.R + "rs = recs\nfor r in rs:\n    r['mcc']", S, alias=True)
        self.assertEqual((r["outcome"], r["missing"]), ("block", "mcc"))

    def test_change_through_alias_stops_both(self):
        for change in ("q.pop('merchant')", "f(q)", "q.drop(columns=['merchant'], inplace=True)", "z = [q]"):
            code = H + "q = p\n" + change + "\np['date']"
            self.assertNotEqual(decide(code, S, alias=True)["outcome"], "block", change)
            code = H + "q = p\n" + change.replace("q", "p", 1) + "\nq['date']"
            self.assertNotEqual(decide(code, S, alias=True)["outcome"], "block", change)

    def test_new_column_through_alias_is_seen(self):
        self.assertEqual(decide(H + "q = p\nq['date'] = 1\np['date']", S, alias=True)["outcome"], "run")

    def test_conditional_alias_stops_tracking(self):
        code = H + "if c:\n    q = p\np['date']"
        self.assertNotEqual(decide(code, S, alias=True)["outcome"], "block")

    def test_aliases_are_not_exported(self):
        st = GateState()
        decide(H + "q = p", S, st, alias=True)
        st.commit(failed=False)
        self.assertNotIn("q", st.frames)
        self.assertNotIn("p", st.frames)


class LoopTargets(unittest.TestCase):
    """Found by the DataAgentBench census: a loop over a display of frames changes them."""

    def test_loop_over_tuple_of_frames(self):
        code = H + "q = pd.read_csv('payments.csv')\nfor d in (p, q):\n    d['date'] = 1\np['date']"
        self.assertNotEqual(decide(code, S)["outcome"], "block")
        code = H + "for i, d in enumerate([p]):\n    d['date'] = 1\np['date']"
        self.assertNotEqual(decide(code, S)["outcome"], "block")

    def test_comprehension_over_display(self):
        code = H + "[d.insert(0, 'date', 1) for d in (p,)]\np['date']"
        self.assertNotEqual(decide(code, S)["outcome"], "block")

    def test_loop_over_rows_still_checked(self):
        code = H + "for _, row in p.iterrows():\n    row['date']"
        self.assertEqual(decide(code, S)["outcome"], "block")


class SystemExitRaise(unittest.TestCase):
    def test_conditional_raise_systemexit_makes_rest_conditional(self):
        for r in ("raise SystemExit", "raise SystemExit(0)", "raise SystemExit()"):
            code = H + f"if c:\n    {r}\np['date']"
            self.assertEqual(decide(code, S)["outcome"], "unknown", r)

    def test_other_conditional_raise_keeps_block(self):
        code = H + "if c:\n    raise ValueError('x')\np['date']"
        self.assertEqual(decide(code, S)["outcome"], "block")


class MethodArguments(unittest.TestCase):
    """Gate v6, fix 1: column names that pandas methods require."""

    def blocks(self, tail, missing="date"):
        r = decide(H + tail, S)
        return r["outcome"] == "block" and r["missing"] == missing

    def test_methods_block(self):
        for tail in ("p.groupby('date').size()", "p.groupby(['merchant', 'date']).size()",
                     "p.sort_values('date')", "p.sort_values(by=['year', 'date'])",
                     "p.drop_duplicates(subset=['date'])", "p.dropna(subset=['date'])",
                     "p.set_index('date')", "p.pivot_table(index='date', values='eur_amount')",
                     "p.drop(columns=['date'])", "p.drop('date', axis=1)",
                     "p.loc[p['year'] > 0, 'date']", "p.loc[:, ['merchant', 'date']]", "p.at[0, 'date']"):
            self.assertTrue(self.blocks(tail), tail)

    def test_present_columns_run(self):
        for tail in ("p.groupby('merchant').size()", "p.sort_values('year')", "p.loc[p['year'] > 0, 'merchant']",
                     "p.drop(columns=['merchant'])"):
            self.assertEqual(decide(H + tail, S)["outcome"], "run", tail)

    def test_not_checked(self):
        for tail in ("p.dropna(axis=1, subset=[0])", "p.drop(columns=['date'], errors='ignore')",
                     "p.drop('date')", "p.groupby(level=0).size()", "p.rename(columns={'date': 'd'})",
                     "p.loc['date']", "p.loc[:, 'a':'date']", "p.sort_values('date', axis=1)",
                     "p.groupby(*cols)"):
            self.assertNotEqual(decide(H + tail, S)["outcome"], "block", tail)

    def test_merge(self):
        r = "q = pd.read_csv('payments.csv')\n"
        self.assertTrue(self.blocks("p.merge(other, on='date')"))                 # left side tracked
        self.assertTrue(self.blocks(r + "p.merge(q, left_on='merchant', right_on='date')"))
        self.assertTrue(self.blocks(r + "pd.merge(p, q, on='date')"))
        self.assertEqual(decide(H + r + "p.merge(q, on='merchant')", S)["outcome"], "run")
        self.assertNotEqual(decide(H + "other.merge(p, left_on='date', right_on='merchant')", S)["outcome"], "block")

    def test_conditional_method_is_possible(self):
        self.assertEqual(decide(H + "if c:\n    p.groupby('date')", S)["outcome"], "unknown")


class DerivedFrames(unittest.TestCase):
    """Gate v6, fix 2: columns of derived frames."""
    Q = "q = pd.read_csv('payments.csv')\n"

    def out(self, tail):
        return decide(H + self.Q + tail, S)["outcome"]

    def test_blocks(self):
        for tail in ("f = p[p['year'] > 0]\nf['date']", "p[p['year'] > 0]['date']", "p[(p.year > 0) & ~p.merchant.isna()]['date']",
                     "f = p[['merchant', 'year']]\nf['eur_amount']", "p.loc[p['year'] > 0]['date']",
                     "p.sort_values('year').head()['date']", "p.copy()['date']", "p.query('year > 0')['date']",
                     "g = p.groupby('merchant')['eur_amount'].sum().reset_index()\ng['year']",
                     "g = p.groupby('merchant', as_index=False).agg(total=('eur_amount', 'sum'))\ng['eur_amount']",
                     "g = p.groupby(['merchant', 'year']).size().reset_index(name='n')\ng['eur_amount']",
                     "r = p.rename(columns={'year': 'y'})\nr['year']", "d = p.drop(columns=['year'])\nd['year']",
                     "c = pd.concat([p, q])\nc['date']", "t = p.reset_index(drop=True)\nt['index']",
                     "p.groupby('merchant')['date']", "p[p.year > 0].groupby('date').size()",
                     "m = p.merge(q, on='merchant')\nm['date']"):
            self.assertEqual(self.out(tail), "block", tail)

    def test_runs(self):
        for tail in ("g = p.groupby('merchant', as_index=False).agg(total=('eur_amount', 'sum'))\ng['total']",
                     "g = p.groupby('merchant')['eur_amount'].sum().reset_index()\ng['eur_amount']",
                     "g = p.groupby('merchant')['eur_amount'].sum().reset_index(name='s')\ng['s']",
                     "r = p.rename(columns={'year': 'y'})\nr['y']", "a = p.assign(z=1)\na['z']",
                     "t = p.reset_index()\nt['index']", "h = p.head()\nh['date'] = 1\nh['date']"):
            self.assertEqual(self.out(tail), "run", tail)

    def test_merge_keeps_both_spellings(self):
        # year is in both frames: pandas names it year_x / year_y; the gate keeps year too (never a false block)
        for tail in ("m = p.merge(q, on='merchant')\nm['year']", "m = p.merge(q, on='merchant')\nm['year_x']",
                     "m = p.merge(q)\nm['year_x']", "m = p.merge(q, on='merchant', suffixes=('_l', '_r'))\nm['year_l']"):
            self.assertNotEqual(self.out(tail), "block", tail)

    def test_not_tracked(self):
        for tail in ("f = p[mask]\nf['date']", "x = p.dropna(inplace=True)\nx['date']",
                     "g = p.groupby('merchant').sum()\ng['date']", "f = p.set_index('year')\nf['date']",
                     "p.merge(q, left_on='year', right_on='year')['date']", "p.iloc[:, [0, 1]]['date']"):
            self.assertNotEqual(self.out(tail), "block", tail)

    def test_filtered_rows_may_be_empty(self):
        self.assertEqual(self.out("for _, row in p[p.year > 0].iterrows():\n    row['date']"), "unknown")


class HelperFunctions(unittest.TestCase):
    """Gate v6, fix 4: pure loader functions."""
    LOAD = ("import json, os\nimport pandas as pd\n"
            "def load(v):\n    if isinstance(v, str) and v.endswith('.json'):\n"
            "        with open(v, 'r', encoding='utf-8') as f:\n            return json.load(f)\n    return v\n")

    def test_loader_gives_records(self):
        r = decide(self.LOAD + "recs = load('merchant_data.json')\nfor x in recs:\n    x['mcc']", S)
        self.assertEqual((r["outcome"], r["missing"]), ("block", "mcc"))
        r = decide(self.LOAD + "df = pd.DataFrame(load('merchant_data.json'))\ndf['mcc']", S)
        self.assertEqual((r["outcome"], r["missing"]), ("block", "mcc"))
        self.assertEqual(decide(self.LOAD + "for x in load('merchant_data.json'):\n    x['mcc']", S)["outcome"], "block")

    def test_loader_checks_the_file(self):
        gone = lambda p: False  # noqa: E731
        r = decide(self.LOAD + "recs = load('missing.json')", S, exists=gone)
        self.assertEqual((r["outcome"], r["kind"]), ("block", "file"))

    def test_identity_branch_needs_alias_mode(self):
        base = "import json\nrecs = json.load(open('merchant_data.json'))\n"
        code = self.LOAD.replace("import json, os\n", "import json, os\n" + base) + "r2 = load(recs)\nfor x in r2:\n    x['mcc']"
        self.assertEqual(decide(code, S, alias=True)["outcome"], "block")
        self.assertNotEqual(decide(code, S)["outcome"], "block")
        # without alias mode, the argument escapes through the result: stop tracking it
        code2 = self.LOAD.replace("import json, os\n", "import json, os\n" + base) + "r2 = load(recs)\nr2.append({})\nfor x in recs:\n    x['mcc']"
        self.assertNotEqual(decide(code2, S)["outcome"], "block")

    def test_impure_or_unknown_helpers_are_not_followed(self):
        for body in ("def load(v):\n    data = json.load(open(v))\n    data.append({'mcc': 1})\n    return data\n",
                     "def load(v):\n    print(v)\n    return json.load(open(v))\n",
                     "def load(v):\n    with open(v, 'w') as f:\n        return json.load(f)\n",
                     "def load(v, w):\n    return json.load(open(v))\n",
                     "def load(v):\n    if cond:\n        return json.load(open(v))\n    return v\n"):
            code = "import json\n" + body + "for x in load('merchant_data.json'):\n    x['mcc']"
            self.assertNotEqual(decide(code, S)["outcome"], "block", body)

    def test_rebinding_the_name_forgets_the_helper(self):
        code = self.LOAD + "load = other\nfor x in load('merchant_data.json'):\n    x['mcc']"
        self.assertNotEqual(decide(code, S)["outcome"], "block")

    def test_exists_guard(self):
        code = ("import json, os\ndef load(p):\n    if os.path.exists(p):\n        return json.load(open(p))\n    return []\n"
                "for x in load('merchant_data.json'):\n    x['mcc']")
        self.assertEqual(decide(code, S, exists=lambda p: True)["outcome"], "block")
        self.assertNotEqual(decide(code, S)["outcome"], "block")


class ReaderWhitelist(unittest.TestCase):
    """Gate v7, rule 2: read_csv / read_json / csv.DictReader arguments."""
    RC = "import pandas as pd\np = pd.read_csv('payments.csv'{})\np['month']"

    def out(self, args):
        return decide(self.RC.format(args), S)["outcome"]

    def test_arguments_that_keep_the_names(self):
        for args in ("", ", low_memory=False", ", dtype={'year': int}", ", nrows=5", ", parse_dates=['year']",
                     ", parse_dates=True", ", header=0", ", header='infer'", ", sep=','", ", delimiter=','",
                     ", index_col=None", ", index_col=False", ", encoding='utf-8'", ", encoding='UTF8'",
                     ", encoding='utf-8-sig'", ", encoding=None", ", skip_blank_lines=True", ", engine='python'",
                     ", na_values=['NA']", ", thousands=','", ", on_bad_lines='skip'", ", filepath_or_buffer=None"):
            if "filepath" in args:
                continue
            self.assertEqual(self.out(args), "block", args)
        r = decide("import pandas as pd\np = pd.read_csv(filepath_or_buffer='payments.csv', low_memory=False)\n"
                   "p['month']", S)
        self.assertEqual(r["outcome"], "block")

    def test_arguments_that_may_change_the_names(self):
        for args in (", skiprows=1", ", header=1", ", header=[0, 1]", ", header=False", ", sep=';'", ", sep='\\t'",
                     ", delimiter='|'", ", sep=None", ", index_col=0", ", index_col='year'", ", encoding='latin-1'",
                     ", skip_blank_lines=False", ", parse_dates={'d': ['year', 'day_of_year']}",
                     ", parse_dates=[['year', 'day_of_year']]", ", comment='#'", ", skipinitialspace=True",
                     ", quotechar=\"'\"", ", **kw", ", ';'", ", sep=s", ", encoding=enc", ", prefix='c'"):
            r = decide(self.RC.format(args), S)
            self.assertEqual(r["outcome"], "unknown", args)

    def test_usecols_with_an_unknown_argument_is_not_checked(self):
        r = decide("import pandas as pd\np = pd.read_csv('payments.csv', sep=';', usecols=['month'])", S)
        self.assertEqual(r["outcome"], "unknown")
        r = decide("import pandas as pd\np = pd.read_csv('payments.csv', low_memory=False, usecols=['month'])", S)
        self.assertEqual((r["outcome"], r["missing"]), ("block", "month"))

    def test_read_json(self):
        base = "import pandas as pd\nm = pd.read_json('merchant_data.json'{})\nm['mcc']"
        self.assertEqual(decide(base.format(""), S)["outcome"], "block")
        self.assertEqual(decide(base.format(", dtype=False"), S)["outcome"], "block")
        self.assertEqual(decide(base.format(", encoding='utf-8'"), S)["outcome"], "block")
        for args in (", orient='records'", ", lines=True", ", typ='series'", ", encoding='latin-1'", ", nrows=3"):
            self.assertEqual(decide(base.format(args), S)["outcome"], "unknown", args)

    def test_chunks(self):
        base = "import pandas as pd\nfor c in pd.read_csv('payments.csv', chunksize=10{}):\n    c['month']"
        self.assertEqual(decide(base.format(""), S)["outcome"], "block")
        for args in (", sep=';'", ", header=None", ", names=['a', 'month']", ", skiprows=2"):
            self.assertEqual(decide(base.format(args), S)["outcome"], "unknown", args)

    def test_dictreader(self):
        base = "import csv\nwith open('payments.csv') as f:\n    for r in csv.DictReader(f{}):\n        r['month']"
        self.assertEqual(decide(base.format(""), S)["outcome"], "block")
        for args in (", delimiter=';'", ", fieldnames=['month']", ", ['month']", ", skipinitialspace=True"):
            self.assertEqual(decide(base.format(args), S)["outcome"], "unknown", args)
        rows = "import csv\nwith open('payments.csv') as f:\n    rd = csv.DictReader(f{})\n    rows = list(rd)\nrows[0]['month']"
        self.assertEqual(decide(rows.format(""), S)["outcome"], "block")
        self.assertEqual(decide(rows.format(", delimiter=';'"), S)["outcome"], "unknown")

    def test_helper_loader_checks_the_encoding(self):
        code = ("import pandas as pd\ndef load(p):\n    return pd.read_csv(p, encoding='{}')\n"
                "df = load('payments.csv')\ndf['month']")
        self.assertEqual(decide(code.format("utf-8"), S)["outcome"], "block")
        self.assertNotEqual(decide(code.format("latin-1"), S)["outcome"], "block")

    def test_run_needs_every_read_tracked(self):
        code = ("import pandas as pd\np = pd.read_csv('payments.csv')\nq = pd.read_csv('payments.csv', sep=';')\n"
                "p['year']")
        self.assertEqual(decide(code, S)["outcome"], "unknown")


class CsvSchema(unittest.TestCase):
    """Gate v7, rule 1: the union of the pandas and csv.reader headers."""

    def cols(self, text, encoding="utf-8"):
        import tempfile
        from gate import csv_columns
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as f:
            f.write(text.encode(encoding))
        try:
            return csv_columns(f.name)
        finally:
            os.unlink(f.name)

    def test_plain_header(self):
        self.assertEqual(self.cols("a,b\n1,2\n"), {"a", "b"})

    def test_empty_field_and_repeated_name(self):
        self.assertEqual(self.cols(",a,a\n0,1,2\n"), {"", "Unnamed: 0", "a", "a.1"})

    def test_byte_order_mark(self):
        c = self.cols("\ufeffid,x\n1,2\n")
        self.assertIn("id", c)
        self.assertIn("\ufeffid", c)                   # csv.DictReader keeps the mark

    def test_leading_blank_line(self):
        self.assertIn("a", self.cols("\na,b\n1,2\n"))

    def test_unreadable_file_is_left_out(self):
        self.assertIsNone(self.cols(""))
        self.assertIsNone(self.cols("a,b\n\xe9,1\n", encoding="latin-1"))


class GateV8(unittest.TestCase):
    """Gate v8: R1 directory calls, R2 names bound to literals as keys, R3 file locations."""
    G = staticmethod(lambda p: False)  # nothing exists

    def test_r1_listdir_scandir_chdir(self):
        for call in ("os.listdir('/mnt/data')", "os.scandir('/mnt/data')", "os.chdir('/mnt/data')",
                     "os.listdir(path='/mnt/data')"):
            r = decide("import os\n" + call, S, exists=self.G)
            self.assertEqual((r["outcome"], r["kind"], r["source"]), ("block", "file", "/mnt/data"), call)
            self.assertIn("directory", r["reason"])

    def test_r1_not_checked(self):
        cases = ["import os\nos.listdir()",                                   # the working directory
                 "import os\nos.walk('/mnt/data')",                           # raises nothing
                 "import glob\nglob.glob('/mnt/data/*')",
                 "from pathlib import Path\nPath('/mnt/data').iterdir()",     # raises only when consumed
                 "import os\nos.makedirs('out')\nos.listdir('out')",         # a cell that creates directories
                 "from pathlib import Path\nPath('out').mkdir()\nos.listdir('out')",
                 "import os\nif os.path.isdir('/d'):\n    os.listdir('/d')",  # conditional
                 "import os\ntry:\n    os.listdir('/d')\nexcept FileNotFoundError:\n    pass",
                 "from os import listdir\nlistdir('/d')",                     # bare name: not os.listdir
                 "import os\nd = '/a' if x else '/b'\nos.listdir(d)"]         # unresolved path
        for code in cases:
            self.assertNotEqual(decide(code, S, exists=self.G)["outcome"], "block", code)
        self.assertNotEqual(decide("import os\nos.listdir('/d')", S)["outcome"], "block")   # no exists
        self.assertNotEqual(decide("import os\nos.listdir('/d')", S, exists=lambda p: None)["outcome"], "block")
        self.assertNotEqual(decide("import os\nos.listdir('/d')", S, exists=lambda p: True)["outcome"], "block")

    def test_r1_written_file_in_directory(self):
        code = "import pandas as pd\np = pd.read_csv('payments.csv')\np.to_csv('out/a.csv')\nimport os\nos.listdir('out')"
        self.assertNotEqual(decide(code, S, exists=lambda p: p == "payments.csv")["outcome"], "block")

    def test_r2_string_name(self):
        r = decide(H + "col = 'month'\np[col]", S)
        self.assertEqual((r["outcome"], r["missing"]), ("block", "month"))
        self.assertEqual(decide(H + "col = 'year'\np[col]", S)["outcome"], "run")
        st = GateState()
        decide(H + "col = 'month'", S, st)
        st.commit(failed=False)
        self.assertEqual(decide("p[col]", S, st)["outcome"], "block")      # carried to the next cell

    def test_r2_string_name_not_resolved(self):
        for tail in ("col = 'month'\ncol = other\np[col]",
                     "col = 'month'\nfor col in cols:\n    p[col]",
                     "if x:\n    col = 'month'\np[col]",
                     "col = 'month'\ncol += 'x'\np[col]",
                     "col = 'month'\n[p[col] for col in p.columns]",
                     "col = 'month'\ndef f(col):\n    return p[col]",
                     "col = 'month'\n(col := 'year')\np[col]",
                     "col = 'month'\nexec('col = 1')\np[col]"):
            self.assertNotEqual(decide(H + tail, S)["outcome"], "block", tail)

    def test_r2_list_name(self):
        r = decide(H + "cols = ['year', 'month']\nx = p[cols]", S)
        self.assertEqual((r["outcome"], r["missing"]), ("block", "month"))
        r = decide(H + "cols = ['year', 'month']\nprint(len(cols))\nfor c in cols:\n    pass\n"
                       "if 'x' in cols:\n    pass\np[cols]", S)
        self.assertEqual(r["outcome"], "block")
        self.assertEqual(decide(H + "cols = ['year', 'merchant']\np[cols]", S)["outcome"], "run")

    def test_r2_list_name_not_resolved(self):
        for tail in ("cols = ['year', 'month']\ncols.append('x')\np[cols]",
                     "cols = ['year', 'month']\ncols.remove('month')\np[cols]",
                     "cols = ['year', 'month']\nf(cols)\np[cols]",
                     "cols = ['year', 'month']\nc2 = cols\nc2.pop()\np[cols]",
                     "cols = ['year', 'month']\ncols[1] = 'year'\np[cols]",
                     "cols = ['year', 'month']\ncols += []\np[cols]",
                     "cols = ['year', 'month']\ncols = ['year']\np[cols]",
                     "if x:\n    cols = ['month']\np[cols]",
                     "cols = ['year', 'month']\ndel cols[1]\np[cols]",
                     "cols = ['year', x]\np[cols]"):
            self.assertNotEqual(decide(H + tail, S)["outcome"], "block", tail)
        st = GateState()
        decide(H + "cols = ['month']", S, st)
        st.commit(failed=False)
        self.assertNotEqual(decide("p[cols]", S, st)["outcome"], "block")   # lists are not carried

    def test_r2_record_keys(self):
        code = "import json\nrecs = json.load(open('fees.json'))\nk = 'fee'\nfor r in recs:\n    r[k]"
        self.assertEqual(decide(code, S)["outcome"], "block")

    def test_r3_found_in(self):
        loc = lambda b: ["input/payments.csv"] if b == "payments.csv" else []  # noqa: E731
        code = "import pandas as pd\np = pd.read_csv('/mnt/data/payments.csv')"
        r = decide(code, S, exists=self.G, locate=loc)
        self.assertEqual((r["outcome"], r["found_in"]), ("block", ["input/payments.csv"]))
        r = decide("import os\nos.listdir('/mnt/data')", S, exists=self.G, locate=loc)
        self.assertEqual(r["available"], ["input"])
        self.assertEqual(decide(code, S, exists=self.G)["found_in"], [])        # without locate: unchanged


if __name__ == "__main__":
    unittest.main()
