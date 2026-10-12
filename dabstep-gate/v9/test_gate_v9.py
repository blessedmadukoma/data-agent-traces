"""Unit tests for release 4 (checker version 9).  Run from this folder:  python3 -m unittest test_gate_v9.py

1. Every test of version 8 (../test_gate.py) runs against version 9. Two of them change on
   purpose and are replaced: a `with` block of an unknown context manager may not run to its
   end (C3), and a group-by list or a pivot_table index may be row labels (N1).
2. RUN_CASES: programs that run without an error in pandas and that version 8 wrongly blocks.
   Each runs in a fresh copy of the data of differential_test.py; version 9 must not block it,
   and the program must run.
3. FAIL_CASES: programs that fail on a missing name or file; version 9 must still block them.
4. 400 programs from the generator of differential_test.py (seed 11, not a development seed):
   version 9 blocks none that runs.
"""
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate  # noqa: E402  version 9
import differential_test as dt  # noqa: E402

sys.modules["gate"] = gate
_spec = importlib.util.spec_from_file_location("v8tests", os.path.join(HERE, "..", "test_gate.py"))
v8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v8)

for _name in dir(v8):
    _obj = getattr(v8, _name)
    if isinstance(_obj, type) and issubclass(_obj, unittest.TestCase) and _obj is not unittest.TestCase:
        globals()["V8_" + _name] = _obj


class V8_ControlFlow(v8.ControlFlow):                  # noqa: N801  replaces the version 8 class
    def test_with_body_runs(self):
        self.assertEqual(v8.out(v8.H + "with open('notes.md') as fh:\n    p['date']"), "block")
        self.assertEqual(v8.out(v8.H + "with warnings.catch_warnings():\n    p['date']"), "block")
        self.assertEqual(v8.out(v8.H + "with ctx():\n    p['date']"), "unknown")                       # C3
        self.assertEqual(v8.out(v8.H + "with contextlib.suppress(KeyError):\n    p['date']"), "unknown")


class V8_MethodArguments(v8.MethodArguments):          # noqa: N801
    def test_methods_block(self):
        for tail in ("p.groupby('date').size()", "p.groupby(('merchant', 'date')).size()",
                     "p.sort_values('date')", "p.sort_values(by=['year', 'date'])",
                     "p.drop_duplicates(subset=['date'])", "p.dropna(subset=['date'])",
                     "p.set_index('date')", "p.pivot_table(index='merchant', values='date')",
                     "p.drop(columns=['date'])", "p.drop('date', axis=1)",
                     "p.loc[p['year'] > 0, 'date']", "p.loc[:, ['merchant', 'date']]", "p.at[0, 'date']"):
            self.assertTrue(self.blocks(tail), tail)
        for tail in ("p.groupby(['merchant', 'date']).size()",                                          # N1
                     "p.pivot_table(index='date', values='eur_amount')",
                     "p.pivot_table(index='merchant', columns='date', aggfunc='size')",
                     "p[p.year > 0].drop_duplicates(subset=['date'])"):                                   # N2
            self.assertEqual(gate.decide(v8.H + tail, v8.S)["outcome"], "unknown", tail)


def _schema():
    return dt.schema(None)


def case(steps, run=True):
    """Decide each step with version 9 in a fresh copy of the test data, then run the program.
    Returns (decisions, outcomes)."""
    if isinstance(steps, str):
        steps = [steps]
    work = tempfile.mkdtemp(prefix="v9case_")
    try:
        dt.make_data(work)
        outs = []
        if run:
            runner = os.path.join(work, "_run.py")
            with open(runner, "w") as fh:
                fh.write("import json, sys, os\n"
                         "class _End(BaseException):\n    pass\n"
                         "def final_answer(x=None):\n    raise _End()\n"
                         f"g = {{'__name__': '__main__', 'final_answer': final_answer}}\nexec({dt.PRELUDE!r}, g)\n"
                         "res = []\n"
                         "snap0 = sorted(os.listdir('.'))\n"
                         f"for code in {steps!r}:\n"
                         "    snap = []\n"
                         "    for d, _, fs in os.walk('.'):\n"
                         "        snap += [os.path.normpath(os.path.join(d, f)) for f in fs]\n"
                         "        if d != '.':\n            snap.append(os.path.normpath(d) + '/')\n"
                         "    try:\n        exec(compile(code, '<step>', 'exec'), g)\n        res.append(['ok', '', snap])\n"
                         "    except _End:\n        res.append(['end', '', snap])\n"
                         "    except SystemExit:\n        res.append(['end', '', snap])\n"
                         "    except BaseException as e:\n        res.append(['error', type(e).__name__, snap])\n"
                         "sys.stdout = sys.__stdout__\nprint('@@' + json.dumps(res))\n")
            env = dict(os.environ, HOME=work)
            p = subprocess.run([sys.executable, "-I", runner], cwd=work, capture_output=True, text=True,
                               timeout=60, env=env)
            line = next(x for x in p.stdout.splitlines()[::-1] if x.startswith("@@"))
            outs = json.loads(line[2:])
        decisions = dt.decide_program(gate, steps, outs or [["ok", "", None]] * len(steps), _schema())
        return [d["outcome"] for d in decisions], [o[0] for o in outs]
    finally:
        shutil.rmtree(work, ignore_errors=True)


H = "df = pd.read_csv('sales.csv')\n"
H2 = "df = pd.read_csv('two.csv')\n"
H1 = "df = pd.read_csv('one.csv')\n"
R = "recs = json.load(open('fees.json'))\n"

# Programs that run without an error and that version 8 blocks. The first three are the
# reader's programs; the rest come from the differential test (one per construct).
RUN_CASES = [
    # the reader's programs
    R + "selected = [r for r in recs if False if r['absent']]",
    H2 + "c = pd.concat([df, df], axis=1, keys=['k1', 'k2'])\nv = c['k1']",
    H2 + "v = df.groupby(['merchant', 'absent']).size()",
    # C1 chained comparisons
    H + "v = (1 > 2 > len(df['absent']))",
    H + "if 0 > 1 < len(df['absent']):\n    pass",
    # C2 later conditions of a comprehension
    R + "v = [r for r in recs if r['card'] is None if r['absent']]",
    # C3 context managers that suppress errors
    H + "with contextlib.suppress(KeyError):\n    v = df['absent']",
    H + "from contextlib import suppress\nwith suppress(Exception):\n    v = df['absent']",
    H + "class S:\n    def __enter__(self):\n        return self\n    def __exit__(self, *a):\n        return True\n"
        "with S():\n    v = df['absent']",
    H + "@contextlib.contextmanager\ndef cm():\n    try:\n        yield\n    except Exception:\n        pass\n"
        "with cm():\n    v = df['absent']",
    H + "with contextlib.ExitStack() as es:\n    es.enter_context(contextlib.suppress(KeyError))\n    v = df['absent']",
    # C4 functions of the trace that end the step, in this step or an earlier one
    [H + "def done(x):\n    final_answer(x)", "done(1)\nv = df['absent']"],
    H + "stop = final_answer\nstop(1)\nv = df['absent']",
    H + "stop = lambda: final_answer(1)\nstop()\nv = df['absent']",
    H + "from sys import exit as stop\nstop()\nv = df['absent']",
    H + "import sys as s\ns.exit()\nv = df['absent']",
    H + "class A:\n    def __init__(self):\n        final_answer(1)\nA()\nv = df['absent']",
    H + "class B:\n    def go(self):\n        final_answer(1)\nB().go()\nv = df['absent']",
    H + "print = final_answer\nprint(1)\nv = df['absent']",
    # C5 enders used as values, code the checker cannot see, rebound builtins
    H + "@final_answer\ndef g():\n    pass\nv = df['absent']",
    H + "list(map(final_answer, [1]))\nv = df['absent']",
    H + "print(final_answer(1), df['absent'])",
    H + "v = [final_answer(1), df['absent']]",
    H + "getattr(sys, 'exit')()\nv = df['absent']",
    H + "import builtins\nbuiltins.exec('final_answer(1)')\nv = df['absent']",
    H + "fa = {'f': final_answer}\nfa['f'](1)\nv = df['absent']",
    # N1 group-by lists and pivot_table labels that pandas reads as row labels
    H1 + "v = df.groupby(['absent']).size()",
    H1 + "v = df.pivot_table(index=['absent'], values='merchant', aggfunc='count')",
    "df = pd.read_csv('sub/inner.csv')\nv = df.pivot_table(index='merchant', columns=['size'], aggfunc='size')",
    # N2 drop_duplicates on an empty frame
    H + "e = df[df['merchant'] != df['merchant']]\nv = e.drop_duplicates(subset=['absent'])",
    # N3 concat with keys
    H + "t = pd.concat([df, df], keys=['k1', 'k2'], names=['idx', None]).reset_index()\nv = t['idx']",
    # N4 index names, also through another object that shares the index
    H + "df.index.name = 'idx'\nv = df.groupby('idx').size()",
    H + "ix = df.index\nix.name = 'idx'\nv = df.sort_values('idx')",
    H + "df.index.names = ['idx']\nv = df.merge(df, on='idx')",
    H + "df.index.set_names('idx', inplace=True)\nv = df.groupby('idx').size()",
    # A1 escapes of frames
    H + "L = df.loc\nL[:, 'x_new'] = 1\nv = df['x_new']",
    H + "print((d := df).shape)\nd['x_new'] = 1\nv = df['x_new']",
    H + "import operator\noperator.setitem(df, 'x_new', 1)\nv = df['x_new']",
    H + "df.__setitem__('x_new', 1)\nv = df['x_new']",
    H + "getattr(df, '__setitem__')('x_new', 1)\nv = df['x_new']",
    H + "set_col = df.__setitem__\nset_col('x_new', 1)\nv = df['x_new']",
    H + "class M:\n    def f(self, d):\n        d['x_new'] = 1\nM().f(df)\nv = df['x_new']",
    # A2 function bodies
    H + "def add():\n    d = df\n    d['x_new'] = 1\nadd()\nv = df['x_new']",
    H + "def add(d=df):\n    d['x_new'] = 1\nadd()\nv = df['x_new']",
    H + "def get():\n    return df\nget()['x_new'] = 1\nv = df['x_new']",
    # A1 escapes of records
    R + "x = [recs[0]]\nx[0]['x_new'] = 1\nv = recs[0]['x_new']",
    R + "def add(r):\n    r['x_new'] = 1\nadd(recs[0])\nv = recs[0]['x_new']",
    R + "def add():\n    for r in recs:\n        r['x_new'] = 1\nadd()\nv = recs[0]['x_new']",
    R + "class S:\n    def __init__(self, r):\n        self.r = r\ns = S(recs[0])\ns.r['x_new'] = 1\nv = recs[0]['x_new']",
    R + "m = {'r': recs[0]}\nm['r']['x_new'] = 1\nv = recs[0]['x_new']",
    R + "recs[0].setdefault('x_new', 1)\nv = recs[0]['x_new']",
    R + "recs[0].update({'x_new': 1})\nv = recs[0]['x_new']",
    R + "rc = copy.copy(recs)\nrc[0]['x_new'] = 1\nv = recs[0]['x_new']",
    R + "sel = [r for r in recs]\nsel[0]['x_new'] = 1\nv = recs[0]['x_new']",
    R + "first = next(r for r in recs)\nfirst['x_new'] = 1\nv = recs[0]['x_new']",
    R + "dict.update(recs[0], {'x_new': 1})\nv = recs[0]['x_new']",
    R + "recs2 = recs.copy()\nrecs2[0]['x_new'] = 1\nv = recs[0]['x_new']",
    R + "for r in recs[:1]:\n    r['x_new'] = 1\nv = recs[0]['x_new']",
    # F1 files that the step creates without a name the checker knows
    H + "df.to_string('new.txt')\nt = open('new.txt').read()",
    H + "df.to_html('new.html')\nt = open('new.html').read()",
    "np.savetxt('new.txt', np.zeros(2))\nt = open('new.txt').read()",
    "Path('new.txt').touch()\nt = open('new.txt').read()",
    "Path('new.txt').write_bytes(b'x')\nt = open('new.txt').read()",
    "name = 'new' + str(1) + '.txt'\nopen(name, 'w').write('x')\nt = open('new1.txt').read()",
    "with open(f'new{1}.txt', 'w') as fh:\n    fh.write('x')\nt = open('new1.txt').read()",
    "import shutil\nshutil.copy('sales.csv', 'copy.csv')\nd = pd.read_csv('copy.csv')",
    "os.rename('notes.md', 'moved.md')\nt = open('moved.md').read()",
    # F2 paths that are not local file names
    "d = pd.read_csv('~/sales.csv')",
    # F3 Path.open and a rebound open
    "with Path('new.txt').open('w') as fh:\n    fh.write('x')\nt = open('new.txt').read()",
    "def open(p, *a, **k):\n    return io.StringIO('x')\nt = open('missing.csv').read()",
    # F4 directory changes through aliases and in an earlier step
    "from os import chdir as cd\ncd('sub')\nd = pd.read_csv('inner.csv')",
    "import os as o\no.chdir('sub')\nd = pd.read_csv('inner.csv')",
    "from os import makedirs as md\nmd('newdir')\nv = os.listdir('newdir')",
    ["os.chdir('sub')", "d = pd.read_csv('inner.csv')"],
    # R1, R2 readers whose rows were used before the loop
    "fh = open('sales.csv')\nfh.readline()\nrd = csv.DictReader(fh)\nfor row in rd:\n    v = row['Crossfit']",
    "rd = csv.DictReader(open('one.csv'))\nnext(rd)\nfor row in rd:\n    v = row['absent']",
    "n0 = 0\nd = pd.read_csv('sales.csv', nrows=n0)\nfor _, row in d.iterrows():\n    v = row['absent']",
]
# programs that run without an error and that version 8 does not block either; version 9 must keep it so
SAFE_CASES = [
    H + "def done(x):\n    final_answer(x)\ndone(1)\nv = df['absent']",
    H + "def gen():\n    final_answer(1)\n    yield 1\nlist(gen())\nv = df['absent']",
    H + "exec('final_answer(1)')\nv = df['absent']",
    H + "eval('final_answer(1)')\nv = df['absent']",
    H + "class C:\n    final_answer(1)\nv = df['absent']",
    H + "def f(x=final_answer(1)):\n    pass\nv = df['absent']",
    H + "def any(x):\n    return True\nv = any(df['absent'] for _ in [1])",
    H + "list = lambda it: []\nv = list(df['absent'] for _ in [1])",
    H + "df.rename_axis('idx', inplace=True)\nv = df.groupby('idx').size()",
    H + "df.index = df.index.rename('idx')\nv = df.groupby('idx').size()",
    H + "box = [df]\nbox[0]['x_new'] = 1\nv = df['x_new']",
    H + "class H:\n    pass\nh = H()\nh.d = df\nh.d['x_new'] = 1\nv = df['x_new']",
    H + "store = {'a': df}\nstore['a']['x_new'] = 1\nv = df['x_new']",
    H + "pd.DataFrame.insert(df, 0, 'x_new', 1)\nv = df['x_new']",
    R + "pick = (lambda xs: xs[0])\npick(recs)['x_new'] = 1\nv = recs[0]['x_new']",
]
# programs that only some pandas versions run without an error (version 8 blocks them there)
VERSION_CASES = [
    "d = pd.read_json('[{\"a\": 1}]')",                                   # pandas 1.5 and 2.2
    H + "d2 = pd.DataFrame(df)\nd2['x_new'] = 1\nv = df['x_new']",          # pandas 1.5
    H + "b = df[['merchant']]\nb.index.name = 'idx'\nv = df.groupby('idx').size()",   # 1.5 and 2.2
    H + "s = df['merchant']\ns.index.name = 'idx'\nv = df.groupby('idx').size()",     # 1.5 and 2.2
]

FAIL_CASES = [
    H + "v = df['absent']",
    H + "v = df[['merchant', 'absent']]",
    H + "v = df.groupby('absent').size()",
    H + "v = df.sort_values(['merchant', 'absent'])",
    H + "v = df.drop(columns=['absent'])",
    H + "v = df.loc[:, ['absent']]",
    H + "final_answer(df['absent'].sum())",
    H + "if df['absent'].any() and True:\n    pass",
    H + "with open('notes.md') as fh:\n    v = df['absent']",
    R + "v = recs[0]['absent']",
    R + "for r in recs:\n    v = r['absent']",
    R + "v = [r for r in recs if r['absent']]",
    "rd = csv.DictReader(open('sales.csv'))\nfor row in rd:\n    v = row['absent']",
    "rows = list(csv.DictReader(open('sales.csv')))\nv = rows[0]['absent']",
    "d = pd.read_csv('missing.csv')",
    "t = open('missing.txt').read()",
    "v = os.listdir('missing_dir')",
    H + "t = df.groupby('merchant')['amount'].sum().reset_index()\nv = t['absent']",
    H + "def helper(x):\n    return x\nv = df['absent']",
]


class RunCases(unittest.TestCase):
    def test_reader_and_differential_cases_run_and_are_not_blocked(self):
        for steps in RUN_CASES:
            with self.subTest(steps=steps):
                dec, outs = case(steps)
                self.assertTrue(all(o in ("ok", "end") for o in outs[-1:]), (steps, outs))
                self.assertNotIn("block", dec, (steps, dec))

    def test_safe_cases_run_and_are_not_blocked(self):
        for steps in SAFE_CASES:
            with self.subTest(steps=steps):
                dec, outs = case(steps)
                self.assertTrue(all(o in ("ok", "end") for o in outs[-1:]), (steps, outs))
                self.assertNotIn("block", dec, (steps, dec))

    def test_version_cases_are_not_blocked(self):
        for steps in VERSION_CASES:
            with self.subTest(steps=steps):
                dec, _ = case(steps, run=False)
                self.assertNotIn("block", dec, (steps, dec))

    def test_version_8_blocks_them(self):
        """The cases are real gaps of version 8: it blocks each one (some only on other pandas versions)."""
        spec = importlib.util.spec_from_file_location("gate_v8", os.path.join(HERE, "..", "gate.py"))
        g8 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g8)
        missed = []
        for steps in RUN_CASES + VERSION_CASES:
            steps = [steps] if isinstance(steps, str) else steps
            work = tempfile.mkdtemp(prefix="v8case_")
            try:
                dt.make_data(work)
                snap = []
                for d, _, fs in os.walk(work):
                    snap += [os.path.normpath(os.path.relpath(os.path.join(d, f), work)) for f in fs]
                    if d != work:
                        snap.append(os.path.relpath(d, work) + "/")
                try:
                    dec = dt.decide_program(g8, steps, [["ok", "", snap]] * len(steps), _schema())
                except Exception:                            # noqa: BLE001  a crash of version 8 counts
                    dec = [{"outcome": "block"}]
                if not any(d["outcome"] in ("block", "crash") for d in dec):
                    missed.append(steps)
            finally:
                shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(missed, [])


class FailCases(unittest.TestCase):
    def test_still_blocked(self):
        for steps in FAIL_CASES:
            with self.subTest(steps=steps):
                dec, outs = case(steps)
                self.assertEqual(outs[-1], "error", steps)
                self.assertEqual(dec[-1], "block", steps)


class Generated(unittest.TestCase):
    def test_no_false_block_on_400_generated_programs(self):
        rng = random.Random(11)
        programs = [dt.program(rng)[0] for _ in range(400)]
        with tempfile.TemporaryDirectory() as work:
            data = os.path.join(work, "data")
            os.makedirs(data)
            dt.make_data(data)
            outs = dt.run_programs(sys.executable, programs, data, 10.0, work)
        false = []
        for steps, o in zip(programs, outs):
            dec = dt.decide_program(gate, steps, o, _schema())
            for d, (kind, _, _) in zip(dec, o):
                if d["outcome"] == "block" and kind in ("ok", "end"):
                    false.append(steps)
                self.assertNotEqual(d["outcome"], "crash", steps)
        self.assertEqual(false, [])


class State(unittest.TestCase):
    def test_facts_carry_across_steps(self):
        st = gate.GateState()
        gate.decide("import os as o\nfrom sys import exit as stop\ndef done(x):\n    final_answer(x)", {}, st)
        st.commit(failed=False)
        self.assertIn("done", st.enders)
        self.assertIn("stop", st.enders)
        self.assertEqual(st.aliases.get("o"), "os")
        r = gate.decide("df = pd.read_csv('sales.csv')\ndone(1)\ndf['absent']", _schema(), st)
        self.assertNotEqual(r["outcome"], "block")

    def test_mixed_type_names_do_not_crash(self):
        r = gate.decide(H + "t = df.groupby('merchant').size().reset_index()\nv = t['absent']", _schema())
        self.assertEqual(r["outcome"], "block")
        self.assertIn(0, r["available"])


if __name__ == "__main__":
    unittest.main()
