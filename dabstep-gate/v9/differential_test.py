#!/usr/bin/env python3
"""Differential test of the checker against pandas (release 4, version 9).

Generates random Python programs over a small set of data files, runs each
program with one or more Python interpreters (each with its own pandas
version), and compares the run with the checker's decision. A false block is
a program step that the checker blocks and that ran without an error (it ran
to its end, or it ended through final_answer or SystemExit).

Each program has one or two steps. The steps of a program run in order in
one namespace, as in a harness that keeps variables between steps; the
checker decides each step with a GateState carried from the step before.

Every program runs in a child process of its own (os.fork), in a fresh copy
of the data folder, with HOME set to that folder and a time limit, because
some programs crash pandas 1.5.

Usage:
  python3 differential_test.py --n 20000 --seed 7 \\
      --python python3 --python /path/pd22/bin/python --python /path/pd15/bin/python \\
      --gate-dir ../ --gate-dir . --out results.json

--gate-dir may be given more than once; each folder must hold a gate.py.
The program list depends only on --n and --seed, not on the interpreters.
"""
import argparse
import ast
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------- data files
# Every data file has at least one record: this is a condition of the checker.
SALES = [
    {"merchant": "Crossfit", "amount": 10.5, "card": "visa", "day": 1, "country": "NL"},
    {"merchant": "Golfclub", "amount": 20.0, "card": "mc", "day": 2, "country": "BE"},
    {"merchant": "Rafa", "amount": 30.25, "card": "visa", "day": 3, "country": "NL"},
    {"merchant": "Belles", "amount": 5.0, "card": "amex", "day": 4, "country": "IT"},
    {"merchant": "Martinis", "amount": 7.5, "card": "mc", "day": 5, "country": "NL"},
]
TWO = [{"merchant": "Crossfit", "amount": 1.0, "card": "visa"},
       {"merchant": "Rafa", "amount": 2.0, "card": "mc"}]
ONE = [{"merchant": "Golfclub", "amount": 3.0}]
INNER = [{"merchant": "Crossfit", "fee": 0.1}, {"merchant": "Rafa", "fee": 0.2}]
FEES = [{"ID": 1, "rate": 0.5, "card": "visa", "merchants": ["Crossfit"]},
        {"ID": 2, "rate": 0.7, "card": "mc", "merchants": []},
        {"ID": 3, "rate": 0.9, "card": "amex", "merchants": ["Rafa"], "note": "x"}]
ONEJ = [{"ID": 7, "rate": 0.1}]

CSV_FILES = {"sales.csv": SALES, "two.csv": TWO, "one.csv": ONE, "sub/inner.csv": INNER}
JSON_FILES = {"fees.json": FEES, "one.json": ONEJ}
COLS = {"sales.csv": ["merchant", "amount", "card", "day", "country"], "two.csv": ["merchant", "amount", "card"],
        "one.csv": ["merchant", "amount"], "sub/inner.csv": ["merchant", "fee"]}
KEYS = {"fees.json": ["ID", "rate", "card", "merchants", "note"], "one.json": ["ID", "rate"]}
ROWS = {"sales.csv": 5, "two.csv": 2, "one.csv": 1, "sub/inner.csv": 2}
ABSENT = ["absent", "month", "merchant_id", "Merchant", "k1", "idx", "total", "size", "index",
          "level_0", "fee", "rate", "Crossfit", "visa", "x_new"]
MISSING_FILES = ["missing.csv", "data/sales.csv", "/mnt/data/sales.csv", "sales.CSV", "fees2.json",
                 "sub/missing.csv"]

PRELUDE = ("import pandas as pd\nimport numpy as np\nimport json, csv, os, sys, io, contextlib, warnings, copy\n"
           "from pathlib import Path\n")


def make_data(folder):
    import csv as _csv
    for name, rows in CSV_FILES.items():
        p = os.path.join(folder, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for name, recs in JSON_FILES.items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
            json.dump(recs, fh)
    with open(os.path.join(folder, "notes.md"), "w", encoding="utf-8") as fh:
        fh.write("# notes\n")
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": 1}, fh)


def schema(gate):
    out = {}
    for name in CSV_FILES:
        out[os.path.basename(name)] = set(COLS[name])
    for name in JSON_FILES:
        out[name] = set(KEYS[name])
    return out


# ---------------------------------------------------------------- generator
class G:
    """Random choices for one program."""

    def __init__(self, rng):
        self.r = rng

    def p(self, prob):
        return self.r.random() < prob

    def ch(self, seq):
        return self.r.choice(seq)

    def name(self, present, p_absent=0.5):
        return repr(self.ch(ABSENT)) if self.p(p_absent) or not present else repr(self.ch(present))

    def csvfile(self):
        return self.ch(["sales.csv", "sales.csv", "two.csv", "one.csv", "sub/inner.csv"])


def load_frame(g, var="df"):
    f = g.csvfile()
    k = g.r.randrange(10)
    if k == 0:
        line = f"path = {f!r}\n{var} = pd.read_csv(path)"
    elif k == 1 and "/" not in f:
        line = f"{var} = pd.read_csv(os.path.join('.', {f!r}))"
    elif k == 2:
        line = f"{var} = pd.read_csv(Path({f!r}))"
    elif k == 3:
        line = f"{var} = pd.read_csv({f!r}, nrows={g.ch([0, 1, 2, 'n'])})"
        if "nrows=n" in line:
            line = f"n = {g.ch([0, 1, 3])}\n" + line
    elif k == 4:
        line = f"{var} = pd.read_csv({f!r}, usecols={g.ch([[c] for c in COLS[f]] + [COLS[f][:2]])!r})"
    else:
        line = f"{var} = pd.read_csv({f!r})"
    return line, f


def load_records(g, var="recs"):
    k = g.r.randrange(6)
    if k == 0:
        return f"with open('fees.json') as fh:\n    {var} = json.load(fh)", "fees.json"
    if k == 1:
        return f"{var} = json.load(open('one.json'))", "one.json"
    if k == 2:
        f = g.ch(["sales.csv", "two.csv", "one.csv"])
        return f"{var} = list(csv.DictReader(open({f!r})))", f
    if k == 3:
        f = g.ch(["sales.csv", "two.csv"])
        return f"{var} = pd.read_csv({f!r}).to_dict('records')", f
    return f"{var} = json.load(open('fees.json'))", "fees.json"


def names_of(f):
    return COLS.get(f) or KEYS.get(f) or []


# Each template returns code that uses df (a frame of file f) or recs (records of file f).
def t_frame_reads(g, f):
    P, X = names_of(f), g.name(names_of(f))
    Y = g.name(names_of(f))
    k = g.r.randrange(16)
    return [
        f"v = df[{X}]",
        f"v = df[[{Y}, {X}]]",
        f"v = df.loc[:, {X}]",
        f"v = df.loc[df[{Y}].notna(), [{X}]]",
        f"v = df.at[0, {X}]",
        f"v = df[df[{X}] == df[{X}]]",
        f"v = df.groupby({X}).size()",
        f"v = df.groupby([{Y}, {X}]).size()",
        f"v = df.sort_values([{Y}, {X}])",
        f"v = df.drop_duplicates(subset=[{X}])",
        f"e = df[df[{repr(P[0])}] != df[{repr(P[0])}]]\nv = e.drop_duplicates(subset=[{X}])",
        f"e = df.head(0)\nv = e.dropna(subset=[{X}])",
        f"v = df.set_index([{Y}, {X}])",
        f"v = df.drop(columns=[{X}])",
        f"v = df.pivot_table(index=[{X}], values={Y}, aggfunc='count')",
        f"v = df.pivot_table(index={repr(P[0])}, columns=[{X}], aggfunc='size')",
    ][k], "frame_reads:" + str(k)


def t_derived(g, f):
    P = names_of(f)
    A, B = repr(P[0]), repr(P[-1])
    X, N = g.name(P), repr(g.ch(ABSENT + P))
    k = g.r.randrange(16)
    return [
        f"t = df.groupby({A})[{B}].sum().reset_index()\nv = t[{X}]",
        f"t = df.groupby({A}).size().reset_index(name={N})\nv = t[{X}]",
        f"t = df.groupby({A}, as_index=False).size()\nv = t[{X}]",
        f"t = df.groupby({A}).size().reset_index()\nv = t[{X}]",
        f"t = df.groupby({A}).agg(total=({B}, 'sum')).reset_index()\nv = t[{X}]",
        f"t = df.groupby({A}, as_index=False).agg({{{B}: 'sum'}})\nv = t[{X}]",
        f"t = df.groupby({A}, as_index=False)[{B}].count()\nv = t[{X}]",
        f"t = df.rename(columns={{{A}: {N}}})\nv = t[{X}]",
        f"t = df.assign(x_new=1)\nv = t[{X}]",
        f"t = df.reset_index()\nv = t[{X}]",
        f"t = df.merge(df, on={A})\nv = t[{X}]",
        f"t = pd.merge(df, df, on={A}, suffixes=('_l', '_r'))\nv = t[{X}]",
        f"t = pd.concat([df, df])\nv = t[{X}]",
        f"t = pd.concat([df, df], axis=1, keys=['k1', 'k2'])\nv = t[{X}]",
        f"t = pd.concat([df, df], keys=['k1', 'k2'], names=['idx', None]).reset_index()\nv = t[{X}]",
        f"t = df[[{A}]].copy()\nv = t[{X}]",
    ][k], "derived:" + str(k)


def t_index_names(g, f):
    P = names_of(f)
    A = repr(P[0])
    X = g.name(P)
    k = g.r.randrange(10)
    use = g.ch([f"v = df.groupby({X}).size()", f"v = df.sort_values({X})",
                f"v = df.merge(df, on={X})", f"v = df.groupby([{X}]).size()"])
    return [
        f"df.index.name = {X}\n{use}",
        f"b = df[[{A}]]\nb.index.name = {X}\n{use}",
        f"df.rename_axis({X}, inplace=True)\n{use}",
        f"df.index.names = [{X}]\n{use}",
        f"df.index = df.index.rename({X})\n{use}",
        f"b = df.copy()\nb.index.name = {X}\n{use}",
        f"ix = df.index\nix.name = {X}\n{use}",
        f"df.index.set_names({X}, inplace=True)\n{use}",
        f"df = df.set_index({A})\n{use}",
        f"s = df[{A}]\ns.index.name = {X}\n{use}",
    ][k], "index_names:" + str(k)


def t_mutation(g, f):
    P = names_of(f)
    N = repr(g.ch(ABSENT))
    k = g.r.randrange(28)
    return [
        f"L = df.loc\nL[:, {N}] = 1\nv = df[{N}]",
        f"box = [df]\nbox[0][{N}] = 1\nv = df[{N}]",
        f"class H:\n    pass\nh = H()\nh.d = df\nh.d[{N}] = 1\nv = df[{N}]",
        f"def add(d):\n    d[{N}] = 1\nadd(df)\nv = df[{N}]",
        f"def add():\n    df[{N}] = 1\nadd()\nv = df[{N}]",
        f"def add():\n    d = df\n    d[{N}] = 1\nadd()\nv = df[{N}]",
        f"def add(d=df):\n    d[{N}] = 1\nadd()\nv = df[{N}]",
        f"def get():\n    return df\nget()[{N}] = 1\nv = df[{N}]",
        f"class M:\n    def f(self, d):\n        d[{N}] = 1\nM().f(df)\nv = df[{N}]",
        f"df.__setitem__({N}, 1)\nv = df[{N}]",
        f"df.insert(0, {N}, 1)\nv = df[{N}]",
        f"df.loc[:, {N}] = 1\nv = df[{N}]",
        f"store = {{'a': df}}\nstore['a'][{N}] = 1\nv = df[{N}]",
        f"print((d := df).shape)\nd[{N}] = 1\nv = df[{N}]",
        f"x = y = df\nx[{N}] = 1\nv = df[{N}]",
        f"import operator\noperator.setitem(df, {N}, 1)\nv = df[{N}]",
        f"pd.DataFrame.insert(df, 0, {N}, 1)\nv = df[{N}]",
        f"df.columns = [{N}] + list(df.columns[1:])\nv = df[{N}]",
        f"df.columns.values[0] = {N}\nv = df[{N}]",
        f"exec(\"df[{N}] = 1\")\nv = df[{N}]",
        f"globals()['df'][{N}] = 1\nv = df[{N}]",
        f"for d in [df]:\n    d[{N}] = 1\nv = df[{N}]",
        f"frames = [df]\nfor d in frames:\n    d[{N}] = 1\nv = df[{N}]",
        f"df2 = df.pipe(lambda d: d.assign(**{{{N}: 1}}))\nv = df[{N}]",
        f"d2 = pd.DataFrame(df)\nd2[{N}] = 1\nv = df[{N}]",
        f"df2 = df.pipe(lambda d: d)\ndf2[{N}] = 1\nv = df[{N}]",
        f"getattr(df, '__setitem__')({N}, 1)\nv = df[{N}]",
        f"set_col = df.__setitem__\nset_col({N}, 1)\nv = df[{N}]",
    ][k], "mutation:" + str(k)


def t_record_reads(g, f):
    P = names_of(f)
    X, Y = g.name(P), g.name(P)
    k = g.r.randrange(14)
    return [
        f"v = recs[0][{X}]",
        f"for r in recs:\n    v = r[{X}]",
        f"v = [r[{X}] for r in recs]",
        f"v = [r for r in recs if r[{X}]]",
        f"v = [r for r in recs if False if r[{X}]]",
        f"v = [r for r in recs if r[{Y}] is None if r[{X}]]",
        f"v = [r[{X}] for r in recs if False]",
        f"v = any(r[{X}] for r in recs)",
        f"v = next((r[{X}] for r in recs), None)",
        f"v = [a for r in recs for a in [r[{X}]]]",
        f"v = sorted(recs, key=lambda r: r[{X}])",
        f"v = recs[-1][{X}]",
        f"for i, r in enumerate(recs):\n    v = r[{X}]",
        f"v = {{r[{Y}]: r[{X}] for r in recs}}",
    ][k], "record_reads:" + str(k)


def t_record_mutation(g, f):
    N = repr(g.ch(ABSENT))
    k = g.r.randrange(20)
    return [
        f"x = [recs[0]]\nx[0][{N}] = 1\nv = recs[0][{N}]",
        f"def add(r):\n    r[{N}] = 1\nadd(recs[0])\nv = recs[0][{N}]",
        f"for r in recs:\n    r[{N}] = 1\nv = recs[0][{N}]",
        f"def add():\n    for r in recs:\n        r[{N}] = 1\nadd()\nv = recs[0][{N}]",
        f"class S:\n    def __init__(self, r):\n        self.r = r\ns = S(recs[0])\ns.r[{N}] = 1\nv = recs[0][{N}]",
        f"m = {{'r': recs[0]}}\nm['r'][{N}] = 1\nv = recs[0][{N}]",
        f"recs[0].setdefault({N}, 1)\nv = recs[0][{N}]",
        f"recs[0].update({{{N}: 1}})\nv = recs[0][{N}]",
        f"rc = copy.copy(recs)\nrc[0][{N}] = 1\nv = recs[0][{N}]",
        f"s = sorted(recs, key=str)\nfor r in s:\n    r[{N}] = 1\nv = recs[0][{N}]",
        f"sel = [r for r in recs]\nsel[0][{N}] = 1\nv = recs[0][{N}]",
        f"first = next(r for r in recs)\nfirst[{N}] = 1\nv = recs[0][{N}]",
        f"dict.update(recs[0], {{{N}: 1}})\nv = recs[0][{N}]",
        f"for r in reversed(recs):\n    r[{N}] = 1\nv = recs[0][{N}]",
        f"recs2 = list(recs)\nrecs2[0][{N}] = 1\nv = recs[0][{N}]",
        f"for i in range(len(recs)):\n    recs[i][{N}] = 1\nv = recs[0][{N}]",
        f"recs2 = recs.copy()\nrecs2[0][{N}] = 1\nv = recs[0][{N}]",
        f"for r in recs[:1]:\n    r[{N}] = 1\nv = recs[0][{N}]",
        f"import random\nrandom.choice(recs[:1])[{N}] = 1\nv = recs[0][{N}]",
        f"pick = (lambda xs: xs[0])\npick(recs)[{N}] = 1\nv = recs[0][{N}]",
    ][k], "record_mutation:" + str(k)


READS_DF = ["v = df[{X}]", "v = df[{X}].sum()", "v = df.groupby({X}).size()", "v = df[[{X}]]"]


def t_control(g, f, rec=False):
    P = names_of(f)
    X = g.name(P, 0.8)
    read = (f"v = recs[0][{X}]" if rec else g.ch(READS_DF).format(X=X))
    expr = (f"recs[0][{X}]" if rec else f"df[{X}]")
    k = g.r.randrange(40)
    return [
        f"v = (1 > 2 > len({expr}))",
        f"if 0 > 1 < len({expr}):\n    pass",
        f"v = 0 < 1 < len({expr})",
        f"with contextlib.suppress(KeyError):\n    {read}",
        f"from contextlib import suppress\nwith suppress(Exception):\n    {read}",
        f"with contextlib.suppress(KeyError), open('notes.md') as fh:\n    {read}",
        f"class S:\n    def __enter__(self):\n        return self\n    def __exit__(self, *a):\n        return True\nwith S():\n    {read}",
        f"@contextlib.contextmanager\ndef cm():\n    try:\n        yield\n    except Exception:\n        pass\nwith cm():\n    {read}",
        f"with open('notes.md') as fh:\n    {read}",
        f"with warnings.catch_warnings():\n    {read}",
        f"def done(x):\n    final_answer(x)\ndone(1)\n{read}",
        f"stop = final_answer\nstop(1)\n{read}",
        f"stop = lambda: final_answer(1)\nstop()\n{read}",
        f"@final_answer\ndef g():\n    pass\n{read}",
        f"list(map(final_answer, [1]))\n{read}",
        f"print(final_answer(1), {expr})",
        f"v = [final_answer(1), {expr}]",
        f"from sys import exit as stop\nstop()\n{read}",
        f"import sys as s\ns.exit()\n{read}",
        f"exec('final_answer(1)')\n{read}",
        f"eval('final_answer(1)')\n{read}",
        f"getattr(sys, 'exit')()\n{read}",
        f"class A:\n    def __init__(self):\n        final_answer(1)\nA()\n{read}",
        f"class B:\n    def go(self):\n        final_answer(1)\nB().go()\n{read}",
        f"def gen():\n    final_answer(1)\n    yield 1\nlist(gen())\n{read}",
        f"for _ in [1]:\n    final_answer(1)\n{read}",
        f"print = final_answer\nprint(1)\n{read}",
        f"def any(x):\n    return True\nv = any({expr} for _ in [1])",
        f"list = lambda it: []\nv = list({expr} for _ in [1])",
        f"try:\n    {read}\nexcept KeyError:\n    v = None",
        f"v = {expr} if False else 0",
        f"v = False and {expr}",
        f"raise SystemExit\n{read}",
        f"sys.exit(0)\n{read}",
        f"import builtins\nbuiltins.exec('final_answer(1)')\n{read}",
        f"class C:\n    final_answer(1)\n{read}",
        f"def f(x=final_answer(1)):\n    pass\n{read}",
        f"with contextlib.ExitStack() as es:\n    es.enter_context(contextlib.suppress(KeyError))\n    {read}",
        f"fa = {{'f': final_answer}}\nfa['f'](1)\n{read}",
        f"import threading\nt = threading.Thread(target=final_answer, args=(1,))\nt.run()\n{read}",
    ][k], "control:" + str(k)


def t_files(g):
    M = repr(g.ch(MISSING_FILES))
    k = g.r.randrange(33)
    return [
        f"d = pd.read_csv({M})",
        f"d = pd.read_csv('~/sales.csv')",
        f"d = pd.read_json('[{{\"a\": 1}}]')",
        f"d = pd.read_json({M})",
        f"df = pd.read_csv('sales.csv')\ndf.to_csv('new.csv')\nd = pd.read_csv('new.csv')",
        f"df = pd.read_csv('sales.csv')\ndf.to_string('new.txt')\nt = open('new.txt').read()",
        f"df = pd.read_csv('sales.csv')\ndf.to_html('new.html')\nt = open('new.html').read()",
        f"df = pd.read_csv('sales.csv')\ndf.to_latex('new.tex')\nt = open('new.tex').read()",
        f"np.savetxt('new.txt', np.zeros(2))\nt = open('new.txt').read()",
        f"Path('new.txt').touch()\nt = open('new.txt').read()",
        f"Path('new.txt').write_bytes(b'x')\nt = open('new.txt').read()",
        f"name = 'new' + str(1) + '.txt'\nopen(name, 'w').write('x')\nt = open('new1.txt').read()",
        f"with open(f'new{{1}}.txt', 'w') as fh:\n    fh.write('x')\nt = open('new1.txt').read()",
        f"import shutil\nshutil.copy('sales.csv', 'copy.csv')\nd = pd.read_csv('copy.csv')",
        f"os.rename('notes.md', 'moved.md')\nt = open('moved.md').read()",
        f"with Path('new.txt').open('w') as fh:\n    fh.write('x')\nt = open('new.txt').read()",
        f"fh = Path('new.txt').open('w')\nfh.write('x')\nfh.close()\nt = open('new.txt').read()",
        f"def open(p, *a, **k):\n    return io.StringIO('x')\nt = open({M}).read()",
        f"from os import chdir as cd\ncd('sub')\nd = pd.read_csv('inner.csv')",
        f"import os as o\no.chdir('sub')\nd = pd.read_csv('inner.csv')",
        f"from os import makedirs as md\nmd('newdir')\nv = os.listdir('newdir')",
        f"v = os.listdir({g.ch(['missing_dir', 'sub', '.'])!r})",
        f"if os.path.exists({M}):\n    d = pd.read_csv({M})",
        f"t = Path({M}).read_text()",
        f"fh = open('sales.csv')\nfh.readline()\nrd = csv.DictReader(fh)\nfor row in rd:\n    v = row[{g.ch(['Crossfit', 'merchant', '10.5', 'absent'])!r}]",
        f"rd = csv.DictReader(open('one.csv'))\nnext(rd)\nfor row in rd:\n    v = row[{g.name(COLS['one.csv'])}]",
        f"rd = csv.DictReader(open('two.csv'))\nfirst = next(rd)\nfor row in rd:\n    v = row[{g.name(COLS['two.csv'])}]",
        f"d = pd.read_csv('sales.csv', nrows=n0)\nfor _, row in d.iterrows():\n    v = row[{g.name(COLS['sales.csv'])}]",
        f"d = pd.read_csv('https://example.invalid/sales.csv')",
        f"d = pd.read_csv('file://' + os.path.abspath('sales.csv'))\nv = d[{g.name(COLS['sales.csv'])}]",
        ["os.chdir('sub')", "d = pd.read_csv('inner.csv')"],
        ["from os import chdir\nchdir('sub')", "t = open('inner.csv').read()"],
        f"d = pd.read_csv(io.StringIO('a,b\\n1,2\\n'))\nv = d[{g.name(['a', 'b'])}]",
    ][k], "files:" + str(k)


def program(rng):
    """One program: a list of one or two step sources."""
    g = G(rng)
    kind = g.r.randrange(10)
    if kind == 9:
        code, tag = t_files(g)
        if isinstance(code, list):
            steps = code
        else:
            if "n0" in code:
                code = f"n0 = {g.ch([0, 1])}\n" + code
            steps = [code]
    else:
        rec = kind in (6, 7, 8)
        if rec:
            load, f = load_records(g)
        else:
            load, f = load_frame(g)
        if kind in (0, 1):
            body, tag = t_frame_reads(g, f)
        elif kind == 2:
            body, tag = t_derived(g, f)
        elif kind == 3:
            body, tag = t_index_names(g, f)
        elif kind == 4:
            body, tag = t_mutation(g, f)
        elif kind == 5:
            body, tag = t_control(g, f, rec=False)
        elif kind == 6:
            body, tag = t_record_reads(g, f)
        elif kind == 7:
            body, tag = t_record_mutation(g, f)
        else:
            body, tag = t_control(g, f, rec=True)
        if g.p(0.25):                       # two steps: the load (and maybe a definition) first
            first = load
            if g.p(0.3):
                first += "\ndef done(x):\n    final_answer(x)"
                body = "done(1)\n" + body
            steps = [first, body]
        else:
            steps = [load + "\n" + body]
    return steps, tag


# ---------------------------------------------------------------- running
RUNNER = r'''
import json, os, sys, shutil, tempfile, select, signal
DATA, INP, OUT, TIMEOUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
PRELUDE = json.loads(sys.argv[5])
exec(PRELUDE, {})          # import the modules once, before the children are forked

def snapshot(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root))
        if d != root:
            out.append(os.path.relpath(d, root) + "/")
    return sorted(out)

def child(steps, wfd):
    work = tempfile.mkdtemp(prefix="dt_")
    shutil.copytree(DATA, work, dirs_exist_ok=True)
    os.chdir(work)
    os.environ["HOME"] = work
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    class _End(BaseException):
        pass
    def final_answer(x=None):
        raise _End()
    g = {"__name__": "__main__", "final_answer": final_answer}
    exec(PRELUDE, g)
    res = []
    for code in steps:
        snap = snapshot(work)
        try:
            exec(compile(code, "<step>", "exec"), g)
            res.append(["ok", "", snap])
        except _End:
            res.append(["end", "", snap])
        except SystemExit:
            res.append(["end", "SystemExit", snap])
        except BaseException as e:
            res.append(["error", (type(e).__name__ + ": " + str(e))[:300], snap])
    os.write(wfd, json.dumps(res).encode())
    os._exit(0)

out = open(OUT, "w")
for line in open(INP):
    steps = json.loads(line)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            child(steps, w)
        finally:
            os._exit(1)
    os.close(w)
    buf, deadline = b"", TIMEOUT
    import time
    t0 = time.time()
    timed_out = False
    while True:
        left = TIMEOUT - (time.time() - t0)
        if left <= 0:
            timed_out = True
            break
        rd, _, _ = select.select([r], [], [], left)
        if not rd:
            timed_out = True
            break
        chunk = os.read(r, 65536)
        if not chunk:
            break
        buf += chunk
    if timed_out:
        os.kill(pid, signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    os.close(r)
    if timed_out:
        res = [["timeout", "", []] for _ in steps]
    elif not buf:
        res = [["crash", f"status {status}", []] for _ in steps]
    else:
        res = json.loads(buf)
    out.write(json.dumps(res) + "\n")
out.close()
'''


def run_programs(python, programs, data, timeout, workdir):
    inp = os.path.join(workdir, "programs.jsonl")
    with open(inp, "w") as fh:
        for steps in programs:
            fh.write(json.dumps(steps) + "\n")
    script = os.path.join(workdir, "runner.py")
    with open(script, "w") as fh:
        fh.write(RUNNER)
    out = os.path.join(workdir, f"out_{abs(hash(python))}.jsonl")
    subprocess.run([python, "-I", script, data, inp, out, str(timeout), json.dumps(PRELUDE)], check=True)
    with open(out) as fh:
        return [json.loads(line) for line in fh]


def version_of(python):
    return subprocess.run([python, "-c", "import pandas, sys; print(pandas.__version__)"],
                          capture_output=True, text=True, check=True).stdout.strip()


def load_gate(gate_dir, tag):
    spec = importlib.util.spec_from_file_location(f"gate_{tag}", os.path.join(gate_dir, "gate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decide_program(gate, steps, outcomes, sch):
    """The checker's decision for each step, with the files present before that step."""
    state = gate.GateState()
    out = []
    for i, code in enumerate(steps):
        snap = set(outcomes[i][2]) if i < len(outcomes) and outcomes[i][2] else None

        def exists(path, snap=snap):
            if snap is None or os.path.isabs(path):
                return None if snap is None else os.path.exists(path)
            p = os.path.normpath(path)
            return p in snap or (p + "/") in snap or p == "."
        try:
            r = gate.decide(code, sch, state, exists)
        except Exception as e:                      # noqa: BLE001  a checker crash is reported, not hidden
            r = {"outcome": "crash", "reason": f"{type(e).__name__}: {e}"}
        out.append(r)
        if r.get("outcome") != "crash":
            state.commit(failed=outcomes[i][0] not in ("ok", "end"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--python", action="append", required=True)
    ap.add_argument("--gate-dir", action="append", required=True)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", type=int, default=40)
    ap.add_argument("--per-tag", type=int, default=3, help="false-block examples kept per template")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tagged = [program(rng) for _ in range(a.n)]
    programs = [p for p, _ in tagged]
    tags = [t for _, t in tagged]
    for steps in programs:
        for s in steps:
            ast.parse(s)                          # every generated step is valid Python
    gates = [(d, load_gate(d, i)) for i, d in enumerate(a.gate_dir)]
    sch = schema(None)
    report = {"n": a.n, "seed": a.seed, "versions": {}}
    with tempfile.TemporaryDirectory() as work:
        data = os.path.join(work, "data")
        os.makedirs(data)
        make_data(data)
        for py in a.python:
            ver = version_of(py)
            outs = run_programs(py, programs, data, a.timeout, work)
            vr = {"python": py, "steps": 0, "ran": 0, "errors": 0, "data_errors": 0, "timeouts": 0, "crashes": 0,
                  "gates": {}}
            for d, _ in gates:
                vr["gates"][d] = {"block": 0, "false_block": 0, "true_block": 0, "crash": 0, "examples": [],
                                  "false_by_tag": {}, "crash_examples": [], "by_tag": {}}
            decisions = {d: [decide_program(gate, steps, o, sch) for steps, o in zip(programs, outs)]
                         for d, gate in gates}
            for pi, (steps, o) in enumerate(zip(programs, outs)):
                for si, code in enumerate(steps):
                    kind, msg = o[si][0], o[si][1]
                    vr["steps"] += 1
                    ran = kind in ("ok", "end")
                    vr["ran"] += ran
                    vr["errors"] += kind == "error"
                    vr["data_errors"] += kind == "error" and msg.split(":")[0] in ("KeyError", "FileNotFoundError")
                    vr["timeouts"] += kind == "timeout"
                    vr["crashes"] += kind == "crash"
                    for d, _ in gates:
                        r = decisions[d][pi][si]
                        gr = vr["gates"][d]
                        if r["outcome"] == "crash":
                            gr["crash"] += 1
                            if len(gr["crash_examples"]) < 5:
                                gr["crash_examples"].append({"steps": steps, "reason": r.get("reason")})
                            continue
                        if r["outcome"] != "block":
                            continue
                        gr["block"] += 1
                        if ran:
                            gr["false_block"] += 1
                            gr["false_by_tag"][tags[pi]] = gr["false_by_tag"].get(tags[pi], 0) + 1
                            bt = gr["by_tag"].setdefault(tags[pi], [])
                            if len(bt) < a.per_tag:
                                bt.append(steps)
                            if len(gr["examples"]) < a.examples:
                                gr["examples"].append({"program": pi, "tag": tags[pi], "step": si, "steps": steps, "run": kind,
                                                       "missing": r.get("missing"), "location": r.get("location")})
                        elif kind == "error":
                            gr["true_block"] += 1
            report["versions"][ver] = vr
            print(f"pandas {ver}: {vr['steps']} steps, {vr['ran']} ran without error, {vr['errors']} errors "
                  f"({vr['data_errors']} KeyError or FileNotFoundError), {vr['timeouts']} timeouts, "
                  f"{vr['crashes']} crashes")
            for d, _ in gates:
                gr = vr["gates"][d]
                print(f"  {d}: {gr['block']} blocks, {gr['false_block']} false blocks, "
                      f"{gr['true_block']} blocks of failing steps, {gr['crash']} checker crashes")
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
