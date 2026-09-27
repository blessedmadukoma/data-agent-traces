#!/usr/bin/env python3
"""Minimal exact pre-execution gate for agent-generated pandas code (runbook step 4).

decide(code, schema, state=None, exists=None) -> dict with "outcome" in
{"run", "block", "unknown"}.

schema maps a source file name to its columns (CSV) or record keys (JSON that
holds a list of objects), for example
    {"payments.csv": {"merchant", "card_scheme", ...}, "fees.json": {"ID", ...}}
For JSON, a key is missing only if no record has it, so a read of that key
fails for every record.

Tracked objects (version 4):
  * frame    df = pd.read_csv(<path>) / pd.read_json(<path>); pd.DataFrame(records)
  * records  json.load(f) / json.loads(text) of a context JSON file;
             list(csv.DictReader(f)); df.to_dict("records")
  * record   one element: a loop variable over records or over a
             csv.DictReader, recs[0] / recs[-1], or row in df.iterrows()
  * rows     csv.DictReader(f), which can be read once
  * chunks   pd.read_csv(<path>, chunksize=...) or iterator=True
  Paths: a string literal, a name bound to a literal, os.path.join, "+",
  an f-string, Path(...) and "/" of literals. The file is matched by base
  name. A file handle from open(<path>) or text from f.read(),
  open(<path>).read() or Path(<path>).read_text() is followed into
  json.load / json.loads. Harness tools that return text are not followed.

Checks:
  * df["col"], df[["a", "b"]] on a frame; rec["key"] on a record;
    usecols=[...] on read_csv.
  * (v6) Derived frames (fix 2): the columns of df[mask], df[["a"]],
    df.loc/iloc row selections, copy/head/sort_values/dropna/query/...,
    assign, drop(columns), rename(columns={...}), reset_index, merge on
    known keys, pd.concat, and groupby reductions with reset_index or
    as_index=False (see _Walker.frame_of). A column that both merged frames
    have is kept with and without its suffix, so the known columns never
    miss a real column.
  * (v6) Helper functions (fix 4): a module-level function with one
    parameter whose body only tests the parameter (isinstance, endswith,
    os.path.exists), opens and loads the file it names (json.load,
    pd.read_csv, pd.read_json) or returns it. A call is run abstractly on
    its argument: it gives the file's records or frame, or the argument
    itself, and a file it must open is checked like open().
  * (v6) Column names that pandas methods require on a tracked frame:
    groupby, sort_values, drop_duplicates, dropna(subset), set_index,
    pivot and pivot_table keywords, drop(columns=...) or drop(labels,
    axis=1) unless errors="ignore", merge on / left_on / right_on (each
    side if it is tracked), and df.loc[rows, "col"] / df.at[i, "col"].
    rename and get are not checked: they do not raise for a missing name.
  * (v7) Reader arguments: a whitelist (rule 2). A tracked read_csv gives
    the file's columns only if every keyword is one that pandas documents
    as not changing the column names (READ_CSV_ANY_VALUE, or header, sep,
    delimiter, index_col, skip_blank_lines, encoding and parse_dates at
    their default values). read_json allows READ_JSON_ANY_VALUE and a UTF-8
    encoding. csv.DictReader is tracked only with one argument and no
    keyword. Any other argument makes the read unknown.
  * (v7) Schema (rule 1): csv_columns(path) gives the union of the columns
    of pd.read_csv(path, nrows=0) and the raw first row from csv.reader.
    gate_adapter.context_schema uses it when the loaded gate has it.
  * (v8) R1: with exists=callable, os.listdir(p), os.scandir(p) and
    os.chdir(p) need p to exist (they raise FileNotFoundError at the call).
    Not checked in a cell that may create a directory (DIR_WRITER_NAMES).
    R2: df[name] where name is bound to a string literal (as paths already
    are), or bound once in the cell to a list of string literals that is
    only read (_list_consts), is checked like df["literal"].
    R3: with locate=callable, a file receipt lists where files with the
    same base name exist (found_in).
  * With exists=callable: every file that a reader, open() or json.load
    reads must exist. exists(path) returns True, False or None (unknown).
    In a live run, pass os.path.exists. A file written earlier in the same
    cell (to_csv, open(..., "w"), ...) counts as existing.

Executors: with tolerant_keys=True, a missing dict key that has a close
match (difflib, cutoff 0.6) is not a violation, because smolagents 1.0-1.4
silently reads the closest key instead (evaluate_subscript). Such reads are
listed in result["substitutions"]. DataFrame columns are not affected.

Loops over a non-empty source (a context file is never empty): the body
runs at least once, so the first pass through the body is checked as code
that always runs, up to the first statement that may skip the rest
(break, continue, final_answer). The same holds for the first filter of a
list comprehension, and of a generator passed to next(), list(), any() and
similar.

These stop tracking an object, because they can change its columns or
keys in a way the gate does not follow: reassigning it; an in-place method
(inplace=True, insert, pop, update, pipe, append, extend, setdefault, clear,
remove; and drop, rename, set_index, reset_index, join as a bare
statement); df.loc/iloc/at/iat[...] = ...; df.attr = ...; del; binding it to
another name or putting it in a container; iterating over a list, tuple,
set or dict display that contains it (for df in (a, b)); passing it to a function that is
not known to be pure; a function body that changes it; exec, eval,
globals, locals, vars or setattr (these stop tracking all objects).
rec["new"] = ... adds the key to every tracked record of the same file.

Control flow:
  * final_answer(...), exit(), quit(), sys.exit(), os._exit() and
    raise SystemExit end the cell without an error. Code after such a call does not run. Code after a
    statement that may make such a call runs only conditionally.
  * Code after an unconditional raise does not run. A conditional raise does
    not make later code conditional: if the branch runs, the cell fails
    anyway, so a block is not a false block.
  * In `a and b` and `a or b`, only the first operand always runs. The same
    holds for the branches of `x if c else y` and for lambda bodies.
  * The bodies of if, while, try, match and function definitions, and of
    loops over other iterables, may not run.

Aliases (alias=True, for harnesses without state across cells, such as
DataAgentBench): `x = y` in code that always runs, where y is tracked, binds
x to the same object instead of dropping both. Forgetting any name of an
alias group forgets the whole group, so a change through either name stops
tracking both. Aliased names are not exported to a GateState.

State across cells:
  Pass the same GateState to every cell of one trace, in order, and call
  state.commit(failed=...) after the cell. If the cell failed, every name
  that the cell would change is dropped. If the cell did not run at all (it
  was blocked), do not call commit.

Decisions:
  * block    a violation in code that always runs.
  * run      no violation, at least one tracked read, and every source known.
  * unknown  invalid Python; no tracked read; a path that cannot be
             resolved; a file that is not in the schema; or a violation only
             in code that may not run.

A block carries a receipt: location, kind (column, key or file), missing
name, available names, other sources that contain the name, and close
matches. result["files"] lists the files the cell reads, as
(path, conditional, line), and result["written"] the files it writes, for
evaluation.
"""
import ast
import difflib
import os
import warnings

READERS = {"read_csv", "read_json"}
FILE_READERS = READERS | {"read_excel", "read_parquet", "read_table", "read_feather", "read_pickle"}
WRITERS = {"to_csv", "to_json", "to_excel", "to_parquet", "to_pickle", "to_feather", "write_text"}
PATH_KWARGS = ("filepath_or_buffer", "path_or_buf", "io", "path", "file")
# gate v7, rule 2: read_csv keywords that pandas documents as not changing the column
# names, with any value. names and usecols are handled in bind_reader.
READ_CSV_ANY_VALUE = frozenset({
    "dtype", "usecols", "names", "nrows", "chunksize", "iterator", "low_memory", "na_values",
    "keep_default_na", "na_filter", "date_format", "dayfirst", "cache_dates", "thousands", "decimal",
    "converters", "true_values", "false_values", "float_precision", "memory_map", "on_bad_lines",
    "compression", "storage_options", "encoding_errors", "dtype_backend", "verbose", "engine"})
READ_JSON_ANY_VALUE = frozenset({"dtype", "convert_dates", "keep_default_dates", "precise_float", "date_unit"})
UTF8_NAMES = frozenset({"utf-8", "utf8", "utf_8", "u8", "utf-8-sig", "utf_8_sig"})
# methods that change an object when their result is discarded (bare statement)
INPLACE_METHODS = {"insert", "pop", "drop", "rename", "set_index", "reset_index",
                   "update", "join", "pipe"}
# methods that change an object whatever happens to their result
MUTATING_METHODS = {"insert", "pop", "update", "pipe", "append", "extend", "setdefault",
                    "clear", "remove"}
# methods on another object that keep a reference to their argument
ESCAPE_METHODS = {"append", "extend", "insert", "add", "setdefault", "update", "put"}
# functions that do not change an object passed to them
PURE_FUNCTIONS = {"print", "len", "display", "str", "repr", "type", "isinstance", "id",
                  "list", "tuple", "set", "dict", "sorted", "reversed", "enumerate",
                  "zip", "min", "max", "sum", "any", "all", "bool", "int", "float",
                  "round", "abs", "hasattr", "getattr", "iter", "next", "format",
                  "hash", "final_answer"}
# calls that consume at least the first element of a generator passed to them
CONSUMERS = {"next", "list", "tuple", "set", "frozenset", "sorted", "any", "all", "sum",
             "min", "max", "dict"}
UNTRACKABLE = {"exec", "eval", "globals", "locals", "vars", "setattr"}
# gate v8, R1: calls that may create a directory; a cell with one of them gets no directory check
DIR_WRITER_NAMES = {"os.makedirs", "os.mkdir", "os.rename", "os.replace", "os.renames", "os.system", "os.symlink",
                    "shutil.copytree", "shutil.move", "shutil.unpack_archive", "shutil.copy", "shutil.copyfile",
                    "shutil.copy2", "subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_call",
                    "subprocess.check_output", "makedirs", "mkdir", "system", "copytree", "urlretrieve"}
DIR_WRITER_ATTRS = {"makedirs", "mkdir", "extractall", "copytree", "urlretrieve", "unpack_archive"}
# gate v8, R2: calls that only read a list of names passed to them
LIST_SAFE_CALLS = {"len", "print", "sorted", "list", "set", "tuple", "str", "repr"}
# pandas methods whose name arguments must be columns of the frame (gate v6, fix 1):
# method -> (position of the argument, keywords). A missing column raises KeyError.
COLUMN_ARGS = {"groupby": (0, ("by",)), "sort_values": (0, ("by",)), "drop_duplicates": (0, ("subset",)),
               "dropna": (None, ("subset",)), "set_index": (0, ("keys",)),
               "pivot": (None, ("index", "columns", "values")),
               "pivot_table": (None, ("index", "columns", "values"))}
ROW_AXIS = (0, "index", "rows")
# methods whose result has the same columns as the frame (gate v6, fix 2); the
# second set also keeps every row, so a non-empty frame stays non-empty
SAME_COLUMNS = {"copy", "head", "tail", "sample", "sort_values", "sort_index", "drop_duplicates", "dropna",
                "fillna", "query", "astype", "nlargest", "nsmallest", "round", "replace", "ffill", "bfill"}
KEEP_ROWS = {"copy", "sort_values", "sort_index", "fillna", "astype", "round", "replace", "ffill", "bfill"}
# groupby reductions whose result columns are the selected (or all) columns
REDUCE = {"sum", "mean", "count", "max", "min", "median", "nunique", "first", "last", "std", "var", "prod"}
# calls that return a boolean Series, so df[<call>] keeps the columns
MASK_CALLS = {"isin", "notna", "isna", "notnull", "isnull", "between", "contains", "startswith", "endswith",
              "duplicated", "eq", "ne", "gt", "lt", "ge", "le", "match", "fullmatch"}
END_NAMES = {"final_answer", "exit", "quit"}
END_ATTRS = {("sys", "exit"), ("os", "_exit")}
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
CELL, LOOP = "cell", "loop"   # a path that ends the cell / ends this loop iteration


class GateState:
    """Tracked objects and string constants carried across the cells of one trace."""

    def __init__(self):
        self.frames = {}    # name -> (source file, frozenset of names, kind, non-empty)
        self.consts = {}    # name -> literal string, or ("handle" | "text", path)
        self.pending = None

    def commit(self, failed):
        """Apply the state after the cell that decide() last saw.

        failed=False: the cell ran to the end. failed=True: the cell stopped
        at an unknown line, so names that it would change are dropped."""
        if self.pending is None:
            return
        frames, consts = self.pending
        self.pending = None
        if not failed:
            self.frames, self.consts = frames, consts
            return
        self.frames = {k: v for k, v in self.frames.items()
                       if frames.get(k) == v}
        self.consts = {k: v for k, v in self.consts.items()
                       if consts.get(k) == v}


def _resolve(node, consts):
    """Return a string for a literal path expression, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        v = consts.get(node.id)
        return v if isinstance(v, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        a, b = _resolve(node.left, consts), _resolve(node.right, consts)
        if a is None or b is None:
            return None
        return a + b if isinstance(node.op, ast.Add) else os.path.join(a, b)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                r = _resolve(v.value, consts)
                if r is None:
                    return None
                parts.append(r)
        return "".join(parts)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "join" and isinstance(f.value, ast.Attribute) \
                and f.value.attr == "path":  # os.path.join
            parts = [_resolve(a, consts) for a in node.args]
            return os.path.join(*parts) if parts and all(p is not None for p in parts) else None
        if (isinstance(f, ast.Name) and f.id in ("Path", "PurePath")) or \
                (isinstance(f, ast.Attribute) and f.attr in ("Path", "PurePath")):
            parts = [_resolve(a, consts) for a in node.args]
            return os.path.join(*parts) if parts and all(p is not None for p in parts) else None
    return None


def _path_arg(call):
    if call.args:
        return call.args[0]
    return next((k.value for k in call.keywords if k.arg in PATH_KWARGS), None)


def _reader_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in READERS)


def _const_is(node, values):
    """A constant equal to one of values, with the same type (header=False is not header=0)."""
    return isinstance(node, ast.Constant) and any(
        type(node.value) is type(v) and node.value == v for v in values)


def _utf8(node):
    return isinstance(node, ast.Constant) and (
        node.value is None or (isinstance(node.value, str) and node.value.strip().lower() in UTF8_NAMES))


def _reader_args(call):
    """Gate v7, rule 2. None if every argument of a read_csv / read_json call is one
    that pandas documents as not changing the column names, else the reason."""
    kind = call.func.attr
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return f"{kind} with *args or **kwargs"
    if len(call.args) > 1:
        return f"{kind} with a second positional argument"
    for k in call.keywords:
        a, v = k.arg, k.value
        if a in PATH_KWARGS and not call.args:
            continue
        if kind == "read_csv":
            if a in READ_CSV_ANY_VALUE:
                continue
            if a == "header" and _const_is(v, (0, "infer")):
                continue
            if a in ("sep", "delimiter") and _const_is(v, (",",)):
                continue
            if a == "index_col" and _const_is(v, (None, False)):
                continue
            if a == "skip_blank_lines" and _const_is(v, (True,)):
                continue
            if a == "parse_dates" and (_const_is(v, (True, False)) or _literal_list(v) is not None):
                continue
        elif a in READ_JSON_ANY_VALUE:
            continue
        if a == "encoding" and _utf8(v):
            continue
        return f"{kind}({a}=...) may change the column names"
    return None


def _plain_dictreader(call):
    """Gate v7, rule 2: csv.DictReader(f) with one argument and no keyword. A
    second positional argument is fieldnames."""
    return len(call.args) == 1 and not isinstance(call.args[0], ast.Starred) and not call.keywords


def csv_columns(path):
    """Gate v7, rule 1: the column names a CSV file can give, or None if pandas
    cannot read it with default arguments. The union of the columns of
    pd.read_csv(path, nrows=0), which covers a byte-order mark, leading blank
    lines, 'Unnamed: n' for an empty field and 'x.1' for a repeated name, and
    the raw first row from csv.reader, which csv.DictReader uses."""
    import csv
    import pandas as pd
    try:
        cols = {str(c) for c in pd.read_csv(path, nrows=0).columns}
    except Exception:                       # noqa: BLE001  pandas raises many error types
        return None
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            row = next(csv.reader(fh), None)
    except (UnicodeDecodeError, csv.Error, OSError):
        row = None
    return cols | set(row or [])


def _call_name(node):
    """'json.load' for json.load(...), 'open' for open(...), else the attribute or name."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        if isinstance(f.value, ast.Name):
            return f"{f.value.id}.{f.attr}"
        return f.attr
    return None


def _open_mode(call):
    mode = call.args[1] if len(call.args) > 1 else next(
        (k.value for k in call.keywords if k.arg == "mode"), None)
    if mode is None:
        return "r"
    return mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else None


def _literal_list(node):
    if isinstance(node, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
        return [e.value for e in node.elts]
    return None


def _literal_names(node):
    """String literals in a name argument: 'a' -> ['a']; ['a', x, 'b'] -> ['a', 'b']."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


# calls that copy or only read a list passed to them: a helper result given straight to them
# does not make the helper's argument reachable by another name (fix 4)
SAFE_CONSUMERS = {"pd.DataFrame", "pandas.DataFrame", "DataFrame", "len", "print", "list", "sorted", "json.dumps",
                  "set", "tuple", "enumerate", "display"}
HELPER_CALLS = {"isinstance", "open", "json.load", "json.loads", "load", "loads", "pd.read_csv", "pd.read_json",
                "pandas.read_csv", "pandas.read_json", "os.path.exists", "os.path.isfile", "exists", "isfile",
                "endswith", "startswith", "lower", "strip", "read"}


def _helper_pure(d):
    """True if a function body only tests its parameter, opens and loads a file, binds
    local names and returns: it cannot change any object (gate v6, fix 4)."""
    ok_stmts = (ast.Expr, ast.If, ast.Return, ast.With, ast.Assign, ast.Pass)
    for n in ast.walk(ast.Module(body=d.body, type_ignores=[])):
        if isinstance(n, ast.stmt) and not isinstance(n, ok_stmts):
            return False
        if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant):
            return False
        if isinstance(n, ast.Assign) and not (len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)):
            return False
        if isinstance(n, ast.Call) and _call_name(n) not in HELPER_CALLS and not (
                isinstance(n.func, ast.Attribute) and n.func.attr in ("endswith", "startswith", "lower", "strip", "read")):
            return False
        if isinstance(n, ast.Call) and _call_name(n) == "open" and _open_mode(n) not in ("r", "rb", "rt"):
            return False                            # writing a file changes the workspace
        if isinstance(n, (ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)):
            return False
    return True


def _is_mask(node):
    """An expression that gives a boolean Series (df[mask] keeps every column)."""
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Invert, ast.Not)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return True
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MASK_CALLS


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _is_const(node, values):
    return isinstance(node, ast.Constant) and node.value in values


def _subscript_keys(node):
    """Column names read by df[...] with literal keys, else None."""
    s = node.slice
    if isinstance(s, ast.Constant) and isinstance(s.value, str):
        return [s.value]
    return _literal_list(s)


def _walk_no(node, skip):
    todo = [node]
    while todo:
        n = todo.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if not isinstance(c, skip):
                todo.append(c)


def _walk_no_defs(node):
    """ast.walk that does not enter function, class or lambda bodies."""
    return _walk_no(node, DEFS)


def _is_end_call(n):
    if not isinstance(n, ast.Call):
        return False
    f = n.func
    if isinstance(f, ast.Name):
        return f.id in END_NAMES
    return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
        and (f.value.id, f.attr) in END_ATTRS


def _is_exit_raise(n):
    """raise SystemExit / raise SystemExit(...): ends the process, with exit status 0 if no code is given."""
    if not isinstance(n, ast.Raise) or n.exc is None:
        return False
    e = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
    return (isinstance(e, ast.Name) and e.id == "SystemExit") or \
        (isinstance(e, ast.Attribute) and e.attr == "SystemExit")


def _may_end(stmt):
    """True if the statement may end the cell without an error."""
    return any(_is_end_call(n) or _is_exit_raise(n) for n in _walk_no_defs(stmt))


def _may_skip(stmt):
    """True if the statement may skip the rest of the enclosing loop body."""
    if isinstance(stmt, LOOPS):     # a nested loop owns the break and continue in its body
        return any(_may_skip(s) for s in stmt.orelse)
    return any(isinstance(n, (ast.Break, ast.Continue)) for n in _walk_no(stmt, DEFS + LOOPS))


def _escaping(expr):
    """Bare names whose object becomes reachable through the assigned value."""
    if isinstance(expr, ast.Name):
        return {expr.id}
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        return set().union(*(_escaping(e) for e in expr.elts)) if expr.elts else set()
    if isinstance(expr, ast.Dict):
        return set().union(*(_escaping(v) for v in expr.values if v is not None)) if expr.values else set()
    if isinstance(expr, ast.IfExp):
        return _escaping(expr.body) | _escaping(expr.orelse)
    if isinstance(expr, ast.BoolOp):
        return set().union(*(_escaping(v) for v in expr.values))
    if isinstance(expr, (ast.Starred, ast.NamedExpr)):
        return _escaping(expr.value)
    return set()


def _display_escapes(expr):
    """Bare names inside list, tuple, set or dict displays in expr. Iterating
    over such a display binds each object to the loop target, which can then
    change it (for df in (a, b): df["x"] = 1)."""
    out = set()
    for n in ast.walk(expr):
        if isinstance(n, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            out |= _escaping(n)
    return out


def _names(node):
    return {e.id for e in ast.walk(node) if isinstance(e, ast.Name)}


def _records_arg(call):
    """True for to_dict("records") and to_dict(orient="records")."""
    a = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "orient"), None)
    return isinstance(a, ast.Constant) and a.value == "records"


def _dir_writers(tree):
    """Gate v8, R1: True if the cell has a call that may create a directory."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = _call_name(n)
            if name in DIR_WRITER_NAMES or (isinstance(n.func, ast.Attribute) and n.func.attr in DIR_WRITER_ATTRS):
                return True
    return False


def _list_consts(tree):
    """Gate v8, R2: {name: (line, tuple of str)} for names that the cell binds exactly once, in a
    top-level `name = [\"a\", \"b\"]` (list or tuple of string literals), and uses only read-only
    after that line: as a subscript key, as the iterable of a loop or comprehension, as an
    argument of LIST_SAFE_CALLS, or as the right operand of `in` / `not in`."""
    cand = {}
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                and isinstance(st.value, (ast.List, ast.Tuple)) and st.value.elts \
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in st.value.elts):
            cand.setdefault(st.targets[0].id, []).append((st.lineno, tuple(e.value for e in st.value.elts)))
    if not cand:
        return {}
    parent = {}
    for n in ast.walk(tree):
        for ch in ast.iter_child_nodes(n):
            parent[ch] = n
    stores, bad = {}, set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            bad |= set(n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores[n.name] = stores.get(n.name, 0) + 1
            a = getattr(n, "args", None)
            if a is not None:
                for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
                    if p is not None:
                        stores[p.arg] = stores.get(p.arg, 0) + 1
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                k = (al.asname or al.name).split(".")[0]
                stores[k] = stores.get(k, 0) + 1
        elif isinstance(n, ast.ExceptHandler) and n.name:
            stores[n.name] = stores.get(n.name, 0) + 1
        elif isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            stores[n.id] = stores.get(n.id, 0) + 1
    out = {}
    for name, binds in cand.items():
        if len(binds) != 1 or stores.get(name, 0) != 1 or name in bad:
            continue
        line, vals = binds[0]
        ok = True
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)):
                continue
            p = parent.get(n)
            if n.lineno <= line:
                ok = False
            elif isinstance(p, ast.Subscript) and p.slice is n:
                pass
            elif isinstance(p, (ast.For, ast.AsyncFor, ast.comprehension)) and p.iter is n:
                pass
            elif isinstance(p, ast.Call) and n in p.args and isinstance(p.func, ast.Name) \
                    and p.func.id in LIST_SAFE_CALLS:
                pass
            elif isinstance(p, ast.Compare) and n in p.comparators and all(
                    isinstance(o, (ast.In, ast.NotIn)) for o in p.ops):
                pass
            else:
                ok = False
            if not ok:
                break
        if ok:
            out[name] = (line, vals)
    return out


class _Walker:
    def __init__(self, schema, state=None, exists=None, tolerant_keys=False, alias=False, locate=None,
                 list_consts=None, dir_writers=False):
        self.schema = {os.path.basename(k): set(v) for k, v in schema.items()}
        self.locate = locate              # gate v8, R3: base name -> known paths of files with that name
        self.list_consts = list_consts or {}   # gate v8, R2: name -> (line, tuple of str), this cell only
        self.dir_writers = dir_writers    # gate v8, R1: the cell has a call that may create a directory
        self.alias = alias
        self.groups = {}          # name -> shared set of names bound to the same object (alias mode)
        self.helpers = {}         # name -> FunctionDef of a pure loader function in this cell (fix 4)
        self.skip_call = None     # the helper call that the current statement binds as an alias
        if state is not None:
            self.frames = {k: (s, set(c), kind, ne) for k, (s, c, kind, ne) in state.frames.items()}
            self.consts = dict(state.consts)
        else:
            self.frames = {}      # name -> (source file, set of names, kind, non-empty)
            self.consts = {}      # name -> literal string, or ("handle" | "text", path)
        self.exists = exists
        self.tolerant = tolerant_keys
        self.substitutions = []   # dict keys that the executor would replace silently
        self.written = set()      # paths written earlier in this cell
        self.files = []           # (path, conditional, line) of every file read
        self.dirs = []            # gate v8: (path, conditional, line) of every directory call
        self.cwd_changed = False
        self.in_loop = False
        self.unknown_reasons = []
        self.definite, self.possible = [], []
        self.reads = 0

    def export(self):
        shared = {n for n, g in self.groups.items() if len(g) > 1}
        return ({k: (s, frozenset(c), kind, ne) for k, (s, c, kind, ne) in self.frames.items()
                 if k not in shared},
                dict(self.consts))

    def _untracked(self, why):
        """Gate v7: a read of a context file with arguments that may change the column
        names is not tracked, and the cell cannot be run-certified."""
        if why not in self.unknown_reasons:
            self.unknown_reasons.append(why)

    def forget(self, names):
        names = set(names)
        if self.alias:
            for n in list(names):
                names |= self.groups.get(n, set())
        for n in names:
            self.frames.pop(n, None)
            self.consts.pop(n, None)
            self.helpers.pop(n, None)
            g = self.groups.pop(n, None)
            if g is not None:
                g.discard(n)

    def is_alias(self, s, targets, value, conditional):
        """True for an unconditional `x = y` where y is tracked (alias mode only)."""
        return (self.alias and not conditional and isinstance(s, ast.Assign) and len(targets) == 1
                and isinstance(targets[0], ast.Name) and isinstance(value, ast.Name)
                and value.id in self.frames and value.id != targets[0].id)

    def lookup(self, name, env):
        return env[name] if name in env else self.frames.get(name)

    # ---------------------------------------------------------------- receipts
    def receipt(self, obj, name, lineno):
        src, cols, kind, _ = obj
        found = sorted(f for f, c in self.schema.items() if name in c and f != src)
        near = difflib.get_close_matches(name, sorted(cols), n=3, cutoff=0.7)
        what = "column" if kind == "frame" else "key"
        reason = f"{src} has no column '{name}'" if kind == "frame" else \
            f"no record in {src} has the key '{name}'"
        return {"location": f"line {lineno}", "kind": what, "source": src, "missing": name,
                "reason": reason, "available": sorted(cols), "found_in": found, "near_matches": near}

    def file_receipt(self, path, lineno):
        base = os.path.basename(path)
        near = difflib.get_close_matches(base, sorted(self.schema), n=3, cutoff=0.6)
        found = sorted(self.locate(base) or []) if self.locate else []     # gate v8, R3
        return {"location": f"line {lineno}", "kind": "file", "source": path, "missing": base,
                "reason": f"file '{path}' does not exist", "available": sorted(self.schema),
                "found_in": found, "near_matches": near}

    def dir_receipt(self, path, lineno):
        """Gate v8, R1: a directory that os.listdir / os.scandir / os.chdir needs."""
        dirs = set()
        if self.locate:
            for b in self.schema:
                dirs |= {os.path.dirname(p) or "." for p in (self.locate(b) or [])}
        return {"location": f"line {lineno}", "kind": "file", "source": path, "missing": path,
                "reason": f"directory '{path}' does not exist", "available": sorted(dirs),
                "found_in": [], "near_matches": difflib.get_close_matches(path, sorted(dirs), n=3, cutoff=0.6)}

    # ------------------------------------------------------------ file sources
    def path_of(self, expr):
        return _resolve(expr, self.consts) if expr is not None else None

    def text_path(self, expr, env):
        """Path of the file whose text `expr` is, or None."""
        if isinstance(expr, ast.Name):
            v = self.consts.get(expr.id)
            return v[1] if isinstance(v, tuple) and v[0] == "text" else None
        if not isinstance(expr, ast.Call):
            return None
        f = expr.func
        if isinstance(f, ast.Attribute) and f.attr in ("read", "read_text"):
            return self.handle_path(f.value) or (self.path_of(f.value) if f.attr == "read_text" else None)
        # A harness tool such as read_file(path) is not followed: in one
        # submission the agent defined read_file itself and returned invented data.
        return None

    def handle_path(self, expr):
        """Path of an open file handle, or None."""
        if isinstance(expr, ast.Name):
            v = self.consts.get(expr.id)
            return v[1] if isinstance(v, tuple) and v[0] == "handle" else None
        if isinstance(expr, ast.Call) and _call_name(expr) == "open" and _open_mode(expr) == "r":
            return self.path_of(_path_arg(expr))
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "open" \
                and _open_mode(expr) in ("r", None):
            return self.path_of(expr.func.value)    # Path(p).open()
        return None

    def context(self, path, suffix):
        """Base name of a context file with this suffix, or None."""
        if path is None:
            return None
        base = os.path.basename(path)
        return base if base in self.schema and base.endswith(suffix) else None

    def records_from(self, expr, env):
        """(source, keys) if expr evaluates to the records of a context file."""
        name = _call_name(expr)
        if name in ("json.load", "load") and isinstance(expr, ast.Call) and expr.args:
            base = self.context(self.handle_path(expr.args[0]), ".json")
        elif name in ("json.loads", "loads") and isinstance(expr, ast.Call) and expr.args:
            base = self.context(self.text_path(expr.args[0], env), ".json")
        else:
            base = None
        return (base, self.schema[base]) if base else None

    def elem(self, it, env):
        """(object, non-empty) for the elements of an iteration, ("pair", ...) for
        (index, element) pairs, or None."""
        if isinstance(it, ast.Name):
            o = self.lookup(it.id, env)
            if o and o[2] in ("records", "rows"):
                return (o[0], o[1], "record", True), o[3]
            if o and o[2] == "chunks":
                return (o[0], set(o[1]), "frame", True), o[3]
            return None
        if not isinstance(it, ast.Call):
            return None
        rec = self.records_from(it, env)
        if rec:
            return (rec[0], set(rec[1]), "record", True), True
        hr = self.records_of(it, env) if isinstance(it, ast.Call) else None
        if hr:                                           # for r in load(p) (fix 4)
            return (hr[0], set(hr[1]), "record", True), hr[3]
        name, f = _call_name(it), it.func
        if name in ("csv.DictReader", "DictReader") and it.args:
            base = self.context(self.handle_path(it.args[0]), ".csv")
            if base and _plain_dictreader(it):
                return (base, set(self.schema[base]), "record", True), True
            if base:
                self._untracked(f"csv.DictReader of {base} with arguments that may change the field names")
        if isinstance(f, ast.Attribute) and f.attr == "iterrows":
            o = self.frame_of(f.value, env)
            if o:
                return ("pair", (o[0], set(o[1]), "record", True)), o[3]
        if isinstance(f, ast.Attribute) and f.attr == "to_dict" and _records_arg(it) \
                and isinstance(f.value, ast.Name):
            o = self.lookup(f.value.id, env)
            if o and o[2] == "frame":
                return (o[0], set(o[1]), "record", True), o[3]
        if _reader_call(it) and any(k.arg in ("chunksize", "iterator") for k in it.keywords):
            base = self.context(self.path_of(_path_arg(it)), ".csv")
            # gate v7: names= and header=None are not applied to chunks, so such reads are not tracked
            why = _reader_args(it) or ("names= or header=None" if _kw(it, "names") is not None
                                       or _const_is(_kw(it, "header"), (None,)) else None)
            if base and not why:
                return (base, set(self.schema[base]), "frame", True), True
            if base:
                self._untracked(f"chunked read of {base}: {why}")
        if name == "enumerate" and it.args:
            inner = self.elem(it.args[0], env)
            if inner and inner[0][0] != "pair":
                return ("pair", inner[0]), inner[1]
        return None

    def bind_target(self, target, elem, env=None):
        """Bind a loop or comprehension target to the elements of an iteration."""
        put = (lambda k, v: env.__setitem__(k, v)) if env is not None else \
            (lambda k, v: self.frames.__setitem__(k, v))
        for n in _names(target):
            if env is not None:
                env[n] = None
            else:
                self.forget([n])
        if elem is None:
            return
        obj = elem[0]
        if obj[0] == "pair":
            if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2 \
                    and isinstance(target.elts[1], ast.Name):
                put(target.elts[1].id, obj[1])
        elif isinstance(target, ast.Name):
            put(target.id, obj)

    # ------------------------------------------------------------------- reads
    def check_reads(self, node, conditional):
        self._reads(node, conditional, {})

    def _reads(self, n, cond, env):
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load):
            self._check_sub(n, cond, env)
        if isinstance(n, ast.Call):
            self._check_file_call(n, cond)
            self._check_method(n, cond, env)
            self._check_helper_call(n, cond, env)
        if isinstance(n, ast.BoolOp):
            self._reads(n.values[0], cond, env)
            for v in n.values[1:]:
                self._reads(v, True, env)
            return
        if isinstance(n, ast.IfExp):
            self._reads(n.test, cond, env)
            self._reads(n.body, True, env)
            self._reads(n.orelse, True, env)
            return
        if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            self._reads(n.args, cond, env)
            inner = dict(env)
            a = n.args
            for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
                if p is not None:
                    inner[p.arg] = None
            if not isinstance(n, ast.Lambda):
                for x in ast.walk(n):
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                        inner[x.id] = None
                for d in n.decorator_list:
                    self._reads(d, cond, env)
            for b in (n.body if isinstance(n.body, list) else [n.body]):
                self._reads(b, True, inner)
            return
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self._comp(n, cond, env, eager=not isinstance(n, ast.GeneratorExp))
            return
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in CONSUMERS:
            for a in n.args:
                if isinstance(a, ast.GeneratorExp):
                    self._comp(a, cond, env, eager=True)
                else:
                    self._reads(a, cond, env)
            for k in n.keywords:
                self._reads(k.value, cond, env)
            return
        for ch in ast.iter_child_nodes(n):
            self._reads(ch, cond, env)

    def _comp(self, n, cond, env, eager):
        gens = n.generators
        self._reads(gens[0].iter, cond, env)
        inner = dict(env)
        el = self.elem(gens[0].iter, env)
        self.bind_target(gens[0].target, el, inner)
        first = eager and el is not None and el[1] and len(gens) == 1
        c_if = cond if first else True
        for f in gens[0].ifs:
            self._reads(f, c_if, inner)
        for g in gens[1:]:
            self._reads(g.iter, True, inner)
            self.bind_target(g.target, self.elem(g.iter, inner), inner)
            for f in g.ifs:
                self._reads(f, True, inner)
        c_elt = c_if if first and not gens[0].ifs else True
        for p in ([n.key, n.value] if isinstance(n, ast.DictComp) else [n.elt]):
            self._reads(p, c_elt, inner)

    # ------------------------------------------------ helper functions (fix 4)
    def register_helper(self, d, conditional):
        """Keep a module-level function with one parameter whose body only loads the file
        its parameter names, or returns the parameter."""
        if conditional or not isinstance(d, ast.FunctionDef) or d.decorator_list:
            return
        a = d.args
        if len(a.args) != 1 or a.posonlyargs or a.vararg or a.kwonlyargs or a.kwarg or a.defaults:
            return
        if _helper_pure(d):
            self.helpers[d.name] = d

    def helper_arg(self, expr, env):
        """("path", p) for a resolvable path, ("obj", name, object) for a tracked name, or None."""
        p = _resolve(expr, self.consts)
        if p is not None:
            return ("path", p)
        if isinstance(expr, ast.Name):
            o = self.lookup(expr.id, env)
            if o:
                return ("obj", expr.id, o)
        return None

    def _isinstance(self, arg, types):
        names = {t.split(".")[-1] for t in types}
        known = {"str", "bytes", "list", "dict", "tuple", "DataFrame", "Path", "PathLike", "PurePath"}
        if not names <= known:
            return None
        if arg[0] == "path":
            return bool(names & {"str"})
        kind = arg[2][2]
        want = {"records": "list", "record": "dict", "frame": "DataFrame"}.get(kind)
        if kind == "rows":
            return False if not names & {"list", "dict", "str", "tuple"} else None
        return want in names if want else None

    def eval_helper(self, d, arg):
        """Run a pure helper abstractly on one argument. Returns (result, opened): result is a
        tracked object, "same" (the argument itself) or None; opened lists the paths it opens."""
        p = d.args.args[0].arg
        path = arg[1] if arg and arg[0] == "path" else None
        local, opened = {}, []

        def test(t):
            if isinstance(t, ast.BoolOp):
                for v in t.values:
                    r = test(v)
                    if r is None:
                        return None
                    if isinstance(t.op, ast.And) and r is False:
                        return False
                    if isinstance(t.op, ast.Or) and r is True:
                        return True
                return isinstance(t.op, ast.And)
            if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
                r = test(t.operand)
                return None if r is None else not r
            if not isinstance(t, ast.Call) or arg is None:
                return None
            fn = _call_name(t)
            if fn == "isinstance" and len(t.args) == 2 and isinstance(t.args[0], ast.Name) and t.args[0].id == p:
                ty = t.args[1].elts if isinstance(t.args[1], ast.Tuple) else [t.args[1]]
                return self._isinstance(arg, [ast.unparse(x) for x in ty])
            if fn in ("os.path.exists", "os.path.isfile", "exists", "isfile") and t.args \
                    and isinstance(t.args[0], ast.Name) and t.args[0].id == p:
                return self.exists(path) if path is not None and self.exists is not None else None
            f = t.func
            if isinstance(f, ast.Attribute) and f.attr in ("endswith", "startswith") and path is not None and t.args:
                recv, low = f.value, False
                if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr == "lower":
                    recv, low = recv.func.value, True
                lit = _literal_names(t.args[0])
                if isinstance(recv, ast.Name) and recv.id == p and lit:
                    v = path.lower() if low else path
                    return any(getattr(v, f.attr)(x) for x in lit)
            return None

        def load_of(v):
            if not isinstance(v, ast.Call):
                return None
            fn = _call_name(v)
            if fn in ("json.load", "load") and len(v.args) == 1:
                h, src = v.args[0], None
                if isinstance(h, ast.Name) and isinstance(local.get(h.id), tuple) and local[h.id][0] == "handle":
                    src = local[h.id][1]
                elif isinstance(h, ast.Call) and _call_name(h) == "open" and h.args and isinstance(h.args[0], ast.Name) \
                        and h.args[0].id == p and _open_mode(h) in ("r", "rb") and path is not None:
                    src = path
                    opened.append(path)
                if src is None:
                    return None
                base = self.context(src, ".json")
                return (base, set(self.schema[base]), "records", True) if base else None
            if isinstance(v.func, ast.Attribute) and v.func.attr in ("read_csv", "read_json") and len(v.args) == 1 \
                    and isinstance(v.args[0], ast.Name) and v.args[0].id == p and path is not None \
                    and all(k.arg == "encoding" for k in v.keywords) and _reader_args(v) is None:
                opened.append(path)
                base = self.context(path, ".csv" if v.func.attr == "read_csv" else ".json")
                return (base, set(self.schema[base]), "frame", True) if base else None
            return None

        def value(v):
            if isinstance(v, ast.Name):
                if v.id == p:
                    return "same"
                o = local.get(v.id)
                return o if isinstance(o, tuple) and o[0] != "handle" and len(o) == 4 else None
            return load_of(v)

        def run(stmts):
            for st in stmts:
                if isinstance(st, (ast.Pass,)) or (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)):
                    continue
                if isinstance(st, ast.Return):
                    return ("ret", value(st.value) if st.value is not None else None)
                if isinstance(st, ast.If):
                    r = test(st.test)
                    if r is None:
                        return "unknown"
                    out = run(st.body if r else st.orelse)
                    if out is not None:
                        return out
                    continue
                if isinstance(st, ast.With):
                    it = st.items[0] if len(st.items) == 1 else None
                    c = it.context_expr if it else None
                    if not (isinstance(c, ast.Call) and _call_name(c) == "open" and c.args
                            and isinstance(c.args[0], ast.Name) and c.args[0].id == p
                            and _open_mode(c) in ("r", "rb") and path is not None):
                        return "unknown"
                    opened.append(path)
                    if isinstance(it.optional_vars, ast.Name):
                        local[it.optional_vars.id] = ("handle", path)
                    out = run(st.body)
                    if out is not None:
                        return out
                    continue
                if isinstance(st, ast.Assign):
                    local[st.targets[0].id] = value(st.value)
                    continue
                return "unknown"
            return None

        out = run(d.body)
        res = out[1] if isinstance(out, tuple) else None
        if res == "same" and not (arg and arg[0] == "obj"):
            res = None
        return res, opened

    def helper_value(self, call, env):
        """(object or "same", argument) for a call of a pure helper with one argument, else (None, None)."""
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in self.helpers
                and len(call.args) == 1 and not call.keywords):
            return None, None
        arg = self.helper_arg(call.args[0], env)
        if arg is None:
            return None, None
        res, _ = self.eval_helper(self.helpers[call.func.id], arg)
        return res, arg

    def _check_helper_call(self, n, cond, env):
        """A helper call that opens a file: the file must exist (like open())."""
        if not (isinstance(n.func, ast.Name) and n.func.id in self.helpers and len(n.args) == 1 and not n.keywords):
            return
        arg = self.helper_arg(n.args[0], env)
        if arg is None:
            return
        _, opened = self.eval_helper(self.helpers[n.func.id], arg)
        for path in opened[:1]:
            self.files.append((path, cond, n.lineno))
            if self.exists is None or path in self.written or (self.cwd_changed and not os.path.isabs(path)):
                continue
            if self.exists(path) is False:
                (self.possible if cond else self.definite).append(self.file_receipt(path, n.lineno))

    def records_of(self, expr, env):
        """Tracked records for a name or a helper call that returns records, else None."""
        if isinstance(expr, ast.Name):
            o = self.lookup(expr.id, env)
            return o if o and o[2] == "records" else None
        res, arg = self.helper_value(expr, env)
        if res == "same" and arg[2][2] == "records":
            return arg[2]
        return res if isinstance(res, tuple) and res[2] == "records" else None

    def _frame(self, expr, env):
        """The tracked frame that expr names or derives (gate v6, fix 2), or None."""
        return self.frame_of(expr, env)

    def frame_of(self, e, env, depth=0):
        """(source, columns, "frame", non-empty) for an expression whose result is a
        DataFrame with known columns, or None. The columns may include names the
        frame does not have (never the reverse), so a block stays exact:
          df[mask], df[a:b], df.loc[mask], df.loc[mask, ["a"]], df.iloc[rows], df[["a", "b"]],
          df.copy() / head / tail / sort_values / dropna / query / ... (SAME_COLUMNS),
          df.assign(x=...), df.drop(columns=...), df.rename(columns={...}),
          df.reset_index(drop=True) and df.reset_index() (adds "index" or "level_0"),
          df.merge(other, on=...) and pd.merge(a, b, on=...): keys, the other columns of
          both, and name + suffix for columns in both,
          pd.concat([a, b, ...]): all columns of all frames,
          df.groupby(keys)[cols].sum().reset_index(), groupby(..., as_index=False),
          .agg(name=(col, f)), .size().reset_index(name="n")."""
        if depth > 20:
            return None
        if isinstance(e, ast.Name):
            o = self.lookup(e.id, env)
            return o if o and o[2] == "frame" else None
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in self.helpers:
            res, arg = self.helper_value(e, env)           # df = load_frame(p) (fix 4)
            if res == "same":
                res = arg[2]
            return res if isinstance(res, tuple) and res[2] == "frame" else None
        if isinstance(e, ast.Subscript) and isinstance(e.ctx, ast.Load):
            v, sl = e.value, e.slice
            if isinstance(v, ast.Attribute) and v.attr in ("loc", "iloc"):
                base = self.frame_of(v.value, env, depth + 1)
                if base is None:
                    return None
                if isinstance(sl, ast.Tuple):
                    if v.attr == "iloc" or len(sl.elts) != 2:
                        return None
                    cols = sl.elts[1]
                    if isinstance(cols, ast.Slice) and cols.lower is None and cols.upper is None and cols.step is None:
                        return (base[0], set(base[1]), "frame", False)
                    names = _literal_list(cols) if isinstance(cols, (ast.List, ast.Tuple)) else None
                    if names is not None and all(n in base[1] for n in names):
                        return (base[0], set(names), "frame", False)
                    return None
                if v.attr == "iloc":
                    return (base[0], set(base[1]), "frame", False) if isinstance(sl, (ast.Slice, ast.List)) else None
                return (base[0], set(base[1]), "frame", False) if _is_mask(sl) or isinstance(sl, ast.Slice) else None
            base = self.frame_of(v, env, depth + 1)
            if base is None:
                return None
            if isinstance(sl, ast.List):
                names = _literal_list(sl)
                return (base[0], set(names), "frame", base[3]) if names is not None and all(
                    n in base[1] for n in names) else None
            if isinstance(sl, ast.Slice) or _is_mask(sl):
                return (base[0], set(base[1]), "frame", False)
            return None
        if not (isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute)):
            return None
        if any(isinstance(a, ast.Starred) for a in e.args) or any(k.arg is None for k in e.keywords):
            return None
        if _is_const(_kw(e, "inplace"), (True,)) or (_kw(e, "inplace") is not None
                                                      and not _is_const(_kw(e, "inplace"), (False,))):
            return None                             # inplace=True returns None
        m, recv = e.func.attr, e.func.value
        if isinstance(recv, ast.Name) and recv.id in ("pd", "pandas"):
            if m == "DataFrame" and len(e.args) == 1 and not e.keywords:
                rec = self.records_of(e.args[0], env)          # pd.DataFrame(load(p)) (fix 4)
                return (rec[0], set(rec[1]), "frame", rec[3]) if rec else None
            if m == "merge":
                left = e.args[0] if e.args else _kw(e, "left")
                right = e.args[1] if len(e.args) > 1 else _kw(e, "right")
                if left is None or right is None:
                    return None
                return self._merge_frame(self.frame_of(left, env, depth + 1), self.frame_of(right, env, depth + 1), e)
            if m == "concat" and e.args and isinstance(e.args[0], (ast.List, ast.Tuple)) and e.args[0].elts:
                parts = [self.frame_of(x, env, depth + 1) for x in e.args[0].elts]
                if all(parts):
                    return ("concat", set().union(*(p[1] for p in parts)), "frame", any(p[3] for p in parts))
            return None
        gb = self._groupby_frame(e, env, depth)
        if gb is not None:
            return gb
        base = self.frame_of(recv, env, depth + 1)
        if base is None:
            return None
        cols, ne = set(base[1]), base[3]
        if m in SAME_COLUMNS:
            axis = _kw(e, "axis")
            if axis is not None and not _is_const(axis, ROW_AXIS) and m not in ("sort_index",):
                return None
            return (base[0], cols, "frame", ne if m in KEEP_ROWS else False)
        if m == "reset_index":
            if e.args or any(k.arg not in ("drop",) for k in e.keywords):
                return None
            drop = _kw(e, "drop")
            if drop is not None and not isinstance(drop, ast.Constant):
                return None
            return (base[0], cols if _is_const(drop, (True,)) else cols | {"index", "level_0"}, "frame", ne)
        if m == "assign":
            return (base[0], cols | {k.arg for k in e.keywords}, "frame", ne) if not e.args else None
        if m == "drop":
            if _kw(e, "index") is not None and _kw(e, "columns") is None:
                return (base[0], cols, "frame", False)
            names = _literal_names(_kw(e, "columns")) if _kw(e, "columns") is not None else None
            if names is None and _is_const(_kw(e, "axis"), (1, "columns")):
                lab = e.args[0] if e.args else _kw(e, "labels")
                names = _literal_names(lab) if lab is not None else None
            return (base[0], cols - set(names), "frame", ne) if names else None
        if m == "rename":
            mp = _kw(e, "columns")
            if mp is None and _kw(e, "index") is not None and not e.args:
                return (base[0], cols, "frame", ne)
            if isinstance(mp, ast.Dict) and all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                                                and isinstance(v, ast.Constant) and isinstance(v.value, str)
                                                for k, v in zip(mp.keys, mp.values)):
                ren = {k.value: v.value for k, v in zip(mp.keys, mp.values)}
                return (base[0], {ren.get(c, c) for c in cols}, "frame", ne)
            return None
        if m == "merge":
            right = e.args[0] if e.args else _kw(e, "right")
            return self._merge_frame(base, self.frame_of(right, env, depth + 1), e) if right is not None else None
        return None

    def _merge_frame(self, lo, ro, e):
        """Columns of a merge on known keys; a column in both frames appears with and without suffixes."""
        if lo is None or ro is None:
            return None
        if any(k.arg in ("left_index", "right_index", "left_on", "right_on") for k in e.keywords):
            return None
        how = _kw(e, "how")
        if how is not None and not (isinstance(how, ast.Constant) and how.value in ("inner", "left", "right", "outer")):
            return None
        on = _kw(e, "on")
        if on is not None:
            keys = set(_literal_names(on))
            if not keys or (isinstance(on, (ast.List, ast.Tuple)) and len(keys) != len(on.elts)):
                return None
        else:
            keys = lo[1] & ro[1]
        suf = _kw(e, "suffixes")
        sx, sy = "_x", "_y"
        if suf is not None:
            lit = _literal_list(suf) if isinstance(suf, (ast.List, ast.Tuple)) else None
            if not lit or len(lit) != 2:
                return None
            sx, sy = lit
        cols = keys | (lo[1] - keys) | (ro[1] - keys)
        both = (lo[1] - keys) & (ro[1] - keys)
        if on is None:
            both |= keys        # the known columns may include names a frame lacks, so a guessed key may be suffixed
        cols |= {c + sx for c in both} | {c + sy for c in both}
        ind = _kw(e, "indicator")
        if ind is not None:
            if _is_const(ind, (True,)):
                cols.add("_merge")
            elif isinstance(ind, ast.Constant) and isinstance(ind.value, str):
                cols.add(ind.value)
            elif not _is_const(ind, (False,)):
                return None
        return ("merge", cols, "frame", False)

    def _groupby_frame(self, e, env, depth):
        """Columns of df.groupby(keys)[sel].<reduce>().reset_index() and similar, or None."""
        reset, rname = False, None
        call = e
        if e.func.attr == "reset_index":
            if e.args or any(k.arg != "name" for k in e.keywords):
                return None
            nm = _kw(e, "name")
            if nm is not None:
                if not (isinstance(nm, ast.Constant) and isinstance(nm.value, str)):
                    return None
                rname = nm.value
            reset, call = True, e.func.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
                return None
        agg, target = call.func.attr, call.func.value
        sel, series = None, False
        if isinstance(target, ast.Subscript):
            sl = target.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                sel, series = [sl.value], True
            elif isinstance(sl, ast.List) and _literal_list(sl) is not None:
                sel = _literal_list(sl)
            else:
                return None
            target = target.value
        if not (isinstance(target, ast.Call) and isinstance(target.func, ast.Attribute)
                and target.func.attr == "groupby"):
            return None
        base = self.frame_of(target.func.value, env, depth + 1)
        if base is None or _kw(target, "level") is not None or any(
                isinstance(a, ast.Starred) for a in target.args) or any(k.arg is None for k in target.keywords):
            return None
        by = target.args[0] if target.args else _kw(target, "by")
        keys = _literal_names(by) if by is not None else []
        if not keys or (isinstance(by, (ast.List, ast.Tuple)) and len(keys) != len(by.elts)) \
                or not all(k in base[1] for k in keys):
            return None
        if sel is not None and not all(c in base[1] for c in sel):
            return None
        as_index = _kw(target, "as_index")
        flat = reset or _is_const(as_index, (False,))
        if not flat or (as_index is not None and not isinstance(as_index, ast.Constant)):
            return None
        if rname is not None and not series and agg != "size":
            return None
        if agg in REDUCE and not call.args:
            vals = set(sel) if sel is not None else set(base[1]) - set(keys)
            if rname is not None:
                vals = {rname}
        elif agg == "size" and not call.args and not call.keywords:
            vals = {rname or ("size" if not reset else 0)}
        elif agg in ("agg", "aggregate") and not call.args and call.keywords and all(
                isinstance(k.value, ast.Tuple) and len(k.value.elts) == 2 for k in call.keywords):
            vals = {k.arg for k in call.keywords}
        elif agg in ("agg", "aggregate") and len(call.args) == 1 and not call.keywords \
                and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            vals = set(sel) if sel is not None else set(base[1]) - set(keys)
        else:
            return None
        return (base[0], set(keys) | vals, "frame", False)

    def _check_columns(self, obj, names, lineno, cond):
        if not names:
            return
        self.reads += 1
        for k in names:
            if k not in obj[1]:
                self.violation(obj, k, lineno, cond)

    def _check_method(self, n, cond, env):
        """Column names that a pandas method requires (gate v6, fix 1)."""
        f = n.func
        if not isinstance(f, ast.Attribute):
            return
        if any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords):
            return                                  # *args or **kwargs: the arguments are not known
        m = f.attr
        if m == "merge":
            if isinstance(f.value, ast.Name) and f.value.id in ("pd", "pandas"):
                left = n.args[0] if n.args else _kw(n, "left")
                right = n.args[1] if len(n.args) > 1 else _kw(n, "right")
            else:
                left, right = f.value, (n.args[0] if n.args else _kw(n, "right"))
            lo = self._frame(left, env) if left is not None else None
            ro = self._frame(right, env) if right is not None else None
            on = _kw(n, "on")
            for obj, key in ((lo, "left_on"), (ro, "right_on")):
                if obj is None:
                    continue
                names = _literal_names(on) if on is not None else []
                v = _kw(n, key)
                if v is not None:
                    names += _literal_names(v)
                self._check_columns(obj, names, n.lineno, cond)
            return
        obj = self._frame(f.value, env)
        if obj is None:
            return
        if m in COLUMN_ARGS:
            axis = _kw(n, "axis")
            if axis is not None and not _is_const(axis, ROW_AXIS):
                return
            if m == "groupby" and _kw(n, "level") is not None:
                return
            pos, kws = COLUMN_ARGS[m]
            names = _literal_names(n.args[pos]) if pos is not None and len(n.args) > pos else []
            for k in kws:
                v = _kw(n, k)
                if v is not None:
                    names += _literal_names(v)
            self._check_columns(obj, names, n.lineno, cond)
        elif m == "drop":
            errors = _kw(n, "errors")
            if errors is not None and not _is_const(errors, ("raise",)):
                return
            cols = _kw(n, "columns")
            names = _literal_names(cols) if cols is not None else []
            if _is_const(_kw(n, "axis"), (1, "columns")):
                lab = n.args[0] if n.args else _kw(n, "labels")
                if lab is not None:
                    names += _literal_names(lab)
            self._check_columns(obj, names, n.lineno, cond)

    def _check_sub(self, n, cond, env):
        v, obj = n.value, None
        if isinstance(v, ast.Attribute) and v.attr in ("loc", "at") and isinstance(n.slice, ast.Tuple) \
                and len(n.slice.elts) == 2:
            fr = self._frame(v.value, env)        # df.loc[rows, 'k'], df.at[i, 'k'] (gate v6, fix 1)
            if fr is not None:
                col = n.slice.elts[1]
                self._check_columns(fr, _literal_names(col) if not isinstance(col, ast.Slice) else [],
                                    n.lineno, cond)
            return
        if isinstance(v, ast.Name):
            obj = self.lookup(v.id, env)
        elif isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and isinstance(v.slice, ast.Constant) \
                and type(v.slice.value) is int and v.slice.value in (0, -1):
            base = self.lookup(v.value.id, env)
            if base and base[2] == "records" and base[3]:
                obj = (base[0], base[1], "record", True)
        elif isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "groupby":
            obj = self.frame_of(v.func.value, env)     # df.groupby(...)["col"] selects a column (fix 2)
        elif isinstance(v, (ast.Subscript, ast.Call)):
            obj = self.frame_of(v, env)                # df[mask]["col"], df.merge(...)["col"] (fix 2)
        if not obj or obj[2] not in ("frame", "record"):
            return
        keys = _subscript_keys(n)
        one_name = False
        if keys is None and isinstance(n.slice, ast.Name) and n.slice.id not in env:
            v = self.consts.get(n.slice.id)             # gate v8, R2: df[col] with col = "x"
            lc = self.list_consts.get(n.slice.id)
            if isinstance(v, str):
                keys, one_name = [v], True
            elif lc is not None and obj[2] == "frame" and n.lineno > lc[0]:
                keys = list(lc[1])                      # df[cols] with cols = ["a", "b"]
        if not keys or (obj[2] == "record" and not (isinstance(n.slice, ast.Constant) or one_name)):
            return
        self.reads += 1
        for k in keys:
            if k not in obj[1]:
                self.violation(obj, k, n.lineno, cond)

    def violation(self, obj, k, lineno, cond):
        """A missing column or key. With tolerant_keys, a missing dict key that has
        a close match is replaced silently by the executor (smolagents 1.0-1.4,
        evaluate_subscript), so it is not an error."""
        if obj[2] == "record" and self.tolerant:
            near = difflib.get_close_matches(k, sorted(obj[1]))
            if near:
                self.substitutions.append({"location": f"line {lineno}", "source": obj[0], "written": k,
                                           "used": near[0], "conditional": cond})
                return
        (self.possible if cond else self.definite).append(self.receipt(obj, k, lineno))

    def _check_file_call(self, n, cond):
        """Record writes; check that a read file exists."""
        name, f = _call_name(n), n.func
        if isinstance(f, ast.Attribute) and f.attr in WRITERS:
            p = self.path_of(_path_arg(n)) if f.attr != "write_text" else self.path_of(f.value)
            if p is not None:
                self.written.add(p)
            return
        if name in ("os.listdir", "os.scandir", "os.chdir", "chdir"):
            if name.startswith("os."):            # gate v8, R1
                self._check_dir_call(n, name, cond)
            if name in ("os.chdir", "chdir"):
                self.cwd_changed = True
            return
        path = None
        if name == "open":
            mode = _open_mode(n)
            p = self.path_of(_path_arg(n))
            if mode is None or any(c in mode for c in "wax+"):
                if p is not None:
                    self.written.add(p)
                return
            path = p
        elif isinstance(f, ast.Attribute) and f.attr in FILE_READERS:
            path = self.path_of(_path_arg(n))
        elif isinstance(f, ast.Attribute) and f.attr in ("read_text", "read_bytes"):
            path = self.path_of(f.value)
        if path is None:
            return
        self.files.append((path, cond, n.lineno))
        if self.exists is None or path in self.written or (self.cwd_changed and not os.path.isabs(path)):
            return
        if self.exists(path) is False:
            (self.possible if cond else self.definite).append(self.file_receipt(path, n.lineno))

    def _check_dir_call(self, n, name, cond):
        """Gate v8, R1: os.listdir(p), os.scandir(p) and os.chdir(p) raise FileNotFoundError at the
        call when p does not exist. Not checked: a call without a path, a cell that may create a
        directory (DIR_WRITERS), a path written in this cell, a relative path after os.chdir."""
        arg = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "path"), None)
        path = self.path_of(arg) if arg is not None else None
        if path is not None:
            self.dirs.append((path, cond, n.lineno))      # for replay evidence, as result["files"]
        if path is None or self.exists is None or self.dir_writers \
                or (self.cwd_changed and not os.path.isabs(path)):
            return
        p = os.path.normpath(path)
        if any(os.path.normpath(w) == p or os.path.normpath(w).startswith(p.rstrip("/") + "/")
               for w in self.written):
            return
        if self.exists(path) is False:
            (self.possible if cond else self.definite).append(self.dir_receipt(path, n.lineno))

    # ----------------------------------------------------------------- effects
    def helper_escapes(self, node):
        """A helper call that returns its tracked argument makes that object reachable by
        another name, unless the result goes straight into a loop or a copying call, or
        the statement binds it as an alias (alias mode). Stop tracking the argument then."""
        if not self.helpers:
            return
        par = {}
        for n in _walk_no_defs(node):
            for c in ast.iter_child_nodes(n):
                par[c] = n
        for n in _walk_no_defs(node):
            if n is self.skip_call or not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                            and n.func.id in self.helpers):
                continue
            res, arg = self.helper_value(n, {})
            if res != "same":
                continue
            p = par.get(n)
            if (isinstance(p, ast.Call) and n in p.args and _call_name(p) in SAFE_CONSUMERS) or \
                    (isinstance(p, (ast.For, ast.comprehension)) and p.iter is n):
                continue
            self.forget_source(arg[1])

    def effects(self, node):
        """Stop tracking objects that this code may change by another path."""
        self.helper_escapes(node)
        for n in _walk_no_defs(node):
            if isinstance(n, ast.comprehension):
                for name in _display_escapes(n.iter) & set(self.frames):
                    self.forget_source(name)
            if isinstance(n, ast.NamedExpr):
                self.forget(_names(n.target))
            elif isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)) \
                    and isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name):
                self.forget([n.value.value.id])        # df.loc[...] = ...
            elif isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name) and f.id in UNTRACKABLE:
                    self.frames.clear()
                    self.consts.clear()
                    continue
                args = [*n.args, *(k.value for k in n.keywords)]
                bare = {a.id for a in args if isinstance(a, ast.Name)} | \
                    {a.value.id for a in args if isinstance(a, ast.Starred) and isinstance(a.value, ast.Name)}
                if isinstance(f, ast.Attribute):
                    obj = f.value
                    if isinstance(obj, ast.Name) and obj.id in self.frames and (
                            f.attr in MUTATING_METHODS or any(k.arg == "inplace" for k in n.keywords)):
                        self.forget_source(obj.id)
                    if f.attr in ESCAPE_METHODS:
                        self.forget(bare & set(self.frames))   # other.append(df)
                elif isinstance(f, ast.Name):
                    if f.id not in PURE_FUNCTIONS and f.id not in self.helpers:
                        self.forget(bare & set(self.frames))   # a user function may change it
                else:
                    self.forget(bare & set(self.frames))

    def forget_source(self, name):
        """Stop tracking `name`; for records, also every record of the same file."""
        obj = self.frames.get(name)
        self.forget([name])
        if obj and obj[2] in ("records", "record"):
            self.forget([k for k, o in self.frames.items() if o[0] == obj[0] and o[2] in ("records", "record")])

    def add_keys(self, name, keys):
        obj = self.frames[name]
        if obj[2] in ("records", "record"):
            for o in self.frames.values():
                if o[0] == obj[0] and o[2] in ("records", "record"):
                    o[1].update(keys)
        else:
            obj[1].update(keys)

    def def_effects(self, d):
        """A function or class body may run later and change an object it names."""
        hit = set()
        for n in ast.walk(d):
            if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.ctx, (ast.Store, ast.Del)):
                hit |= _names(n.value)
            elif isinstance(n, ast.Global):
                hit |= set(n.names)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and isinstance(n.func.value, ast.Name) and (
                        n.func.attr in INPLACE_METHODS | MUTATING_METHODS
                        or any(k.arg == "inplace" for k in n.keywords)):
                hit.add(n.func.value.id)
        for name in hit & (set(self.frames) | set(self.consts)):
            self.forget_source(name)

    # ---------------------------------------------------------------- bindings
    def bind_reader(self, target, call, lineno):
        arg = _path_arg(call)
        if arg is None:
            self.unknown_reasons.append(f"line {lineno}: reader without a path")
            self.forget([target])
            return
        path = _resolve(arg, self.consts)
        if path is None:
            self.unknown_reasons.append(f"line {lineno}: dynamic path")
            self.forget([target])
            return
        base = os.path.basename(path)
        if base not in self.schema:
            self.unknown_reasons.append(
                f"line {lineno}: {base} is not in the schema (workspace file?)")
            self.forget([target])
            return
        kws = {k.arg: k.value for k in call.keywords}
        if "header" in kws and isinstance(kws["header"], ast.Constant) and kws["header"].value is None:
            self.unknown_reasons.append(
                f"line {lineno}: header=None gives integer column names")
            self.forget([target])
            return
        why = _reader_args(call)
        if why:                                       # gate v7, rule 2
            self.unknown_reasons.append(f"line {lineno}: {why}")
            self.forget([target])
            return
        self.reads += 1
        cols = set(self.schema[base])
        if "names" in kws:
            names = _literal_list(kws["names"])
            if names is None:
                self.unknown_reasons.append(
                    f"line {lineno}: non-literal names=")
                self.forget([target])
                return
            cols = set(names)
        for kw in call.keywords:
            if kw.arg == "usecols":
                lits = _literal_list(kw.value)
                if lits is None:
                    self.unknown_reasons.append(
                        f"line {lineno}: non-literal usecols")
                    self.forget([target])
                    return
                missing = [c for c in lits if c not in cols]
                for c in missing:
                    self.definite.append({"location": f"line {lineno}", "kind": "column", "source": base,
                                          "missing": c,
                                          "reason": f"usecols requests '{c}', which {base} does not have",
                                          "available": sorted(cols),
                                          "found_in": sorted(f for f, s in self.schema.items() if c in s and f != base),
                                          "near_matches": difflib.get_close_matches(c, sorted(cols), n=3, cutoff=0.7)})
                cols = set(lits) & cols
        nrows = kws.get("nrows")
        nonempty = not (isinstance(nrows, ast.Constant) and nrows.value == 0)
        kind = "chunks" if any(k in kws for k in ("chunksize", "iterator")) else "frame"
        self.consts.pop(target, None)
        self.frames[target] = (base, cols, kind, nonempty)

    def bind_value(self, target, value, lineno):
        """Bind `target = value` (unconditional). Return True if handled."""
        if self.alias and isinstance(value, ast.Name) and value.id in self.frames and value.id != target:
            self.consts.pop(target, None)
            self.frames[target] = self.frames[value.id]     # the same object: shared names
            g = self.groups.get(value.id) or {value.id}
            g.add(target)
            for n in g:
                self.groups[n] = g
            return True
        if _reader_call(value):
            self.bind_reader(target, value, lineno)
            return True
        res, arg = self.helper_value(value, {})
        if isinstance(res, tuple):                       # recs = load(p) (fix 4)
            self.consts.pop(target, None)
            self.frames[target] = (res[0], set(res[1]), res[2], res[3])
            return True
        if res == "same" and self.alias and arg[1] != target:   # recs = load(recs_or_path), alias mode
            return self.bind_value(target, ast.Name(id=arg[1], ctx=ast.Load()), lineno)
        rec = self.records_from(value, {})
        if rec:
            self.reads += 1
            self.consts.pop(target, None)
            self.frames[target] = (rec[0], set(rec[1]), "records", True)
            return True
        name, f = _call_name(value), getattr(value, "func", None)
        if name in ("csv.DictReader", "DictReader") and value.args:
            base = self.context(self.handle_path(value.args[0]), ".csv")
            if base and _plain_dictreader(value):
                self.consts.pop(target, None)
                self.frames[target] = (base, set(self.schema[base]), "rows", True)
                return True
            if base:                                   # gate v7, rule 2
                self._untracked(f"line {lineno}: csv.DictReader of {base} with arguments that may change "
                                "the field names")
        if name == "list" and value.args and len(value.args) == 1:
            el = self.elem(value.args[0], {})
            if el and el[0][0] != "pair" and el[0][2] == "record":
                self.consts.pop(target, None)
                self.frames[target] = (el[0][0], set(el[0][1]), "records", el[1])
                if isinstance(value.args[0], ast.Name) and value.args[0].id in self.frames:
                    o = self.frames[value.args[0].id]
                    if o[2] == "rows":        # the reader is used up
                        self.frames[value.args[0].id] = (o[0], o[1], o[2], False)
                return True
        if isinstance(f, ast.Attribute) and f.attr == "to_dict" and _records_arg(value) \
                and isinstance(f.value, ast.Name) and self.frames.get(f.value.id, (0, 0, 0))[2] == "frame":
            o = self.frames[f.value.id]
            self.consts.pop(target, None)
            self.frames[target] = (o[0], set(o[1]), "records", o[3])
            return True
        if isinstance(f, ast.Attribute) and f.attr == "DataFrame" and value.args \
                and isinstance(value.args[0], ast.Name) and len(value.args) == 1 and not value.keywords:
            o = self.frames.get(value.args[0].id)
            if o and o[2] == "records":
                self.consts.pop(target, None)
                self.frames[target] = (o[0], set(o[1]), "frame", o[3])
                return True
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) \
                and isinstance(value.slice, ast.Constant) and type(value.slice.value) is int \
                and value.slice.value in (0, -1):
            o = self.frames.get(value.value.id)
            if o and o[2] == "records" and o[3]:
                self.consts.pop(target, None)
                self.frames[target] = (o[0], o[1], "record", True)
                return True
        if isinstance(value, ast.Call) and _call_name(value) == "open" and _open_mode(value) == "r":
            p = self.path_of(_path_arg(value))
            if p is not None:
                self.frames.pop(target, None)
                self.consts[target] = ("handle", p)
                return True
        tp = self.text_path(value, {})
        if tp is not None:
            self.frames.pop(target, None)
            self.consts[target] = ("text", tp)
            return True
        if not isinstance(value, ast.Name):
            fr = self.frame_of(value, {})             # t = df[mask], t = a.merge(b, on=...), ... (fix 2)
            if fr is not None:
                self.consts.pop(target, None)
                self.frames[target] = (fr[0], set(fr[1]), "frame", fr[3])
                return True
        return False

    # -------------------------------------------------------------- statements
    def block(self, stmts, conditional):
        """Process a statement list. Return CELL if it always ends the cell,
        LOOP if it always ends the current loop iteration, else None."""
        for s in stmts:
            if self.in_loop and isinstance(s, (ast.Break, ast.Continue)):
                return LOOP
            r = self.stmt(s, conditional)
            if r:
                return r
            if not conditional and (_may_end(s) or (self.in_loop and _may_skip(s))):
                conditional = True
        return None

    def loop_body(self, stmts, conditional):
        saved, self.in_loop = self.in_loop, True
        try:
            return self.block(stmts, conditional)
        finally:
            self.in_loop = saved

    def stmt(self, s, conditional):
        """Process one statement. Return CELL or LOOP if it always ends that scope."""
        # Reads in the statement are checked before its own assignment takes effect.
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.check_reads(s, True)
            self.def_effects(s)
            self.forget([s.name])
            self.register_helper(s, conditional)
            return None
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            self.forget([(a.asname or a.name).split(".")[0] for a in s.names])
            return None
        if isinstance(s, ast.Delete):
            for t in s.targets:
                self.check_reads(t, conditional)
                self.forget(_names(t))
            return None
        if isinstance(s, (ast.For, ast.AsyncFor)):
            self.check_reads(s.iter, conditional)
            self.effects(s.iter)
            for name in _display_escapes(s.iter) & set(self.frames):
                self.forget_source(name)          # for df in (a, b): the loop target is each object
            el = self.elem(s.iter, {})
            self.bind_target(s.target, el)
            first = el is not None and el[1] and not conditional
            r = self.loop_body(s.body, not first)
            if isinstance(s.iter, ast.Name) and s.iter.id in self.frames and self.frames[s.iter.id][2] in ("rows", "chunks"):
                o = self.frames[s.iter.id]      # a reader can be read once
                self.frames[s.iter.id] = (o[0], o[1], o[2], False)
            self.block(s.orelse, True)
            return CELL if first and r == CELL else None
        if isinstance(s, (ast.If, ast.While, ast.Try)) or type(s).__name__ == "TryStar":
            h = getattr(s, "test", None)
            if h is not None:
                self.check_reads(h, conditional)
                self.effects(h)
            body = self.loop_body if isinstance(s, ast.While) else self.block
            body(s.body, True)
            for field in ("orelse", "finalbody"):
                self.block(getattr(s, field, []) or [], True)
            for hd in getattr(s, "handlers", []) or []:
                if hd.name:
                    self.forget([hd.name])
                self.block(hd.body, True)
            return None
        if isinstance(s, ast.Match):
            self.check_reads(s.subject, conditional)
            self.effects(s.subject)
            for case in s.cases:
                self.forget(_names(case.pattern))
                self.block(case.body, True)
            return None
        if isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                self.check_reads(it.context_expr, conditional)
                self.effects(it.context_expr)
                if it.optional_vars is not None:
                    self.forget(_names(it.optional_vars))
                    p = self.handle_path(it.context_expr)
                    if p is not None and isinstance(it.optional_vars, ast.Name) and not conditional:
                        self.consts[it.optional_vars.id] = ("handle", p)
            return self.block(s.body, conditional)

        value = getattr(s, "value", None)
        self.skip_call = None
        if self.helpers and self.alias and not conditional and isinstance(s, ast.Assign) and len(s.targets) == 1 \
                and isinstance(s.targets[0], ast.Name) and isinstance(value, ast.Call):
            res, _ = self.helper_value(value, {})
            if res == "same":
                self.skip_call = value
        if value is not None:
            self.check_reads(value, conditional)
        if isinstance(s, ast.Assert):
            self.check_reads(s.test, conditional)
        targets = s.targets if isinstance(s, ast.Assign) else \
            [s.target] if isinstance(s, (ast.AnnAssign, ast.AugAssign)) else []
        for t in targets:
            if not isinstance(t, ast.Name):
                self.check_reads(t, conditional)   # df.loc[df["x"] > 0, "y"] = ...
        if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Subscript) \
                and isinstance(s.target.value, ast.Name) and s.target.value.id in self.frames:
            obj = self.frames[s.target.value.id]   # df["c"] += 1 reads df["c"] first
            if obj[2] in ("frame", "record"):
                for k in _subscript_keys(s.target) or []:
                    self.reads += 1
                    if k not in obj[1]:
                        self.violation(obj, k, s.lineno, conditional)
        self.effects(s)
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute):
            obj = s.value.func.value
            if isinstance(obj, ast.Name) and obj.id in self.frames and (
                    s.value.func.attr in INPLACE_METHODS
                    or any(k.arg == "inplace" for k in s.value.keywords)):
                self.forget_source(obj.id)
        if targets and value is not None and not self.is_alias(s, targets, value, conditional):
            # x = df, x = [df, ...]: df becomes reachable through another name
            for n in {n for n in _escaping(value) if n in self.frames} \
                    - {t.id for t in targets if isinstance(t, ast.Name)}:
                self.forget_source(n)
        for t in targets:
            if isinstance(t, ast.Name):
                self.helpers.pop(t.id, None)
                if value is not None and not conditional and not isinstance(s, ast.AugAssign) \
                        and self.bind_value(t.id, value, s.lineno):
                    continue
                self.frames.pop(t.id, None)
                p = _resolve(value, self.consts) if value is not None else None
                if p is not None and not conditional and isinstance(s, ast.Assign):
                    self.consts[t.id] = p        # also os.path.join(...), f-strings, Path(...)
                else:
                    self.consts.pop(t.id, None)
            elif isinstance(t, ast.Subscript):
                self.store_subscript(t)
            elif isinstance(t, (ast.Tuple, ast.List, ast.Starred)):
                for n in _names(t):
                    self.forget_source(n)
            elif isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                self.forget_source(t.value.id)
        return CELL if isinstance(s, ast.Raise) or (isinstance(s, ast.Expr) and _is_end_call(s.value)) else None

    def store_subscript(self, t):
        """obj[...] = value: add a column or key, or stop tracking."""
        base = t.value
        while isinstance(base, ast.Subscript):
            base = base.value
        if not (isinstance(base, ast.Name) and base.id in self.frames):
            return
        obj = self.frames[base.id]
        keys = _subscript_keys(t)
        if isinstance(t.value, ast.Name):
            if obj[2] in ("frame", "record") and keys:
                self.add_keys(base.id, keys)
            elif obj[2] in ("frame", "record"):
                self.forget_source(base.id)
            else:
                self.forget_source(base.id)          # recs[0] = {...}
        elif obj[2] in ("records", "record"):        # recs[0]["k"] = ..., rec["a"]["b"] = ...
            if keys and isinstance(t.value, ast.Subscript) and isinstance(t.value.value, ast.Name) \
                    and obj[2] == "records":
                self.add_keys(base.id, keys)
            elif obj[2] == "records":
                self.forget_source(base.id)
        # a chained assignment on a frame (df["a"]["b"] = ...) adds no column


def decide(code, schema, state=None, exists=None, tolerant_keys=False, alias=False, locate=None):
    """locate (gate v8, R3): optional callable, base name -> known paths of files with that name,
    listed in file receipts."""
    if state is not None:
        state.pending = None
    try:
        with warnings.catch_warnings():
            # agent code often has invalid escapes such as "\d"
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError) as e:
        return {"outcome": "unknown", "reason": f"invalid Python: {type(e).__name__}", "files": [],
                "written": [], "substitutions": []}
    w = _Walker(schema, state, exists, tolerant_keys, alias, locate=locate,
                list_consts=_list_consts(tree), dir_writers=_dir_writers(tree))
    w.block(tree.body, False)
    if state is not None:
        state.pending = w.export()
    files = w.files
    extra = {"files": files, "dirs": w.dirs, "written": sorted(w.written), "substitutions": w.substitutions}
    if w.definite:
        first = w.definite[0]
        return {"outcome": "block", **first, "violations": w.definite, **extra}
    if w.reads == 0 and not w.possible:
        reason = "; ".join(
            w.unknown_reasons) or "no supported source read in this cell"
        return {"outcome": "unknown", "reason": reason, **extra}
    if w.possible:
        return {"outcome": "unknown", "reason": "violation only in code that may not run",
                "violations": w.possible, **extra}
    if w.unknown_reasons:
        return {"outcome": "unknown", "reason": "; ".join(w.unknown_reasons), **extra}
    return {"outcome": "run", "reads": w.reads, **extra}


if __name__ == "__main__":
    demo = "import pandas as pd\npayments = pd.read_csv('payments.csv')\npayments['account_type']"
    print(decide(demo, {"payments.csv": {"merchant"},
          "merchant_data.json": {"merchant", "account_type"}}))
