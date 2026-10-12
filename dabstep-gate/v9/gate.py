#!/usr/bin/env python3
"""Exact pre-execution checker for agent-generated pandas code, version 9 (release 4).

Version 9 corrects gaps in the model of Python and pandas of version 8 (frozen
release 3, dabstep-gate/gate.py). The gaps were found by programs that a reader
sent and by a differential test against pandas 1.5, 2.2 and 3.0
(differential_test.py). Version 8 is unchanged; this file is a separate release.
Every change makes the checker block less, never more. The changes:

  Control flow (code that may not run)
    C1  In a chained comparison a < b < c, the links after the first may not run.
    C2  In a comprehension, the conditions after the first `if` may not run.
    C3  A `with` block whose context manager may suppress an error (anything
        but open() and a few standard managers, SAFE_CONTEXTS) may not run to its end.
    C4  A function, lambda or class defined in the trace, in this step or an
        earlier one, that may call final_answer, sys.exit or os._exit (or a
        function that does), is an "ender": a call of it may end the step. So
        is a name bound to an ender (stop = final_answer; from sys import
        exit as stop; import sys as s; s.exit()).
    C5  An ender used as a value (a decorator, an argument such as
        map(final_answer, ...)), exec, eval, compile, __import__ and getattr on a
        module or with a computed name may end the step at a point the checker
        cannot see: the rest of that statement and every later statement may
        not run. A statement that calls an ender anywhere but as its whole
        value (final_answer(x) or y = final_answer(x)) has no checked reads.
        Builtins that the trace rebinds (any = ..., list = ..., print = ...)
        lose their special meaning.
  Names that pandas may read as something else
    N1  df.groupby([...]) with a list: pandas reads the list as one group
        label per row when it is as long as the frame, so a missing name in a
        list gives unknown. pivot_table index= and columns= the same.
    N2  drop_duplicates(subset=...) on a frame that may be empty does not
        raise: unknown unless the frame is known to be non-empty.
    N3  pd.concat with keys=, names= or levels= is not followed (it builds
        hierarchical labels).
    N4  Index level names count as names for groupby, sort_values and merge.
        Any change of an index name (x.name = ..., x.names = ...,
        rename_axis/set_names/set_axis/rename with inplace) stops the checker
        from following every frame, because frames can share an index.
  Objects changed by another path (aliasing)
    A1  A tracked object that escapes (bound to another name, put in a
        container or an attribute, returned, passed to a function or method
        that is not known to leave it unchanged, df.loc taken as a value, a
        record reached by recs[i], next(...), sorted(...) and similar) is
        no longer followed; for records, no record of that file is followed.
        pd.DataFrame(df) counts as an escape (pandas 1.5 shares the data).
    A2  A function, lambda or class body that may change a tracked object,
        through any name (an alias in the body, a default argument, a loop
        variable over records, global), stops the checker from following it.
    A3  A store through .columns, .index or .values of any object stops the
        checker from following every frame. __setitem__, __delitem__ and
        similar method calls count as changes.
  Files
    F1  A step that may create a file the checker cannot name (to_string,
        to_html, savefig, np.save, Path.touch, shutil, os.rename, an open()
        for writing of a path it cannot resolve, ...) gets no file block.
    F2  URLs (scheme://...) are not files. pandas readers expand "~", so a
        path that starts with "~" is not checked. read_json given literal JSON
        text is not a file read.
    F3  Path(p).open(mode) takes the mode as its first argument. open() counts
        only when the trace does not rebind it.
    F4  os.chdir, os.makedirs and the readers are recognised through import
        aliases (from os import chdir as cd). A change of directory in an
        earlier step stops relative-path checks in later steps.
  Records and readers
    R1  A file handle that the code uses before json.load or csv.DictReader
        reads it (fh.readline(), next(fh), ...) is not followed: the reader
        may see a different header.
    R2  A csv.DictReader used in any way before a loop (next(rd), ...) may be
        empty in that loop. read_csv(nrows=...) gives a frame that may be
        empty unless nrows is a positive literal.
  Other
    O1  Readers are pandas readers only: pd.read_csv with pd bound to pandas
        (or from pandas import read_csv); json.load only through json.
    O2  Receipts sort names with key=str (a frame can have integer column
        names, for example after size().reset_index()).

The checker never runs the code. decide(code, schema, state=None, exists=None,
tolerant_keys=False, alias=False, locate=None) has the same arguments and
result as version 8: see dabstep-gate/gate.py for the full description of the
tracked objects, checks, control flow, state and decisions.
"""
import ast
import difflib
import os
import threading
import warnings

READERS = {"read_csv", "read_json"}
FILE_READERS = READERS | {"read_excel", "read_parquet", "read_table", "read_feather", "read_pickle"}
WRITERS = {"to_csv", "to_json", "to_excel", "to_parquet", "to_pickle", "to_feather", "write_text"}
PATH_KWARGS = ("filepath_or_buffer", "path_or_buf", "io", "path", "file")
READ_CSV_ANY_VALUE = frozenset({
    "dtype", "usecols", "names", "nrows", "chunksize", "iterator", "low_memory", "na_values",
    "keep_default_na", "na_filter", "date_format", "dayfirst", "cache_dates", "thousands", "decimal",
    "converters", "true_values", "false_values", "float_precision", "memory_map", "on_bad_lines",
    "compression", "storage_options", "encoding_errors", "dtype_backend", "verbose", "engine"})
READ_JSON_ANY_VALUE = frozenset({"dtype", "convert_dates", "keep_default_dates", "precise_float", "date_unit"})
UTF8_NAMES = frozenset({"utf-8", "utf8", "utf_8", "u8", "utf-8-sig", "utf_8_sig"})
INPLACE_METHODS = {"insert", "pop", "drop", "rename", "set_index", "reset_index",
                   "update", "join", "pipe"}
# v9, A3: dunder methods that change the object too
MUTATING_METHODS = {"insert", "pop", "update", "pipe", "append", "extend", "setdefault",
                    "clear", "remove", "popitem", "__setitem__", "__delitem__", "__setattr__", "__delattr__",
                    "__init__", "__iadd__", "__ior__"}
ESCAPE_METHODS = {"append", "extend", "insert", "add", "setdefault", "update", "put", "__setitem__", "appendleft"}
PURE_FUNCTIONS = {"print", "len", "display", "str", "repr", "type", "isinstance", "id",
                  "list", "tuple", "set", "dict", "sorted", "reversed", "enumerate",
                  "zip", "min", "max", "sum", "any", "all", "bool", "int", "float",
                  "round", "abs", "hasattr", "getattr", "iter", "next", "format",
                  "hash", "final_answer"}
CONSUMERS = {"next", "list", "tuple", "set", "frozenset", "sorted", "any", "all", "sum",
             "min", "max", "dict"}
# v9, C5: calls that run code the checker cannot see, or reach objects by name
UNTRACKABLE = {"exec", "eval", "globals", "locals", "vars", "setattr", "delattr", "compile", "__import__",
               "builtins.exec", "builtins.eval", "builtins.compile", "builtins.__import__", "builtins.setattr",
               "builtins.globals", "builtins.locals", "builtins.vars", "importlib.import_module", "runpy.run_path",
               "runpy.run_module", "code.interact"}
DIR_WRITER_NAMES = {"os.makedirs", "os.mkdir", "os.rename", "os.replace", "os.renames", "os.system", "os.symlink",
                    "shutil.copytree", "shutil.move", "shutil.unpack_archive", "shutil.copy", "shutil.copyfile",
                    "shutil.copy2", "subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_call",
                    "subprocess.check_output", "makedirs", "mkdir", "system", "copytree", "urlretrieve",
                    "os.popen", "os.link", "os.execv", "os.spawnv"}
DIR_WRITER_ATTRS = {"makedirs", "mkdir", "extractall", "copytree", "urlretrieve", "unpack_archive"}
# v9, F1: calls that may create a file whose path the checker does not record
# methods that write to the path in their first argument when they have one (to_string() alone returns text)
BUF_WRITERS = {"to_string", "to_html", "to_latex", "to_markdown", "to_xml", "to_hdf", "to_stata", "to_orc",
               "savefig", "save", "savez", "savez_compressed", "savetxt", "tofile", "to_file", "extract",
               "extractall", "writestr", "copyfile", "copy2", "copytree", "move", "unpack_archive"}
# methods that write to the path of their receiver: Path(p).touch()
SELF_WRITERS = {"touch", "write_bytes", "symlink_to", "hardlink_to"}
# functions that write to the path in their second argument: shutil.copy(src, dst)
DST_WRITERS = {"os.rename", "os.replace", "os.renames", "os.link", "os.symlink", "shutil.copy", "shutil.copyfile",
               "shutil.copy2", "shutil.copytree", "shutil.move", "urllib.request.urlretrieve"}
# functions whose file effects the checker cannot know
OPAQUE_WRITERS = {"os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.Popen",
                  "subprocess.check_call", "subprocess.check_output", "shutil.unpack_archive", "tempfile.mkstemp",
                  "tempfile.mkdtemp", "os.execv", "os.spawnv", "numpy.save", "numpy.savez", "numpy.savetxt",
                  "joblib.dump", "pickle.dump"}
LIST_SAFE_CALLS = {"len", "print", "sorted", "list", "set", "tuple", "str", "repr"}
COLUMN_ARGS = {"groupby": (0, ("by",)), "sort_values": (0, ("by",)), "drop_duplicates": (0, ("subset",)),
               "dropna": (None, ("subset",)), "set_index": (0, ("keys",)),
               "pivot": (None, ("index", "columns", "values")),
               "pivot_table": (None, ("index", "columns", "values"))}
ROW_AXIS = (0, "index", "rows")
SAME_COLUMNS = {"copy", "head", "tail", "sample", "sort_values", "sort_index", "drop_duplicates", "dropna",
                "fillna", "query", "astype", "nlargest", "nsmallest", "round", "replace", "ffill", "bfill"}
KEEP_ROWS = {"copy", "sort_values", "sort_index", "fillna", "astype", "round", "replace", "ffill", "bfill"}
REDUCE = {"sum", "mean", "count", "max", "min", "median", "nunique", "first", "last", "std", "var", "prod"}
MASK_CALLS = {"isin", "notna", "isna", "notnull", "isnull", "between", "contains", "startswith", "endswith",
              "duplicated", "eq", "ne", "gt", "lt", "ge", "le", "match", "fullmatch"}
END_NAMES = {"final_answer", "exit", "quit"}
END_ATTRS = {("sys", "exit"), ("os", "_exit")}
# v9, C4: qualified names of calls that end the process
END_QUALIFIED = {"sys.exit", "os._exit", "os.abort", "builtins.exit", "builtins.quit", "unittest.main", "pytest.main"}
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
CELL, LOOP = "cell", "loop"
# v9, C3: context managers that do not suppress errors
SAFE_CONTEXTS = {"open", "io.open", "codecs.open", "gzip.open", "bz2.open", "lzma.open", "builtins.open",
                 "warnings.catch_warnings", "pandas.option_context", "numpy.errstate", "contextlib.redirect_stdout",
                 "contextlib.redirect_stderr", "contextlib.nullcontext", "tempfile.TemporaryDirectory",
                 "tempfile.NamedTemporaryFile", "tempfile.TemporaryFile", "zipfile.ZipFile", "tarfile.open"}
# v9, O1/F4: what a bare name means before the trace binds it (the usual imports of a harness)
DEFAULT_ALIASES = {"pd": "pandas", "pandas": "pandas", "np": "numpy", "numpy": "numpy", "json": "json", "csv": "csv",
                   "os": "os", "sys": "sys", "io": "io", "contextlib": "contextlib", "warnings": "warnings",
                   "copy": "copy", "shutil": "shutil", "pathlib": "pathlib", "Path": "pathlib.Path", "glob": "glob",
                   "math": "math", "re": "re", "builtins": "builtins", "operator": "operator", "codecs": "codecs",
                   "gzip": "gzip", "subprocess": "subprocess", "tempfile": "tempfile", "zipfile": "zipfile",
                   "datetime": "datetime", "collections": "collections", "itertools": "itertools",
                   "functools": "functools", "random": "random", "statistics": "statistics", "pickle": "pickle",
                   "importlib": "importlib", "time": "time", "string": "string", "decimal": "decimal"}
STAR_EXPORTS = {"os": ("chdir", "makedirs", "mkdir", "listdir", "scandir", "rename", "replace", "remove", "system",
                       "symlink", "link", "popen"),
                "sys": ("exit",), "shutil": ("copy", "copyfile", "copy2", "copytree", "move")}
# v9, A1: attributes of a frame that give no way to change its column names
SAFE_ATTRS = {"columns", "shape", "dtypes", "empty", "size", "ndim", "values", "index", "T", "axes", "dtype",
              "keys", "items", "get", "head", "tail", "describe", "info", "plot", "str", "dt", "cat", "iloc",
              "loc", "at", "iat", "name", "nunique", "unique", "value_counts"}
# v9, A1: attributes whose value gives write access to a frame or a record
UNSAFE_ATTRS = {"loc", "iloc", "at", "iat", "__dict__", "_mgr", "_data", "__class__", "__setitem__", "__delitem__"}
ACCESSORS = {"str", "dt", "cat", "sparse", "plot", "style", "attrs", "flags"}
# v9, A1: module functions that change an object passed to them, or return it or its parts
MUTATOR_FUNCS = {"operator.setitem", "operator.delitem", "operator.iadd", "operator.ior", "setattr", "delattr",
                 "builtins.setattr", "copy.copy", "pandas.DataFrame", "pandas.Series", "pandas.DataFrame.insert",
                 "dict.update", "dict.setdefault", "dict.pop", "dict.clear", "dict.__setitem__", "list.append",
                 "list.extend", "list.insert", "itertools.chain", "random.choice", "random.sample",
                 "random.choices", "heapq.nlargest", "heapq.nsmallest", "functools.reduce", "operator.itemgetter",
                 "collections.deque", "collections.ChainMap", "types.SimpleNamespace", "weakref.ref"}
# v9, A1: calls that leave a record or a list of records unchanged and keep no reference to it
RECORD_SAFE_CALLS = {"len", "print", "str", "repr", "type", "isinstance", "id", "bool", "hash", "display",
                     "json.dumps", "json.dump", "pandas.DataFrame", "pandas.json_normalize", "copy.deepcopy",
                     "pprint.pprint", "dict"}
# v9, A1: calls that leave a frame unchanged and keep no reference to it
FRAME_SAFE_CALLS = {"len", "print", "str", "repr", "type", "isinstance", "id", "bool", "hash", "display", "list",
                    "tuple", "set", "sorted", "enumerate", "zip", "iter", "min", "max", "sum", "any", "all", "dict",
                    "reversed", "next", "pandas.concat", "pandas.merge", "pandas.melt", "pandas.get_dummies",
                    "pandas.crosstab", "pandas.pivot_table", "pandas.to_numeric", "pandas.to_datetime",
                    "pandas.isna", "pandas.notna", "pandas.isnull", "pandas.notnull", "pandas.unique",
                    "pandas.cut", "pandas.qcut", "pandas.merge_asof", "pandas.wide_to_long", "json.dumps",
                    "copy.deepcopy"}
# v9, A1: methods of a tracked frame that take other frames as arguments without keeping or changing them
FRAME_ARG_METHODS = {"merge", "join", "equals", "isin", "compare", "combine_first", "corrwith", "dot", "align",
                     "reindex_like", "add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow", "eq", "ne",
                     "lt", "le", "gt", "ge", "radd", "rsub", "rmul", "rdiv", "where", "mask", "concat", "append"}


class GateState:
    """Tracked objects, string constants and facts about names, carried across the steps of one trace."""

    def __init__(self):
        self.frames = {}
        self.consts = {}
        self.pending = None
        # v9: facts that only grow over a trace
        self.enders = set()          # names whose call may end the step (C4)
        self.attr_enders = set()     # method names whose call may end the step (C4)
        self.creator_funcs = set()   # names whose call may write a file (F1)
        self.attr_creators = set()   # method names whose call may write a file (F1)
        self.defined = set()         # names that the trace binds (O1, C5)
        self.aliases = {}            # name -> qualified name, from imports (F4)
        self.unaliased = set()       # names bound by something other than an import
        self.cwd_changed = False     # an earlier step may have changed the working directory (F4)

    def commit(self, failed):
        if self.pending is None:
            return
        frames, consts, facts = self.pending
        self.pending = None
        # facts are kept whether or not the step failed: the step may have bound them before it failed
        self.enders |= facts["enders"]
        self.attr_enders |= facts["attr_enders"]
        self.creator_funcs |= facts["creator_funcs"]
        self.attr_creators |= facts["attr_creators"]
        self.defined |= facts["defined"]
        self.unaliased |= facts["unaliased"]
        for k, v in facts["aliases"].items():
            if k not in self.unaliased:
                self.aliases[k] = v
        for k in self.unaliased:
            self.aliases.pop(k, None)
        self.cwd_changed = self.cwd_changed or facts["cwd_changed"]
        if facts["forget_all_frames"]:
            self.frames = {k: v for k, v in self.frames.items() if v[2] not in ("frame", "chunks")}
        if not failed:
            self.frames, self.consts = frames, consts
            return
        self.frames = {k: v for k, v in self.frames.items()
                       if frames.get(k) == v}
        self.consts = {k: v for k, v in self.consts.items()
                       if consts.get(k) == v}


_TLS = threading.local()        # the child lists of the step being decided (speed only)


def _children(n):
    kids = getattr(_TLS, "kids", None)
    if kids is not None:
        c = kids.get(id(n))
        if c is not None:
            return c
    return list(ast.iter_child_nodes(n))


def _walk(node):
    """All nodes under node (node included), like ast.walk but in depth-first order."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(_children(n))


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
                if r is None or v.conversion != -1 or v.format_spec is not None:
                    return None
                parts.append(r)
        return "".join(parts)
    if isinstance(node, ast.Call):
        f = node.func
        if node.keywords:
            return None
        if isinstance(f, ast.Attribute) and f.attr == "join" and isinstance(f.value, ast.Attribute) \
                and f.value.attr == "path":  # os.path.join
            parts = [_resolve(a, consts) for a in node.args]
            return os.path.join(*parts) if parts and all(p is not None for p in parts) else None
        if (isinstance(f, ast.Name) and f.id in ("Path", "PurePath")) or \
                (isinstance(f, ast.Attribute) and f.attr in ("Path", "PurePath")):
            parts = [_resolve(a, consts) for a in node.args]
            return os.path.join(*parts) if parts and all(p is not None for p in parts) else None
    return None


def _not_a_file(path):
    """v9, F2: a URL is not a local file."""
    return "://" in path


def _path_arg(call):
    if call.args:
        return call.args[0]
    return next((k.value for k in call.keywords if k.arg in PATH_KWARGS), None)


def _const_is(node, values):
    return isinstance(node, ast.Constant) and any(
        type(node.value) is type(v) and node.value == v for v in values)


def _utf8(node):
    return isinstance(node, ast.Constant) and (
        node.value is None or (isinstance(node.value, str) and node.value.strip().lower() in UTF8_NAMES))


def _reader_args(call):
    """Gate v7, rule 2 (unchanged)."""
    kind = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
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
    return len(call.args) == 1 and not isinstance(call.args[0], ast.Starred) and not call.keywords


def csv_columns(path):
    """Gate v7, rule 1 (unchanged)."""
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


def _literal_list(node):
    if isinstance(node, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
        return [e.value for e in node.elts]
    return None


def _literal_names(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


SAFE_CONSUMERS = {"pd.DataFrame", "pandas.DataFrame", "DataFrame", "len", "print", "list", "sorted", "json.dumps",
                  "set", "tuple", "enumerate", "display"}
HELPER_CALLS = {"isinstance", "open", "json.load", "json.loads", "load", "loads", "pd.read_csv", "pd.read_json",
                "pandas.read_csv", "pandas.read_json", "os.path.exists", "os.path.isfile", "exists", "isfile",
                "endswith", "startswith", "lower", "strip", "read"}


def _helper_pure(d):
    ok_stmts = (ast.Expr, ast.If, ast.Return, ast.With, ast.Assign, ast.Pass)
    for n in _walk(ast.Module(body=d.body, type_ignores=[])):
        if isinstance(n, ast.stmt) and not isinstance(n, ok_stmts):
            return False
        if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant):
            return False
        if isinstance(n, ast.Assign) and not (len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)):
            return False
        if isinstance(n, ast.Call) and _call_name(n) not in HELPER_CALLS and not (
                isinstance(n.func, ast.Attribute) and n.func.attr in ("endswith", "startswith", "lower", "strip", "read")):
            return False
        if isinstance(n, ast.Call) and _call_name(n) == "open" and _builtin_open_mode(n) not in ("r", "rb", "rt"):
            return False
        if isinstance(n, (ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)):
            return False
    return True


def _builtin_open_mode(call):
    """The mode of open(path, mode); None if it is not a literal."""
    mode = call.args[1] if len(call.args) > 1 else next(
        (k.value for k in call.keywords if k.arg == "mode"), None)
    if mode is None:
        return "r"
    return mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else None


def _method_open_mode(call):
    """v9, F3: the mode of Path(p).open(mode); the mode is the first argument."""
    mode = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "mode"), None)
    if mode is None:
        return "r"
    return mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else None


def _is_mask(node):
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
    s = node.slice
    if isinstance(s, ast.Constant) and isinstance(s.value, str):
        return [s.value]
    return _literal_list(s)


def _walk_no(node, skip):
    todo = [node]
    while todo:
        n = todo.pop()
        yield n
        for c in _children(n):
            if not isinstance(c, skip):
                todo.append(c)


def _walk_no_defs(node):
    return _walk_no(node, DEFS)


def _walk_run_now(node):
    """v9: the nodes of `node` that run when it runs: class bodies, decorators and default
    values run at definition; function and lambda bodies do not."""
    todo = [node]
    while todo:
        n = todo.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node:
            todo.extend(n.decorator_list)
            todo.extend(d for d in n.args.defaults + n.args.kw_defaults if d is not None)
            continue
        if isinstance(n, ast.Lambda) and n is not node:
            todo.extend(d for d in n.args.defaults + n.args.kw_defaults if d is not None)
            continue
        todo.extend(_children(n))


def _is_exit_raise(n):
    if not isinstance(n, ast.Raise) or n.exc is None:
        return False
    e = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
    return (isinstance(e, ast.Name) and e.id == "SystemExit") or \
        (isinstance(e, ast.Attribute) and e.attr == "SystemExit")


def _may_skip(stmt):
    if isinstance(stmt, LOOPS):
        return any(_may_skip(s) for s in stmt.orelse)
    return any(isinstance(n, (ast.Break, ast.Continue)) for n in _walk_no(stmt, DEFS + LOOPS))


def _escaping(expr):
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
    out = set()
    for n in _walk(expr):
        if isinstance(n, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            out |= _escaping(n)
    return out


def _names(node):
    return {e.id for e in _walk(node) if isinstance(e, ast.Name)}


def _records_arg(call):
    a = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "orient"), None)
    return isinstance(a, ast.Constant) and a.value == "records"


def _parents(node):
    par = {}
    for n in _walk(node):
        for c in _children(n):
            par[c] = n
    return par


def _index(tree):
    """Child lists and the parent map of a parsed step, in one pass."""
    kids, par = {}, {}
    stack = [tree]
    while stack:
        n = stack.pop()
        c = list(ast.iter_child_nodes(n))
        kids[id(n)] = c
        for x in c:
            par[x] = n
        stack.extend(c)
    return kids, par


def _module_bindings(tree):
    """v9: (names bound at module level other than by import, {alias: qualified name} from imports,
    names declared global in function bodies)."""
    bound, aliases = set(), {}
    nodes = sorted(_walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    for n in nodes:                          # in source order, so that a later import wins
        if isinstance(n, ast.Import):
            for al in n.names:
                if al.asname:
                    aliases[al.asname] = al.name
                else:
                    top = al.name.split(".")[0]
                    aliases[top] = top
        elif isinstance(n, ast.ImportFrom):
            mod = ("." * (n.level or 0)) + (n.module or "")
            for al in n.names:
                if al.name == "*":
                    for k in STAR_EXPORTS.get(mod, ()):
                        aliases[k] = f"{mod}.{k}"
                    continue
                aliases[al.asname or al.name] = f"{mod}.{al.name}"
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Global):
            bound |= set(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            bound.add(n.rest)
    # names bound inside function bodies are local to them, unless declared global; a
    # comprehension variable is local too. Counting them is conservative: a bound name only
    # loses a special meaning.
    return bound, aliases


def _list_consts(tree):
    """Gate v8, R2 (unchanged)."""
    cand = {}
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                and isinstance(st.value, (ast.List, ast.Tuple)) and st.value.elts \
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in st.value.elts):
            cand.setdefault(st.targets[0].id, []).append((st.lineno, tuple(e.value for e in st.value.elts)))
    if not cand:
        return {}
    parent = _parents(tree)
    stores, bad = {}, set()
    for n in _walk(tree):
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
        for n in _walk(tree):
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


class Facts:
    """v9: what names mean in one step, from this step and the earlier steps of the trace."""

    def __init__(self, tree, state, par=None):
        self.par = par if par is not None else _parents(tree)    # one parent map for the whole step
        self.tree_nodes = set(map(id, self.par)) | {id(tree)}
        bound, imported = _module_bindings(tree)
        prev_aliases = dict(state.aliases) if state is not None else {}
        prev_unaliased = set(state.unaliased) if state is not None else set()
        self.unaliased = bound
        self.aliases = dict(DEFAULT_ALIASES)
        self.aliases.update(prev_aliases)
        self.aliases.update(imported)
        for k in bound | prev_unaliased:
            self.aliases.pop(k, None)
        self.new_aliases = {k: v for k, v in imported.items() if k not in bound}
        self.defined = bound | (set(state.defined) if state is not None else set())
        self.enders = set(state.enders) if state is not None else set()
        self.attr_enders = set(state.attr_enders) if state is not None else set()
        self.creator_funcs = set(state.creator_funcs) if state is not None else set()
        self.attr_creators = set(state.attr_creators) if state is not None else set()
        self._solve_enders(tree)
        self._solve_creators(tree)

    def qname(self, e):
        """The qualified name of a module, function or attribute chain ('pandas.read_csv',
        'os.chdir', 'open'), or None for a value that the trace binds."""
        if isinstance(e, ast.Name):
            if e.id in self.aliases:
                return self.aliases[e.id]
            if e.id in self.defined:
                return None
            return e.id
        if isinstance(e, ast.Attribute):
            base = self.qname(e.value)
            return None if base is None else f"{base}.{e.attr}"
        return None

    def builtin(self, name):
        """True if `name` still means the builtin (or harness) function of that name."""
        return name not in self.defined and name not in self.aliases

    def is_end_name(self, e):
        q = self.qname(e)
        if q in END_QUALIFIED or (q in END_NAMES and isinstance(e, ast.Name)):
            return True
        return isinstance(e, ast.Name) and e.id in self.enders

    def is_end_call(self, n):
        if not isinstance(n, ast.Call):
            return False
        f = n.func
        if self.is_end_name(f):
            return True
        if isinstance(f, ast.Name) and f.id in END_NAMES:          # final_answer even when rebound
            return True
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in END_ATTRS:
            return True
        return isinstance(f, ast.Attribute) and f.attr in self.attr_enders

    def poison(self, n, par):
        """C5: n ends the step at a point the checker cannot see, or runs code it cannot see."""
        if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(getattr(n, "ctx", None), ast.Load) \
                and self.is_end_name(n):
            p = par.get(n)
            return not (isinstance(p, ast.Call) and p.func is n) and not isinstance(p, ast.Attribute)
        if isinstance(n, ast.Call):
            q = self.qname(n.func)
            if q in UNTRACKABLE:
                return True
            if q in ("getattr", "builtins.getattr", "operator.attrgetter", "operator.methodcaller"):
                if len(n.args) < 2 or not (isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)):
                    return True
                name = n.args[1].value
                base = self.qname(n.args[0])
                return name in ("exit", "_exit", "abort", "main") or name in self.attr_enders or \
                    name in self.enders or (base is not None and base.split(".")[0] in ("sys", "os", "builtins"))
        return False

    def parents(self, node):
        return self.par if id(node) in self.tree_nodes else _parents(node)

    def expr_may_end(self, node, par=None):
        """True if running `node` may end the step (function and lambda bodies do not run)."""
        par = par if par is not None else self.parents(node)
        for n in _walk_run_now(node):
            if self.is_end_call(n) or _is_exit_raise(n) or self.poison(n, par):
                return True
        return False

    def body_may_end(self, d):
        """True if a call of the function, lambda or class `d` may end the step."""
        par = self.parents(d)
        body = d.body if isinstance(d.body, list) else [d.body]
        for st in body:
            for n in _walk_run_now(st):
                if self.is_end_call(n) or _is_exit_raise(n) or self.poison(n, par):
                    return True
        return False

    def _solve_enders(self, tree):
        """C4: names whose call may end the step, to a fixed point."""
        funcs, classes, links = {}, {}, []
        for n in _walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(n.name, []).append(n)
            elif isinstance(n, ast.ClassDef):
                classes.setdefault(n.name, []).append(n)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if isinstance(n.value, ast.Lambda):
                    for t in names:
                        funcs.setdefault(t, []).append(n.value)
                elif n.value is not None:
                    links.append((names, n.value))
        # the values that a binding makes callable through its target: x = f, x = f if c else g
        for k, v in self.aliases.items():
            if v in END_QUALIFIED:
                self.enders.add(k)
        changed = True
        while changed:
            changed = False
            for name, defs in funcs.items():
                if name not in self.enders and any(self.body_may_end(d) for d in defs):
                    self.enders.add(name)
                    changed = True
            for name, defs in classes.items():
                for c in defs:
                    for m in c.body:
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name not in self.attr_enders \
                                and self.body_may_end(m):
                            self.attr_enders.add(m.name)
                            changed = True
                if name not in self.enders and any(
                        isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name in self.attr_enders
                        for c in defs for m in c.body):
                    self.enders.add(name)
                    changed = True
            for names, value in links:
                hit = False
                todo = [value]
                while todo:
                    v = todo.pop()
                    if isinstance(v, (ast.Name, ast.Attribute)) and self.is_end_name(v):
                        hit = True
                        break
                    if isinstance(v, ast.IfExp):
                        todo += [v.body, v.orelse]
                    elif isinstance(v, ast.BoolOp):
                        todo += v.values
                    elif isinstance(v, ast.NamedExpr):
                        todo.append(v.value)
                if hit:
                    for t in names:
                        if t not in self.enders:
                            self.enders.add(t)
                            changed = True

    def _solve_creators(self, tree):
        """F1: functions and methods of the trace whose body may write a file."""
        def writes(d):
            for n in _walk(d):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                q = self.qname(f)
                if q in OPAQUE_WRITERS or q in DST_WRITERS or (isinstance(f, ast.Attribute) and (
                        f.attr in SELF_WRITERS | BUF_WRITERS | WRITERS | {"open"})) \
                        or (q in ("open", "io.open", "codecs.open") and _builtin_open_mode(n) != "r") \
                        or (isinstance(f, ast.Name) and f.id in self.creator_funcs) \
                        or (isinstance(f, ast.Attribute) and f.attr in self.attr_creators):
                    return True
            return False
        changed = True
        while changed:
            changed = False
            for n in _walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name not in self.creator_funcs \
                        and writes(n):
                    self.creator_funcs.add(n.name)
                    self.attr_creators.add(n.name)
                    changed = True
                elif isinstance(n, ast.ClassDef) and n.name not in self.creator_funcs and writes(n):
                    self.creator_funcs.add(n.name)
                    changed = True
                elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda) and writes(n.value):
                    for t in n.targets:
                        if isinstance(t, ast.Name) and t.id not in self.creator_funcs:
                            self.creator_funcs.add(t.id)
                            changed = True

    def export(self, cwd_changed, forget_all_frames):
        return {"enders": set(self.enders), "attr_enders": set(self.attr_enders),
                "creator_funcs": set(self.creator_funcs), "attr_creators": set(self.attr_creators),
                "defined": set(self.defined), "aliases": dict(self.new_aliases),
                "unaliased": set(self.unaliased), "cwd_changed": cwd_changed,
                "forget_all_frames": forget_all_frames}


class _Walker:
    def __init__(self, schema, state=None, exists=None, tolerant_keys=False, alias=False, locate=None,
                 list_consts=None, dir_writers=False, facts=None):
        self.schema = {os.path.basename(k): set(v) for k, v in schema.items()}
        self.locate = locate
        self.list_consts = list_consts or {}
        self.dir_writers = dir_writers
        self.facts = facts
        self.alias = alias
        self.groups = {}
        self.helpers = {}
        self.skip_call = None
        if state is not None:
            self.frames = {k: (s, set(c), kind, ne) for k, (s, c, kind, ne) in state.frames.items()}
            self.consts = dict(state.consts)
        else:
            self.frames = {}
            self.consts = {}
        self.exists = exists
        self.tolerant = tolerant_keys
        self.substitutions = []
        self.written = set()
        self.files = []
        self.dirs = []
        self.cwd_changed = bool(state is not None and state.cwd_changed)
        self.cwd_changed_here = False
        self.forgot_all_frames = False
        self.cur_cond = False              # the statement being processed may not run
        self._may_end_cache = {}
        self.creator_seen = False          # F1: a call before this point may have created a file
        self.in_loop = False
        self.unknown_reasons = []
        self.definite, self.possible = [], []
        self.reads = 0

    def export(self):
        shared = {n for n, g in self.groups.items() if len(g) > 1}
        return ({k: (s, frozenset(c), kind, ne) for k, (s, c, kind, ne) in self.frames.items()
                 if k not in shared},
                dict(self.consts))

    # ----------------------------------------------------------------- names
    def qname(self, e):
        return self.facts.qname(e)

    def is_pandas(self, e):
        return self.qname(e) == "pandas"

    def is_reader(self, node):
        """O1: pd.read_csv / pd.read_json with pd bound to pandas (attribute form, as in version 8)."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in READERS):
            return False
        return self.qname(node.func) in ("pandas.read_csv", "pandas.read_json")

    def _untracked(self, why):
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

    def forget_frames(self):
        """N4, A3: stop following every frame (an index or column label may have changed)."""
        self.forgot_all_frames = True
        self.forget([k for k, v in self.frames.items() if v[2] in ("frame", "chunks")])

    def is_alias(self, s, targets, value, conditional):
        return (self.alias and not conditional and isinstance(s, ast.Assign) and len(targets) == 1
                and isinstance(targets[0], ast.Name) and isinstance(value, ast.Name)
                and value.id in self.frames and value.id != targets[0].id)

    def lookup(self, name, env):
        return env[name] if name in env else self.frames.get(name)

    # ---------------------------------------------------------------- receipts
    def receipt(self, obj, name, lineno):
        src, cols, kind, _ = obj
        found = sorted(f for f, c in self.schema.items() if name in c and f != src)
        strs = sorted((c for c in cols if isinstance(c, str)), key=str)
        near = difflib.get_close_matches(name, strs, n=3, cutoff=0.7)
        what = "column" if kind == "frame" else "key"
        reason = f"{src} has no column '{name}'" if kind == "frame" else \
            f"no record in {src} has the key '{name}'"
        return {"location": f"line {lineno}", "kind": what, "source": src, "missing": name,
                "reason": reason, "available": sorted(cols, key=str), "found_in": found, "near_matches": near}

    def file_receipt(self, path, lineno):
        base = os.path.basename(path)
        near = difflib.get_close_matches(base, sorted(self.schema), n=3, cutoff=0.6)
        found = sorted(self.locate(base) or []) if self.locate else []
        return {"location": f"line {lineno}", "kind": "file", "source": path, "missing": base,
                "reason": f"file '{path}' does not exist", "available": sorted(self.schema),
                "found_in": found, "near_matches": near}

    def dir_receipt(self, path, lineno):
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

    def is_builtin_open(self, call):
        """F3: open(...) where open is the builtin (io.open and codecs.open open files too)."""
        q = self.qname(call.func)
        return q in ("open", "io.open", "builtins.open", "codecs.open")

    def text_path(self, expr, env):
        if isinstance(expr, ast.Name):
            v = self.consts.get(expr.id)
            return v[1] if isinstance(v, tuple) and v[0] == "text" else None
        if not isinstance(expr, ast.Call):
            return None
        f = expr.func
        if isinstance(f, ast.Attribute) and f.attr in ("read", "read_text") and not expr.args and not expr.keywords:
            return self.handle_path(f.value) or (self.path_of(f.value) if f.attr == "read_text" else None)
        return None

    def handle_path(self, expr):
        if isinstance(expr, ast.Name):
            v = self.consts.get(expr.id)
            return v[1] if isinstance(v, tuple) and v[0] == "handle" else None
        if isinstance(expr, ast.Call) and self.is_builtin_open(expr) and _call_name(expr) == "open" \
                and _builtin_open_mode(expr) == "r":
            return self.path_of(_path_arg(expr))
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "open" \
                and isinstance(expr.func.value, ast.Call) \
                and self.qname(expr.func.value.func) in ("pathlib.Path", "pathlib.PurePath") \
                and _method_open_mode(expr) == "r":
            return self.path_of(expr.func.value)
        return None

    def context(self, path, suffix):
        if path is None:
            return None
        base = os.path.basename(path)
        return base if base in self.schema and base.endswith(suffix) else None

    def records_from(self, expr, env):
        if not isinstance(expr, ast.Call):
            return None
        q = self.qname(expr.func)
        cn = _call_name(expr)
        if q == "json.load" and cn in ("json.load", "load") and len(expr.args) == 1 and not expr.keywords:
            base = self.context(self.handle_path(expr.args[0]), ".json")
        elif q == "json.loads" and cn in ("json.loads", "loads") and len(expr.args) == 1 and not expr.keywords:
            base = self.context(self.text_path(expr.args[0], env), ".json")
        else:
            base = None
        return (base, self.schema[base]) if base else None

    def elem(self, it, env):
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
        if hr:
            return (hr[0], set(hr[1]), "record", True), hr[3]
        q, f = self.qname(it.func), it.func
        if q == "csv.DictReader" and _call_name(it) in ("csv.DictReader", "DictReader") and it.args:
            base = self.context(self.handle_path(it.args[0]), ".csv")
            if base and _plain_dictreader(it):
                return (base, set(self.schema[base]), "record", True), True
            if base:
                self._untracked(f"csv.DictReader of {base} with arguments that may change the field names")
        if isinstance(f, ast.Attribute) and f.attr == "iterrows" and not it.args and not it.keywords:
            o = self.frame_of(f.value, env)
            if o:
                return ("pair", (o[0], set(o[1]), "record", True)), o[3]
        if isinstance(f, ast.Attribute) and f.attr == "to_dict" and _records_arg(it) \
                and isinstance(f.value, ast.Name):
            o = self.lookup(f.value.id, env)
            if o and o[2] == "frame":
                return (o[0], set(o[1]), "record", True), o[3]
        if self.is_reader(it) and any(k.arg in ("chunksize", "iterator") for k in it.keywords):
            base = self.context(self.path_of(_path_arg(it)), ".csv")
            why = _reader_args(it) or ("names= or header=None" if _kw(it, "names") is not None
                                       or _const_is(_kw(it, "header"), (None,)) else None)
            if base and not why:
                return (base, set(self.schema[base]), "frame", True), self._rows_kept(it)
            if base:
                self._untracked(f"chunked read of {base}: {why}")
        if q == "enumerate" and it.args and len(it.args) == 1 and not it.keywords:
            inner = self.elem(it.args[0], env)
            if inner and inner[0][0] != "pair":
                return ("pair", inner[0]), inner[1]
        return None

    def bind_target(self, target, elem, env=None):
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
        if isinstance(n, ast.Compare) and len(n.comparators) > 1:      # C1
            self._reads(n.left, cond, env)
            self._reads(n.comparators[0], cond, env)
            for c in n.comparators[1:]:
                self._reads(c, True, env)
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
                for x in _walk(n):
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
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in CONSUMERS \
                and self.facts.builtin(n.func.id):                      # C5: a rebound builtin is not a consumer
            for a in n.args:
                if isinstance(a, ast.GeneratorExp):
                    self._comp(a, cond, env, eager=True)
                else:
                    self._reads(a, cond, env)
            for k in n.keywords:
                self._reads(k.value, cond, env)
            return
        for ch in _children(n):
            self._reads(ch, cond, env)

    def _comp(self, n, cond, env, eager):
        gens = n.generators
        self._reads(gens[0].iter, cond, env)
        inner = dict(env)
        el = self.elem(gens[0].iter, env)
        self.bind_target(gens[0].target, el, inner)
        first = eager and el is not None and el[1] and len(gens) == 1 and not gens[0].is_async
        c_if = cond if first else True
        for i, f in enumerate(gens[0].ifs):
            self._reads(f, c_if if i == 0 else True, inner)              # C2
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
        if conditional or not isinstance(d, ast.FunctionDef) or d.decorator_list:
            return
        a = d.args
        if len(a.args) != 1 or a.posonlyargs or a.vararg or a.kwonlyargs or a.kwarg or a.defaults:
            return
        # O1: the helper's json.load / pd.read_csv must be the library functions
        for n in _walk(d):
            if isinstance(n, ast.Call):
                cn = _call_name(n)
                if cn in ("load", "loads") and self.qname(n.func) not in ("json.load", "json.loads"):
                    return
                if cn in ("json.load", "json.loads") and self.qname(n.func) not in ("json.load", "json.loads"):
                    return
                if cn in ("pd.read_csv", "pd.read_json", "pandas.read_csv", "pandas.read_json") \
                        and self.qname(n.func) not in ("pandas.read_csv", "pandas.read_json"):
                    return
                if cn == "open" and not self.is_builtin_open(n):
                    return
                if cn in ("isinstance", "os.path.exists", "os.path.isfile", "exists", "isfile") \
                        and cn.split(".")[0] in self.facts.defined:
                    return
        if _helper_pure(d):
            self.helpers[d.name] = d

    def helper_arg(self, expr, env):
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
                        and h.args[0].id == p and _builtin_open_mode(h) in ("r", "rb") and path is not None:
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
                            and _builtin_open_mode(c) in ("r", "rb") and path is not None):
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
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in self.helpers
                and len(call.args) == 1 and not call.keywords):
            return None, None
        arg = self.helper_arg(call.args[0], env)
        if arg is None:
            return None, None
        res, _ = self.eval_helper(self.helpers[call.func.id], arg)
        return res, arg

    def _check_helper_call(self, n, cond, env):
        if not (isinstance(n.func, ast.Name) and n.func.id in self.helpers and len(n.args) == 1 and not n.keywords):
            return
        arg = self.helper_arg(n.args[0], env)
        if arg is None:
            return
        _, opened = self.eval_helper(self.helpers[n.func.id], arg)
        for path in opened[:1]:
            self._check_file(path, cond, n.lineno)

    def records_of(self, expr, env):
        if isinstance(expr, ast.Name):
            o = self.lookup(expr.id, env)
            return o if o and o[2] == "records" else None
        res, arg = self.helper_value(expr, env)
        if res == "same" and arg[2][2] == "records":
            return arg[2]
        return res if isinstance(res, tuple) and res[2] == "records" else None

    def _frame(self, expr, env):
        return self.frame_of(expr, env)

    def frame_of(self, e, env, depth=0):
        if depth > 20:
            return None
        if isinstance(e, ast.Name):
            o = self.lookup(e.id, env)
            return o if o and o[2] == "frame" else None
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in self.helpers:
            res, arg = self.helper_value(e, env)
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
            return None
        m, recv = e.func.attr, e.func.value
        if self.is_pandas(recv):
            if m == "DataFrame" and len(e.args) == 1 and not e.keywords:
                rec = self.records_of(e.args[0], env)
                return (rec[0], set(rec[1]), "frame", rec[3]) if rec else None
            if m == "merge":
                left = e.args[0] if e.args else _kw(e, "left")
                right = e.args[1] if len(e.args) > 1 else _kw(e, "right")
                if left is None or right is None:
                    return None
                return self._merge_frame(self.frame_of(left, env, depth + 1), self.frame_of(right, env, depth + 1), e)
            if m == "concat" and e.args and isinstance(e.args[0], (ast.List, ast.Tuple)) and e.args[0].elts \
                    and not any(k.arg in ("keys", "names", "levels") for k in e.keywords):        # N3
                parts = [self.frame_of(x, env, depth + 1) for x in e.args[0].elts]
                if all(parts):
                    return ("concat", set().union(*(p[1] for p in parts)), "frame", any(p[3] for p in parts))
            return None
        if isinstance(recv, ast.Name) and recv.id in ("pd", "pandas"):
            return None                     # pd bound to something else (O1)
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
            both |= keys
        cols |= {c + sx for c in both if isinstance(c, str)} | {c + sy for c in both if isinstance(c, str)}
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
        f = n.func
        if not isinstance(f, ast.Attribute):
            return
        if any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords):
            return
        m = f.attr
        if m == "merge":
            if self.is_pandas(f.value):
                left = n.args[0] if n.args else _kw(n, "left")
                right = n.args[1] if len(n.args) > 1 else _kw(n, "right")
            elif isinstance(f.value, ast.Name) and f.value.id in ("pd", "pandas"):
                return
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
            if m == "pivot_table":                                     # N1: only values= is certain
                v = _kw(n, "values")
                self._check_columns(obj, _literal_names(v) if v is not None else [], n.lineno, cond)
                amb = []
                for k in ("index", "columns"):
                    v = _kw(n, k)
                    if v is not None:
                        amb += _literal_names(v)
                self._check_columns(obj, amb, n.lineno, True)
                return
            names = _literal_names(n.args[pos]) if pos is not None and len(n.args) > pos else []
            for k in kws:
                v = _kw(n, k)
                if v is not None:
                    names += _literal_names(v)
            c = cond
            if m == "groupby":                                         # N1: a list may be row labels
                by = n.args[0] if n.args else _kw(n, "by")
                c = cond or isinstance(by, ast.List)
            if m == "drop_duplicates" and not obj[3]:                  # N2: an empty frame does not raise
                c = True
            self._check_columns(obj, names, n.lineno, c)
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
            fr = self._frame(v.value, env)
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
            obj = self.frame_of(v.func.value, env)
        elif isinstance(v, (ast.Subscript, ast.Call)):
            obj = self.frame_of(v, env)
        if not obj or obj[2] not in ("frame", "record"):
            return
        keys = _subscript_keys(n)
        one_name = False
        if keys is None and isinstance(n.slice, ast.Name) and n.slice.id not in env:
            val = self.consts.get(n.slice.id)
            lc = self.list_consts.get(n.slice.id)
            if isinstance(val, str):
                keys, one_name = [val], True
            elif lc is not None and obj[2] == "frame" and n.lineno > lc[0]:
                keys = list(lc[1])
        if not keys or (obj[2] == "record" and not (isinstance(n.slice, ast.Constant) or one_name)):
            return
        self.reads += 1
        for k in keys:
            if k not in obj[1]:
                self.violation(obj, k, n.lineno, cond)

    def violation(self, obj, k, lineno, cond):
        if obj[2] == "record" and self.tolerant:
            near = difflib.get_close_matches(k, sorted((c for c in obj[1] if isinstance(c, str)), key=str))
            if near:
                self.substitutions.append({"location": f"line {lineno}", "source": obj[0], "written": k,
                                           "used": near[0], "conditional": cond})
                return
        (self.possible if cond else self.definite).append(self.receipt(obj, k, lineno))

    def _check_file(self, path, cond, lineno):
        """A file that the code reads must exist (exists() answers for the live workspace)."""
        self.files.append((path, cond, lineno))
        if self.exists is None or path in self.written or (self.cwd_changed and not os.path.isabs(path)):
            return
        if _not_a_file(path):                                          # F2
            return
        if self.exists(path) is False:
            (self.possible if cond or self.creator_seen else self.definite).append(self.file_receipt(path, lineno))

    def _check_file_call(self, n, cond):
        f = n.func
        q = self.qname(f)
        if isinstance(f, ast.Attribute) and f.attr in WRITERS:
            p = self.path_of(_path_arg(n)) if f.attr != "write_text" else self.path_of(f.value)
            if p is not None:
                self.written.add(p)
            return
        if self.write_target(n) is not None:
            return
        if q in ("os.listdir", "os.scandir", "os.chdir"):
            if _call_name(n) in ("os.listdir", "os.scandir", "os.chdir"):     # as version 8 (no new checks)
                self._check_dir_call(n, q, cond)
            if q == "os.chdir":
                self.cwd_changed = True
                self.cwd_changed_here = True
            return
        if isinstance(f, ast.Name) and f.id == "chdir" or (isinstance(f, ast.Attribute) and f.attr == "chdir"):
            self.cwd_changed = True                                     # a chdir of any kind (F4)
            self.cwd_changed_here = True
            return
        path = None
        if self.is_builtin_open(n) and _call_name(n) == "open":
            mode = _builtin_open_mode(n)
            p = self.path_of(_path_arg(n))
            if mode is None or any(c in mode for c in "wax+"):
                if p is not None:
                    self.written.add(p)
                return
            path = p
        elif isinstance(f, ast.Attribute) and f.attr == "open":
            # F3: <path>.open(mode) takes the mode first (Path(p).open("w")); a write is recorded, a read
            # is not checked (version 8 did not check it either)
            mode = _method_open_mode(n)
            p = self.path_of(f.value)
            if (mode is None or any(c in mode for c in "wax+")) and p is not None:
                self.written.add(p)
            return
        elif isinstance(f, ast.Attribute) and q is not None and q.startswith("pandas.") \
                and f.attr in FILE_READERS and len(q.split(".")) == 2:
            path = self.path_of(_path_arg(n))
            if path is not None and path.lstrip().startswith("~"):      # F2: pandas expands "~"
                return
            if path is not None and q.endswith("read_json") and path.lstrip()[:1] in ("[", "{"):
                return                                                  # F2: literal JSON text
        elif isinstance(f, ast.Attribute) and f.attr in ("read_text", "read_bytes"):
            path = self.path_of(f.value)
        if path is None:
            return
        self._check_file(path, cond, n.lineno)

    def write_target(self, n):
        """F1: for a call that writes a file other than by open() and the pandas WRITERS: the path it
        writes ("" if the checker cannot resolve it), or None if the call writes no file."""
        f = n.func
        q = self.qname(f)
        if q in OPAQUE_WRITERS or (isinstance(f, ast.Name) and f.id in self.facts.creator_funcs):
            return ""
        if q in DST_WRITERS:
            dst = n.args[1] if len(n.args) > 1 else _kw(n, "dst")
            return (self.path_of(dst) or "") if dst is not None else ""
        if isinstance(f, ast.Attribute) and f.attr in SELF_WRITERS:
            return self.path_of(f.value) or ""
        if isinstance(f, ast.Attribute) and f.attr in BUF_WRITERS:
            arg = n.args[0] if n.args else next((k.value for k in n.keywords
                                                 if k.arg in ("buf", "path_or_buf", "path_or_buffer", "path",
                                                              "fname", "file", "filename", "dst", "fp")), None)
            if arg is None:
                return None
            return self.path_of(arg) or ""
        if isinstance(f, ast.Attribute) and f.attr in self.facts.attr_creators:
            return ""
        return None

    def scan_creators(self, node):
        """F1: before the reads of a statement are checked, note any call in it that may create a file
        whose path the checker does not know; record the paths it does know."""
        for n in _walk_run_now(node):
            if not isinstance(n, ast.Call):
                continue
            t = self.write_target(n)
            if t == "":
                self.creator_seen = True
            elif t is not None:
                self.written.add(t)
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in WRITERS:
                p = self.path_of(_path_arg(n)) if f.attr != "write_text" else self.path_of(f.value)
                if p is None and (f.attr == "write_text" or _path_arg(n) is not None):
                    self.creator_seen = True
            if self.is_builtin_open(n) or (isinstance(f, ast.Attribute) and f.attr == "open"
                                           and not self.is_builtin_open(n)):
                mode = _builtin_open_mode(n) if self.is_builtin_open(n) else _method_open_mode(n)
                p = self.path_of(_path_arg(n)) if self.is_builtin_open(n) else self.path_of(f.value)
                if (mode is None or any(c in mode for c in "wax+")) and p is None:
                    self.creator_seen = True

    def _check_dir_call(self, n, name, cond):
        arg = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "path"), None)
        path = self.path_of(arg) if arg is not None else None
        if path is not None:
            self.dirs.append((path, cond, n.lineno))
        if path is None or self.exists is None or self.dir_writers \
                or (self.cwd_changed and not os.path.isabs(path)) or _not_a_file(path):
            return
        p = os.path.normpath(path)
        if any(os.path.normpath(w) == p or os.path.normpath(w).startswith(p.rstrip("/") + "/")
               for w in self.written):
            return
        if self.exists(path) is False:
            (self.possible if cond else self.definite).append(self.dir_receipt(path, n.lineno))

    # ----------------------------------------------------------------- effects
    def helper_escapes(self, node):
        if not self.helpers:
            return
        par = {}
        for n in _walk_no_defs(node):
            for c in _children(n):
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

    def _safe_call_for(self, call, kind):
        """A1: True if `call` leaves an argument of this kind unchanged and keeps no reference to it."""
        f = call.func
        q = self.qname(f)
        if isinstance(f, ast.Name) and f.id in self.helpers:
            return True                     # pure loader helpers (fix 4); helper_escapes handles returns
        if kind == "rows":
            return True                     # a csv.DictReader gives new dicts; handle_uses marks it as used
        if kind in ("records", "record"):
            return q in RECORD_SAFE_CALLS
        if q in MUTATOR_FUNCS:
            return False
        if q in FRAME_SAFE_CALLS:
            return True
        if isinstance(f, ast.Attribute):
            if f.attr in FRAME_ARG_METHODS and self.frame_of(f.value, {}) is not None:
                return True
            # a function of an imported module (not a method of an object of the trace)
            mods = {v.split(".")[0] for v in self.facts.aliases.values()}
            top = q.split(".")[0] if q is not None else None
            if top in mods and top not in ("builtins", "operator", "copy") and q not in UNTRACKABLE \
                    and not q.startswith("pandas.DataFrame") and not q.startswith("pandas.Series"):
                return True
        return False

    def escapes(self, node, in_body=False, root=None):
        """A1/A2: names of tracked objects that `node` lets escape: after it, another path may
        change them. In a function body (in_body=True), any binding of the object to another
        name counts too, because the checker does not follow names inside bodies."""
        out = set()
        tracked = self.frames
        if not tracked:
            return out
        par = self.facts.parents(node)
        root = node if root is None else root
        for n in _walk(node):
            if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in tracked):
                continue
            if not self._safe_use(n, par, tracked[n.id][2], in_body, root):
                out.add(n.id)
        return out

    def _safe_use(self, n, par, kind, in_body, root=None):
        cur, start = n, n
        while True:
            if cur is root and cur is not start:
                return True                         # the value of the analysed statement or expression
            p = par.get(cur)
            if p is None:
                return True
            if isinstance(p, ast.Subscript) and p.value is cur:
                if kind == "records" and cur is start:
                    if isinstance(p.slice, ast.Slice):
                        kind = "records"            # recs[1:] holds the same records
                    else:
                        kind = "record"             # recs[i] is a record: check what happens to it
                    if isinstance(p.ctx, ast.Load):
                        cur = p
                        continue
                    return True                     # recs[i] = ...: store_subscript decides
                return True                         # a column, a value of a record, or a new frame
            if isinstance(p, ast.Attribute) and p.value is cur:
                gp = par.get(p)
                if isinstance(gp, ast.Call) and gp.func is p:
                    m = p.attr
                    if m in MUTATING_METHODS or any(k.arg == "inplace" for k in gp.keywords):
                        return False
                    if kind in ("records", "rows") and m in ("copy", "__iter__", "__getitem__", "index", "pop"):
                        return False                # a list that holds the same records
                    if kind == "frame" and m in ("pipe", "__getattribute__", "__getattr__"):
                        return False
                    return True
                if isinstance(gp, ast.Subscript) and gp.value is p and p.attr in ("loc", "iloc", "at", "iat"):
                    return True
                if p.attr in UNSAFE_ATTRS or p.attr in MUTATING_METHODS or p.attr.startswith("_"):
                    return False                    # df.loc as a value, df.__dict__, recs.append as a value
                if kind == "frame":
                    if p.attr in SAFE_ATTRS or p.attr in ACCESSORS or p.attr in self.frames[start.id][1]:
                        return True                 # a column (df.year), a property or a pandas accessor
                    # an unknown accessor whose method is called (df.ta.sma(append=True)) may change the frame
                    q = gp
                    while isinstance(q, ast.Attribute):
                        q = par.get(q)
                    return not (isinstance(q, ast.Call) and isinstance(q.func, ast.Attribute) and q.func is not p)
                return kind not in ("records", "record") or p.attr in ("keys", "items", "values", "get", "count",
                                                                       "index", "copy")
            if isinstance(p, ast.Call):
                if cur is p.func:
                    return True
                gp = par.get(p)
                if cur is start and not in_body and kind in ("records", "rows") and self.qname(p.func) == "list" \
                        and len(p.args) == 1 and not p.keywords and isinstance(gp, ast.Assign) and gp.value is p \
                        and len(gp.targets) == 1 and isinstance(gp.targets[0], ast.Name) and not self.cur_cond:
                    return True                     # recs2 = list(recs): bind_value follows recs2 (same file)
                if cur is start and not in_body and isinstance(gp, (ast.For, ast.AsyncFor)) and gp.iter is p \
                        and self.elem(p, {}) is not None:
                    return True                     # for i, r in enumerate(recs): the loop variable is followed
                return self._safe_call_for(p, kind)
            if isinstance(p, ast.keyword):
                call = par.get(p)
                return isinstance(call, ast.Call) and self._safe_call_for(call, kind)
            if isinstance(p, (ast.Starred, ast.List, ast.Tuple, ast.Set)) or (isinstance(p, ast.Dict) and cur in p.values):
                cur = p
                continue
            if isinstance(p, ast.Dict):
                return True
            if isinstance(p, ast.IfExp):
                if cur is p.test:
                    return True
                cur = p
                continue
            if isinstance(p, ast.BoolOp):
                cur = p
                continue
            if isinstance(p, ast.BinOp):
                if kind == "frame":
                    return True
                cur = p                             # recs + [...] holds the same records
                continue
            if isinstance(p, (ast.Compare, ast.UnaryOp, ast.FormattedValue, ast.JoinedStr, ast.Assert, ast.If,
                              ast.While, ast.Slice, ast.withitem, ast.Expr, ast.Await)):
                return True
            if isinstance(p, ast.NamedExpr):
                return False
            if isinstance(p, (ast.Assign, ast.AnnAssign)) and cur is p.value:
                if in_body:
                    return False
                targets = p.targets if isinstance(p, ast.Assign) else [p.target]
                if cur is start:
                    return True                     # x = df: the statement's own rule decides (_escaping, aliases)
                if kind == "record" and len(targets) == 1 and isinstance(targets[0], ast.Name) \
                        and isinstance(cur, ast.Subscript) and cur.value is start and par.get(cur) is p \
                        and not self.cur_cond and isinstance(cur.slice, ast.Constant) \
                        and type(cur.slice.value) is int and cur.slice.value in (0, -1):
                    return True                     # r = recs[0]: bind_value follows it as a record
                return False
            if isinstance(p, ast.AugAssign):
                return kind == "frame" and cur is p.value
            if isinstance(p, (ast.For, ast.AsyncFor)) and cur is p.iter:
                if in_body:
                    return False
                if cur is start and self.elem(cur, {}) is not None:
                    return True                     # for r in recs: the loop variable is followed
                return False
            if isinstance(p, ast.comprehension) and cur is p.iter:
                if cur is not start:
                    return False
                comp = par.get(p)
                if kind == "frame":
                    return True                     # for c in df: column names
                return self._comp_keeps(comp, p, par)
            if isinstance(p, (ast.Return, ast.Yield, ast.YieldFrom)):
                return False
            if isinstance(p, ast.arguments):
                return False                        # a default value: the parameter is another name
            if isinstance(p, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return False
            return False

    def _comp_keeps(self, comp, gen, par):
        """True if a comprehension over records only reads its loop variable by key."""
        if not isinstance(gen.target, ast.Name):
            return False
        v = gen.target.id
        for n in _walk(comp):
            if isinstance(n, ast.Name) and n.id == v and isinstance(n.ctx, ast.Load):
                p = par.get(n)
                if isinstance(p, ast.Subscript) and p.value is n:
                    continue
                if isinstance(p, ast.Attribute) and p.value is n and p.attr in ("get", "keys", "items", "values") \
                        and isinstance(par.get(p), ast.Call):
                    continue
                if isinstance(p, ast.Compare) or (isinstance(p, ast.Call) and n in p.args
                                                  and self.qname(p.func) in RECORD_SAFE_CALLS):
                    continue
                return False
        return True

    def effects(self, node, root=None):
        """Stop tracking objects that this code may change by another path. For the header of a
        compound statement, root is the statement (for r in recs[:1] binds r to a record)."""
        self.helper_escapes(node)
        for name in self.escapes(node, root=root):                       # A1
            self.forget_source(name)
        for n in _walk_no_defs(node):
            if isinstance(n, ast.comprehension):
                for name in _display_escapes(n.iter) & set(self.frames):
                    self.forget_source(name)
            if isinstance(n, ast.NamedExpr):
                self.forget(_names(n.target))
                for name in _escaping(n.value) & set(self.frames):
                    self.forget_source(name)
            elif isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
                if n.attr in ("name", "names"):                         # N4
                    self.forget_frames()
                self._store_chain(n)
            elif isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)):
                if isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name):
                    self.forget([n.value.value.id])
                self._store_chain(n)
            elif isinstance(n, ast.Call):
                f = n.func
                q = self.qname(f)
                if q in UNTRACKABLE or (isinstance(f, ast.Name) and f.id in UNTRACKABLE):
                    self.frames.clear()
                    self.consts.clear()
                    self.forgot_all_frames = True
                    continue
                if isinstance(f, ast.Attribute) and f.attr in ("rename_axis", "set_names", "set_axis", "rename") \
                        and any(k.arg == "inplace" and not _is_const(k.value, (False,)) for k in n.keywords) \
                        and (f.attr != "rename" or isinstance(f.value, ast.Attribute)):
                    self.forget_frames()                                # N4
                args = [*n.args, *(k.value for k in n.keywords)]
                bare = {a.id for a in args if isinstance(a, ast.Name)} | \
                    {a.value.id for a in args if isinstance(a, ast.Starred) and isinstance(a.value, ast.Name)}
                if isinstance(f, ast.Attribute):
                    obj = f.value
                    root = obj
                    while isinstance(root, (ast.Subscript, ast.Attribute)):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in self.frames and (
                            f.attr in MUTATING_METHODS or any(k.arg == "inplace" for k in n.keywords)):
                        self.forget_source(root.id)                     # recs[0].update(...), A3
                    if f.attr in ESCAPE_METHODS:
                        self.forget(bare & set(self.frames))
                elif isinstance(f, ast.Name):
                    if not (f.id in PURE_FUNCTIONS and self.facts.builtin(f.id)) and f.id not in self.helpers:
                        self.forget(bare & set(self.frames))
                else:
                    self.forget(bare & set(self.frames))

    def _store_chain(self, t):
        """A3: a store through attributes or subscripts of a tracked object (h.d[...] = ...,
        df.columns.values[0] = ...)."""
        chain, base = [], t
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            chain.append(base)
            base = base.value
        attrs = [c.attr for c in chain if isinstance(c, ast.Attribute)]
        if any(a in ("columns", "index", "values", "axes", "_mgr", "_data", "array", "name", "names") for a in attrs[1:]) \
                or (len(chain) > 1 and any(a in ("columns", "index") for a in attrs)):
            self.forget_frames()
        if isinstance(base, ast.Name) and base.id in self.frames:
            direct = len(chain) == 1 and isinstance(chain[0], ast.Subscript)
            two = len(chain) == 2 and all(isinstance(c, ast.Subscript) for c in chain)
            if not (direct or two):
                self.forget_source(base.id)

    def forget_source(self, name):
        obj = self.frames.get(name)
        self.forget([name])
        if obj and obj[2] in ("records", "record", "rows"):
            self.forget([k for k, o in self.frames.items() if o[0] == obj[0] and o[2] in ("records", "record", "rows")])

    def add_keys(self, name, keys):
        obj = self.frames[name]
        if obj[2] in ("records", "record"):
            for o in self.frames.values():
                if o[0] == obj[0] and o[2] in ("records", "record"):
                    o[1].update(keys)
        else:
            obj[1].update(keys)

    def def_effects(self, d):
        """A2: a function, lambda or class body may run later and change an object it names."""
        hit = set()
        for n in _walk(d):
            if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.ctx, (ast.Store, ast.Del)):
                hit |= _names(n.value)
                if isinstance(n, ast.Attribute) and n.attr in ("name", "names"):
                    self.forget_frames()
                if isinstance(n, (ast.Subscript, ast.Attribute)):
                    base = n.value
                    attrs = []
                    while isinstance(base, (ast.Subscript, ast.Attribute)):
                        if isinstance(base, ast.Attribute):
                            attrs.append(base.attr)
                        base = base.value
                    if any(a in ("columns", "index", "values", "axes") for a in attrs):
                        self.forget_frames()
            elif isinstance(n, ast.Global):
                hit |= set(n.names)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and (
                    n.func.attr in INPLACE_METHODS | MUTATING_METHODS
                    or any(k.arg == "inplace" for k in n.keywords)):
                root = n.func.value
                while isinstance(root, (ast.Subscript, ast.Attribute)):
                    root = root.value
                if isinstance(root, ast.Name):
                    hit.add(root.id)
                if n.func.attr in ("rename_axis", "set_names", "set_axis", "rename") and \
                        any(k.arg == "inplace" for k in n.keywords):
                    self.forget_frames()
            elif isinstance(n, ast.Call) and (self.qname(n.func) in UNTRACKABLE):
                self.frames.clear()
                self.consts.clear()
                self.forgot_all_frames = True
                return
        # A2: any use in the body that would let the object escape, with bindings in the body counted
        # a default value becomes the parameter, and a lambda returns its body: climbing from a name to
        # the def or lambda (root=d) reaches ast.arguments or ast.Lambda, which count as escapes
        defaults = [] if isinstance(d, ast.ClassDef) else \
            [x for x in d.args.defaults + d.args.kw_defaults if x is not None]
        parts = ([d.body] if isinstance(d, ast.Lambda) else list(d.body)) + defaults
        for part in parts:
            hit |= self.escapes(part, in_body=True, root=d)
            for n in _walk(part):
                if isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                    for name in _names(n.iter) & set(self.frames):
                        if self.frames[name][2] in ("records", "rows", "record"):
                            hit.add(name)
        for name in hit & (set(self.frames) | set(self.consts)):
            self.forget_source(name)

    # ---------------------------------------------------------------- bindings
    def _rows_kept(self, call):
        """R2: True if read_csv(...) keeps at least one row of a non-empty file."""
        nrows = _kw(call, "nrows")
        if nrows is None:
            return True
        return isinstance(nrows, ast.Constant) and type(nrows.value) is int and nrows.value > 0

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
        if base not in self.schema or _not_a_file(path) or path.lstrip().startswith("~"):
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
        if why:
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
                                          "available": sorted(cols, key=str),
                                          "found_in": sorted(f for f, s in self.schema.items() if c in s and f != base),
                                          "near_matches": difflib.get_close_matches(
                                              c, sorted((x for x in cols if isinstance(x, str)), key=str),
                                              n=3, cutoff=0.7)})
                cols = set(lits) & cols
        nonempty = self._rows_kept(call)                                 # R2
        kind = "chunks" if any(k in kws for k in ("chunksize", "iterator")) else "frame"
        self.consts.pop(target, None)
        self.frames[target] = (base, cols, kind, nonempty)

    def bind_value(self, target, value, lineno):
        if self.alias and isinstance(value, ast.Name) and value.id in self.frames and value.id != target:
            self.consts.pop(target, None)
            self.frames[target] = self.frames[value.id]
            g = self.groups.get(value.id) or {value.id}
            g.add(target)
            for n in g:
                self.groups[n] = g
            return True
        if self.is_reader(value):
            self.bind_reader(target, value, lineno)
            return True
        res, arg = self.helper_value(value, {})
        if isinstance(res, tuple):
            self.consts.pop(target, None)
            self.frames[target] = (res[0], set(res[1]), res[2], res[3])
            return True
        if res == "same" and self.alias and arg[1] != target:
            return self.bind_value(target, ast.Name(id=arg[1], ctx=ast.Load()), lineno)
        rec = self.records_from(value, {})
        if rec:
            self.reads += 1
            self.consts.pop(target, None)
            self.frames[target] = (rec[0], set(rec[1]), "records", True)
            return True
        q, f = (self.qname(value.func) if isinstance(value, ast.Call) else None), getattr(value, "func", None)
        if q == "csv.DictReader" and _call_name(value) in ("csv.DictReader", "DictReader") and value.args:
            base = self.context(self.handle_path(value.args[0]), ".csv")
            if base and _plain_dictreader(value):
                self.consts.pop(target, None)
                self.frames[target] = (base, set(self.schema[base]), "rows", True)
                return True
            if base:
                self._untracked(f"line {lineno}: csv.DictReader of {base} with arguments that may change "
                                "the field names")
        if q == "list" and value.args and len(value.args) == 1 and not value.keywords:
            el = self.elem(value.args[0], {})
            if el and el[0][0] != "pair" and el[0][2] == "record":
                self.consts.pop(target, None)
                self.frames[target] = (el[0][0], set(el[0][1]), "records", el[1])
                if isinstance(value.args[0], ast.Name) and value.args[0].id in self.frames:
                    o = self.frames[value.args[0].id]
                    if o[2] == "rows":
                        self.frames[value.args[0].id] = (o[0], o[1], o[2], False)
                return True
        if isinstance(f, ast.Attribute) and f.attr == "to_dict" and _records_arg(value) \
                and isinstance(f.value, ast.Name) and self.frames.get(f.value.id, (0, 0, 0))[2] == "frame":
            o = self.frames[f.value.id]
            self.consts.pop(target, None)
            self.frames[target] = (o[0], set(o[1]), "records", o[3])
            return True
        if q == "pandas.DataFrame" and value.args \
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
        if isinstance(value, ast.Call) and self.is_builtin_open(value) and _call_name(value) == "open" \
                and _builtin_open_mode(value) == "r":
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
            fr = self.frame_of(value, {})
            if fr is not None:
                self.consts.pop(target, None)
                self.frames[target] = (fr[0], set(fr[1]), "frame", fr[3])
                return True
        return False

    def handle_uses(self, s):
        """R1/R2: a file handle used other than as the argument of json.load, csv.DictReader or
        csv.reader is not followed; a csv.DictReader used other than as a loop source may be
        empty afterwards."""
        par = self.facts.parents(s)
        for n in _walk(s):
            if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
                continue
            c = self.consts.get(n.id)
            p = par.get(n)
            if isinstance(c, tuple) and c[0] == "handle":
                if isinstance(p, ast.Call) and n in p.args and len(p.args) == 1 and \
                        self.qname(p.func) in ("json.load", "csv.DictReader", "csv.reader"):
                    continue
                self.consts.pop(n.id, None)
            o = self.frames.get(n.id)
            if o is not None and o[2] == "rows":
                if isinstance(p, (ast.For, ast.AsyncFor, ast.comprehension)) and p.iter is n:
                    continue
                if isinstance(p, ast.Call) and self.qname(p.func) == "list" and n in p.args:
                    continue
                self.frames[n.id] = (o[0], o[1], o[2], False)

    # -------------------------------------------------------------- statements
    def _safe_context(self, e):
        """C3: a context manager that does not suppress errors."""
        if not isinstance(e, ast.Call):
            return False
        if self.qname(e.func) in SAFE_CONTEXTS:
            return True
        return isinstance(e.func, ast.Attribute) and e.func.attr == "open" and isinstance(e.func.value, ast.Call) \
            and self.qname(e.func.value.func) in ("pathlib.Path", "pathlib.PurePath")

    def may_end(self, stmt):
        """True if the statement may end the step without an error (C4, C5)."""
        k = id(stmt)
        if k not in self._may_end_cache:
            self._may_end_cache[k] = self._may_end(stmt)
        return self._may_end_cache[k]

    def _may_end(self, stmt):
        par = self.facts.parents(stmt)
        for n in _walk_no_defs(stmt):
            if self.facts.is_end_call(n) or _is_exit_raise(n) or self.facts.poison(n, par):
                return True
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in stmt.decorator_list:
                if self.facts.expr_may_end(d, par):
                    return True
            if isinstance(stmt, ast.ClassDef):
                for st in stmt.body:
                    if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                            self.facts.expr_may_end(st, par):
                        return True
        return False

    def tail_end(self, s):
        """C5: the ender call that is the whole value of `final_answer(x)` or `y = final_answer(x)`,
        if its own arguments cannot end the step."""
        v = s.value if isinstance(s, (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return)) else None
        if not (isinstance(v, ast.Call) and self.facts.is_end_call(v)):
            return None
        par = self.facts.parents(v)
        for a in [*v.args, *(k.value for k in v.keywords)]:
            if self.facts.expr_may_end(a, par):
                return None
        if isinstance(s, ast.Assign) and any(not isinstance(t, ast.Name) for t in s.targets):
            return None
        return v

    def block(self, stmts, conditional):
        for s in stmts:
            if self.in_loop and isinstance(s, (ast.Break, ast.Continue)):
                return LOOP
            r = self.stmt(s, conditional)
            if r:
                return r
            if not conditional and (self.may_end(s) or (self.in_loop and _may_skip(s))):
                conditional = True
        return None

    def loop_body(self, stmts, conditional):
        saved, self.in_loop = self.in_loop, True
        try:
            return self.block(stmts, conditional)
        finally:
            self.in_loop = saved

    def header_cond(self, expr, conditional):
        return conditional or (expr is not None and self.facts.expr_may_end(expr))

    def stmt(self, s, conditional):
        self.cur_cond = conditional
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in s.decorator_list:
                self.scan_creators(d)
            if not isinstance(s, ast.ClassDef):
                for d in s.args.defaults + s.args.kw_defaults:
                    if d is not None:
                        self.scan_creators(d)
            self.check_reads(s, True)
            for d in s.decorator_list:
                self.effects(d)
            self.def_effects(s)
            if isinstance(s, ast.ClassDef):
                for st in s.body:
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.def_effects(st)
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
                self.effects(t)
            return None
        if isinstance(s, (ast.For, ast.AsyncFor)):
            hc = self.header_cond(s.iter, conditional)
            self.cur_cond = hc
            self.scan_creators(s.iter)
            self.handle_uses(s.iter)
            self.check_reads(s.iter, hc)
            self.effects(s.iter, root=s)
            for name in _display_escapes(s.iter) & set(self.frames):
                self.forget_source(name)
            el = self.elem(s.iter, {})
            self.bind_target(s.target, el)
            self.effects_target(s.target)
            first = el is not None and el[1] and not hc and not isinstance(s, ast.AsyncFor)
            r = self.loop_body(s.body, not first)
            if isinstance(s.iter, ast.Name) and s.iter.id in self.frames and self.frames[s.iter.id][2] in ("rows", "chunks"):
                o = self.frames[s.iter.id]
                self.frames[s.iter.id] = (o[0], o[1], o[2], False)
            self.block(s.orelse, True)
            return CELL if first and r == CELL else None
        if isinstance(s, (ast.If, ast.While, ast.Try)) or type(s).__name__ == "TryStar":
            h = getattr(s, "test", None)
            if h is not None:
                self.cur_cond = self.header_cond(h, conditional)
                self.scan_creators(h)
                self.handle_uses(h)
                self.check_reads(h, self.cur_cond)
                self.effects(h, root=s)
            body = self.loop_body if isinstance(s, ast.While) else self.block
            body(s.body, True)
            for field in ("orelse", "finalbody"):
                self.block(getattr(s, field, []) or [], True)
            for hd in getattr(s, "handlers", []) or []:
                if hd.name:
                    self.forget([hd.name])
                if hd.type is not None:
                    self.effects(hd.type)
                self.block(hd.body, True)
            return None
        if isinstance(s, ast.Match):
            self.cur_cond = self.header_cond(s.subject, conditional)
            self.scan_creators(s.subject)
            self.handle_uses(s.subject)
            self.check_reads(s.subject, self.header_cond(s.subject, conditional))
            self.effects(s.subject, root=s)
            for case in s.cases:
                self.forget(_names(case.pattern))
                self.block(case.body, True)
            return None
        if isinstance(s, (ast.With, ast.AsyncWith)):
            safe = not isinstance(s, ast.AsyncWith) and all(self._safe_context(it.context_expr) for it in s.items)
            hc = conditional
            for it in s.items:
                hc = self.header_cond(it.context_expr, hc)
                self.cur_cond = hc
                self.scan_creators(it.context_expr)
                self.handle_uses(it.context_expr)
                self.check_reads(it.context_expr, hc)
                self.effects(it.context_expr, root=s)
                if it.optional_vars is not None:
                    self.forget(_names(it.optional_vars))
                    self.effects_target(it.optional_vars)
                    p = self.handle_path(it.context_expr)
                    if p is not None and isinstance(it.optional_vars, ast.Name) and not hc:
                        self.consts[it.optional_vars.id] = ("handle", p)
            r = self.block(s.body, hc or not safe)                       # C3
            return r if safe else None

        value = getattr(s, "value", None)
        # C5: a statement that may end the step anywhere but in its own tail call has no checked reads
        if not conditional and self.may_end(s) and self.tail_end(s) is None:
            conditional = True
        self.cur_cond = conditional
        self.scan_creators(s)
        self.handle_uses(s)
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
            if s.msg is not None:
                self.check_reads(s.msg, True)
        targets = s.targets if isinstance(s, ast.Assign) else \
            [s.target] if isinstance(s, (ast.AnnAssign, ast.AugAssign)) else []
        for t in targets:
            if not isinstance(t, ast.Name):
                self.check_reads(t, conditional)
        if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Subscript) \
                and isinstance(s.target.value, ast.Name) and s.target.value.id in self.frames:
            obj = self.frames[s.target.value.id]
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
            for n in {n for n in _escaping(value) if n in self.frames} \
                    - {t.id for t in targets if isinstance(t, ast.Name)}:
                self.forget_source(n)
            if len(targets) > 1 or any(not isinstance(t, ast.Name) for t in targets):
                for n in _escaping(value) & set(self.frames):
                    self.forget_source(n)                               # x = y = df, a.b = df
        for t in targets:
            if isinstance(t, ast.Name):
                self.helpers.pop(t.id, None)
                if value is not None and not conditional and not isinstance(s, ast.AugAssign) \
                        and len(targets) == 1 and self.bind_value(t.id, value, s.lineno):
                    continue
                self.frames.pop(t.id, None)
                p = _resolve(value, self.consts) if value is not None else None
                if p is not None and not conditional and isinstance(s, ast.Assign):
                    self.consts[t.id] = p
                else:
                    self.consts.pop(t.id, None)
            elif isinstance(t, ast.Subscript):
                self.store_subscript(t)
            elif isinstance(t, (ast.Tuple, ast.List, ast.Starred)):
                for n in _names(t):
                    self.forget_source(n)
                self.effects_target(t)
            elif isinstance(t, ast.Attribute):
                root = t.value
                while isinstance(root, (ast.Subscript, ast.Attribute)):
                    root = root.value
                if isinstance(root, ast.Name):
                    self.forget_source(root.id)
        ended = isinstance(s, ast.Raise) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                                             and self._certain_end(s.value))
        return CELL if ended else None

    def _certain_end(self, call):
        """A call that always ends the step: final_answer, exit or quit (not rebound), sys.exit, os._exit."""
        f = call.func
        if isinstance(f, ast.Name) and f.id in END_NAMES:
            return True
        return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in END_ATTRS

    def effects_target(self, t):
        """A target that stores into a tracked object (for df["x"] in ..., with ... as h.f)."""
        for n in _walk(t):
            if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.ctx, ast.Store):
                root = n.value
                while isinstance(root, (ast.Subscript, ast.Attribute)):
                    root = root.value
                if isinstance(root, ast.Name):
                    self.forget_source(root.id)
                if isinstance(n, ast.Attribute) and n.attr in ("name", "names"):
                    self.forget_frames()

    def store_subscript(self, t):
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
                self.forget_source(base.id)
        elif obj[2] in ("records", "record"):
            if keys and isinstance(t.value, ast.Subscript) and isinstance(t.value.value, ast.Name) \
                    and obj[2] == "records":
                self.add_keys(base.id, keys)
            elif obj[2] == "records":
                self.forget_source(base.id)
            else:
                self.forget_source(base.id)            # rec["a"]["b"] = ...: a nested object


def _dir_writers(tree, facts):
    """Gate v8, R1, with import aliases (F4)."""
    for n in _walk(tree):
        if isinstance(n, ast.Call):
            q = facts.qname(n.func)
            name = _call_name(n)
            if q in DIR_WRITER_NAMES or name in DIR_WRITER_NAMES or \
                    (isinstance(n.func, ast.Attribute) and n.func.attr in DIR_WRITER_ATTRS):
                return True
    return False


def decide(code, schema, state=None, exists=None, tolerant_keys=False, alias=False, locate=None):
    """Same arguments and result as version 8."""
    if state is not None:
        state.pending = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError) as e:
        return {"outcome": "unknown", "reason": f"invalid Python: {type(e).__name__}", "files": [],
                "written": [], "substitutions": []}
    kids, par = _index(tree)
    _TLS.kids = kids
    try:
        facts = Facts(tree, state, par)
        w = _Walker(schema, state, exists, tolerant_keys, alias, locate=locate,
                    list_consts=_list_consts(tree), dir_writers=_dir_writers(tree, facts), facts=facts)
        w.block(tree.body, False)
    finally:
        _TLS.kids = None
    if state is not None:
        frames, consts = w.export()
        state.pending = (frames, consts, facts.export(w.cwd_changed_here or w.cwd_changed, w.forgot_all_frames))
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
