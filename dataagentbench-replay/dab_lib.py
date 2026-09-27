"""Shared code for the DataAgentBench (DAB) replay.

DAB [Ma et al., arXiv:2603.20576] runs a ReAct agent with four tools:
list_db, query_db (SQL or MongoDB), execute_python and return_answer. The
baseline logs are in branch refs/pull/6/head (commit 0a1a9c0c), one
tool_calls.jsonl per run. This module reads them and models the harness as
it is written at DAB commit 0290945:

  * Every successful tool result is stored under var_<tool_call_id>. A
    result longer than 10,000 characters (JSON) goes to
    file_storage/<tool_call_id>.json and the variable holds that path.
  * execute_python writes  code = \"\"\"<agent code>\"\"\"  followed by
    exec(code, env_args)  to a file and runs it in a new Python process
    (ExecTool._exec). Python decodes escape sequences in that literal before
    the agent's code runs. env_args holds the stored results only, so no
    Python name survives from one call to the next.
  * The agent sees only the last line of a Python error.
  * MongoDB queries without "limit" return at most 5 documents
    (MongoQueryDBTool.check_args).
  * query_db turns every result into a DataFrame and returns
    df.to_dict("records") (db_config.serialize), for SQL and MongoDB alike.
    Every record therefore has every column, so the first record of a stored
    result gives its exact key set.
"""
import ast
import json
import os
import re
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

PREVIEW = 10000          # DataAgent.PREVIEW_LENGTH and BaseTool log truncation
MODEL_NAMES = {"gpt5.1": "gpt-5.1"}


# ------------------------------------------------------------------ reading
def run_files(root):
    """Yield (model, dataset, query, run, path) for every tool_calls.jsonl."""
    for dirpath, _, files in os.walk(root):
        if "tool_calls.jsonl" not in files:
            continue
        parts = os.path.relpath(dirpath, root).split(os.sep)
        if not parts[0].startswith("results-"):
            continue
        model = parts[0][len("results-"):]
        model = MODEL_NAMES.get(model, model)
        run = next((p for p in parts if re.fullmatch(r"run_\d+", p)), parts[-1])
        yield model, parts[1].removeprefix("query_"), parts[2], run, os.path.join(dirpath, "tool_calls.jsonl")


def read_calls(path):
    """Tool-call records of one run, in order. Unreadable lines are skipped and counted."""
    out, bad = [], 0
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                bad += 1
    return out, bad


def result_text(preview):
    """The logged result preview (json.dumps(result)[:10000]) as a string."""
    if not isinstance(preview, str):
        return json.dumps(preview)
    for p in (preview, preview + '"'):
        try:
            v = json.loads(p)
            return v if isinstance(v, str) else preview
        except ValueError:
            pass
    return preview


def result_value(preview):
    """The logged result as a Python value, or None if the preview is cut."""
    try:
        return json.loads(preview)
    except (ValueError, TypeError):
        return None


def first_record_keys(preview):
    """Keys of the first record of a list-of-dicts result, also from a cut preview."""
    if not isinstance(preview, str) or not preview.startswith("[{"):
        return None
    try:
        rec, _ = json.JSONDecoder().raw_decode(preview, 1)
    except ValueError:
        return None
    return set(rec) if isinstance(rec, dict) else None


# ------------------------------------------------------------ code wrapper
def compiles(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # invalid escapes such as "\d" in agent code
            compile(code, "<cell>", "exec", dont_inherit=True)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def wrapper(code):
    """Reproduce ExecTool._exec.

    Returns (own, status, decoded): own is True if the agent's code compiles
    on its own; status is "same" (the harness runs the agent's code),
    "changed" (it runs a different but valid program), "outer_syntax" (the
    file does not parse), "outer_structure" (a \"\"\" in the code ends the
    literal early) or "inner_syntax" (the decoded code does not compile);
    decoded is the code that runs, or None."""
    c = (code or "").strip()
    own = compiles(c)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(f'code = """{c}"""')
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return own, "outer_syntax", None
    b = tree.body
    if not (len(b) == 1 and isinstance(b[0], ast.Assign) and isinstance(b[0].value, ast.Constant)
            and isinstance(b[0].value.value, str)):
        return own, "outer_structure", None
    dec = b[0].value.value
    if not compiles(dec):
        return own, "inner_syntax", None
    return own, ("same" if dec == c else "changed"), dec


SYNTAX = re.compile(r"\b(SyntaxError|IndentationError|TabError)\b")


def harness_syntax(own, status, error):
    """A failure that the code wrapper causes: valid code, SyntaxError after decoding."""
    return bool(own and status in ("outer_syntax", "outer_structure", "inner_syntax")
                and SYNTAX.search(error or ""))


# ---------------------------------------------------------- classification
SQL_COL = re.compile(r"UndefinedColumn|no such column|Referenced column .{1,200}? not found|"
                     r"column .{1,200}? does not exist|does not have a column named", re.S)
SQL_TAB = re.compile(r"UndefinedTable|no such table|Table with name .{1,200}? does not exist|"
                     r"relation .{1,200}? does not exist", re.S)
SQL_HINT = re.compile(r"Candidate bindings|Did you mean|HINT:|Perhaps you meant")
SQL_SYNTAX = re.compile(r"syntax error|Parser Error|ParserException|SyntaxError|unrecognized token|incomplete input", re.I)
SQL_FUNC = re.compile(r"UndefinedFunction|No function matches|function .{1,120}? does not exist|"
                      r"Scalar Function with name|Could not choose a best candidate function", re.S)
PY_LAST = re.compile(r"^Execution failed with exit code (-?\d+)\n(.*)$", re.S)
PY_TYPE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))\b:?\s?(.*)$", re.S)
VAR = re.compile(r"var_[\w.:\-]+|file_storage|\b\d{15,}\.json\b|function-call-\d+|functions\.\w+:\d+")


def engine(msg):
    for pat, e in (("DuckDB", "duckdb"), ("SQLite", "sqlite"), ("Postgres", "postgres"),
                   ("psycopg", "postgres"), ("Mongo", "mongo")):
        if pat in msg:
            return e
    return None


def key_list(text):
    """Keys named in a KeyError message: 'k', or pandas lists such as
    "None of [Index(['a', 'b'], dtype='object')] are in the [columns]"."""
    text = text.strip()
    m = re.match(r"^Index\(\[(.*?)\]", text)
    if m:
        return re.findall(r"'([^']*)'", re.sub(r"dtype='[^']*'", "", m.group(1)))
    try:
        v = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        v = text
    if isinstance(v, (str, int, float)):
        s = str(v)
        m = (re.search(r"\[Index\(\[(.*?)\]", s) or re.match(r"^\[(.*)\] not in index$", s)
             or re.search(r"None of \[(.*?)\] are in the", s) or re.search(r"Label\(s\) \[(.*?)\] do not exist", s))
        if m:
            return re.findall(r"'([^']*)'", re.sub(r"dtype='[^']*'", "", m.group(1)))
        return [s]
    if isinstance(v, (tuple, list)):
        return [str(x) for x in v]
    return []


def classify(tool, db_type, error):
    """(group, detail) for a failed tool call.

    group is one of: sql:missing-column, sql:missing-table, sql:syntax,
    sql:function, sql:other, mongo:query-json, mongo:other, py:<ErrorType>,
    py:data-key, py:data-file, interface:var-key, interface:var-file,
    interface:var-name, py:print-format, py:timeout, other."""
    e = error or ""
    if tool == "query_db":
        if e.startswith("Collection does not exist"):
            return "mongo:missing-collection", e.split(":", 1)[-1].strip()
        if e.startswith("Invalid Mongo query") or db_type == "mongo":
            return ("mongo:query-json" if "JSON" in e[:60] else "mongo:other"), None
        if SQL_COL.search(e):
            return "sql:missing-column", None
        if SQL_TAB.search(e):
            return "sql:missing-table", None
        if SQL_FUNC.search(e):
            return "sql:function", None
        if SQL_SYNTAX.search(e):
            return "sql:syntax", None
        return "sql:other", None
    if tool != "execute_python":
        return "other", None
    if e.startswith("Failed to parse the printed result"):
        return "py:print-format", None
    m = PY_LAST.match(e)
    last = m.group(2) if m else e
    if m and m.group(1) == "124" or "timed out" in e.lower():
        return "py:timeout", None
    t = PY_TYPE.match(last)
    if not t:
        return "py:other", None
    et, rest = t.group(1).split(".")[-1], t.group(2)
    if et in ("IndentationError", "TabError"):
        et = "SyntaxError"
    if et == "KeyError":
        return ("interface:var-key" if VAR.search(rest) else "py:data-key"), key_list(rest)
    if et == "FileNotFoundError":
        pm = re.search(r"No such file or directory: '([^']+)'|File (\S+) does not exist", rest)
        path = next((g for g in pm.groups() if g), None) if pm else None
        return ("interface:var-file" if VAR.search(rest) else "py:data-file"), path
    if et == "NameError":
        nm = re.search(r"name '([^']+)' is not defined", rest)
        name = nm.group(1) if nm else None
        return ("interface:var-name" if name and name.startswith("var_") else "py:NameError"), name
    return "py:" + et, None


DATA_REF = {"sql:missing-column", "sql:missing-table", "mongo:missing-collection", "py:data-key", "py:data-file"}
INTERFACE = {"interface:var-key", "interface:var-file", "interface:var-name"}


# ----------------------------------------------------- results and names
def ident(name):
    """A Python identifier for a DAB variable name (Gemini and Kimi ids are not identifiers)."""
    if name.isidentifier():
        return name
    return "_dabvar_" + re.sub(r"\W", "_", name)


class _EnvAccess(ast.NodeTransformer):
    """locals()['var_x'] / globals()['var_x'] / .get('var_x') -> a plain name.

    At module level of exec(code, env_args), locals() is env_args, so this
    keeps the meaning. Inside a function, locals() is the function's own
    namespace, so only globals() is rewritten there."""

    def __init__(self, names):
        self.names, self.depth, self.n = names, 0, 0

    def _scope(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = visit_ClassDef = _scope

    def _env_call(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args \
                and not node.keywords:
            return node.func.id == "globals" or (node.func.id == "locals" and self.depth == 0)
        return False

    def visit_Subscript(self, node):
        self.generic_visit(node)
        s = node.slice
        if self._env_call(node.value) and isinstance(node.ctx, ast.Load) and isinstance(s, ast.Constant) \
                and s.value in self.names:
            self.n += 1
            return ast.copy_location(ast.Name(id=ident(s.value), ctx=ast.Load()), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "get" and self._env_call(f.value) \
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in self.names:
            self.n += 1
            return ast.copy_location(ast.Name(id=ident(node.args[0].value), ctx=ast.Load()), node)
        return node


def rewrite_env_access(code, names):
    """Return (code, n_rewritten). Code that does not parse is returned unchanged."""
    if not names or ("locals()" not in code and "globals()" not in code):
        return code, 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code, 0
    t = _EnvAccess(set(names))
    tree = t.visit(tree)
    if not t.n:
        return code, 0
    return ast.unparse(ast.fix_missing_locations(tree)), t.n


def env_state(env, stored_keys, new_state):
    """Gate schema and GateState for the variables that exist before a Python call.

    env: the logged env_args {var name: value}. stored_keys: {file name:
    set of keys} for stored query_db results (exact: every record has every
    column). Returns (schema, state, info)."""
    schema, st = {}, new_state()
    info = {"records": 0, "record": 0, "paths": 0, "exact_files": 0}
    for name, v in env.items():
        n = ident(name)
        if isinstance(v, str):
            st.consts[n] = v
            if v.startswith("file_storage/") and v.endswith(".json"):
                info["paths"] += 1
                base = os.path.basename(v)
                if base in stored_keys:
                    schema[base] = set(stored_keys[base])
                    info["exact_files"] += 1
        elif isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
            keys = set().union(*(set(map(str, r)) for r in v))
            st.frames[n] = (name, frozenset(keys), "records", True)
            schema[name] = keys
            info["records"] += 1
        elif isinstance(v, dict):
            st.frames[n] = (name, frozenset(map(str, v)), "record", True)
            schema[name] = set(map(str, v))
            info["record"] += 1
    return schema, st, info


# --------------------------------------------------------- workspace model
WRITE_CALLS = {"to_csv", "to_json", "to_excel", "to_parquet", "to_pickle", "to_feather", "write_text",
               "write_bytes", "dump", "save", "savetxt", "savez", "system", "run", "Popen", "call",
               "check_call", "check_output", "copy", "copyfile", "copy2", "move", "rename", "replace",
               "makedirs", "mkdir", "urlretrieve", "savefig", "to_sql", "remove", "unlink", "rmtree"}


def write_calls(code):
    """Number of calls that may create, move or delete files (open() in a write mode included)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return 0
    n = 0
    for c in ast.walk(tree):
        if not isinstance(c, ast.Call):
            continue
        f = c.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        if name == "open":
            mode = c.args[1] if len(c.args) > 1 else next((k.value for k in c.keywords if k.arg == "mode"), None)
            if mode is not None and not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)
                                         and set(mode.value) <= set("rbt")):
                n += 1
        elif name in WRITE_CALLS:
            n += 1
    return n


class Workspace:
    """The files in the exec container of one run, as far as the log shows them.

    Each run gets a new container from python:3.12-slim with pandas and
    pyarrow (DAB Dockerfile); autogen mounts the work directory at /workspace
    and runs code there (DockerCommandLineCodeExecutor). The work directory
    starts empty. It gains file_storage/<id>.json for each stored result, and
    the files that earlier Python calls write. /tmp also starts empty in the
    new container (gate v6, fix 3). Other absolute paths are unknown. After a
    call whose writes the gate cannot resolve (dynamic paths, shell, moves), a
    path that is not known to exist is unknown."""

    EMPTY_ROOTS = ("/tmp",)

    def __init__(self, empty_tmp=True):
        self.stored, self.written, self.opaque = set(), set(), False
        self.empty_tmp = empty_tmp

    @staticmethod
    def norm(path):
        p = os.path.normpath(path)
        if p.startswith("./"):
            p = p[2:]
        if p == "/workspace":
            return "."
        if p.startswith("/workspace/"):
            p = p[len("/workspace/"):]
        return p

    def exists(self, path):
        p = self.norm(path)
        if p in (".", "file_storage") or p in self.stored or p in self.written:
            return True
        if self.opaque or p.startswith("~"):
            return None
        if os.path.isabs(p):
            if self.empty_tmp and any(p == r for r in self.EMPTY_ROOTS):
                return True
            if self.empty_tmp and any(p.startswith(r + "/") for r in self.EMPTY_ROOTS):
                return False
            return None
        return False

    def after_call(self, resolved_writes, n_write_calls, ran):
        """Record the writes of a Python call. A failed call may have stopped before a write."""
        if not ran:
            if n_write_calls:
                self.opaque = True
            return
        for w in resolved_writes:
            self.written.add(self.norm(w))
        if n_write_calls > len(resolved_writes):
            self.opaque = True
