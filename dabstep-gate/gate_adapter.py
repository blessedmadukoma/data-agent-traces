"""Small adapter between the analysis scripts and your gate.py.

It imports decide() from your gate directory, builds a schema from the
DABstep context files, and reads the gate's outcome and reason whatever
type decide() returns (str, dict, tuple or an object with attributes).
"""
import collections
import csv
import json
import os
import re
import sys


def load_decide(gate_dir):
    sys.path.insert(0, os.path.abspath(gate_dir))
    from gate import decide  # noqa: E402  (your gate.py)
    return decide


def load_state(gate_dir):
    """Return the GateState class of your gate.py (for --carry)."""
    sys.path.insert(0, os.path.abspath(gate_dir))
    from gate import GateState  # noqa: E402
    return GateState


def csv_reader_of_gate():
    """The CSV schema rule of the loaded gate: gate.csv_columns (v7, rule 1) if the
    gate module has it, else None (v6: the raw first row)."""
    g = sys.modules.get("gate")
    return getattr(g, "csv_columns", None)


def context_schema(data):
    """{basename: set(columns or record keys)} for the DABstep context files.

    CSV files give their header (v6) or the union of the pandas and csv.reader
    headers (v7, when the loaded gate has csv_columns). JSON files that hold a
    list of records give the union of the record keys (used by read_json and
    by receipts). Call after load_decide / load_state.
    """
    c = os.path.join(data, "data/context")
    cols_of = csv_reader_of_gate()
    out = {}
    for fn in os.listdir(c):
        p = os.path.join(c, fn)
        if fn.endswith(".csv") and cols_of is not None:
            cols = cols_of(p)
            if cols is not None:
                out[fn] = cols
        elif fn.endswith(".csv"):
            with open(p) as f:
                out[fn] = set(next(csv.reader(f)))
        elif fn.endswith(".json"):
            recs = json.load(open(p))
            if isinstance(recs, list) and recs and all(isinstance(r, dict) for r in recs):
                keys = set()
                for r in recs:
                    keys |= set(r)
                out[fn] = keys
    return out


def _names(code):
    """(names bound, names read) in one cell, or (None, None) for invalid code."""
    import ast
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None, None
    bound, read = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, ast.Store) else read).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
    return bound, read


def stateful_submissions(traces, min_cells=20, limit=0.5):
    """Decide per submission whether names persist from one cell to the next.

    traces: {(submission_file, task_id, trace_line): [cells in order]}.
    A cell that reads a name bound only in an earlier cell of the trace fails
    with NameError if the harness runs every cell in a fresh process. The
    submission is stateless if more than `limit` of these cells fail with
    NameError (at least `min_cells` cells). Returns {file: (stateful, k, n)}.
    """
    import builtins
    skip = set(dir(builtins)) | {"final_answer", "display", "read_csv_metadata_file",
                                 "read_json_metadata_file", "safe_read_file"}
    cnt = {}
    for (fn, _, _), cells in traces.items():
        k, n = cnt.get(fn, (0, 0))
        before = set()
        for r in sorted(cells, key=lambda r: r["cell_index"]):
            bound, read = _names(r["code"] or "")
            if bound is None:
                continue
            if (read - bound - skip) & before:
                n += 1
                k += bool(r["failed"] and r.get("error_type") == "NameError")
            before |= bound
        cnt[fn] = (k, n)
    return {fn: (not (n >= min_cells and k / n > limit), k, n) for fn, (k, n) in cnt.items()}


KEY_LISTS = [re.compile(p, re.S) for p in (
    r"None of \[Index\(\[(.*?)\]", r"None of \[(.*?)\] are in the", r"\[([^\[\]]*?)\] not in index",
    r"KeyError:?\s*Index\(\[(.*?)\]", r"KeyError:?\s*\[([^\[\]]*?)\]", r"Label\(s\) \[(.*?)\] do not exist")]


def error_keys(r):
    """Missing names of a failed cell: the recorded error key, or every quoted name in a
    pandas message that lists keys ("None of [Index(['a'])] are in the [columns]",
    "['a'] not in index", "KeyError: Index(['a'], dtype='object')")."""
    if r.get("error_key"):
        return {r["error_key"]}
    if not str(r.get("error_type") or "").startswith(("KeyError", "index-error")):
        return set()
    out = set()
    for rx in KEY_LISTS:
        for m in rx.finditer(r.get("output") or ""):
            out |= set(re.findall(r"'([^'\n]{1,80})'", re.sub(r"dtype='[^']*'", "", m.group(1))))
    return out


FNF_PATH = re.compile(r"No such file or directory: '([^']+)'|File (\S+) does not exist|"
                      r"\[Errno 2\][^'\n]*'([^']+)'")


def fnf_path(output):
    """The path in a FileNotFoundError message, or None."""
    m = FNF_PATH.search(output or "")
    return next((g for g in m.groups() if g), None) if m else None


def same_path(a, b):
    a, b = os.path.normpath(a), os.path.normpath(b)
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


class FileEvidence:
    """Replay stand-in for os.path.exists (gate version 4, --files).

    A recorded trace does not list its workspace. For the cell being
    decided, a path exists if an earlier cell of the same trace wrote it, or
    if another cell of the same submission read it in code that always runs
    and did not fail. A path is missing if another cell of the submission
    failed with FileNotFoundError on it. Otherwise it is unknown. The cell's
    own outcome is never used. In a live run, pass os.path.exists instead.
    """

    def __init__(self):
        self.ok = collections.defaultdict(lambda: collections.defaultdict(set))
        self.missing = collections.defaultdict(lambda: collections.defaultdict(set))
        self.writes = collections.defaultdict(list)

    def exists_for(self, fn, tkey, cell_index):
        me = (tkey, cell_index)

        def exists(path):
            p = os.path.normpath(path)
            if any(ci < cell_index and same_path(w, p) for ci, w in self.writes[tkey]):
                return True
            if self.ok[fn].get(p, set()) - {me}:
                return True
            if self.missing[fn].get(p, set()) - {me}:
                return False
            return None
        return exists


def file_evidence(replay):
    """First pass over all cells: which files each cell reads and writes."""
    ev = FileEvidence()
    for tkey, cells in replay.traces.items():
        fn = tkey[0]
        for r, res, _, _ in replay.run(tkey, cells, with_files=False):
            me = (tkey, r["cell_index"])
            for p in res.get("written", []):
                ev.writes[tkey].append((r["cell_index"], os.path.normpath(p)))
            if not r["failed"]:
                # gate v8: a directory that a cell listed without error exists, as a file read does
                for p, cond, _ in res.get("files", []) + res.get("dirs", []):
                    if not cond:
                        ev.ok[fn][os.path.normpath(p)].add(me)
            elif r.get("error_type") == "FileNotFoundError":
                m = fnf_path(r.get("output"))
                if m:
                    ev.missing[fn][os.path.normpath(m)].add(me)
                    for p, _, _ in res.get("files", []):
                        if same_path(p, m):
                            ev.missing[fn][os.path.normpath(p)].add(me)
    return ev


def tolerant_submissions(traces, man):
    """Submissions whose executor may replace a missing dict key silently.

    smolagents 1.0-1.4 return the closest key for a missing dict key; from
    1.5 the executor raises and adds "Maybe you meant one of these indexes".
    A smolagents submission counts as tolerant unless its errors show that
    message (the version is not recorded)."""
    newer = {fn for (fn, _, _), cells in traces.items()
             if any("Maybe you meant one of these indexes" in (r.get("output") or "") for r in cells)}
    return {fn for (fn, _, _) in traces
            if man.get(fn, {}).get("trace_format") == "smolagents" and fn not in newer}


class Replay:
    """Run the gate over recorded traces, with the options of evaluate_gate.py.

    carry:    carry tracked objects across cells (not for stateless harnesses)
    files:    check file existence with FileEvidence
    tolerant: "auto" (smolagents submissions, see tolerant_submissions),
              "all", or "none"
    """

    def __init__(self, gate_dir, data, traces, man, carry=False, files=False, tolerant="none"):
        self.decide = load_decide(gate_dir)
        self.new_state = load_state(gate_dir) if carry else None
        self.schema = context_schema(data)
        self.traces = traces
        self.stateful = stateful_submissions(traces) if carry else {}
        subs = {fn for (fn, _, _) in traces}
        self.tolerant = tolerant_submissions(traces, man) if tolerant == "auto" else \
            subs if tolerant == "all" else set()
        self.ev = None
        if files:
            self.ev = file_evidence(self)

    def run(self, tkey, cells, with_files=True):
        """Yield (cell, result, outcome, milliseconds) for the cells of one trace, in order."""
        import time
        fn = tkey[0]
        st = self.new_state() if self.new_state and self.stateful[fn][0] else None
        kw = {}
        if fn in self.tolerant:
            kw["tolerant_keys"] = True
        for r in sorted(cells, key=lambda r: r["cell_index"]):
            if with_files and self.ev is not None:
                kw["exists"] = self.ev.exists_for(fn, tkey, r["cell_index"])
            t0 = time.perf_counter()
            try:
                res = self.decide(r["code"] or "", self.schema, st, **kw) if (st is not None or kw) \
                    else self.decide(r["code"] or "", self.schema)
                o = outcome(res)
            except Exception as e:  # a crash in the gate is a gate defect, not a block
                res, o = {"outcome": "gate_error", "reason": f"{type(e).__name__}: {e}"[:500]}, "gate_error"
            dt = (time.perf_counter() - t0) * 1000
            if st is not None:
                # the recorded cell ran; a block means it would have failed
                st.commit(failed=r["failed"] or o == "block")
            yield r, res, o, dt


def outcome(res):
    if isinstance(res, str):
        return res.lower()
    if isinstance(res, dict):
        for k in ("outcome", "decision", "result", "status"):
            if k in res:
                return str(res[k]).lower()
    if isinstance(res, (tuple, list)) and res and isinstance(res[0], str):
        return res[0].lower()
    for k in ("outcome", "decision", "result"):
        if hasattr(res, k):
            return str(getattr(res, k)).lower()
    return str(res).lower()


def reason(res):
    if isinstance(res, dict):
        if "reason" in res:
            loc = res.get("location")
            extra = f" (found in: {', '.join(res['found_in'])})" if res.get(
                "found_in") else ""
            return (f"{loc}: " if loc else "") + str(res["reason"]) + extra
        return json.dumps({k: v for k, v in res.items() if k != "outcome"}, default=str)[:500]
    if isinstance(res, (tuple, list)) and len(res) > 1:
        return str(res[1:])[:500]
    return str(res)[:500]


def safe_decide(decide, code, schema, state=None, exists=None):
    try:
        if exists is not None:
            res = decide(code, schema, state, exists)
        else:
            res = decide(code, schema, state) if state is not None else decide(code, schema)
        return outcome(res), reason(res)
    except Exception as e:  # a crash in the gate is a gate defect, not a block
        return "gate_error", f"{type(e).__name__}: {e}"[:500]
