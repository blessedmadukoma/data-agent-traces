"""Build manifest.jsonl for the DABstep trace corpus.

For every submission file it records: size, SHA-256, Git-LFS pointer
status, rows, rows with a trace, trace format, submitter, a trace hash
for duplicate detection, and the inclusion decision with a reason.

Usage:
  python3 build_manifest.py --data $DATA --out $RUN/manifest.jsonl \
      --scripts-commit "$(git rev-parse HEAD)"

It stops with an error if any Git-LFS pointer file is found.
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess

REV = "c8fb51b3b898e1755f3b5a00c8dc5f25a52ff678"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def detect(t):
    if "<Code>" in t and "<Execute>" in t:
        return "xml"
    if "python_interpreter" in t and ("arguments" in t):
        return "smolagents"
    if "Traceback (most recent call last)" in t or "Code execution failed" in t:
        return "other_with_errors"
    if "```py" in t or "<code>" in t or "Code:" in t:
        return "code_no_standard_output"
    return "summary"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scripts-commit", default="unknown")
    a = ap.parse_args()
    sd = os.path.join(a.data, "data/submissions")
    pointers = []
    rows = []
    for fn in sorted(os.listdir(sd)):
        p = os.path.join(sd, fn)
        size = os.path.getsize(p)
        if size < 200 and open(p, "rb").read(60).startswith(b"version https://git-lfs"):
            pointers.append(fn)
            continue
        n = ntr = err = 0
        fmts = collections.Counter()
        th = []
        first = None
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            first = first or r
            n += 1
            t = r.get("reasoning_trace")
            th.append((str(r.get("task_id")), str(t)))
            if not t or (isinstance(t, str) and not t.strip()):
                continue
            t = t if isinstance(t, str) else json.dumps(t)
            ntr += 1
            fmts[detect(t)] += 1
            err += ("Traceback (most recent call last)" in t) or (
                "Code execution failed" in t) or ("[Error]" in t)
        h = hashlib.sha256()
        for tid, t in sorted(th):
            h.update(tid.encode())
            h.update(t.encode())
        org = (first or {}).get("organisation") or ""
        fmt = fmts.most_common(1)[0][0] if fmts else "none"
        rows.append(dict(
            submission_file=fn, submission_id=(
                first or {}).get("submission_id"),
            submitter=org.split("user")[-1].strip() if "user" in org else org,
            organisation=org, agent_name=(first or {}).get("agent_name"), model_family=(first or {}).get("model_family"),
            trace_format=fmt, rows=n, rows_with_trace=ntr, traces_with_error_text=err,
            dataset_commit=REV, file_sha256=sha256(p), trace_sha256=h.hexdigest(), size_bytes=size,
            scripts_commit=a.scripts_commit))
    if pointers:
        raise SystemExit(
            f"{len(pointers)} Git-LFS pointer files found, e.g. {pointers[:3]}. Run `git lfs pull` first.")
    seen = {}
    for r in rows:
        dup = seen.get(r["trace_sha256"]) if r["rows_with_trace"] else None
        seen.setdefault(r["trace_sha256"], r["submission_file"])
        ntr = r["rows_with_trace"]
        if ntr == 0:
            inc, why = False, "no_trace"
        elif dup:
            inc, why = False, f"duplicate_of:{dup}"
        elif r["trace_format"] in ("summary", "none"):
            inc, why = False, "summary_only"
        elif r["trace_format"] == "code_no_standard_output":
            inc, why = False, "parser_not_written"
        elif r["traces_with_error_text"] == 0:
            inc, why = True, "no_error_signal (controls only)"
        else:
            inc, why = True, None
        r["included"], r["exclusion_reason"] = inc, why
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as w:
        for r in rows:
            w.write(json.dumps(r) + "\n")
    c = collections.Counter(
        (r["included"], r["exclusion_reason"] or "included", r["trace_format"]) for r in rows)
    print(
        f"files: {len(rows)}; with trace: {sum(r['rows_with_trace'] > 0 for r in rows)}")
    for (inc, why, fmt), v in sorted(c.items(), key=lambda x: -x[1]):
        if why not in ("no_trace",):
            print(f"  {v:5d}  included={inc!s:5}  {fmt:24s} {why}")
    inc_err = [r for r in rows if r["included"]
               and r["exclusion_reason"] is None]
    print(f"included with error text: {len(inc_err)} submissions, "
          f"{len({r['submitter'] for r in inc_err})} submitters")


if __name__ == "__main__":
    main()
