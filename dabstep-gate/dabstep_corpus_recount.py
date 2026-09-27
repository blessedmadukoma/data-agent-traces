#!/usr/bin/env python3
"""Independent recount of the DABstep trace corpus at a pinned commit.

This script does not reproduce the main parsers. It is a second,
heuristic measurement that you can compare against them.

Steps:
  1. Download the dataset at a pinned revision (submissions, task_scores,
     context, tasks). About 5.6 GB. Git-LFS objects are resolved.
  2. Census every submission file: rows, rows with a trace, trace format.
  3. Select code-bearing submissions and remove byte-identical duplicates.
  4. Extract distinct error events per trace (smolagents "Code execution
     failed ... due to:" messages and Python tracebacks).
  5. Classify events: executor, data-reference, other.
  6. Classify missing keys against the context-file schemas.
  7. Estimate the association between a data-reference error in a trace
     and a correct final answer, using public task_scores, with task and
     submission fixed effects, clustered by submission and by submitter.
  8. XML traces: cell counts, recovery after KeyError, and reads of files
     that are neither context files nor written earlier in the trace.
  9. Median agent step duration in one smolagents submission.

Requirements: python>=3.10, huggingface_hub, pandas, statsmodels.
Usage: python3 dabstep_corpus_recount.py [--rev <commit>] [--dir dab]
"""
import argparse
import collections
import csv
import difflib
import hashlib
import json
import math
import os
import re
import warnings
warnings.filterwarnings("ignore")

DEFAULT_REV = "c8fb51b3b898e1755f3b5a00c8dc5f25a52ff678"

DATA = {"KeyError", "KeyError(wrapped)", "index-error(wrapped)",
        "FileNotFoundError", "ValueError:usecols"}
EXE = {"executor:forbidden", "executor:budget", "ModuleNotFoundError"}
ANSI = re.compile(r"\x1b\[[0-9;]*m|\\u001b\[[0-9;]*m")
TB = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_.]*(?:Error|Exception)): ?([^\n]{0,300})$", re.M)


def download(rev, d):
    from huggingface_hub import snapshot_download
    snapshot_download("adyen/DABstep", repo_type="dataset", revision=rev, local_dir=d,
                      allow_patterns=["data/submissions/*", "data/task_scores/*",
                                      "data/context/*", "data/tasks/*"], max_workers=16)


def schemas(d):
    c = os.path.join(d, "data/context/")
    s = {f: next(csv.reader(open(c + f))) for f in
         ("payments.csv", "acquirer_countries.csv", "merchant_category_codes.csv")}
    for f in ("merchant_data.json", "fees.json"):
        keys = set()
        for m in json.load(open(c + f)):
            keys |= set(m)
        s[f] = sorted(keys)
    return s


def census(d):
    out = []
    sd = os.path.join(d, "data/submissions")
    for fn in sorted(os.listdir(sd)):
        n = ntr = code = exe = 0
        with open(os.path.join(sd, fn)) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                t = r.get("reasoning_trace")
                if not t or (isinstance(t, str) and not t.strip()):
                    continue
                t = t if isinstance(t, str) else json.dumps(t)
                ntr += 1
                if "<Code>" in t or "```py" in t or "Code:" in t or "<code>" in t:
                    code += 1
                if any(k in t for k in ("<Execute>", "Execution logs:", "Observation:", "Traceback")):
                    exe += 1
        out.append(dict(file=fn, rows=n, with_trace=ntr, code=code, exe=exe))
    return out


def trace_hash(path):
    h = hashlib.sha256()
    recs = [json.loads(l) for l in open(path) if l.strip()]
    for x in sorted(recs, key=lambda x: str(x["task_id"])):
        h.update(str(x["task_id"]).encode())
        h.update(str(x.get("reasoning_trace")).encode())
    return h.hexdigest()


def norm(t):
    t = ANSI.sub("", t)
    for a, b in (("\\\\n", "\n"), ("\\n", "\n"), ("\\'", "'"), ('\\"', '"'), ("\\\\", "\\")):
        t = t.replace(a, b)
    return t


def classify(win):
    w, head = win[:3000], win[:200]
    if re.match(r"\s*InterpreterError", head) or not re.match(r"\s*[A-Za-z]+(Error|Exception)", head):
        if re.search(r"not permitted to evaluate other functions|Forbidden function|is not among the explicitly allowed|"
                     r"Import of .{1,60} is not allowed|Forbidden access|Cannot assign to name .{1,40} erase the existing tool", w[:600]):
            return "executor:forbidden", None
        if re.search(r"max number of operations|Reached the max|operations limit|timed? ?out", w[:600], re.I):
            return "executor:budget", None
        m = re.search(
            r"Could not index .{0,2500}? with '([^'\n]{1,80})'", w, re.S)
        if m and "Could not index" in head + w[:40]:
            return "index-error(wrapped)", m.group(1)
        if "Could not index" in head + w[:40]:
            # the message repeats the whole DataFrame, which can be longer than the window
            m = re.search(r" with '([^'\n]{1,80})': KeyError: '", win, re.S)
            if m:
                return "index-error(wrapped)", m.group(1)
        m = re.search(r"KeyError: '([^'\n]{1,80})'", w[:800])
        if m:
            return "KeyError(wrapped)", m.group(1)
        for pat, c in ((r"is not defined|The variable `", "NameError"),
                       (r"SyntaxError|invalid syntax|Code parsing failed|unterminated|IndentationError", "SyntaxError"),
                       (r"has no attribute", "AttributeError"),
                       (r"TypeError|unsupported operand|not callable|not subscriptable|not iterable", "TypeError"),
                       (r"ValueError|could not convert|invalid literal", "ValueError"),
                       (r"IndexError|out of bounds|index out of range", "IndexError"),
                       (r"FileNotFound|No such file", "FileNotFoundError")):
            if re.search(pat, w[:1500]):
                return c, None
        return "other", None
    et = re.match(r"\s*([A-Za-z]+(?:Error|Exception))", head).group(1)
    if et == "IndentationError":
        et = "SyntaxError"
    if et == "ValueError" and "Usecols do not match" in w[:300]:
        return "ValueError:usecols", None
    k = None
    if et == "KeyError":
        m = re.match(r"\s*KeyError:?\s*'([^'\n]{1,80})'", head)
        k = m.group(1) if m else None
    return et, k


def keyclass(k, sch):
    pay = set(sch["payments.csv"])
    where = collections.defaultdict(list)
    for f, cs in sch.items():
        for c in cs:
            if c:
                where[c].append(f)
    if k is None:
        return "non-string/unknown"
    if k in pay:
        return "exists-in-payments.csv"
    if k in where:
        return "exists-only-in:" + "/".join(where[k])
    if k.lower() in {c.lower() for c in where}:
        return "case-mismatch"
    if difflib.get_close_matches(k, list(where), n=1, cutoff=0.8):
        return "near-miss"
    return "absent-from-all-sources"


def events_for(t):
    t = norm(t)
    ev = {}
    for m in re.finditer(r"Code execution failed", t):
        cand = [x for x in (t.find("due to", m.start(), m.start() + 20000),
                            t.find("because of the following error", m.start(), m.start() + 20000)) if x >= 0]
        if not cand:
            continue
        s = t.find(":", min(cand)) + 1
        c, k = classify(t[s:s + 3000])
        ev[(c, k, t[s:s + 120])] = (c, k)
    if "Traceback (most recent call last)" in t or "[Error]" in t:
        for m in TB.finditer(t):
            c, k = classify(m.group(1) + ": " + m.group(2))
            ev[(c, k, m.group(0)[:120])] = (c, k)
    return list(ev.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default=DEFAULT_REV)
    ap.add_argument("--dir", default="dab")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    if not a.skip_download:
        download(a.rev, a.dir)
    sch = schemas(a.dir)
    cen = census(a.dir)
    print(
        f"submission files: {len(cen)}; with any trace: {sum(c['with_trace'] > 0 for c in cen)}")
    cb = [c for c in cen if c["with_trace"] and c["code"] /
          c["with_trace"] > 0.3 and c["exe"] / c["with_trace"] > 0.3]
    sd = os.path.join(a.dir, "data/submissions/")
    seen, files = {}, []
    for c in cb:
        h = trace_hash(sd + c["file"])
        if h in seen:
            print(f"  duplicate traces: {c['file']} == {seen[h]}")
            continue
        seen[h] = c["file"]
        files.append(c["file"])
    print(f"code-bearing files: {len(cb)}; distinct: {len(files)}")

    tasks = {str(json.loads(l)["task_id"]): json.loads(l)["level"]
             for l in open(os.path.join(a.dir, "data/tasks/all.jsonl"))}
    rows, famc, kc = [], collections.defaultdict(
        collections.Counter), collections.Counter()
    for fn in files:
        sp = os.path.join(a.dir, "data/task_scores/", fn)
        sc = {str(json.loads(l)["task_id"]): json.loads(l)["score"]
              for l in open(sp) if l.strip()} if os.path.exists(sp) else {}
        for line in open(sd + fn):
            if not line.strip():
                continue
            r = json.loads(line)
            t = r.get("reasoning_trace") or ""
            ev = events_for(t if isinstance(t, str) else json.dumps(t))
            for c, k in ev:
                famc[fn]["exe" if c in EXE else "data" if c in DATA else "other"] += 1
                if c in ("KeyError", "KeyError(wrapped)", "index-error(wrapped)"):
                    kc[keyclass(k, sch)] += 1
            rows.append(dict(sub=fn, task=str(r["task_id"]), level=tasks.get(str(r["task_id"])),
                             y=sc.get(str(r["task_id"])), any_err=bool(ev), d=any(c in DATA for c, _ in ev)))
    print("\nshare of distinct error events per submission (executor / data-reference / other):")
    for fn, c in famc.items():
        n = sum(c.values())
        print(
            f"  {fn[4:60]:56s} {c['exe']/n:4.0%} {c['data']/n:4.0%} {c['other']/n:4.0%}  n={n}")
    n = sum(kc.values())
    print(f"\nmissing-key classes (n={n}):")
    for k, v in kc.most_common():
        print(f"  {k}: {v} ({v/n:.1%})")

    xml_checks(a.dir, files)
    step_durations(a.dir, "v1__msr-basic5__04-09-2025.jsonl")

    import pandas as pd
    import statsmodels.api as sm
    submitter = {}
    for fn in famc:
        org = json.loads(open(sd + fn).readline()).get("organisation") or ""
        submitter[fn] = org.split("user")[-1].strip() if "user" in org else fn
    for r in rows:
        r["submitter"] = submitter.get(r["sub"], r["sub"])
    df = pd.DataFrame(
        [r for r in rows if r["y"] is not None and r["sub"] in famc])
    df["y"] = df["y"].astype(bool).astype(int)
    df["d"] = df["d"].astype(int)
    df = df[df.groupby("task")["y"].transform(
        "nunique") > 1].reset_index(drop=True)
    print(
        f"\nsubmissions: {df['sub'].nunique()}; submitters: {df['submitter'].nunique()}")
    for fe, cl in ((["task"], "sub"), (["sub"], "sub"), (["sub", "task"], "sub"), (["sub", "task"], "submitter")):
        X = pd.get_dummies(df[fe].astype(str), drop_first=True).astype(float)
        X.insert(0, "d", df["d"])
        X = sm.add_constant(X)
        m = sm.Logit(df["y"], X).fit(disp=0, method="newton", maxiter=100,
                                     cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cl])[0]})
        b, se = m.params["d"], m.bse["d"]
        print(f"OR(correct | data-reference error), FE={fe}, clustered by {cl}: {math.exp(b):.2f} "
              f"[{math.exp(b - 1.96 * se):.2f}, {math.exp(b + 1.96 * se):.2f}] (n={len(df)})")


CELL = re.compile(r"<Code>(.*?)</Code>\s*<Execute>(.*?)</Execute>", re.S)
CONTEXT = {"payments.csv", "acquirer_countries.csv", "merchant_category_codes.csv", "fees.json",
           "merchant_data.json", "manual.md", "payments-readme.md"}


def xml_checks(d, files):
    """XML (DeepAnalyze-style) traces: cell counts, recovery after KeyError, reads outside the context."""
    sd = os.path.join(d, "data/submissions/")
    head = {f: open(sd + f).read(200000) for f in files}
    xml = [f for f in files if "<Code>" in head[f] and "<Execute>" in head[f]]
    cells = errs = kerr = next_ok = no_later_ok = 0
    print("\nfiles with XML <Code>/<Execute> markers:", len(xml))
    for fn in xml:
        reads = ran = traces = 0
        for line in open(sd + fn):
            t = json.loads(line).get("reasoning_trace") or ""
            cs = CELL.findall(t)
            st = [("Traceback (most recent call last)" in o or o.strip(
            ).startswith("[Error]"), c, o) for c, o in cs]
            cells += len(st)
            errs += sum(e for e, _, _ in st)
            for i, (e, _, o) in enumerate(st):
                if e and re.search(r"^KeyError", o, re.M):
                    kerr += 1
                    next_ok += i + 1 < len(st) and not st[i + 1][0]
                    no_later_ok += not any(not x[0] for x in st[i + 1:])
            written, hit = set(), False
            for e, c, o in st:
                for m in re.findall(r"to_csv\(\s*['\"]([^'\"]+)['\"]", c):
                    written.add(m.split("/")[-1])
                for m in re.findall(r"(?:read_csv|read_json|open)\(\s*['\"]([^'\"]+)['\"]", c):
                    b = m.split("/")[-1]
                    if b in CONTEXT or b in written:
                        continue
                    reads += 1
                    hit = True
                    ran += not e
            traces += hit
        print(f"  {fn[4:60]:56s} reads of non-context files not written earlier: {reads} "
              f"(ran without error: {ran}); traces: {traces}")
    print(f"  cells {cells}; error cells {errs}; KeyError cells {kerr}; next cell ran without error {next_ok}; "
          f"no later successful cell {no_later_ok}")


def step_durations(d, fn):
    p = os.path.join(d, "data/submissions/", fn)
    if not os.path.exists(p):
        return
    ds = []
    for line in open(p):
        t = json.loads(line).get("reasoning_trace") or ""
        ds += [float(x) for x in re.findall(
            r"Timing\(start_time=[0-9.]+, end_time=[0-9.]+, duration=([0-9.]+)\)", t)]
    if ds:
        ds.sort()
        print(
            f"\nstep duration in {fn}: median {ds[len(ds)//2]:.1f} s, p90 {ds[int(.9*len(ds))]:.1f} s, n={len(ds)}")


if __name__ == "__main__":
    main()
