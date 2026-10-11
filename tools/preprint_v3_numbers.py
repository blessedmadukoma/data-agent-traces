#!/usr/bin/env python3
"""Numbers added to the preprint in v3.6-v3.10 (11 October 2026), from the recorded runs.

Prints:
  1. Live experiment (Table 6 and Section 3.4): correct answers per arm over 822 runs; the paired
     differences over question-seed pairs (821 for the column-name arms, because one run did not
     finish); a bootstrap that resamples questions (10,000 samples, seed 20261011); per-seed
     differences with Newcombe intervals.
  2. Unrecovered lost answers per arm (runs whose model defined or assigned final_answer and that
     still have no answer after recovery), for the worst-case bound in Section 3.4.
  3. Receipt replay (Section 3.4): the primary analysis, and the sensitivity analysis that keeps
     the pairs in which an arm stopped early (that arm at 7 steps unless it reached the outcome).
  4. Literal answers in the receipt replay: finished arms whose first literal-answer step is
     final_answer(<literal>), and how many of these are 'Not Applicable'.
  5. Causes of the arms that stopped early (TimeoutError or ExecutorLost).
  6. Receipt replay restricted to pairs whose rebuilt earlier steps all reproduced their recorded
     outcome, and re-scored with a rule that also counts file and directory listings as inspection.
  7. Live experiment: executed data-reference errors and blocks by kind (missing file or key), in the
     plain-prompt arms up to the answer, and all blocked steps as recorded.
  8. Recovered answers that the recorded step which gave them already shows (a literal in its code, or a
     line of its printed output that is the answer alone or after ':' or '=').
  9. With --audit: prevalence and recall intervals with Jeffreys' prior per stratum, which keep
     uncertainty for a stratum with no positive sample (400,000 draws, seed 20261011).
  10. Blocked steps: whether the lines before the missing read, which do not run when the step is
     blocked, define names that later steps need (NameError or a stale value).

Usage (from the repository root, after `tar xzf recorded-runs-2026-09.tar.gz` and
`uv run live/fa_rescore.py --rec recorded --replays results/fa-replay --out runs/fa_recorded`):

  uv run tools/preprint_v3_numbers.py --qrdata data/QRData/benchmark/QRData.json \
      --recorded recorded --recovered runs/fa_recorded
"""
import argparse
import ast
import collections
import json
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("live", "rq3-pilot", "dabstep-gate"):
    sys.path.insert(0, os.path.join(ROOT, d))
import e1_report as E            # noqa: E402
import fa_rescore as F           # noqa: E402
import rq3_full_report as RQ     # noqa: E402
from recovery_episodes import gave_up  # noqa: E402


def load_live(path, gold):
    by = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by[r["task"]].setdefault(r["arm"], {})[r["seed"]] = E.per_arm(r, gold)
    return by


def live(label, path, gold):
    by = load_live(path, gold)
    q = sorted(by)
    print(f"== live experiment, {label}")
    for arm in ("P0G0", "P0G1", "P1G0", "P1G1"):
        xs = [v for t in q for v in by[t].get(arm, {}).values()]
        print(f"  {arm}: runs {len(xs)}, correct {sum(v['correct'] for v in xs)}, "
              f"not finished {sum(not v['done'] for v in xs)}")
    for x, y in (("P0G0", "P0G1"), ("P1G0", "P1G1"), ("P0G0", "P1G0")):
        out = {}
        E.compare(by, q, x, y, f"  {x} -> {y}", out)
        per_q = []
        for t in q:
            if x in by[t] and y in by[t]:
                seeds = [s for s in by[t][x] if s in by[t][y] and by[t][x][s]["done"] and by[t][y][s]["done"]]
                if seeds:
                    per_q.append([by[t][y][s]["correct"] - by[t][x][s]["correct"] for s in seeds])
        rng, n, vals = random.Random(20261011), 10000, []
        for _ in range(n):
            smp = [per_q[rng.randrange(len(per_q))] for _ in per_q]
            flat = [d for ds in smp for d in ds]
            vals.append(sum(flat) / len(flat))
        vals.sort()
        est = sum(d for ds in per_q for d in ds) / sum(len(ds) for ds in per_q)
        print(f"    question bootstrap: {est:+.4f} [{vals[int(.025 * n)]:+.4f}, {vals[int(.975 * n) - 1]:+.4f}]")
        for s in sorted({s for t in q for a in by[t].values() for s in a}):
            pairs = [(by[t][x][s], by[t][y][s]) for t in q if x in by[t] and y in by[t]
                     and s in by[t][x] and s in by[t][y]]
            a = sum(1 for u, v in pairs if u["correct"] and v["correct"])
            b = sum(1 for u, v in pairs if v["correct"] and not u["correct"])
            c = sum(1 for u, v in pairs if u["correct"] and not v["correct"])
            d, lo, hi = E.newcombe_paired(a, b, c, len(pairs) - a - b - c)
            print(f"    seed {s}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}], McNemar p {E.sign_test(b, c):.3f}")


def unrecovered(recorded, recovered):
    rec = [json.loads(line) for line in open(recorded)]
    new = [json.loads(line) for line in open(recovered)]
    c = collections.Counter()
    for r, n in zip(rec, new):
        cells = [x for x in r["cells"] if "code" in x and not x.get("no_code") and not x.get("blocked")]
        if any(F.defines_final_answer(x["code"]) for x in cells):
            c[(r["arm"], "defined")] += 1
            if r.get("final_answer") is not None:
                c[(r["arm"], "answer recorded anyway")] += 1
            elif n.get("final_answer") is None:
                c[(r["arm"], "still no answer")] += 1
    print("== lost answers not recovered, by arm")
    for k in sorted(c):
        print(f"  {k[0]} {k[1]}: {c[k]}")


def replay(results):
    rows = [json.loads(line) for line in open(results)]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["episode"], r["seed"])][r["arm"]] = r

    def y(r):
        v = r.get("steps_to_fix")
        return v if v is not None else 7

    print("== receipt replay")
    for mode in ("primary", "stopped arms kept at 7"):
        acc, C = collections.defaultdict(list), collections.Counter()
        for k, a in by.items():
            t, rc = a.get("traceback"), a.get("receipt")
            stopped = t.get("status") != "done" or rc.get("status") != "done"
            if stopped and mode == "primary":
                C["stopped"] += 1
                continue
            if not t.get("blocked_names_missing"):
                C["does not reproduce"] += 1
                continue
            if stopped:
                C["stopped, kept"] += 1
            acc[k[0]].append(y(t) - y(rc))
        print(f"  {mode}: {dict(C)}; pairs {sum(len(v) for v in acc.values())}")
        RQ.summarise(f"  {mode}", {e: statistics.mean(v) for e, v in acc.items()})
    causes = collections.Counter((r.get("steps") or [{}])[-1].get("error_type")
                                 for r in rows if r.get("status") != "done")
    print(f"  arms that stopped early, by last error: {dict(causes)}")
    lit, na = 0, 0
    for r in rows:
        if r.get("status") != "done":
            continue
        for st in r.get("steps") or []:
            if gave_up(st.get("code") or ""):
                lit += 1
                for n in ast.walk(ast.parse(st["code"])):
                    if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "final_answer":
                        na += str(n.args[0].value) == "Not Applicable"
                break
    print(f"  finished arms with a literal answer: {lit}; of these 'Not Applicable': {na}")


def replay_sensitivity(results):
    import recovery_episodes as RE
    rows = [json.loads(line) for line in open(results)]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["episode"], r["seed"])][r["arm"]] = r
    extra = {"walk", "iterdir", "scandir", "rglob", "exists", "isfile", "isdir", "getsize", "stat", "ls"}

    def y(r):
        return next((s["step"] for s in r.get("steps") or [] if s.get("ok") and not RE.is_inspection(s.get("code") or "")), 7)

    print("== receipt replay sensitivity")
    for mode in ("every earlier step reproduced", "file listings count as inspection"):
        if mode.startswith("file"):
            RE.LOOK_CALLS |= extra
            RE.PLAIN |= extra
        acc = collections.defaultdict(list)
        for k, a in by.items():
            t, rc = a.get("traceback"), a.get("receipt")
            if t.get("status") != "done" or rc.get("status") != "done" or not t.get("blocked_names_missing"):
                continue
            if mode.startswith("every") and (t.get("prefix_mismatch", 0) or rc.get("prefix_mismatch", 0)):
                continue
            acc[k[0]].append(y(t) - y(rc))
        print(f"  {mode}: pairs {sum(len(v) for v in acc.values())}")
        RQ.summarise(f"  {mode}", {e: statistics.mean(v) for e, v in acc.items()})


def live_kinds(recorded, recovered):
    C, B = collections.Counter(), collections.Counter()
    for line in open(recovered):
        r = json.loads(line)
        for c in r["cells"]:
            if r["arm"] == "P0G0" and c.get("ok") is False and c.get("error_type") in ("KeyError", "FileNotFoundError"):
                C[c["error_type"]] += 1
            if r["arm"] == "P0G1" and c.get("blocked") and isinstance(c.get("receipt"), dict):
                B[c["receipt"].get("kind")] += 1
    print(f"== live experiment, plain prompt (recovered): executed errors {dict(C)}; blocks by kind {dict(B)}")
    A = collections.Counter()
    for line in open(recorded):
        r = json.loads(line)
        for c in r["cells"]:
            if c.get("blocked"):
                A[c["receipt"].get("kind") if isinstance(c.get("receipt"), dict) else None] += 1
    print(f"  all blocked steps, both arms with the checker (recorded): {sum(A.values())}; by kind {dict(A)}")


def skipped_lines(recorded):
    """Does a blocked step lose the effect of the lines before its missing read?

    Without the checker, those lines run before the error; with it, the step does not run at all.
    Counts, over all blocked steps (as recorded): later steps that fail with NameError on a name
    that those lines define (before a working step defines it again), and later working steps that
    read a name that those lines would have reassigned, before redefining it.
    """
    import re
    undef = re.compile(r"name '([^']+)' is not defined")

    def bound(code, before=10 ** 9):
        try:
            tree = ast.parse(code or "")
        except (SyntaxError, ValueError):
            return set()
        out = set()
        for st in tree.body:
            if st.lineno >= before:
                break
            for n in ast.walk(st):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
                elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    out.add(n.name)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        return out

    def reads(code):
        try:
            tree = ast.parse(code or "")
        except (SyntaxError, ValueError):
            return set()
        return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    C = collections.Counter()
    for line in open(recorded):
        r = json.loads(line)
        cells = r["cells"]
        for i, c in enumerate(cells):
            m = re.search(r"line (\d+)", (c.get("receipt") or {}).get("location") or "") \
                if c.get("blocked") and isinstance(c.get("receipt"), dict) else None
            if not m:
                continue
            skipped = bound(c["code"], int(m.group(1)))
            existing = set().union(*[bound(d["code"]) for d in cells[:i] if d.get("ok") is True])
            missing, stale = set(skipped), skipped & existing
            C["blocked"] += 1
            C["skipped lines reassign an existing name"] += bool(stale)
            name_error = stale_read = False
            for d in cells[i + 1:]:
                if d.get("blocked"):
                    continue
                if d.get("ok") is False and d.get("error_type") == "NameError":
                    hit = undef.search(d.get("error") or "")
                    name_error = name_error or bool(hit and hit.group(1) in missing)
                if d.get("ok") is True:
                    defined = bound(d["code"])
                    stale_read = stale_read or bool(reads(d["code"]) & (stale - defined))
                    missing -= defined
                    stale -= defined
            C["later NameError on a skipped name"] += name_error
            C["later working step reads an old value"] += stale_read
    print(f"== blocked steps and the lines before the missing read (recorded): {dict(C)}")


def answer_evidence(recorded, replays, arms):
    """Recovered answers that the recorded step which gave them already shows.

    A plain substring test is too weak, because many QRData answers are one letter ('a' to 'd').
    An answer counts as shown when it is a literal in the step's code, or when a line of the step's
    printed output is the answer alone or ends with ': <answer>' or '= <answer>'.
    """
    import re
    norm = lambda x: re.sub(r"\s+", " ", str(x)).strip().lower()  # noqa: E731

    def literals(code):
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return set()
        return {norm(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and isinstance(n.value, (str, int, float)) and not isinstance(n.value, bool)}

    def printed(a, stdout):
        for ln in str(stdout).splitlines():
            line = norm(ln)
            if line and (line == a or line.strip("'\"()[] ") == a
                         or re.search(r"[:=]\s*" + re.escape(a) + r"[\s.]*$", line)):
                return True
        return False

    recs = [json.loads(line) for line in open(recorded)]
    C = collections.Counter()
    for line in open(replays):
        o = json.loads(line)
        r = recs[int(o["key"].split(":")[1])]
        if r.get("arm") not in arms or o.get("answer") is None or r.get("final_answer") is not None:
            continue
        stop = o["answer_cell"]
        if any(not (bool(r["cells"][c["cell"]].get("ok")) == c["ok"] and
                    (r["cells"][c["cell"]].get("error_type") or None) == (c["error_type"] or None))
               for c in o["cells"] if c["cell"] < stop):
            continue
        cell, a = r["cells"][stop], norm(o["answer"])
        lit, pr = a in literals(cell.get("code") or ""), printed(a, cell.get("stdout") or "")
        C["used"] += 1
        C["literal in code"] += lit
        C["printed line"] += pr
        C["either"] += lit or pr
    print(f"  {os.path.basename(replays)}: {dict(C)}")


def jeffreys(audit, sizes):
    import numpy as np
    res = json.load(open(audit))
    rng = np.random.default_rng(20261011)
    print("== audit intervals with Jeffreys' prior (stratum share ~ Beta(k+1/2, n-k+1/2))")
    for corpus, (N, denom) in sizes.items():
        cnt = res[corpus]["counts"]
        ks, ns = [], []
        for st in ("pos", "neg-sus", "neg-rest"):
            c = {int(k): v for k, v in cnt[st].items()}
            ks.append(c.get(1, 0))
            ns.append(sum(v for k, v in c.items() if k != 4))      # unclear labels left out
        p = np.stack([rng.beta(k + 0.5, n - k + 0.5, 400000) for k, n in zip(ks, ns)])
        count = (np.array(N)[:, None] * p).sum(0)
        rec = N[0] * p[0] / (N[0] * p[0] + N[1] * p[1] + N[2] * p[2])
        lo, hi = np.percentile(count / denom * 100, [2.5, 97.5])
        rlo, rhi = np.percentile(rec, [2.5, 97.5])
        print(f"  {corpus}: share {lo:.1f}% to {hi:.1f}%; recall {rlo:.2f} to {rhi:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrdata", required=True)
    ap.add_argument("--recorded", required=True, help="folder with e1/ and receipts/ from the recorded-runs archive")
    ap.add_argument("--recovered", required=True, help="output folder of live/fa_rescore.py")
    ap.add_argument("--replays", default=os.path.join(ROOT, "results/fa-replay"))
    ap.add_argument("--audit", help="audit/results_FINAL.json, for the Jeffreys intervals")
    a = ap.parse_args()
    qs = json.load(open(a.qrdata))
    gold = {f"qr-{i}": (q["answer"], (q.get("meta_data") or {}).get("question_type")) for i, q in enumerate(qs)}
    live("recorded", os.path.join(a.recorded, "e1/results.jsonl"), gold)
    live("recovered", os.path.join(a.recovered, "e1/results.jsonl"), gold)
    unrecovered(os.path.join(a.recorded, "e1/results.jsonl"), os.path.join(a.recovered, "e1/results.jsonl"))
    replay(os.path.join(a.recorded, "receipts/results_gpt-oss.jsonl"))
    replay_sensitivity(os.path.join(a.recorded, "receipts/results_gpt-oss.jsonl"))
    live_kinds(os.path.join(a.recorded, "e1/results.jsonl"), os.path.join(a.recovered, "e1/results.jsonl"))
    skipped_lines(os.path.join(a.recorded, "e1/results.jsonl"))
    print("== recovered answers found in the recorded step that gave them")
    answer_evidence(os.path.join(a.recorded, "e1/results.jsonl"), os.path.join(a.replays, "live_gpt.jsonl"),
                    {"P0G0", "P0G1", "P1G0", "P1G1"})
    answer_evidence(os.path.join(a.recorded, "e1/results_deepseek41.jsonl"), os.path.join(a.replays, "live_ds.jsonl"),
                    {"P0G0", "P0G1"})
    if a.audit:
        jeffreys(a.audit, {"DABstep": ([4429, 2577, 19259], 19466), "DAB": ([4751, 3626, 19607], 27984)})


if __name__ == "__main__":
    main()
