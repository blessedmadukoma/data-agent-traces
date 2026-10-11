"""Draw the hand-audit sample for the data-reference classification (paper v2, Section 2.2).

Population: the failed steps that the paper counts.
  DABstep: every failed cell in runs/dabstep/parsed_*_cells.jsonl (26,265).
  DAB: every failed call in runs/dab/cells_*.jsonl that the harness did not cause (27,984).
Strata per corpus:
  pos      steps that the classifier labels as data-reference errors; simple random sample of 100.
  neg-sus  other failed steps whose error type can hide a missing name (AttributeError, ValueError,
           IndexError, unclassified errors, other SQL/MongoDB errors and so on); 50.
  neg-rest all remaining failed steps; 50.
Selection order is SHA-256(salt|corpus|uid), so the same inputs and salt always give the same sample.
The page that the labeller sees has no stratum and no classifier label. Items are shuffled within
each corpus by a second hash.

Usage: python3 tools/audit_dre_sample.py --out runs/audit_dre
"""
import argparse
import collections
import glob
import hashlib
import json
import os

SALT = "dre-audit-2026-10-07"
N_POS, N_SUS, N_REST = 100, 50, 50

DABSTEP_POS = {"index-error(wrapped)", "FileNotFoundError", "KeyError", "ValueError:usecols", "KeyError(wrapped)"}
DABSTEP_SUS = {"other", "ValueError", "AttributeError", "IndexError", "ParserError", "UndefinedVariableError",
               "IndexingError", "MergeError", "PermissionError"}
DAB_POS = {"sql:missing-column", "sql:missing-table", "mongo:missing-collection", "py:data-key", "py:data-file"}
DAB_SUS = {"py:other", "py:ValueError", "py:AttributeError", "py:IndexError", "sql:other", "mongo:other",
           "py:OSError", "py:DatabaseError", "py:OperationalError"}

CODE_MAX, ERR_MAX, PREV_MAX, N_PREV = 6000, 2500, 1500, 3


def h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def cut_middle(s, n):
    s = s or ""
    if len(s) <= n:
        return s
    k = n // 2
    return s[:k] + f"\n... [{len(s) - n} characters cut] ...\n" + s[-k:]


def tail(s, n):
    s = s or ""
    return s if len(s) <= n else f"[{len(s) - n} earlier characters cut]\n" + s[-n:]


def dabstep_items(root):
    tasks = {}
    for line in open(os.path.join(root, "data/dabstep/data/tasks/all.jsonl")):
        t = json.loads(line)
        tasks[str(t["task_id"])] = t["question"]
    traces = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, "runs/dabstep/parsed_*_cells.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            traces[(r["submission_file"], str(r["trace_line"]))].append(r)
    out = []
    for (sub, tl), cells in traces.items():
        cells.sort(key=lambda r: int(r["cell_index"]))
        for j, r in enumerate(cells):
            if not r.get("failed"):
                continue
            et = str(r.get("error_type"))
            stratum = "pos" if et in DABSTEP_POS else "neg-sus" if et in DABSTEP_SUS else "neg-rest"
            prev = [{"code": cut_middle(c.get("code"), PREV_MAX), "output": tail(c.get("output"), 600)}
                    for c in cells[max(0, j - N_PREV):j]]
            out.append({
                "uid": f"{sub}|{tl}|{r['cell_index']}", "corpus": "DABstep", "stratum": stratum, "group": et,
                "context": f"DABstep task {r['task_id']} ({sub.split('__')[1] if '__' in sub else sub})",
                "question": tasks.get(str(r["task_id"]), ""), "kind": "python",
                "code": cut_middle(r.get("code"), CODE_MAX), "error": tail(r.get("output"), ERR_MAX), "prev": prev})
    return out


def dab_items(root):
    qtext = {}
    qdir = os.path.join(root, "runs/audit_dre/dab_queries.json")
    if os.path.exists(qdir):
        qtext = json.load(open(qdir))
    out = []
    for p in sorted(glob.glob(os.path.join(root, "runs/dab/cells_*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            calls = r["calls"]
            for j, x in enumerate(calls):
                if not x["failed"] or x.get("harness_syntax"):
                    continue
                g = x["group"]
                stratum = "pos" if g in DAB_POS else "neg-sus" if g in DAB_SUS else "neg-rest"
                if x["tool"] == "query_db":
                    kind, code = f"query_db ({x.get('db_type')})", x.get("query") or ""
                else:
                    kind, code = x["tool"], x.get("code") or ""
                prev = []
                for c in calls[max(0, j - N_PREV):j]:
                    body = c.get("query") if c["tool"] == "query_db" else c.get("code")
                    prev.append({"code": f"[{c['tool']}] " + cut_middle(str(body or ""), PREV_MAX),
                                 "output": tail(c.get("error") or ("(succeeded)" if not c["failed"] else ""), 600)})
                out.append({
                    "uid": f"{r['trace']}|{x['i']}", "corpus": "DAB", "stratum": stratum, "group": g,
                    "context": f"DAB {r['dataset']} {r['query']} ({r['model']})", "dataset": r["dataset"],
                    "question": qtext.get(f"{r['dataset']}/{r['query']}", ""), "kind": kind,
                    "code": cut_middle(str(code), CODE_MAX), "error": tail(x.get("error"), ERR_MAX), "prev": prev})
    return out


def draw(items, corpus):
    by = collections.defaultdict(list)
    for it in items:
        by[it["stratum"]].append(it)
    sizes = {s: len(v) for s, v in by.items()}
    chosen = []
    for s, n in (("pos", N_POS), ("neg-sus", N_SUS), ("neg-rest", N_REST)):
        pool = sorted(by[s], key=lambda it: h(SALT, corpus, it["uid"]))
        chosen += pool[:n]
    chosen.sort(key=lambda it: h(SALT, "order", corpus, it["uid"]))
    return chosen, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="runs/audit_dre")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sample, strata = [], {}
    for corpus, items in (("DABstep", dabstep_items(a.root)), ("DAB", dab_items(a.root))):
        chosen, sizes = draw(items, corpus)
        strata[corpus] = {"population": sizes, "sample": dict(collections.Counter(it["stratum"] for it in chosen)),
                          "failed_total": len(items)}
        sample += chosen
    for i, it in enumerate(sample, 1):
        it["id"] = i
    with open(os.path.join(a.out, "sample_full.jsonl"), "w") as f:
        for it in sample:
            f.write(json.dumps(it) + "\n")
    blind = [{k: it[k] for k in ("id", "corpus", "context", "question", "kind", "code", "error", "prev")} | (
        {"dataset": it["dataset"]} if "dataset" in it else {}) for it in sample]
    with open(os.path.join(a.out, "sample_blind.json"), "w") as f:
        json.dump(blind, f)
    json.dump(strata, open(os.path.join(a.out, "strata.json"), "w"), indent=1)
    print(json.dumps(strata, indent=1))
    for name in ("sample_full.jsonl", "sample_blind.json", "strata.json"):
        print(name, h(open(os.path.join(a.out, name)).read())[:16])


if __name__ == "__main__":
    main()
