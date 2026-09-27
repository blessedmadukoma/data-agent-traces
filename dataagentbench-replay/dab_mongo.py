#!/usr/bin/env python3
"""MongoDB checks on DataAgentBench: fields that no document has, and the default limit.

Usage:
  uv run --with pymongo --with mongomock dataagentbench-replay/dab_mongo.py \
      --cells $DAB_RUN/cells_*.jsonl --dumps dataagentbench/mongo --json $DAB_RUN/mongo.json

MongoDB returns no error for a filter or projection on a field that no
document has; the filter matches nothing (or everything, for $exists: false
and similar) and the projection drops the field. For every logged query:
  * field paths in the filter (keys outside $-operators, through $and, $or,
    $nor and $elemMatch) and in the projection;
  * a path is missing if no document of the collection has it (arrays of
    sub-documents count, as in MongoDB dot notation);
  * a query without "limit" gets limit 5 from the harness. With the dump
    loaded into mongomock, the true number of matching documents shows
    whether the 5 records were a cut.
Collections and their source (commit 0a1a9c0c of refs/pull/6/head):
  yelp.business, yelp.checkin, agnews.articles, civic_unstructured.civic_docs:
  BSON dumps. paper_unstructured.paper_docs: no dump in the repository; its
  fields {_id, filename, text} come from create_databases.py, and no
  truncation check is made.
"""
import argparse
import collections
import json
import os

DUMPS = {("yelp", "business"): "business.bson", ("yelp", "checkin"): "checkin.bson",
         ("agnews", "articles"): "articles.bson", ("civic_unstructured", "civic_docs"): "civic_docs.bson"}
SCRIPT_FIELDS = {("paper_unstructured", "paper_docs"): {"_id", "filename", "text"}}
LOGICAL = {"$and", "$or", "$nor"}
EMPTY_OK = {"$exists", "$ne", "$nin", "$not"}     # can match documents that lack the field


def paths(doc, prefix=""):
    out = set()
    if isinstance(doc, dict):
        for k, v in doc.items():
            p = f"{prefix}{k}"
            out.add(p)
            out |= paths(v, p + ".")
    elif isinstance(doc, list):
        for v in doc:
            out |= paths(v, prefix)
    return out


def filter_fields(f, prefix=""):
    """[(path, may_match_missing)] for a filter document."""
    out = []
    if not isinstance(f, dict):
        return out
    for k, v in f.items():
        if k in LOGICAL and isinstance(v, list):
            for sub in v:
                out += filter_fields(sub, prefix)
        elif k.startswith("$"):
            continue                      # $expr, $where, $text, ...: not followed
        else:
            p = prefix + k
            loose = v is None or (isinstance(v, dict) and (set(v) & EMPTY_OK or v.get("$in") and None in v["$in"]))
            out.append((p, bool(loose)))
            if isinstance(v, dict) and isinstance(v.get("$elemMatch"), dict):
                out += filter_fields(v["$elemMatch"], p + ".")
    return out


def proj_fields(pr):
    if not isinstance(pr, dict):
        return []
    return [k for k in pr if not k.startswith("$") and k != "_id"]


class Unsupported(Exception):
    pass


def _values(doc, path):
    """Values at a dotted path, through arrays of sub-documents (MongoDB dot notation)."""
    cur = [doc]
    for part in path.split("."):
        nxt = []
        for c in cur:
            if isinstance(c, dict) and part in c:
                nxt.append(c[part])
            elif isinstance(c, list):
                nxt += [x[part] for x in c if isinstance(x, dict) and part in x]
        cur = nxt
    return cur


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _key(x):
    """Hashable key under MongoDB equality: numbers compare across int and float."""
    if _num(x):
        return ("n", float(x))
    if isinstance(x, (str, bool)) or x is None:
        return (type(x).__name__, x)
    return ("r", repr(x))


def _cmp(a, b, op):
    if _num(a) and _num(b) or isinstance(a, str) and isinstance(b, str):
        return {"$gt": a > b, "$gte": a >= b, "$lt": a < b, "$lte": a <= b}[op]
    return False                      # different type brackets never match


def _compile_value(cond):
    """Predicate over the values at a path (an empty list if the field is missing)."""
    import re

    def flat(vals):
        out = []
        for v in vals:
            out.append(v)
            if isinstance(v, list):
                out += v
        return out

    if not isinstance(cond, dict) or not any(k.startswith("$") for k in cond):
        k = _key(cond)
        return lambda vals: any(_key(v) == k for v in flat(vals)) or (cond is None and not vals)
    tests = []
    for op, arg in cond.items():
        if op == "$options":
            continue
        if op == "$regex":
            fl = 0
            for ch in cond.get("$options", ""):
                fl |= {"i": re.I, "m": re.M, "s": re.S, "x": re.X}.get(ch, 0)
            rx = re.compile(arg, fl)
            tests.append(lambda vals, rx=rx: any(isinstance(v, str) and rx.search(v) for v in flat(vals)))
        elif op in ("$in", "$nin"):
            ks = {_key(x) for x in arg}
            none_in = None in arg
            t = (lambda vals, ks=ks, n=none_in: any(_key(v) in ks for v in flat(vals)) or (n and not vals))
            tests.append(t if op == "$in" else (lambda vals, t=t: not t(vals)))
        elif op == "$ne":
            k = _key(arg)
            tests.append(lambda vals, k=k: not any(_key(v) == k for v in flat(vals)))
        elif op == "$exists":
            tests.append(lambda vals, a=bool(arg): bool(vals) == a)
        elif op in ("$gt", "$gte", "$lt", "$lte"):
            tests.append(lambda vals, op=op, arg=arg: any(_cmp(v, arg, op) for v in flat(vals)))
        else:
            raise Unsupported(op)
    return lambda vals: all(t(vals) for t in tests)


def compile_filter(f):
    """MongoDB find() semantics for the operators seen in the DAB logs; Unsupported otherwise."""
    parts = []
    for k, v in (f or {}).items():
        if k in ("$and", "$or", "$nor"):
            subs = [compile_filter(x) for x in v]
            if k == "$and":
                parts.append(lambda d, subs=subs: all(p(d) for p in subs))
            elif k == "$or":
                parts.append(lambda d, subs=subs: any(p(d) for p in subs))
            else:
                parts.append(lambda d, subs=subs: not any(p(d) for p in subs))
        elif k.startswith("$"):
            raise Unsupported(k)
        else:
            pv = _compile_value(v)
            parts.append(lambda d, k=k, pv=pv: pv(_values(d, k)))
    return lambda d: all(p(d) for p in parts)


def missing(p, fields):
    parts = p.split(".")
    if any(x.isdigit() for x in parts):
        return None                       # array index: not checked
    return p not in fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--json")
    ap.add_argument("--cache", help="JSON file of true match counts; reused and extended across runs")
    ap.add_argument("--budget", type=float, default=140, help="seconds to spend on new counts in this run")
    a = ap.parse_args()
    import time
    t_start = time.time()
    import bson
    import mongomock
    fields, colls, docs_of = {}, {}, {}
    client = mongomock.MongoClient()
    for key, fn in DUMPS.items():
        data = open(os.path.join(a.dumps, fn), "rb").read()
        docs = bson.decode_all(data)
        # mongorestore gives every document an _id
        fields[key] = {"_id"} | (set().union(*(paths(d) for d in docs)) if docs else set())
        docs_of[key] = docs
        c = client["dab"][f"{key[0]}__{key[1]}"]
        if docs and len(docs) < 1000:      # mongomock is a fallback for small collections only
            c.insert_many(docs)
        colls[key] = c
        print(f"{key[0]}.{key[1]}: {len(docs)} documents, {len(fields[key])} field paths")
    fields.update(SCRIPT_FIELDS)

    C = collections.Counter()
    cache = {}
    if a.cache and os.path.exists(a.cache):
        cache = {tuple(json.loads(k)): v for k, v in json.load(open(a.cache)).items()}
    pending = 0
    per = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    for p in a.cells:
        for line in open(p):
            r = json.loads(line)
            for c in r["calls"]:
                if c.get("db_type") != "mongo" or not c.get("mongo"):
                    continue
                m = c["mongo"]
                key = (r["dataset"], m.get("collection"))
                pc = per[f"{r['model']}"]
                for cc in (C, pc):
                    cc["queries"] += 1
                if c["failed"]:
                    continue
                for cc in (C, pc):
                    cc["ok"] += 1
                fs = fields.get(key)
                if fs is None:
                    C["unknown_collection"] += 1
                    continue
                ff = filter_fields(m.get("filter"))
                strict_missing = [f for f, loose in ff if not loose and missing(f, fs)]
                loose_missing = [f for f, loose in ff if loose and missing(f, fs)]
                pm = [f for f in proj_fields(m.get("projection")) if missing(f, fs)]
                n = c.get("n_records")
                for cc in (C, pc):
                    if strict_missing:
                        cc["filter_missing"] += 1
                        cc["filter_missing_empty"] += n == 0
                    if loose_missing:
                        cc["filter_missing_loose"] += 1
                    if pm:
                        cc["proj_missing"] += 1
                if strict_missing and len(examples["filter"]) < 5:
                    examples["filter"].append((r["trace"], c["i"], strict_missing, n))
                if pm and len(examples["proj"]) < 5:
                    examples["proj"].append((r["trace"], c["i"], pm))
                # default limit
                if not m.get("limit_given"):
                    for cc in (C, pc):
                        cc["nolimit"] += 1
                    if key in colls:
                        ck = (key[0], key[1], json.dumps(m.get("filter"), sort_keys=True, default=str))
                        if ck not in cache and time.time() - t_start > a.budget:
                            pending += 1
                            continue
                        if ck not in cache:
                            try:
                                pred = compile_filter(m.get("filter"))
                                cache[ck] = sum(1 for d in docs_of[key] if pred(d))
                            except Unsupported:
                                cache[ck] = None
                                if len(docs_of[key]) < 1000:     # small collection: mongomock is fast enough
                                    try:
                                        cache[ck] = colls[key].count_documents(m.get("filter") or {})
                                    except Exception:
                                        pass
                            except Exception:
                                cache[ck] = None
                        true = cache[ck]
                        if true is None:
                            C["nolimit_count_error"] += 1
                            continue
                        for cc in (C, pc):
                            cc["nolimit_checked"] += 1
                            cc["nolimit_zero"] += true == 0
                            cc["nolimit_cut"] += true > 5
                            cc["nolimit_true_gt5_sum"] += true if true > 5 else 0
    if a.cache:
        json.dump({json.dumps(list(k)): v for k, v in cache.items()}, open(a.cache, "w"))
    if pending:
        print(f"INCOMPLETE: {pending} queries still need a count; run again with the same --cache")
    print("\nall models:", json.dumps(dict(C), sort_keys=True))
    for mdl in sorted(per):
        print(f"  {mdl}: {dict(per[mdl])}")
    for k, v in examples.items():
        print(f"examples ({k}):")
        for e in v:
            print("   ", e)
    if a.json:
        json.dump({"all": dict(C), "per_model": {k: dict(v) for k, v in per.items()},
                   "examples": examples}, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
