"""Analyse the hand audit of the data-reference classification (audit/PLAN.md).

Inputs:
  --sample   audit/sample_ids.csv, or runs/audit_dre/sample_full.jsonl from tools/audit_dre_sample.py
             (strata and ids)
  --strata   audit/strata.json (population sizes)
  --coder1   JSON {id: {"label": 1-4, "note": "...", "raw_label": ...}} (final labels, after adjudication;
             raw_label is the label before adjudication, present only where adjudication changed it)
  --coder2   JSONL lines {"id", "label", "note"} (LLM coder), optional
  --coder1-raw  coder 1 labels before adjudication, for kappa (optional; by default they are rebuilt
             from the raw_label fields of --coder1)
Prints precision, recall, corrected counts and shares with 95% intervals, per corpus, and agreement.

Release note (October 2026): two changes from the version used on 7 October, neither of which changes
a number: --sample also reads the CSV of identifiers that the repository publishes, and the labels
before adjudication are rebuilt from raw_label when --coder1-raw is not given (the 7 October run passed
them as a separate file).

Usage: python3 tools/audit_dre_analysis.py --sample audit/sample_ids.csv --strata audit/strata.json \
           --coder1 audit/labels_coder1_final.json --coder2 audit/labels_coder2.jsonl --json /tmp/results.json
"""
import argparse
import collections
import csv
import json
import math
import random

B = 10000


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def kappa(a, b, cats):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def estimates(items, N, total):
    """items: {stratum: [label, ...]} with labels 1-3 only. Returns dict of estimates."""
    def share(lst):
        lst = [x for x in lst if x in (1, 2, 3)]
        return (sum(x == 1 for x in lst) / len(lst)) if lst else 0.0
    prec = share(items["pos"])
    tp = N["pos"] * prec
    fn = sum(N[h] * share(items[h]) for h in ("neg-sus", "neg-rest"))
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": prec, "tp": tp, "fn": fn, "recall": rec, "corrected": tp + fn,
            "corrected_share": (tp + fn) / total, "rule_share": N["pos"] / total}


def load_sample(path):
    if path.endswith(".csv"):
        return [{"id": int(r["id"]), "corpus": r["corpus"], "stratum": r["stratum"]} for r in csv.DictReader(open(path))]
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="audit/sample_ids.csv")
    ap.add_argument("--strata", default="audit/strata.json")
    ap.add_argument("--coder1", required=True)
    ap.add_argument("--coder1-raw")
    ap.add_argument("--coder2")
    ap.add_argument("--json")
    a = ap.parse_args()
    sample = load_sample(a.sample)
    strata = json.load(open(a.strata))
    c1 = {int(k): v for k, v in json.load(open(a.coder1)).items()}
    if a.coder1_raw:
        c1raw = {int(k): v for k, v in json.load(open(a.coder1_raw)).items()}
    else:
        c1raw = {k: {"label": v.get("raw_label", v.get("label"))} for k, v in c1.items()}
    c2 = {}
    if a.coder2:
        for path in a.coder2.split(","):
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    c2[int(r["id"])] = r
    out = {}
    rng = random.Random(20261007)
    for corpus in ("DABstep", "DAB"):
        N = strata[corpus]["population"]
        total = strata[corpus]["failed_total"]
        its = [s for s in sample if s["corpus"] == corpus]
        missing = [s["id"] for s in its if s["id"] not in c1 or not c1[s["id"]].get("label")]
        by = collections.defaultdict(list)
        counts = collections.defaultdict(collections.Counter)
        for s in its:
            lab = c1.get(s["id"], {}).get("label")
            if lab:
                by[s["stratum"]].append(int(lab))
                counts[s["stratum"]][int(lab)] += 1
        est = estimates(by, N, total)
        boots = collections.defaultdict(list)
        for _ in range(B):
            rs = {h: [rng.choice(v) for _ in v] if v else [] for h, v in by.items()}
            for h in ("pos", "neg-sus", "neg-rest"):
                rs.setdefault(h, [])
            e = estimates(rs, N, total)
            for k in ("precision", "recall", "corrected", "corrected_share"):
                boots[k].append(e[k])
        ci = {k: (sorted(v)[int(0.025 * B)], sorted(v)[int(0.975 * B) - 1]) for k, v in boots.items()}
        pos = [x for x in by["pos"] if x in (1, 2, 3)]
        k1 = sum(x == 1 for x in pos)
        unclear_pos = sum(x == 4 for x in by["pos"])
        res = {"labelled": sum(len(v) for v in by.values()), "missing": missing, "counts": {h: dict(c) for h, c in counts.items()},
               "estimates": est, "ci95_bootstrap": ci, "precision_wilson": wilson(k1, len(pos)),
               "precision_unclear_as_not": k1 / max(1, len(by["pos"])), "unclear_in_pos": unclear_pos}
        if c2:
            ids = [s["id"] for s in its if s["id"] in c2 and s["id"] in c1raw and c1raw[s["id"]].get("label")]
            l1 = [int(c1raw[i]["label"]) for i in ids]
            l2 = [int(c2[i]["label"]) for i in ids]
            res["agreement"] = {"n": len(ids), "raw": sum(x == y for x, y in zip(l1, l2)) / max(1, len(ids)),
                                "kappa4": kappa(l1, l2, [1, 2, 3, 4]),
                                "kappa_binary": kappa([x == 1 for x in l1], [y == 1 for y in l2], [True, False]),
                                "disagreements": [i for i, x, y in zip(ids, l1, l2) if x != y]}
        out[corpus] = res
        e = est
        print(f"\n== {corpus}: {res['labelled']} labelled, {len(missing)} missing")
        print("   counts by stratum:", res["counts"])
        print(f"   precision {e['precision']:.3f} (bootstrap {ci['precision'][0]:.3f}-{ci['precision'][1]:.3f}; "
              f"Wilson {res['precision_wilson'][0]:.3f}-{res['precision_wilson'][1]:.3f}); unclear counted as not: {res['precision_unclear_as_not']:.3f}")
        print(f"   missed (FN) {e['fn']:.0f}; recall {e['recall']:.3f} ({ci['recall'][0]:.3f}-{ci['recall'][1]:.3f})")
        print(f"   corrected count {e['corrected']:.0f} ({ci['corrected'][0]:.0f}-{ci['corrected'][1]:.0f}); "
              f"share {e['corrected_share']:.3f} ({ci['corrected_share'][0]:.3f}-{ci['corrected_share'][1]:.3f}) vs rule {e['rule_share']:.3f}")
        if "agreement" in res:
            g = res["agreement"]
            print(f"   agreement with coder 2: n={g['n']}, raw {g['raw']:.3f}, kappa(4) {g['kappa4']:.3f}, "
                  f"kappa(binary) {g['kappa_binary']:.3f}, disagreements {len(g['disagreements'])}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
