"""Second human coder of the data-reference audit (October 2026): agreement and estimates.

The second human coder labelled the 400 audit items on a private page, blind to the strata, the rules'
decisions and the other coders, with the frozen codebook and clarification C1 (audit/README.md). Part 1 holds
a stratified random 100 items per corpus (50 pos, 25 neg-sus, 25 neg-rest) and the 34 items on which coder 1
and the LLM coder disagreed before adjudication; part 2 holds the rest (audit/human2/make_order.py).

Reports, per corpus:
  * agreement (raw, Cohen's kappa on the four labels and on label 1 against the rest) of the human coder with
    coder 1 after adjudication, coder 1 before adjudication, and the LLM coder: on the random part-1 items and on
    every item that the human coder labelled;
  * precision, recall and the corrected share from the human coder's labels (the estimator and bootstrap of
    tools/audit_dre_analysis.py), on every labelled item;
  * the items where the human coder and coder 1 (final) disagree, for adjudication.

Usage: python3 tools/audit_human2_analysis.py --human2 audit/human2/labels_human2.json [--json out.json]
"""
import argparse, collections, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_dre_analysis import B, estimates, kappa, load_sample  # noqa: E402


def agree(ids, x, y):
    ids = [i for i in ids if i in x and i in y and x[i] and y[i]]
    a, b = [x[i] for i in ids], [y[i] for i in ids]
    return {"n": len(ids), "raw": sum(p == q for p, q in zip(a, b)) / max(1, len(ids)),
            "kappa4": kappa(a, b, [1, 2, 3, 4]),
            "kappa_binary": kappa([p == 1 for p in a], [q == 1 for q in b], [True, False])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human2", required=True, help="JSON: {id: {label, note}} or a list of {id, label, note}")
    ap.add_argument("--audit", default="audit")
    ap.add_argument("--json")
    a = ap.parse_args()
    raw = json.load(open(a.human2))
    rows = raw if isinstance(raw, list) else [dict(v, id=k) for k, v in raw.items()]
    h2 = {int(r["id"]): (int(r["label"]) if r.get("label") else None) for r in rows}
    c1f = {int(k): v["label"] for k, v in json.load(open(os.path.join(a.audit, "labels_coder1_final.json"))).items()}
    c1r = {int(k): v.get("raw_label", v["label"]) for k, v in json.load(open(os.path.join(a.audit, "labels_coder1_final.json"))).items()}
    llm = {int(json.loads(l)["id"]): json.loads(l)["label"] for l in open(os.path.join(a.audit, "labels_coder2.jsonl")) if l.strip()}
    comp = json.load(open(os.path.join(a.audit, "human2", "part1_composition_SEALED.json")))
    random_part = set(comp["random_sample"])
    sample = load_sample(os.path.join(a.audit, "sample_ids.csv"))
    strata = json.load(open(os.path.join(a.audit, "strata.json")))
    out = {}
    rng = random.Random(20261012)
    for corpus in ("DABstep", "DAB"):
        its = [s for s in sample if s["corpus"] == corpus]
        ids_all = [s["id"] for s in its if h2.get(s["id"])]
        ids_rand = [i for i in ids_all if i in random_part]
        res = {"labelled": len(ids_all), "labelled_random_part1": len(ids_rand)}
        for name, other in (("coder1_final", c1f), ("coder1_before_adjudication", c1r), ("llm_coder", llm)):
            res[name] = {"random_part1": agree(ids_rand, h2, other), "all_labelled": agree(ids_all, h2, other)}
        by = collections.defaultdict(list)
        for s in its:
            if h2.get(s["id"]):
                by[s["stratum"]].append(h2[s["id"]])
        for h in ("pos", "neg-sus", "neg-rest"):
            by.setdefault(h, [])
        N, total = strata[corpus]["population"], strata[corpus]["failed_total"]
        est = estimates(by, N, total)
        boots = collections.defaultdict(list)
        for _ in range(B):
            rs = {h: [rng.choice(v) for _ in v] if v else [] for h, v in by.items()}
            e = estimates(rs, N, total)
            for k in ("precision", "recall", "corrected_share"):
                boots[k].append(e[k])
        res["estimates_human2"] = est
        res["ci95_human2"] = {k: (sorted(v)[int(0.025 * B)], sorted(v)[int(0.975 * B) - 1]) for k, v in boots.items()}
        res["disagreements_with_coder1_final"] = [
            {"id": i, "human2": h2[i], "coder1_final": c1f.get(i), "llm": llm.get(i)} for i in ids_all if h2[i] != c1f.get(i)]
        out[corpus] = res
        g = res["coder1_final"]
        print(f"== {corpus}: {len(ids_all)} labelled ({len(ids_rand)} in the random part)")
        for name in ("coder1_final", "coder1_before_adjudication", "llm_coder"):
            for scope in ("random_part1", "all_labelled"):
                x = res[name][scope]
                print(f"   {name:28s} {scope:13s} n={x['n']:3d} raw {x['raw']:.3f} kappa4 {x['kappa4']:.3f} "
                      f"kappa(1 vs rest) {x['kappa_binary']:.3f}")
        e, ci = est, res["ci95_human2"]
        print(f"   human coder's estimates: precision {e['precision']:.3f} {tuple(round(v, 3) for v in ci['precision'])}, "
              f"recall {e['recall']:.3f}, corrected share {e['corrected_share']:.3f} "
              f"{tuple(round(v, 3) for v in ci['corrected_share'])} (rules {e['rule_share']:.3f})")
        print(f"   disagreements with coder 1 (final): {len(res['disagreements_with_coder1_final'])}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
