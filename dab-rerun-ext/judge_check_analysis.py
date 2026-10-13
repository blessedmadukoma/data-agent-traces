"""H4b: agreement between the human check and DAB's failure-analysis judge (pre-registration section 4).

The human coder labelled each of the 40 failed runs drawn by judge_check_items.py with DAB's failure modes FM1 to FM4,
blind to the arm and to the judge's labels. A run can have more than one mode. This script compares, for each mode,
"the human gave this mode" with "the judge said yes", by Cohen's kappa, per arm and pooled, with a 95% bootstrap
interval over runs (10,000 resamples, seed 20261012) for the pooled kappa.

Inputs: the human labels (modes per item; no trace content) and the sealed key (item -> arm and the judge's labels).
Usage: python3 dab-rerun-ext/judge_check_analysis.py --human dab-rerun-ext/judge_check_human_labels.json \
           --key $DAB_OUT/judge_check/key_SEALED.json [--json out.json]
"""
import argparse
import json
import random

MODES = ["FM1", "FM2", "FM3", "FM4"]


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, ((po - pe) / (1 - pe) if pe < 1 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    human = json.load(open(a.human))["decisions"]
    key = json.load(open(a.key))
    ids = sorted(key, key=int)
    out = {"n": len(ids), "unsure": sum(bool(human[i].get("unsure")) for i in ids), "modes": {}}
    rng = random.Random(20261012)
    for m in MODES:
        res = {}
        for scope in ("pooled", "A", "B"):
            sel = [i for i in ids if scope == "pooled" or key[i]["arm"] == scope]
            j = [bool(key[i]["judge"][m]) for i in sel]
            h = [m in human[i]["modes"] for i in sel]
            po, k = kappa(j, h)
            res[scope] = {"n": len(sel), "agree": sum(x == y for x, y in zip(j, h)), "raw": po, "kappa": k,
                          "judge_yes": sum(j), "human_yes": sum(h)}
        boots = []
        for _ in range(10000):
            s = [rng.choice(ids) for _ in ids]
            k = kappa([bool(key[i]["judge"][m]) for i in s], [m in human[i]["modes"] for i in s])[1]
            if k is not None:
                boots.append(k)
        boots.sort()
        res["pooled"]["kappa_ci95"] = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else None
        out["modes"][m] = res
        p = res["pooled"]
        kp = "n/a" if p["kappa"] is None else f"{p['kappa']:.3f}"
        ci = p["kappa_ci95"]
        print(f"{m}: pooled agree {p['agree']}/{p['n']} kappa {kp}"
              + (f" (95% CI {ci[0]:.2f} to {ci[1]:.2f})" if ci else "")
              + f"; judge yes {p['judge_yes']}, human yes {p['human_yes']}; "
              + "; ".join(f"arm {s}: agree {res[s]['agree']}/{res[s]['n']}, kappa "
                          + ("n/a" if res[s]["kappa"] is None else f"{res[s]['kappa']:.3f}")
                          + f", judge {res[s]['judge_yes']}, human {res[s]['human_yes']}" for s in ("A", "B")))
    print(f"runs marked unsure by the human: {out['unsure']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
