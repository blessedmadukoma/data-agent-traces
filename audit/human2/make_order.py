#!/usr/bin/env python3
"""Order of the 400 audit items for the second human coder (12 October 2026).

Part 1: per corpus, a random sample of 100 items stratified as the audit (50 pos, 25 neg-sus, 25 neg-rest),
plus every item on which coder 1 and the LLM coder disagreed before adjudication (results_FINAL.json),
shuffled together so that the coder cannot tell which items are which. Part 2: the other items, shuffled.
Salt: second-human-coder-2026-10-12. Writes order.json: {"part1": [ids], "part2": [ids]}.
"""
import csv, json, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
rng = random.Random("second-human-coder-2026-10-12")
rows = list(csv.DictReader(open(os.path.join(AUD, "sample_ids.csv"))))
res = json.load(open(os.path.join(AUD, "results_FINAL.json")))
dis = {i for c in res for i in res[c]["agreement"]["disagreements"]}
part1 = set()
for corpus in ("DABstep", "DAB"):
    for stratum, k in (("pos", 50), ("neg-sus", 25), ("neg-rest", 25)):
        ids = sorted(int(r["id"]) for r in rows if r["corpus"] == corpus and r["stratum"] == stratum)
        part1 |= set(rng.sample(ids, k))
random_part = set(part1)
part1 |= dis
part2 = sorted({int(r["id"]) for r in rows} - part1)
p1 = sorted(part1)
rng.shuffle(p1)
rng.shuffle(part2)
json.dump({"part1": p1, "part2": part2}, open(os.path.join(HERE, "order.json"), "w"))
json.dump({"random_sample": sorted(random_part), "disagreements": sorted(dis)},
          open(os.path.join(HERE, "part1_composition_SEALED.json"), "w"))
print(f"part 1: {len(p1)} items ({len(random_part)} random, {len(dis - random_part)} disagreements added); "
      f"part 2: {len(part2)} items")
