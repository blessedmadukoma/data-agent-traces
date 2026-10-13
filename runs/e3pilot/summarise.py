import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'live'))
from e2_report import run_measures
here = os.path.dirname(os.path.abspath(__file__))
qs = json.load(open(os.path.join(here, '..', '..', 'data/QRData/benchmark/QRData.json')))
gold = {f"qr-{i}": (q["answer"], (q.get("meta_data") or {}).get("question_type")) for i, q in enumerate(qs)}
for f in sorted(glob.glob(os.path.join(here, '*.jsonl'))):
    rs = [json.loads(l) for l in open(f)]
    m = [run_measures(r, gold) for r in rs]
    n = len(m)
    dre = sum(1 for v in m if v['kinds'].get('file', 0) + v['kinds'].get('key_checker_block', 0) > 0)
    files = sum(v['kinds'].get('file', 0) for v in m); cols = sum(v['kinds'].get('key_checker_block', 0) for v in m)
    ans = sum(v['answered'] for v in m); acc = sum(v['correct'] for v in m)
    cost = sum(r['job_cost_usd'] for r in rs)
    print(f"{os.path.basename(f):28s} runs {n:2d}  runs with a data-reference error {dre:2d} ({100*dre/max(n,1):4.1f}%)  "
          f"[file errors {files}, column errors {cols}]  answered {ans} ({100*ans/max(n,1):.0f}%)  correct {acc}  ${cost:.3f}")
