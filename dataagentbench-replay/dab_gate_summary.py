#!/usr/bin/env python3
"""Pool the per-model summaries that dab_gate_eval.py writes.  Usage: dab_gate_summary.py DIR"""
import collections
import glob
import json
import os
import statistics
import sys

d = sys.argv[1]
C, K, L = collections.Counter(), collections.Counter(), []
for p in sorted(glob.glob(os.path.join(d, "gate_*.jsonl.summary.json"))):
    s = json.load(open(p))
    C.update(s["counts"])
    K.update(s["kinds"])
    L += s["latency_ms"]
ran = max(1, C["ran"])
print(f"DAB {d}: ran {C['ran']}; data-reference failures {C['data_fail']}, caught {C['data_cause']} "
      f"({C['data_cause'] / max(1, C['data_fail']):.1%}), by kind {dict(K)}, not flagged by pyflakes "
      f"{C['data_cause_not_lint']}; interface misuse {C['iface_fail']}, caught {C['iface_cause']}; "
      f"error-free calls {C['ok']}, blocked {C['ok_block']}; failed calls blocked {C['failed_block']}, "
      f"cause-matched {C['cause']}; outcomes " + ", ".join(
          f"{k[2:]} {v / ran:.1%}" for k, v in sorted(C.items()) if k.startswith("o:"))
      + (f"; latency median {statistics.median(L):.2f} ms" if L else ""))
