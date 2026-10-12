#!/bin/sh
# Release 4 of the checker (version 9, dabstep-gate/v9): score every recorded step again and compare
# its decisions with release 3 (version 8, dabstep-gate/gate.py). No model calls.
# Needs: make dabstep, make dab (the parsed cells and the version 8 predictions) and sh tools/rescore.sh.
# Usage: sh tools/release4.sh      Writes runs/release4/ and prints the comparison.
set -e
DATA=${DATA:-data/dabstep}
RUN=${RUN:-runs/dabstep}
DAB_RUN=${DAB_RUN:-runs/dab}
RESCORE=${RESCORE:-runs/rescore}
REC=${REC:-recorded}
BENCH=${BENCH:-data}
OUT=${OUT:-runs/release4}
DAB_MODELS="gemini-2.5-flash gemini-3-pro gpt-5-mini gpt-5.1 gpt-5.2 kimi-k2-thinking"
mkdir -p "$OUT/dab"

echo "1. DABstep traces"
uv run dabstep-gate/evaluate_gate.py --gate-dir dabstep-gate/v9 --data "$DATA" --manifest "$RUN/manifest.jsonl" \
  --cells "$RUN"/parsed_*_cells.jsonl --pred "$OUT/dabstep_v9.jsonl" \
  --lint --carry --files --tolerant auto > "$OUT/dabstep_v9_report.txt"

echo "2. DAB logs"
for m in $DAB_MODELS; do
  PYTHONPATH=dabstep-gate uv run dataagentbench-replay/dab_gate_eval.py --gate-dir dabstep-gate/v9 \
    --cells "$DAB_RUN/cells_$m.jsonl" \
    --pred "$OUT/dab/gate_$m.jsonl" --lint > /dev/null
done
uv run dataagentbench-replay/dab_gate_summary.py "$OUT/dab" > "$OUT/dab_v9_report.txt"

echo "3. Our runs (the four later evaluations), with the evidence of the traces and with the live file check"
RUNS="ia_gpt-oss:ia_kb:$REC/ia/results_gpt-oss.jsonl
ia_deepseek41:ia_kb:$REC/ia/results_deepseek41.jsonl
qr_gpt-oss:qr_kb:$REC/qr/results_gpt-oss.jsonl
db_gpt-oss:db_kb:$REC/db/results_gpt-oss.jsonl"
for run in $RUNS; do
  name=${run%%:*}
  rest=${run#*:}
  kb=$RESCORE/${rest%%:*}
  results=${rest#*:}
  uv run kramabench/kb_gate.py --gate-dir dabstep-gate/v9 --kb "$kb" --results "$results" \
    --out "$OUT/${name}_v9.jsonl" > "$OUT/${name}_v9.txt"
  uv run infiagent/mount_exists.py --roots /home/claude/iaenv /usr -- --gate-dir dabstep-gate/v9 --kb "$kb" \
    --results "$results" --out "$OUT/${name}_v9_live.jsonl" > "$OUT/${name}_v9_live.txt"
done

echo "4. Release 3 against release 4, step by step"
PAIRS="--pair dabstep $RUN/gate_predictions.jsonl $OUT/dabstep_v9.jsonl"
for m in $DAB_MODELS; do
  PAIRS="$PAIRS --pair dab_$m $DAB_RUN/gate_$m.jsonl $OUT/dab/gate_$m.jsonl"
done
for name in ia_gpt-oss ia_deepseek41 qr_gpt-oss db_gpt-oss; do
  PAIRS="$PAIRS --pair $name $RESCORE/${name}_v8.jsonl $OUT/${name}_v9.jsonl"
  PAIRS="$PAIRS --pair ${name}_live $RESCORE/${name}_v8_live.jsonl $OUT/${name}_v9_live.jsonl"
done
# shellcheck disable=SC2086
uv run dabstep-gate/v9/compare_releases.py $PAIRS --json "$OUT/compare.json" | tee "$OUT/compare.txt"

echo "5. The live QRData experiment: every step of the runs with the checker, decided again"
uv run dabstep-gate/v9/live_replay.py --results "$REC/e1/results.jsonl" --tables "$BENCH/QRData/benchmark/data" \
  --kb "$RESCORE/qr_kb" | tee "$OUT/live_replay.txt"
