#!/bin/sh
# Score the recorded model runs again, without calling a model.
# Needs the recorded runs (recorded-runs-2026-09.tar.gz) and the benchmark data (tools/get_benchmarks.sh).
# Usage: sh tools/rescore.sh     Writes runs/rescore/ and prints the lines to compare with the preprint.
set -e
BENCH=${BENCH:-data}
REC=${REC:-recorded}
OUT=${OUT:-runs/rescore}
IA_TABLES=$BENCH/InfiAgent/examples/DA-Agent/data/da-dev-tables
QR_TABLES=$BENCH/QRData/benchmark/data
QR_QUESTIONS=$BENCH/QRData/benchmark/QRData.json
DB_REAL=$BENCH/discoverybench/discoverybench/real

if [ ! -d "$REC" ]; then
  tar xzf recorded-runs-2026-09.tar.gz
fi
mkdir -p "$OUT"
uv run python tools/make_kb.py infiagent --data "$IA_TABLES" --out "$OUT/ia_kb"
uv run python tools/make_kb.py qrdata --data "$QR_TABLES" --out "$OUT/qr_kb"
uv run python tools/make_kb.py discoverybench --data "$DB_REAL" --out "$OUT/db_kb"

# name, kb folder, results file
RUNS="ia_gpt-oss:ia_kb:$REC/ia/results_gpt-oss.jsonl
ia_deepseek41:ia_kb:$REC/ia/results_deepseek41.jsonl
qr_gpt-oss:qr_kb:$REC/qr/results_gpt-oss.jsonl
db_gpt-oss:db_kb:$REC/db/results_gpt-oss.jsonl"

for run in $RUNS; do
  name=${run%%:*}
  rest=${run#*:}
  kb=$OUT/${rest%%:*}
  results=${rest#*:}
  for version in v6 v7 v8; do
    gate=dabstep-gate/$version
    [ "$version" = v8 ] && gate=dabstep-gate
    uv run kramabench/kb_gate.py --gate-dir "$gate" --kb "$kb" --results "$results" \
      --out "$OUT/${name}_$version.jsonl" > "$OUT/${name}_$version.txt"
  done
  uv run infiagent/ia_sandbox_exists.py --gate-dir dabstep-gate/v7 --kb "$kb" --results "$results" \
    --out "$OUT/${name}_v7_live.jsonl" > "$OUT/${name}_v7_live.txt"
  # the Python folders that the sandbox of the recorded runs mounted
  uv run infiagent/mount_exists.py --roots /home/claude/iaenv /usr -- --gate-dir dabstep-gate --kb "$kb" \
    --results "$results" --out "$OUT/${name}_v8_live.jsonl" > "$OUT/${name}_v8_live.txt"
  uv run baselines/dry_run_score.py --dry "$REC/dry/${name}_s100.jsonl" --gate "$OUT/${name}_v8.jsonl" \
    > "$OUT/${name}_dry_s100.json"
done
uv run baselines/llm_checker_score.py "$REC"/llm_check/*.jsonl > "$OUT/llm_checker.txt"
cat "$REC"/e1/validation_seed1.jsonl "$REC"/e1/validation_seed2.jsonl > "$OUT/e1_validation.jsonl"
uv run live/e1_report.py --results "$REC/e1/results.jsonl" --questions "$QR_QUESTIONS" \
  --validation "$OUT/e1_validation.jsonl" > "$OUT/e1_gpt-oss.txt"
uv run live/e1_report.py --results "$REC/e1/results_deepseek41.jsonl" --questions "$QR_QUESTIONS" \
  --validation "$REC/e1/validation_deepseek41.jsonl" > "$OUT/e1_deepseek41.txt"
uv run rq3-pilot/rq3_full_report.py --episodes "$REC/receipts/episodes.jsonl" \
  --results "$REC/receipts/results_gpt-oss.jsonl" > "$OUT/receipts_gpt-oss.txt"
uv run rq3-pilot/constant_answers.py "$REC/receipts/results_gpt-oss.jsonl" > "$OUT/constant_answers_gpt-oss.txt"

echo
echo "Table 3 (working cells, blocked; failures caught with the version of the row):"
for f in ia_gpt-oss_v7 qr_gpt-oss_v7 ia_deepseek41_v7 db_gpt-oss_v8; do
  echo "  $f: $(grep -h 'cells without an error' "$OUT/$f.txt") | $(grep -h 'caught with a cause match' "$OUT/$f.txt" | cut -d'(' -f1)"
done
echo "Table 3, live file check:"
for f in ia_gpt-oss_v7_live qr_gpt-oss_v7_live ia_deepseek41_v7_live db_gpt-oss_v8_live; do
  echo "  $f: $(grep -h 'caught with a cause match' "$OUT/$f.txt" | cut -d'(' -f1)"
done
echo "v6 on the v7 runs (blocks on working cells):"
for f in ia_gpt-oss_v6 qr_gpt-oss_v6 ia_deepseek41_v6; do
  echo "  $f: $(grep -h 'cells without an error' "$OUT/$f.txt")"
done
echo "Baselines: see $OUT/*_dry_s100.json (caught, false_blocks) and $OUT/llm_checker.txt"
echo "Table 4: $OUT/e1_gpt-oss.txt and $OUT/e1_deepseek41.txt"
echo "Receipt replay: $(grep -h 'all episodes' "$OUT/receipts_gpt-oss.txt")"
echo "Constant answers: $(tail -1 "$OUT/constant_answers_gpt-oss.txt")"
