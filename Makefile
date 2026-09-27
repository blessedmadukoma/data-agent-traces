# Reproduce the preprint. Run `make help` for the list of targets.
# Paths can be changed on the command line, for example: make dabstep DATA=/data/dabstep

DATA ?= data/dabstep
RUN ?= runs/dabstep
DAB_LOGS ?= data/dab-logs
DAB_MONGO ?= data/dab-mongo
DAB_BOOKREVIEW ?= data/dab-bookreview-rerun
DAB_RUN ?= runs/dab
RQ3_RUN ?= runs/receipts
RQ3_MODEL ?= gpt-oss:120b

DABSTEP_COMMIT = c8fb51b3b898e1755f3b5a00c8dc5f25a52ff678
DAB_PR6 = 0a1a9c0cbb8dd8475c61a88142a778160acccfe9
DAB_BOOKREVIEW_COMMIT = 1b2d3b95e8a20aa66aee5ba8a2dd0d9b3e686f3e
DAB_URL = https://github.com/ucbepic/DataAgentBench.git
DAB_INIT = ( [ -d .git ] || (git init -q && git remote add origin $(DAB_URL) && git config extensions.partialClone origin) )
SHA256 = $(shell command -v sha256sum > /dev/null && echo sha256sum || echo shasum -a 256)
DAB_MODELS = gemini-2.5-flash gemini-3-pro gpt-5-mini gpt-5.1 gpt-5.2 kimi-k2-thinking

.PHONY: help setup test check-frozen dabstep-data dabstep dabstep-extra dabstep-recount dab-data dab \
        bench-data rescore fa-rescore fa-replay \
        receipts-select receipts-cost-check receipts-report

help:
	@echo "setup            install the Python environment (uv sync)"
	@echo "test             run all unit tests"
	@echo "check-frozen     check the SHA-256 hashes of the frozen checker files"
	@echo "dabstep-data     download DABstep at the pinned commit (about 5.6 GB)"
	@echo "dabstep          parse the DABstep traces and run the checker on every cell"
	@echo "dabstep-extra    recovery episodes, silent filters, syntax-error check, outcome regression, error types"
	@echo "dabstep-recount  independent heuristic recount of the DABstep corpus (missing-key classes)"
	@echo "dab-data         download the DataAgentBench logs, MongoDB dumps and bookreview re-run (about 2 GB)"
	@echo "dab              DataAgentBench replay: prevalence, checker, harness counts, MongoDB"
	@echo "bench-data       clone InfiAgent-DABench, QRData and DiscoveryBench at the pinned commits (about 540 MB)"
	@echo "rescore          score the recorded model runs again, without model calls"
	@echo "fa-rescore       score the recorded runs again with the answers that our harness lost recovered"
	@echo "fa-replay        replay the recorded runs with final_answer protected (Linux, bubblewrap; about 1 hour)"
	@echo "receipts-select  cut the 1,000 DABstep episodes of the receipt replay (no model calls)"
	@echo "receipts-cost-check  run the first 10 episodes with a model (needs OLLAMA_API_KEY; about USD 0.07)"
	@echo "receipts-report  report the receipt replay"

setup:
	uv sync --all-groups

test:
	cd dabstep-gate && uv run python -m unittest test_gate.py
	cd dataagentbench-replay && uv run python -m unittest test_dab.py
	cd rq3-pilot && DABSTEP_DATA=$(abspath $(DATA)) uv run python -m unittest test_rq3.py
	cd live && uv run python -m unittest test_fa_runner.py

check-frozen:
	$(SHA256) -c frozen/v6.sha256 frozen/v7.sha256 frozen/v8.sha256

dabstep-data:
	uv run python tools/download_dabstep.py --revision $(DABSTEP_COMMIT) --out $(DATA)

dabstep:
	mkdir -p $(RUN)
	uv run dabstep-gate/build_manifest.py --data $(DATA) --out $(RUN)/manifest.jsonl \
	  --scripts-commit "$$(git rev-parse HEAD 2>/dev/null || echo unknown)"
	uv run dabstep-gate/extract_xml_trace_cells.py $(DATA)/data/submissions $(RUN)/parsed_xml_cells.jsonl \
	  --manifest $(RUN)/manifest.jsonl
	uv run dabstep-gate/extract_smolagents_trace_cells.py $(DATA)/data/submissions $(RUN)/parsed_smolagents_cells.jsonl \
	  --manifest $(RUN)/manifest.jsonl
	uv run dabstep-gate/extract_generic_trace_cells.py $(DATA)/data/submissions $(RUN)/parsed_generic_cells.jsonl \
	  --manifest $(RUN)/manifest.jsonl --also "v1__ByteDance DataPlatform-LLM-data_agent__27-08-2025.jsonl"
	uv run dabstep-gate/evaluate_gate.py --gate-dir dabstep-gate --data $(DATA) --manifest $(RUN)/manifest.jsonl \
	  --cells $(RUN)/parsed_*_cells.jsonl --pred $(RUN)/gate_predictions.jsonl \
	  --lint --carry --files --tolerant auto | tee $(RUN)/checker_report.txt

dabstep-extra:
	uv run dabstep-gate/recovery_episodes.py --manifest $(RUN)/manifest.jsonl --cells $(RUN)/parsed_*_cells.jsonl \
	  --pred $(RUN)/gate_predictions.jsonl | tee $(RUN)/recovery_report.txt
	uv run dabstep-gate/silent_filter_check.py --data $(DATA) --manifest $(RUN)/manifest.jsonl \
	  --cells $(RUN)/parsed_*_cells.jsonl | tee $(RUN)/silent_filter_report.txt
	uv run dabstep-gate/dabstep_syntax_check.py --manifest $(RUN)/manifest.jsonl \
	  --cells $(RUN)/parsed_*_cells.jsonl | tee $(RUN)/syntax_check_report.txt
	uv run dabstep-gate/recovery_from_traces.py --data $(DATA) --manifest $(RUN)/manifest.jsonl \
	  --cells $(RUN)/parsed_*_cells.jsonl --out $(RUN)/recovery.csv | tee $(RUN)/outcome_report.txt
	uv run python tools/error_types.py $(RUN)/parsed_*_cells.jsonl | tee $(RUN)/error_types.txt
	uv run python tools/fresh_process.py --manifest $(RUN)/manifest.jsonl --cells $(RUN)/parsed_*_cells.jsonl \
	  | tee $(RUN)/fresh_process.txt

dabstep-recount:
	uv run dabstep-gate/dabstep_corpus_recount.py --skip-download --dir $(DATA) | tee $(RUN)/recount_report.txt

dab-data:
	mkdir -p $(DAB_LOGS) $(DAB_MONGO) $(DAB_BOOKREVIEW)
	cd $(DAB_LOGS) && $(DAB_INIT) && git fetch -q --depth 1 --filter=blob:none origin refs/pull/6/head \
	  && test "$$(git rev-parse FETCH_HEAD)" = "$(DAB_PR6)" \
	  && git sparse-checkout set --no-cone 'results-*/**/tool_calls.jsonl' && git checkout -q --detach $(DAB_PR6)
	cd $(DAB_LOGS) && for p in query_yelp/query_dataset/yelp_business/yelp_db/business.bson \
	  query_yelp/query_dataset/yelp_business/yelp_db/checkin.bson \
	  query_agnews/query_dataset/agnews_articles/articles_db/articles.bson \
	  query_civic_unstructured/query_dataset/civic_docs_dump/civic_db/civic_docs.bson; do \
	  git show $(DAB_PR6):$$p > $(abspath $(DAB_MONGO))/$$(basename $$p); done
	cd $(DAB_BOOKREVIEW) && $(DAB_INIT) && git fetch -q --depth 1 --filter=blob:none origin $(DAB_BOOKREVIEW_COMMIT) \
	  && git sparse-checkout set --no-cone 'results-*/query_bookreview/**/tool_calls.jsonl' \
	  && git checkout -q --detach $(DAB_BOOKREVIEW_COMMIT)

dab:
	mkdir -p $(DAB_RUN)
	for m in $(DAB_MODELS); do uv run dataagentbench-replay/extract_dab_cells.py --logs $(DAB_LOGS) --model $$m \
	  --out $(DAB_RUN)/cells_$$m.jsonl; done
	uv run dataagentbench-replay/dab_rq1.py $(DAB_RUN)/cells_*.jsonl --json $(DAB_RUN)/rq1.json | tee $(DAB_RUN)/rq1_report.txt
	for m in $(DAB_MODELS); do uv run dataagentbench-replay/dab_gate_eval.py --gate-dir dabstep-gate \
	  --cells $(DAB_RUN)/cells_$$m.jsonl --pred $(DAB_RUN)/gate_$$m.jsonl --lint > /dev/null; done
	uv run dataagentbench-replay/dab_gate_summary.py $(DAB_RUN) | tee $(DAB_RUN)/checker_report.txt
	cd $(DAB_LOGS) && uv run --project $(CURDIR) python $(CURDIR)/dataagentbench-replay/dab_harness_count.py \
	  | tee $(abspath $(DAB_RUN))/harness_report.txt
	cd $(DAB_LOGS) && uv run --project $(CURDIR) python $(CURDIR)/dataagentbench-replay/dab_harness_count.py \
	  --paper-matched --bookreview-rerun $(abspath $(DAB_BOOKREVIEW)) | tee $(abspath $(DAB_RUN))/harness_report_paper_matched.txt
	# The MongoDB count stops after a time budget and resumes from its cache, so run it until it is complete.
	for i in 1 2 3 4 5 6 7 8 9 10; do \
	  uv run --group dab dataagentbench-replay/dab_mongo.py --cells $(DAB_RUN)/cells_*.jsonl --dumps $(DAB_MONGO) \
	    --cache $(DAB_RUN)/mongo_counts.json --json $(DAB_RUN)/mongo.json > $(DAB_RUN)/mongo_report.txt; \
	  grep -q INCOMPLETE $(DAB_RUN)/mongo_report.txt || break; done
	grep "all models" $(DAB_RUN)/mongo_report.txt
	if grep -q INCOMPLETE $(DAB_RUN)/mongo_report.txt; then echo "MongoDB count still incomplete: run make dab again"; fi

bench-data:
	sh tools/get_benchmarks.sh

rescore:
	sh tools/rescore.sh

# Answers lost by our harness (preprint, Section 3.4). fa-rescore uses the released replays in
# results/fa-replay; fa-replay makes them again. SANDBOX_PY is a Python with requirements/sandbox.txt.
FA_REPLAYS ?= results/fa-replay
SANDBOX_PY ?= sandbox-env/bin/python
BENCH ?= data

fa-rescore:
	[ -d recorded ] || tar xzf recorded-runs-2026-09.tar.gz
	mkdir -p runs
	uv run live/fa_rescore.py --rec recorded --replays $(FA_REPLAYS) --out runs/fa_recorded | tee runs/fa_rescore_summary.txt
	BENCH=$(BENCH) REC=runs/fa_recorded OUT=runs/rescore_fa sh tools/rescore.sh

fa-replay:
	[ -d recorded ] || tar xzf recorded-runs-2026-09.tar.gz
	uv run python tools/make_kb.py discoverybench --data $(BENCH)/discoverybench/discoverybench/real --out runs/fa_db_kb
	$(SANDBOX_PY) live/fa_replay.py --tag live_gpt --results recorded/e1/results.jsonl \
	  --data $(BENCH)/QRData/benchmark/data --out runs/fa-replay/live_gpt.jsonl
	$(SANDBOX_PY) live/fa_replay.py --tag live_ds --results recorded/e1/results_deepseek41.jsonl \
	  --data $(BENCH)/QRData/benchmark/data --out runs/fa-replay/live_ds.jsonl
	$(SANDBOX_PY) live/fa_replay.py --tag ia_gpt --results recorded/ia/results_gpt-oss.jsonl \
	  --data $(BENCH)/InfiAgent/examples/DA-Agent/data/da-dev-tables --out runs/fa-replay/ia_gpt.jsonl
	$(SANDBOX_PY) live/fa_replay.py --tag ia_ds --results recorded/ia/results_deepseek41.jsonl \
	  --data $(BENCH)/InfiAgent/examples/DA-Agent/data/da-dev-tables --out runs/fa-replay/ia_ds.jsonl
	$(SANDBOX_PY) live/fa_replay.py --tag qr_gpt --results recorded/qr/results_gpt-oss.jsonl \
	  --data $(BENCH)/QRData/benchmark/data --out runs/fa-replay/qr_gpt.jsonl
	$(SANDBOX_PY) live/fa_replay.py --tag db_gpt --results recorded/db/results_gpt-oss.jsonl \
	  --data-per-domain runs/fa_db_kb/data --out runs/fa-replay/db_gpt.jsonl
	@echo "Now run: make fa-rescore FA_REPLAYS=runs/fa-replay"

receipts-select:
	mkdir -p $(RQ3_RUN)/gate-v6
	cp dabstep-gate/v6/gate.py dabstep-gate/gate_adapter.py $(RQ3_RUN)/gate-v6/
	uv run rq3-pilot/select_prefixes.py --gate-dir $(RQ3_RUN)/gate-v6 --data $(DATA) --manifest $(RUN)/manifest.jsonl \
	  --cells $(RUN)/parsed_*_cells.jsonl --out $(RQ3_RUN)/episodes.jsonl \
	  --n 5000 --seed 3 --max-share 1.0 --max-prefix 12 --exclude rq3-pilot/excluded_pairs.jsonl
	$(SHA256) $(RQ3_RUN)/episodes.jsonl

receipts-cost-check:
	uv run rq3-pilot/rq3_run.py --episodes $(RQ3_RUN)/episodes.jsonl --data $(DATA) --limit 10 \
	  --out $(RQ3_RUN)/results_cost10.jsonl --model $(RQ3_MODEL) --seeds 1 --max-seconds 150 \
	  --transcripts $(RQ3_RUN)/transcripts_cost10

receipts-report:
	uv run rq3-pilot/rq3_report.py $(RQ3_RUN)/results_cost10.jsonl
