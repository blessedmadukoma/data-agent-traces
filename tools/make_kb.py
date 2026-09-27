"""Build the data folder that kramabench/kb_gate.py reads, without running a model.

The run scripts build this folder next to their output. Use this script to score
recorded runs that you did not run yourself.

Usage:
  uv run python tools/make_kb.py infiagent --data data/InfiAgent/examples/DA-Agent/data/da-dev-tables --out runs/ia/ia_kb
  uv run python tools/make_kb.py qrdata --data data/QRData/benchmark/data --out runs/qr/qr_kb
  uv run python tools/make_kb.py discoverybench --data data/discoverybench/discoverybench/real --out runs/db/db_kb
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def link_tables(tables, out, domain):
    folder = os.path.join(out, "data", domain)
    os.makedirs(folder, exist_ok=True)
    link = os.path.join(folder, "input")
    if not os.path.islink(link):
        os.symlink(os.path.abspath(tables), link)
    return link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["infiagent", "qrdata", "discoverybench"])
    ap.add_argument("--data", required=True, help="the benchmark's data folder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not os.path.isdir(a.data):
        sys.exit(f"Data folder not found: {a.data}")
    if a.benchmark in ("infiagent", "qrdata"):
        link = link_tables(a.data, a.out, a.benchmark)
        print(f"linked {link} -> {os.path.abspath(a.data)}")
        return
    sys.path.insert(0, os.path.join(HERE, "..", "discoverybench"))
    import db_run  # noqa: E402

    tasks = db_run.load_tasks(a.data, a.out)
    domains = {t["domain"] for t in tasks}
    print(f"copied the data files of {len(tasks)} tasks in {len(domains)} folders to {a.out}/data")


if __name__ == "__main__":
    main()
