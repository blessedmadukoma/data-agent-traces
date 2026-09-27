"""Count failed DABstep cells by error type, from the parsed cell files.

Usage: uv run python tools/error_types.py runs/dabstep/parsed_*_cells.jsonl
"""
import collections
import json
import sys


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit("Give the parsed cell files, for example runs/dabstep/parsed_*_cells.jsonl")
    counts = collections.Counter()
    failed = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                cell = json.loads(line)
                if cell.get("failed"):
                    failed += 1
                    counts[cell.get("error_type")] += 1
    print(f"failed cells: {failed}")
    for error_type, n in counts.most_common():
        print(f"  {str(error_type):28} {n:6}  {n / failed:6.1%}")


if __name__ == "__main__":
    main()
