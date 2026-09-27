"""Download the DABstep dataset at a pinned commit and check it.

Usage: uv run python tools/download_dabstep.py --revision <commit> --out data/dabstep
Set HF_TOKEN to a Hugging Face read token for faster downloads.
"""
import argparse
import hashlib
import pathlib
import sys

from huggingface_hub import snapshot_download

PAYMENTS_SHA256 = "5fbb26210a45427d7a6560cfab3a362a08e4067f27cd03695f211a51c47ffc25"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    snapshot_download(
        "adyen/DABstep", repo_type="dataset", revision=a.revision, local_dir=a.out,
        allow_patterns=["data/submissions/*", "data/task_scores/*", "data/context/*", "data/tasks/*"],
    )
    root = pathlib.Path(a.out) / "data"
    pointers = [p for p in root.rglob("*") if p.is_file() and p.stat().st_size < 200
                and b"git-lfs.github.com/spec" in p.read_bytes()]
    if pointers:
        sys.exit(f"{len(pointers)} files are Git LFS pointers, not data. Run the download again.")
    if sha256(root / "context" / "payments.csv") != PAYMENTS_SHA256:
        sys.exit("payments.csv does not match the expected SHA-256. Check the revision.")
    n = len(list((root / "submissions").iterdir()))
    print(f"OK: {a.out} at {a.revision}, {n} submission files")


if __name__ == "__main__":
    main()
