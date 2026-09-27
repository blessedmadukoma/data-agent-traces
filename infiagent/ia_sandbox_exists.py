#!/usr/bin/env python3
"""Supplementary analysis (not pre-registered, added after the InfiAgent-DABench run):
kb_gate.py with a file-existence model that also knows the sandbox's mount table.

The frozen replay model (kb_gate.make_exists) answers "unknown" for an absolute
path outside /work and /tmp. The bubblewrap sandbox binds only /usr, /bin, /lib,
/lib64, /sbin, /etc, /proc, /dev, /tmp, /work and the Python roots, so a path whose
first component is none of these, and is not a parent of a Python root (/home),
cannot exist inside it. This is what a live os.path.exists check in the sandbox
would return. The gate itself is unchanged (frozen v7).

Usage: python3 infiagent/ia_sandbox_exists.py --gate-dir dabstep-gate/v7 --kb runs/ia/ia_kb \
    --results runs/ia/results_gpt-oss.jsonl --out runs/ia/gate_v7_sandbox_exists.jsonl
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kramabench"))
import kb_gate  # noqa: E402

MOUNTED = {"usr", "bin", "lib", "lib64", "sbin", "etc", "proc", "dev", "tmp", "work", "home"}
frozen_make_exists = kb_gate.make_exists


def make_exists(lake, workspace, cwd_known):
    inner = frozen_make_exists(lake, workspace, cwd_known)

    def exists(path):
        r = inner(path)
        p = str(path)
        if r is None and os.path.isabs(p):
            top = os.path.normpath(p).split("/")[1] if p != "/" else ""
            if top and top not in MOUNTED:
                return False
        return r
    return exists


if __name__ == "__main__":
    kb_gate.make_exists = make_exists
    kb_gate.main()
