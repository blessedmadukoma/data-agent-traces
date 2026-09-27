#!/usr/bin/env python3
"""Replay scoring with the exact sandbox mount table (evidence model E-M, plan
2026-09-24j, section 1). Secondary scoring model; the gate is unchanged.

kb_gate.make_exists (frozen) answers "unknown" for an absolute path outside /work
and /tmp. The bubblewrap sandbox of kb_run.KBExecutor binds only /usr, /bin, /lib,
/lib64, /sbin and /etc read-only, mounts /proc, /dev and a private /tmp, binds the
work directory at /work, and binds the Python roots of the run. So:

  * a path under a bind is unknown (its contents are not recorded);
  * a proper parent of a bind ("/", "/home", "/home/claude") exists;
  * every other absolute path does not exist.

This replaces the coarse model of ia_sandbox_exists.py, which treated all of /home
as unknown.

Usage: python3 infiagent/mount_exists.py --roots /home/claude/iaenv /usr -- \
    --gate-dir dabstep-gate --kb runs/qr/qr_kb --results runs/qr/results_gpt-oss.jsonl --out ...
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kramabench"))
import kb_gate  # noqa: E402

FIXED = ["/usr", "/bin", "/lib", "/lib64", "/sbin", "/etc", "/proc", "/dev", "/tmp", "/work"]
frozen_make_exists = kb_gate.make_exists


def mount_model(roots):
    binds = sorted({os.path.normpath(r) for r in FIXED + list(roots)})

    def make_exists(lake, workspace, cwd_known):
        inner = frozen_make_exists(lake, workspace, cwd_known)

        def exists(path):
            r = inner(path)
            p = str(path)
            if r is not None or not os.path.isabs(p):
                return r
            n = os.path.normpath(p)
            if any(n == b or n.startswith(b + "/") for b in binds):
                return None
            if n == "/" or any(b.startswith(n.rstrip("/") + "/") for b in binds):
                return True
            return False
        return exists
    return make_exists


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--roots" not in argv or "--" not in argv:
        raise SystemExit(__doc__)
    i, j = argv.index("--roots"), argv.index("--")
    roots = argv[i + 1:j]
    sys.argv = [sys.argv[0]] + argv[j + 1:]
    kb_gate.make_exists = mount_model(roots)
    kb_gate.main()
