"""A persistent Python executor for one episode, inside a sandbox.

Executed code comes from other people's agent traces and from the model
under test, so it runs in an isolated process:

  * bwrap (default, Linux): no network, a private /tmp, read-only system
    directories and Python environment, and one writable work directory.
    The repository and the rest of the home directory are not visible.
  * local: a plain subprocess in the work directory. Unsafe; only for
    unit tests with known code.

The work directory holds the DABstep context files in data/context/.
extra_paths maps other paths that the prefix code reads (absolute, or
relative to the work directory) to context files, so that recorded code
finds its files.
"""
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _python_roots():
    """Directories the interpreter and its packages need (read-only in the sandbox):
    the virtual environment and the base interpreter's installation. Nothing
    under ~/mnt (connected folders) is bound."""
    mnt = os.path.join(os.path.expanduser("~"), "mnt")
    roots = {os.path.abspath(sys.prefix), os.path.abspath(sys.base_prefix),
             os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))}
    p = os.path.abspath(sys.executable)
    while os.path.islink(p):                  # uv links python -> cpython-3.12-... -> cpython-3.12.14-...
        t = os.path.join(os.path.dirname(p), os.readlink(p))
        roots.add(os.path.dirname(os.path.dirname(os.path.abspath(t))))
        p = os.path.abspath(t)
    return sorted(r for r in roots if r and r != "/" and not (r + os.sep).startswith(mnt + os.sep))


class Executor:
    def __init__(self, context_dir, extra_paths=None, mode="bwrap", timeout=60):
        self.timeout, self.mode = timeout, mode
        self.work = tempfile.mkdtemp(prefix="rq3-", dir=os.environ.get("RQ3_SCRATCH"))
        ctx = os.path.join(self.work, "data", "context")
        os.makedirs(ctx)
        for fn in os.listdir(context_dir):
            shutil.copy2(os.path.join(context_dir, fn), ctx)
        shutil.copy2(os.path.join(HERE, "runner.py"), os.path.join(self.work, ".runner.py"))
        binds = []
        for target, src_name in (extra_paths or {}).items():
            src = os.path.join(ctx, src_name)
            if not os.path.exists(src):
                continue
            if os.path.isabs(target):
                binds.append((src, target))
            else:
                dst = os.path.normpath(os.path.join(self.work, target))
                if dst.startswith(self.work + os.sep) and not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                elif not dst.startswith(self.work + os.sep):
                    binds.append((src, "/work" + dst[len(self.work):] if dst.startswith(self.work) else
                                  os.path.normpath(os.path.join("/work", target))))
        self.binds = binds
        self.proc = self._start()

    def _start(self):
        py = os.path.abspath(sys.executable)      # the venv's python, so that its packages load
        if self.mode == "local":
            cmd = [py, "-u", ".runner.py"]
            return subprocess.Popen(cmd, cwd=self.work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=self._err(), text=True, bufsize=1)
        if self.mode != "bwrap":
            raise ValueError(f"unknown sandbox mode {self.mode}")
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for d in ("/usr", "/bin", "/lib", "/lib64", "/sbin", "/etc"):
            if os.path.exists(d):
                cmd += ["--ro-bind", d, d]
        for r in _python_roots():
            cmd += ["--ro-bind", r, r]
        cmd += ["--bind", self.work, "/work", "--chdir", "/work", "--setenv", "HOME", "/work",
                "--setenv", "MPLBACKEND", "Agg"]
        for src, dst in self.binds:
            if not dst.startswith(("/usr/", "/bin/", "/lib", "/etc/", "/proc", "/dev", "/work/.runner")):
                cmd += ["--ro-bind", src, dst]
        cmd += [py, "-u", "/work/.runner.py"]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err(),
                                text=True, bufsize=1)

    def _err(self):
        self.err_path = os.path.join(self.work, ".executor-stderr.txt")
        return open(self.err_path, "w")

    def _lost(self, why):
        try:
            tail = open(self.err_path).read()[-2000:]
        except OSError:
            tail = ""
        return {"ok": False, "error_type": "ExecutorLost", "stdout": "", "final_answer": None, "lost": True,
                "error": why + (("\n" + tail) if tail else "")}

    def run(self, code):
        """Run one cell. Returns the runner's result, or a timeout / crash result."""
        if self.proc.poll() is not None:
            return self._lost("the executor process has ended")
        self.proc.stdin.write(json.dumps({"code": code}) + "\n")
        self.proc.stdin.flush()
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            r, _, _ = select.select([self.proc.stdout], [], [], 0.5)
            if r:
                line = self.proc.stdout.readline()
                if not line:
                    return self._lost("the executor process ended during the cell")
                return json.loads(line)
        self.close()
        return {"ok": False, "error_type": "TimeoutError", "stdout": "", "final_answer": None, "lost": True,
                "error": f"TimeoutError: the cell did not finish within {self.timeout} s"}

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        shutil.rmtree(self.work, ignore_errors=True)
