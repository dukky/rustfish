#!/usr/bin/env python3
"""Pre-release UCI probe for rustfish (CCRL readiness).

Runs the protocol surface a CCRL tester's GUI (Cute Chess, Arena, Banksia)
exercises against a built binary and prints PASS/FAIL for each check. Stdlib
only, so it runs anywhere Python 3 does.

    python3 scripts/ccrl_check.py ./target/release/rustfish
    python3 scripts/ccrl_check.py rustfish-v1.0.0-windows-x86-64-v3.exe
"""
import subprocess
import sys
import threading
import time

BIN = sys.argv[1] if len(sys.argv) > 1 else "./target/release/rustfish"
EXPECTED_NAME = "Rustfish 1.0.0"
EXPECTED_AUTHOR = "dukky"


class Engine:
    def __init__(self):
        self.p = subprocess.Popen(
            [BIN],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.p.stdin is not None and self.p.stdout is not None
        self.lines = []
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\n"))

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait_for(self, prefix, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                for line in self.lines:
                    if line.startswith(prefix):
                        return line
            time.sleep(0.01)
        return None

    def drain(self):
        with self.lock:
            out = list(self.lines)
            self.lines.clear()
        return out

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + (("  (" + detail + ")") if detail else ""))


def main():
    e = Engine()
    e.send("uci")
    if e.wait_for("uciok", 10) is None:
        check("uci -> uciok", False)
        e.close()
        return 1
    lines = e.drain()
    id_lines = [l for l in lines if l.startswith("id ")]
    options = [l for l in lines if l.startswith("option name ")]
    name = next((l[8:] for l in id_lines if l.startswith("id name ")), "")
    author = next((l[10:] for l in id_lines if l.startswith("id author ")), "")
    ownbook = next((l for l in options if "name OwnBook " in l), "")
    threads = next((l for l in options if "name Threads " in l), "")
    hash_opt = next((l for l in options if "name Hash " in l), "")

    check("uci -> uciok", True)
    check("id name is the release version", name == EXPECTED_NAME, name or "missing")
    check("id author is set", author == EXPECTED_AUTHOR, author or "missing")
    check("Hash option advertised", bool(hash_opt), hash_opt)
    check("Threads option advertised, default 1", "default 1" in threads, threads or "missing")
    check("OwnBook advertises default false", "default false" in ownbook, ownbook or "missing")

    e.send("isready")
    check("isready -> readyok", e.wait_for("readyok", 5) is not None)

    e.send("ucinewgame")
    e.send("isready")
    e.wait_for("readyok", 5)
    e.send("position startpos moves e2e4 e7e5 g1f3")
    e.send("go depth 6")
    check("position moves + go depth 6", e.wait_for("bestmove", 15) is not None)
    e.drain()

    e.send("setoption name Hash value 256")
    e.send("setoption name Clear Hash")
    e.send("isready")
    check("setoption Hash + Clear Hash", e.wait_for("readyok", 5) is not None)
    e.drain()

    start = time.time()
    e.send("position startpos")
    e.send("go movetime 300")
    best = e.wait_for("bestmove", 5)
    elapsed = time.time() - start
    check("go movetime 300 searches (0.1-1.0 s)", best is not None and 0.1 <= elapsed <= 1.0,
          "%.2fs" % elapsed)
    e.drain()

    start = time.time()
    e.send("position startpos")
    e.send("go wtime 120000 btime 120000 winc 1000 binc 1000 movestogo 40")
    check("clock + movestogo returns a move", e.wait_for("bestmove", 15) is not None,
          "%.2fs" % (time.time() - start))
    e.drain()

    e.send("position startpos")
    e.send("go infinite")
    time.sleep(0.5)
    start = time.time()
    e.send("stop")
    check("go infinite + stop -> bestmove", e.wait_for("bestmove", 5) is not None,
          "%.2fs" % (time.time() - start))
    e.drain()

    e.send("position startpos")
    e.send("go infinite")
    time.sleep(0.2)
    e.send("isready")
    check("isready during search", e.wait_for("readyok", 3) is not None)
    e.send("stop")
    e.wait_for("bestmove", 5)
    e.drain()
    e.close()

    benches = []
    for _ in range(2):
        proc = subprocess.run([BIN], input="bench\nquit\n", text=True,
                              capture_output=True, timeout=300)
        benches.append(proc.stdout)
    nodes = []
    for out in benches:
        for line in out.splitlines():
            if line.startswith("Nodes searched"):
                nodes.append(line.split(":", 1)[1].strip())
    check("bench deterministic across two runs", len(nodes) == 2 and nodes[0] == nodes[1], str(nodes))
    noisy = [l for l in benches[0].splitlines()
             if l and not l.startswith(("info ", "bestmove ", "Nodes searched", "Nodes/second",
                                        "Evals hashed", "Eval hash", "bench", "Total time"))]
    check("bench stdout is protocol-clean", not noisy, "; ".join(noisy[:3]))

    failed = len(results) - sum(results)
    print("\n%d checks, %d failed" % (len(results), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
