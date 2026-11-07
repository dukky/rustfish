#!/usr/bin/env python3
"""Report what is in flight on the laptop to the rustfish dashboard.

Read-only: inspects processes and the logs in ~/rustfish-data, then POSTs a
small JSON payload to https://rustfish-dash.dev.duk.im/api/laptop with
`Authorization: Bearer <token>` (env LAPTOP_TOKEN, else
~/.config/rustfish/laptop-token). `--print` prints the payload instead.
Stdlib only; meant to run from launchd every 2 minutes.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

DATA = os.path.expanduser("~/rustfish-data")
URL = "https://rustfish-dash.dev.duk.im/api/laptop"
TOKEN_FILE = os.path.expanduser("~/.config/rustfish/laptop-token")
MAX_LINE = 200


def ps():
    """[(pid, stat, etime, command)] for every process."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,stat=,etime=,command="],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            rows.append((int(parts[0]), parts[1], parts[2], parts[3]))
    return rows


def tail(path, n=1, max_bytes=65536):
    """Last n non-empty lines of a file ([] if missing)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [l.strip() for l in data.splitlines() if l.strip()]
    return lines[-n:]


def trim(s, n=MAX_LINE):
    s = (s or "").replace(os.path.expanduser("~"), "~")
    return s if len(s) <= n else s[: n - 1] + "…"


HEADER_RE = re.compile(r"^train .*?epochs=(\d+) x (\d+)")
EPOCH_RE = re.compile(r"^epoch (\d+)\b.*?\bval ([0-9.]+)(?: \(quantized ([0-9.]+)\))?.*?\b(\d+) pos/s")


def parse_train_log(text):
    """Digest a trainer log: epochs, positions/epoch, last epoch, val, pos/s, ETA."""
    r = {"epochs": None, "epoch": 0, "positions": None, "val": None, "val_q": None,
         "pos_s": None, "eta_s": None, "done": False, "keep_best": False}
    for line in text.splitlines():
        line = line.strip()
        m = HEADER_RE.match(line)
        if m:
            r["epochs"], r["positions"] = int(m.group(1)), int(m.group(2))
            continue
        m = EPOCH_RE.match(line)
        if m:
            r["epoch"] = int(m.group(1))
            r["val"] = float(m.group(2))
            r["val_q"] = float(m.group(3)) if m.group(3) else None
            r["pos_s"] = int(m.group(4))
            continue
        if line.startswith("keep-best"):
            r["keep_best"] = True
        if line.startswith("wrote"):
            r["done"] = True
    if r["epochs"] and r["positions"] and r["pos_s"] and not r["done"]:
        r["eta_s"] = int((r["epochs"] - r["epoch"]) * r["positions"] / r["pos_s"])
    return r


def arg_of(cmd, flag):
    m = re.search(r"(?:^|\s)" + re.escape(flag) + r"\s+(\S+)", cmd)
    return m.group(1) if m else None


def trainers(procs):
    out = []
    for pid, stat, etime, cmd in procs:
        if "nnue-trainer" not in cmd or " train" not in cmd:
            continue
        name = arg_of(cmd, "--out")
        if not name:
            continue
        name = os.path.basename(name)
        stem = name[:-4] if name.endswith(".bin") else name
        text = ""
        try:
            with open(os.path.join(DATA, stem + ".log"), "r", errors="replace") as fh:
                text = fh.read()[-200000:]
        except OSError:
            pass
        t = parse_train_log(text)
        t.update({"net": stem, "kind": "metal" if "metal" in cmd else "cpu",
                  "elapsed": etime, "paused": stat.startswith("T")})
        out.append(t)
    return out


def chains(procs):
    out = []
    for pid, stat, etime, cmd in procs:
        m = re.search(r"(chain\d*)\.sh\b", cmd)
        if m and ("zsh" in cmd or "sh " in cmd or cmd.startswith("./")):
            name = m.group(1)
            last = tail(os.path.join(DATA, name + ".log"))
            out.append({"name": name, "elapsed": etime, "last": trim(last[0] if last else "")})
    return out


def datagen(procs):
    gate_line = tail(os.path.join(DATA, "datagen-gate.log"))
    gate_running = any("datagen-gate.sh" in c for _, _, _, c in procs)
    engines = [(s, e, c) for _, s, e, c in procs if re.search(r"(^|\s|/)\.\./bin/rustfish-\S+", c)]
    r = {"gate_running": gate_running, "gate_last": trim(gate_line[0] if gate_line else ""),
         "running": bool(engines), "state": None, "elapsed": None, "file": None, "bytes": None}
    if engines:
        stat, etime, _ = engines[0]
        r["state"] = "paused" if stat.startswith("T") else "running"
        r["elapsed"] = etime
    files = []
    try:
        for f in os.listdir(os.path.join(DATA, "local")):
            if re.fullmatch(r"L\d+\.txt", f):
                p = os.path.join(DATA, "local", f)
                files.append((os.path.getmtime(p), f, os.path.getsize(p)))
    except OSError:
        pass
    if files:
        mtime, f, size = max(files)
        r.update(file=f, bytes=size, modified_s=int(time.time() - mtime))
    return r


def netloop(procs):
    running = any("netloop.py" in c and " auto" in c for _, _, _, c in procs)
    last = tail(os.path.join(DATA, "netloop-auto.log"))
    return {"running": running, "last": trim(last[0] if last else "")}


def caffeinate(procs):
    out = []
    for pid, stat, etime, cmd in procs:
        if cmd.startswith("caffeinate"):
            t = arg_of(cmd, "-t")
            el = etime_s(etime)
            rem = int(t) - el if t and t.isdigit() and el is not None else None
            out.append({"args": trim(cmd[len("caffeinate"):].strip(), 60), "remaining_s": rem})
    return out


def etime_s(s):
    """ps etime ([[dd-]hh:]mm:ss) -> seconds."""
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(x) for x in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError:
        return None


def collect():
    procs = [p for p in ps() if p[0] != os.getpid()]
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = None
    try:
        free = shutil.disk_usage(os.path.expanduser("~")).free
    except OSError:
        free = None
    return {
        "ts": int(time.time()),
        "trainers": trainers(procs),
        "chains": chains(procs),
        "datagen": datagen(procs),
        "netloop": netloop(procs),
        "caffeinate": caffeinate(procs),
        "load": load,
        "disk_free_gb": round(free / 1e9, 1) if free is not None else None,
    }


def token():
    t = os.environ.get("LAPTOP_TOKEN", "").strip()
    if not t:
        try:
            with open(TOKEN_FILE) as fh:
                t = fh.read().strip()
        except OSError:
            pass
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print", action="store_true", help="print the payload instead of posting")
    ap.add_argument("--url", default=os.environ.get("LAPTOP_STATUS_URL", URL))
    args = ap.parse_args()
    body = json.dumps(collect(), separators=(",", ":")).encode()
    if args.print:
        print(json.dumps(json.loads(body), indent=2))
        print(f"({len(body)} bytes)", file=sys.stderr)
        return 0
    tok = token()
    if not tok:
        print("no token: set LAPTOP_TOKEN or write ~/.config/rustfish/laptop-token", file=sys.stderr)
        return 2
    req = urllib.request.Request(args.url, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {tok}",
        "User-Agent": "rustfish-laptop-status"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except Exception as exc:
        print(f"post failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
