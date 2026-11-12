#!/usr/bin/env python3
"""Push a training machine's status to the rustfish dashboard.

Read-only: reads the trainer and datagen logs you point it at, then POSTs a
small JSON payload (schema "machine-status/1", well under 16 KB) to
https://rustfish-dash.dev.duk.im/api/machine with
`Authorization: Bearer <token>` (env MACHINE_TOKEN, else --token-file,
default ~/.config/rustfish/machine-token). Stdlib only; meant to run from
cron every few minutes, e.g.

  */5 * * * * python3 ~/rustfish/scripts/machine_status.py --machine desktop \\
      --gpu-log '~/rustfish-data/*.log' --datagen-log '~/rustfish-data/D*.log' \\
      --disk ~/rustfish-data

--print shows the payload without posting.

Logs: --gpu-log takes nnue-trainer logs (CPU, CUDA and Metal print the same
`train H=...` header and one `epoch k ...` line per epoch); --datagen-log
takes the stderr of `rustfish datagen` (`datagen: games a..b seed s ...`
header and periodic `datagen: games d/t positions p ...` lines). Each flag
takes a path or a glob and may repeat; the most recently modified matching
log is the current job, and one untouched for longer than --gpu-idle /
--datagen-idle is reported as idle instead.
"""
import argparse
import glob
import json
import os
import re
import shutil
import socket
import sys
import time
import urllib.request

URL = "https://rustfish-dash.dev.duk.im/api/machine"
TOKEN_FILE = "~/.config/rustfish/machine-token"
SCHEMA = "machine-status/1"
MAX_TAIL = 400_000  # bytes of each log read (the latest stage is near the end)

TRAIN_RE = re.compile(r"^train H=(\d+) buckets=(\d+)(.*?) params=\d+ epochs=(\d+) x (\d+)")
EPOCH_RE = re.compile(r"^epoch (\d+)\b.*?\bval ([0-9.]+)(?: \(quantized ([0-9.]+)\))?.*?\b(\d+) pos/s")
KB_RE = re.compile(r"king-buckets=(\S+)")
THREATS_RE = re.compile(r"threats[= ](\S+)")
DG_HEAD_RE = re.compile(r"^datagen: games (\d+)\.\.(\d+) seed (\d+) nodes (\d+).*?->\s*(\S+)")
DG_PROG_RE = re.compile(r"^datagen( done)?: games (\d+)/(\d+) positions (\d+)\b.*?([0-9.]+) games/s ([0-9.]+) pos/s")


def read_tail(path, max_bytes=MAX_TAIL):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def short_path(p):
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if p.startswith(home) else p


def newest(patterns, match=None):
    """(path, mtime) of the most recently modified file matching any pattern
    whose tail passes `match` (a text -> bool), or (None, None)."""
    files = set()
    for pat in patterns or []:
        files.update(f for f in glob.glob(os.path.expanduser(pat)) if os.path.isfile(f))
    best = (None, None)
    for f in sorted(files, key=os.path.getmtime, reverse=True):
        if match is None or match(read_tail(f)):
            return f, os.path.getmtime(f)
    return best


def parse_train_log(text, mtime=None, now=None):
    """The last training stage in an nnue-trainer log.

    Returns epoch k/N, val / quantized val of the last epoch, pos/s, the shape
    from the `train` header, overall progress (epochs done plus the part of the
    current epoch implied by the time since the log was last written, which
    happens once per epoch) and ETA. None if the text has no `train` header.
    """
    r = None
    for line in text.splitlines():
        line = line.strip()
        m = TRAIN_RE.match(line)
        if m:  # a new stage (a chain log can hold several): start over
            rest = m.group(3)
            kb = KB_RE.search(rest)
            th = THREATS_RE.search(rest)
            r = {"hidden": int(m.group(1)), "buckets": int(m.group(2)),
                 "king_buckets": kb.group(1) if kb else None, "threats": th.group(1) if th else None,
                 "epochs": int(m.group(4)), "positions": int(m.group(5)), "epoch": 0,
                 "val": None, "val_q": None, "pos_s": None, "done": False, "keep_best": None, "out": None}
            continue
        if r is None:
            continue
        m = EPOCH_RE.match(line)
        if m:
            r.update(epoch=int(m.group(1)), val=float(m.group(2)),
                     val_q=float(m.group(3)) if m.group(3) else None, pos_s=int(m.group(4)))
            continue
        m = re.match(r"^keep-best: epoch (\d+)", line)
        if m:
            r["keep_best"] = int(m.group(1))
        m = re.match(r"^wrote (\S+)", line)
        if m:
            r.update(done=True, out=os.path.basename(m.group(1)))
    if r is None:
        return None
    now = now or time.time()
    progress = r["epoch"] / r["epochs"] if r["epochs"] else None
    eta = None
    if not r["done"] and r["pos_s"] and r["positions"] and r["epochs"]:
        since = max(0.0, now - mtime) if mtime else 0.0
        part = min(0.99, since * r["pos_s"] / r["positions"])
        left = r["epochs"] - r["epoch"] - part
        progress = (r["epoch"] + part) / r["epochs"]
        eta = int(max(0.0, left) * r["positions"] / r["pos_s"])
    if r["done"]:
        progress = 1.0
    r.update(progress=round(progress, 4) if progress is not None else None, eta_s=eta)
    return r


def parse_datagen_log(text):
    """The last run in a `rustfish datagen` stderr log: seed, games done/total,
    positions, rates and ETA. None if there is no datagen line."""
    r = None
    for line in text.splitlines():
        line = line.strip()
        m = DG_HEAD_RE.match(line)
        if m:
            first, end = int(m.group(1)), int(m.group(2))
            r = {"seed": int(m.group(3)), "nodes": int(m.group(4)), "out": os.path.basename(m.group(5)),
                 "games_done": 0, "games_total": end - first, "positions": 0,
                 "games_s": None, "pos_s": None, "done": False}
            continue
        m = DG_PROG_RE.match(line)
        if m:
            if r is None:
                r = {"seed": None, "nodes": None, "out": None}
            r.update(done=bool(m.group(1)), games_done=int(m.group(2)), games_total=int(m.group(3)),
                     positions=int(m.group(4)), games_s=float(m.group(5)), pos_s=float(m.group(6)))
    if r is None:
        return None
    left = (r.get("games_total") or 0) - (r.get("games_done") or 0)
    r["eta_s"] = int(left / r["games_s"]) if r.get("games_s") and not r["done"] and left > 0 else None
    return r


def gpu_job(patterns, idle_s, now):
    path, mtime = newest(patterns, lambda t: "\ntrain H=" in "\n" + t)
    if not path:
        return None
    r = parse_train_log(read_tail(path), mtime, now)
    if r is None:
        return None
    r.update(name=re.sub(r"\.log$", "", os.path.basename(path)), log=short_path(path),
             updated_s=int(now - mtime), idle=bool(not r["done"] and now - mtime > idle_s))
    return r


def datagen_job(patterns, idle_s, now):
    path, mtime = newest(patterns, lambda t: "datagen" in t)
    if not path:
        return None
    r = parse_datagen_log(read_tail(path))
    if r is None:
        return None
    r.update(name=re.sub(r"\.(txt|bin)$", "", r.get("out") or "") or re.sub(r"\.log$", "", os.path.basename(path)),
             log=short_path(path), updated_s=int(now - mtime),
             idle=bool(not r["done"] and now - mtime > idle_s))
    return r


def disks(paths):
    out = []
    for p in paths or ["~"]:
        try:
            u = shutil.disk_usage(os.path.expanduser(p))
            out.append({"path": p, "free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1)})
        except OSError:
            out.append({"path": p, "free_gb": None, "total_gb": None})
    return out


def collect(args, now=None):
    now = now or time.time()
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        load = None
    return {"schema": SCHEMA, "machine": args.machine, "ts": int(now),
            "gpu": gpu_job(args.gpu_log, args.gpu_idle, now),
            "datagen": datagen_job(args.datagen_log, args.datagen_idle, now),
            "disk": disks(args.disk), "load": load}


def token(path):
    t = os.environ.get("MACHINE_TOKEN", "").strip()
    if not t:
        try:
            with open(os.path.expanduser(path)) as fh:
                t = fh.read().strip()
        except OSError:
            pass
    return t


def parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--machine", default=socket.gethostname().split(".")[0][:40] or "machine",
                    help="name shown on the dashboard (default: short hostname)")
    ap.add_argument("--gpu-log", action="append", metavar="PATH_OR_GLOB", help="nnue-trainer log(s)")
    ap.add_argument("--datagen-log", action="append", metavar="PATH_OR_GLOB", help="rustfish datagen stderr log(s)")
    ap.add_argument("--disk", action="append", metavar="PATH", help="report free space here (default ~)")
    ap.add_argument("--gpu-idle", type=int, default=3 * 3600,
                    help="seconds without a log write before the GPU job counts as idle (an epoch can take hours)")
    ap.add_argument("--datagen-idle", type=int, default=1800,
                    help="seconds without a log write before datagen counts as idle")
    ap.add_argument("--url", default=os.environ.get("MACHINE_STATUS_URL", URL))
    ap.add_argument("--token-file", default=TOKEN_FILE)
    ap.add_argument("--print", action="store_true", help="print the payload instead of posting")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    body = json.dumps(collect(args), separators=(",", ":")).encode()
    if args.print:
        print(json.dumps(json.loads(body), indent=2))
        print(f"({len(body)} bytes)", file=sys.stderr)
        return 0
    tok = token(args.token_file)
    if not tok:
        print(f"no token: set MACHINE_TOKEN or write {args.token_file}", file=sys.stderr)
        return 2
    req = urllib.request.Request(args.url, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {tok}",
        "User-Agent": "rustfish-machine-status"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except Exception as exc:
        print(f"post failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
