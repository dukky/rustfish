#!/usr/bin/env python3
"""Helpers for .github/workflows/train-sweep.yml (stdlib only).

  plan     validate the variants JSON, print the job matrix
  pool     resolve datagen seeds -> runs, fetch, convert, x2-fix, cut the
           validation sets, subsample to --positions, shuffle
  init     fetch a warm-start net (<ref>:<path>, https URL or run:<id>:<artifact>:<file>)
  finish   after a variant's training: pick its net, eval --best-scale on both val sets
  report   markdown table of every variant's finish JSON

Validation sets (the ones used on the laptop, ~/rustfish-data):
  val.bin      the last 1,000,000 positions of seed 9 (net scale)
  val17x2.bin  the last 1,000,000 positions of seed 17, scores x2
               (fix/val17x2.bin; the rest of seed 17 is fix/s17x2.bin)
Both are always cut from their seed, so they are never trained on. The
expected sha256 of each is checked against the laptop copies.

Score scale: CI seeds 14-18 were written at the engine's halved scale
(eval = net/2 since ad53daa) and get every score doubled, saturating, exactly
as ~/rustfish-data/fix/x2.py and netloop.x2_copy do. Seeds >= 19 are at net
scale. Seed 7 is quarantined (never used).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from array import array
from pathlib import Path

REC = 32
VAL_N = 1_000_000
QUARANTINED = {7}
VAL_SHA = {
    "val.bin": "7d2d9157462d1fc28db6d21c6ef199da823f20160c61d3f7532a6676e3af85dd",
    "val17x2.bin": "abe51b808e34a42c8eb65d02a1041be0c4da7b7c4b2632393d1a298e7cc404b1",
}
BLOCK = 65536  # records per contiguous block when subsampling a seed
REPO = os.environ.get("GITHUB_REPOSITORY", "dukky/rustfish")


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), file=sys.stderr, flush=True)
    return subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


def out(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def x2(buf):
    """Double every score (int16 at word 12 of each 16-word record), saturating."""
    a = array("h")
    a.frombytes(buf)
    a[12::16] = array("h", (32767 if x > 16383 else -32767 if x < -16383 else 2 * x for x in a[12::16]))
    return a.tobytes()


def parse_int_set(s):
    r = set()
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            r.update(range(int(a), int(b) + 1))
        else:
            r.add(int(part))
    return r


# ---------------------------------------------------------------- plan

def cmd_plan(a):
    v = json.loads(a.variants)
    assert isinstance(v, list) and v, "variants must be a non-empty JSON list"
    names = set()
    for x in v:
        assert isinstance(x, dict) and "name" in x and "args" in x, f"variant needs name and args: {x}"
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", x["name"]), f"bad variant name {x['name']!r}"
        assert x["name"] not in names, f"duplicate variant name {x['name']}"
        names.add(x["name"])
        for bad in ("--data", "--val", "--out", "--threads", "--checkpoint"):
            assert bad not in x["args"].split(), f"{x['name']}: {bad} is added by the workflow"
        x.setdefault("init", True)
    print(json.dumps(v))


# ---------------------------------------------------------------- pool

def datagen_runs():
    """seed -> newest successful datagen run id (with its merged artifact name)."""
    rows = json.loads(out(["gh", "run", "list", "-R", REPO, "-w", "datagen.yml", "--status", "success",
                           "-L", "400", "--json", "databaseId,displayTitle"]))
    runs = {}
    for r in rows:  # newest first
        m = re.search(r"\bseed (\d+)\b", r["displayTitle"])
        if m and int(m.group(1)) not in runs:
            runs[int(m.group(1))] = str(r["databaseId"])
    return runs


def resolve(spec, runs):
    """Seeds spec -> {seed: run_id}. Explicit seeds must exist; ranges skip gaps."""
    spec = spec.strip()
    try:
        j = json.loads(spec)
    except json.JSONDecodeError:
        j = spec
    if isinstance(j, dict):
        res = {int(k): str(v) for k, v in j.items()}
        bad = QUARANTINED & res.keys()
        assert not bad, f"seed(s) {sorted(bad)} are quarantined"
        return res
    items = j if isinstance(j, list) else [j]
    res = {}
    for it in items:
        it = str(it)
        if it == "all":
            seeds, explicit = sorted(runs), False
        elif "-" in it:
            lo, hi = map(int, it.split("-"))
            seeds, explicit = [s for s in sorted(runs) if lo <= s <= hi], False
        else:
            seeds, explicit = [int(it)], True
        for s in seeds:
            if s in QUARANTINED:
                assert not explicit, f"seed {s} is quarantined"
                print(f"seed {s}: quarantined, skipped", file=sys.stderr)
                continue
            assert s in runs, f"no successful datagen run for seed {s}"
            res[s] = runs[s]
    return res


def artifact_name(seed, run):
    names = out(["gh", "api", f"repos/{REPO}/actions/runs/{run}/artifacts?per_page=100",
                 "--jq", '.artifacts[] | select(.expired | not) | .name']).split()
    m = [n for n in names if re.fullmatch(rf"datagen-s{seed}-n\d+", n)]
    assert m, f"no merged datagen artifact for seed {seed} in run {run} (expired?)"
    return m[0]


def subsample(src, dst, start, n, k, double):
    """Write k of the records [start, start+n) of src to dst as evenly spaced
    contiguous blocks (whole games stay together; the shuffle mixes them)."""
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        if k >= n:
            picks = [(start, n)]
        else:
            nb = -(-n // BLOCK)
            m = -(-k // BLOCK)
            picks = [(start + (i * nb // m) * BLOCK, BLOCK) for i in range(m)]
        left = k
        for off, cnt in picks:
            fi.seek(off * REC)
            while cnt > 0 and left > 0:
                c = min(cnt, left, 1 << 20)
                b = fi.read(c * REC)
                if not b:
                    break
                fo.write(x2(b) if double else b)
                got = len(b) // REC
                cnt -= got
                left -= got
    return k - left


def cut_tail(src, n, dst, double):
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fi.seek((n - VAL_N) * REC)
        while True:
            b = fi.read(REC << 20)
            if not b:
                break
            fo.write(x2(b) if double else b)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_pool(a):
    w = Path(a.work)
    w.mkdir(parents=True, exist_ok=True)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    x2_seeds = parse_int_set(a.x2_seeds)
    runs = datagen_runs()
    train = resolve(a.seeds, runs)
    assert train, "no training seeds"
    need = dict(train)
    for s in (9, 17):  # validation seeds are always fetched
        if s not in need:
            assert s in runs, f"validation seed {s} has no datagen run"
            need[s] = runs[s]
    print("seeds:", json.dumps({s: need[s] for s in sorted(need)}), file=sys.stderr)

    # 1. download every merged artifact (gzip text) and count its lines.
    lines = {}
    for s in sorted(need):
        d = w / f"dl-{s}"
        if not d.exists():
            sh(["gh", "run", "download", need[s], "-R", REPO, "-n", artifact_name(s, need[s]), "-D", str(d)])
        lines[s] = int(out(["bash", "-c", f"gunzip -c {d}/*.gz | wc -l"]).strip())
        print(f"seed {s}: {lines[s]} lines", file=sys.stderr)
    avail = sum(lines[s] - (VAL_N if s in (9, 17) else 0) for s in train)
    cap = a.positions if a.positions > 0 else avail
    frac = min(1.0, cap / avail)
    st = os.statvfs(w)
    free = st.f_bavail * st.f_frsize
    est = int(min(cap, avail) * REC * 2.1)
    print(f"available {avail}, cap {cap}, fraction {frac:.4f}; disk free {free >> 30} GB, "
          f"need ~{est >> 30} GB more", file=sys.stderr)
    assert est < free, "not enough disk for this pool: lower positions"

    # 2. per seed: convert, cut validation tails, x2-fix, subsample.
    T = a.trainer
    manifest = {"seeds": {}, "x2_seeds": sorted(x2_seeds & train.keys()), "quarantined": sorted(QUARANTINED),
                "positions_cap": a.positions, "fraction": frac}
    parts = []
    for s in sorted(need):
        full = w / "full.bin"
        sh(f"gunzip -c {w}/dl-{s}/*.gz | {T} convert - {full}")
        sh(["rm", "-rf", str(w / f"dl-{s}")])
        n = full.stat().st_size // REC
        dbl = s in x2_seeds
        if s == 9:
            cut_tail(full, n, dest / "val.bin", False)
            n -= VAL_N
        if s == 17:
            cut_tail(full, n, dest / "val17x2.bin", True)  # always x2: seed 17 is half scale
            n -= VAL_N
        if s in train:
            k = round(n * frac)
            p = w / f"s{s}.bin"
            got = subsample(full, p, 0, n, k, dbl)
            parts.append(str(p))
            manifest["seeds"][s] = {"run": need[s], "positions": n, "used": got, "x2": dbl}
            print(f"seed {s}: {got} of {n} positions{' (x2)' if dbl else ''}", file=sys.stderr)
        full.unlink()

    # 3. global shuffle.
    sh([T, "shuffle", "--seed", str(a.shuffle_seed), "--out", str(dest / "pool.bin")] + parts)
    for p in parts:
        os.unlink(p)
    manifest["pool"] = (dest / "pool.bin").stat().st_size // REC
    manifest["val"] = {}
    for v in ("val.bin", "val17x2.bin"):
        h = sha256(dest / v)
        manifest["val"][v] = {"sha256": h, "matches_laptop": h == VAL_SHA[v]}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------- init

def cmd_init(a):
    spec, dst = a.spec.strip(), Path(a.out)
    if spec.startswith("https://"):
        sh(["curl", "-fsSL", "-o", str(dst), spec])
    elif spec.startswith("run:"):
        _, run, art, f = spec.split(":", 3)
        d = dst.parent / "init-dl"
        sh(["gh", "run", "download", run, "-R", REPO, "-n", art, "-D", str(d)])
        os.replace(d / f, dst)
    else:
        ref, path = spec.split(":", 1) if ":" in spec else ("main", spec)
        with open(dst, "wb") as fh:
            sh(["gh", "api", "-H", "Accept: application/vnd.github.raw",
                f"repos/{REPO}/contents/{path}?ref={ref}"], stdout=fh)
    print(f"init net {spec}: {dst.stat().st_size} bytes, sha256 {sha256(dst)}")


# ---------------------------------------------------------------- finish

def eval_args(args):
    """The --buckets / --king-buckets / --lambda of a train command line, for eval."""
    t, r = args.split(), []
    for i, x in enumerate(t):
        if x in ("--buckets", "--king-buckets", "--lambda") and i + 1 < len(t):
            r += [x, t[i + 1]]
    return r


def best_scale(trainer, net, val, args, threads):
    o = out([trainer, "eval", "--net", str(net), "--val", str(val), "--best-scale", "--threads", str(threads)]
            + eval_args(args))
    loss = re.search(r"val positions \d+ loss ([0-9.]+)", o)
    bs = re.search(r"best scale x([0-9.]+): loss ([0-9.]+)", o)
    return {"loss": float(loss.group(1)), "scale": float(bs.group(1)), "best_scale_loss": float(bs.group(2))}


def cpu_model():
    try:
        for line in out(["lscpu"]).splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "?"


def cmd_finish(a):
    log = Path(a.log).read_text(errors="replace")
    net = Path(a.net)
    epochs = [(int(e), float(v), float(q), int(p)) for e, v, q, p in re.findall(
        r"^epoch (\d+) .*? val ([0-9.]+) \(quantized ([0-9.]+)\).*? ([0-9]+) pos/s", log, re.M)]
    start = re.search(r"^start val loss ([0-9.]+) \(quantized ([0-9.]+)\)", log, re.M)
    kb = re.search(r"^keep-best: epoch (\d+)", log, re.M)
    res = {"name": a.name, "args": a.args, "init": a.init, "cpu": cpu_model(), "nproc": os.cpu_count(),
           "rc": a.rc, "timed_out": a.rc == 124, "epochs": len(epochs),
           "start_q": float(start.group(2)) if start else None}
    if epochs:
        e, v, q, p = epochs[-1]
        b = min(epochs, key=lambda x: x[2])
        res.update(final_val=v, final_q=q, pos_s=p, best_epoch=b[0], best_q=b[2])
    if kb:
        res["output_epoch"] = int(kb.group(1))
    elif epochs:
        # Killed before --keep-best wrote <out>: use the best finished epoch.
        best = net.with_suffix(f".e{res['best_epoch']}.bin")
        if best.exists():
            net.write_bytes(best.read_bytes())
            res["output_epoch"] = res["best_epoch"]
    # The output net (best epoch by --keep-best, possibly the start net) and,
    # when that is not the last epoch, the last epoch too: what training did.
    nets = [("", net)]
    if epochs and res.get("output_epoch") != epochs[-1][0]:
        nets.append(("last_", net.with_suffix(f".e{epochs[-1][0]}.bin")))
    for pre, n in nets:
        if not n.exists():
            continue
        for key, v in (("val", a.val), ("val17x2", a.val17)):
            try:
                res[pre + key] = best_scale(a.trainer, n, v, a.args, a.threads)
            except Exception as ex:  # keep the row even if eval fails
                res[pre + key] = {"error": str(ex)}
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_evalnet(a):
    """Reference row (the warm-start net) for the report."""
    res = {"name": a.name, "args": a.args, "reference": True}
    for key, v in (("val", a.val), ("val17x2", a.val17)):
        res[key] = best_scale(a.trainer, a.net, v, a.args, os.cpu_count())
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- report

def cmd_report(a):
    rows = [json.loads(p.read_text()) for p in sorted(Path(a.dir).rglob("*.json")) if p.name != "manifest.json"]

    def bs(r, k):
        return r.get(k, {}).get("best_scale_loss")

    def key(r):
        x, y = bs(r, "val"), bs(r, "val17x2")
        return (0, x + y) if x is not None and y is not None else (1, 0)

    f = lambda x, n=6: "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))

    def pair(r, pre):
        v, w = r.get(pre + "val", {}), r.get(pre + "val17x2", {})
        if not v and not w:
            return "-"
        return (f"{f(v.get('best_scale_loss'))} / {f(w.get('best_scale_loss'))} "
                f"(x{f(v.get('scale'), 3)} / x{f(w.get('scale'), 3)})")

    print("Sorted by best-scale loss (`eval --best-scale`, scale profiled out; lower is better) of the "
          "output net on val.bin + val17x2. The output is the best epoch by quantised val.bin loss "
          "(`--keep-best`, e0 = the start net); when it is not the last epoch, the last epoch is shown "
          "too. `val (q)` = float (quantised) val.bin loss as printed by the trainer.\n")
    print("| variant | best-scale val.bin / val17x2 (scale) | out | last epoch best-scale | epochs | "
          "final val (q) | best epoch (q) | start q | pos/s | CPU | config |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=key):
        ref = r.get("reference")
        ep = f"{r.get('epochs', '-')}{' (timeout)' if r.get('timed_out') else ''}"
        print(f"| {r['name']} | {pair(r, '')} | {'-' if ref else 'e' + f(r.get('output_epoch'))} | "
              f"{pair(r, 'last_')} | {'-' if ref else ep} | {f(r.get('final_val'))} ({f(r.get('final_q'))}) | "
              f"{f(r.get('best_epoch'))} ({f(r.get('best_q'))}) | {f(r.get('start_q'))} | {f(r.get('pos_s'))} | "
              f"{'-' if ref else r.get('cpu', '?')} | `{r.get('args', '')}`{' +init' if r.get('init') else ''} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("plan")
    q.add_argument("--variants", required=True)
    q = sp.add_parser("pool")
    q.add_argument("--seeds", required=True)
    q.add_argument("--positions", type=int, default=300_000_000)
    q.add_argument("--x2-seeds", default="14-18")
    q.add_argument("--shuffle-seed", type=int, default=1)
    q.add_argument("--trainer", required=True)
    q.add_argument("--work", required=True)
    q.add_argument("--dest", required=True)
    q = sp.add_parser("init")
    q.add_argument("--spec", required=True)
    q.add_argument("--out", required=True)
    q = sp.add_parser("finish")
    for k in ("name", "log", "net", "trainer", "val", "val17", "json"):
        q.add_argument(f"--{k}", required=True)
    q.add_argument("--args", default="")
    q.add_argument("--init", action="store_true")
    q.add_argument("--rc", type=int, default=0)
    q.add_argument("--threads", type=int, default=os.cpu_count())
    q = sp.add_parser("evalnet")
    for k in ("name", "net", "trainer", "val", "val17", "json"):
        q.add_argument(f"--{k}", required=True)
    q.add_argument("--args", default="")
    q = sp.add_parser("report")
    q.add_argument("--dir", required=True)
    a = p.parse_args()
    {"plan": cmd_plan, "pool": cmd_pool, "init": cmd_init, "finish": cmd_finish, "evalnet": cmd_evalnet, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
