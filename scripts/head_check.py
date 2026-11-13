#!/usr/bin/env python3
"""Build a deeper-head engine variant and check it against the trainer.

Engine inference for `nnue-trainer --head` nets lives in src/nnue_head.rs
behind `HEAD` in src/nnue.rs (default off). This script:

1. copies the engine sources (src/, net/, Cargo.*) to `--workdir`, sets
   `H`, `KING_LAYOUT` and `HEAD = true` there and embeds `--net`;
2. (unless --no-check) takes `--n` positions from a 32-byte record file,
   writes them as FENs, gets the trainer's float evals with
   `nnue-trainer fen-eval`, and runs the engine test
   `head_matches_trainer_fixture` on them (mean / p95 / max |diff|);
3. builds the release binary (`<workdir>/target/release/rustfish`), for
   bench / NPS runs.

Example (laptop):
  python3 scripts/head_check.py --net ~/rustfish-data/head/t1.bin \\
      --hidden 512 --king-buckets kb8m --records ~/rustfish-data/val.bin \\
      --trainer tools/nnue-trainer/target/release/nnue-trainer \\
      --workdir ~/rustfish-data/head/engine-h512
"""
import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KING_LAYOUTS = {"none": ("LAYOUT_NONE", 1), "m1": ("LAYOUT_M1", 1), "kb4": ("LAYOUT_KB4", 4),
                "kb4m": ("LAYOUT_KB4M", 4), "kb8m": ("LAYOUT_KB8M", 8)}
REC = 32


def head_net_bytes(h, buckets, kb):
    plain = kb * 768 * h + h + buckets * (2 * h + 1)
    head = buckets * (16 * 2 * h + 16 + 32 * 16 + 32 + 32 + 1)
    return 2 * (plain + head)


def rec_to_fen(r):
    """A trainer record (see `encode` in tools/nnue-trainer) -> FEN."""
    occ = struct.unpack_from("<Q", r, 0)[0]
    board = [None] * 64
    n = 0
    for sq in range(64):
        if occ >> sq & 1:
            nib = r[8 + n // 2] >> (4 * (n & 1)) & 0xF
            c = "pnbrqk"[nib & 7]
            board[sq] = c if nib >> 3 else c.upper()
            n += 1
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for f in range(8):
            p = board[rank * 8 + f]
            if p is None:
                empty += 1
            else:
                row += (str(empty) if empty else "") + p
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    stm = "b" if r[27] == 1 else "w"
    return "/".join(rows) + f" {stm} - - 0 1"


def set_const(src, name, typ, value):
    pat = rf"^(pub\(crate\) const {name}: {typ} = )[^;]+;"
    if not re.search(pat, src, re.MULTILINE):
        sys.exit(f"no `pub(crate) const {name}: {typ}` in src/nnue.rs")
    return re.sub(pat, rf"\g<1>{value};", src, count=1, flags=re.MULTILINE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="nnue-trainer --head net file")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--buckets", type=int, default=8, help="output buckets (the engine has BUCKETS = 8)")
    ap.add_argument("--king-buckets", default="kb8m", choices=sorted(KING_LAYOUTS))
    ap.add_argument("--workdir", required=True, help="scratch dir for the engine copy")
    ap.add_argument("--records", help="32-byte record file to take positions from (e.g. val.bin)")
    ap.add_argument("--n", type=int, default=500, help="positions for the check")
    ap.add_argument("--skip", type=int, default=0, help="records to skip first")
    ap.add_argument("--trainer", default=str(REPO / "tools/nnue-trainer/target/release/nnue-trainer"))
    ap.add_argument("--no-check", action="store_true", help="only build the variant")
    ap.add_argument("--plain", action="store_true",
                    help="HEAD = false and a plain net of the same H (a speed baseline); implies --no-check")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    net = Path(args.net).expanduser().resolve()
    layout, kb = KING_LAYOUTS[args.king_buckets]
    args.no_check |= args.plain
    want = head_net_bytes(args.hidden, args.buckets, kb)
    if args.plain:
        want = 2 * (kb * 768 * args.hidden + args.hidden + args.buckets * (2 * args.hidden + 1))
    if net.stat().st_size != want:
        sys.exit(f"{net} is {net.stat().st_size} bytes; a H={args.hidden} {args.king_buckets} "
                 f"{args.buckets}-bucket {'plain' if args.plain else 'head'} net is {want}")
    wd = Path(args.workdir).expanduser().resolve()
    wd.mkdir(parents=True, exist_ok=True)
    for d in ("src", "net"):
        if (wd / d).exists():
            shutil.rmtree(wd / d)
        shutil.copytree(REPO / d, wd / d)
    for f in ("Cargo.toml", "Cargo.lock"):
        shutil.copyfile(REPO / f, wd / f)
    nnue_rs = wd / "src/nnue.rs"
    src = nnue_rs.read_text()
    src = set_const(src, "H", "usize", args.hidden)
    src = set_const(src, "BUCKETS", "usize", args.buckets)
    src = set_const(src, "HEAD", "bool", "false" if args.plain else "true")
    src = set_const(src, "KING_LAYOUT", "KingLayout", layout)
    nnue_rs.write_text(src)
    shutil.copyfile(net, wd / "net/nnue.bin")
    env = dict(os.environ, CARGO_TARGET_DIR=str(wd / "target"))

    if not args.no_check:
        if not args.records:
            sys.exit("--records needed for the check (or --no-check)")
        fens = []
        with open(Path(args.records).expanduser(), "rb") as f:
            f.seek(args.skip * REC)
            for _ in range(args.n):
                r = f.read(REC)
                if len(r) < REC:
                    break
                fens.append(rec_to_fen(r))
        (wd / "fens.txt").write_text("\n".join(fens) + "\n")
        out = subprocess.run([args.trainer, "fen-eval", "--net", str(net), "--buckets", str(args.buckets),
                              "--king-buckets", args.king_buckets, "--fens", str(wd / "fens.txt")],
                             check=True, capture_output=True, text=True).stdout
        (wd / "fixture.txt").write_text(out)
        env_t = dict(env, RUSTFISH_HEAD_FIXTURE=str(wd / "fixture.txt"))
        r = subprocess.run(["cargo", "test", "--release", "-j", str(args.jobs), "nnue::",
                            "--", "--nocapture", "--test-threads", str(args.jobs)],
                           cwd=wd, env=env_t, capture_output=True, text=True)
        for line in (r.stdout + r.stderr).splitlines():
            if "head fixture:" in line or "test result" in line or "panicked" in line or "FAILED" in line:
                print(line)
        if r.returncode != 0:
            sys.exit("head check failed")
    subprocess.run(["cargo", "build", "--release", "-j", str(args.jobs)], cwd=wd, env=env, check=True,
                   capture_output=True)
    print(f"engine: {wd / 'target/release/rustfish'}")


if __name__ == "__main__":
    main()
