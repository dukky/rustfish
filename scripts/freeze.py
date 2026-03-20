#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ENGINE = ROOT / "target" / "release" / "rustfish"
BIN_DIR = ROOT / "bin"


def run(cmd: list[str], cwd: pathlib.Path = ROOT, capture: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    return proc.stdout if capture else ""


def git_info() -> dict[str, str | bool | None]:
    try:
        sha = run(["git", "rev-parse", "--short", "HEAD"], capture=True).strip()
        dirty = bool(run(["git", "status", "--short"], capture=True).strip())
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True).strip()
        return {"sha": sha, "dirty": dirty, "branch": branch}
    except Exception:
        return {"sha": None, "dirty": None, "branch": None}


def freeze_name(engine_name: str, git_sha: str | None, stamp: str) -> str:
    sha_part = git_sha or "nogit"
    return f"{engine_name}_{sha_part}_{stamp}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and freeze a versioned rustfish binary with metadata.")
    ap.add_argument("--engine-name", default="rustfish")
    ap.add_argument("--engine", default=str(DEFAULT_ENGINE))
    ap.add_argument("--bench-depth", type=int, default=5)
    ap.add_argument("--note", default="")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    engine_path = pathlib.Path(args.engine)
    if not args.skip_build:
        run(["cargo", "build", "--release"])

    if not engine_path.exists():
        print(f"error: engine binary not found: {engine_path}", file=sys.stderr)
        return 2

    BIN_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_meta = git_info()
    frozen_name = freeze_name(args.engine_name, git_meta["sha"], stamp)
    frozen_path = BIN_DIR / frozen_name
    meta_path = BIN_DIR / f"{frozen_name}.json"

    shutil.copy2(engine_path, frozen_path)

    bench_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bench.py"),
        "--engine",
        str(frozen_path),
        "--depth",
        str(args.bench_depth),
    ]
    bench_output = run(bench_cmd, capture=True)

    metadata = {
        "frozen_path": str(frozen_path),
        "created_utc": stamp,
        "engine_name": args.engine_name,
        "source_engine": str(engine_path),
        "git": git_meta,
        "bench_depth": args.bench_depth,
        "bench_output": bench_output.strip().splitlines(),
        "note": args.note,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(frozen_path)
    print(meta_path)
    print(bench_output.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
