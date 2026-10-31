#!/usr/bin/env python3
"""Drive the GitHub Actions test pipeline from the laptop (needs `gh`).

  sprt <branch>   (branch must already be pushed) dispatch an SPRT of <branch> vs main, follow
                  every wave to a verdict, append it to tests/sprt-log.jsonl
  ladder          dispatch the CCRL ladder on main, wait, append the rating
                  to tests/ladder-log.jsonl
  status          show recent SPRT / ladder runs

Examples:
  python3 scripts/ci.py sprt tm-soft-limit                  # STC gainer [0, 5]
  python3 scripts/ci.py sprt tm-soft-limit --ltc            # LTC non-regression [-5, 0]
  python3 scripts/ci.py sprt tm-soft-limit --ltc --gainer   # LTC gainer [0, 5] (time management)
  python3 scripts/ci.py sprt refactor-movegen --nonreg      # must not lose [-5, 0]
  python3 scripts/ci.py sprt my-branch --no-wait            # dispatch and exit
  python3 scripts/ci.py sprt --resume <test_id>             # re-attach to a test
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "tests")

PROFILES = {
    # rounds_per_shard is the first wave; each later wave doubles up to the max.
    "stc": {"tc": "10+0.1", "hash": 16, "shards": 16, "rounds_per_shard": 16, "max_rounds_per_shard": 64},
    "ltc": {"tc": "60+0.6", "hash": 64, "shards": 20, "rounds_per_shard": 6, "max_rounds_per_shard": 24},
}


def gh(*args, capture=True):
    # Read-only calls retry through network blips; dispatches never retry, so a
    # lost response can't launch a test twice.
    attempts = 1 if args[:2] == ("workflow", "run") else 6
    for attempt in range(attempts):
        res = subprocess.run(["gh", *args], cwd=ROOT, capture_output=capture, text=True)
        if res.returncode == 0:
            return res.stdout if capture else ""
        if attempt + 1 < attempts:
            time.sleep(30 * (attempt + 1))
    sys.exit(f"gh {' '.join(args)} failed:\n{res.stderr if capture else ''}")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def append_log(name, record):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, name), "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def runs_titled(workflow, needle):
    out = gh("run", "list", "--workflow", workflow, "--limit", "60",
             "--json", "databaseId,displayTitle,status,conclusion,createdAt,url")
    return [r for r in json.loads(out) if needle in r["displayTitle"]]


def wait_for_run(run_id, quiet=False):
    while True:
        info = json.loads(gh("run", "view", str(run_id), "--json", "status,conclusion"))
        if info["status"] == "completed":
            return info["conclusion"]
        if not quiet:
            print(".", end="", flush=True)
        time.sleep(60)


def download_json(run_id, artifact, filename):
    with tempfile.TemporaryDirectory() as tmp:
        # Artifacts can lag the run's completion or hit network blips; retry.
        for attempt in range(6):
            res = subprocess.run(["gh", "run", "download", str(run_id), "-n", artifact, "-D", tmp],
                                 cwd=ROOT, capture_output=True, text=True)
            if res.returncode == 0:
                with open(os.path.join(tmp, filename)) as fh:
                    return json.load(fh)
            time.sleep(30 * (attempt + 1))
        return None


def follow_sprt(test_id):
    """Wait for each wave in turn until one reports final, and return its result."""
    wave = 1
    while True:
        title = f"SPRT {test_id} wave {wave}"
        deadline = time.time() + 600
        runs = []
        while not runs:
            runs = [r for r in runs_titled("sprt.yml", f"SPRT {test_id} ")
                    if r["displayTitle"] == title]
            if not runs:
                if time.time() > deadline:
                    sys.exit(f"no run titled {title!r} appeared; check the previous wave's report job")
                time.sleep(15)
        run = runs[0]
        print(f"wave {wave}: {run['url']} ", end="", flush=True)
        conclusion = wait_for_run(run["databaseId"])
        res = download_json(run["databaseId"], "sprt-result", "sprt-result.json")
        if res is None:
            sys.exit(f"\nwave {wave} ended {conclusion} without a result artifact: {run['url']}")
        print(f" LLR {res['llr']:+.2f}, Elo {res['elo']:+.1f}, {res['games']} games -> {res['status']}")
        if res["final"]:
            res["url"] = run["url"]
            return res
        wave += 1


def cmd_sprt(args):
    if args.resume:
        test_id = args.resume
    else:
        branch = args.candidate or sys.exit("candidate branch required")
        remote = git("ls-remote", "origin", f"refs/heads/{branch}")
        if not remote:
            sys.exit(f"branch {branch!r} is not on origin; push it first: git push -u origin {branch}")
        local = git("rev-parse", "--verify", "--quiet", branch) if branch else ""
        if local and remote.split()[0] != local:
            print(f"warning: origin/{branch} ({remote.split()[0][:7]}) differs from local ({local[:7]})")
        profile = "ltc" if args.ltc else "stc"
        settings = dict(PROFILES[profile])
        # An LTC run confirms an STC pass, so it defaults to non-regression:
        # a real gain clears [-5, 0] in far fewer games than [0, 5].
        nonreg = args.nonreg or (args.ltc and not args.gainer)
        elo0, elo1 = (-5, 0) if nonreg else (args.elo0, args.elo1)
        settings.update(elo0=elo0, elo1=elo1, max_waves=args.max_waves)
        # Pin both refs to SHAs now: a test may sit queued for hours, and main
        # (or the candidate branch) moving meanwhile must not change what it tests.
        cand_sha = remote.split()[0]
        base_remote = git("ls-remote", "origin", f"refs/heads/{args.base}")
        base_sha = base_remote.split()[0] if base_remote else args.base
        settings.update(candidate_name=branch, base_name=args.base)
        stamp = datetime.datetime.now().strftime("%m%d-%H%M")
        test_id = f"{re.sub(r'[^A-Za-z0-9._-]', '-', branch)}-{profile}-{stamp}"
        gh("workflow", "run", "sprt.yml", "--ref", "main",
           "-f", f"test_id={test_id}", "-f", f"candidate={cand_sha}", "-f", f"base={base_sha}",
           "-f", f"settings={json.dumps(settings)}", capture=False)
        print(f"dispatched {test_id}: {branch} ({cand_sha[:7]}) vs {args.base} ({base_sha[:7]}), "
              f"{settings['tc']}, [{elo0}, {elo1}]")
        if args.no_wait:
            print(f"re-attach later with: python3 scripts/ci.py sprt --resume {test_id}")
            return
    res = follow_sprt(test_id)
    res["logged_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    append_log("sprt-log.jsonl", res)
    verdict = {"H1": "PASSED", "H0": "FAILED"}.get(res["status"], res["status"].upper())
    print(f"\n{verdict}: {res['candidate']} vs {res['base']}  Elo {res['elo']:+.1f} "
          f"(95% CI {res['elo_ci95'][0]:+.1f} .. {res['elo_ci95'][1]:+.1f}), {res['games']} games")
    print("logged to tests/sprt-log.jsonl")


def cmd_ladder(args):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gh("workflow", "run", "ccrl-ladder.yml", "--ref", args.ref,
       "-f", f"games_per_anchor={args.games}", "-f", f"tc={args.tc}", capture=False)
    time.sleep(10)
    run = json.loads(gh("run", "list", "--workflow", "ccrl-ladder.yml", "--limit", "1",
                        "--json", "databaseId,url,headSha"))[0]
    print(f"ladder {run['url']} ", end="", flush=True)
    wait_for_run(run["databaseId"])
    res = download_json(run["databaseId"], "ccrl-ladder-results", "results.json")
    if res is None:
        sys.exit(f"\nladder finished without results: {run['url']}")
    res.update(url=run["url"], sha=run["headSha"], tc=args.tc, logged_at=stamp)
    append_log("ladder-log.jsonl", res)
    print(f"\nCCRL Blitz-scale {res['rating']:.0f} (95% CI {res['ci'][0]:.0f} .. {res['ci'][1]:.0f}), "
          f"{res['games']} games; logged to tests/ladder-log.jsonl")


def cmd_status(_args):
    for wf in ("sprt.yml", "ccrl-ladder.yml"):
        print(f"== {wf}")
        out = gh("run", "list", "--workflow", wf, "--limit", "8",
                 "--json", "displayTitle,status,conclusion,createdAt")
        for r in json.loads(out):
            print(f"  {r['createdAt'][:16]}  {r['status']:<11} {r['conclusion'] or '':<9} {r['displayTitle']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sprt")
    sp.add_argument("candidate", nargs="?")
    sp.add_argument("--base", default="main")
    sp.add_argument("--ltc", action="store_true", help="60+0.6 instead of 10+0.1")
    sp.add_argument("--nonreg", action="store_true", help="non-regression bounds [-5, 0]")
    sp.add_argument("--gainer", action="store_true", help="with --ltc: gainer bounds instead of non-regression")
    sp.add_argument("--elo0", type=float, default=0)
    # [0, 10] while the big gains are being harvested: same 5% false-accept
    # rate, ~4x faster on borderline patches, but real +2..5 gains often fail.
    # Return to --elo1 5 once typical patches get small.
    sp.add_argument("--elo1", type=float, default=10)
    sp.add_argument("--max-waves", type=int, default=10)
    sp.add_argument("--no-wait", action="store_true")
    sp.add_argument("--resume", metavar="TEST_ID")
    sp.set_defaults(func=cmd_sprt)

    lp = sub.add_parser("ladder")
    lp.add_argument("--ref", default="main")
    lp.add_argument("--games", type=int, default=96, help="games per anchor")
    lp.add_argument("--tc", default="120+1")
    lp.set_defaults(func=cmd_ladder)

    st = sub.add_parser("status")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
