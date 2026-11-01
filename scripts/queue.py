#!/usr/bin/env python3
"""Serial merge queue, run by .github/workflows/queue.yml.

State lives on the `queue-state` branch (queue.json + log.jsonl), so `main`
only ever moves by fast-forward merges of candidates that passed.

Each `advance`:
  1. If RESULT holds the final SPRT result for the current item: H1 ->
     fast-forward main to the tested candidate (only if main hasn't moved;
     otherwise requeue it at the front); anything else -> drop. Log it.
  2. Start pending items in order until one is dispatched: rebase onto main,
     build, test, perft, selftest, bench (refreshing the Bench: trailer), push
     the rebased branch, and dispatch an SPRT whose final report calls back
     here. Conflicts and failed checks are logged as needs-human and skipped.
     'speedup' items merge directly if bench nodes equal main's.

Item modes: stc (gainer [0, 10]), nonreg ([-5, 0] STC), ltc-gainer
([0, 10] at 60+0.6), speedup (no SPRT; bench must be unchanged).
"""
import datetime
import json
import os
import re
import subprocess
import sys

REPO = os.getcwd()
STATE = os.path.join(REPO, ".queue-state")
STATE_BRANCH = "queue-state"

SETTINGS = {
    "stc": {"tc": "10+0.1", "hash": 16, "elo0": 0, "elo1": 10, "shards": 16,
            "rounds_per_shard": 16, "max_rounds_per_shard": 64, "max_waves": 10},
    "nonreg": {"tc": "10+0.1", "hash": 16, "elo0": -5, "elo1": 0, "shards": 16,
               "rounds_per_shard": 16, "max_rounds_per_shard": 64, "max_waves": 10},
    "ltc-gainer": {"tc": "60+0.6", "hash": 64, "elo0": 0, "elo1": 10, "shards": 20,
                   "rounds_per_shard": 6, "max_rounds_per_shard": 24, "max_waves": 8},
}


def sh(*cmd, cwd=REPO, check=True):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{res.stdout}\n{res.stderr}")
    return res


def out(*cmd, cwd=REPO):
    return sh(*cmd, cwd=cwd).stdout.strip()


def now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_state():
    sh("git", "fetch", "-q", "origin", STATE_BRANCH)
    if not os.path.isdir(STATE):
        sh("git", "worktree", "add", "-q", "-B", "qs", STATE, f"origin/{STATE_BRANCH}")
    with open(os.path.join(STATE, "queue.json")) as fh:
        return json.load(fh)


def save_state(state, message):
    with open(os.path.join(STATE, "queue.json"), "w") as fh:
        json.dump(state, fh, indent=2)
        fh.write("\n")
    sh("git", "add", "-A", cwd=STATE)
    if sh("git", "diff", "--cached", "--quiet", cwd=STATE, check=False).returncode != 0:
        sh("git", "commit", "-q", "-m", message, cwd=STATE)
        sh("git", "push", "-q", "origin", f"HEAD:{STATE_BRANCH}", cwd=STATE)


def log(entry):
    entry["logged_at"] = now()
    with open(os.path.join(STATE, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry))


def main_bench():
    """Latest Bench: trailer reachable from origin/main."""
    for body in out("git", "log", "-z", "--format=%B", "origin/main").split("\0"):
        m = re.findall(r"^Bench:\s*(\d+)\s*$", body, re.M)
        if m:
            return int(m[-1])
    return None


def run_engine(cmds):
    return subprocess.run(["./target/release/rustfish"], input=cmds, cwd=REPO,
                          capture_output=True, text=True).stdout


def fast_forward_main(sha):
    sh("git", "fetch", "-q", "origin", "main")
    main_sha = out("git", "rev-parse", "origin/main")
    if sh("git", "merge-base", "--is-ancestor", main_sha, sha, check=False).returncode != 0:
        return False
    sh("git", "push", "-q", "origin", f"{sha}:refs/heads/main")
    return True


def start(item):
    """Prepare and launch one item. Returns the in-flight record, or None if it finished/skipped."""
    branch, mode = item["branch"], item["mode"]
    base = {"branch": branch, "mode": mode}
    sh("git", "fetch", "-q", "origin", "main", branch)
    sh("git", "checkout", "-q", "-B", "candidate", f"origin/{branch}")
    if sh("git", "rebase", "-q", "origin/main", check=False).returncode != 0:
        sh("git", "rebase", "--abort", check=False)
        log(dict(base, outcome="needs-human", reason="rebase conflict"))
        return None
    if not out("git", "rev-list", "origin/main..candidate"):
        log(dict(base, outcome="skipped", reason="no commits beyond main"))
        return None

    if sh("cargo", "build", "--release", "-q", check=False).returncode != 0:
        log(dict(base, outcome="needs-human", reason="build failed after rebase"))
        return None
    if sh("cargo", "test", "--release", "-q", check=False).returncode != 0:
        log(dict(base, outcome="needs-human", reason="cargo test failed"))
        return None
    checks = run_engine("perfttest\nselftest 300\nquit\n")
    if not re.search(r"perfttest summary (\d+)/\1 passed", checks) or "selftest ok" not in checks:
        log(dict(base, outcome="needs-human", reason="perft/selftest failed"))
        return None
    m = re.search(r"Nodes searched\s*:\s*(\d+)", run_engine("bench\nquit\n"))
    nodes = int(m.group(1)) if m else None
    mnodes = main_bench()

    msg = out("git", "log", "-1", "--format=%B")
    if nodes != mnodes:
        if re.search(r"^Bench: \d+$", msg, re.M):
            msg = re.sub(r"^Bench: \d+$", f"Bench: {nodes}", msg, flags=re.M)
        else:
            msg = msg.replace("Co-Authored-By:", f"Bench: {nodes}\n\nCo-Authored-By:", 1)
        sh("git", "commit", "-q", "--amend", "-m", msg)
    sha = out("git", "rev-parse", "candidate")
    sh("git", "push", "-q", "--force", "origin", f"candidate:{branch}")

    if mode == "speedup":
        if nodes != mnodes:
            log(dict(base, outcome="needs-human", reason=f"speedup changed bench {mnodes} -> {nodes}"))
        elif fast_forward_main(sha):
            log(dict(base, outcome="merged", sha=sha, reason="speedup, bench identical"))
        else:
            log(dict(base, outcome="needs-human", reason="main moved during speedup merge"))
        return None

    main_sha = out("git", "rev-parse", "origin/main")
    settings = dict(SETTINGS[mode], candidate_name=branch, base_name="main", callback="queue")
    test_id = f"q-{branch}-{datetime.datetime.utcnow().strftime('%m%d-%H%M')}"
    sh("gh", "workflow", "run", "sprt.yml", "--ref", "main",
       "-f", f"test_id={test_id}", "-f", f"candidate={sha}", "-f", f"base={main_sha}",
       "-f", f"settings={json.dumps(settings)}")
    log(dict(base, outcome="dispatched", test_id=test_id, sha=sha, base_sha=main_sha, bench=nodes))
    return dict(base, test_id=test_id, sha=sha, base_sha=main_sha, started=now())


def advance():
    state = load_state()
    raw = os.environ.get("RESULT", "").strip()
    cur = state.get("current")
    if raw:
        res = json.loads(raw)
        if not cur or res.get("test_id") != cur["test_id"]:
            print(f"ignoring result for {res.get('test_id')}: not the current item")
            return
        verdict = res.get("status")
        record = {k: cur[k] for k in ("branch", "mode", "test_id", "sha")}
        record.update(elo=res.get("elo"), elo_ci95=res.get("elo_ci95"), games=res.get("games"),
                      llr=res.get("llr"), verdict=verdict)
        if verdict == "H1":
            if fast_forward_main(cur["sha"]):
                log(dict(record, outcome="merged"))
            else:
                log(dict(record, outcome="requeued", reason="main moved during test"))
                state["pending"].insert(0, {"branch": cur["branch"], "mode": cur["mode"]})
        else:
            log(dict(record, outcome="rejected"))
        state["current"] = None
    elif cur:
        print(f"{cur['test_id']} still in flight; nothing to do")
        return

    while state["pending"] and not state.get("current"):
        item = state["pending"].pop(0)
        try:
            state["current"] = start(item)
        except Exception as exc:  # keep the queue moving past unexpected failures
            log({"branch": item["branch"], "mode": item["mode"], "outcome": "needs-human",
                 "reason": f"queue error: {exc}"[:500]})
            sh("git", "rebase", "--abort", check=False)
            state["current"] = None
    save_state(state, f"queue: {state['current']['test_id'] if state.get('current') else 'idle'}")


if __name__ == "__main__":
    if sys.argv[1:] != ["advance"]:
        sys.exit("usage: queue.py advance  (RESULT env = final SPRT result JSON, optional)")
    advance()
