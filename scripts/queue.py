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

# Per-mode SPRT settings, shared with the dashboard (deploy/dashboard). The
# ltc-check mode is the periodic check queued automatically: main vs main
# CHECK_EVERY merges back.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue-settings.json")) as _fh:
    _CFG = json.load(_fh)
SETTINGS = _CFG["modes"]
CHECK_EVERY = _CFG["check_every"]


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
    """queue.json plus any requests waiting in inbox/ (consumed here, removed on save).

    Additions arrive as inbox files rather than workflow inputs: GitHub keeps
    only one pending run per concurrency group and cancels the rest, which
    would silently drop an input, while a file stays until a run reads it."""
    sh("git", "fetch", "-q", "origin", STATE_BRANCH)
    if not os.path.isdir(STATE):
        sh("git", "worktree", "add", "-q", "-B", "qs", STATE, f"origin/{STATE_BRANCH}")
    with open(os.path.join(STATE, "queue.json")) as fh:
        state = json.load(fh)
    inbox = os.path.join(STATE, "inbox")
    for name in sorted(os.listdir(inbox)) if os.path.isdir(inbox) else []:
        path = os.path.join(inbox, name)
        with open(path) as fh:
            req = json.load(fh)
        adds = parse_adds(req.get("add", ""))
        if req.get("front"):
            state["pending"][:0] = adds
        else:
            state["pending"] += adds
        if req.get("unpause"):
            state.pop("paused", None)
        os.remove(path)
        print(f"inbox {name}: {req}")
    return state


def push_state():
    """Push queue-state, rebasing over inbox files that arrived meanwhile."""
    for _ in range(5):
        if sh("git", "push", "-q", "origin", f"HEAD:{STATE_BRANCH}", cwd=STATE, check=False).returncode == 0:
            return
        sh("git", "pull", "-q", "--rebase", "origin", STATE_BRANCH, cwd=STATE)
    raise RuntimeError("could not push queue-state")


def save_state(state, message):
    with open(os.path.join(STATE, "queue.json"), "w") as fh:
        json.dump(state, fh, indent=2)
        fh.write("\n")
    sh("git", "add", "-A", cwd=STATE)
    if sh("git", "diff", "--cached", "--quiet", cwd=STATE, check=False).returncode != 0:
        sh("git", "commit", "-q", "-m", message, cwd=STATE)
        push_state()


def request(req):
    """Local helper: drop a request into queue-state's inbox and nudge the queue."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="queue-inbox-")
    sh("git", "fetch", "-q", "origin", STATE_BRANCH)
    sh("git", "worktree", "add", "-q", "--detach", tmp, f"origin/{STATE_BRANCH}")
    try:
        os.makedirs(os.path.join(tmp, "inbox"), exist_ok=True)
        name = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S%f") + ".json"
        with open(os.path.join(tmp, "inbox", name), "w") as fh:
            json.dump(req, fh)
        sh("git", "add", "-A", cwd=tmp)
        sh("git", "commit", "-q", "-m", f"queue request: {req}", cwd=tmp)
        for _ in range(5):
            if sh("git", "push", "-q", "origin", f"HEAD:{STATE_BRANCH}", cwd=tmp, check=False).returncode == 0:
                break
            sh("git", "pull", "-q", "--rebase", "origin", STATE_BRANCH, cwd=tmp)
        else:
            raise SystemExit("could not push the request")
    finally:
        sh("git", "worktree", "remove", "--force", tmp, check=False)
    # A nudge; if it is cancelled in the concurrency group, the heartbeat picks the request up.
    sh("gh", "workflow", "run", "queue.yml", check=False)
    print(f"requested {req}")


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


ENGINE_PATHS = ["src", "Cargo.toml", "Cargo.lock", "build.rs"]


def fast_forward_main(sha, base_sha=None, branch=None):
    """Move main to the tested sha. If main moved since base_sha but only outside
    the engine sources (tooling, docs, logs), the tested binary is unchanged, so
    replay the candidate on top of main instead of retesting."""
    sh("git", "fetch", "-q", "origin", "main")
    main_sha = out("git", "rev-parse", "origin/main")
    if sh("git", "merge-base", "--is-ancestor", main_sha, sha, check=False).returncode != 0:
        if not base_sha or sh("git", "diff", "--quiet", base_sha, main_sha, "--",
                              *ENGINE_PATHS, check=False).returncode != 0:
            return False
        sh("git", "checkout", "-q", "-B", "replay", sha)
        if sh("git", "rebase", "-q", main_sha, check=False).returncode != 0:
            sh("git", "rebase", "--abort", check=False)
            return False
        if sh("git", "diff", "--quiet", sha, "replay", "--", *ENGINE_PATHS,
              check=False).returncode != 0:
            return False
        sha = out("git", "rev-parse", "replay")
        if branch:
            sh("git", "push", "-q", "--force", "origin", f"replay:{branch}")
    sh("git", "push", "-q", "origin", f"{sha}:refs/heads/main")
    return True


def start(item):
    """Prepare and launch one item. Returns the in-flight record, or None if it finished/skipped."""
    branch, mode = item["branch"], item["mode"]
    base = {"branch": branch, "mode": mode}
    if mode == "ltc-check":
        return dispatch(base, out("git", "rev-parse", "origin/main"), item["base"], "main-ltc-check")
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

    return dict(dispatch(base, sha, out("git", "rev-parse", "origin/main"), branch), bench=nodes)


def dispatch(base, sha, base_sha, name):
    """Start the SPRT for an item; its final report calls back into the queue."""
    settings = dict(SETTINGS[base["mode"]], candidate_name=name, base_name=base_sha[:7],
                    callback="queue")
    test_id = f"q-{name}-{datetime.datetime.utcnow().strftime('%m%d-%H%M')}"
    sh("gh", "workflow", "run", "sprt.yml", "--ref", "main",
       "-f", f"test_id={test_id}", "-f", f"candidate={sha}", "-f", f"base={base_sha}",
       "-f", f"settings={json.dumps(settings)}")
    log(dict(base, outcome="dispatched", test_id=test_id, sha=sha, base_sha=base_sha))
    return dict(base, test_id=test_id, sha=sha, base_sha=base_sha, started=now())


def find_result(test_id):
    """Final SPRT result for test_id from its runs, or None while it is still going.

    Lets a scheduled or manual run recover if the sprt.yml callback was lost
    (e.g. a pending queue run superseded in the concurrency group). A last wave
    that finished without a report (build failure, cancellation) yields
    status "broken"."""
    runs = json.loads(out("gh", "run", "list", "--workflow", "sprt.yml", "--limit", "200",
                          "--json", "databaseId,displayTitle,status"))
    waves = [(int(m.group(1)), r) for r in runs
             if (m := re.fullmatch(rf"SPRT {re.escape(test_id)} wave (\d+)", r["displayTitle"]))]
    if not waves:
        return None
    _, run = max(waves, key=lambda w: w[0])
    if run["status"] != "completed":
        return None
    tmp = os.path.join(REPO, ".sprt-result")
    sh("rm", "-rf", tmp)
    # Shard failures can fail the run while its report still succeeded, so
    # judge by the report artifact rather than the run conclusion.
    if sh("gh", "run", "download", str(run["databaseId"]), "-n", "sprt-result", "-D", tmp,
          check=False).returncode != 0:
        return {"test_id": test_id, "status": "broken", "final": True}
    with open(os.path.join(tmp, "sprt-result.json")) as fh:
        res = json.load(fh)
    return res if res.get("final") else None


def parse_adds(spec):
    """'branch:mode,branch2:mode2' -> pending items (mode defaults to stc)."""
    items = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        branch, _, mode = part.partition(":")
        mode = mode or "stc"
        if mode not in SETTINGS and mode != "speedup" or mode == "ltc-check":
            raise SystemExit(f"unknown mode {mode!r} for {branch}")
        items.append({"branch": branch, "mode": mode})
    return items


def advance():
    state = load_state()
    raw = os.environ.get("RESULT", "").strip()
    cur = state.get("current")
    if not raw and cur:
        found = find_result(cur["test_id"])
        if found:
            print(f"recovered final result for {cur['test_id']}")
            raw = json.dumps(found)
    if raw:
        res = json.loads(raw)
        if not cur or res.get("test_id") != cur["test_id"]:
            print(f"ignoring result for {res.get('test_id')}: not the current item")
            save_state(state, "queue: add items")
            return
        verdict = res.get("status")
        record = {k: cur[k] for k in ("branch", "mode", "test_id", "sha")}
        record.update(elo=res.get("elo"), elo_ci95=res.get("elo_ci95"), games=res.get("games"),
                      llr=res.get("llr"), verdict=verdict)
        item = {k: cur[k] for k in ("branch", "mode", "base") if k in cur}
        if verdict == "broken":
            # Infrastructure failure: retry once, then hand to a human.
            if retry(state, item, cur):
                log(dict(record, outcome="retrying", reason="SPRT run ended without a report"))
            else:
                log(dict(record, outcome="needs-human", reason="SPRT run ended without a report, twice"))
        elif cur["mode"] == "ltc-check":
            if verdict == "H0":
                state["paused"] = f"LTC check {cur['test_id']} failed: main regressed vs {cur['base_sha'][:7]}"
                log(dict(record, outcome="regression-alert", reason=state["paused"]))
            else:
                log(dict(record, outcome="check-passed" if verdict == "H1" else "check-inconclusive"))
        elif verdict == "H1":
            if fast_forward_main(cur["sha"], cur.get("base_sha"), cur["branch"]):
                log(dict(record, outcome="merged"))
                schedule_check(state, cur["base_sha"])
            else:
                log(dict(record, outcome="requeued", reason="main moved during test"))
                state["pending"].insert(0, item)
        else:
            log(dict(record, outcome="rejected"))
        state["current"] = None
    elif cur:
        print(f"{cur['test_id']} still in flight; nothing to do")
        save_state(state, "queue: add items")
        return

    if state.get("paused"):
        print(f"paused: {state['paused']} (resume with: python3 scripts/queue.py unpause)")
    while state["pending"] and not state.get("current") and not state.get("paused"):
        item = state["pending"].pop(0)
        try:
            launched = start(item)
            # Keep item fields (retried, base) on the in-flight record.
            state["current"] = dict(item, **launched) if launched else None
        except Exception as exc:  # keep the queue moving past unexpected failures
            sh("git", "rebase", "--abort", check=False)
            reason = f"queue error: {exc}"[:500]
            if retry(state, item, item, back=True):
                log({"branch": item["branch"], "mode": item["mode"], "outcome": "retrying", "reason": reason})
            else:
                log({"branch": item["branch"], "mode": item["mode"], "outcome": "needs-human", "reason": reason})
            state["current"] = None
    save_state(state, f"queue: {state['current']['test_id'] if state.get('current') else 'idle'}")


def retry(state, item, prev, back=False):
    """Requeue an item once after an infrastructure failure. Returns False if it already was."""
    if prev.get("retried"):
        return False
    item = dict(item, retried=True)
    if back:
        state["pending"].append(item)
    else:
        state["pending"].insert(0, item)
    return True


def schedule_check(state, base_sha):
    """After every CHECK_EVERY merges, queue an LTC non-regression of main vs the
    main from CHECK_EVERY merges back (the slow-erosion check in development.md)."""
    state.setdefault("check_base", base_sha)
    state["merges_since_check"] = state.get("merges_since_check", 0) + 1
    if state["merges_since_check"] >= CHECK_EVERY:
        state["pending"].insert(0, {"branch": "main", "mode": "ltc-check", "base": state["check_base"]})
        sh("git", "fetch", "-q", "origin", "main")
        state["check_base"] = out("git", "rev-parse", "origin/main")
        state["merges_since_check"] = 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["advance"]:
        advance()
    elif len(args) >= 2 and args[0] == "add" and set(args[2:]) <= {"--front"}:
        parse_adds(args[1])  # validate before pushing
        request({"add": args[1], "front": "--front" in args})
    elif args == ["unpause"]:
        request({"unpause": True})
    else:
        sys.exit("usage: queue.py advance            (CI; RESULT env = final SPRT result JSON)\n"
                 "       queue.py add <branch:mode,...> [--front]\n"
                 "       queue.py unpause")
