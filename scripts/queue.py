#!/usr/bin/env python3
"""Serial merge queue, run by .github/workflows/queue.yml.

State lives on the `queue-state` branch (queue.json + log.jsonl), so `main`
only ever moves by fast-forward merges of candidates that passed.

Each `advance`:
  1. If RESULT holds the final SPRT result for the current item: H1 ->
     fast-forward main to the tested candidate (replayed over main if main
     moved only outside the engine or by behaviour-identical speedup merges,
     see fast_forward_main; otherwise requeue it at the front); anything
     else -> drop. Log it.
  2. Merge every pending 'speedup' item (bench and evalhash identical to
     main's, no games), even while an SPRT or ladder is in flight: they cost
     no runners, and their merges are recorded in state["identical_merges"]
     so the in-flight candidate can still be replayed over them.
  3. If nothing is in flight, start pending items until one is dispatched,
     picked by pick_next (front items, then scheduled ladders/ltc-checks in
     place, else the best expected Elo per runner-hour from log.jsonl base
     rates): rebase onto main, build, test, perft, selftest, bench (refreshing
     the Bench: trailer), push the rebased branch (with a lease), pin the
     tested sha under refs/queue/<test_id>, and dispatch an SPRT whose final
     report calls back here. Conflicts and failed checks are logged as
     needs-human and skipped.
     Datagen bursts: when the best pending item scores below the "burst"
     settings' below_score (or nothing is pending), the foreground runners
     (MAX_FOREGROUND_SHARDS, from the "lanes" settings) generate one datagen
     seed instead (see burst_reason); the item keeps its place and the queue
     re-picks when the burst ends.
  4. Keep the background lane (datagen) busy, within the runner budget left
     by other workflows (ladders, SPSA, sweeps, hand-started SPRTs);
     datagen.yml calls back here when a run ends, so the next seed starts at
     once (and a burst ends).

State is saved after every merge and dispatch, not only at the end, so a run
killed by the job timeout loses at most the step it was in.

Item modes: the SPRT modes in queue-settings.json (stc, nonreg, ltc-gainer,
net-gate; ltc-check is queued automatically), speedup (no SPRT; bench and
evalhash must be unchanged) and ladder (CCRL ladder on main).
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

# Runner split (queue-settings.json "lanes"): the background datagen lane
# holds background_parallel runners, the queue's own job 1, and the foreground
# (SPRTs, ladders, bursts) gets the rest of the 20; anything wider would
# starve both. background_shards must be a multiple of background_parallel.
RUNNERS = 20
_LANES = _CFG.get("lanes", {})
BACKGROUND_PARALLEL = _LANES.get("background_parallel", 6)
BACKGROUND_SHARDS = _LANES.get("background_shards", 3 * BACKGROUND_PARALLEL)
MAX_FOREGROUND_SHARDS = RUNNERS - 1 - BACKGROUND_PARALLEL


def validate_settings(cfg):
    """Fail loudly at load time on settings that would oversubscribe runners."""
    lanes = cfg.get("lanes", {})
    bpar = lanes.get("background_parallel", BACKGROUND_PARALLEL)
    bshards = lanes.get("background_shards", 3 * bpar if isinstance(bpar, int) else None)
    if not isinstance(bpar, int) or not 1 <= bpar <= RUNNERS - 2:
        raise SystemExit(f"queue-settings.json: lanes.background_parallel must be 1..{RUNNERS - 2}, "
                         f"got {bpar!r}")
    if not isinstance(bshards, int) or bshards < 1 or bshards % bpar:
        raise SystemExit(f"queue-settings.json: lanes.background_shards ({bshards!r}) must be a "
                         f"positive multiple of lanes.background_parallel ({bpar}), or the remainder "
                         f"runs as a second, mostly idle round (datagen.yml)")
    max_fg = RUNNERS - 1 - bpar
    bad = [f"{mode} (shards={s.get('shards')!r})" for mode, s in cfg["modes"].items()
           if not isinstance(s.get("shards"), int) or not 1 <= s["shards"] <= max_fg]
    if bad:
        raise SystemExit(f"queue-settings.json: foreground modes must use 1..{max_fg} "
                         f"shards (the rest of the {RUNNERS} runners are the background lane and the "
                         f"queue's own job); fix: {', '.join(bad)}")
    burst = cfg.get("burst")
    if burst and burst.get("enabled"):
        par, shards = burst.get("max_parallel"), burst.get("shards")
        if not isinstance(par, int) or not 1 <= par <= max_fg:
            raise SystemExit(f"queue-settings.json: burst.max_parallel must be 1..{max_fg} "
                             f"(the foreground runners), got {par!r}")
        if not isinstance(shards, int) or shards < 1 or shards % par:
            raise SystemExit(f"queue-settings.json: burst.shards ({shards!r}) must be a positive "
                             f"multiple of burst.max_parallel ({par}), or the remainder runs as a "
                             f"second, mostly idle round (datagen.yml)")
    # The ladder runs on the foreground too; start() passes max_parallel =
    # MAX_FOREGROUND_SHARDS unless the ladder settings set a smaller one.
    lpar = cfg.get("ladder", {}).get("max_parallel")
    if lpar is not None:
        try:
            ok = 1 <= int(lpar) <= max_fg
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise SystemExit(f"queue-settings.json: ladder.max_parallel must be 1..{max_fg} "
                             f"(the foreground runners), got {lpar!r}")


validate_settings(_CFG)
SETTINGS = _CFG["modes"]
BURST = _CFG.get("burst", {})
CHECK_EVERY = _CFG["check_every"]
# A CCRL ladder on main every LADDER_EVERY merges (ccrl-ladder.yml logs it).
LADDER_EVERY = _CFG["ladder_every"]
LADDER = _CFG["ladder"]


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
        if req.get("remove"):
            # queue.py remove <branch>[:mode]: drop waiting items (never the one in flight).
            branch, _, mode = req["remove"].partition(":")
            gone = [p for p in state["pending"] if p["branch"] == branch and (not mode or p["mode"] == mode)]
            state["pending"] = [p for p in state["pending"] if p not in gone]
            for p in gone:
                log({"branch": p["branch"], "mode": p["mode"], "outcome": "removed", "reason": "queue.py remove"})
            os.remove(path)
            print(f"inbox {name}: removed {len(gone)} item(s) for {req['remove']}")
            continue
        adds = parse_adds(req.get("add", ""), req.get("expect"))
        if req.get("front"):
            # Front items run in order before anything is scored (pick_next).
            adds = [dict(a, front=True) for a in adds]
        # Re-adding a waiting branch moves it instead of testing it twice.
        keys = {(a["branch"], a["mode"]) for a in adds}
        state["pending"] = [p for p in state["pending"] if (p["branch"], p["mode"]) not in keys]
        if req.get("front"):
            state["pending"][:0] = adds
        else:
            state["pending"] += adds
        if req.get("unpause"):
            state.pop("paused", None)
        if req.get("preempt"):
            preempt(state, len(adds))
        os.remove(path)
        print(f"inbox {name}: {req}")
    return state


def preempt(state, after):
    """Stop the in-flight SPRT so the items just put at the front run first.

    The interrupted item goes back to the queue right behind them (position
    `after`) and is tested from scratch later; ladders and LTC checks are never
    interrupted. Its runs are cancelled, so a late callback for it is ignored
    as "not the current item"."""
    cur = state.get("current")
    if not cur or cur.get("mode") in ("ladder", "ltc-check", BURST_MODE):  # bursts are short
        return
    # A wave dispatched moments ago (or the next wave its report is starting
    # right now) may not be listed yet: remember the title and keep
    # cancelling it on later runs too (sweep_cancelled).
    until = (datetime.datetime.utcnow() + datetime.timedelta(hours=CANCEL_SWEEP_HOURS)
             ).isoformat(timespec="seconds") + "Z"
    state.setdefault("cancel_sweep", []).append({"prefix": f"SPRT {cur['test_id']} ", "until": until})
    sweep_cancelled(state)
    item = dict(requeue_item(cur), front=True)
    state["pending"].insert(after, item)
    state["current"] = None
    drop_tested_ref(cur["test_id"])
    log({"branch": cur["branch"], "mode": cur["mode"], "test_id": cur["test_id"],
         "outcome": "preempted", "reason": "a front item with --preempt was added"})


CANCEL_SWEEP_HOURS = 12


def sweep_cancelled(state):
    """Cancel any live sprt.yml run whose title matches a preempted test
    (state["cancel_sweep"]); entries expire after CANCEL_SWEEP_HOURS."""
    sweep = [s for s in state.get("cancel_sweep", [])
             if _parse_time(s["until"]) > datetime.datetime.utcnow()]
    if sweep:
        state["cancel_sweep"] = sweep
    else:
        state.pop("cancel_sweep", None)
        return
    res = sh("gh", "run", "list", "-w", "sprt.yml", "-L", "50",
             "--json", "databaseId,displayTitle,status", check=False)
    try:
        runs = json.loads(res.stdout or "[]")
    except ValueError:
        runs = []
    for run in runs:
        title = run.get("displayTitle") or ""
        if run.get("status") != "completed" and any(title.startswith(s["prefix"]) for s in sweep):
            print(f"cancelling preempted run {run['databaseId']}: {title}")
            sh("gh", "run", "cancel", str(run["databaseId"]), check=False)


def tested_ref(test_id):
    """Private ref that pins an SPRT's tested sha while it runs: the branch can
    be force-pushed (by a human or a replay) without losing the commit."""
    return f"refs/queue/{test_id}"


def drop_tested_ref(test_id):
    sh("git", "push", "-q", "origin", f":{tested_ref(test_id)}", check=False)


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
    """main's bench as measured when the queue last merged (main-bench.json), if
    main has not changed the engine since; else the latest Bench: trailer, which
    can be stale (a cherry-picked trailer once hid a net change)."""
    try:
        with open(os.path.join(STATE, "main-bench.json")) as fh:
            rec = json.load(fh)
        if sh("git", "diff", "--quiet", rec["sha"], "origin/main", "--", *ENGINE_PATHS,
              check=False).returncode == 0:
            return rec["nodes"]
    except (OSError, ValueError, KeyError):
        pass
    for body in out("git", "log", "-z", "--format=%B", "origin/main").split("\0"):
        m = BENCH_RE.findall(body)
        if m:
            return int(m[-1])
    return None


# Same trailer pattern as scripts/check_bench.py (what CI enforces).
BENCH_RE = re.compile(r"^Bench:\s*(\d+)\s*$", re.M)
# For rewriting: one line only (BENCH_RE's \s* can run across newlines).
_BENCH_LINE_RE = re.compile(r"^Bench:[ \t]*\d+[ \t]*$", re.M)


def with_bench_trailer(msg, nodes):
    """msg with every Bench: trailer set to nodes, or one added: before the
    first Co-Authored-By: line if there is one, else at the end."""
    if _BENCH_LINE_RE.search(msg):
        return _BENCH_LINE_RE.sub(f"Bench: {nodes}", msg)
    m = re.search(r"^Co-Authored-By:", msg, re.M | re.I)
    if m:
        return f"{msg[:m.start()]}Bench: {nodes}\n\n{msg[m.start():]}"
    return f"{msg.rstrip()}\n\nBench: {nodes}\n"


def run_engine(cmds):
    return subprocess.run(["./target/release/rustfish"], input=cmds, cwd=REPO,
                          capture_output=True, text=True).stdout


ENGINE_PATHS = ["src", "net/nnue.bin", "Cargo.toml", "Cargo.lock", "build.rs"]


def engine_moves(base_sha, main_sha, identical):
    """How main moved since base_sha, as far as the engine is concerned: False
    if no commit in base_sha..main_sha touches ENGINE_PATHS, True if the only
    ones that do are recorded behaviour-identical merges (speedups), None if
    any other engine change landed (or main was rewritten)."""
    if sh("git", "merge-base", "--is-ancestor", base_sha, main_sha, check=False).returncode != 0:
        return None
    known, touched = set(identical), False
    for c in out("git", "rev-list", f"{base_sha}..{main_sha}").split():
        if sh("git", "diff", "--quiet", f"{c}^", c, "--", *ENGINE_PATHS, check=False).returncode == 0:
            continue
        if c not in known:
            return None
        touched = True
    if not touched and sh("git", "diff", "--quiet", base_sha, main_sha, "--", *ENGINE_PATHS,
                          check=False).returncode != 0:
        return None
    return touched


def checked_bench():
    """Build the checkout and run the start() checks (cargo test, perft,
    selftest); bench nodes, or None if anything failed. The replay path
    compares the evalhash separately (fast_forward_main)."""
    if sh("cargo", "build", "--release", "-q", check=False).returncode != 0:
        return None
    if sh("cargo", "test", "--release", "-q", check=False).returncode != 0:
        return None
    checks = run_engine("perfttest\nselftest 300\nquit\n")
    if not re.search(r"perfttest summary (\d+)/\1 passed", checks) or "selftest ok" not in checks:
        return None
    m = re.search(r"Nodes searched\s*:\s*(\d+)", run_engine("bench\nquit\n"))
    return int(m.group(1)) if m else None


MAIN_PUSH_ATTEMPTS = 3


def fast_forward_main(sha, base_sha=None, branch=None, bench=None, identical=(), test_id=None,
                      evalhash=None):
    """Move main to the tested sha. Returns None on success, else why not.

    If main moved since base_sha, replay the candidate on top of main instead
    of retesting when the tested binary's behaviour is unchanged: main moved
    only outside the engine sources (tooling, docs, logs; the replayed engine
    tree is then byte-identical), or also by merges recorded in `identical`
    (speedups: bench and evalhash equal to main's). In the second case the
    replayed tree differs, so it is rebuilt, checked, and must give exactly
    the tested candidate's `bench` (and `evalhash`, when recorded) before
    main moves.

    The push to main is not forced, so it loses to anyone pushing main at the
    same moment (netloop rows, ladder logs, hand-pushed plan commits): then
    main is re-fetched and the same decision is made again (replay or refuse),
    MAIN_PUSH_ATTEMPTS times. The branch is updated to the replayed commit
    only after main moved, under a lease on the tested sha."""
    if test_id:  # the pinned tested sha (dispatch), even if the branch was force-pushed
        sh("git", "fetch", "-q", "origin", f"+{tested_ref(test_id)}:{tested_ref(test_id)}", check=False)
    if branch:
        sh("git", "fetch", "-q", "origin", branch, check=False)
    if sh("git", "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode != 0:
        return f"tested sha {sha[:7]} is on no ref (branch force-pushed during the test?)"
    verified = None  # replay commit whose engine tree passed checked_bench
    for attempt in range(MAIN_PUSH_ATTEMPTS):
        sh("git", "fetch", "-q", "origin", "main")
        main_sha = out("git", "rev-parse", "origin/main")
        target = sha
        if sh("git", "merge-base", "--is-ancestor", main_sha, sha, check=False).returncode != 0:
            if not base_sha:
                return "main moved"
            moved = engine_moves(base_sha, main_sha, identical)
            if moved is None:
                return "main moved during test (engine change)"
            sh("git", "checkout", "-q", "-B", "replay", sha)
            if sh("git", "rebase", "-q", main_sha, check=False).returncode != 0:
                sh("git", "rebase", "--abort", check=False)
                return "main moved during test (replay conflict)"
            if sh("git", "diff", "--quiet", sha, "replay", "--", *ENGINE_PATHS,
                  check=False).returncode != 0 and not (
                    verified and sh("git", "diff", "--quiet", verified, "replay", "--", *ENGINE_PATHS,
                                    check=False).returncode == 0):
                if not moved or bench is None:
                    return "main moved during test (replayed engine differs)"
                got = checked_bench()
                if got != bench:
                    return f"main moved during test (replayed bench {got} != tested {bench})"
                if evalhash:
                    ehash = eval_hash(os.path.join(REPO, "target", "release", "rustfish"))
                    if ehash != evalhash:
                        return f"main moved during test (replayed evalhash {ehash} != tested {evalhash})"
                verified = out("git", "rev-parse", "replay")
            target = out("git", "rev-parse", "replay")
        if sh("git", "push", "-q", "origin", f"{target}:refs/heads/main", check=False).returncode == 0:
            break
        print(f"push to main refused (attempt {attempt + 1}/{MAIN_PUSH_ATTEMPTS}): main moved; re-fetching")
    else:
        return f"main kept moving: push refused {MAIN_PUSH_ATTEMPTS} times"
    if target != sha and branch:
        lease = f"--force-with-lease=refs/heads/{branch}:{sha}"
        if sh("git", "push", "-q", lease, "origin", f"{target}:refs/heads/{branch}",
              check=False).returncode != 0:
            log({"branch": branch, "outcome": "needs-human", "sha": target,
                 "reason": f"merged (replayed as {target[:7]}), but {branch} changed during the "
                           f"test, so it was left as is: commits added after {sha[:7]} were not merged"})
    if bench:
        with open(os.path.join(STATE, "main-bench.json"), "w") as fh:
            json.dump({"sha": target, "nodes": bench}, fh)
            fh.write("\n")
    return None


def start(item, state=None, priority=None):
    """Prepare and launch one item. Returns the in-flight record, or None if it finished/skipped."""
    branch, mode = item["branch"], item["mode"]
    base = {"branch": branch, "mode": mode}
    if mode == "ltc-check":
        return dispatch(base, out("git", "rev-parse", "origin/main"), item["base"], "main-ltc-check",
                        priority)
    if mode == "ladder":
        main_sha = out("git", "rev-parse", "origin/main")
        test_id = f"q-ladder-{datetime.datetime.utcnow().strftime('%m%d-%H%M')}"
        ladder = dict({"max_parallel": MAX_FOREGROUND_SHARDS}, **LADDER)
        sh("gh", "workflow", "run", "ccrl-ladder.yml", "--ref", "main", "-f", f"test_id={test_id}",
           *[a for k, v in ladder.items() for a in ("-f", f"{k}={v}")])
        # On the foreground runners (max_parallel = MAX_FOREGROUND_SHARDS);
        # the background lane keeps generating data meanwhile.
        log(dict(base, outcome="dispatched", test_id=test_id, sha=main_sha,
                 **({"priority": priority} if priority else {})))
        return dict(base, test_id=test_id, sha=main_sha, base_sha=main_sha, started=now())
    sh("git", "fetch", "-q", "origin", "main", branch)
    pushed = out("git", "rev-parse", f"origin/{branch}")  # lease for the push below
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
    stale = [int(b) for b in BENCH_RE.findall(msg) if int(b) != nodes]
    if nodes != mnodes or stale:
        new_msg = with_bench_trailer(msg, nodes)
        if new_msg != msg:
            sh("git", "commit", "-q", "--amend", "-m", new_msg)
    sha = out("git", "rev-parse", "candidate")
    # Leased, not forced: if someone pushed to the branch while it was being
    # prepared, keep their commits and try again with them.
    if sh("git", "push", "-q", f"--force-with-lease=refs/heads/{branch}:{pushed}", "origin",
          f"candidate:refs/heads/{branch}", check=False).returncode != 0:
        reason = f"{branch} changed while it was being prepared (push lease on {pushed[:7]} failed)"
        if state is not None and retry(state, item, item):
            log(dict(base, outcome="retrying", reason=reason))
        else:
            log(dict(base, outcome="needs-human", reason=reason))
        return None

    if mode == "speedup":
        mismatch = None if nodes != mnodes else eval_mismatch()
        if nodes != mnodes:
            log(dict(base, outcome="needs-human", reason=f"speedup changed bench {mnodes} -> {nodes}"))
        elif mismatch:
            log(dict(base, outcome="needs-human", reason=f"speedup changed the eval ({mismatch}); "
                     "eval changes need an SPRT: re-add as stc or nonreg"))
        else:
            commits = out("git", "rev-list", "origin/main..candidate").split()
            refused = fast_forward_main(sha, bench=nodes)
            if refused and refused.startswith("main ") and state is not None:
                # main moved while it was being built: nothing wrong with the
                # speedup, so rebase and check it again first thing.
                state["pending"].insert(0, dict(item, front=True))
                log(dict(base, outcome="requeued", reason=f"speedup merge refused: {refused}"))
            elif refused:
                log(dict(base, outcome="needs-human", reason=f"speedup merge refused: {refused}"))
            else:
                if state is not None:
                    # Behaviour-identical: an in-flight SPRT can be replayed over these.
                    ident = state.setdefault("identical_merges", [])
                    ident += commits
                    del ident[:-IDENTICAL_KEEP]
                log(dict(base, outcome="merged", sha=sha, reason="speedup, bench identical"))
        return None

    # The tested binary's evalhash, so a replay over speedups can be held to it.
    ehash = eval_hash(os.path.join(REPO, "target", "release", "rustfish"))
    return dict(dispatch(base, sha, out("git", "rev-parse", "origin/main"), branch, priority),
                bench=nodes, **({"evalhash": ehash} if ehash else {}))


def eval_hash(binary):
    try:
        res = subprocess.run([binary], input="evalhash\nquit\n", capture_output=True, text=True)
    except OSError:
        return None
    m = re.search(r"Eval hash\s*:\s*([0-9a-f]+)", res.stdout)
    return m.group(1) if m else None


def eval_mismatch():
    """Compare the candidate's `evalhash` with origin/main's. Bench nodes only
    cover search at bench depth; a speedup must also leave the eval itself
    bit-identical. Returns a description of the mismatch, or None."""
    cand = eval_hash(os.path.join(REPO, "target", "release", "rustfish"))
    wt = os.path.join(REPO, ".main-build")
    sh("git", "worktree", "remove", "--force", wt, check=False)
    sh("git", "worktree", "add", "-q", "--detach", wt, "origin/main")
    try:
        sh("cargo", "build", "--release", "-q", cwd=wt)
        main = eval_hash(os.path.join(wt, "target", "release", "rustfish"))
    finally:
        sh("git", "worktree", "remove", "--force", wt, check=False)
    if main is None:
        print("main has no evalhash command yet; skipping the eval check")
        return None
    if cand != main:
        return f"evalhash {main} -> {cand}"
    return None


def dispatch(base, sha, base_sha, name, priority=None):
    """Start the SPRT for an item; its final report calls back into the queue."""
    settings = dict(SETTINGS[base["mode"]], candidate_name=name, base_name=base_sha[:7],
                    callback="queue")
    test_id = f"q-{name}-{datetime.datetime.utcnow().strftime('%m%d-%H%M')}"
    # Pin the tested sha: a force-push to the branch during the test must not
    # leave the commit fast_forward_main needs on no ref. Best effort: the
    # branch still holds it unless someone rewrites it.
    if sh("git", "push", "-q", "--force", "origin", f"{sha}:{tested_ref(test_id)}",
          check=False).returncode != 0:
        print(f"warning: could not pin {sha[:7]} as {tested_ref(test_id)}")
    sh("gh", "workflow", "run", "sprt.yml", "--ref", "main",
       "-f", f"test_id={test_id}", "-f", f"candidate={sha}", "-f", f"base={base_sha}",
       "-f", f"settings={json.dumps(settings)}")
    log(dict(base, outcome="dispatched", test_id=test_id, sha=sha, base_sha=base_sha,
             **({"priority": priority} if priority else {})))
    return dict(base, test_id=test_id, sha=sha, base_sha=base_sha, started=now())


def find_result(test_id):
    """Final SPRT result for test_id from its runs, or None while it is still going.

    Lets a scheduled or manual run recover if the sprt.yml callback was lost
    (e.g. a pending queue run superseded in the concurrency group). A last wave
    that finished without a report (build failure, cancellation) yields
    status "broken"."""
    ladder = test_id.startswith("q-ladder-")
    runs = json.loads(out("gh", "run", "list", "--workflow",
                          "ccrl-ladder.yml" if ladder else "sprt.yml", "--limit", "200",
                          "--json", "databaseId,displayTitle,status"))
    pattern = rf"CCRL ladder {re.escape(test_id)}()" if ladder else \
        rf"SPRT {re.escape(test_id)} wave (\d+)"
    waves = [(int(m.group(1) or 1), r) for r in runs
             if (m := re.fullmatch(pattern, r["displayTitle"]))]
    if not waves:
        return None
    _, run = max(waves, key=lambda w: w[0])
    if run["status"] != "completed":
        return None
    tmp = os.path.join(REPO, ".sprt-result")
    sh("rm", "-rf", tmp)
    # Shard failures can fail the run while its report still succeeded, so
    # judge by the report artifact rather than the run conclusion.
    name = "ccrl-ladder-results" if ladder else "sprt-result"
    if sh("gh", "run", "download", str(run["databaseId"]), "-n", name, "-D", tmp,
          check=False).returncode != 0:
        return {"test_id": test_id, "status": "broken", "final": True}
    if ladder:
        with open(os.path.join(tmp, "results.json")) as fh:
            fit = json.load(fh)
        return {"test_id": test_id, "status": "ladder", "final": True, "rating": fit.get("rating"),
                "ci": fit.get("ci"), "games": fit.get("games")}
    with open(os.path.join(tmp, "sprt-result.json")) as fh:
        res = json.load(fh)
    return res if res.get("final") else None


def parse_adds(spec, expect=None):
    """'branch:mode,branch2:mode2' -> pending items (mode defaults to stc).

    `expect` (Elo) replaces the base-rate gain in pick_next's score for these
    items, e.g. for a net with a big validation-loss lead."""
    items = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        branch, _, mode = part.partition(":")
        mode = mode or "stc"
        if mode not in SETTINGS and mode not in ("speedup", "ladder") or mode == "ltc-check":
            raise SystemExit(f"unknown mode {mode!r} for {branch}")
        item = {"branch": branch, "mode": mode}
        if expect is not None:
            item["expect"] = float(expect)
        items.append(item)
    return items


# Priority. Base rates per SPRT mode come from log.jsonl, shrunk towards these
# priors with PSEUDO pseudo-samples so a mode with few results gets sane
# numbers. Gains are clipped to GAIN_CAP: a handful of +100 bug fixes would
# otherwise dominate the mean.
PRIOR_P, PRIOR_GAIN, PSEUDO, GAIN_CAP = 0.25, 5.0, 3, 30.0
PRIOR_HOURS = {"stc": 1.0, "nonreg": 1.5, "ltc-gainer": 3.0, "net-gate": 0.3}
SCHEDULED = ("ladder", "ltc-check")  # measurement: never reordered
BURST_MODE = "datagen-burst"
MAX_PASSED_OVER = 3
IDENTICAL_KEEP = 500


def _parse_time(stamp):
    return datetime.datetime.fromisoformat(stamp.rstrip("Z"))


def base_rates(entries):
    """Per SPRT mode: acceptance rate, mean (clipped) Elo of accepted tests,
    mean wall-clock hours of accepted and of other final tests (dispatched ->
    final row), and the sample count, each shrunk towards the priors."""
    started, stats = {}, {}
    for e in entries:
        if e.get("outcome") == "dispatched" and e.get("test_id"):
            started[e["test_id"]] = e.get("logged_at")
            continue
        mode = e.get("mode")
        if e.get("outcome") not in ("merged", "rejected", "requeued") or mode not in SETTINGS \
                or e.get("verdict") not in ("H1", "H0", "inconclusive"):
            continue
        st = stats.setdefault(mode, {"n": 0, "acc": 0, "gain": 0.0, "h_acc": [], "h_rej": []})
        st["n"] += 1
        ok = e["verdict"] == "H1"
        if ok:
            st["acc"] += 1
            st["gain"] += max(0.0, min(GAIN_CAP, float(e.get("elo") or 0.0)))
        try:
            hours = (_parse_time(e["logged_at"]) - _parse_time(started[e["test_id"]])).total_seconds() / 3600
            st["h_acc" if ok else "h_rej"].append(hours)
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    rates = {}
    for mode in SETTINGS:
        st = stats.get(mode, {"n": 0, "acc": 0, "gain": 0.0, "h_acc": [], "h_rej": []})
        h0 = PRIOR_HOURS.get(mode, 1.0)
        rates[mode] = {
            "n": st["n"],
            "p_accept": (st["acc"] + PSEUDO * PRIOR_P) / (st["n"] + PSEUDO),
            "gain": (st["gain"] + PSEUDO * PRIOR_GAIN) / (st["acc"] + PSEUDO),
            "hours_accept": (sum(st["h_acc"]) + PSEUDO * h0) / (len(st["h_acc"]) + PSEUDO),
            "hours_reject": (sum(st["h_rej"]) + PSEUDO * h0) / (len(st["h_rej"]) + PSEUDO),
        }
    return rates


def priority_of(item, rates):
    """Expected Elo per runner-hour: p_accept * gain / expected runner-hours,
    where runner-hours = shards * (p * hours_accept + (1 - p) * hours_reject).
    Speedups cost no runners and score infinity."""
    mode = item["mode"]
    if mode == "speedup":
        return {"rule": "speedup", "score": float("inf")}
    r = rates.get(mode)
    if r is None:
        return {"rule": "score", "score": 0.0, "reason": f"no base rate for {mode}"}
    p = r["p_accept"]
    expect = item.get("expect")
    gain = expect if expect is not None else r["gain"]
    runner_hours = SETTINGS[mode]["shards"] * (p * r["hours_accept"] + (1 - p) * r["hours_reject"])
    return {"rule": "score", "score": round(p * gain / runner_hours, 4), "p_accept": round(p, 3),
            "gain": round(gain, 2), "gain_source": "expect" if expect is not None else "base-rate",
            "runner_hours": round(runner_hours, 2), "samples": r["n"]}


def pick_next(pending, rates):
    """Index of the item to start next, and the priority record for its log.

    Items added with --front (and requeued/preempted ones) go first, in order.
    A ladder or ltc-check keeps its place: nothing queued behind it can jump
    ahead, so measurement isn't starved. Among the items before it, the best
    score wins (ties: the earlier one), except that an item jumped over
    max(MAX_PASSED_OVER, number of items waiting) times goes next (cheap stc items would otherwise starve a
    costly ltc-gainer forever)."""
    end = next((i for i, p in enumerate(pending) if p["mode"] in SCHEDULED), len(pending))
    for i, p in enumerate(pending[:end]):
        if p.get("front"):
            return i, {"rule": "front"}
    if end == 0:
        return 0, {"rule": "scheduled"}
    # The guard scales with the queue: with N items waiting, every pick passes
    # over up to N-1 of them, so a fixed 3 turned priority into FIFO.
    limit = max(MAX_PASSED_OVER, end)
    for i, p in enumerate(pending[:end]):
        if p.get("passed_over", 0) >= limit:
            return i, dict(priority_of(p, rates), rule="starvation-guard", passed_over=p["passed_over"])
    scored = [(priority_of(p, rates), i) for i, p in enumerate(pending[:end])]
    best, idx = max(scored, key=lambda s: (s[0]["score"], -s[1]))
    for p in pending[:idx]:
        p["passed_over"] = p.get("passed_over", 0) + 1
    return idx, dict(best, considered=end)


def read_log():
    entries = []
    try:
        with open(os.path.join(STATE, "log.jsonl")) as fh:
            for line in fh:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return entries


def take_seed(state):
    """The next datagen seed, shared by the background lane and bursts (never reused)."""
    seed = state.get("next_seed", 14)
    state["next_seed"] = seed + 1
    return seed


def datagen_title_re(seed):
    """datagen.yml's run-name: "Datagen <ref> seed <seed> (...)"."""
    return re.compile(rf"Datagen \S+ seed {int(seed)} \(")


def dispatch_datagen(seed, shards, games, nodes, max_parallel):
    sh("gh", "workflow", "run", "datagen.yml", "--ref", "main", "-f", "ref=main",
       "-f", f"shards={shards}", "-f", f"games_per_shard={games}", "-f", f"nodes={nodes}",
       "-f", f"seed={seed}", "-f", f"max_parallel={max_parallel}")


# Runners other workflows take while they run, charged against the
# background lane (the foreground and the queue's own job are already
# reserved). Their max-parallel: spsa.yml and train-sweep.yml 16, a
# hand-started ladder or ci.py SPRT the foreground width. Conservative: a run
# in its last (report) job still counts in full until it completes.
OTHER_WORKFLOWS = {"spsa.yml": 16, "train-sweep.yml": 16,
                   "ccrl-ladder.yml": MAX_FOREGROUND_SHARDS, "sprt.yml": MAX_FOREGROUND_SHARDS}


# The queue's own SPRTs and ladders (test ids q-...) are the foreground.
QUEUE_TITLES = ("SPRT q-", "CCRL ladder q-")


def other_runners():
    """Runners held by live runs of OTHER_WORKFLOWS that the queue did not
    start (its own SPRT waves and ladders are the foreground, whose runners
    are reserved; a just-finished wave still uploading must not count), and
    their titles."""
    used, who = 0, []
    for wf, width in OTHER_WORKFLOWS.items():
        runs = json.loads(out("gh", "run", "list", "--workflow", wf, "--limit", "20",
                              "--json", "databaseId,status,displayTitle"))
        for r in runs:
            title = r.get("displayTitle") or ""
            if r["status"] == "completed" or title.startswith(QUEUE_TITLES):
                continue
            used += width
            who.append(title or wf)
    return used, who


def background_lane(state):
    """Keep the background runners (BACKGROUND_PARALLEL) busy: when no other
    datagen run is queued or running, generate the next datagen seed from main
    (only merged nets generate data). A foreground burst's run
    (state["current"]) uses the foreground runners, so it does not block this
    lane. Runners held by other workflows (other_runners) shrink the run to
    what is left of the lane, keeping its rounds, or skip it."""
    # datagen.yml's final job dispatches the queue with its own run id
    # (FINISHED_RUN): that run is still in progress while it calls back.
    finished = os.environ.get("FINISHED_RUN", "").strip()
    cur = state.get("current") or {}
    burst_re = datagen_title_re(cur["seed"]) if cur.get("mode") == BURST_MODE else None
    runs = json.loads(out("gh", "run", "list", "--workflow", "datagen.yml", "--limit", "10",
                          "--json", "databaseId,status,displayTitle"))
    if any(r["status"] != "completed" and str(r.get("databaseId")) != finished
           and not (burst_re and burst_re.match(r.get("displayTitle") or "")) for r in runs):
        return
    used, who = other_runners()
    par = min(BACKGROUND_PARALLEL, BACKGROUND_PARALLEL - used)
    if par < 1:
        print(f"background lane: skipped, other runs hold {used} runners: {who}")
        return
    shards = BACKGROUND_SHARDS // BACKGROUND_PARALLEL * par
    seed = take_seed(state)
    dispatch_datagen(seed, shards, 15000, 5000, par)
    row = {"branch": "main", "mode": "datagen", "outcome": "dispatched", "seed": seed,
           "sha": out("git", "rev-parse", "origin/main"), "reason": "background lane idle"}
    if par < BACKGROUND_PARALLEL:
        row.update(shards=shards, max_parallel=par,
                   reason=f"background lane idle; shrunk, other runs hold {used} runners: {who}")
    log(row)


def burst_reason(state, pending, rates):
    """Why the foreground should run a datagen burst now instead of the next
    item, or None. A burst runs when nothing is pending (burst.when_idle), or
    when the item pick_next would start scores below burst.below_score, at
    most burst.max_consecutive times in a row while items wait. Front and
    scheduled (ladder/ltc-check) items never wait for a burst; a
    starvation-guard pick can, within the cap."""
    if not BURST.get("enabled"):
        return None
    if not pending:
        return "foreground idle" if BURST.get("when_idle", True) else None
    if state.get("consecutive_bursts", 0) >= BURST.get("max_consecutive", 2):
        return None
    # Peek on a copy: a burst passes nobody over, so the starvation guard's
    # passed_over counts only move when an item really runs.
    idx, pri = pick_next(json.loads(json.dumps(pending)), rates)
    item = pending[idx]
    if pri.get("rule") not in ("score", "starvation-guard"):
        return None
    threshold = BURST.get("below_score", 0.0)
    if pri["score"] < threshold:
        return f"best item {item['branch']}:{item['mode']} scores {pri['score']} < {threshold}"
    return None


def start_burst(state, reason, pending):
    """Dispatch one datagen seed on the foreground runners; returns its in-flight record."""
    seed = take_seed(state)
    dispatch_datagen(seed, BURST["shards"], BURST.get("games_per_shard", 15000),
                     BURST.get("nodes", 5000), BURST["max_parallel"])
    # Only bursts taken while items wait count towards the cap.
    state["consecutive_bursts"] = state.get("consecutive_bursts", 0) + 1 if pending else 0
    main_sha = out("git", "rev-parse", "origin/main")
    test_id = f"q-burst-s{seed}"
    log({"branch": "main", "mode": BURST_MODE, "outcome": "dispatched", "test_id": test_id,
         "seed": seed, "sha": main_sha, "reason": reason,
         "consecutive": state["consecutive_bursts"]})
    return {"branch": "main", "mode": BURST_MODE, "test_id": test_id, "seed": seed,
            "sha": main_sha, "base_sha": main_sha, "started": now()}


BURST_LOST_HOURS = 2.0


def burst_done(cur):
    """True once the burst's datagen run has ended: completed, or it is the run
    calling back right now (FINISHED_RUN, still in progress while it does).
    A run that never shows up within BURST_LOST_HOURS counts as ended too, so
    a lost dispatch cannot wedge the foreground."""
    finished = os.environ.get("FINISHED_RUN", "").strip()
    runs = json.loads(out("gh", "run", "list", "--workflow", "datagen.yml", "--limit", "20",
                          "--json", "databaseId,status,displayTitle"))
    pat = datagen_title_re(cur["seed"])
    mine = [r for r in runs if pat.match(r.get("displayTitle") or "")]
    if not mine:
        try:
            age = datetime.datetime.utcnow() - _parse_time(cur["started"])
        except (KeyError, TypeError, ValueError, AttributeError):
            return False
        return age.total_seconds() > BURST_LOST_HOURS * 3600
    return all(r["status"] == "completed" or str(r.get("databaseId")) == finished for r in mine)


def requeue_item(cur):
    """The pending item to put back for an in-flight record."""
    return {k: cur[k] for k in ("branch", "mode", "base", "expect") if k in cur}


def finish(state, cur, res):
    """Act on the final result for the current item and log it."""
    verdict = res.get("status")
    record = {k: cur[k] for k in ("branch", "mode", "test_id", "sha")}
    record.update(elo=res.get("elo"), elo_ci95=res.get("elo_ci95"), games=res.get("games"),
                  llr=res.get("llr"), verdict=verdict)
    item = requeue_item(cur)
    if verdict == "broken":
        # Infrastructure failure: retry once, then hand to a human.
        if retry(state, item, cur):
            log(dict(record, outcome="retrying", reason="SPRT run ended without a report"))
        else:
            log(dict(record, outcome="needs-human", reason="SPRT run ended without a report, twice"))
    elif cur["mode"] == "ladder":
        log(dict(record, outcome="ladder-logged", rating=res.get("rating"), ci=res.get("ci")))
    elif cur["mode"] == "ltc-check":
        if verdict == "H0":
            state["paused"] = f"LTC check {cur['test_id']} failed: main regressed vs {cur['base_sha'][:7]}"
            log(dict(record, outcome="regression-alert", reason=state["paused"]))
        else:
            log(dict(record, outcome="check-passed" if verdict == "H1" else "check-inconclusive"))
    elif verdict == "H1":
        refused = fast_forward_main(cur["sha"], cur.get("base_sha"), cur["branch"], cur.get("bench"),
                                    state.get("identical_merges", []), test_id=cur.get("test_id"),
                                    evalhash=cur.get("evalhash"))
        if not refused:
            log(dict(record, outcome="merged"))
            schedule_check(state, cur["base_sha"])
        else:
            log(dict(record, outcome="requeued", reason=refused))
            state["pending"].insert(0, dict(item, front=True))
    else:
        log(dict(record, outcome="rejected"))


def run_item(state, item, priority=None):
    """start() with the queue's error handling. Returns the in-flight record or None."""
    try:
        launched = start(item, state, priority)
        # Keep item fields (retried, base, expect) on the in-flight record.
        return dict(item, **launched) if launched else None
    except Exception as exc:  # keep the queue moving past unexpected failures
        sh("git", "rebase", "--abort", check=False)
        reason = f"queue error: {exc}"[:500]
        if retry(state, item, item, back=True):
            log({"branch": item["branch"], "mode": item["mode"], "outcome": "retrying", "reason": reason})
        else:
            log({"branch": item["branch"], "mode": item["mode"], "outcome": "needs-human", "reason": reason})
        return None


def finish_safely(state, cur, res):
    """finish(), but an exception (a git or gh failure mid-merge) requeues or
    logs the item instead of killing the run before state is saved, which
    would leave the same result to fail on every later run."""
    try:
        finish(state, cur, res)
    except Exception as exc:
        sh("git", "rebase", "--abort", check=False)
        reason = f"queue error while finishing: {exc}"[:500]
        row = {k: cur.get(k) for k in ("branch", "mode", "test_id", "sha")}
        row["verdict"] = res.get("status")
        if res.get("status") in ("H1", "broken") and cur.get("mode") != "ladder" \
                and retry(state, requeue_item(cur), cur):
            log(dict(row, outcome="retrying", reason=reason))
        else:
            log(dict(row, outcome="needs-human", reason=reason))
    drop_tested_ref(cur["test_id"])


# Speedups merged per run: each builds the candidate and main (evalhash), so
# a long list must not run into the job timeout; the rest wait for the next run.
MAX_SPEEDUPS_PER_RUN = 2


def process_speedups(state):
    """Merge pending speedups now, whatever is in flight (no runners needed),
    up to MAX_SPEEDUPS_PER_RUN; state is saved after each one. Returns how
    many are left for the next run."""
    if state.get("paused"):
        return 0
    todo = [p for p in state["pending"] if p["mode"] == "speedup"]
    for item in todo[:MAX_SPEEDUPS_PER_RUN]:
        state["pending"].remove(item)
        run_item(state, item, {"rule": "speedup"})
        save_state(state, f"queue: speedup {item['branch']}")
    left = max(0, len(todo) - MAX_SPEEDUPS_PER_RUN)
    if left:
        print(f"{left} speedups left for the next run")
    return left


def advance():
    state = load_state()
    try:
        sweep_cancelled(state)
    except Exception as exc:  # never let the sweep stall the queue
        print(f"cancel sweep: {exc}")
    raw = os.environ.get("RESULT", "").strip()
    cur = state.get("current")
    if cur and cur.get("mode") == BURST_MODE:
        if burst_done(cur):
            log({"branch": "main", "mode": BURST_MODE, "outcome": "burst-done",
                 "test_id": cur["test_id"], "seed": cur["seed"], "sha": cur.get("sha")})
            state["current"] = cur = None
        if raw:
            print("ignoring result: no SPRT in flight (datagen burst)")
            raw = ""
    elif not raw and cur:
        found = find_result(cur["test_id"])
        if found:
            print(f"recovered final result for {cur['test_id']}")
            raw = json.dumps(found)
    if raw:
        res = json.loads(raw)
        if not cur or res.get("test_id") != cur["test_id"]:
            print(f"ignoring result for {res.get('test_id')}: not the current item")
        else:
            finish_safely(state, cur, res)
            state["current"] = None
            save_state(state, f"queue: finished {cur['test_id']}")

    if state.get("paused"):
        print(f"paused: {state['paused']} (resume with: python3 scripts/queue.py unpause)")
    speedups_left = process_speedups(state)
    if state.get("current"):
        print(f"{state['current']['test_id']} still in flight")
    rates = None
    while not state.get("current") and not state.get("paused"):
        rates = rates or base_rates(read_log())
        reason = burst_reason(state, state["pending"], rates)
        if reason:
            try:
                state["current"] = start_burst(state, reason, state["pending"])
            except Exception as exc:  # fall back to the item rather than stall
                print(f"datagen burst: {exc}")
            else:
                save_state(state, f"queue: {state['current']['test_id']}")
        if state.get("current") or not state["pending"]:
            break
        idx, priority = pick_next(state["pending"], rates)
        item = state["pending"].pop(idx)
        state["current"] = run_item(state, item, priority)
        if state.get("current"):
            state["consecutive_bursts"] = 0
        # Saved at once: a run killed later (job timeout) must not forget a
        # dispatched test (and dispatch it again) or a skipped item.
        save_state(state, f"queue: {state['current']['test_id'] if state.get('current') else 'skipped ' + item['branch']}")
    try:
        background_lane(state)
    except Exception as exc:  # never let the background lane stall the queue
        print(f"background lane: {exc}")
    save_state(state, f"queue: {state['current']['test_id'] if state.get('current') else 'idle'}")
    if speedups_left:  # the rest of the speedups go on the next run, now
        sh("gh", "workflow", "run", "queue.yml", "--ref", "main", check=False)


def retry(state, item, prev, back=False):
    """Requeue an item once after an infrastructure failure. Returns False if it already was."""
    if prev.get("retried"):
        return False
    item = dict(item, retried=True)
    if back:
        item.pop("front", None)
        state["pending"].append(item)
    else:
        state["pending"].insert(0, dict(item, front=True))
    return True


def schedule_check(state, base_sha):
    """After every CHECK_EVERY merges, queue an LTC non-regression of main vs the
    main from CHECK_EVERY merges back (the slow-erosion check in development.md)."""
    state.setdefault("check_base", base_sha)
    state["merges_since_check"] = state.get("merges_since_check", 0) + 1
    state["merges_since_ladder"] = state.get("merges_since_ladder", 0) + 1
    if state["merges_since_ladder"] >= LADDER_EVERY:
        state["pending"].insert(0, {"branch": "main", "mode": "ladder"})
        state["merges_since_ladder"] = 0
    if state["merges_since_check"] >= CHECK_EVERY:
        state["pending"].insert(0, {"branch": "main", "mode": "ltc-check", "base": state["check_base"]})
        sh("git", "fetch", "-q", "origin", "main")
        state["check_base"] = out("git", "rev-parse", "origin/main")
        state["merges_since_check"] = 0


def show_rates():
    """Dry run: base rates and the pick order for origin/queue-state (read-only)."""
    sh("git", "fetch", "-q", "origin", STATE_BRANCH, check=False)
    entries = [json.loads(l) for l in out("git", "show", f"origin/{STATE_BRANCH}:log.jsonl").splitlines() if l]
    state = json.loads(out("git", "show", f"origin/{STATE_BRANCH}:queue.json"))
    rates = base_rates(entries)
    for mode, r in rates.items():
        print(f"{mode:11s} n={r['n']:3d} p={r['p_accept']:.2f} gain={r['gain']:.1f} "
              f"h_acc={r['hours_accept']:.2f} h_rej={r['hours_reject']:.2f}")
    print(f"speedups, merged at once: {[p['branch'] for p in state['pending'] if p['mode'] == 'speedup']}")
    pending = [p for p in state["pending"] if p["mode"] != "speedup"]
    cur = state.get("current") or {}
    if cur.get("mode") == BURST_MODE:
        print(f"in flight: datagen burst seed {cur.get('seed')}")
    reason = burst_reason(state, pending, rates)
    print(f"burst: {'yes, ' + reason if reason else 'no'} (enabled={bool(BURST.get('enabled'))}, "
          f"below_score={BURST.get('below_score')}, "
          f"consecutive={state.get('consecutive_bursts', 0)}/{BURST.get('max_consecutive', 2)})")
    while pending:
        idx, pri = pick_next(pending, rates)
        item = pending.pop(idx)
        print(f"next: {item['branch']}:{item['mode']}  {json.dumps(pri)}")


def parse_add_args(args):
    """['spec', '--front', '--expect', '12'] -> request dict; SystemExit on bad args."""
    spec, flags, expect, rest = args[0], set(), None, list(args[1:])
    while rest:
        a = rest.pop(0)
        if a in ("--front", "--preempt"):
            flags.add(a)
        elif a == "--expect" and rest:
            try:
                expect = float(rest.pop(0))
            except ValueError:
                raise SystemExit("--expect takes an Elo number")
        else:
            raise SystemExit(f"unknown argument {a!r}")
    parse_adds(spec, expect)  # validate before pushing
    req = {"add": spec, "front": bool(flags), "preempt": "--preempt" in flags}
    if expect is not None:
        req["expect"] = expect
    return req


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["advance"]:
        advance()
    elif len(args) >= 2 and args[0] == "add":
        request(parse_add_args(args[1:]))
    elif len(args) == 2 and args[0] == "remove":
        request({"remove": args[1]})
    elif args == ["unpause"]:
        request({"unpause": True})
    elif args == ["rates"]:
        show_rates()
    else:
        sys.exit("usage: queue.py advance            (CI; RESULT env = final SPRT result JSON)\n"
                 "       queue.py add <branch:mode,...> [--front] [--preempt] [--expect <elo>]\n"
                 "       queue.py remove <branch[:mode]>  (drop waiting items)\n"
                 "       queue.py unpause\n"
                 "       queue.py rates              (read-only: base rates and pick order)")
