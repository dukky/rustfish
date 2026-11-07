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
     the Bench: trailer), push the rebased branch, and dispatch an SPRT whose
     final report calls back here. Conflicts and failed checks are logged as
     needs-human and skipped.
     Datagen bursts: when the best pending item scores below the "burst"
     settings' below_score (or nothing is pending), the 16 foreground runners
     generate one datagen seed instead (~15-20 min, see burst_reason); the
     item keeps its place and the queue re-picks when the burst ends.
  4. Keep the background lane (datagen) busy; datagen.yml calls back here
     when a run ends, so the next seed starts at once (and a burst ends).

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
# holds background_parallel runners, the queue's own job 1, and foreground
# SPRTs get the rest of the 20; a mode wider than that would starve both.
# background_shards should be a multiple of background_parallel.
_LANES = _CFG.get("lanes", {})
BACKGROUND_PARALLEL = _LANES.get("background_parallel", 6)
BACKGROUND_SHARDS = _LANES.get("background_shards", 3 * BACKGROUND_PARALLEL)
MAX_FOREGROUND_SHARDS = 19 - BACKGROUND_PARALLEL


def validate_settings(cfg):
    """Fail loudly at load time on settings that would oversubscribe runners."""
    bad = [f"{mode} (shards={s.get('shards')!r})" for mode, s in cfg["modes"].items()
           if not isinstance(s.get("shards"), int) or not 1 <= s["shards"] <= MAX_FOREGROUND_SHARDS]
    if bad:
        raise SystemExit(f"queue-settings.json: foreground modes must use 1..{MAX_FOREGROUND_SHARDS} "
                         f"shards (the rest of the 20 runners are the background lane and the "
                         f"queue's own job); fix: {', '.join(bad)}")
    burst = cfg.get("burst")
    if burst and burst.get("enabled"):
        par, shards = burst.get("max_parallel"), burst.get("shards")
        if not isinstance(par, int) or not 1 <= par <= MAX_FOREGROUND_SHARDS:
            raise SystemExit(f"queue-settings.json: burst.max_parallel must be 1..{MAX_FOREGROUND_SHARDS} "
                             f"(the foreground runners), got {par!r}")
        if not isinstance(shards, int) or shards < 1 or shards % par:
            raise SystemExit(f"queue-settings.json: burst.shards ({shards!r}) must be a positive "
                             f"multiple of burst.max_parallel ({par}), or the remainder runs as a "
                             f"second, mostly idle round (datagen.yml)")


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
    runs = json.loads(sh("gh", "run", "list", "-w", "sprt.yml", "-L", "50",
                         "--json", "databaseId,displayTitle,status").stdout or "[]")
    for run in runs:
        if run["displayTitle"].startswith(f"SPRT {cur['test_id']} ") and run["status"] != "completed":
            sh("gh", "run", "cancel", str(run["databaseId"]), check=False)
    item = dict(requeue_item(cur), front=True)
    state["pending"].insert(after, item)
    state["current"] = None
    log({"branch": cur["branch"], "mode": cur["mode"], "test_id": cur["test_id"],
         "outcome": "preempted", "reason": "a front item with --preempt was added"})


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
        m = re.findall(r"^Bench:\s*(\d+)\s*$", body, re.M)
        if m:
            return int(m[-1])
    return None


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
    """Build the checkout and run the start() checks; bench nodes, or None if anything failed."""
    if sh("cargo", "build", "--release", "-q", check=False).returncode != 0:
        return None
    checks = run_engine("perfttest\nselftest 300\nquit\n")
    if not re.search(r"perfttest summary (\d+)/\1 passed", checks) or "selftest ok" not in checks:
        return None
    m = re.search(r"Nodes searched\s*:\s*(\d+)", run_engine("bench\nquit\n"))
    return int(m.group(1)) if m else None


def fast_forward_main(sha, base_sha=None, branch=None, bench=None, identical=()):
    """Move main to the tested sha. Returns None on success, else why not.

    If main moved since base_sha, replay the candidate on top of main instead
    of retesting when the tested binary's behaviour is unchanged: main moved
    only outside the engine sources (tooling, docs, logs; the replayed engine
    tree is then byte-identical), or also by merges recorded in `identical`
    (speedups: bench and evalhash equal to main's). In the second case the
    replayed tree differs, so it is rebuilt, checked, and must give exactly
    the tested candidate's `bench` before main moves."""
    sh("git", "fetch", "-q", "origin", "main")
    if branch:  # make sure the tested sha is present in this checkout
        sh("git", "fetch", "-q", "origin", branch, check=False)
    main_sha = out("git", "rev-parse", "origin/main")
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
              check=False).returncode != 0:
            if not moved or bench is None:
                return "main moved during test (replayed engine differs)"
            got = checked_bench()
            if got != bench:
                return f"main moved during test (replayed bench {got} != tested {bench})"
        sha = out("git", "rev-parse", "replay")
        if branch:
            sh("git", "push", "-q", "--force", "origin", f"replay:{branch}")
    sh("git", "push", "-q", "origin", f"{sha}:refs/heads/main")
    if bench:
        with open(os.path.join(STATE, "main-bench.json"), "w") as fh:
            json.dump({"sha": sha, "nodes": bench}, fh)
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
        sh("gh", "workflow", "run", "ccrl-ladder.yml", "--ref", "main", "-f", f"test_id={test_id}",
           *[a for k, v in LADDER.items() for a in ("-f", f"{k}={v}")])
        # Foreground on 16 runners: on the 4-runner background lane a 120+1
        # ladder would take ~12 h. Datagen keeps the other 4 busy meanwhile.
        log(dict(base, outcome="dispatched", test_id=test_id, sha=main_sha,
                 **({"priority": priority} if priority else {})))
        return dict(base, test_id=test_id, sha=main_sha, base_sha=main_sha, started=now())
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
    stale = [int(b) for b in re.findall(r"^Bench: (\d+)$", msg, re.M) if int(b) != nodes]
    if nodes != mnodes or stale:
        if re.search(r"^Bench: \d+$", msg, re.M):
            msg = re.sub(r"^Bench: \d+$", f"Bench: {nodes}", msg, flags=re.M)
        else:
            msg = msg.replace("Co-Authored-By:", f"Bench: {nodes}\n\nCo-Authored-By:", 1)
        sh("git", "commit", "-q", "--amend", "-m", msg)
    sha = out("git", "rev-parse", "candidate")
    sh("git", "push", "-q", "--force", "origin", f"candidate:{branch}")

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
            if refused:
                log(dict(base, outcome="needs-human", reason=f"speedup merge refused: {refused}"))
            else:
                if state is not None:
                    # Behaviour-identical: an in-flight SPRT can be replayed over these.
                    ident = state.setdefault("identical_merges", [])
                    ident += commits
                    del ident[:-IDENTICAL_KEEP]
                log(dict(base, outcome="merged", sha=sha, reason="speedup, bench identical"))
        return None

    return dict(dispatch(base, sha, out("git", "rev-parse", "origin/main"), branch, priority),
                bench=nodes)


def eval_hash(binary):
    res = subprocess.run([binary], input="evalhash\nquit\n", capture_output=True, text=True)
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


def background_lane(state):
    """Keep the background runners (BACKGROUND_PARALLEL) busy (1 of 20 stays free for the queue's own job): when no datagen or SPSA run
    is queued or running, generate the next datagen seed from main (only
    merged nets generate data). A foreground burst's run (state["current"])
    uses the foreground runners, so it does not block this lane."""
    # datagen.yml's final job dispatches the queue with its own run id
    # (FINISHED_RUN): that run is still in progress while it calls back.
    finished = os.environ.get("FINISHED_RUN", "").strip()
    cur = state.get("current") or {}
    burst_re = datagen_title_re(cur["seed"]) if cur.get("mode") == BURST_MODE else None
    for wf in ("datagen.yml", "spsa.yml"):
        runs = json.loads(out("gh", "run", "list", "--workflow", wf, "--limit", "10",
                              "--json", "databaseId,status,displayTitle"))
        if any(r["status"] != "completed" and str(r.get("databaseId")) != finished
               and not (burst_re and burst_re.match(r.get("displayTitle") or "")) for r in runs):
            return
    seed = take_seed(state)
    dispatch_datagen(seed, BACKGROUND_SHARDS, 15000, 5000, BACKGROUND_PARALLEL)
    log({"branch": "main", "mode": "datagen", "outcome": "dispatched", "seed": seed,
         "sha": out("git", "rev-parse", "origin/main"), "reason": "background lane idle"})


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
                                    state.get("identical_merges", []))
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


def process_speedups(state):
    """Merge every pending speedup now, whatever is in flight (no runners needed)."""
    if state.get("paused"):
        return
    for item in [p for p in state["pending"] if p["mode"] == "speedup"]:
        state["pending"].remove(item)
        run_item(state, item, {"rule": "speedup"})


def advance():
    state = load_state()
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
            finish(state, cur, res)
            state["current"] = None

    if state.get("paused"):
        print(f"paused: {state['paused']} (resume with: python3 scripts/queue.py unpause)")
    process_speedups(state)
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
        if state.get("current") or not state["pending"]:
            break
        idx, priority = pick_next(state["pending"], rates)
        state["current"] = run_item(state, state["pending"].pop(idx), priority)
        if state.get("current"):
            state["consecutive_bursts"] = 0
    try:
        background_lane(state)
    except Exception as exc:  # never let the background lane stall the queue
        print(f"background lane: {exc}")
    save_state(state, f"queue: {state['current']['test_id'] if state.get('current') else 'idle'}")


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
    elif args == ["unpause"]:
        request({"unpause": True})
    elif args == ["rates"]:
        show_rates()
    else:
        sys.exit("usage: queue.py advance            (CI; RESULT env = final SPRT result JSON)\n"
                 "       queue.py add <branch:mode,...> [--front] [--preempt] [--expect <elo>]\n"
                 "       queue.py unpause\n"
                 "       queue.py rates              (read-only: base rates and pick order)")
