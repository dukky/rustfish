#!/usr/bin/env python3
"""rustfish progress dashboard: a stdlib-only HTTP server.

Serves index.html and /api/status, a JSON digest of the public repo
dukky/rustfish built from raw.githubusercontent.com files and the
unauthenticated GitHub REST API, plus /api/lichess for the Lichess bot card.
Every upstream fetch is cached in memory and only refreshed when a page asks
for data, so an idle dashboard makes no requests and a busy one stays well
inside the 60 requests/hour API limit. Training machines push their status to
POST /api/machine (scripts/machine_status.py).
"""
import datetime
import hmac
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.environ.get("REPO", "dukky/rustfish")
RAW = f"https://raw.githubusercontent.com/{REPO}"
API = f"https://api.github.com/repos/{REPO}"
HERE = os.path.dirname(os.path.abspath(__file__))
# Goals come from scripts/ccrl-targets.json on main (real CCRL Blitz single-CPU
# list thresholds); this copy is only the fallback if that file can't be fetched.
CCRL_TARGETS_FALLBACK = {
    "source": "CCRL Blitz rating list, single-CPU engines, best versions only",
    "url": "https://computerchess.org.uk/ccrl/404/", "computed": "2026-10-31",
    "targets": [{"name": "3400 (short term)", "rating": 3400, "rank": 131},
                {"name": "Top 100 (mid term)", "rating": 3504, "rank": 100},
                {"name": "Top 10 (long term)", "rating": 3751, "rank": 10}],
    "reference": [{"rank": 1, "name": "Stockfish 19", "rating": 3792}]}

# Seconds each source stays fresh. Worst-case API calls per hour: 20 for the
# in-progress runs list + 10 for the recent runs list + 3 for commits + at most
# JOBS_PER_HOUR job-list fetches shared by the live runs (each live run's jobs
# TTL stretches with the number of live runs), i.e. ~53/h, under the 60/h
# unauthenticated limit even with the page open all day. ETag 304s do not count.
TTL_RAW = 120
TTL_RUNS_ACTIVE = 180
TTL_RUNS_RECENT = 360
TTL_JOBS_MIN = 180
TTL_COMMITS = 1200
JOBS_PER_HOUR = 20

# Fallback rounds per shard per wave, used only if scripts/queue-settings.json
# can't be fetched (waves start at rounds_per_shard and double up to
# max_rounds_per_shard). Used only to estimate games played in the in-flight SPRT.
ROUNDS = {"stc": (16, 64), "nonreg": (16, 64), "ltc-gainer": (6, 24), "ltc-check": (6, 24)}

# GitHub's concurrent-job cap for the account (all workflows share it).
RUNNER_CAP = 20
# The live CI view: in-progress jobs of every live run, grouped by workflow file.
RUNNER_GROUPS = (("sprt", "SPRT / queue", ("sprt.yml", "queue.yml")),
                 ("spsa", "SPSA", ("spsa.yml",)),
                 ("ladder", "CCRL ladder", ("ccrl-ladder.yml",)),
                 ("other", "Other", None))
# Workflows with exactly one job: counted without fetching their job list.
SINGLE_JOB_WORKFLOWS = ("queue.yml", "ci.yml")
LIVE_STATUSES = ("in_progress", "queued", "waiting", "pending", "requested")


def workflow_file(run):
    return os.path.basename((run.get("path") or "").split("@")[0])


def jobs_ttl(n_live):
    """Per-run job-list TTL: the live runs share JOBS_PER_HOUR fetches."""
    return max(TTL_JOBS_MIN, int(3600 * max(1, n_live) / JOBS_PER_HOUR))


def runner_group(wf):
    for key, label, files in RUNNER_GROUPS:
        if files is None or wf in files:
            return key, label
    return "other", "Other"


def build_runner_usage(runs, jobs_by_run, cap=RUNNER_CAP):
    """What is actually running on GitHub Actions right now.

    runs: the live workflow runs (any of LIVE_STATUSES); jobs_by_run: run id ->
    that run's job list, or None when it could not be fetched. Single-job
    workflows (SINGLE_JOB_WORKFLOWS) need no job list. Counts in-progress jobs
    (each holds one of the account's `cap` concurrent runners) and queued jobs
    (waiting for one), grouped as in RUNNER_GROUPS. A run whose job list is
    unknown is listed with jobs None and makes the total a lower bound.
    """
    groups = {key: {"key": key, "label": label, "running": 0, "queued": 0, "runs": []}
              for key, label, _ in RUNNER_GROUPS}
    unknown = 0
    for r in runs:
        status = r.get("status")
        if status not in LIVE_STATUSES:
            continue
        wf = workflow_file(r)
        jobs = jobs_by_run.get(r.get("id"))
        if wf in SINGLE_JOB_WORKFLOWS and jobs is None:
            running, queued = (1, 0) if status == "in_progress" else (0, 1)
        elif jobs is None:
            running = queued = None
        else:
            running = sum(1 for j in jobs if j.get("status") == "in_progress")
            queued = sum(1 for j in jobs if j.get("status") in ("queued", "waiting", "pending"))
            if not running and not queued:
                continue  # all jobs finished, the run is wrapping up
        key, _ = runner_group(wf)
        g = groups[key]
        if running is None:
            unknown += 1
        else:
            g["running"] += running
            g["queued"] += queued
        g["runs"].append({"id": r.get("id"), "workflow": wf, "title": r.get("display_title") or r.get("name"),
                          "url": r.get("html_url"), "status": status, "running": running, "queued": queued,
                          "started": r.get("run_started_at") or r.get("created_at")})
    out = [g for g in groups.values()]
    in_use = sum(g["running"] for g in out)
    return {"cap": cap, "in_use": in_use, "idle": max(0, cap - in_use),
            "queued": sum(g["queued"] for g in out), "unknown_runs": unknown,
            "groups": out}


# Net generations: one row per net, newest first, with its shape read from the
# training record or, failing that, from the branch name (g8h1024kb8m-ep:
# gen 8, H 1024, king buckets kb8m, threat inputs ep).
GEN_RE = re.compile(r"^(?:nnue-|local-)?g(\d+)", re.I)
NAME_H_RE = re.compile(r"h(\d{3,4})(?=kb|-|$)", re.I)
NAME_KB_RE = re.compile(r"(kb\d+m?)", re.I)
NAME_THREATS_RE = re.compile(r"-(ep|threats?)(?:-|$)", re.I)
BEST_SCALE_PAIR_RE = re.compile(r"best-scale[^.;]*?val\.bin\s*/\s*val17x2[:\s]*([0-9.]+)\s*/\s*([0-9.]+)", re.I)
BEST_SCALE_VAL_RE = re.compile(r"best-scale.*?val\.bin\s*([0-9.]+)", re.I | re.S)
BEST_SCALE_17_RE = re.compile(r"best-scale.*?val17x2\s*([0-9.]+)", re.I | re.S)


def parse_best_scale(notes):
    """(val.bin, val17x2) best-scale losses quoted in a netloop notes string."""
    if not notes:
        return None, None
    def num(s):
        try:
            return float(s.rstrip("."))
        except (AttributeError, ValueError):
            return None
    m = BEST_SCALE_PAIR_RE.search(notes)
    if m:
        return num(m.group(1)), num(m.group(2))
    a, b = BEST_SCALE_VAL_RE.search(notes), BEST_SCALE_17_RE.search(notes)
    return num(a.group(1)) if a else None, num(b.group(1)) if b else None


def net_shape_from_name(name):
    """Shape hints in a net/branch name: {gen, hidden, king_buckets, threats}."""
    base = re.sub(r"^(nnue-|local-)", "", name or "")
    m = GEN_RE.match(base)
    h, kb, th = NAME_H_RE.search(base), NAME_KB_RE.search(base), NAME_THREATS_RE.search(base)
    return {"gen": int(m.group(1)) if m else None, "hidden": int(h.group(1)) if h else None,
            "king_buckets": kb.group(1).lower() if kb else None,
            "threats": th.group(1).lower() if th else None}


def build_generations(nets):
    """Rows of the generation table from net_gates["v2"] (newest first).

    Each row: gen, name, shape (hidden, output buckets, king buckets, threats),
    pool size, best-scale val losses (record fields best_scale_val /
    best_scale_val17x2, else parsed from notes), gate result, and which fields
    were inferred from the name because the training record is missing.
    """
    rows = []
    for g in nets:
        name = g.get("id") or re.sub(r"^nnue-", "", g.get("branch") or "")
        hint = net_shape_from_name(g.get("id") or g.get("branch"))
        if hint["gen"] is None:
            hint = net_shape_from_name(g.get("branch"))
        rec = bool(g.get("training_record"))
        shape = {"hidden": g.get("hidden"), "king_buckets": g.get("king_buckets"),
                 "threats": g.get("threats") or g.get("inputs")}
        inferred = []
        for k in shape:
            if shape[k] is None and hint[k]:
                shape[k] = hint[k]
                inferred.append(k)
        hidden, kb, threats = shape["hidden"], shape["king_buckets"], shape["threats"]
        bs_val, bs_17 = g.get("best_scale_val"), g.get("best_scale_val17x2")
        if bs_val is None and bs_17 is None:
            bs_val, bs_17 = parse_best_scale(g.get("notes"))
        gen = hint["gen"]
        label = f"g{gen}" if gen is not None else ("E" if re.match(r"^E\d", name or "") else None)
        rows.append({
            "gen": gen, "gen_label": label, "name": name, "branch": g.get("branch"),
            "training_record": rec, "local": bool(g.get("local")), "inferred": inferred,
            "hidden": hidden, "buckets": g.get("buckets") if rec else None,
            "king_buckets": kb, "threats": threats,
            "pool": g.get("positions_per_epoch"), "train_data": g.get("train_data"),
            "epochs": g.get("epochs"), "val_loss": g.get("val_loss"),
            "val_loss_quantized": g.get("val_loss_quantized"),
            "best_scale_val": bs_val, "best_scale_val17x2": bs_17, "baseline": g.get("baseline"),
            "notes": g.get("notes"), "mode": g.get("mode"), "tc": g.get("tc"), "status": g.get("status"),
            "elo": g.get("elo"), "ci": g.get("ci"), "games": g.get("games"), "games_est": g.get("games_est"),
            "reason": g.get("reason"), "when": g.get("when") or g.get("first_at"), "run_url": g.get("run_url"),
            "current": False})
    # Grouped by generation (newest generation first; the E series before gen
    # 2 counts as gen 1, unnumbered local experiments last), newest net first.
    rank = lambda r: r["gen"] if r["gen"] is not None else (1 if r["gen_label"] == "E" else -1)
    rows.sort(key=lambda r: (rank(r), r["when"] or "9999"), reverse=True)
    # The net on main now: the most recently merged one.
    merged = [r for r in rows if r["status"] == "merged"]
    if merged:
        max(merged, key=lambda r: r["when"] or "")["current"] = True
    return rows

# NNUE nets: the v1 chain (net1..net13, net/chain.json) ran until the trainer-v2
# pivot; since then each net is an experiment (E0, E1, ..., generations g2a, ...)
# recorded by `scripts/netloop.py record` in net/experiments.jsonl. Gate results
# come straight from the queue log, so a net appears as soon as it is queued.
NET_PIVOT = "2026-11-06T08:00:00Z"
NET_V1_AUDIT = "docs/nnue-experiments.md#audit-of-the-v1-chain-2026-11-06"
# Where net/experiments.jsonl lives: main once merged, else the netloop branch.
EXPERIMENT_REFS = ("main", "netloop")
# Optional experiments.jsonl fields passed through when a row has them.
NET_EXTRA_FIELDS = ("king_buckets", "mean_abs_eval", "best_scale_val", "best_scale_val17x2",
                    "best_scale_factor", "output_scale", "init", "train_data", "threats", "inputs")


class Cache:
    """URL -> (fetched_at, payload, etag). Serves stale data when GitHub fails."""

    def __init__(self):
        self.lock = threading.Lock()
        self.entries = {}
        self.rate = {}

    def get(self, url, ttl, parse, headers=None):
        now = time.time()
        with self.lock:
            ent = self.entries.get(url)
        if ent and now - ent["at"] < ttl:
            return ent["data"], ent["at"] if ent["data"] is not None else None, ent.get("error")
        headers = dict(headers or {"User-Agent": "rustfish-dashboard",
                                   "Accept": "application/vnd.github+json"})
        if ent and ent.get("etag"):
            headers["If-None-Match"] = ent["etag"]
        # Back off from the API if the unauthenticated quota is nearly used up.
        if url.startswith(API) and self.rate.get("remaining", 99) < 5 and \
                self.rate.get("reset", 0) > now and ent:
            return ent["data"], ent["at"], "GitHub API rate limit low; showing cached data"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                if url.startswith(API):
                    self._note_rate(resp.headers)
                body = resp.read().decode("utf-8")
                data = parse(body)
                ent = {"at": now, "data": data, "etag": resp.headers.get("ETag")}
        except urllib.error.HTTPError as exc:
            if url.startswith(API):
                self._note_rate(exc.headers)
            if exc.code == 304 and ent:
                ent = dict(ent, at=now, error=None)
            else:
                return self._fail(url, ent, f"HTTP {exc.code} from {url}")
        except Exception as exc:  # network down, bad JSON, timeout
            return self._fail(url, ent, f"{type(exc).__name__}: {exc}"[:300])
        with self.lock:
            self.entries[url] = ent
        return ent["data"], ent["at"], None

    def _fail(self, url, ent, msg):
        # Keep stale data (or remember the failure when there is none) and retry
        # after a minute rather than on every request.
        with self.lock:
            self.entries[url] = dict(ent or {"data": None}, at=time.time() - max(0, TTL_RAW - 60),
                                     error=msg)
        if ent and ent["data"] is not None:
            return ent["data"], ent["at"], msg
        return None, None, msg

    def _note_rate(self, headers):
        try:
            self.rate = {"remaining": int(headers.get("X-RateLimit-Remaining")),
                         "reset": int(headers.get("X-RateLimit-Reset"))}
        except (TypeError, ValueError):
            pass


CACHE = Cache()
BUILD_LOCK = threading.Lock()  # one refresh at a time, so requests share fetches


def jsonl(body):
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds") \
        if ts else None


def parse_when(s):
    """Normalise the repo's several timestamp spellings to ISO 8601 UTC."""
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})?", s)
    if m:
        y, mo, d, h, mi, sec = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{sec or '00'}Z"
    s = s.rstrip("Z")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", s):
        s += ":00"
    return s + "Z"


def is_net_row(branch, mode):
    """A net candidate: an nnue-* branch or a net-gate test, but not engine code
    shipped as a speedup (nnue-inference, nnue-arch-r2, ...)."""
    return mode == "net-gate" or (bool(branch) and branch.startswith("nnue-") and mode != "speedup"
                                  and not branch.startswith("nnue-scale"))  # eval-scale tests, not nets


def build_net_gates(qlog, queue, experiments, v1_branches):
    """Every net branch's latest queue status joined with its training record.

    Returns {"v2": [...], "v1": [...]}: v2 is everything gated after the
    trainer-v2 pivot (or with an experiments row), newest first; v1 is the older
    gate history. A gated branch without an experiments row is kept and flagged
    (training_record False) so a missing record is visible, not silently absent.
    """
    by_branch = {}
    for r in experiments:
        b = r.get("branch") or (f"nnue-{str(r['id']).lower()}" if r.get("id") else None)
        if b and b != "main":
            by_branch[b] = r
    current = queue.get("current") or {}
    gates = {}  # branch -> latest state
    for r in qlog:
        b = r.get("branch")
        if not is_net_row(b, r.get("mode")):
            continue
        when = parse_when(r.get("logged_at"))
        g = gates.setdefault(b, {"branch": b, "first_at": when})
        g.update(mode=r.get("mode"), outcome=r.get("outcome"), when=when,
                 sha=(r.get("sha") or "")[:7] or g.get("sha"), test_id=r.get("test_id") or g.get("test_id"))
        if r.get("outcome") != "dispatched":
            g.update(elo=r.get("elo"), ci=r.get("elo_ci95"), games=r.get("games"), llr=r.get("llr"),
                     verdict=r.get("verdict"), reason=r.get("reason"))
    if is_net_row(current.get("branch"), current.get("mode")):
        started = parse_when(current.get("started"))
        g = gates.setdefault(current["branch"], {"branch": current["branch"], "first_at": started})
        g.update(mode=current.get("mode"), status="testing", test_id=current.get("test_id"),
                 when=started, sha=(current.get("sha") or "")[:7])
    for p in queue.get("pending", []):
        if not is_net_row(p.get("branch"), p.get("mode")):
            continue
        g = gates.setdefault(p["branch"], {"branch": p["branch"], "first_at": None})
        if g.get("status") != "testing":
            g.update(status="queued", mode=p.get("mode"))
    for g in gates.values():
        if not g.get("status"):
            # A dispatch with no final row yet is still being judged.
            g["status"] = "testing" if g.get("outcome") == "dispatched" else g.get("outcome")
    # Trained nets nobody has queued yet still belong in the table; local-only
    # experiments (branch "local-<id>", never pushed) are labelled as such.
    for b, r in by_branch.items():
        if b not in gates:
            local = b.startswith("local-")
            gates[b] = {"branch": b, "first_at": parse_when(r.get("date")),
                        "status": "local only" if local else "not gated", "local": local}
    v1, v2 = [], []
    for g in gates.values():
        e = by_branch.get(g["branch"])
        g["training_record"] = bool(e)
        if e:
            sprt = e.get("sprt") or {}
            g.update(id=e.get("id"), hidden=e.get("hidden"), buckets=e.get("buckets"),
                     seeds=e.get("seeds"), epochs=e.get("epochs"),
                     positions_per_epoch=e.get("positions_per_epoch"), val_loss=e.get("val_loss"),
                     val_loss_quantized=e.get("val_loss_quantized"), baseline=e.get("baseline"),
                     notes=e.get("notes"), tc=sprt.get("tc"),
                     **{k: e[k] for k in NET_EXTRA_FIELDS if e.get(k) is not None})
            # A gate run outside the queue only exists in the training record.
            if g.get("elo") is None and sprt.get("elo") is not None:
                g.update(elo=sprt.get("elo"), ci=sprt.get("ci"), games=sprt.get("games"),
                         mode=g.get("mode") or sprt.get("mode"), gate_from="experiments.jsonl")
        old = g["branch"] in v1_branches or (g.get("first_at") or "9999") < NET_PIVOT
        (v1 if old and not e else v2).append(g)
    order = lambda g: g.get("first_at") or "9999"  # queued, never dispatched: newest
    v2.sort(key=order, reverse=True)
    v1.sort(key=order, reverse=True)
    return {"v2": v2, "v1": v1, "pivot": NET_PIVOT}


def build_status():
    sources = {}

    def fetch(name, url, ttl, parse):
        data, at, err = CACHE.get(url, ttl, parse)
        sources[name] = {"fetched_at": iso(at), "error": err}
        return data

    ladder = fetch("ladder", f"{RAW}/main/tests/ladder-log.jsonl", TTL_RAW, jsonl) or []
    sprt = fetch("sprt", f"{RAW}/main/tests/sprt-log.jsonl", TTL_RAW, jsonl) or []
    queue = fetch("queue", f"{RAW}/queue-state/queue.json", TTL_RAW, json.loads) or {}
    qlog = fetch("queue_log", f"{RAW}/queue-state/log.jsonl", TTL_RAW, jsonl) or []
    plan = fetch("plan", f"{RAW}/main/plan.md", TTL_RAW, lambda b: b)
    m = re.search(r"Last updated:\s*([0-9][0-9 :T-]*[0-9](?: UTC)?)", plan or "")
    plan_updated = m.group(1).strip() if m else None
    # Two run lists: every in-progress run however old (a long SPSA round falls
    # out of a recent-runs page), and the latest runs (queued runs, the last
    # completed SPRT wave). Merged by id, the fresher in-progress copy wins.
    active_runs = (fetch("runs", f"{API}/actions/runs?status=in_progress&per_page=100",
                         TTL_RUNS_ACTIVE, json.loads) or {}).get("workflow_runs", [])
    recent_runs = (fetch("runs_recent", f"{API}/actions/runs?per_page=30", TTL_RUNS_RECENT,
                         json.loads) or {}).get("workflow_runs", [])
    by_id = {r.get("id"): r for r in recent_runs}
    by_id.update({r.get("id"): r for r in active_runs})
    runs = sorted(by_id.values(), key=lambda r: r.get("created_at") or "", reverse=True)
    commits = fetch("commits", f"{API}/commits?sha=main&per_page=25", TTL_COMMITS, json.loads) or []

    # Job lists of the live runs (single-job workflows need none). They share
    # JOBS_PER_HOUR fetches, so each run's TTL grows with the number of runs.
    live = [r for r in runs if r.get("status") in LIVE_STATUSES]
    need_jobs = [r for r in live if workflow_file(r) not in SINGLE_JOB_WORKFLOWS]
    live_ttl = jobs_ttl(len(need_jobs))
    jobs_by_run, jobs_errors = {}, []
    for r in need_jobs:
        data, at, err = CACHE.get(f"{API}/actions/runs/{r['id']}/jobs?per_page=100", live_ttl, json.loads)
        jobs_by_run[r["id"]] = data.get("jobs", []) if isinstance(data, dict) else None
        if err:
            jobs_errors.append(err)
    sources["ci_jobs"] = {"fetched_at": iso(time.time()) if not jobs_errors else None,
                          "error": jobs_errors[0] if jobs_errors else None}
    # Wave sizes per queue mode, from the file scripts/queue.py itself reads.
    qcfg = fetch("queue_settings", f"{RAW}/main/scripts/queue-settings.json", TTL_RAW, json.loads) or {}
    ccrl = fetch("ccrl_targets", f"{RAW}/main/scripts/ccrl-targets.json", TTL_RAW, json.loads)
    if not (isinstance(ccrl, dict) and ccrl.get("targets")):
        ccrl = dict(CCRL_TARGETS_FALLBACK, fallback=True)
    short = min(ccrl["targets"], key=lambda t: t["rating"])
    def as_target(t, reached=()):
        return {"rating": t["rating"], "name": t.get("name"), "reached": list(reached),
                "label": "CCRL Blitz" + (f", #{t['rank']} on the single-CPU list" if t.get("rank") else "")}
    target = as_target(short)
    rounds = {m: (s["rounds_per_shard"], s.get("max_rounds_per_shard", s["rounds_per_shard"]))
              for m, s in qcfg.get("modes", {}).items()} or ROUNDS

    # NNUE net chain: on main once merged, otherwise the net-chain branch.
    chain, chain_ref, chain_err = None, None, None
    for ref in ("main", "net-chain"):
        data, at, err = CACHE.get(f"{RAW}/{ref}/net/chain.json", TTL_RAW, json.loads)
        if isinstance(data, dict):
            chain, chain_ref = data, ref
            sources["net_chain"] = {"fetched_at": iso(at), "error": None, "ref": ref}
            break
        chain_err = err
    if chain is None:
        sources["net_chain"] = {"fetched_at": None, "error": chain_err or "no net/chain.json"}
    nets = []
    for n in (chain or {}).get("nets", []) if isinstance(chain, dict) else []:
        gate = n.get("gate") or {}
        nets.append({"net": n.get("net"), "branch": n.get("branch"), "positions": n.get("positions"),
                     "nodes": n.get("nodes"), "seed": n.get("seed"), "epochs": n.get("epochs"),
                     "loss": n.get("loss"), "loss_note": n.get("loss_note"), "bench": n.get("bench"),
                     "note": n.get("note"), "gate_elo": gate.get("elo"), "gate_ci": gate.get("ci"),
                     "gate_games": gate.get("games"), "gate_verdict": gate.get("verdict")})

    # NNUE experiments (trainer v2): training metadata joined to queue gates.
    experiments, exp_ref, exp_err = None, None, None
    for ref in EXPERIMENT_REFS:
        data, at, err = CACHE.get(f"{RAW}/{ref}/net/experiments.jsonl", TTL_RAW, jsonl)
        if data:
            experiments, exp_ref = data, ref
            sources["net_experiments"] = {"fetched_at": iso(at), "error": None, "ref": ref}
            break
        exp_err = exp_err or err
    if experiments is None:
        sources["net_experiments"] = {"fetched_at": None, "error": exp_err or "no net/experiments.jsonl"}
    net_gates = build_net_gates(qlog, queue, experiments or [], {n["branch"] for n in nets})
    net_gates["experiments_ref"] = exp_ref

    # Strength over time.
    points = []
    for r in ladder:
        if r.get("rating") is None:
            continue
        # A cancelled or partial run: far fewer games than its anchors x
        # games_per_anchor (or < 40 games per anchor when that is not logged).
        n_anchors = len(r.get("anchors", [])) or 1
        gpa = r.get("games_per_anchor")
        try:
            expected = int(gpa) * n_anchors if gpa else 40 * n_anchors
        except (TypeError, ValueError):
            expected = 40 * n_anchors
        partial = bool(r.get("partial") or r.get("status") in ("cancelled", "partial")
                       or (r.get("games") or 0) < 0.9 * expected)
        points.append({"partial": partial, "games_per_anchor": gpa,
                       "ci_width": round(r["ci"][1] - r["ci"][0], 1) if r.get("ci") else None,
"when": parse_when(r.get("logged_at")), "rating": r["rating"],
                       "ci": r.get("ci"), "sha": (r.get("sha") or "")[:7], "note": r.get("note"),
                       # Rows from 2026-11-07 on: ci is pentanomial + anchor CCRL error and
                       # ci_binomial the old interval; older rows have only the binomial ci.
                       "ci_method": r.get("ci_method") or "binomial", "ci_binomial": r.get("ci_binomial"),
                       "games": r.get("games"), "tc": r.get("tc"), "url": r.get("url"),
                       "anchors_file": r.get("anchors_file", "scripts/anchors.json"),
                       "rating_with_black_marlin": r.get("rating_with_black_marlin"),
                       "anchors": [{k: a.get(k) for k in ("anchor", "ccrl", "score", "games")}
                                   for a in r.get("anchors", [])]})
    points.sort(key=lambda p: p["when"] or "")

    # SPRT results, minus tests later marked invalid.
    invalid = {r["test_id"] for r in sprt if r.get("status") == "invalid"}
    tests = []
    for r in sprt:
        if r.get("test_id") in invalid or r.get("status") == "invalid":
            continue
        tests.append({"test_id": r.get("test_id"), "candidate": r.get("candidate"),
                      "status": r.get("status"), "elo": r.get("elo"), "ci": r.get("elo_ci95"),
                      "games": r.get("games"), "tc": r.get("tc"), "bounds": [r.get("elo0"), r.get("elo1")],
                      "base": r.get("base"), "when": parse_when(r.get("logged_at")),
                      "url": r.get("url"), "note": r.get("note"), "source": "sprt-log"})

    # Queue outcomes (the queue's own SPRTs are not in sprt-log.jsonl).
    events = []
    for r in qlog:
        events.append({"branch": r.get("branch"), "mode": r.get("mode"), "outcome": r.get("outcome"),
                       "elo": r.get("elo"), "ci": r.get("elo_ci95"), "games": r.get("games"),
                       "verdict": r.get("verdict"), "reason": r.get("reason"),
                       "test_id": r.get("test_id"), "sha": (r.get("sha") or "")[:7],
                       "when": parse_when(r.get("logged_at")), "seed": r.get("seed"),
                       # ladder-logged rows carry a CCRL fit, not a gate Elo
                       "rating": r.get("rating"), "rating_ci": r.get("ci"),
                       "tc": (qcfg.get("modes", {}).get(r.get("mode")) or {}).get("tc")})

    # Accepted changes and the Elo each measured against the main of its day.
    # A queue "merged" entry is authoritative. Before the queue existed, merges
    # were manual, so an H1 for a branch not waiting in the queue again is only
    # "passed" (likely merged, possibly folded into a combined branch).
    waiting = {p.get("branch") for p in queue.get("pending", [])}
    if queue.get("current"):
        waiting.add(queue["current"].get("branch"))
    merged = {}
    for t in sorted(tests, key=lambda t: t["when"] or ""):
        name = t["candidate"]
        if (t["status"] == "H1" and t["elo"] is not None and name and name != "main"
                and name not in waiting and t["games"]):
            merged[name] = {"name": name, "elo": t["elo"], "ci": t["ci"], "when": t["when"],
                            "games": t["games"], "how": f"SPRT {t['tc']}", "certain": False}
    for e in events:
        if e["outcome"] == "merged":
            merged.pop(e["branch"], None)
            merged[e["branch"] + "@" + (e["when"] or "")] = {
                "name": e["branch"], "elo": e["elo"], "ci": e["ci"], "when": e["when"],
                "games": e["games"], "mode": e["mode"], "tc": e["tc"],
                "how": "speedup, bench identical (no Elo)" if e["mode"] == "speedup"
                       else {"net-gate": "net gate"}.get(e["mode"], f"{e['mode']} gate")
                       + (f" at {e['tc']}" if e["tc"] else ""), "certain": True}
    merged = sorted(merged.values(), key=lambda m: m["when"] or "")

    # In-flight foreground item: SPRT runs are titled "SPRT <test_id> wave <n>",
    # queue ladders "CCRL ladder <test_id>" (one run, 2 shard jobs per anchor).
    current = queue.get("current")
    inflight = None
    if current:
        inflight = dict(current)
        tid = re.escape(current.get("test_id", ""))
        is_ladder = current.get("mode") == "ladder"
        waves = []
        for run in runs:
            title = run.get("display_title") or ""
            m = re.fullmatch(rf"CCRL ladder {tid}()" if is_ladder else rf"SPRT {tid} wave (\d+)", title)
            if m:
                waves.append((int(m.group(1) or 1), run))
        if waves:
            wave, run = max(waves, key=lambda w: w[0])
            inflight.update(wave=None if is_ladder else wave, run_url=run.get("html_url"),
                            run_status=run.get("status"),
                            run_started=run.get("run_started_at") or run.get("created_at"))
            if run.get("id") in jobs_by_run:  # live: fetched above
                jobs = jobs_by_run[run["id"]] or []
            else:  # the last wave has finished: its job list no longer changes. A
                # separate cache key (filter=latest is the API default) so a copy
                # cached while the run was live is not reused.
                jobs = (fetch("jobs", f"{API}/actions/runs/{run['id']}/jobs?per_page=100&filter=latest",
                              3600, json.loads) or {}).get("jobs", [])
            shard_re = r".+ / shard \d+" if is_ladder else r"shard \d+"
            shards = [j for j in jobs if re.fullmatch(shard_re, j.get("name", ""))]
            done = sum(1 for j in shards if j.get("status") == "completed")
            running = sum(1 for j in shards if j.get("status") == "in_progress")
            inflight.update(shards_total=len(shards), shards_done=done, shards_running=running)
            if running:  # the run reads "queued" while some shards wait for runners
                inflight["run_status"] = "in_progress"
            base, cap = rounds.get(current.get("mode"), (None, None))
            if is_ladder and shards:
                # ccrl-ladder.yml default: 2 shards per anchor.
                per_job = int((qcfg.get("ladder") or {}).get("games_per_anchor") or 96) // 2
                inflight.update(games_est=done * per_job, games_wave_total=len(shards) * per_job,
                                games_prior=0)
            elif base and shards:
                rpw = [min(cap, base * 2 ** i) for i in range(wave)]
                prior = sum(len(shards) * r * 2 for r in rpw[:-1])
                inflight.update(games_prior=prior, games_wave_total=len(shards) * rpw[-1] * 2,
                                games_est=prior + done * rpw[-1] * 2)

    runner_usage = build_runner_usage(live, jobs_by_run)
    runner_usage["as_of"] = sources.get("runs", {}).get("fetched_at")
    # Configured shards per SPRT wave (queue-settings.json), shown as context only.
    runner_usage["test_shards"] = (qcfg.get("modes", {}).get("stc") or {}).get("shards")

    recent_commits = []
    for c in commits if isinstance(commits, list) else []:
        msg = (c.get("commit") or {}).get("message", "")
        bench = re.findall(r"^Bench:\s*(\d+)\s*$", msg, re.M)
        recent_commits.append({"sha": c.get("sha", "")[:7], "title": msg.splitlines()[0] if msg else "",
                               "when": ((c.get("commit") or {}).get("committer") or {}).get("date"),
                               "bench": int(bench[-1]) if bench else None, "url": c.get("html_url")})

    for g in net_gates["v2"] + net_gates["v1"]:
        if inflight and g["branch"] == inflight.get("branch"):
            g.update(games_est=inflight.get("games_est"), run_url=inflight.get("run_url"))

    latest = points[-1] if points else None
    # The headline goal is the next CCRL target the last ladder has not reached.
    if latest and latest.get("rating") is not None:
        done = [t for t in ccrl["targets"] if t["rating"] <= latest["rating"]]
        ahead = [t for t in ccrl["targets"] if t["rating"] > latest["rating"]]
        nxt = min(ahead, key=lambda t: t["rating"]) if ahead else max(ccrl["targets"], key=lambda t: t["rating"])
        target = as_target(nxt, [t.get("name") for t in done])
    # Measured strength is the last ladder run; estimated adds the queue SPRTs
    # merged after it. Estimated is never presented as an official CCRL rating.
    measured = None
    if latest:
        measured = {k: latest[k] for k in ("rating", "ci", "ci_method", "when", "sha", "url", "games")}
    components = []
    if measured and measured.get("when"):
        for e in events:
            if e["outcome"] != "merged" or not e["branch"] or e["branch"] == "main":
                continue
            if e["elo"] is None or not e["when"] or e["when"] <= measured["when"]:
                continue
            components.append({"branch": e["branch"], "elo": e["elo"], "games": e["games"],
                               "when": e["when"]})
    components.sort(key=lambda c: c["when"])
    delta = round(sum(c["elo"] for c in components), 1)
    strength = None
    if measured:
        strength = {
            "measured": measured,
            "estimated": {"rating": round(measured["rating"] + delta, 1), "delta": delta,
                          "components": components},
            "label": "Estimated = last measured ladder rating + queue SPRT Elo merged since; "
                     "not an official CCRL rating.",
        }

    return {
        "generated_at": iso(time.time()),
        "repo": REPO,
        "target": target,
        "ccrl_targets": ccrl,
        "strength": strength,
        "net_chain": {"nets": nets, "ref": chain_ref, "audit_doc": NET_V1_AUDIT},
        "net_gates": net_gates,
        "generations": build_generations(net_gates["v2"]),
        "runner_usage": runner_usage,
        "roadmap": {"source": "plan.md", "ref": "main", "last_updated": plan_updated},
        "current": latest and {k: latest[k] for k in ("rating", "ci", "ci_method", "sha", "when", "url", "games")},
        "gap": latest and round(target["rating"] - latest["rating"], 1),
        "ladder": points,
        "queue": {"current": inflight, "pending": queue.get("pending", []), "paused": queue.get("paused")},
        "queue_events": events[-40:][::-1],
        "sprt": sorted(tests, key=lambda t: t["when"] or "", reverse=True)[:40],
        "merged": merged,
        "commits": recent_commits,
        "plan": plan,
        "sources": sources,
        "api_rate": CACHE.rate,
    }


# Machine status push: a training box (scripts/machine_status.py) POSTs its
# latest snapshot to /api/machine; the old laptop reporter
# (scripts/laptop_status.py) still posts to /api/laptop and is stored as
# machine "laptop". Latest payload per machine, in memory and in
# $DATA_DIR/machines.json when DATA_DIR is configured. Accepted only with
# `Authorization: Bearer $MACHINE_TOKEN` (or the legacy LAPTOP_TOKEN).
MACHINE_MAX_BODY = 16 * 1024
MACHINE_MAX = 8  # machines kept; the longest-silent one is dropped beyond this
MACHINE_SCHEMA = "machine-status/1"
MACHINE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}")
MACHINE_LOCK = threading.Lock()
MACHINES = {}  # name -> {"payload": {...}, "received": epoch seconds}
DATA_DIR = os.environ.get("DATA_DIR")


def machines_load():
    if not DATA_DIR:
        return
    try:
        with open(os.path.join(DATA_DIR, "machines.json")) as fh:
            d = json.load(fh)
        for name, ent in d.items():
            MACHINES[name] = {"payload": ent["payload"], "received": float(ent["received"])}
        return
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    try:  # before the rename: one laptop.json
        with open(os.path.join(DATA_DIR, "laptop.json")) as fh:
            d = json.load(fh)
        MACHINES["laptop"] = {"payload": d["payload"], "received": float(d["received"])}
    except (OSError, ValueError, KeyError, TypeError):
        pass


def machine_name(payload, default):
    name = payload.get("machine")
    return name if isinstance(name, str) and MACHINE_NAME_RE.fullmatch(name) else default


def machine_store(name, payload, now=None):
    with MACHINE_LOCK:
        MACHINES[name] = {"payload": payload, "received": now or time.time()}
        while len(MACHINES) > MACHINE_MAX:
            del MACHINES[min(MACHINES, key=lambda k: MACHINES[k]["received"])]
        if DATA_DIR:
            try:
                tmp = os.path.join(DATA_DIR, "machines.json.tmp")
                with open(tmp, "w") as fh:
                    json.dump(MACHINES, fh)
                os.replace(tmp, os.path.join(DATA_DIR, "machines.json"))
            except OSError:
                pass


def machines_status():
    """[{name, schema, payload, received, age_s}], most recently heard first."""
    now = time.time()
    with MACHINE_LOCK:
        out = [{"name": k, "schema": v["payload"].get("schema"), "payload": v["payload"],
                "received": iso(v["received"]), "age_s": int(now - v["received"])}
               for k, v in MACHINES.items()]
    return sorted(out, key=lambda m: m["age_s"])


def machine_authorized(header):
    if not header or not header.startswith("Bearer "):
        return False
    got = header[7:].strip().encode()
    tokens = [t for t in (os.environ.get("MACHINE_TOKEN", ""), os.environ.get("LAPTOP_TOKEN", "")) if t]
    # Compare against every configured token so the timing does not tell which matched.
    ok = [hmac.compare_digest(got, t.encode()) for t in tokens]
    return any(ok)


# Lichess bot card: the bot account's public profile, recent games and rating
# history from the public Lichess API (no token). Served by /api/lichess, apart
# from /api/status, so a Lichess outage never delays or breaks the rest of the
# page. Each URL is cached TTL_LICHESS and refetched only when a page asks: at
# most 4 requests per 5 minutes however many pages are open, and a failure is
# remembered for a minute before retrying (Lichess asks clients to wait a full
# minute after a 429). The account is token-bound in lichess-bot, so its name
# is not in deploy/lichess-bot/config.yml: set LICHESS_USER (dokku config:set).
LICHESS = "https://lichess.org"
LICHESS_USER = os.environ.get("LICHESS_USER", "").strip()
TTL_LICHESS = 300
LICHESS_SPEEDS = ("bullet", "blitz", "rapid", "classical")
LICHESS_GAMES = 10
LICHESS_HISTORY_POINTS = 60  # last N rating-history points per speed, for the sparklines
LICHESS_UA = "rustfish-dashboard/1.0 (+https://github.com/dukky/rustfish; status page for the rustfish bot)"
LICHESS_LOCK = threading.Lock()  # one refresh at a time, so requests share fetches


def lichess_headers(accept="application/json"):
    return {"User-Agent": LICHESS_UA, "Accept": accept}


def parse_lichess_user(d):
    """/api/user/{name} -> profile with ratings per speed (only speeds played)."""
    if not isinstance(d, dict) or not d.get("id"):
        raise ValueError("no user in response")
    perfs = []
    for speed in LICHESS_SPEEDS:
        p = (d.get("perfs") or {}).get(speed) or {}
        if p.get("games"):
            perfs.append({"speed": speed, "rating": p.get("rating"), "rd": p.get("rd"),
                          "prog": p.get("prog"), "prov": bool(p.get("prov")), "games": p["games"]})
    count = d.get("count") or {}
    return {"id": d["id"], "username": d.get("username") or d["id"], "title": d.get("title"),
            "url": d.get("url") or f"{LICHESS}/@/{d['id']}",
            "disabled": bool(d.get("disabled")), "tos_violation": bool(d.get("tosViolation")),
            "seen_at": iso(d["seenAt"] / 1000) if d.get("seenAt") else None,
            "created_at": iso(d["createdAt"] / 1000) if d.get("createdAt") else None,
            "playing_url": d.get("playing"),
            "perfs": perfs,
            "count": {k: count.get(k) for k in ("all", "rated", "win", "loss", "draw")}}


def parse_lichess_online(rows):
    """/api/users/status?ids=... -> {online, playing, playing_id} for the one user."""
    r = rows[0] if isinstance(rows, list) and rows else {}
    return {"online": bool(r.get("online")), "playing": bool(r.get("playing")),
            "playing_id": r.get("playingId")}


def lichess_result(g, color):
    if g.get("status") in ("aborted", "noStart", "created", "started"):
        return "aborted" if g.get("status") in ("aborted", "noStart") else "playing"
    w = g.get("winner")
    return "draw" if not w else "win" if w == color else "loss"


def parse_lichess_games(rows, user_id):
    """NDJSON rows of /api/games/user -> newest-first list from the bot's side."""
    out = []
    uid = (user_id or "").lower()
    for g in rows:
        players = g.get("players") or {}
        white = (players.get("white") or {}).get("user") or {}
        color = "white" if white.get("id", "").lower() == uid else "black"
        me, opp = players.get(color) or {}, players.get("black" if color == "white" else "white") or {}
        ou = opp.get("user") or {}
        out.append({
            "id": g.get("id"), "url": f"{LICHESS}/{g.get('id')}" + ("/black" if color == "black" else ""),
            "speed": g.get("speed"), "rated": bool(g.get("rated")), "variant": g.get("variant"),
            "color": color, "result": lichess_result(g, color), "status": g.get("status"),
            "opponent": ou.get("name") or (f"Stockfish level {opp['aiLevel']}" if opp.get("aiLevel") else "anonymous"),
            "opponent_title": ou.get("title"), "opponent_rating": opp.get("rating"),
            "rating": me.get("rating"), "rating_diff": me.get("ratingDiff"),
            "clock": f"{g['clock'].get('initial', 0) / 60:g}+{g['clock'].get('increment', 0)}"
                     if isinstance(g.get("clock"), dict) else None,
            "when": iso((g.get("lastMoveAt") or g.get("createdAt") or 0) / 1000)})
    out.sort(key=lambda x: x["when"] or "", reverse=True)
    return out


def parse_lichess_history(rows):
    """/api/user/{name}/rating-history -> {speed: [[YYYY-MM-DD, rating], ...]}.
    Lichess points are [year, month (0-based), day, rating], one per day played."""
    out = {}
    for r in rows if isinstance(rows, list) else []:
        speed = str(r.get("name", "")).lower()
        if speed not in LICHESS_SPEEDS:
            continue
        pts = [[f"{p[0]:04d}-{p[1] + 1:02d}-{p[2]:02d}", p[3]] for p in r.get("points") or []
               if isinstance(p, list) and len(p) == 4]
        if pts:
            out[speed] = pts[-LICHESS_HISTORY_POINTS:]
    return out


def build_lichess(user=None, cache=None):
    """The Lichess card. Never raises: an unreachable Lichess gives available False."""
    user = LICHESS_USER if user is None else user
    cache = cache or CACHE
    if not user:
        return {"available": False, "configured": False, "error": "LICHESS_USER not set"}
    q = urllib.parse.quote(user)
    sources = {}

    def fetch(name, url, parse, accept="application/json"):
        data, at, err = cache.get(url, TTL_LICHESS, parse, headers=lichess_headers(accept))
        sources[name] = {"fetched_at": iso(at), "error": err}
        return data

    profile = fetch("user", f"{LICHESS}/api/user/{q}", lambda b: parse_lichess_user(json.loads(b)))
    if profile is None:
        return {"available": False, "configured": True, "user": user, "sources": sources,
                "error": sources["user"]["error"] or "no data"}
    status = fetch("status", f"{LICHESS}/api/users/status?ids={q}&withGameIds=true",
                   lambda b: parse_lichess_online(json.loads(b)))
    games = fetch("games", f"{LICHESS}/api/games/user/{q}?max={LICHESS_GAMES}&moves=false&opening=false",
                  lambda b: parse_lichess_games(jsonl(b), profile["id"]), accept="application/x-ndjson")
    history = fetch("history", f"{LICHESS}/api/user/{q}/rating-history",
                    lambda b: parse_lichess_history(json.loads(b)))
    online = status or {"online": None, "playing": bool(profile["playing_url"]), "playing_id": None}
    if online.get("playing_id") and not profile["playing_url"]:
        profile["playing_url"] = f"{LICHESS}/{online['playing_id']}"
    return {"available": True, "configured": True, "user": user, "profile": profile,
            "online": online["online"], "playing": online["playing"] or bool(profile["playing_url"]),
            "games": games, "history": history, "sources": sources,
            "generated_at": iso(time.time())}


class Handler(BaseHTTPRequestHandler):
    server_version = "rustfish-dash"

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            try:
                with BUILD_LOCK:
                    status = build_status()
                status["machines"] = machines_status()
                body = json.dumps(status).encode()
                self._send(200, "application/json", body)
            except Exception as exc:
                self._send(500, "application/json", json.dumps({"error": str(exc)}).encode())
        elif path == "/api/lichess":
            try:
                with LICHESS_LOCK:
                    body = json.dumps(build_lichess()).encode()
                self._send(200, "application/json", body)
            except Exception as exc:
                self._send(500, "application/json", json.dumps({"available": False, "error": str(exc)}).encode())
        elif path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                self._send(200, "text/html; charset=utf-8", fh.read())
        elif path == "/healthz":
            self._send(200, "text/plain", b"ok")
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        # /api/laptop is the old name, kept for scripts/laptop_status.py.
        if path not in ("/api/machine", "/api/laptop"):
            return self._send(404, "text/plain", b"not found")
        if not machine_authorized(self.headers.get("Authorization")):
            return self._send(401, "text/plain", b"unauthorized")
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._send(411, "text/plain", b"length required")
        if n < 0 or n > MACHINE_MAX_BODY:
            return self._send(413, "text/plain", b"too large")
        try:
            payload = json.loads(self.rfile.read(n))
        except ValueError:
            return self._send(400, "text/plain", b"bad json")
        if not isinstance(payload, dict):
            return self._send(400, "text/plain", b"expected an object")
        name = machine_name(payload, "laptop" if path == "/api/laptop" else "default")
        machine_store(name, payload)
        self._send(204, "text/plain", b"")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    machines_load()
    print(f"rustfish dashboard on :{port}", flush=True)
    ThreadingHTTPServer(("", port), Handler).serve_forever()
