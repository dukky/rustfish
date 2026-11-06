#!/usr/bin/env python3
"""rustfish progress dashboard: a stdlib-only HTTP server.

Serves index.html and /api/status, a JSON digest of the public repo
dukky/rustfish built from raw.githubusercontent.com files and the
unauthenticated GitHub REST API. Every upstream fetch is cached in memory
and only refreshed when a page asks for data, so an idle dashboard makes no
requests and a busy one stays well inside the 60 requests/hour API limit.
"""
import datetime
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.environ.get("REPO", "dukky/rustfish")
RAW = f"https://raw.githubusercontent.com/{REPO}"
API = f"https://api.github.com/repos/{REPO}"
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = {"rating": 3400, "label": "Goal: 3400 CCRL Blitz"}

# Seconds each source stays fresh. API calls: ~20/h runs + ~20/h jobs + ~12/h
# datagen jobs + 6/h commits at worst, i.e. under the 60/h unauthenticated limit
# even with someone watching all day.
TTL_RAW = 120
TTL_RUNS = 180
TTL_JOBS = 180
TTL_DATAGEN_JOBS = 300
TTL_COMMITS = 600

# Fallback rounds per shard per wave, used only if scripts/queue-settings.json
# can't be fetched (waves start at rounds_per_shard and double up to
# max_rounds_per_shard). Used only to estimate games played in the in-flight SPRT.
ROUNDS = {"stc": (16, 64), "nonreg": (16, 64), "ltc-gainer": (6, 24), "ltc-check": (6, 24)}

# Positions per game in the net3 datagen set (measured, not a guess).
POSITIONS_PER_GAME = 79

# The strategy switched after plan.md was written; point the roadmap at the audit.
AUDIT_DOC = "docs/opus-audit-2026-11-04.md"


class Cache:
    """URL -> (fetched_at, payload, etag). Serves stale data when GitHub fails."""

    def __init__(self):
        self.lock = threading.Lock()
        self.entries = {}
        self.rate = {}

    def get(self, url, ttl, parse):
        now = time.time()
        with self.lock:
            ent = self.entries.get(url)
        if ent and now - ent["at"] < ttl:
            return ent["data"], ent["at"], ent.get("error")
        headers = {"User-Agent": "rustfish-dashboard", "Accept": "application/vnd.github+json"}
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
        if ent:
            # Keep stale data but retry after a minute rather than every request.
            with self.lock:
                self.entries[url] = dict(ent, at=time.time() - max(0, TTL_RAW - 60), error=msg)
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
    runs = (fetch("runs", f"{API}/actions/runs?per_page=50", TTL_RUNS, json.loads) or {}) \
        .get("workflow_runs", [])
    commits = fetch("commits", f"{API}/commits?sha=main&per_page=25", TTL_COMMITS, json.loads) or []
    # Wave sizes per queue mode, from the file scripts/queue.py itself reads.
    qcfg = fetch("queue_settings", f"{RAW}/main/scripts/queue-settings.json", TTL_RAW, json.loads) or {}
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

    # Net-training experiments (file may not exist yet; a 404 just means none logged).
    exp_file = os.environ.get("EXPERIMENTS_FILE")  # local override, for testing only
    if exp_file:
        try:
            with open(exp_file) as fh:
                exp_rows = jsonl(fh.read())
            sources["experiments"] = {"fetched_at": iso(time.time()), "error": None}
        except OSError as exc:
            exp_rows = []
            sources["experiments"] = {"fetched_at": None, "error": str(exc)}
    else:
        exp_rows, at, err = CACHE.get(f"{RAW}/main/net/experiments.jsonl", TTL_RAW, jsonl)
        if err and "HTTP 404" in err:
            err = None
        sources["experiments"] = {"fetched_at": iso(at), "error": err}
    experiments = sorted((r for r in exp_rows or [] if isinstance(r, dict)),
                         key=lambda r: str(r.get("date") or ""))

    # Strength over time.
    points = []
    for r in ladder:
        if r.get("rating") is None:
            continue
        points.append({"when": parse_when(r.get("logged_at")), "rating": r["rating"],
                       "ci": r.get("ci"), "sha": (r.get("sha") or "")[:7], "note": r.get("note"),
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
                       "when": parse_when(r.get("logged_at"))})

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
                "games": e["games"], "how": f"queue {e['mode']}", "certain": True}
    merged = sorted(merged.values(), key=lambda m: m["when"] or "")

    # In-flight test: its SPRT runs are titled "SPRT <test_id> wave <n>".
    current = queue.get("current")
    inflight = None
    if current:
        inflight = dict(current)
        waves = []
        for run in runs:
            m = re.fullmatch(rf"SPRT {re.escape(current.get('test_id', ''))} wave (\d+)",
                             run.get("display_title") or "")
            if m:
                waves.append((int(m.group(1)), run))
        if waves:
            wave, run = max(waves, key=lambda w: w[0])
            inflight.update(wave=wave, run_url=run.get("html_url"), run_status=run.get("status"),
                            run_started=run.get("run_started_at") or run.get("created_at"))
            ttl = TTL_JOBS if run.get("status") != "completed" else 3600
            jobs = (fetch("jobs", f"{API}/actions/runs/{run['id']}/jobs?per_page=100", ttl,
                          json.loads) or {}).get("jobs", [])
            shards = [j for j in jobs if re.fullmatch(r"shard \d+", j.get("name", ""))]
            done = sum(1 for j in shards if j.get("status") == "completed")
            running = sum(1 for j in shards if j.get("status") == "in_progress")
            inflight.update(shards_total=len(shards), shards_done=done, shards_running=running)
            base, cap = rounds.get(current.get("mode"), (None, None))
            if base and shards:
                rpw = [min(cap, base * 2 ** i) for i in range(wave)]
                prior = sum(len(shards) * r * 2 for r in rpw[:-1])
                inflight.update(games_prior=prior, games_wave_total=len(shards) * rpw[-1] * 2,
                                games_est=prior + done * rpw[-1] * 2)

    active = [{"name": r.get("name"), "title": r.get("display_title"), "status": r.get("status"),
               "url": r.get("html_url"), "started": r.get("run_started_at") or r.get("created_at")}
              for r in runs if r.get("status") in ("in_progress", "queued", "waiting", "pending")
              and r.get("name") != "CI"]

    # Live datagen wave for the next net: shard progress and positions produced.
    datagen = None
    dg = next((r for r in runs
               if ((r.get("name") or "").startswith("Datagen")
                   or (r.get("display_title") or "").startswith("Datagen"))
               and r.get("status") in ("in_progress", "queued", "waiting", "pending")), None)
    if dg:
        title = dg.get("display_title") or dg.get("name") or ""
        jobs_doc = fetch("datagen_jobs", f"{API}/actions/runs/{dg['id']}/jobs?per_page=100",
                         TTL_DATAGEN_JOBS, json.loads)
        jobs = (jobs_doc if isinstance(jobs_doc, dict) else {}).get("jobs", [])
        shards = [j for j in jobs if re.fullmatch(r"shard \d+", j.get("name", ""))]
        done = sum(1 for j in shards if j.get("status") == "completed")
        running = sum(1 for j in shards if j.get("status") == "in_progress")
        queued = sum(1 for j in shards if j.get("status") == "queued")
        m = re.search(r"\((\d+)x(\d+)\s+games?,\s*(\d+)\s+nodes\)", title, re.I)
        planned = games = nodes = None
        if m:
            planned, games, nodes = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            mm = re.search(r"([\d,]+)\s+games?", title, re.I)
            games = int(mm.group(1).replace(",", "")) if mm else None
            mm = re.search(r"(\d+)\s+nodes?", title, re.I)
            nodes = int(mm.group(1)) if mm else None
        mm = re.search(r"Datagen\s+(\S+)\s+seed", title, re.I)
        total = len(shards) or planned or 0
        datagen = {"title": title, "net": mm.group(1) if mm else None,
                   "run_url": dg.get("html_url"), "status": dg.get("status"),
                   "started": dg.get("run_started_at") or dg.get("created_at"),
                   "shards_total": total, "shards_done": done, "shards_running": running,
                   "shards_queued": queued, "games_per_shard": games, "nodes": nodes,
                   "estimated_positions": done * games * POSITIONS_PER_GAME if games else None}

    recent_commits = []
    for c in commits if isinstance(commits, list) else []:
        msg = (c.get("commit") or {}).get("message", "")
        bench = re.findall(r"^Bench:\s*(\d+)\s*$", msg, re.M)
        recent_commits.append({"sha": c.get("sha", "")[:7], "title": msg.splitlines()[0] if msg else "",
                               "when": ((c.get("commit") or {}).get("committer") or {}).get("date"),
                               "bench": int(bench[-1]) if bench else None, "url": c.get("html_url")})

    latest = points[-1] if points else None
    # Measured strength is the last ladder run; estimated adds the queue SPRTs
    # merged after it. Estimated is never presented as an official CCRL rating.
    measured = None
    if latest:
        measured = {k: latest[k] for k in ("rating", "ci", "when", "sha", "url", "games")}
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
        "target": TARGET,
        "strength": strength,
        "net_chain": {"nets": nets, "ref": chain_ref},
        "datagen": datagen,
        "experiments": experiments,
        "roadmap": {"audit_doc": AUDIT_DOC, "plan_stale": True},
        "current": latest and {k: latest[k] for k in ("rating", "ci", "sha", "when", "url", "games")},
        "gap": latest and round(TARGET["rating"] - latest["rating"], 1),
        "ladder": points,
        "queue": {"current": inflight, "pending": queue.get("pending", []), "paused": queue.get("paused")},
        "queue_events": events[-40:][::-1],
        "sprt": sorted(tests, key=lambda t: t["when"] or "", reverse=True)[:40],
        "merged": merged,
        "active_runs": active,
        "commits": recent_commits,
        "plan": plan,
        "sources": sources,
        "api_rate": CACHE.rate,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "rustfish-dash"

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            try:
                with BUILD_LOCK:
                    status = build_status()
                body = json.dumps(status).encode()
                self._send(200, "application/json", body)
            except Exception as exc:
                self._send(500, "application/json", json.dumps({"error": str(exc)}).encode())
        elif path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                self._send(200, "text/html; charset=utf-8", fh.read())
        elif path == "/healthz":
            self._send(200, "text/plain", b"ok")
        else:
            self._send(404, "text/plain", b"not found")

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
    print(f"rustfish dashboard on :{port}", flush=True)
    ThreadingHTTPServer(("", port), Handler).serve_forever()
