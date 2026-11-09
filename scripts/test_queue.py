#!/usr/bin/env python3
"""Unit tests for scripts/queue.py (stdlib unittest only).

    python3 scripts/test_queue.py

The fast-forward / replay tests run real git against a throwaway bare
"origin" and clone in a temp dir; cargo, gh and the engine are mocked, so
nothing touches GitHub or builds anything.
"""
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mergequeue", os.path.join(HERE, "queue.py"))
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)

BENCH = 634443


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


class GitRepo(unittest.TestCase):
    """A bare origin with main, a clone the queue works in, and a state dir."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="queue-test-")
        self.origin = os.path.join(self.tmp, "origin.git")
        self.work = os.path.join(self.tmp, "work")
        self.dev = os.path.join(self.tmp, "dev")  # where "humans" make branches
        q_state = os.path.join(self.tmp, "state")
        os.makedirs(q_state)
        git("init", "-q", "--bare", "-b", "main", self.origin, cwd=self.tmp)
        git("clone", "-q", self.origin, self.dev, cwd=self.tmp)
        for k, v in (("user.name", "t"), ("user.email", "t@t"), ("commit.gpgsign", "false")):
            git("config", k, v, cwd=self.dev)
        self.commit(self.dev, "src/search.rs", "search v1\n", "init")
        self.commit(self.dev, "src/eval.rs", "eval v1\n", "eval")
        git("push", "-q", "origin", "HEAD:main", cwd=self.dev)
        git("clone", "-q", self.origin, self.work, cwd=self.tmp)
        for k, v in (("user.name", "q"), ("user.email", "q@q"), ("commit.gpgsign", "false")):
            git("config", k, v, cwd=self.work)

        real_sh = q.sh
        self.calls = []

        self.before = None  # hook: called with each command before it runs

        def fake_sh(*cmd, cwd=q.REPO, check=True):
            self.calls.append(cmd)
            if self.before:
                self.before(cmd)
            if cmd[0] in ("cargo", "gh"):
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return real_sh(*cmd, cwd=self.work if cwd == q.REPO else cwd, check=check)

        self.logs = []
        engine = f"perfttest summary 6/6 passed\nselftest ok\nNodes searched  : {BENCH}\n"
        self.patches = [
            mock.patch.object(q, "sh", fake_sh),
            mock.patch.object(q, "STATE", q_state),
            mock.patch.object(q, "log", self.logs.append),
            mock.patch.object(q, "run_engine", lambda cmds: engine),
            mock.patch.object(q, "eval_mismatch", lambda: None),
            mock.patch.object(q, "main_bench", lambda: BENCH),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def commit(self, repo, path, text, msg):
        full = os.path.join(repo, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(text)
        git("add", "-A", cwd=repo)
        git("commit", "-q", "-m", f"{msg}\n\nBench: {BENCH}\n\nCo-Authored-By: x", cwd=repo)
        return git("rev-parse", "HEAD", cwd=repo)

    def branch(self, name, path, text):
        """Push a branch off the current origin/main with one commit."""
        git("fetch", "-q", "origin", cwd=self.dev)
        git("checkout", "-q", "-B", name, "origin/main", cwd=self.dev)
        sha = self.commit(self.dev, path, text, name)
        git("push", "-q", "--force", "origin", f"{name}:{name}", cwd=self.dev)
        return sha

    def main_sha(self):
        return git("rev-parse", "main", cwd=self.origin)

    def tested_candidate(self):
        """An SPRT candidate tested against the current main: (cur, result)."""
        base = self.main_sha()
        sha = self.branch("cand", "src/search.rs", "search v2 (tested)\n")
        cur = {"branch": "cand", "mode": "stc", "test_id": "q-cand-1", "sha": sha,
               "base_sha": base, "bench": BENCH + 1}
        return cur, {"test_id": "q-cand-1", "status": "H1", "elo": 6.0, "games": 4000}

    def merge_speedup(self, state):
        """Queue a speedup through the real start() (cargo/engine mocked)."""
        self.branch("fast", "src/eval.rs", "eval v1 but faster\n")
        self.assertIsNone(q.start({"branch": "fast", "mode": "speedup"}, state))
        self.assertEqual(self.logs[-1]["outcome"], "merged", self.logs[-1])


class SpeedupAndReplay(GitRepo):
    def test_speedup_records_identical_merge(self):
        state = {"pending": []}
        before = self.main_sha()
        self.merge_speedup(state)
        self.assertNotEqual(self.main_sha(), before)
        self.assertEqual(state["identical_merges"], [self.main_sha()])

    def test_sprt_after_identical_merge_is_replayed_with_bench_check(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        self.merge_speedup(state)
        speedup_main = self.main_sha()
        with mock.patch.object(q, "checked_bench", return_value=BENCH + 1) as cb:
            q.finish(state, cur, res)
        cb.assert_called_once()
        self.assertEqual(self.logs[-1]["outcome"], "merged", self.logs[-1])
        new_main = self.main_sha()
        self.assertNotEqual(new_main, cur["sha"])  # replayed, not the tested sha
        self.assertEqual(git("rev-parse", f"{new_main}^", cwd=self.origin), speedup_main)
        self.assertEqual(git("show", f"{new_main}:src/search.rs", cwd=self.origin), "search v2 (tested)")
        self.assertEqual(git("show", f"{new_main}:src/eval.rs", cwd=self.origin), "eval v1 but faster")
        self.assertEqual(git("rev-parse", "cand", cwd=self.origin), new_main)
        with open(os.path.join(q.STATE, "main-bench.json")) as fh:
            self.assertEqual(json.load(fh), {"sha": new_main, "nodes": BENCH + 1})
        self.assertEqual(state["pending"], [])

    def test_replay_bench_mismatch_requeues(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        self.merge_speedup(state)
        speedup_main = self.main_sha()
        with mock.patch.object(q, "checked_bench", return_value=BENCH + 7):
            q.finish(state, cur, res)
        self.assertEqual(self.logs[-1]["outcome"], "requeued")
        self.assertIn("replayed bench", self.logs[-1]["reason"])
        self.assertEqual(self.main_sha(), speedup_main)
        self.assertEqual(state["pending"], [{"branch": "cand", "mode": "stc", "front": True}])

    def test_sprt_after_non_identical_engine_change_is_requeued(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        git("checkout", "-q", "-B", "direct", "origin/main", cwd=self.dev)
        self.commit(self.dev, "src/eval.rs", "eval v2 (functional)\n", "eval change")
        git("push", "-q", "origin", "direct:main", cwd=self.dev)
        moved = self.main_sha()
        with mock.patch.object(q, "checked_bench") as cb:
            q.finish(state, cur, res)
        cb.assert_not_called()
        self.assertEqual(self.logs[-1]["outcome"], "requeued")
        self.assertIn("engine change", self.logs[-1]["reason"])
        self.assertEqual(self.main_sha(), moved)
        self.assertEqual(state["pending"][0]["branch"], "cand")

    def test_identical_merge_plus_functional_change_is_requeued(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        self.merge_speedup(state)
        git("fetch", "-q", "origin", cwd=self.dev)
        git("checkout", "-q", "-B", "direct", "origin/main", cwd=self.dev)
        self.commit(self.dev, "src/eval.rs", "eval v2 (functional)\n", "eval change")
        git("push", "-q", "origin", "direct:main", cwd=self.dev)
        q.finish(state, cur, res)
        self.assertEqual(self.logs[-1]["outcome"], "requeued")

    def test_tooling_only_move_replays_without_rebuild(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        git("checkout", "-q", "-B", "direct", "origin/main", cwd=self.dev)
        self.commit(self.dev, "docs/x.md", "notes\n", "docs")
        git("push", "-q", "origin", "direct:main", cwd=self.dev)
        with mock.patch.object(q, "checked_bench") as cb:
            q.finish(state, cur, res)
        cb.assert_not_called()
        self.assertEqual(self.logs[-1]["outcome"], "merged")
        self.assertEqual(git("show", "main:docs/x.md", cwd=self.origin), "notes")

    def test_unmoved_main_fast_forwards_to_tested_sha(self):
        cur, res = self.tested_candidate()
        q.finish({"pending": [], "current": cur}, cur, res)
        self.assertEqual(self.main_sha(), cur["sha"])


class Advance(unittest.TestCase):
    def run_advance(self, state, start_result=None):
        started, logs = [], []

        def fake_start(item, st=None, priority=None):
            started.append((item["branch"], priority))
            return start_result(item) if start_result else None

        with mock.patch.object(q, "load_state", return_value=state), \
                mock.patch.object(q, "save_state"), \
                mock.patch.object(q, "find_result", return_value=None), \
                mock.patch.object(q, "background_lane"), \
                mock.patch.object(q, "read_log", return_value=[]), \
                mock.patch.object(q, "log", logs.append), \
                mock.patch.object(q, "start", fake_start), \
                mock.patch.dict(os.environ, {"RESULT": ""}):
            q.advance()
        return started, logs

    def test_speedups_processed_while_sprt_in_flight(self):
        cur = {"branch": "ltc", "mode": "ltc-gainer", "test_id": "q-ltc-1", "sha": "a", "base_sha": "b"}
        state = {"pending": [{"branch": "s1", "mode": "speedup"}, {"branch": "x", "mode": "stc"},
                             {"branch": "s2", "mode": "speedup"}], "current": cur}
        started, _ = self.run_advance(state)
        self.assertEqual([b for b, _ in started], ["s1", "s2"])
        self.assertIs(state["current"], cur)
        self.assertEqual(state["pending"], [{"branch": "x", "mode": "stc"}])

    def test_speedups_processed_while_ladder_in_flight(self):
        cur = {"branch": "main", "mode": "ladder", "test_id": "q-ladder-1", "sha": "a", "base_sha": "a"}
        state = {"pending": [{"branch": "s1", "mode": "speedup"}], "current": cur}
        started, _ = self.run_advance(state)
        self.assertEqual([b for b, _ in started], ["s1"])
        self.assertIs(state["current"], cur)

    def test_paused_queue_merges_nothing(self):
        state = {"pending": [{"branch": "s1", "mode": "speedup"}], "current": None, "paused": "x"}
        started, _ = self.run_advance(state)
        self.assertEqual(started, [])

    def test_idle_queue_dispatches_by_score_after_speedups(self):
        state = {"pending": [{"branch": "ltc", "mode": "ltc-gainer"}, {"branch": "s", "mode": "speedup"},
                             {"branch": "stc", "mode": "stc"}], "current": None}
        started, _ = self.run_advance(
            state, lambda item: {"test_id": f"q-{item['branch']}", "sha": "x", "base_sha": "y"})
        self.assertEqual([b for b, _ in started], ["s", "stc"])
        self.assertEqual(started[1][1]["rule"], "score")
        self.assertEqual(state["current"]["test_id"], "q-stc")
        self.assertEqual(state["pending"], [{"branch": "ltc", "mode": "ltc-gainer", "passed_over": 1}])


def log_rows():
    """A small history: stc accepts 1 in 4 at +10, ltc-gainer accepts 1 in 4 at +10 but 3x longer."""
    rows = []
    for mode, hours in (("stc", 1), ("ltc-gainer", 3), ("net-gate", 0.2)):
        for i, verdict in enumerate(["H1", "H0", "H0", "H0"] * 3):
            tid = f"{mode}-{i}"
            rows.append({"outcome": "dispatched", "test_id": tid, "mode": mode,
                         "logged_at": "2026-11-01T00:00:00Z"})
            end = f"2026-11-01T{int(hours):02d}:{int(hours % 1 * 60):02d}:00Z"
            rows.append({"outcome": "merged" if verdict == "H1" else "rejected", "test_id": tid,
                         "mode": mode, "verdict": verdict, "elo": 10.0 if verdict == "H1" else -3.0,
                         "logged_at": end})
    return rows


class Priority(unittest.TestCase):
    def setUp(self):
        self.rates = q.base_rates(log_rows())

    def test_base_rates(self):
        stc = self.rates["stc"]
        self.assertEqual(stc["n"], 12)
        self.assertAlmostEqual(stc["p_accept"], (3 + 3 * q.PRIOR_P) / 15)
        self.assertAlmostEqual(stc["gain"], (30 + 3 * q.PRIOR_GAIN) / 6)
        self.assertAlmostEqual(stc["hours_reject"], 1.0)
        # Few samples: falls back to the priors.
        self.assertEqual(self.rates["nonreg"]["n"], 0)
        self.assertAlmostEqual(self.rates["nonreg"]["p_accept"], q.PRIOR_P)
        self.assertAlmostEqual(self.rates["nonreg"]["gain"], q.PRIOR_GAIN)

    def test_gain_is_clipped(self):
        rows = [{"outcome": "merged", "test_id": "t", "mode": "stc", "verdict": "H1", "elo": 500.0,
                 "logged_at": "2026-11-01T00:00:00Z"}]
        self.assertAlmostEqual(q.base_rates(rows)["stc"]["gain"], (q.GAIN_CAP + 3 * q.PRIOR_GAIN) / 4)

    def test_cheaper_mode_first(self):
        pending = [{"branch": "l", "mode": "ltc-gainer"}, {"branch": "s", "mode": "stc"}]
        idx, pri = q.pick_next(pending, self.rates)
        self.assertEqual(idx, 1)
        self.assertEqual(pri["rule"], "score")
        for k in ("score", "p_accept", "gain", "gain_source", "runner_hours", "samples"):
            self.assertIn(k, pri)
        self.assertEqual(pending[0]["passed_over"], 1)

    def test_fifo_on_ties(self):
        pending = [{"branch": "a", "mode": "stc"}, {"branch": "b", "mode": "stc"}]
        self.assertEqual(q.pick_next(pending, self.rates)[0], 0)

    def test_expect_overrides_gain(self):
        pending = [{"branch": "s", "mode": "stc"}, {"branch": "l", "mode": "ltc-gainer", "expect": 80.0}]
        idx, pri = q.pick_next(pending, self.rates)
        self.assertEqual(idx, 1)
        self.assertEqual(pri["gain_source"], "expect")
        self.assertEqual(pri["gain"], 80.0)

    def test_front_overrides_score(self):
        pending = [{"branch": "s", "mode": "stc"}, {"branch": "l", "mode": "ltc-gainer", "front": True}]
        self.assertEqual(q.pick_next(pending, self.rates), (1, {"rule": "front"}))

    def test_scheduled_items_keep_their_place(self):
        pending = [{"branch": "l", "mode": "ltc-gainer"}, {"branch": "main", "mode": "ladder"},
                   {"branch": "s", "mode": "stc"}]
        self.assertEqual(q.pick_next(pending, self.rates)[0], 0)  # stc can't jump the ladder
        pending.pop(0)
        self.assertEqual(q.pick_next(pending, self.rates), (0, {"rule": "scheduled"}))

    def test_front_item_behind_ladder_waits(self):
        pending = [{"branch": "main", "mode": "ltc-check", "base": "x"},
                   {"branch": "f", "mode": "stc", "front": True}]
        self.assertEqual(q.pick_next(pending, self.rates)[0], 0)

    def test_starvation_guard(self):
        pending = [{"branch": "l", "mode": "ltc-gainer"}] + \
                  [{"branch": f"s{i}", "mode": "stc"} for i in range(5)]
        picks = []
        while pending:
            picks.append(pending.pop(q.pick_next(pending, self.rates)[0])["branch"])
        self.assertEqual(picks[:4], ["s0", "s1", "s2", "l"])

    def test_requeue_and_retry_go_to_front(self):
        state = {"pending": [{"branch": "s", "mode": "stc"}]}
        q.retry(state, {"branch": "r", "mode": "net-gate", "expect": 9.0}, {})
        self.assertEqual(state["pending"][0], {"branch": "r", "mode": "net-gate", "expect": 9.0,
                                               "retried": True, "front": True})
        q.retry(state, {"branch": "b", "mode": "stc", "front": True}, {}, back=True)
        self.assertNotIn("front", state["pending"][-1])

    def test_add_args(self):
        self.assertEqual(q.parse_add_args(["nnue-g3:net-gate", "--expect", "25"]),
                         {"add": "nnue-g3:net-gate", "front": False, "preempt": False, "expect": 25.0})
        self.assertEqual(q.parse_add_args(["a:stc", "--preempt"]),
                         {"add": "a:stc", "front": True, "preempt": True})
        self.assertEqual(q.parse_adds("a:stc,b:nonreg", 4),
                         [{"branch": "a", "mode": "stc", "expect": 4.0},
                          {"branch": "b", "mode": "nonreg", "expect": 4.0}])
        with self.assertRaises(SystemExit):
            q.parse_add_args(["a:stc", "--expect", "lots"])
        with self.assertRaises(SystemExit):
            q.parse_add_args(["a:stc", "--bogus"])


class Guard(unittest.TestCase):
    def test_live_settings_pass(self):
        with open(os.path.join(HERE, "queue-settings.json")) as fh:
            q.validate_settings(json.load(fh))

    def test_too_many_shards_fails_loudly(self):
        cfg = {"modes": {"stc": {"shards": q.MAX_FOREGROUND_SHARDS}, "ltc-gainer": {"shards": 19}}}
        with self.assertRaises(SystemExit) as ctx:
            q.validate_settings(cfg)
        self.assertIn("ltc-gainer (shards=19)", str(ctx.exception))
        self.assertNotIn("stc (", str(ctx.exception))

    def test_missing_shards_fails(self):
        with self.assertRaises(SystemExit):
            q.validate_settings({"modes": {"stc": {"tc": "10+0.1"}}})


def runs_by_workflow(datagen=(), others=None):
    """A fake `out` for gh run list: datagen.yml runs, plus {workflow: runs} for the rest."""
    others = others or {}

    def fake_out(*c, **k):
        if c[:3] == ("gh", "run", "list"):
            wf = c[c.index("--workflow") + 1]
            return json.dumps(list(datagen) if wf == "datagen.yml" else others.get(wf, []))
        return "sha"
    return fake_out


class BackgroundLane(unittest.TestCase):
    def lane(self, runs, finished="", others=None, logs=None):
        cmds = []
        with mock.patch.object(q, "out", runs_by_workflow(runs, others)), \
                mock.patch.object(q, "sh", lambda *c, **k: cmds.append(c)), \
                mock.patch.object(q, "log", (logs if logs is not None else []).append), \
                mock.patch.dict(os.environ, {"FINISHED_RUN": finished}):
            state = {"next_seed": 20}
            q.background_lane(state)
        return state, cmds

    def test_calling_back_run_is_ignored(self):
        state, cmds = self.lane([{"databaseId": 42, "status": "in_progress"}], "42")
        self.assertEqual(state["next_seed"], 21)
        self.assertIn("seed=20", cmds[0])

    def test_other_running_datagen_blocks(self):
        state, cmds = self.lane([{"databaseId": 41, "status": "in_progress"}], "42")
        self.assertEqual((state["next_seed"], cmds), (20, []))

    def test_running_burst_does_not_block(self):
        title = "Datagen main seed 19 (16x15000 games, 5000 nodes)"
        cmds = []
        runs = [{"databaseId": 41, "status": "in_progress", "displayTitle": title}]
        with mock.patch.object(q, "out", runs_by_workflow(runs)), \
                mock.patch.object(q, "sh", lambda *c, **k: cmds.append(c)), \
                mock.patch.object(q, "log"), \
                mock.patch.dict(os.environ, {"FINISHED_RUN": ""}):
            state = {"next_seed": 20, "current": {"mode": q.BURST_MODE, "seed": 19, "test_id": "q-burst-s19"}}
            q.background_lane(state)
        self.assertEqual(state["next_seed"], 21)
        self.assertIn("seed=20", cmds[0])
        self.assertIn(f"max_parallel={q.BACKGROUND_PARALLEL}", cmds[0])
        self.assertIn(f"shards={q.BACKGROUND_SHARDS}", cmds[0])


BURST_ON = {"enabled": True, "below_score": 0.04, "max_consecutive": 2, "when_idle": True,
            "shards": q.MAX_FOREGROUND_SHARDS, "max_parallel": q.MAX_FOREGROUND_SHARDS, "games_per_shard": 15000, "nodes": 5000}
LOW = {"branch": "simp", "mode": "nonreg", "expect": 1.0}     # score ~0.01
HIGH = {"branch": "feat", "mode": "stc", "expect": 20.0}      # score ~0.3


class Burst(unittest.TestCase):
    """Datagen bursts on the foreground runners (burst_reason / start_burst / burst_done)."""

    def run_advance(self, state, burst=BURST_ON, runs=(), finished="", real_lane=False):
        """One advance with gh mocked. Returns (started items, datagen dispatches, log rows)."""
        started, dispatched, logs = [], [], []

        def fake_start(item, st=None, priority=None):
            started.append(item["branch"])
            return {"test_id": f"q-{item['branch']}", "sha": "x", "base_sha": "y"}

        def fake_out(*cmd, **kw):
            if cmd[:3] == ("gh", "run", "list"):
                return json.dumps(list(runs))
            return "mainsha"

        def fake_sh(*cmd, **kw):
            if cmd[:4] == ("gh", "workflow", "run", "datagen.yml"):
                dispatched.append(dict(a.split("=", 1) for a in cmd if "=" in a and not a.startswith("--")))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        patches = [mock.patch.object(q, "load_state", return_value=state),
                   mock.patch.object(q, "save_state"),
                   mock.patch.object(q, "find_result", return_value=None),
                   mock.patch.object(q, "read_log", return_value=log_rows()),
                   mock.patch.object(q, "log", logs.append),
                   mock.patch.object(q, "start", fake_start),
                   mock.patch.object(q, "out", fake_out),
                   mock.patch.object(q, "sh", fake_sh),
                   mock.patch.object(q, "BURST", burst),
                   mock.patch.dict(os.environ, {"RESULT": "", "FINISHED_RUN": finished})]
        if not real_lane:
            patches.append(mock.patch.object(q, "background_lane"))
        for p in patches:
            p.start()
        try:
            q.advance()
        finally:
            for p in reversed(patches):
                p.stop()
        return started, dispatched, logs

    def done_runs(self, state):
        """The in-flight burst's datagen run, completed."""
        seed = state["current"]["seed"]
        return [{"databaseId": 900 + seed, "status": "completed",
                 "displayTitle": f"Datagen main seed {seed} (16x15000 games, 5000 nodes)"}]

    def test_burst_when_best_score_below_threshold(self):
        state = {"pending": [dict(LOW)], "current": None, "next_seed": 30}
        started, dispatched, logs = self.run_advance(state)
        self.assertEqual(started, [])
        self.assertEqual(dispatched, [{"ref": "main", "shards": str(q.MAX_FOREGROUND_SHARDS), "games_per_shard": "15000",
                                       "nodes": "5000", "seed": "30", "max_parallel": str(q.MAX_FOREGROUND_SHARDS)}])
        self.assertEqual(state["current"]["mode"], "datagen-burst")
        self.assertEqual(state["current"]["seed"], 30)
        self.assertEqual(state["next_seed"], 31)
        self.assertEqual(state["pending"], [LOW])  # keeps its place, not passed over
        self.assertEqual(state["consecutive_bursts"], 1)
        row = logs[-1]
        self.assertEqual((row["mode"], row["outcome"], row["seed"]), ("datagen-burst", "dispatched", 30))
        self.assertIn("simp:nonreg", row["reason"])

    def test_no_burst_when_high_value_item_pending(self):
        state = {"pending": [dict(LOW), dict(HIGH)], "current": None, "next_seed": 30}
        started, dispatched, _ = self.run_advance(state)
        self.assertEqual((started, dispatched), (["feat"], []))
        self.assertEqual(state["current"]["test_id"], "q-feat")
        self.assertEqual(state["next_seed"], 30)

    def test_no_burst_for_front_or_scheduled_items(self):
        for item in (dict(LOW, front=True), {"branch": "main", "mode": "ladder"}):
            state = {"pending": [item], "current": None, "next_seed": 30}
            started, dispatched, _ = self.run_advance(state)
            self.assertEqual((started, dispatched), ([item["branch"]], []))

    def test_consecutive_burst_cap(self):
        state = {"pending": [dict(LOW), dict(LOW, branch="simp2")], "current": None, "next_seed": 30}
        seq = []
        for _ in range(6):
            runs = self.done_runs(state) if state.get("current", {}) and \
                state["current"]["mode"] == "datagen-burst" else []
            if state.get("current") and state["current"]["mode"] != "datagen-burst":
                state["current"] = None  # the item's SPRT ended
            started, dispatched, _ = self.run_advance(state, runs=runs)
            seq += [f"burst{d['seed']}" for d in dispatched] + started
        self.assertEqual(seq, ["burst30", "burst31", "simp", "burst32", "burst33", "simp2"])
        self.assertEqual(state["next_seed"], 34)
        self.assertEqual(state["consecutive_bursts"], 0)

    def test_burst_ends_on_its_own_callback_only(self):
        title = "Datagen main seed 30 (16x15000 games, 5000 nodes)"
        bg = "Datagen main seed 29 (16x15000 games, 5000 nodes)"
        cur = {"branch": "main", "mode": "datagen-burst", "test_id": "q-burst-s30", "seed": 30,
               "started": q.now()}
        runs = [{"databaseId": 7, "status": "in_progress", "displayTitle": title},
                {"databaseId": 6, "status": "in_progress", "displayTitle": bg}]
        # The background lane's run calling back does not end the burst.
        state = {"pending": [dict(HIGH)], "current": dict(cur), "next_seed": 31}
        started, _, _ = self.run_advance(state, runs=runs, finished="6")
        self.assertEqual((started, state["current"]["test_id"]), ([], "q-burst-s30"))
        # Its own callback does (the run is still in progress while it calls back),
        # and the queue re-picks at once: the high-value item added meanwhile goes.
        started, dispatched, logs = self.run_advance(state, runs=runs, finished="7")
        self.assertEqual((started, dispatched), (["feat"], []))
        self.assertIn({"mode": "datagen-burst", "outcome": "burst-done", "seed": 30},
                      [{k: r.get(k) for k in ("mode", "outcome", "seed")} for r in logs])

    def test_lost_burst_does_not_wedge(self):
        old = (datetime.datetime.utcnow() - datetime.timedelta(hours=3)).isoformat(timespec="seconds") + "Z"
        cur = {"branch": "main", "mode": "datagen-burst", "test_id": "q-burst-s30", "seed": 30, "started": old}
        state = {"pending": [dict(HIGH)], "current": dict(cur), "next_seed": 31}
        started, _, _ = self.run_advance(state)
        self.assertEqual(started, ["feat"])
        state = {"pending": [dict(HIGH)], "current": dict(cur, started=q.now()), "next_seed": 31}
        started, _, _ = self.run_advance(state)
        self.assertEqual(started, [])  # just dispatched, not listed yet: keep waiting

    def test_seed_counter_shared_with_background_lane(self):
        state = {"pending": [dict(LOW)], "current": None, "next_seed": 40}
        _, dispatched, logs = self.run_advance(state, real_lane=True)
        self.assertEqual([(d["seed"], d["max_parallel"]) for d in dispatched], [("40", str(q.MAX_FOREGROUND_SHARDS)), ("41", str(q.BACKGROUND_PARALLEL))])
        self.assertEqual([r["mode"] for r in logs], ["datagen-burst", "datagen"])
        self.assertEqual(state["next_seed"], 42)

    def test_idle_foreground_bursts(self):
        state = {"pending": [], "current": None, "next_seed": 50, "consecutive_bursts": 2}
        _, dispatched, logs = self.run_advance(state)
        self.assertEqual([d["seed"] for d in dispatched], ["50"])
        self.assertEqual(logs[-1]["reason"], "foreground idle")
        self.assertEqual(state["consecutive_bursts"], 0)  # nothing waited on it
        state = {"pending": [], "current": None, "next_seed": 50}
        _, dispatched, _ = self.run_advance(state, burst=dict(BURST_ON, when_idle=False))
        self.assertEqual((dispatched, state["current"]), ([], None))

    def test_disabled(self):
        for burst in (dict(BURST_ON, enabled=False), {}):
            state = {"pending": [dict(LOW)], "current": None, "next_seed": 30}
            started, dispatched, _ = self.run_advance(state, burst=burst)
            self.assertEqual((started, dispatched), (["simp"], []))
            state = {"pending": [], "current": None, "next_seed": 30}
            _, dispatched, _ = self.run_advance(state, burst=burst)
            self.assertEqual(dispatched, [])

    def test_paused_queue_does_not_burst(self):
        state = {"pending": [dict(LOW)], "current": None, "next_seed": 30, "paused": "x"}
        self.assertEqual(self.run_advance(state)[1], [])

    def test_settings_validation(self):
        base = {"modes": {"stc": {"shards": q.MAX_FOREGROUND_SHARDS}}}
        q.validate_settings(dict(base, burst=BURST_ON))
        q.validate_settings(dict(base, burst=dict(BURST_ON, enabled=False, shards=5)))
        for bad in ({"shards": q.MAX_FOREGROUND_SHARDS + 1}, {"max_parallel": 17, "shards": 17}, {"max_parallel": 8, "shards": 12}):
            with self.assertRaises(SystemExit):
                q.validate_settings(dict(base, burst=dict(BURST_ON, **bad)))


class RacesAndPins(GitRepo):
    """Lost races for main, force-pushed branches and leases (no wedges)."""

    def push_to_main(self, path, text, msg):
        git("fetch", "-q", "origin", cwd=self.dev)
        git("checkout", "-q", "-B", "direct", "origin/main", cwd=self.dev)
        self.commit(self.dev, path, text, msg)
        git("push", "-q", "origin", "direct:main", cwd=self.dev)
        return self.main_sha()

    def race_main_push(self, path, times=1):
        """Someone pushes to main right before each of the queue's next `times` main pushes."""
        count = [0]

        def before(cmd):
            if cmd[:2] == ("git", "push") and cmd[-1].endswith(":refs/heads/main") and count[0] < times:
                count[0] += 1
                self.push_to_main(path, f"racer {count[0]}\n", f"racer {count[0]}")
        self.before = before
        return count

    def test_lost_race_to_tooling_commit_is_retried_and_replayed(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        count = self.race_main_push("docs/plan.md")
        q.finish(state, cur, res)
        self.assertEqual(count[0], 1)
        self.assertEqual(self.logs[-1]["outcome"], "merged", self.logs[-1])
        new_main = self.main_sha()
        self.assertEqual(git("show", f"{new_main}:docs/plan.md", cwd=self.origin), "racer 1")
        self.assertEqual(git("show", f"{new_main}:src/search.rs", cwd=self.origin), "search v2 (tested)")
        self.assertEqual(git("rev-parse", "cand", cwd=self.origin), new_main)
        self.assertEqual(state["pending"], [])

    def test_lost_race_to_engine_change_requeues_and_keeps_branch(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        self.race_main_push("src/eval.rs")
        q.finish(state, cur, res)  # must not raise
        self.assertEqual(self.logs[-1]["outcome"], "requeued")
        self.assertIn("engine change", self.logs[-1]["reason"])
        self.assertEqual(state["pending"], [{"branch": "cand", "mode": "stc", "front": True}])
        # Nothing was force-pushed before main moved: the tested sha is still the branch.
        self.assertEqual(git("rev-parse", "cand", cwd=self.origin), cur["sha"])

    def test_main_that_keeps_moving_requeues(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        count = self.race_main_push("docs/plan.md", times=10)
        q.finish(state, cur, res)
        self.assertEqual(count[0], q.MAIN_PUSH_ATTEMPTS)
        self.assertEqual(self.logs[-1]["outcome"], "requeued")
        self.assertIn("kept moving", self.logs[-1]["reason"])
        self.assertEqual(git("rev-parse", "cand", cwd=self.origin), cur["sha"])

    def force_push_branch(self):
        git("fetch", "-q", "origin", cwd=self.dev)
        git("checkout", "-q", "-B", "cand", "origin/main", cwd=self.dev)
        human = self.commit(self.dev, "src/search.rs", "search v3 (rewritten)\n", "rewrite")
        git("push", "-q", "--force", "origin", "cand:cand", cwd=self.dev)
        return human

    def test_force_pushed_branch_without_pin_requeues_instead_of_wedging(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        self.force_push_branch()
        q.finish(state, cur, res)
        self.assertEqual(self.logs[-1]["outcome"], "requeued")
        self.assertIn("on no ref", self.logs[-1]["reason"])

    def test_pinned_sha_merges_after_force_push(self):
        cur, res = self.tested_candidate()
        git("push", "-q", "origin", f"{cur['sha']}:{q.tested_ref(cur['test_id'])}", cwd=self.dev)
        state = {"pending": [], "current": cur}
        human = self.force_push_branch()
        q.finish(state, cur, res)
        self.assertEqual(self.logs[-1]["outcome"], "merged", self.logs[-1])
        self.assertEqual(self.main_sha(), cur["sha"])
        self.assertEqual(git("rev-parse", "cand", cwd=self.origin), human)  # left alone

    def test_replay_does_not_clobber_a_changed_branch(self):
        cur, res = self.tested_candidate()
        git("push", "-q", "origin", f"{cur['sha']}:{q.tested_ref(cur['test_id'])}", cwd=self.dev)
        state = {"pending": [], "current": cur}
        self.push_to_main("docs/x.md", "notes\n", "docs")
        human = self.force_push_branch()
        q.finish(state, cur, res)
        self.assertEqual([r["outcome"] for r in self.logs[-2:]], ["needs-human", "merged"])
        self.assertIn("changed during the test", self.logs[-2]["reason"])
        self.assertEqual(git("show", "main:src/search.rs", cwd=self.origin), "search v2 (tested)")
        self.assertEqual(git("show", "main:docs/x.md", cwd=self.origin), "notes")
        self.assertEqual(git("rev-parse", "cand", cwd=self.origin), human)

    def test_replay_evalhash_mismatch_requeues(self):
        cur, res = self.tested_candidate()
        cur["evalhash"] = "aaaa"
        state = {"pending": [], "current": cur}
        self.merge_speedup(state)
        with mock.patch.object(q, "checked_bench", return_value=BENCH + 1), \
                mock.patch.object(q, "eval_hash", return_value="bbbb"):
            q.finish(state, cur, res)
        self.assertEqual(self.logs[-1]["outcome"], "requeued")
        self.assertIn("evalhash", self.logs[-1]["reason"])

    def test_replay_runs_cargo_test(self):
        cur, res = self.tested_candidate()
        state = {"pending": [], "current": cur}
        self.merge_speedup(state)
        self.calls.clear()
        q.checked_bench()
        self.assertIn(("cargo", "test", "--release", "-q"), self.calls)

    def test_dispatch_pins_tested_sha_and_records_evalhash(self):
        self.branch("feat", "src/search.rs", "search v2\n")
        with mock.patch.object(q, "eval_hash", return_value="abcd"):
            rec = q.start({"branch": "feat", "mode": "stc"}, {"pending": []})
        pinned = git("rev-parse", q.tested_ref(rec["test_id"]), cwd=self.origin)
        self.assertEqual(pinned, rec["sha"])
        self.assertEqual(rec["evalhash"], "abcd")
        self.assertTrue(any(c[:4] == ("gh", "workflow", "run", "sprt.yml") for c in self.calls))

    def test_branch_pushed_during_prepare_is_not_clobbered(self):
        self.branch("feat", "src/search.rs", "search v2\n")
        state = {"pending": []}
        human = []

        def before(cmd):
            if cmd[:2] == ("cargo", "build") and not human:
                git("checkout", "-q", "feat", cwd=self.dev)
                human.append(self.commit(self.dev, "src/search.rs", "search v2b\n", "more"))
                git("push", "-q", "origin", "feat:feat", cwd=self.dev)
        self.before = before
        self.assertIsNone(q.start({"branch": "feat", "mode": "stc"}, state))
        self.assertEqual(self.logs[-1]["outcome"], "retrying")
        self.assertEqual(git("rev-parse", "feat", cwd=self.origin), human[0])
        self.assertEqual(state["pending"], [{"branch": "feat", "mode": "stc", "retried": True, "front": True}])
        self.assertFalse(any(c[:4] == ("gh", "workflow", "run", "sprt.yml") for c in self.calls))

    def test_speedup_refused_because_main_moved_is_requeued(self):
        self.branch("fast", "src/eval.rs", "eval v1 but faster\n")
        state = {"pending": []}
        moved = []

        def before(cmd):
            if cmd[:2] == ("cargo", "build") and not moved:
                moved.append(self.push_to_main("docs/x.md", "notes\n", "docs"))
        self.before = before
        self.assertIsNone(q.start({"branch": "fast", "mode": "speedup"}, state))
        self.assertEqual(self.logs[-1]["outcome"], "requeued", self.logs[-1])
        self.assertEqual(state["pending"], [{"branch": "fast", "mode": "speedup", "front": True}])
        self.assertEqual(self.main_sha(), moved[0])

    def test_bench_trailer_added_without_co_author(self):
        git("fetch", "-q", "origin", cwd=self.dev)
        git("checkout", "-q", "-B", "feat", "origin/main", cwd=self.dev)
        with open(os.path.join(self.dev, "src/search.rs"), "w") as fh:
            fh.write("search v2\n")
        git("commit", "-q", "-am", "Plain message, no trailers", cwd=self.dev)
        git("push", "-q", "origin", "feat:feat", cwd=self.dev)
        with mock.patch.object(q, "main_bench", lambda: BENCH - 5):
            q.start({"branch": "feat", "mode": "stc"}, {"pending": []})
        msg = git("log", "-1", "--format=%B", "feat", cwd=self.origin)
        self.assertEqual(q.BENCH_RE.findall(msg), [str(BENCH)])


class BenchTrailer(unittest.TestCase):
    def test_rewrite(self):
        w = q.with_bench_trailer
        self.assertEqual(w("x\n\nBench: 5\n\nCo-Authored-By: a\n", 7), "x\n\nBench: 7\n\nCo-Authored-By: a\n")
        self.assertEqual(w("x\n\nBench:  5 \n", 7), "x\n\nBench: 7\n")
        self.assertEqual(w("x\n\nCo-authored-by: a\n", 7), "x\n\nBench: 7\n\nCo-authored-by: a\n")
        self.assertEqual(w("x\n", 7), "x\n\nBench: 7\n")
        for msg in ("x", "x\n\nCo-Authored-By: a", "x\n\nBench: 1\n"):
            self.assertEqual(q.BENCH_RE.findall(w(msg, 9)), ["9"])


class Durability(unittest.TestCase):
    """State is saved after each merge/dispatch; errors in finish don't wedge."""

    CUR = {"branch": "cand", "mode": "stc", "test_id": "q-cand-1", "sha": "s", "base_sha": "b"}
    RES = {"test_id": "q-cand-1", "status": "H1", "final": True}

    def advance(self, state, result=None, start=None, finish=None):
        """One advance; fills self.saved (state snapshots), self.logs, self.cmds."""
        self.saved, self.logs, self.cmds = [], [], []

        def fake_start(item, st=None, priority=None):
            if item["mode"] == "speedup":
                self.cmds.append(("start", item["branch"]))
                return None
            return start(item) if start else None

        def fake_sh(*c, **k):
            self.cmds.append(c)
            return subprocess.CompletedProcess(c, 0, "", "")

        patches = [mock.patch.object(q, "load_state", return_value=state),
                   mock.patch.object(q, "save_state",
                                     lambda st, msg: self.saved.append(json.loads(json.dumps(st)))),
                   mock.patch.object(q, "find_result", return_value=None),
                   mock.patch.object(q, "background_lane"),
                   mock.patch.object(q, "read_log", return_value=[]),
                   mock.patch.object(q, "log", self.logs.append),
                   mock.patch.object(q, "start", fake_start),
                   mock.patch.object(q, "sh", fake_sh),
                   mock.patch.object(q, "BURST", {}),
                   mock.patch.dict(os.environ, {"RESULT": json.dumps(result) if result else ""})]
        if finish:
            patches.append(mock.patch.object(q, "finish", finish))
        for p in patches:
            p.start()
        try:
            q.advance()
        finally:
            for p in reversed(patches):
                p.stop()

    @staticmethod
    def merged(state, cur, res):
        state.setdefault("identical_merges", []).append("m1")

    def test_finish_error_requeues_instead_of_wedging(self):
        def boom(state, cur, res):
            raise RuntimeError("git push origin s:refs/heads/main failed")
        state = {"pending": [], "current": dict(self.CUR)}
        self.advance(state, self.RES, finish=boom)
        self.assertIsNone(self.saved[0]["current"])
        # Requeued at the front (and, with start() mocked, picked again at once).
        self.assertEqual(self.saved[0]["pending"],
                         [{"branch": "cand", "mode": "stc", "retried": True, "front": True}])
        self.assertEqual(self.logs[0]["outcome"], "retrying")
        self.assertIn("refs/heads/main failed", self.logs[0]["reason"])
        self.assertIn(("git", "push", "-q", "origin", ":refs/queue/q-cand-1"), self.cmds)
        self.assertIsNone(self.saved[0]["current"])
        # A second failure goes to a human rather than looping.
        state = {"pending": [], "current": dict(self.CUR, retried=True)}
        self.advance(state, self.RES, finish=boom)
        self.assertEqual((self.logs[0]["outcome"], self.saved[0]["pending"]), ("needs-human", []))

    def test_run_killed_after_merge_keeps_the_merge(self):
        def killed(item):
            raise KeyboardInterrupt  # stands in for the job timeout killing the run
        state = {"pending": [{"branch": "next", "mode": "stc"}], "current": dict(self.CUR)}
        with self.assertRaises(KeyboardInterrupt):
            self.advance(state, self.RES, start=killed, finish=self.merged)
        # Saved before the next item started: the merge is not lost, and the
        # item that was being started is still pending.
        last = self.saved[-1]
        self.assertIsNone(last["current"])
        self.assertEqual(last["identical_merges"], ["m1"])
        self.assertEqual(last["pending"], [{"branch": "next", "mode": "stc"}])

    def test_saves_after_finish_speedups_and_dispatch(self):
        state = {"pending": [{"branch": "s1", "mode": "speedup"}, {"branch": "next", "mode": "stc"}],
                 "current": dict(self.CUR)}
        self.advance(state, self.RES, finish=self.merged,
                     start=lambda item: {"test_id": "q-next-1", "sha": "x", "base_sha": "y"})
        after_finish, after_speedup, after_dispatch = self.saved[:3]
        self.assertIsNone(after_finish["current"])
        self.assertEqual(after_finish["identical_merges"], ["m1"])
        self.assertEqual([p["branch"] for p in after_speedup["pending"]], ["next"])
        self.assertEqual(after_dispatch["current"]["test_id"], "q-next-1")
        self.assertEqual(after_dispatch["pending"], [])

    def test_speedups_capped_per_run(self):
        n = q.MAX_SPEEDUPS_PER_RUN
        state = {"pending": [{"branch": f"s{i}", "mode": "speedup"} for i in range(n + 1)],
                 "current": dict(self.CUR)}
        self.advance(state)
        self.assertEqual([c for c in self.cmds if c[0] == "start"], [("start", f"s{i}") for i in range(n)])
        self.assertEqual(state["pending"], [{"branch": f"s{n}", "mode": "speedup"}])
        self.assertIn(("gh", "workflow", "run", "queue.yml", "--ref", "main"), self.cmds)


class Preempt(unittest.TestCase):
    def test_late_listed_wave_is_cancelled_on_a_later_run(self):
        listed, cmds = [[]], []

        def fake_sh(*c, **k):
            cmds.append(c)
            stdout = json.dumps(listed[0]) if c[:3] == ("gh", "run", "list") else ""
            return subprocess.CompletedProcess(c, 0, stdout, "")
        state = {"pending": [{"branch": "net", "mode": "net-gate", "front": True}],
                 "current": {"branch": "a", "mode": "stc", "test_id": "q-a-1", "sha": "s", "base_sha": "b"}}
        with mock.patch.object(q, "sh", fake_sh), mock.patch.object(q, "log"):
            q.preempt(state, 1)  # the wave is not listed yet
            self.assertFalse(any(c[:3] == ("gh", "run", "cancel") for c in cmds))
            self.assertEqual(state["cancel_sweep"][0]["prefix"], "SPRT q-a-1 ")
            listed[0] = [{"databaseId": 5, "status": "queued", "displayTitle": "SPRT q-a-1 wave 1"},
                         {"databaseId": 6, "status": "queued", "displayTitle": "SPRT q-a-10 wave 1"},
                         {"databaseId": 7, "status": "in_progress", "displayTitle": "SPRT q-b-1 wave 1"}]
            q.sweep_cancelled(state)
        self.assertEqual([c for c in cmds if c[:3] == ("gh", "run", "cancel")], [("gh", "run", "cancel", "5")])
        self.assertEqual(state["pending"][1]["branch"], "a")
        self.assertIsNone(state["current"])

    def test_sweep_entries_expire(self):
        old = (datetime.datetime.utcnow() - datetime.timedelta(minutes=1)).isoformat(timespec="seconds") + "Z"
        state = {"cancel_sweep": [{"prefix": "SPRT q-a-1 ", "until": old}]}
        with mock.patch.object(q, "sh") as sh:
            q.sweep_cancelled(state)
        sh.assert_not_called()
        self.assertNotIn("cancel_sweep", state)


class RunnerBudget(unittest.TestCase):
    def lane(self, others):
        cmds, logs = [], []
        with mock.patch.object(q, "out", runs_by_workflow((), others)), \
                mock.patch.object(q, "sh", lambda *c, **k: cmds.append(c)), \
                mock.patch.object(q, "log", logs.append), \
                mock.patch.dict(os.environ, {"FINISHED_RUN": ""}):
            state = {"next_seed": 20}
            q.background_lane(state)
        runs = [dict(a.split("=", 1) for a in c if "=" in a and not a.startswith("--")) for c in cmds
                if c[:4] == ("gh", "workflow", "run", "datagen.yml")]
        return state, runs, logs

    def test_other_ladder_shrinks_the_lane(self):
        _, runs, logs = self.lane({"ccrl-ladder.yml": [
            {"databaseId": 1, "status": "in_progress", "displayTitle": "CCRL ladder "}]})
        par = q.BACKGROUND_PARALLEL - q.OTHER_WORKFLOWS["ccrl-ladder.yml"]
        if par < 1:
            self.assertEqual(runs, [])
            return
        self.assertEqual(int(runs[0]["max_parallel"]), par)
        self.assertEqual(int(runs[0]["shards"]), q.BACKGROUND_SHARDS // q.BACKGROUND_PARALLEL * par)
        self.assertEqual(int(runs[0]["shards"]) % par, 0)
        self.assertIn("shrunk", logs[-1]["reason"])

    def test_queue_own_runs_do_not_count(self):
        _, runs, _ = self.lane({
            "sprt.yml": [{"databaseId": 1, "status": "in_progress", "displayTitle": "SPRT q-x-1 wave 2"}],
            "ccrl-ladder.yml": [{"databaseId": 2, "status": "in_progress",
                                 "displayTitle": "CCRL ladder q-ladder-1107-1200"}]})
        self.assertEqual((int(runs[0]["max_parallel"]), int(runs[0]["shards"])),
                         (q.BACKGROUND_PARALLEL, q.BACKGROUND_SHARDS))

    def test_spsa_or_sweep_skips_the_lane(self):
        for wf in ("spsa.yml", "train-sweep.yml"):
            state, runs, _ = self.lane({wf: [{"databaseId": 1, "status": "in_progress", "displayTitle": "x"}]})
            self.assertEqual((runs, state["next_seed"]), ([], 20))


def total_games(mode):
    """Most games an SPRT of this mode can play: each wave doubles
    rounds_per_shard up to max_rounds_per_shard (sprt.yml's report job), and
    every round is a 2-game pair."""
    rounds, total = mode["rounds_per_shard"], 0
    cap = mode.get("max_rounds_per_shard", rounds)
    for _ in range(mode["max_waves"]):
        total += rounds
        rounds = min(cap, rounds * 2)
    return 2 * mode["shards"] * total


class LiveSettings(unittest.TestCase):
    """The checked-in queue-settings.json and workflows agree with each other."""

    MIN_GAMES = {"stc": 15000, "nonreg": 15000, "ltc-gainer": 4000, "ltc-check": 4000, "net-gate": 1500}

    def setUp(self):
        with open(os.path.join(HERE, "queue-settings.json")) as fh:
            self.cfg = json.load(fh)

    def test_total_game_caps(self):
        self.assertEqual(total_games({"shards": 6, "rounds_per_shard": 8, "max_rounds_per_shard": 40,
                                      "max_waves": 4}), 2 * 6 * (8 + 16 + 32 + 40))
        for mode, minimum in self.MIN_GAMES.items():
            self.assertGreaterEqual(total_games(self.cfg["modes"][mode]), minimum, mode)

    def test_lanes_add_up(self):
        lanes = self.cfg["lanes"]
        self.assertEqual(lanes["background_shards"] % lanes["background_parallel"], 0)
        self.assertEqual(q.MAX_FOREGROUND_SHARDS + q.BACKGROUND_PARALLEL + 1, q.RUNNERS)
        self.assertLessEqual(self.cfg["burst"]["max_parallel"], q.MAX_FOREGROUND_SHARDS)
        for mode, s in self.cfg["modes"].items():
            self.assertLessEqual(s["shards"], q.MAX_FOREGROUND_SHARDS, mode)

    def test_bad_lanes_fail(self):
        base = {"modes": {"stc": {"shards": 1}}}
        good = {"background_parallel": 13, "background_shards": 26}
        q.validate_settings(dict(base, lanes=good))
        for lanes in ({"background_parallel": 13, "background_shards": 20}, {"background_parallel": 0},
                      {"background_parallel": 19}):
            with self.assertRaises(SystemExit):
                q.validate_settings(dict(base, lanes=lanes))
        with self.assertRaises(SystemExit):  # 20 - 1 - 13 = 6 foreground runners
            q.validate_settings({"modes": {"stc": {"shards": 7}}, "lanes": good})
        with self.assertRaises(SystemExit):
            q.validate_settings(dict(base, lanes=good, ladder={"max_parallel": 7}))
        q.validate_settings(dict(base, lanes=good, ladder={"max_parallel": "6"}))

    def test_ladder_gets_the_foreground_width(self):
        with open(os.path.join(HERE, "..", ".github", "workflows", "ccrl-ladder.yml")) as fh:
            wf = fh.read()
        self.assertIn("max-parallel: ${{ fromJSON(inputs.max_parallel) }}", wf)
        self.assertIn("      max_parallel:\n", wf)
        cmds = []
        with mock.patch.object(q, "sh", lambda *c, **k: cmds.append(c)), \
                mock.patch.object(q, "out", lambda *c, **k: "mainsha"), \
                mock.patch.object(q, "log"):
            q.start({"branch": "main", "mode": "ladder"})
        self.assertIn(f"max_parallel={q.MAX_FOREGROUND_SHARDS}", cmds[0])

    def test_ci_sprt_defaults_to_the_foreground_width(self):
        spec = importlib.util.spec_from_file_location("ci_mod", os.path.join(HERE, "ci.py"))
        ci = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ci)
        for name, prof in ci.PROFILES.items():
            self.assertEqual(prof["shards"], q.MAX_FOREGROUND_SHARDS, name)


class InboxRemove(unittest.TestCase):
    """The queue.py remove inbox path drops waiting items, but never the in-flight one."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="queue-remove-test-")
        self.state = os.path.join(self.tmp, "state")
        os.makedirs(os.path.join(self.state, "inbox"))
        self.logs = []
        self.patches = [
            mock.patch.object(q, "STATE", self.state),
            mock.patch.object(q, "sh", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")),
            mock.patch.object(q, "log", self.logs.append),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, pending, request):
        with open(os.path.join(self.state, "queue.json"), "w") as fh:
            json.dump({"pending": pending, "current": {"branch": "live", "mode": "stc"}}, fh)
        with open(os.path.join(self.state, "inbox", "0001.json"), "w") as fh:
            json.dump(request, fh)

    def test_remove_branch_and_mode(self):
        pending = [
            {"branch": "a", "mode": "stc"},
            {"branch": "a", "mode": "nonreg"},
            {"branch": "b", "mode": "stc"},
        ]
        self.write(pending, {"remove": "a:nonreg"})
        state = q.load_state()
        self.assertEqual(state["pending"],
                         [{"branch": "a", "mode": "stc"}, {"branch": "b", "mode": "stc"}])
        self.assertEqual(state["current"], {"branch": "live", "mode": "stc"})
        self.assertEqual([(r["branch"], r["mode"], r["outcome"]) for r in self.logs],
                         [("a", "nonreg", "removed")])
        self.assertFalse(os.path.exists(os.path.join(self.state, "inbox", "0001.json")))

    def test_remove_branch_all_modes(self):
        pending = [
            {"branch": "a", "mode": "stc"},
            {"branch": "a", "mode": "nonreg"},
            {"branch": "b", "mode": "stc"},
        ]
        self.write(pending, {"remove": "a"})
        state = q.load_state()
        self.assertEqual(state["pending"], [{"branch": "b", "mode": "stc"}])
        self.assertEqual(len(self.logs), 2)
        self.assertTrue(all(r["outcome"] == "removed" for r in self.logs))

    def test_missing_item_is_a_noop(self):
        pending = [{"branch": "a", "mode": "stc"}]
        self.write(pending, {"remove": "missing:stc"})
        state = q.load_state()
        self.assertEqual(state["pending"], pending)
        self.assertEqual(self.logs, [])
        self.assertFalse(os.path.exists(os.path.join(self.state, "inbox", "0001.json")))


if __name__ == "__main__":
    unittest.main()
