#!/usr/bin/env python3
"""Unit tests for scripts/queue.py (stdlib unittest only).

    python3 scripts/test_queue.py

The fast-forward / replay tests run real git against a throwaway bare
"origin" and clone in a temp dir; cargo, gh and the engine are mocked, so
nothing touches GitHub or builds anything.
"""
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

        def fake_sh(*cmd, cwd=q.REPO, check=True):
            self.calls.append(cmd)
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
        cfg = {"modes": {"stc": {"shards": 16}, "ltc-gainer": {"shards": 19}}}
        with self.assertRaises(SystemExit) as ctx:
            q.validate_settings(cfg)
        self.assertIn("ltc-gainer (shards=19)", str(ctx.exception))
        self.assertNotIn("stc (", str(ctx.exception))

    def test_missing_shards_fails(self):
        with self.assertRaises(SystemExit):
            q.validate_settings({"modes": {"stc": {"tc": "10+0.1"}}})


class BackgroundLane(unittest.TestCase):
    def lane(self, runs, finished=""):
        cmds = []
        with mock.patch.object(q, "out", lambda *c, **k: json.dumps(runs) if c[:3] == ("gh", "run", "list") else "sha"), \
                mock.patch.object(q, "sh", lambda *c, **k: cmds.append(c)), \
                mock.patch.object(q, "log"), \
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


if __name__ == "__main__":
    unittest.main()
