#!/usr/bin/env python3
"""Stdlib tests for /api/status: the live CI runner view (Actions jobs grouped
by workflow), the net generation table (1024-wide kb8m rows, rows with no
training record) and build_status wired end to end. No network: every
upstream fetch goes through a fake urlopen.

Run: python3 deploy/dashboard/test_status.py
"""
import io
import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.pop("DATA_DIR", None)
import server  # noqa: E402


def run(i, wf, status, title, created="2026-11-12T22:00:00Z"):
    return {"id": i, "path": f".github/workflows/{wf}", "status": status, "display_title": title,
            "name": title, "html_url": f"https://github.com/x/actions/runs/{i}", "created_at": created}


def jobs(*statuses):
    return [{"name": f"shard {k}", "status": s} for k, s in enumerate(statuses)]


SPRT = run(1, "sprt.yml", "in_progress", "SPRT q-feat-x-1112-2305 wave 3", "2026-11-12T23:22:13Z")
SPSA = run(2, "spsa.yml", "in_progress", "SPSA tune/spsa-params.json @ main round 1", "2026-11-12T22:41:08Z")
QUEUE = run(3, "queue.yml", "in_progress", "Merge queue", "2026-11-12T23:30:00Z")
CI = run(4, "ci.yml", "queued", "Plan: tidy", "2026-11-12T23:31:00Z")
LADDER = run(5, "ccrl-ladder.yml", "queued", "CCRL ladder q-ladder-1", "2026-11-12T23:00:00Z")
DATAGEN = run(6, "datagen.yml", "in_progress", "Datagen main seed 200", "2026-11-12T21:00:00Z")
DONE = run(7, "sprt.yml", "completed", "SPRT q-feat-x-1112-2305 wave 2", "2026-11-12T23:11:46Z")


class RunnerUsageTest(unittest.TestCase):
    def test_grouping(self):
        u = server.build_runner_usage(
            [SPRT, SPSA, QUEUE, CI, LADDER, DATAGEN, DONE],
            {1: [{"name": "plan", "status": "completed"}] + jobs(*["in_progress"] * 8, "queued", "queued"),
             2: jobs(*["in_progress"] * 8), 5: jobs("queued", "queued"), 6: None})
        g = {x["key"]: x for x in u["groups"]}
        self.assertEqual([x["key"] for x in u["groups"]], ["sprt", "spsa", "ladder", "other"])
        self.assertEqual((g["sprt"]["running"], g["sprt"]["queued"]), (9, 2))  # 8 shards + the queue job
        self.assertEqual((g["spsa"]["running"], g["ladder"]["running"], g["ladder"]["queued"]), (8, 0, 2))
        self.assertEqual((g["other"]["running"], g["other"]["queued"]), (0, 1))  # ci.yml queued; datagen unknown
        self.assertEqual((u["in_use"], u["idle"], u["queued"], u["cap"]), (17, 3, 5, 20))
        self.assertEqual(u["unknown_runs"], 1)
        dg = next(r for r in g["other"]["runs"] if r["workflow"] == "datagen.yml")
        self.assertIsNone(dg["running"])
        self.assertNotIn(7, [r["id"] for x in u["groups"] for r in x["runs"]])  # completed runs ignored

    def test_finished_jobs_dropped(self):
        u = server.build_runner_usage([SPSA], {2: [{"name": "shard 0", "status": "completed"}]})
        self.assertEqual(u["in_use"], 0)
        self.assertEqual(sum(len(x["runs"]) for x in u["groups"]), 0)

    def test_jobs_ttl_shares_budget(self):
        self.assertEqual(server.jobs_ttl(0), server.TTL_JOBS_MIN)
        self.assertEqual(server.jobs_ttl(1), server.TTL_JOBS_MIN)
        self.assertEqual(server.jobs_ttl(4), 3600 * 4 // server.JOBS_PER_HOUR)
        for n in range(1, 10):  # never more than JOBS_PER_HOUR job fetches an hour
            self.assertLessEqual(n * 3600 / server.jobs_ttl(n), max(server.JOBS_PER_HOUR, 3600 / server.TTL_JOBS_MIN))


# Real rows from net/experiments.jsonl and the queue log (trimmed).
EXPERIMENTS = [
    {"id": "g7kb8m", "date": "2026-11-08T15:55:57Z", "hidden": 512, "buckets": 8, "epochs": 4,
     "positions_per_epoch": 1715621881, "val_loss": 0.026328, "val_loss_quantized": 0.026344,
     "baseline": "g6kb8m", "branch": "nnue-g7kb8m", "king_buckets": "kb8m",
     "notes": "Gen 7 (pool7 ~1.8B): g7 -> g7m1 -> kb8m warm start. Best-scale val.bin 0.025821 vs 0.025832, "
              "val17x2 0.024812 vs 0.024920. Embedded unattended by chain14."},
    {"id": "g8h1024kb8m", "date": "2026-11-12T15:31:15Z", "hidden": 1024, "buckets": 8, "epochs": 4,
     "positions_per_epoch": 3201732989, "val_loss": 0.026636, "val_loss_quantized": 0.026637, "baseline": "main",
     "branch": "nnue-g8h1024kb8m", "king_buckets": "kb8m", "train_data": "pool8.bin", "init": "g8h1024m1.bin",
     "notes": "Gen 8 on the desktop 1080 Ti (CUDA trainer): pool8 (3.20B) ... Best-scale val.bin / val17x2: "
              "0.025574 / 0.024364 vs main g7h1024kb8m 0.025631 / 0.024587."},
    {"id": "g4kb8m", "date": "2026-11-07T12:24:23Z", "hidden": 512, "buckets": 8, "branch": "nnue-g4kb8m",
     "positions_per_epoch": 729869027, "king_buckets": "kb8m",
     "notes": "Best-scale loss: val.bin 0.025943 (kb4 0.026056, ft 0.026086, g4 0.026110); val17x2 0.025316 (kb4 0.025337)."},
    {"id": "h-kbf", "date": "2026-11-06T20:00:00Z", "hidden": 512, "buckets": 8, "branch": "local-h-kbf",
     "positions_per_epoch": 271403224, "notes": "learning curve kb8m+factor at 50% of games (local only)"},
]
QLOG = [
    {"branch": "nnue-g4kb8m", "mode": "stc", "outcome": "merged", "elo": 10.99, "elo_ci95": [4.1, 17.9],
     "games": 5000, "logged_at": "2026-11-07T13:00:00Z", "verdict": "H1"},
    {"branch": "nnue-g7kb8m", "mode": "net-gate", "outcome": "merged", "elo": 30.58, "elo_ci95": [13.12, 48.21],
     "games": 672, "logged_at": "2026-11-08T16:29:29Z", "verdict": "H1"},
    {"branch": "nnue-g7h1024kb8m", "mode": "stc", "outcome": "dispatched", "logged_at": "2026-11-12T13:38:01Z",
     "test_id": "q-nnue-g7h1024kb8m-1112-1337"},
    {"branch": "nnue-g7h1024kb8m", "mode": "stc", "outcome": "merged", "elo": 11.51, "elo_ci95": [5.51, 17.51],
     "games": 4560, "logged_at": "2026-11-12T14:53:42Z", "verdict": "H1"},
    {"branch": "nnue-g8h1024kb8m", "mode": "net-gate", "outcome": "merged", "elo": 37.18, "elo_ci95": [23.58, 50.89],
     "games": 1060, "logged_at": "2026-11-12T17:58:26Z", "verdict": "H1"},
    {"branch": "nnue-g8h1024kb8m-ep", "mode": "stc", "outcome": "rejected", "elo": -23.2, "elo_ci95": [-35.23, -11.22],
     "games": 960, "logged_at": "2026-11-12T23:03:41Z", "verdict": "H0"},
]


class GenerationsTest(unittest.TestCase):
    def rows(self):
        gates = server.build_net_gates(QLOG, {}, EXPERIMENTS, set())
        return {r["name"]: r for r in server.build_generations(gates["v2"])}

    def test_1024_kb8m_with_record(self):
        r = self.rows()["g8h1024kb8m"]
        self.assertEqual((r["gen"], r["hidden"], r["buckets"], r["king_buckets"], r["threats"]), (8, 1024, 8, "kb8m", None))
        self.assertEqual((r["pool"], r["train_data"]), (3201732989, "pool8.bin"))
        self.assertEqual((r["best_scale_val"], r["best_scale_val17x2"]), (0.025574, 0.024364))
        self.assertEqual((r["status"], r["mode"], r["elo"], r["games"]), ("merged", "net-gate", 37.18, 1060))
        self.assertTrue(r["training_record"])
        self.assertEqual(r["inferred"], [])
        self.assertTrue(r["current"])  # the latest merged net is on main

    def test_missing_records_inferred_from_name(self):
        rows = self.rows()
        r = rows["g7h1024kb8m"]
        self.assertFalse(r["training_record"])
        self.assertEqual((r["gen"], r["hidden"], r["king_buckets"], r["status"], r["elo"]), (7, 1024, "kb8m", "merged", 11.51))
        self.assertEqual(sorted(r["inferred"]), ["hidden", "king_buckets"])
        self.assertIsNone(r["pool"])
        self.assertFalse(r["current"])
        ep = rows["g8h1024kb8m-ep"]
        self.assertEqual((ep["gen"], ep["hidden"], ep["king_buckets"], ep["threats"], ep["status"]),
                         (8, 1024, "kb8m", "ep", "rejected"))

    def test_best_scale_note_formats(self):
        rows = self.rows()
        self.assertEqual((rows["g7kb8m"]["best_scale_val"], rows["g7kb8m"]["best_scale_val17x2"]), (0.025821, 0.024812))
        self.assertEqual((rows["g4kb8m"]["best_scale_val"], rows["g4kb8m"]["best_scale_val17x2"]), (0.025943, 0.025316))
        self.assertEqual(server.parse_best_scale("best-scale loss ~ g3"), (None, None))
        self.assertEqual(server.parse_best_scale(None), (None, None))

    def test_explicit_fields_win_and_local_rows(self):
        exp = [dict(EXPERIMENTS[1], best_scale_val=0.1, best_scale_val17x2=0.2, threats="ep")]
        r = server.build_generations(server.build_net_gates([], {}, exp, set())["v2"])[0]
        self.assertEqual((r["best_scale_val"], r["best_scale_val17x2"], r["threats"], r["status"]), (0.1, 0.2, "ep", "not gated"))
        self.assertEqual(r["inferred"], [])
        local = self.rows()["h-kbf"]
        self.assertEqual((local["gen"], local["local"], local["status"]), (None, True, "local only"))

    def test_newest_first(self):
        gates = server.build_net_gates(QLOG, {}, EXPERIMENTS, set())
        names = [r["name"] for r in server.build_generations(gates["v2"])]
        self.assertEqual(names[:3], ["g8h1024kb8m-ep", "g8h1024kb8m", "g7h1024kb8m"])


class FakeResp(io.BytesIO):
    def __init__(self, body):
        super().__init__(body.encode())
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class BuildStatusTest(unittest.TestCase):
    """build_status end to end on canned upstream data."""

    def routes(self):
        api = server.API
        return {
            f"{api}/actions/runs?status=in_progress": {"workflow_runs": [SPRT, SPSA]},
            f"{api}/actions/runs?per_page=30": {"workflow_runs": [CI, DONE, SPRT]},  # SPSA too old for this page
            f"{api}/actions/runs/1/jobs": {"jobs": jobs(*["in_progress"] * 10)},
            f"{api}/actions/runs/2/jobs": {"jobs": jobs(*["in_progress"] * 8)},
            f"{api}/commits": [],
            "/queue-state/queue.json": {"pending": [], "current": {
                "branch": "feat-x", "mode": "stc", "test_id": "q-feat-x-1112-2305", "sha": "b" * 40,
                "base_sha": "a" * 40, "started": "2026-11-12T23:05:31Z"}},
            "/queue-state/log.jsonl": "\n".join(json.dumps(r) for r in QLOG),
            "/net/experiments.jsonl": "\n".join(json.dumps(r) for r in EXPERIMENTS),
            "/scripts/queue-settings.json": {"modes": {"stc": {"tc": "10+0.1", "shards": 10, "rounds_per_shard": 16,
                                                               "max_rounds_per_shard": 116}},
                                             "lanes": {"background_parallel": 9}},
        }

    def build(self):
        calls = []
        routes = self.routes()

        def urlopen(req, timeout=None):
            url = req.full_url
            calls.append(url)
            for key, body in routes.items():
                if key in url:
                    return FakeResp(body if isinstance(body, str) else json.dumps(body))
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        with mock.patch.object(server, "CACHE", server.Cache()), \
                mock.patch.object(urllib.request, "urlopen", urlopen):
            s = server.build_status()
            n_first = len(calls)
            server.build_status()  # inside every TTL: no new requests
        return s, calls, n_first

    def test_runner_usage_from_jobs(self):
        s, calls, n_first = self.build()
        u = s["runner_usage"]
        g = {x["key"]: x for x in u["groups"]}
        self.assertEqual((g["sprt"]["running"], g["spsa"]["running"], g["other"]["queued"]), (10, 8, 1))
        self.assertEqual((u["in_use"], u["idle"], u["test_shards"]), (18, 2, 10))
        self.assertNotIn("background", s)
        self.assertNotIn("datagen", s)
        # One job list per live multi-job run, none for the single-job ci.yml run.
        self.assertEqual(sorted(c for c in calls if "/jobs" in c),
                         [f"{server.API}/actions/runs/1/jobs?per_page=100", f"{server.API}/actions/runs/2/jobs?per_page=100"])
        self.assertEqual(len(calls), n_first)
        cur = s["queue"]["current"]
        self.assertEqual((cur["wave"], cur["shards_total"], cur["shards_running"]), (3, 10, 10))

    def test_generations_in_status(self):
        s, _, _ = self.build()
        gens = {r["name"]: r for r in s["generations"]}
        self.assertEqual(gens["g8h1024kb8m"]["hidden"], 1024)
        self.assertTrue(gens["g8h1024kb8m"]["current"])
        self.assertIn("g7h1024kb8m", gens)


if __name__ == "__main__":
    unittest.main()
