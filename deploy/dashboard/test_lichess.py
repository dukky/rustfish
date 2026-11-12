#!/usr/bin/env python3
"""Stdlib tests for the Lichess card: API parsing and fail-soft /api/lichess.

No network: every upstream fetch goes through a fake urlopen.
Run: python3 deploy/dashboard/test_lichess.py
"""
import io
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.pop("DATA_DIR", None)
import server  # noqa: E402

USER = {"id": "rustfishbot", "username": "RustfishBot", "title": "BOT", "url": "https://lichess.org/@/RustfishBot",
        "createdAt": 1761692400000, "seenAt": 1762900000000,
        "perfs": {"bullet": {"games": 40, "rating": 2401, "rd": 60, "prog": 12},
                  "blitz": {"games": 85, "rating": 2550, "rd": 45, "prog": -8},
                  "rapid": {"games": 0, "rating": 1500, "rd": 500, "prov": True},
                  "puzzle": {"games": 3, "rating": 1600}},
        "count": {"all": 125, "rated": 125, "win": 80, "loss": 30, "draw": 15}}
STATUS = [{"id": "rustfishbot", "name": "RustfishBot", "online": True, "playing": True, "playingId": "abcd1234"}]
GAMES = "\n".join(json.dumps(g) for g in [
    {"id": "g1", "rated": True, "speed": "blitz", "status": "mate", "winner": "white", "lastMoveAt": 1762800000000,
     "clock": {"initial": 180, "increment": 2},
     "players": {"white": {"user": {"name": "RustfishBot", "id": "rustfishbot", "title": "BOT"}, "rating": 2550, "ratingDiff": 6},
                 "black": {"user": {"name": "OtherBot", "id": "otherbot", "title": "BOT"}, "rating": 2600, "ratingDiff": -6}}},
    {"id": "g2", "rated": True, "speed": "bullet", "status": "outoftime", "winner": "white", "lastMoveAt": 1762900000000,
     "clock": {"initial": 30, "increment": 0},
     "players": {"white": {"user": {"name": "Human", "id": "human"}, "rating": 2000},
                 "black": {"user": {"name": "RustfishBot", "id": "rustfishbot"}, "rating": 2401}}},
    {"id": "g3", "rated": False, "speed": "rapid", "status": "stalemate", "lastMoveAt": 1762700000000,
     "players": {"white": {"aiLevel": 8}, "black": {"user": {"name": "RustfishBot", "id": "rustfishbot"}}}},
]) + "\n"
HISTORY = [{"name": "Bullet", "points": [[2026, 9, 29, 1500], [2026, 10, 10, 2401]]},
           {"name": "Blitz", "points": [[2026, 10, d, 2000 + d] for d in range(1, 31)] * 3},
           {"name": "Puzzles", "points": [[2026, 10, 1, 1600]]}, {"name": "Rapid", "points": []}]


class ParseTest(unittest.TestCase):
    def test_user(self):
        u = server.parse_lichess_user(USER)
        self.assertEqual([p["speed"] for p in u["perfs"]], ["bullet", "blitz"])  # unplayed rapid, puzzles dropped
        self.assertEqual(u["perfs"][1], {"speed": "blitz", "rating": 2550, "rd": 45, "prog": -8, "prov": False, "games": 85})
        self.assertEqual(u["count"]["all"], 125)
        self.assertTrue(u["seen_at"].startswith("2025-11-11"))
        with self.assertRaises(ValueError):
            server.parse_lichess_user({"error": "Not found"})

    def test_games(self):
        g = server.parse_lichess_games(server.jsonl(GAMES), "RustfishBot")
        self.assertEqual([x["id"] for x in g], ["g2", "g1", "g3"])  # newest first
        g2, g1, g3 = g
        self.assertEqual((g1["color"], g1["result"], g1["opponent"], g1["opponent_rating"], g1["rating_diff"]),
                         ("white", "win", "OtherBot", 2600, 6))
        self.assertEqual((g1["clock"], g1["url"], g1["opponent_title"]), ("3+2", "https://lichess.org/g1", "BOT"))
        self.assertEqual((g2["color"], g2["result"], g2["clock"], g2["url"]),
                         ("black", "loss", "0.5+0", "https://lichess.org/g2/black"))
        self.assertEqual((g3["result"], g3["opponent"], g3["rated"], g3["clock"]), ("draw", "Stockfish level 8", False, None))

    def test_history(self):
        h = server.parse_lichess_history(HISTORY)
        self.assertEqual(sorted(h), ["blitz", "bullet"])
        self.assertEqual(h["bullet"], [["2026-10-29", 1500], ["2026-11-10", 2401]])  # months are 0-based upstream
        self.assertEqual(len(h["blitz"]), server.LICHESS_HISTORY_POINTS)

    def test_online(self):
        self.assertEqual(server.parse_lichess_online(STATUS), {"online": True, "playing": True, "playing_id": "abcd1234"})
        self.assertEqual(server.parse_lichess_online([]), {"online": False, "playing": False, "playing_id": None})


class FakeResp(io.BytesIO):
    def __init__(self, body):
        super().__init__(body.encode())
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def fake_urlopen(routes, calls):
    def urlopen(req, timeout=None):
        calls.append((req.full_url, dict(req.header_items())))
        for key, body in routes.items():
            if key in req.full_url:
                if isinstance(body, Exception):
                    raise body
                return FakeResp(body)
        raise urllib.error.URLError("no route")
    return urlopen


ROUTES = {"/api/users/status": json.dumps(STATUS), "/api/games/user/": GAMES,
          "/rating-history": json.dumps(HISTORY), "/api/user/rustfishbot": json.dumps(USER)}


class BuildTest(unittest.TestCase):
    def build(self, routes, cache=None):
        calls = []
        cache = cache or server.Cache()
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen(routes, calls)):
            return server.build_lichess("rustfishbot", cache), calls, cache

    def test_ok_and_cached(self):
        r, calls, cache = self.build(ROUTES)
        self.assertTrue(r["available"])
        self.assertTrue(r["online"] and r["playing"])
        self.assertEqual(r["profile"]["playing_url"], "https://lichess.org/abcd1234")
        self.assertEqual(len(r["games"]), 3)
        self.assertEqual(len(calls), 4)
        hdr = {k.lower(): v for k, v in calls[0][1].items()}
        self.assertIn("rustfish-dashboard", hdr["user-agent"])
        games_hdr = {k.lower(): v for u, h in calls if "/api/games/user/" in u for k, v in h.items()}
        self.assertEqual(games_hdr["accept"], "application/x-ndjson")
        self.assertIn("max=10", next(u for u, _ in calls if "/api/games/user/" in u))
        # A second build inside the TTL makes no requests.
        _, calls2, _ = self.build(ROUTES, cache)
        self.assertEqual(calls2, [])

    def test_unreachable(self):
        r, calls, cache = self.build({"lichess.org": urllib.error.URLError("down")})
        self.assertFalse(r["available"])
        self.assertIn("down", r["error"])
        self.assertEqual(len(calls), 1)  # stops after the profile fails
        # The failure is remembered: no retry storm while Lichess is down.
        r2, calls2, _ = self.build(ROUTES, cache)
        self.assertFalse(r2["available"])
        self.assertEqual(calls2, [])

    def test_rate_limited_and_partial(self):
        err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
        r, _, _ = self.build(dict(ROUTES, **{"/api/games/user/": err}))
        self.assertTrue(r["available"])  # profile still shown
        self.assertIsNone(r["games"])
        self.assertIn("429", r["sources"]["games"]["error"])
        r, _, _ = self.build({"/api/user/rustfishbot": "not json"})
        self.assertFalse(r["available"])

    def test_not_configured(self):
        calls = []
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen(ROUTES, calls)):
            r = server.build_lichess("", server.Cache())
        self.assertEqual((r["available"], r["configured"]), (False, False))
        self.assertEqual(calls, [])


class EndpointTest(unittest.TestCase):
    """/api/lichess answers 200 with available False when Lichess is down, and
    /api/status does not depend on it."""

    def test_endpoint_fails_soft(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        real = urllib.request.urlopen
        down = fake_urlopen({"lichess.org": urllib.error.URLError("down")}, [])

        def urlopen(req, timeout=None):  # local requests pass through, upstream is down
            url = req if isinstance(req, str) else req.full_url
            return real(req, timeout=timeout) if url.startswith(base) else down(req, timeout)
        try:
            with mock.patch.object(server, "LICHESS_USER", "rustfishbot"), \
                    mock.patch.object(server, "CACHE", server.Cache()), \
                    mock.patch.object(urllib.request, "urlopen", urlopen):
                with real(base + "/api/lichess", timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    d = json.loads(resp.read())
            self.assertFalse(d["available"])
            self.assertTrue(d["configured"])
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main()
