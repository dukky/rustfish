#!/usr/bin/env python3
"""Unit tests for the scale-matching parts of scripts/netloop.py (stdlib unittest).

    python3 scripts/test_netloop.py

The rescale test runs the real nnue-trainer (tools/nnue-trainer, release
build; skipped if it is not built) on a tiny synthetic net; the auto-screen
test mocks the trainer, git and queue calls.
"""
import argparse
import importlib.util
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("netloop", os.path.join(HERE, "netloop.py"))
nl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nl)

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
    "8/8/4k3/8/2Q5/8/4K3/8 b - - 0 1",
    "4k3/8/8/8/8/8/8/4K2R b - - 0 1",
]


def tiny_net(h, out_mult, seed=1):
    """(768 -> h) x 2 -> 1 i16 net; output layer (weights + bias) times out_mult."""
    rng = random.Random(seed)
    ft = [rng.randint(-40, 40) for _ in range(768 * h)] + [rng.randint(0, 60) for _ in range(h)]
    out = [round(rng.randint(-60, 60) * out_mult) for _ in range(2 * h)] + [round(500 * out_mult)]
    return struct.pack(f"<{len(ft) + len(out)}h", *ft, *out)


class RescaleTo(unittest.TestCase):
    def test_matches_a_reference_at_half_scale(self):
        t = Path(nl.DEFAULT_TRAINER)
        if not t.exists():
            self.skipTest(f"{t} not built")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cand.bin").write_bytes(tiny_net(16, 1.0))
            (d / "ref.bin").write_bytes(tiny_net(16, 0.5))
            text = "".join(f"{f} | 0 | 0.5\n" for f in FENS * 4)
            subprocess.run([str(t), "convert", "-", str(d / "val.bin")], input=text, text=True,
                           check=True, capture_output=True)
            res = nl.rescale_to(t, d / "cand.bin", d / "out.bin", buckets=1, layout="none",
                                ref=d / "ref.bin", ref_buckets=1, ref_layout="none", val=d / "val.bin")
            self.assertAlmostEqual(res["factor"], 0.5, delta=0.03, msg=res["text"])
            self.assertEqual(res["clipped"], 0)
            self.assertIsNotNone(res["achieved"])
            self.assertEqual((d / "out.bin").stat().st_size, (d / "cand.bin").stat().st_size)


class AutoScreenScale(unittest.TestCase):
    def make(self, d, screen_scale):
        args = argparse.Namespace(data_dir=str(d), trainer="/bin/false", dry_run=False, screen_scale=screen_scale,
                                  scale_val="val.bin", queue_front=False, no_queue=True, expect_cap=10,
                                  epochs=3, lr="2e-4", lr_end="1e-5", eval_threads=1, kb_trainer="/bin/false")
        return nl.Auto(args)

    def run_screen(self, screen_scale):
        with tempfile.TemporaryDirectory() as d:
            auto = self.make(Path(d), screen_scale)
            raw = auto.dir / "auto-1.bin"
            raw.write_bytes(b"x")
            main = {"path": str(auto.dir / "main.bin"), "blob": "0123456789abcdef", "hidden": 512,
                    "buckets": 8, "layout": "none"}
            evals, shipped = [], []

            def fake_rescale(trainer_bin, net, out, **kw):
                Path(out).write_bytes(b"s")
                return {"factor": 0.92, "achieved": 0.9201, "clipped": 0, "text": "ok"}

            def fake_eval(net, val, main_):
                evals.append(Path(net).name)
                return {"main.bin": 0.0262, "auto-1-s.bin": 0.0261, "auto-1.bin": 0.0263}[Path(net).name]

            def fake_sh(cmd, **kw):
                cmd = [str(c) for c in cmd]
                if "embed" in cmd:
                    shipped.append(Path(cmd[cmd.index("--net") + 1]).name)
                return ""

            led = {"last_attempt_seq": 1}
            with mock.patch.object(nl, "rescale_to", side_effect=fake_rescale) as rs, \
                    mock.patch.object(auto, "eval_loss", side_effect=fake_eval), \
                    mock.patch.object(auto, "sh", side_effect=fake_sh), \
                    mock.patch.object(auto, "save_ledger"), \
                    mock.patch.object(auto, "queue_has_expect", return_value=False):
                auto.screen_and_ship(led, main, "auto-1", raw, {"name": "auto-1", "new_positions": 1,
                                                                 "replay_positions": 1})
            return rs, evals, shipped

    def test_match_screens_and_ships_the_scaled_net(self):
        rs, evals, shipped = self.run_screen("match")
        self.assertEqual(rs.call_count, 1)
        self.assertTrue(str(rs.call_args.kwargs["ref"]).endswith("main.bin"))
        self.assertIn("auto-1-s.bin", evals)
        self.assertEqual(shipped, ["auto-1-s.bin"], "passes on scaled loss, ships the scaled net")

    def test_off_is_unchanged(self):
        rs, evals, shipped = self.run_screen("off")
        self.assertEqual(rs.call_count, 0)
        self.assertNotIn("auto-1-s.bin", evals)
        self.assertEqual(shipped, [], "raw net is worse than main here, so nothing ships")


class Defaults(unittest.TestCase):
    def parse(self, argv):
        got = {}
        with mock.patch.object(sys, "argv", ["netloop.py", *argv]), \
                mock.patch.object(nl, "cmd_auto", side_effect=lambda a: got.update(vars(a))), \
                mock.patch.object(nl, "cmd_embed", side_effect=lambda a: got.update(vars(a))):
            nl.main()
        return got

    def test_scale_options_default_off(self):
        self.assertEqual(self.parse(["auto"])["screen_scale"], "off")
        e = self.parse(["embed", "--name", "x", "--net", "n.bin", "--hidden", "512", "--buckets", "8"])
        self.assertIsNone(e["match_scale"])
        self.assertEqual(e["match_norm"], "slope")


if __name__ == "__main__":
    unittest.main()
