#!/usr/bin/env python3
"""Unit tests for scripts/netloop.py (stdlib unittest): scale matching and the
experiments-row writer.

    python3 scripts/test_netloop.py

The rescale test runs the real nnue-trainer (tools/nnue-trainer, release
build; skipped if it is not built) on a tiny synthetic net; the auto-screen
test mocks the trainer, git and queue calls; the publish tests push to a
throwaway bare repo in a temp dir.
"""
import argparse
import importlib.util
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
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
                    self.embed_cmd = cmd
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

    def test_embed_records_by_default(self):
        self.run_screen("match")
        cmd = self.embed_cmd
        self.assertEqual(cmd[cmd.index("--baseline") + 1], "main@0123456")
        self.assertNotIn("--no-record", cmd)
        self.assertNotIn("--best-scale", cmd)
        self.assertIn("--trainer", cmd)

    def test_auto_record_flags(self):
        with tempfile.TemporaryDirectory() as d:
            auto = self.make(Path(d), "off")
            auto.a.no_record = True
            self.assertEqual(auto.record_flags(), ["--no-record"])
            auto.a.no_record, auto.a.best_scale = False, True
            self.assertIn("--best-scale", auto.record_flags())


LOG = """# /x/nnue-trainer-metal-kb train --data /d/auto/train-4.bin --val /d/val.bin --out /d/a.bin --init /d/main-b195.bin
train H=512 buckets=8 king-buckets=kb8m (KB=8) factorised params=3154440 epochs=3 x 242145678 batch=16384 lr 2e-4
start val loss 0.026169 mean|eval| 365.6cp
epoch 1 lr 1.53e-4 train 0.025447 val 0.026486 (quantized 0.026491) mean|eval| 390.1cp 1268583 pos/s -> a.e1.bin
epoch 2 lr 1.00e-5 train 0.025082 val 0.026456 (quantized 0.026475) mean|eval| 392.9cp 1259783 pos/s -> a.e2.bin
wrote a.bin (6308880 bytes)
"""


class Rows(unittest.TestCase):
    def test_parse_and_row(self):
        p = nl.parse_log(LOG)
        row = nl.experiment_row("auto-4", p, baseline="main@abc", branch="nnue-auto-4", notes="n")
        self.assertEqual((row["hidden"], row["buckets"], row["epochs"], row["positions_per_epoch"]),
                         (512, 8, 3, 242145678))
        self.assertEqual((row["val_loss"], row["val_loss_quantized"]), (0.026456, 0.026475))
        self.assertEqual(row["king_buckets"], "kb8m")
        self.assertEqual(row["mean_abs_eval"], 392.9)
        self.assertEqual((row["train_data"], row["init"]), ("train-4.bin", "main-b195.bin"))
        self.assertIsNone(row["sprt"])
        old = nl.parse_log("epoch 1 lr 1e-3 train 0.03 val 0.031 (quantized 0.0311)\n")
        self.assertNotIn("mean_abs_eval", nl.experiment_row("x", old, baseline="b", branch="c"))

    def test_upsert_in_place_keeps_gate_fields(self):
        rows = [{"id": "a", "val_loss": 1}, {"id": "b", "val_loss": 2, "sprt": {"elo": 3}, "verdict": "merged"}]
        nl.upsert_row(rows, {"id": "b", "val_loss": 5, "sprt": None, "notes": ""})
        nl.upsert_row(rows, {"id": "c", "val_loss": 6})
        self.assertEqual([r["id"] for r in rows], ["a", "b", "c"])
        self.assertEqual(rows[1], {"id": "b", "val_loss": 5, "sprt": {"elo": 3}, "verdict": "merged"})
        rows = [{"id": "g", "branch": "nnue-g"}, {"id": "h", "branch": "local-h"}]
        nl.upsert_row(rows, {"id": "g", "branch": "local-g"})
        nl.upsert_row(rows, {"id": "h", "branch": "nnue-h"})
        self.assertEqual([r["branch"] for r in rows], ["nnue-g", "nnue-h"])

    def test_record_local_no_push(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "k1.log").write_text(LOG)
            exp = d / "experiments.jsonl"
            exp.write_text('{"id": "k1", "val_loss": 9, "sprt": {"elo": 1}}\n')
            with mock.patch.object(nl, "EXPERIMENTS", exp), mock.patch("sys.stdout"), \
                    mock.patch.object(sys, "argv", ["netloop.py", "record-local", "--name", "k1", "--baseline", "g3f",
                                                    "--data-dir", str(d), "--no-push", "--notes", "kb test"]):
                nl.main()
            rows = nl.parse_rows(exp.read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["branch"], rows[0]["baseline"], rows[0]["val_loss"], rows[0]["sprt"]),
                             ("local-k1", "g3f", 0.026456, {"elo": 1}))

    def test_best_scale_parse(self):
        out = ("val positions 1000000 loss 0.026169 mean|eval| 365.6cp\n"
               "best scale x0.9312: loss 0.026101 mean|eval| 340.5cp\n")
        with mock.patch.object(nl.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0, stdout=out, stderr="")) as r, \
                mock.patch("sys.stdout"):
            got = nl.best_scale("t", "n.bin", "v.bin", buckets=8, king_buckets="kb8m")
        self.assertEqual(got, {"best_scale_val": 0.026101, "best_scale_factor": 0.9312})
        self.assertIn("--king-buckets", r.call_args.args[0])


class EmbedRecords(unittest.TestCase):
    """cmd_embed with git/cargo/engine mocked: the row is pushed after the branch."""

    def embed(self, *extra):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "e9.log").write_text(LOG.replace("king-buckets=kb8m (KB=8) factorised ", "")
                                      .replace("H=512 buckets=8", "H=16 buckets=1"))
            net = d / "e9.bin"
            net.write_bytes(b"\0" * nl.net_bytes(16, 1))
            calls, pushed = [], []

            def fake_run(cmd, **kw):
                cmd = [str(c) for c in cmd]
                calls.append(cmd)
                if "worktree" in cmd and "add" in cmd:
                    wt = Path(cmd[cmd.index("-b") + 2])
                    self.addCleanup(shutil.rmtree, wt, True)
                    (wt / "src").mkdir(parents=True)
                    (wt / "net").mkdir()
                    (wt / "src/nnue.rs").write_text("pub(crate) const H: usize = 16;\n")
                return subprocess.CompletedProcess(cmd, 0)

            def fake_sub(cmd, **kw):
                cmd = [str(c) for c in cmd]
                out = "perfttest summary: ok\n" if cmd[0].endswith("rustfish") else ""
                return subprocess.CompletedProcess(cmd, 1 if "rev-parse" in cmd else 0, stdout=out, stderr="")

            bench = mock.Mock(run_bench=lambda e: (123, 456))
            with mock.patch.object(nl, "run", side_effect=fake_run), \
                    mock.patch.object(nl.subprocess, "run", side_effect=fake_sub), \
                    mock.patch.object(nl, "load_check_bench", return_value=bench), \
                    mock.patch.object(nl, "publish_row", side_effect=lambda row, **kw: pushed.append((row, kw))), \
                    mock.patch.object(sys, "argv", ["netloop.py", "embed", "--name", "e9", "--net", str(net),
                                                    "--hidden", "16", "--buckets", "1", "--data-dir", str(d),
                                                    *extra]), \
                    mock.patch("sys.stdout"):
                nl.main()
            return calls, pushed

    def test_pushes_row_after_branch(self):
        calls, pushed = self.embed("--baseline", "g3f", "--notes", "hello", "--seeds", "4,5")
        self.assertTrue(any(c[-2:] == ["origin", "nnue-e9"] for c in calls if "push" in c))
        self.assertEqual(len(pushed), 1)
        row, kw = pushed[0]
        self.assertEqual(kw, {"why": "embed"})
        self.assertEqual((row["id"], row["branch"], row["baseline"], row["notes"], row["seeds"]),
                         ("e9", "nnue-e9", "g3f", "hello", [4, 5]))
        self.assertEqual((row["hidden"], row["buckets"], row["val_loss"]), (16, 1, 0.026456))

    def test_no_record_and_no_push(self):
        self.assertEqual(self.embed("--no-record")[1], [])
        self.assertEqual(self.embed("--no-push")[1], [])


GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


class PublishRow(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, GIT_ENV)
        self.env.start()
        d = Path(self.td.name)
        self.origin, self.a, self.b = d / "origin.git", d / "a", d / "b"
        self.g("init", "--quiet", "--bare", "-b", "main", self.origin, cwd=d)
        self.g("clone", "--quiet", self.origin, self.a, cwd=d)
        (self.a / "net").mkdir()
        (self.a / "net/experiments.jsonl").write_text('{"id": "old", "val_loss": 1}\n')
        (self.a / "other.txt").write_text("keep\n")
        self.g("add", ".", cwd=self.a)
        self.g("commit", "--quiet", "-m", "init", cwd=self.a)
        self.g("push", "--quiet", "origin", "HEAD:main", cwd=self.a)
        self.g("clone", "--quiet", self.origin, self.b, cwd=d)

    @staticmethod
    def g(*a, cwd):
        subprocess.run(["git", *map(str, a)], cwd=cwd, check=True, capture_output=True)

    def tearDown(self):
        self.env.stop()
        self.td.cleanup()

    def origin_git(self, *a):
        return subprocess.run(["git", "--git-dir", str(self.origin), *a], capture_output=True, text=True,
                              check=True).stdout

    def remote_rows(self):
        return nl.parse_rows(self.origin_git("show", "main:net/experiments.jsonl"))

    def race(self):
        """Another clone lands an unrelated commit on main."""
        self.g("pull", "--quiet", "--rebase", cwd=self.b)
        (self.b / "other.txt").write_text(f"{time.time()}\n")
        self.g("commit", "--quiet", "-am", "race", cwd=self.b)
        self.g("push", "--quiet", "origin", "HEAD:main", cwd=self.b)

    def test_push_dedup_and_update(self):
        with mock.patch("sys.stdout"):
            self.assertIsNotNone(nl.publish_row({"id": "n1", "val_loss": 0.5, "notes": "x"}, repo=self.a))
            self.assertIsNone(nl.publish_row({"id": "n1", "val_loss": 0.5, "notes": "x"}, repo=self.a))
            nl.publish_row({"id": "n1", "val_loss": 0.4, "notes": ""}, repo=self.b)  # from a stale clone
        rows = self.remote_rows()
        self.assertEqual([r["id"] for r in rows], ["old", "n1"])
        self.assertEqual(rows[1], {"id": "n1", "val_loss": 0.4, "notes": "x"})
        stat = self.origin_git("show", "--stat", "--format=%s", "main")
        self.assertIn("Update n1 in net/experiments.jsonl", stat)
        self.assertIn("1 file changed", stat)
        self.assertEqual(sorted(self.origin_git("ls-tree", "-r", "--name-only", "main").split()),
                         ["net/experiments.jsonl", "other.txt"])
        # Working trees are never touched.
        self.assertEqual((self.b / "net/experiments.jsonl").read_text(), '{"id": "old", "val_loss": 1}\n')

    def test_retries_once_after_a_race(self):
        real, pushes = nl.git_out, []

        def racing(repo, *a, **kw):
            if a and a[0] == "push":
                pushes.append(a)
                if len(pushes) == 1:
                    self.race()
            return real(repo, *a, **kw)

        with mock.patch.object(nl, "git_out", side_effect=racing), mock.patch("sys.stdout"):
            nl.publish_row({"id": "n2", "val_loss": 0.3}, repo=self.a)
        self.assertEqual(len(pushes), 2)
        self.assertEqual([r["id"] for r in self.remote_rows()], ["old", "n2"])
        self.assertEqual(self.origin_git("log", "--format=%s", "main").split("\n")[:2],
                         ["Record n2 in net/experiments.jsonl", "race"])

    def test_gives_up_after_two_rejections(self):
        real = nl.git_out

        def always_racing(repo, *a, **kw):
            if a and a[0] == "push":
                self.race()
            return real(repo, *a, **kw)

        with mock.patch.object(nl, "git_out", side_effect=always_racing), mock.patch("sys.stdout"):
            with self.assertRaises(RuntimeError):
                nl.publish_row({"id": "n3"}, repo=self.a)
        self.assertNotIn("n3", [r["id"] for r in self.remote_rows()])


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

    def test_record_options_default(self):
        e = self.parse(["embed", "--name", "x", "--net", "n.bin", "--hidden", "512", "--buckets", "8"])
        self.assertEqual((e["no_record"], e["best_scale"], e["baseline"]), (False, False, "main"))
        a = self.parse(["auto"])
        self.assertEqual((a["no_record"], a["best_scale"]), (False, False))


if __name__ == "__main__":
    unittest.main()
