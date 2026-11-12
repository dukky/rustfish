#!/usr/bin/env python3
"""Stdlib tests for the machine-status push: POST /api/machine (and the old
/api/laptop name), its auth, the stored payloads, and the parsers in
scripts/machine_status.py and scripts/laptop_status.py. No network.

Run: python3 deploy/dashboard/test_machine.py
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
os.environ["MACHINE_TOKEN"] = "s3cret-token"
os.environ["LAPTOP_TOKEN"] = "old-laptop-token"
os.environ.pop("DATA_DIR", None)
import laptop_status  # noqa: E402
import machine_status  # noqa: E402
import server  # noqa: E402

LOG = """init: g5m1.bin is a one-bucket net
train H=512 buckets=8 king-buckets=kb8m (KB=8) factorised params=3154440 epochs=4 x 975777298 batch=16384 lr 0.0002->0.000001
start val loss 0.026254 (quantized 0.026254) mean|eval| 395.1cp
epoch 1 lr 1.71e-4 train 0.025292 val 0.026142 (quantized 0.026167) mean|eval| 398.8cp 1088796 pos/s -> g5kb8m.e1.bin
epoch 2 lr 1.01e-4 train 0.024979 val 0.026114 (quantized 0.026120) mean|eval| 400.1cp 1000000 pos/s -> g5kb8m.e2.bin
"""
# A chain log: a finished plain stage, then a 1024-wide kb8m stage in progress.
CHAIN_LOG = """train H=1024 buckets=8 params=1 epochs=10 x 3201732989 batch=16384 lr 0.001->0.00001
epoch 10 lr 1e-5 train 0.02 val 0.0258 (quantized 0.0258) mean|eval| 450.0cp 2250000 pos/s -> g8h1024.e10.bin
wrote g8h1024.bin (1 bytes, H=1024 buckets=8)
train H=1024 buckets=8 king-buckets=kb8m (KB=8) factorised params=2 epochs=4 x 3201732989 batch=16384 lr 0.0002->0.000001
epoch 1 lr 1.7e-4 train 0.024 val 0.025700 (quantized 0.025710) mean|eval| 460.1cp 2000000 pos/s -> g8h1024kb8m.e1.bin
"""
DG_LOG = """datagen: games 0..200000 seed 2005 nodes 5000 softnodes off threads 9 -> /home/u/rustfish-data/D2005.txt
datagen: games 1000/200000 positions 79000 (79.0/game, 120.0 plies/game, 10 adjudicated) 100s 10.00 games/s 790 pos/s  W/D/L 1/2/3  5000 nodes/move depth 9.00
datagen: games 50000/200000 positions 3950000 (79.0/game, 120.0 plies/game, 10 adjudicated) 5000s 10.00 games/s 790 pos/s  W/D/L 1/2/3  5000 nodes/move depth 9.00
"""


class LaptopParserTest(unittest.TestCase):
    """The old laptop reporter still posts to /api/laptop."""

    def test_in_progress(self):
        r = laptop_status.parse_train_log(LOG)
        self.assertEqual((r["epoch"], r["epochs"]), (2, 4))
        self.assertEqual(r["eta_s"], int(2 * 975777298 / 1000000))

    def test_etime(self):
        self.assertEqual(laptop_status.etime_s("1-02:03:04"), 93784)


class MachineParserTest(unittest.TestCase):
    def test_train_log(self):
        now = 1_000_000.0
        r = machine_status.parse_train_log(LOG, mtime=now - 488, now=now)  # ~half an epoch since the last write
        self.assertEqual((r["hidden"], r["buckets"], r["king_buckets"]), (512, 8, "kb8m"))
        self.assertEqual((r["epoch"], r["epochs"], r["val"], r["val_q"], r["pos_s"]), (2, 4, 0.026114, 0.02612, 1000000))
        self.assertAlmostEqual(r["progress"], (2 + 488e6 / 975777298) / 4, places=3)
        self.assertEqual(r["eta_s"], int((4 - 2 - 488e6 / 975777298) * 975777298 / 1e6))
        self.assertFalse(r["done"])
        done = machine_status.parse_train_log(LOG + "keep-best: epoch 2 (quantized val 0.026120) is the output\n"
                                                    "wrote /x/g5kb8m.bin (1 bytes, H=512 buckets=8)\n")
        self.assertEqual((done["done"], done["keep_best"], done["out"], done["progress"], done["eta_s"]),
                         (True, 2, "g5kb8m.bin", 1.0, None))
        self.assertIsNone(machine_status.parse_train_log("no trainer here\n"))

    def test_chain_log_takes_last_stage(self):
        r = machine_status.parse_train_log(CHAIN_LOG, mtime=100.0, now=100.0)
        self.assertEqual((r["hidden"], r["king_buckets"], r["epoch"], r["epochs"], r["done"]), (1024, "kb8m", 1, 4, False))
        self.assertEqual(r["progress"], 0.25)

    def test_datagen_log(self):
        r = machine_status.parse_datagen_log(DG_LOG)
        self.assertEqual((r["seed"], r["out"], r["games_done"], r["games_total"], r["positions"]),
                         (2005, "D2005.txt", 50000, 200000, 3950000))
        self.assertEqual((r["games_s"], r["pos_s"], r["eta_s"], r["done"]), (10.0, 790.0, 15000, False))
        fin = machine_status.parse_datagen_log(DG_LOG.replace("datagen: games 50000", "datagen done: games 200000"))
        self.assertTrue(fin["done"])
        self.assertIsNone(fin["eta_s"])

    def test_collect_from_globs(self):
        with tempfile.TemporaryDirectory() as d:
            for name, text in (("g8h1024kb8m.log", CHAIN_LOG), ("old.log", LOG), ("D2005.log", DG_LOG)):
                with open(os.path.join(d, name), "w") as fh:
                    fh.write(text)
            os.utime(os.path.join(d, "old.log"), (time.time() - 99999, time.time() - 99999))
            args = machine_status.parser().parse_args(
                ["--machine", "desktop", "--gpu-log", os.path.join(d, "*.log"),
                 "--datagen-log", os.path.join(d, "D*.log"), "--disk", d])
            p = machine_status.collect(args)
        self.assertEqual((p["schema"], p["machine"]), ("machine-status/1", "desktop"))
        self.assertEqual(p["gpu"]["name"], "g8h1024kb8m")  # newest trainer log; D2005.log has no train header
        self.assertFalse(p["gpu"]["idle"])
        self.assertEqual((p["datagen"]["name"], p["datagen"]["seed"]), ("D2005", 2005))
        self.assertEqual(p["disk"][0]["path"], d)
        self.assertIsNotNone(p["disk"][0]["free_gb"])
        self.assertLess(len(json.dumps(p)), server.MACHINE_MAX_BODY)

    def test_nothing_running(self):
        args = machine_status.parser().parse_args(["--machine", "x", "--gpu-log", "/nonexistent/*.log"])
        p = machine_status.collect(args)
        self.assertIsNone(p["gpu"])
        self.assertIsNone(p["datagen"])


class PostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def setUp(self):
        server.MACHINES.clear()

    def post(self, body, token=None, path="/api/machine"):
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = "Bearer " + token
        req = urllib.request.Request(self.base + path, data=body, method="POST", headers=h)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            e.close()
            return e.code

    def test_auth_and_limits(self):
        body = json.dumps({"schema": "machine-status/1", "machine": "desktop", "ts": 1}).encode()
        self.assertEqual(self.post(body), 401)
        self.assertEqual(self.post(body, "wrong"), 401)
        self.assertEqual(self.post(b"x" * 20000, "s3cret-token"), 413)
        self.assertEqual(self.post(b"not json", "s3cret-token"), 400)
        self.assertEqual(self.post(b"[1]", "s3cret-token"), 400)
        self.assertEqual(self.post(body, "s3cret-token"), 204)
        self.assertEqual(self.post(b"x", "s3cret-token", path="/api/other"), 404)
        st = server.machines_status()
        self.assertEqual([m["name"] for m in st], ["desktop"])
        self.assertEqual((st[0]["schema"], st[0]["payload"]["ts"]), ("machine-status/1", 1))
        self.assertLess(st[0]["age_s"], 5)

    def test_machines_kept_apart_and_legacy_path(self):
        self.assertEqual(self.post(json.dumps({"machine": "desktop", "ts": 1}).encode(), "s3cret-token"), 204)
        # laptop_status.py's payload has no machine name: stored as "laptop".
        self.assertEqual(self.post(json.dumps({"ts": 2, "trainers": []}).encode(), "old-laptop-token",
                                   path="/api/laptop"), 204)
        self.assertEqual(self.post(json.dumps({"machine": "../etc", "ts": 3}).encode(), "s3cret-token"), 204)
        names = {m["name"]: m["payload"]["ts"] for m in server.machines_status()}
        self.assertEqual(names, {"desktop": 1, "laptop": 2, "default": 3})  # a bad name falls back

    def test_unset_tokens_reject(self):
        old = os.environ.pop("MACHINE_TOKEN"), os.environ.pop("LAPTOP_TOKEN")
        try:
            self.assertEqual(self.post(b"{}", "anything"), 401)
            self.assertFalse(server.machine_authorized("Bearer "))
        finally:
            os.environ["MACHINE_TOKEN"], os.environ["LAPTOP_TOKEN"] = old

    def test_cap_drops_longest_silent(self):
        for i in range(server.MACHINE_MAX + 2):
            server.machine_store(f"m{i}", {"ts": i}, now=1000 + i)
        self.assertEqual(len(server.MACHINES), server.MACHINE_MAX)
        self.assertNotIn("m0", server.MACHINES)
        self.assertNotIn("m1", server.MACHINES)

    def test_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            old = server.DATA_DIR
            server.DATA_DIR = d
            try:
                server.machine_store("desktop", {"ts": 5})
                server.MACHINES.clear()
                server.machines_load()
                self.assertEqual(server.MACHINES["desktop"]["payload"], {"ts": 5})
                # Upgrading from the single laptop.json.
                os.remove(os.path.join(d, "machines.json"))
                with open(os.path.join(d, "laptop.json"), "w") as fh:
                    json.dump({"payload": {"ts": 7}, "received": 123.0}, fh)
                server.MACHINES.clear()
                server.machines_load()
                self.assertEqual(server.MACHINES["laptop"]["payload"], {"ts": 7})
            finally:
                server.DATA_DIR = old


if __name__ == "__main__":
    unittest.main()
