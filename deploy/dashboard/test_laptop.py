#!/usr/bin/env python3
"""Stdlib tests for the laptop card: log parser and POST /api/laptop auth.

Run: python3 deploy/dashboard/test_laptop.py
"""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
os.environ["LAPTOP_TOKEN"] = "s3cret-token"
os.environ.pop("DATA_DIR", None)
import laptop_status  # noqa: E402
import server  # noqa: E402

LOG = """init: g5m1.bin is a one-bucket net
train H=512 buckets=8 king-buckets=kb8m params=3154440 epochs=4 x 975777298 batch=16384 lr 0.0002->0.000001
start val loss 0.026254 (quantized 0.026254) mean|eval| 395.1cp
epoch 1 lr 1.71e-4 train 0.025292 val 0.026142 (quantized 0.026167) mean|eval| 398.8cp 1088796 pos/s -> g5kb8m.e1.bin
epoch 2 lr 1.01e-4 train 0.024979 val 0.026114 (quantized 0.026120) mean|eval| 400.1cp 1000000 pos/s -> g5kb8m.e2.bin
"""


class ParserTest(unittest.TestCase):
    def test_in_progress(self):
        r = laptop_status.parse_train_log(LOG)
        self.assertEqual((r["epoch"], r["epochs"]), (2, 4))
        self.assertEqual(r["val"], 0.026114)
        self.assertEqual(r["val_q"], 0.026120)
        self.assertEqual(r["pos_s"], 1000000)
        self.assertEqual(r["eta_s"], int(2 * 975777298 / 1000000))
        self.assertFalse(r["done"])

    def test_done(self):
        r = laptop_status.parse_train_log(LOG + "keep-best: epoch 2\nwrote g5kb8m.bin (1 bytes)\n")
        self.assertTrue(r["done"] and r["keep_best"])
        self.assertIsNone(r["eta_s"])

    def test_etime(self):
        self.assertEqual(laptop_status.etime_s("1-02:03:04"), 93784)
        self.assertEqual(laptop_status.etime_s("05:06"), 306)


class PostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}/api/laptop"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def post(self, body, token=None):
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = "Bearer " + token
        req = urllib.request.Request(self.url, data=body, method="POST", headers=h)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def test_auth(self):
        body = json.dumps({"ts": 1, "trainers": []}).encode()
        self.assertEqual(self.post(body), 401)
        self.assertEqual(self.post(body, "wrong"), 401)
        self.assertEqual(self.post(b"x" * 20000, "s3cret-token"), 413)
        self.assertEqual(self.post(b"not json", "s3cret-token"), 400)
        self.assertEqual(self.post(body, "s3cret-token"), 204)
        st = server.laptop_status()
        self.assertEqual(st["payload"]["ts"], 1)
        self.assertLess(st["age_s"], 5)

    def test_unset_token_rejects(self):
        old = os.environ.pop("LAPTOP_TOKEN")
        try:
            self.assertEqual(self.post(b"{}", "anything"), 401)
            self.assertFalse(server.laptop_authorized("Bearer "))
        finally:
            os.environ["LAPTOP_TOKEN"] = old


if __name__ == "__main__":
    unittest.main()
