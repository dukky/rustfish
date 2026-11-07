#!/usr/bin/env python3
"""Unit tests for scripts/ccrl_ladder.py (stdlib unittest only).

    python3 scripts/test_ccrl_ladder.py
"""
import importlib.util
import math
import os
import random
import statistics
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ccrl_ladder", os.path.join(HERE, "ccrl_ladder.py"))
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)


def row_from_pairs(ccrl, pair_scores, err=None):
    penta = [0] * 5
    w = d = l = 0
    for a, b in pair_scores:
        penta[int(round(2 * (a + b)))] += 1
        for s in (a, b):
            w, d, l = w + (s == 1.0), d + (s == 0.5), l + (s == 0.0)
    n = w + d + l
    return {"anchor": f"a{ccrl}", "ccrl": ccrl, "ccrl_err": err, "wins": w, "draws": d, "losses": l,
            "games": n, "score": (w + 0.5 * d) / n, "penta": penta, "unpaired": [0, 0, 0]}


def sample_game(rng, mean, draw):
    """One game score with the given mean and draw rate."""
    u = rng.random()
    win = mean - draw / 2
    return 1.0 if u < win else 0.5 if u < win + draw else 0.0


def sample_pair(rng, e, draw, flip):
    """A colour-reversed pair: with probability `flip` the opening decides it
    (one win, one loss, score 1); otherwise two independent games whose mean is
    set so the pair's expected score is still 2e. This is the negative pair
    correlation that makes the pentanomial variance smaller than the trinomial."""
    if rng.random() < flip:
        return (1.0, 0.0)
    m = (2 * e - flip) / (2 * (1 - flip))
    return (sample_game(rng, m, draw), sample_game(rng, m, draw))


class Variance(unittest.TestCase):
    def test_penta_exact(self):
        # 5 LL + 5 WW: pair mean 1, each pair deviates by 1 -> sum variance 10.
        r = {"penta": [5, 0, 0, 0, 5], "unpaired": [0, 0, 0], "wins": 10, "draws": 0, "losses": 10}
        self.assertAlmostEqual(cl.score_sum_variance(r), 10.0)
        # All pairs 1.5 -> no variance.
        r = {"penta": [0, 0, 0, 7, 0], "unpaired": [0, 0, 0], "wins": 7, "draws": 7, "losses": 0}
        self.assertEqual(cl.score_sum_variance(r), 0.0)

    def test_trinomial_and_mixed(self):
        # Old log row (no penta): W1 D2 L1, mean .5 -> 0.25 + 0 + 0 + 0.25.
        self.assertAlmostEqual(cl.score_sum_variance({"wins": 1, "draws": 2, "losses": 1}), 0.5)
        # Two pairs (0.5, 1.5) -> 2 * 0.25, plus unpaired W, L -> 2 * 0.25.
        r = {"penta": [0, 1, 0, 1, 0], "unpaired": [1, 0, 1], "wins": 3, "draws": 2, "losses": 1}
        self.assertAlmostEqual(cl.score_sum_variance(r), 1.0)

    def test_anchor_sd(self):
        one = [row_from_pairs(3400, [(1.0, 0.0)] * 32, err=19.6)]
        self.assertAlmostEqual(cl.anchor_sd(3400, one), 10.0)
        two = one + [dict(one[0], anchor="b")]
        self.assertAlmostEqual(cl.anchor_sd(3400, two), 10.0 / math.sqrt(2))
        # Missing ccrl_err falls back to the documented default.
        self.assertAlmostEqual(cl.anchor_sd(3400, [dict(one[0], ccrl_err=None)]),
                               cl.DEFAULT_ANCHOR_ERR / cl.Z95)


class Estimate(unittest.TestCase):
    def test_ml_matches_grid_fit(self):
        rng = random.Random(1)
        rows = [row_from_pairs(a, [sample_pair(rng, cl.expected(3450, a), 0.4, 0.2) for _ in range(26)])
                for a in (3300, 3380, 3450, 3500, 3600)]
        best, _, _ = cl.fit([(x["ccrl"], x["wins"], x["draws"], x["losses"]) for x in rows])
        self.assertAlmostEqual(cl.ml_rating(rows), best, delta=0.11)

    def test_old_row_without_pairs(self):
        rows = [{"anchor": "x", "ccrl": 3400, "wins": 20, "draws": 30, "losses": 14, "games": 64,
                 "score": 35 / 64}]
        res = cl.analyse(rows)
        self.assertEqual(res["ci_method"], "trinomial+anchor-err")
        self.assertLess(res["ci"][0], res["rating"])
        self.assertGreater(res["ci"][1], res["rating"])
        lo, hi = res["ci_binomial"]
        self.assertGreater(hi - lo, res["ci"][1] - res["ci"][0] - 2 * cl.Z95 * res["sd_anchors"])

    def test_monte_carlo_coverage(self):
        """Known truth: the pentanomial sd matches the spread of the estimate,
        its 95% interval covers the truth ~95% of the time, and the binomial sd
        overstates the spread when most games are draws."""
        rng = random.Random(7)
        truth = 3450.0
        design = [3320, 3345, 3377, 3417, 3450, 3487, 3491, 3525, 3553, 3579, 3631]
        est, sds, bin_sds, cover = [], [], [], 0
        for _ in range(400):
            rows = [row_from_pairs(a, [sample_pair(rng, cl.expected(truth, a), 0.4, 0.15)
                                       for _ in range(26)]) for a in design]
            r = cl.ml_rating(rows)
            sd = cl.game_sd(r, rows)
            est.append(r)
            sds.append(sd)
            bin_sds.append(math.sqrt(sum(x["games"] * cl.expected(r, x["ccrl"]) * (1 - cl.expected(r, x["ccrl"]))
                                         for x in rows)) / cl.info(r, rows))
            cover += abs(r - truth) <= cl.Z95 * sd
        spread = statistics.pstdev(est)
        self.assertAlmostEqual(statistics.mean(est), truth, delta=3 * spread / math.sqrt(400) + 1)
        self.assertAlmostEqual(statistics.mean(sds) / spread, 1.0, delta=0.12)
        self.assertGreater(cover / 400, 0.91)
        self.assertLess(cover / 400, 0.99)
        self.assertGreater(statistics.mean(bin_sds) / spread, 1.4)

    def test_draw_aware_close(self):
        rng = random.Random(3)
        rows = [row_from_pairs(a, [sample_pair(rng, cl.expected(3450, a), 0.6, 0.15) for _ in range(200)])
                for a in (3350, 3450, 3550)]
        self.assertAlmostEqual(cl.draw_aware_rating(rows), cl.ml_rating(rows), delta=10)


GAME = """[Event "Fastchess Tournament"]
[Round "{rnd}"]
[White "{white}"]
[Black "{black}"]
[Result "{result}"]
[Termination "normal"]

1. {m1} {{book}} Nf6 {{book}} 2. c4 {{-0.10/20 1.0s}} e6 {result}

"""


def pgn(*games):
    return "".join(GAME.format(rnd=r, white=w, black=b, result=res, m1=m1) for r, w, b, res, m1 in games)


class Pairing(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, text):
        path = os.path.join(self.dir.name, name)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def test_pairs_and_unpaired(self):
        a = self.write("s0.pgn", pgn(
            ("1", "rustfish", "x", "1-0", "d4"), ("1", "x", "rustfish", "1/2-1/2", "d4"),  # WD
            ("2", "x", "rustfish", "1-0", "e4"), ("2", "rustfish", "x", "0-1", "e4"),      # LL
            ("3", "rustfish", "x", "1/2-1/2", "c4")))                                      # lone D
        stats, _ = cl.collect([a], "rustfish", {"x": 3400})
        self.assertEqual(stats["x"]["penta"], [1, 0, 0, 1, 0])
        self.assertEqual(stats["x"]["unpaired"], [0, 1, 0])
        self.assertEqual(stats["x"]["wdl"], [1, 2, 2])

    def test_concatenated_shards_and_separate_files(self):
        # all-games.pgn is `cat games/*.pgn`: both shards have a Round 1; the
        # book opening keeps the pairs apart.
        s0 = pgn(("1", "rustfish", "x", "1-0", "d4"), ("1", "x", "rustfish", "0-1", "d4"))   # WW
        s1 = pgn(("1", "rustfish", "x", "0-1", "e4"), ("1", "x", "rustfish", "1-0", "e4"))   # LL
        cat = self.write("all.pgn", s0 + s1)
        stats, _ = cl.collect([cat], "rustfish", {"x": 3400})
        self.assertEqual(stats["x"]["penta"], [1, 0, 0, 0, 1])
        files = [self.write("a.pgn", s0), self.write("b.pgn", s0)]  # same round + opening, two files
        stats, _ = cl.collect(files, "rustfish", {"x": 3400})
        self.assertEqual(stats["x"]["penta"], [0, 0, 0, 0, 2])
        self.assertEqual(stats["x"]["unpaired"], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
