#!/usr/bin/env python3
"""Fit a CCRL-scale rating from fastchess PGNs played against scripts/anchors.json.

Every game must involve the engine under test (default name "rustfish") and one
anchor whose PGN name matches an anchor "name". Results are aggregated per
anchor and fed to the same maximum-likelihood fit as fit_rating.py.

Point estimate: the rating r solving sum_i n_i (s_i - E(r - ccrl_i)) = 0, i.e.
the maximum of fit_rating.log_lik (draws as half points). That equation is
unbiased whatever the draw rate; draws only change its variance.

CI ("ci"): r +- 1.96 * sqrt(sd_games^2 + sd_anchors^2), where
  - sd_games is the sandwich (delta-method) error of that estimating equation
    with the variance of each anchor's total score taken from its game pairs.
    fastchess -repeat plays every opening twice with colours reversed, so the
    pair score (0, 0.5, 1, 1.5, 2) is the independent unit (pentanomial model).
    Games whose partner is missing count as single games (trinomial variance).
  - sd_anchors propagates each anchor's own CCRL error: dr/dccrl_i =
    w_i / sum w with w_i = n_i E_i (1 - E_i). "ccrl_err" in the anchors file
    is CCRL's printed +- (95%); anchors without one get DEFAULT_ANCHOR_ERR.
    Errors are treated as independent between anchors.
"ci_binomial" is the old interval: likelihood ratio of the binomial score
likelihood, game noise only, which counts every game as a full win or loss
and so overstates the noise when most games are draws.

Usage:
  python3 scripts/ccrl_ladder.py --anchors scripts/anchors.json \
      --summary summary.md --json results.json games/*.pgn
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_rating import expected, fit  # noqa: E402

TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
BOOK_RE = re.compile(r'([A-Za-z][^\s{}()]*)\s*\{book\}')
LOW_SCORE, HIGH_SCORE = 0.15, 0.85
K = math.log(10) / 400.0
Z95 = 1.96
DEFAULT_ANCHOR_ERR = 20.0  # CCRL +- (95%) assumed when the anchors file has none
PENTA_SCORES = (0.0, 0.5, 1.0, 1.5, 2.0)
R_MIN, R_MAX = 1000.0, 4000.0


def iter_games(path):
    """Yield the tag dict of each game in a PGN file.

    Adds "_opening": the book moves (SAN marked {book} by fastchess), which
    both games of a -repeat pair share."""
    tags, moves = {}, []

    def done():
        tags["_opening"] = " ".join(BOOK_RE.findall(" ".join(moves)))
        return tags

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TAG_RE.match(line)
            if m:
                if m.group(1) == "Event" and tags:
                    yield done()
                    tags, moves = {}, []
                tags[m.group(1)] = m.group(2)
            elif line.strip():
                moves.append(line.strip())
    if tags:
        yield done()


def collect(paths, engine, anchors):
    """Per-anchor counts from PGN files.

    Returns ({anchor: {"wdl": [w, d, l], "penta": [5 counts], "unpaired": [w, d, l]}},
    time_losses). Two games pair up when they share file, opponent, Round and
    book opening and the engine had each colour once."""
    groups = {}
    time_losses = 0
    for path in paths:
        for g in iter_games(path):
            white, black, result = g.get("White"), g.get("Black"), g.get("Result")
            if engine not in (white, black) or result not in ("1-0", "0-1", "1/2-1/2"):
                continue
            opp = black if white == engine else white
            if opp not in anchors:
                sys.exit(f"unknown opponent {opp!r} in {path}")
            if result == "1/2-1/2":
                score = 0.5
            elif (result == "1-0") == (white == engine):
                score = 1.0
            else:
                score = 0.0
                if "time" in g.get("Termination", "").lower():
                    time_losses += 1
            key = (path, opp, g.get("Round"), g["_opening"])
            groups.setdefault(key, []).append((white == engine, score))

    stats = {}
    for (_, opp, _, _), games in groups.items():
        s = stats.setdefault(opp, {"wdl": [0, 0, 0], "penta": [0] * 5, "unpaired": [0, 0, 0]})
        for _, score in games:
            s["wdl"][wdl_index(score)] += 1
        if len(games) == 2 and games[0][0] != games[1][0]:
            s["penta"][int(round(2 * (games[0][1] + games[1][1])))] += 1
        else:
            for _, score in games:
                s["unpaired"][wdl_index(score)] += 1
    return stats, time_losses


def wdl_index(score):
    return 0 if score == 1.0 else 1 if score == 0.5 else 2


def score_sum_variance(row):
    """Estimated variance of an anchor's total score (sum over its games).

    Pairs contribute m * var(pair score); unpaired games u * var(game score),
    each variance taken around its own sample mean. A row without "penta"
    (old logs) is treated as all unpaired, i.e. trinomial."""
    penta = row.get("penta") or [0] * 5
    unp = row.get("unpaired")
    if unp is None:
        unp = [row["wins"], row["draws"], row["losses"]] if not any(penta) else [0, 0, 0]
    var = 0.0
    m = sum(penta)
    if m:
        mean = sum(c * x for c, x in zip(penta, PENTA_SCORES)) / m
        var += sum(c * (x - mean) ** 2 for c, x in zip(penta, PENTA_SCORES))
    u = sum(unp)
    if u:
        mean = (unp[0] + 0.5 * unp[1]) / u
        var += sum(c * (x - mean) ** 2 for c, x in zip(unp, (1.0, 0.5, 0.0)))
    return var


def ml_rating(rows):
    """Root of sum n (s - E): the same estimate as fit_rating.fit, by bisection."""
    def f(r):
        return sum(x["games"] * (x["score"] - expected(r, x["ccrl"])) for x in rows)
    lo, hi = R_MIN, R_MAX
    if f(lo) <= 0:
        return lo
    if f(hi) >= 0:
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def info(r, rows):
    """-d/dr of sum n (s - E) = K sum n E (1 - E)."""
    return K * sum(x["games"] * expected(r, x["ccrl"]) * (1 - expected(r, x["ccrl"])) for x in rows)


def game_sd(r, rows):
    """Sandwich standard error of the estimate from game noise (pentanomial)."""
    var = sum(score_sum_variance(x) for x in rows)
    if var <= 0:  # every pair scored identically: fall back to the binomial variance
        var = sum(x["games"] * expected(r, x["ccrl"]) * (1 - expected(r, x["ccrl"])) for x in rows)
    return math.sqrt(var) / info(r, rows)


def anchor_sd(r, rows, default_err=DEFAULT_ANCHOR_ERR):
    """Standard error of the estimate from the anchors' own CCRL rating errors."""
    w = [x["games"] * expected(r, x["ccrl"]) * (1 - expected(r, x["ccrl"])) for x in rows]
    sig = [(x.get("ccrl_err") or default_err) / Z95 for x in rows]
    return math.sqrt(sum((wi * si) ** 2 for wi, si in zip(w, sig))) / sum(w)


def draw_aware_rating(rows):
    """Efficient-weight estimate: sum n (s - E) E(1-E) / v = 0 with v each anchor's
    per-game score variance (from pairs). Only a diagnostic: if it lands far from
    the main estimate the draw rate is biasing the equal-weight fit."""
    v = []
    for x in rows:
        v.append(max(score_sum_variance(x) / x["games"], 0.02))

    def f(r):
        return sum(x["games"] * (x["score"] - expected(r, x["ccrl"]))
                   * expected(r, x["ccrl"]) * (1 - expected(r, x["ccrl"])) / vi
                   for x, vi in zip(rows, v))
    lo, hi = R_MIN, R_MAX
    if f(lo) <= 0 or f(hi) >= 0:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analyse(rows, default_err=DEFAULT_ANCHOR_ERR, binomial=True):
    """All fit numbers for per-anchor rows (anchor, ccrl, wins, draws, losses,
    games, score, optional penta / unpaired / ccrl_err)."""
    best = ml_rating(rows)
    sd_g = game_sd(best, rows)
    sd_a = anchor_sd(best, rows, default_err)
    half = Z95 * math.hypot(sd_g, sd_a)
    out = {
        "rating": round(best, 1),
        "ci": [round(best - half, 1), round(best + half, 1)],
        "ci_method": "pentanomial+anchor-err" if any(any(x.get("penta") or []) for x in rows)
                     else "trinomial+anchor-err",
        "sd_games": round(sd_g, 2),
        "sd_anchors": round(sd_a, 2),
        "pairs": sum(sum(x.get("penta") or []) for x in rows),
    }
    da = draw_aware_rating(rows)
    out["rating_draw_aware"] = None if da is None else round(da, 1)
    if binomial:
        _, lo, hi = fit([(x["ccrl"], x["wins"], x["draws"], x["losses"]) for x in rows])
        out["ci_binomial"] = [lo, hi]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--engine", default="rustfish")
    ap.add_argument("--summary", help="write a markdown summary here")
    ap.add_argument("--json", help="write machine-readable results here")
    ap.add_argument("pgns", nargs="+")
    args = ap.parse_args()

    with open(args.anchors) as fh:
        meta = {a["name"]: a for a in json.load(fh)["anchors"]}
    anchors = {name: a["ccrl"] for name, a in meta.items()}

    stats, time_losses = collect(args.pgns, args.engine, anchors)
    if not stats:
        sys.exit("no finished games found")

    rows = []
    for name, s in sorted(stats.items(), key=lambda kv: anchors[kv[0]]):
        w, d, l = s["wdl"]
        n = w + d + l
        rows.append({"anchor": name, "ccrl": anchors[name], "ccrl_err": meta[name].get("ccrl_err"),
                     "wins": w, "draws": d, "losses": l, "games": n, "score": (w + 0.5 * d) / n,
                     "penta": s["penta"], "unpaired": s["unpaired"]})
    total = sum(r["games"] for r in rows)
    res = analyse(rows)
    best, (lo, hi) = res["rating"], res["ci"]
    blo, bhi = res["ci_binomial"]
    unpaired = sum(sum(r["unpaired"]) for r in rows)
    defaulted = [r["anchor"] for r in rows if not r["ccrl_err"]]

    lines = [
        f"## {args.engine}: CCRL Blitz-scale estimate **{best:.0f}** (95% CI {lo:.0f} .. {hi:.0f}, "
        f"+-{(hi - lo) / 2:.1f})",
        "",
        f"{total} games against {len(rows)} anchors ({res['pairs']} colour-reversed pairs, "
        f"{unpaired} unpaired games); {time_losses} losses on time.",
        "",
        "| Anchor | CCRL | W | D | L | Games | Score | Pairs LL/LD/DD+WL/WD/WW |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        err = f" +-{r['ccrl_err']}" if r["ccrl_err"] else ""
        lines.append(f"| {r['anchor']} | {r['ccrl']}{err} | {r['wins']} | {r['draws']} | "
                     f"{r['losses']} | {r['games']} | {100 * r['score']:.1f}% | "
                     f"{'/'.join(str(c) for c in r['penta'])} |")
    da = res["rating_draw_aware"]
    lines += [
        "",
        f"CI: pentanomial game noise +-{Z95 * res['sd_games']:.1f} combined with the anchors' own "
        f"CCRL errors +-{Z95 * res['sd_anchors']:.1f}"
        + (f" (+-{DEFAULT_ANCHOR_ERR:.0f} assumed for {', '.join(defaulted)})" if defaulted else "")
        + f". Old binomial CI (game noise, draws as win+loss): {blo:.0f} .. {bhi:.0f} "
          f"(+-{(bhi - blo) / 2:.1f}).",
        f"Draw-aware weighted estimate: {da:.0f} ({da - best:+.1f})." if da is not None
        else "Draw-aware weighted estimate: undefined (no sign change).",
    ]

    # Games against anchors far from the engine's level carry little rating
    # information; keep roughly half the anchors inside +-150 of the estimate.
    lopsided = [r for r in rows if not LOW_SCORE <= r["score"] <= HIGH_SCORE]
    near = sum(1 for r in rows if abs(r["ccrl"] - best) <= 150)
    if lopsided or near < len(rows) / 2:
        lines += ["", "### Anchor health", ""]
        for r in lopsided:
            lines.append(f"- {r['anchor']} ({r['ccrl']}) scored {100 * r['score']:.0f}%: "
                         f"outside {100 * LOW_SCORE:.0f}-{100 * HIGH_SCORE:.0f}%, consider replacing")
        lines.append(f"- {near}/{len(rows)} anchors within +-150 of {best:.0f}; aim for about half. "
                     f"Pick replacements near {best - 150:.0f}-{best + 250:.0f} from the CCRL complete list.")
    text = "\n".join(lines)
    print(text)

    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write(text + "\n")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(res, engine=args.engine, games=total, time_losses=time_losses, anchors=rows),
                      fh, indent=2)


if __name__ == "__main__":
    main()
