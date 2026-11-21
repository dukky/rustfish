#!/usr/bin/env python3
"""Where are games lost/drawn, from fastchess PGNs with `{score/depth time}` comments.

Stdlib only. Re-implements the round-6 Clover-games method of
docs/search-diagnosis-2026-11-19.md (those scripts were scratch) so the numbers
are comparable, and adds draw/conversion and clock analysis:

  python3 scripts/diag/anchor_games.py --us rustfish games-*.pgn
  python3 scripts/diag/anchor_games.py --us rustfish --by-opponent --json out.json *.pgn

Definitions (all scores in our POV, centipawns, mate = +-10000):
  phase       non-pawn material, N/B 1, R 2, Q 4 (24 = full); open >= 20, mg 9-19, eg <= 8
  lost point  first of OUR moves after which the opponent's score stays <= -200 for good
  won point   same with the opponent's score >= +200 (the opponent is the judge both ways)
  opp loss of our move   opp score before our move (its previous move) minus after (its next)
  agree lag   our moves from the lost point until our own score stays <= -200
  optimism    our score minus the opponent's, at our move before the lost point (with the
              opponent's previous score) and at it (with the opponent's next score),
              raw and with the opponent's score divided by the opp/ours scale ratio
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------- minimal board
FILES = "abcdefgh"
N_D = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
K_D = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
B_D = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
R_D = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def sq(name):
    return FILES.index(name[0]), int(name[1]) - 1


class Board:
    def __init__(self, fen=None):
        self.b = {}
        if fen:
            self.set_fen(fen)
            return
        back = "RNBQKBNR"
        for f in range(8):
            self.b[(f, 0)] = "w" + back[f]
            self.b[(f, 1)] = "wP"
            self.b[(f, 6)] = "bP"
            self.b[(f, 7)] = "b" + back[f]
        self.stm = "w"
        self.castle = set("KQkq")
        self.ep = None

    def set_fen(self, fen):
        parts = fen.split()
        for r, row in enumerate(parts[0].split("/")):
            f = 0
            for c in row:
                if c.isdigit():
                    f += int(c)
                else:
                    self.b[(f, 7 - r)] = ("w" if c.isupper() else "b") + c.upper()
                    f += 1
        self.stm = parts[1] if len(parts) > 1 else "w"
        self.castle = set(parts[2]) - {"-"} if len(parts) > 2 else set()
        self.ep = sq(parts[3]) if len(parts) > 3 and parts[3] != "-" else None

    def key(self):
        return (tuple(sorted(self.b.items())), self.stm, tuple(sorted(self.castle)), self.ep)

    def phase(self):
        w = {"N": 1, "B": 1, "R": 2, "Q": 4}
        return sum(w.get(p[1], 0) for p in self.b.values())

    def men(self):
        return len(self.b)

    def attacked(self, s, by):
        f, r = s
        pr = -1 if by == "w" else 1
        for df in (-1, 1):
            if self.b.get((f + df, r + pr)) == by + "P":
                return True
        for df, dr in N_D:
            if self.b.get((f + df, r + dr)) == by + "N":
                return True
        for df, dr in K_D:
            if self.b.get((f + df, r + dr)) == by + "K":
                return True
        for dirs, kinds in ((B_D, "BQ"), (R_D, "RQ")):
            for df, dr in dirs:
                x, y = f + df, r + dr
                while 0 <= x < 8 and 0 <= y < 8:
                    p = self.b.get((x, y))
                    if p:
                        if p[0] == by and p[1] in kinds:
                            return True
                        break
                    x += df
                    y += dr
        return False

    def can_reach(self, frm, to, kind, capture):
        f, r = frm
        tf, tr = to
        df, dr = tf - f, tr - r
        if kind == "N":
            return (df, dr) in N_D
        if kind == "K":
            return max(abs(df), abs(dr)) == 1
        if kind in "BRQ":
            if kind == "B" and abs(df) != abs(dr):
                return False
            if kind == "R" and df and dr:
                return False
            if kind == "Q" and df and dr and abs(df) != abs(dr):
                return False
            if not df and not dr:
                return False
            sf = (df > 0) - (df < 0)
            sr = (dr > 0) - (dr < 0)
            x, y = f + sf, r + sr
            while (x, y) != to:
                if (x, y) in self.b:
                    return False
                x += sf
                y += sr
            return True
        if kind == "P":
            fwd = 1 if self.stm == "w" else -1
            if capture:
                return abs(df) == 1 and dr == fwd
            if df:
                return False
            if dr == fwd:
                return to not in self.b
            start = 1 if self.stm == "w" else 6
            return dr == 2 * fwd and r == start and (f, r + fwd) not in self.b and to not in self.b
        return False

    def legal_after(self, frm, to, ep_cap=None):
        saved = dict(self.b)
        p = self.b.pop(frm)
        if ep_cap:
            self.b.pop(ep_cap, None)
        self.b[to] = p
        king = next(s for s, q in self.b.items() if q == self.stm + "K")
        ok = not self.attacked(king, "b" if self.stm == "w" else "w")
        self.b = saved
        return ok

    def push_san(self, san):
        """Apply a SAN move. Returns (kind, is_capture)."""
        s = san.rstrip("+#!?")
        me, op = self.stm, ("b" if self.stm == "w" else "w")
        rank0 = 0 if me == "w" else 7
        if s in ("O-O", "O-O-O"):
            kf, rf, kt, rt = (4, 7, 6, 5) if s == "O-O" else (4, 0, 2, 3)
            self.b.pop((kf, rank0))
            self.b.pop((rf, rank0))
            self.b[(kt, rank0)] = me + "K"
            self.b[(rt, rank0)] = me + "R"
            self.castle -= set("KQ" if me == "w" else "kq")
            self.ep = None
            self.stm = op
            return "K", False
        m = re.match(r"^([NBRQK]?)([a-h]?)([1-8]?)(x?)([a-h][1-8])(=?[NBRQ])?$", s)
        if not m:
            raise ValueError("bad SAN " + san)
        kind = m.group(1) or "P"
        dfile, drank, cap, to = m.group(2), m.group(3), bool(m.group(4)), sq(m.group(5))
        promo = (m.group(6) or "").lstrip("=")
        ep_cap = None
        if kind == "P" and cap and to not in self.b:
            ep_cap = (to[0], to[1] - (1 if me == "w" else -1))
        cands = []
        for frm, p in self.b.items():
            if p != me + kind:
                continue
            if dfile and frm[0] != FILES.index(dfile):
                continue
            if drank and frm[1] != int(drank) - 1:
                continue
            if self.can_reach(frm, to, kind, cap):
                cands.append(frm)
        if len(cands) > 1:
            cands = [c for c in cands if self.legal_after(c, to, ep_cap)]
        if len(cands) != 1:
            raise ValueError("ambiguous/illegal SAN %s (%d)" % (san, len(cands)))
        frm = cands[0]
        captured = self.b.get(to)
        self.b.pop(frm)
        if ep_cap:
            self.b.pop(ep_cap)
        self.b[to] = me + (promo if promo else kind)
        # castling rights
        for corner, right in (((0, 0), "Q"), ((7, 0), "K"), ((0, 7), "q"), ((7, 7), "k")):
            if frm == corner or to == corner:
                self.castle.discard(right)
        if kind == "K":
            self.castle -= set("KQ" if me == "w" else "kq")
        self.ep = None
        if kind == "P" and abs(to[1] - frm[1]) == 2:
            self.ep = (frm[0], (frm[1] + to[1]) // 2)
        self.stm = op
        return kind, bool(captured or ep_cap)


# ---------------------------------------------------------------- PGN parsing
TAG = re.compile(r'^\[(\w+) "(.*)"\]$')
SCORE = re.compile(r"^([+-]?)(M?)(\d+(?:\.\d+)?)/(\d+)\s+(\d+(?:\.\d+)?)s")


def parse_pgn(text):
    games, tags, body = [], {}, []
    for line in text.splitlines():
        t = TAG.match(line.strip())
        if t:
            if body:
                games.append((tags, " ".join(body)))
                tags, body = {}, []
            tags[t.group(1)] = t.group(2)
        elif line.strip():
            body.append(line.strip())
    if body:
        games.append((tags, " ".join(body)))
    return games


def tokens(movetext):
    i, n = 0, len(movetext)
    while i < n:
        c = movetext[i]
        if c == "{":
            j = movetext.index("}", i)
            yield "c", movetext[i + 1:j]
            i = j + 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < n and not movetext[j].isspace() and movetext[j] != "{":
                j += 1
            w = movetext[i:j]
            i = j
            if re.match(r"^\d+\.+$", w) or w in ("1-0", "0-1", "1/2-1/2", "*"):
                continue
            w = re.sub(r"^\d+\.+", "", w)
            if w:
                yield "m", w


def parse_score(comment):
    m = SCORE.match(comment.strip())
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    if m.group(2):
        cp = 10000 - int(float(m.group(3)))
    else:
        cp = round(float(m.group(3)) * 100)
    return sign * cp, int(m.group(4)), float(m.group(5))


def load_game(tags, movetext, us):
    """Return per-ply records after the book, scores in OUR POV."""
    if tags.get("White") == us:
        our_col, opp = "w", tags.get("Black")
    elif tags.get("Black") == us:
        our_col, opp = "b", tags.get("White")
    else:
        return None
    res = tags.get("Result")
    pts = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res)
    if pts is None:
        return None
    if our_col == "b":
        pts = 1.0 - pts
    board = Board(tags.get("FEN"))
    plies, pending, term = [], None, ""
    seen = Counter()
    book_plies = 0
    for kind, val in tokens(movetext):
        if kind == "m":
            mover = board.stm
            phase_before, men_before = board.phase(), board.men()
            pk, cap = board.push_san(val)
            seen[board.key()] += 1
            pending = dict(san=val, ours=(mover == our_col), piece=pk, capture=cap,
                           check=val.endswith("+") or val.endswith("#"),
                           phase=phase_before, men=men_before, score=None, depth=None, t=None,
                           rep=seen[board.key()])
            plies.append(pending)
        else:
            if "," in val:
                term = val.split(",", 1)[1].strip()
            if pending is None:
                continue
            if val.strip().startswith("book"):
                book_plies = len(plies)
                pending["book"] = True
                continue
            sc = parse_score(val)
            if sc:
                s, d, t = sc
                if not pending["ours"]:
                    pass
                # fastchess prints the mover's POV; flip to our POV
                pending["score"] = s if pending["ours"] else -s
                pending["depth"], pending["t"] = d, t
    plies = plies[book_plies:]
    tcm = re.match(r"(\d+(?:\.\d+)?)\+(\d+(?:\.\d+)?)", tags.get("TimeControl", ""))
    base, inc = (float(tcm.group(1)), float(tcm.group(2))) if tcm else (0.0, 0.0)
    rem = {True: base, False: base}
    for p in plies:
        if p["t"] is not None:
            rem[p["ours"]] += inc - p["t"]
        p["clk_us"], p["clk_opp"] = rem[True], rem[False]
    order = "KQRBNP"
    final = "v".join("".join(sorted((q[1] for q in board.b.values() if q[0] == c), key=order.index))
                     for c in (our_col, "b" if our_col == "w" else "w"))
    return dict(opp=opp, color=our_col, pts=pts, plies=plies, term=term, final=final,
                tc=tags.get("TimeControl", ""), round=tags.get("Round"), plycount=len(plies))


# ---------------------------------------------------------------- analysis
def phase_name(p):
    return "open" if p >= 20 else ("mg" if p >= 9 else "eg")


def med(xs):
    return st.median(xs) if xs else float("nan")


def mean(xs):
    return st.mean(xs) if xs else float("nan")


def clip(x, c=1000):
    return max(-c, min(c, x))


def analyse_game(g):
    pl = g["plies"]
    ours = [i for i, p in enumerate(pl) if p["ours"] and p["score"] is not None]
    theirs = [i for i, p in enumerate(pl) if not p["ours"] and p["score"] is not None]
    out = {}

    def opp_after(i):  # opponent's next score after ply i
        for j in range(i + 1, len(pl)):
            if not pl[j]["ours"] and pl[j]["score"] is not None:
                return pl[j]["score"]
        return None

    def opp_before(i):
        for j in range(i - 1, -1, -1):
            if not pl[j]["ours"] and pl[j]["score"] is not None:
                return pl[j]["score"]
        return None

    # per-our-move opponent judge loss (positive = our move cost us)
    losses = []
    for i in ours:
        a, b = opp_before(i), opp_after(i)
        if a is None or b is None:
            continue
        if abs(a) >= 300 and abs(b) >= 300:
            loss = None
        else:
            loss = clip(a, 400) - clip(b, 400)
        losses.append((i, loss))
    out["move_losses"] = losses

    def point(th, sign):
        # first of our moves after which the opponent's score stays beyond th for good
        opp_scores = [(i, pl[i]["score"]) for i in theirs]
        if not opp_scores:
            return None
        k = len(opp_scores)
        ok = [False] * k
        good = True
        for j in range(k - 1, -1, -1):
            good = good and (sign * opp_scores[j][1] >= th)
            ok[j] = good
        for j in range(k):
            if ok[j]:
                oi = opp_scores[j][0]
                # our move just before that opponent score
                prev = [i for i in ours if i < oi]
                return prev[-1] if prev else oi
        return None

    def own_point(th, sign):
        k = len(ours)
        good = True
        first = None
        for j in range(k - 1, -1, -1):
            good = good and (sign * pl[ours[j]]["score"] >= th)
            if good:
                first = ours[j]
            else:
                break
        return first

    for name, sign in (("lost", -1), ("won", 1)):
        lp = point(200, sign)
        if lp is None:
            continue
        rec = dict(ply=lp, move=lp // 2 + 1, phase=pl[lp]["phase"], men=pl[lp]["men"],
                   clk_us=pl[lp]["clk_us"], clk_opp=pl[lp]["clk_opp"])
        rec["pname"] = phase_name(rec["phase"])
        bl = [l for i, l in losses if i <= lp and l is not None]
        rec["max_loss"] = max(bl) if bl else 0
        rec["loss_steps"] = bl
        op = own_point(200, sign)
        if op is not None:
            rec["lag"] = (sum(1 for i in ours if lp <= i < op) if op >= lp else
                          -sum(1 for i in ours if op <= i < lp))
        prev = [i for i in ours if i < lp]
        if prev:
            i = prev[-1]
            ob = opp_before(i + 1)
            if ob is not None:
                rec["optimism"] = clip(pl[i]["score"], 2000) - clip(ob, 2000)
                rec["our_before"] = pl[i]["score"]
                rec["opp_before"] = ob
        oa = opp_after(lp)
        if pl[lp]["score"] is not None and oa is not None:
            rec["our_at"], rec["opp_at"] = pl[lp]["score"], oa
        out[name] = rec
    return out


def clock(g, base, inc):
    """Remaining clock after each move for us / them (seconds), approx (book moves free)."""
    rem = {True: base, False: base}
    mins = {True: base, False: base}
    used = {True: [], False: []}
    for p in g["plies"]:
        if p["t"] is None:
            continue
        rem[p["ours"]] += inc - p["t"]
        mins[p["ours"]] = min(mins[p["ours"]], rem[p["ours"]])
        used[p["ours"]].append(p["t"])
    return mins, used


def summarise(games, label, base, inc):
    lines = []
    W = sum(1 for g in games if g["pts"] == 1)
    D = sum(1 for g in games if g["pts"] == 0.5)
    L = sum(1 for g in games if g["pts"] == 0)
    lines.append(f"### {label}: {len(games)} games, +{W} ={D} -{L} ({(W + D / 2) / max(1, len(games)):.1%})")
    an = [(g, analyse_game(g)) for g in games]

    # scale ratio (opp / ours) on same-sign middlegame pairs
    ratios = []
    opp_ratios = defaultdict(list)
    for g in games:
        pl = g["plies"]
        for i in range(len(pl) - 1):
            a, b = pl[i], pl[i + 1]
            if a["score"] is None or b["score"] is None:
                continue
            o, t = (a, b) if a["ours"] else (b, a)
            if not o["ours"] or t["ours"]:
                continue
            lo, hi = sorted((abs(o["score"]), abs(t["score"])))
            if 9 <= o["phase"] and lo >= 50 and hi <= 400 and o["score"] * t["score"] > 0:
                ratios.append(t["score"] / o["score"])
                opp_ratios[g["opp"]].append(t["score"] / o["score"])
    scale = med(ratios) if ratios else 1.0
    oscale = {k: med(v) for k, v in opp_ratios.items()}
    lines.append(f"- scale opp/ours (adjacent same-sign mg pairs, both 0.5-4 pawns): median {scale:.2f} (n={len(ratios)})")

    # side by opponent's first score
    sides = defaultdict(list)
    for g in games:
        first = next((p["score"] for p in g["plies"] if not p["ours"] and p["score"] is not None), None)
        if first is None:
            continue
        s = "better" if first >= 50 else ("worse" if first <= -50 else "level")
        sides[s].append(g)
    parts = []
    for s in ("better", "level", "worse"):
        gs = sides[s]
        if gs:
            w = sum(g["pts"] for g in gs)
            parts.append(f"{s} (opp first score {'>= +0.5' if s=='better' else '<= -0.5' if s=='worse' else 'within 0.5'}) {len(gs)} games {w / len(gs):.1%}")
    lines.append("- side by opponent's first score: " + "; ".join(parts))

    # lost / won points
    for name, res in (("lost", 0.0), ("won", 1.0)):
        recs = []
        for g, a in an:
            if g["pts"] == res and name in a:
                a[name]["scale"] = oscale.get(g["opp"], scale)
                recs.append(a[name])
        tot = sum(1 for g in games if g["pts"] == res)
        if not recs:
            lines.append(f"- {name} point: none ({tot} games with that result)")
            continue
        pc = Counter(r["pname"] for r in recs)
        nobl = sum(1 for r in recs if r["max_loss"] < 150)
        lags = [r["lag"] for r in recs if "lag" in r]
        opt = [r["optimism"] for r in recs if "optimism" in r]
        opt_sc = [clip(r["our_before"], 2000) - clip(r["opp_before"], 2000) / r["scale"] for r in recs if "optimism" in r]
        at = [clip(r["our_at"], 2000) - clip(r["opp_at"], 2000) for r in recs if "our_at" in r]
        at_sc = [clip(r["our_at"], 2000) - clip(r["opp_at"], 2000) / r["scale"] for r in recs if "our_at" in r]
        clk = [(r["clk_us"], r["clk_opp"]) for r in recs]
        steps = [s for r in recs for s in r["loss_steps"] if s > 0]
        lines.append(
            f"- {name} point (opp score {'<= -2' if name=='lost' else '>= +2'} for good) found in {len(recs)}/{tot}: "
            f"open {pc['open']} / mg {pc['mg']} / eg {pc['eg']}; median move after book {med([r['move'] for r in recs])}, "
            f"men {med([r['men'] for r in recs])}; no single move of ours costing >= 1.5 (opp judge) {nobl}/{len(recs)}; "
            f"agree lag median {med(lags)} mean {mean(lags):.1f} our moves; "
            f"ours minus opp (cp, median): move before {med(opt):+.0f} raw / {med(opt_sc):+.0f} rescaled, "
            f"at the point (our move vs opp's next) {med(at):+.0f} raw / {med(at_sc):+.0f} rescaled (opp / its own opp/ours scale); clock at that point us {med([c[0] for c in clk]):.1f}s vs opp {med([c[1] for c in clk]):.1f}s")
        if steps and name == "lost":
            tot_s = sum(steps)
            b = [sum(s for s in steps if lo <= s < hi) / tot_s for lo, hi in ((0, 25), (25, 50), (50, 100), (100, 10 ** 9))]
            lines.append("  - opp-judged loss before the lost point by step size <25/25-50/50-100/>=100: "
                         + " / ".join(f"{x:.0%}" for x in b))
        for g, a in an:
            if g["pts"] == res and name in a and name == "lost":
                r = a[name]
                lines.append(f"  - {g['opp']} r{g['round']} {g['color']}: lost at move {r['move']} ({r['pname']}, phase {r['phase']}), "
                             f"max single loss {r['max_loss']} cp, lag {r.get('lag')}, ours {r.get('our_before')} vs opp {r.get('opp_before')}, "
                             f"plies {g['plycount']}, term '{g['term']}'")

    # draws from winning positions
    draws = [(g, a) for g, a in an if g["pts"] == 0.5]
    peak = []
    for g, a in draws:
        opp_scores = [p["score"] for p in g["plies"] if not p["ours"] and p["score"] is not None]
        our_scores = [p["score"] for p in g["plies"] if p["ours"] and p["score"] is not None]
        peak.append((max(opp_scores or [0]), max(our_scores or [0]), min(opp_scores or [0]), g))
    for th in (100, 150, 200):
        n_opp = sum(1 for po, pu, _, _ in peak if po >= th)
        n_our = sum(1 for po, pu, _, _ in peak if pu >= th)
        n_bad = sum(1 for _, _, mo, _ in peak if mo <= -th)
        lines.append(f"- draws where opp's own score reached >= +{th/100:.1f} for us: {n_opp}/{len(draws)} "
                     f"(ours reached it: {n_our}); opp score <= -{th/100:.1f} (we saved): {n_bad}")
    # where the edge evaporates in better-side draws
    evap = []
    for po, pu, _, g in peak:
        if po < 150:
            continue
        pl = g["plies"]
        idx = max((i for i, p in enumerate(pl) if not p["ours"] and p["score"] is not None),
                  key=lambda i: pl[i]["score"])
        # first opp move after the peak where opp score <= +0.40 for good
        tail = [i for i, p in enumerate(pl) if i > idx and not p["ours"] and p["score"] is not None]
        good, first = True, None
        for i in reversed(tail):
            good = good and pl[i]["score"] <= 40
            if good:
                first = i
            else:
                break
        if first is None:
            continue
        ourprev = [p["score"] for p in pl[:first] if p["ours"] and p["score"] is not None]
        evap.append(dict(peak_move=idx // 2 + 1, peak_phase=pl[idx]["phase"], gone_move=first // 2 + 1,
                         gone_phase=pl[first]["phase"], ours_then=ourprev[-1] if ourprev else None,
                         final_phase=pl[-1]["phase"], term=g["term"]))
    if evap:
        pcp = Counter(phase_name(e["peak_phase"]) for e in evap)
        pcg = Counter(phase_name(e["gone_phase"]) for e in evap)
        lines.append(f"  - of those >= +1.5 draws, edge gone for good (opp <= +0.40) in {len(evap)}: peak at move "
                     f"{med([e['peak_move'] for e in evap])} (open/mg/eg {pcp['open']}/{pcp['mg']}/{pcp['eg']}), gone at move "
                     f"{med([e['gone_move'] for e in evap])} (open/mg/eg {pcg['open']}/{pcg['mg']}/{pcg['eg']}); our score when opp "
                     f"gives up the edge median {med([e['ours_then'] for e in evap if e['ours_then'] is not None])} cp")
    # score trajectories by moves after book, by side
    for s in ("better", "worse"):
        gs = sides.get(s, [])
        if not gs:
            continue
        row = []
        for m in (0, 10, 20, 30, 40):
            us_s, op_s = [], []
            for g in gs:
                pl = g["plies"]
                o = [p["score"] for p in pl[2 * m:2 * m + 2] if p["ours"] and p["score"] is not None]
                t = [p["score"] for p in pl[2 * m:2 * m + 2] if not p["ours"] and p["score"] is not None]
                if o and t:
                    us_s.append(clip(o[0])); op_s.append(clip(t[0]))
            if us_s:
                row.append(f"m{m} {med(op_s):+.0f}|{med(us_s):+.0f}")
        lines.append(f"- {s} side medians, opp|us: " + ", ".join(row))

    # optimism symmetric check: all adjacent pairs, ours minus opp, by opp band
    bands = defaultdict(list)
    for g in games:
        pl = g["plies"]
        for i in range(len(pl) - 1):
            a, b = pl[i], pl[i + 1]
            if a["score"] is None or b["score"] is None or a["ours"] == b["ours"]:
                continue
            o, t = (a, b) if a["ours"] else (b, a)
            if abs(t["score"]) > 600 or t["phase"] < 4:
                continue
            band = int((t["score"] + (50 if t["score"] >= 0 else -50)) / 100)
            band = max(-4, min(4, band))
            bands[band].append(clip(o["score"]) - t["score"])
    lines.append("- ours minus opp score on adjacent plies, by opp score band (pawns): " + ", ".join(
        f"{b:+d}: {med(bands[b]):+.0f} (n={len(bands[b])})" for b in sorted(bands) if len(bands[b]) >= 20))

    # outcome calibration: our points by each engine's score band (our POV)
    cal = {True: defaultdict(list), False: defaultdict(list)}
    for g in games:
        for p in g["plies"]:
            if p["score"] is None or p["phase"] < 4:
                continue
            x = p["score"]
            band = max(-3, min(3, int((x + (50 if x >= 0 else -50)) / 100)))
            cal[p["ours"]][band].append(g["pts"])
    lines.append("- our points by score band (pawns, phase >= 4), us | opp: " + ", ".join(
        f"{b:+d}: {mean(cal[True][b]):.2f}|{mean(cal[False][b]):.2f}" for b in range(-3, 4)
        if len(cal[True][b]) >= 20 and len(cal[False][b]) >= 20))

    # time usage
    for res, nm in ((1.0, "wins"), (0.5, "draws"), (0.0, "losses")):
        gs = [g for g in games if g["pts"] == res]
        if not gs:
            continue
        mins_u, mins_t, first20_u, first20_t, avg_u, avg_t = [], [], [], [], [], []
        for g in gs:
            mins, used = clock(g, base, inc)
            mins_u.append(mins[True]); mins_t.append(mins[False])
            first20_u.append(sum(used[True][:20])); first20_t.append(sum(used[False][:20]))
            avg_u.append(mean(used[True])); avg_t.append(mean(used[False]))
        lines.append(f"- clock in {nm} ({len(gs)}): first 20 moves after book us {med(first20_u):.1f}s vs opp {med(first20_t):.1f}s; "
                     f"mean s/move us {med(avg_u):.2f} vs {med(avg_t):.2f}; min remaining us {med(mins_u):.1f}s (worst {min(mins_u):.1f}) "
                     f"vs opp {med(mins_t):.1f}s (worst {min(mins_t):.1f})")
    # time per move by phase
    tp = defaultdict(lambda: ([], []))
    for g in games:
        for p in g["plies"]:
            if p["t"] is not None:
                tp[phase_name(p["phase"])][0 if p["ours"] else 1].append(p["t"])
    lines.append("- mean s/move by phase us|opp: " + ", ".join(
        f"{k} {mean(tp[k][0]):.2f}|{mean(tp[k][1]):.2f}" for k in ("open", "mg", "eg")))
    # depth by phase
    dp = defaultdict(lambda: ([], []))
    for g in games:
        for p in g["plies"]:
            if p["depth"] is not None:
                dp[phase_name(p["phase"])][0 if p["ours"] else 1].append(p["depth"])
    lines.append("- mean depth by phase us|opp: " + ", ".join(
        f"{k} {mean(dp[k][0]):.1f}|{mean(dp[k][1]):.1f}" for k in ("open", "mg", "eg")))
    # terminations
    tc = Counter((g["pts"], g["term"].replace("White", "X").replace("Black", "X")) for g in games)
    lines.append("- terminations: " + "; ".join(f"{p} {t or '?'} {n}" for (p, t), n in sorted(tc.items())))
    # endgame conversion
    eg_win = Counter()
    for g in games:
        pl = g["plies"]
        hit = False
        for i in range(len(pl) - 1):
            a, b = pl[i], pl[i + 1]
            if a["phase"] <= 8 and a["score"] is not None and b["score"] is not None and a["ours"] != b["ours"]:
                if a["score"] >= 200 and b["score"] >= 200:
                    hit = True
                    break
        if hit:
            eg_win[g["pts"]] += 1
    lines.append(f"- endgames where both say >= +2 for us: won {eg_win[1.0]} / drawn {eg_win[0.5]} / lost {eg_win[0.0]}")
    # our exact 0.00 rate
    z_u = [p for g in games for p in g["plies"] if p["ours"] and p["score"] is not None and p["phase"] >= 4]
    z_t = [p for g in games for p in g["plies"] if not p["ours"] and p["score"] is not None and p["phase"] >= 4]
    lines.append(f"- |score| <= 0.01 on phase >= 4 moves: us {sum(abs(p['score']) <= 1 for p in z_u) / max(1, len(z_u)):.1%}, "
                 f"opp {sum(abs(p['score']) <= 1 for p in z_t) / max(1, len(z_t)):.1%}")
    # opponent-judged loss per our move by phase and type
    by = defaultdict(list)
    for g, a in an:
        pl = g["plies"]
        for i, l in a["move_losses"]:
            if l is None:
                continue
            by[phase_name(pl[i]["phase"])].append(l)
            k = "capture" if pl[i]["capture"] else ("check" if pl[i]["check"] else ("pawn" if pl[i]["piece"] == "P" else ("king" if pl[i]["piece"] == "K" else "piece")))
            by[k].append(l)
    lines.append("- opp-judged mean loss per our move (cp): " + ", ".join(
        f"{k} {mean(by[k]):.1f}" for k in ("open", "mg", "eg", "pawn", "piece", "king", "capture", "check") if by[k]))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", nargs="+")
    ap.add_argument("--us", default="rustfish")
    ap.add_argument("--by-opponent", action="store_true")
    ap.add_argument("--group", action="append", default=[],
                    help="name=opp1,opp2 : extra pooled group")
    args = ap.parse_args()
    games = []
    for path in args.pgn:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for tags, mt in parse_pgn(fh.read()):
                try:
                    g = load_game(tags, mt, args.us)
                except ValueError as e:
                    print("skip:", e, file=sys.stderr)
                    continue
                if g:
                    games.append(g)
    tcm = re.match(r"(\d+(?:\.\d+)?)\+(\d+(?:\.\d+)?)", games[0]["tc"] if games else "")
    base, inc = (float(tcm.group(1)), float(tcm.group(2))) if tcm else (0.0, 0.0)
    print("\n".join(summarise(games, "pooled", base, inc)))
    for spec in args.group:
        name, opps = spec.split("=", 1)
        opps = set(opps.split(","))
        print()
        print("\n".join(summarise([g for g in games if g["opp"] in opps], name, base, inc)))
    if args.by_opponent:
        for opp in sorted({g["opp"] for g in games}):
            print()
            print("\n".join(summarise([g for g in games if g["opp"] == opp], opp, base, inc)))


if __name__ == "__main__":
    main()
