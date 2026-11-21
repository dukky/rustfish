"""Minimal stdlib chess board: FEN in/out, legal moves, SAN -> UCI, make move.
Only what the suite builder needs (PGN replay); speed is irrelevant."""

FILES = 'abcdefgh'
N_DIRS = (-17, -15, -10, -6, 6, 10, 15, 17)
K_DIRS = (-9, -8, -7, -1, 1, 7, 8, 9)
B_DIRS = (-9, -7, 7, 9)
R_DIRS = (-8, -1, 1, 8)


def sq(name):
    return FILES.index(name[0]) + 8 * (int(name[1]) - 1)


def name(s):
    return FILES[s % 8] + str(s // 8 + 1)


def _step_ok(a, b):
    # a -> b is a single king/knight step without wrapping across the board edge
    return 0 <= b < 64 and abs(a % 8 - b % 8) <= 2


class Board:
    def __init__(self, fen):
        parts = fen.split()
        self.b = [None] * 64
        rows = parts[0].split('/')
        for r, row in enumerate(rows):
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    self.b[(7 - r) * 8 + f] = ch
                    f += 1
        self.white = parts[1] == 'w'
        self.castle = '' if parts[2] == '-' else parts[2]
        self.ep = None if parts[3] == '-' else sq(parts[3])
        self.half = int(parts[4]) if len(parts) > 4 else 0
        self.full = int(parts[5]) if len(parts) > 5 else 1

    def fen(self, full=True):
        rows = []
        for r in range(7, -1, -1):
            row, e = '', 0
            for f in range(8):
                p = self.b[r * 8 + f]
                if p is None:
                    e += 1
                else:
                    if e:
                        row += str(e)
                        e = 0
                    row += p
            if e:
                row += str(e)
            rows.append(row)
        ep = '-'
        if self.ep is not None:
            # only emit ep if a capture is actually possible (keeps FENs canonical)
            pawn = 'P' if self.white else 'p'
            d = -8 if self.white else 8
            for df in (-1, 1):
                s = self.ep + d + df
                if 0 <= s < 64 and abs(s % 8 - self.ep % 8) == 1 and self.b[s] == pawn:
                    ep = name(self.ep)
        s = f"{'/'.join(rows)} {'w' if self.white else 'b'} {self.castle or '-'} {ep}"
        return s + (f' {self.half} {self.full}' if full else '')

    def own(self, p, white):
        return p is not None and (p.isupper() == white)

    def attacked(self, s, by_white):
        b = self.b
        # pawns
        if by_white:
            for d in (-7, -9):
                t = s + d
                if 0 <= t < 64 and abs(t % 8 - s % 8) == 1 and b[t] == 'P':
                    return True
        else:
            for d in (7, 9):
                t = s + d
                if 0 <= t < 64 and abs(t % 8 - s % 8) == 1 and b[t] == 'p':
                    return True
        n, k = ('N', 'K') if by_white else ('n', 'k')
        for d in N_DIRS:
            t = s + d
            if _step_ok(s, t) and b[t] == n:
                return True
        for d in K_DIRS:
            t = s + d
            if _step_ok(s, t) and abs(t % 8 - s % 8) <= 1 and b[t] == k:
                return True
        bq = ('B', 'Q') if by_white else ('b', 'q')
        rq = ('R', 'Q') if by_white else ('r', 'q')
        for dirs, pcs in ((B_DIRS, bq), (R_DIRS, rq)):
            for d in dirs:
                t = s
                while True:
                    nt = t + d
                    if not (0 <= nt < 64) or abs(nt % 8 - t % 8) > 1:
                        break
                    t = nt
                    if b[t] is not None:
                        if b[t] in pcs:
                            return True
                        break
        return False

    def king_sq(self, white):
        k = 'K' if white else 'k'
        return self.b.index(k)

    def in_check(self):
        return self.attacked(self.king_sq(self.white), not self.white)

    def pseudo(self):
        b, w = self.b, self.white
        out = []
        for s in range(64):
            p = b[s]
            if p is None or p.isupper() != w:
                continue
            u = p.upper()
            if u == 'P':
                d = 8 if w else -8
                start, last = (1, 7) if w else (6, 0)
                t = s + d
                if 0 <= t < 64 and b[t] is None:
                    self._pawn_add(out, s, t, last)
                    if s // 8 == start and b[t + d] is None:
                        out.append((s, t + d, None))
                for df in (-1, 1):
                    t = s + d + df
                    if 0 <= t < 64 and abs(t % 8 - s % 8) == 1:
                        if (b[t] is not None and not self.own(b[t], w)) or t == self.ep:
                            self._pawn_add(out, s, t, last)
            elif u in 'NK':
                for d in (N_DIRS if u == 'N' else K_DIRS):
                    t = s + d
                    if _step_ok(s, t) and (u == 'N' or abs(t % 8 - s % 8) <= 1) and not self.own(b[t], w):
                        out.append((s, t, None))
            else:
                dirs = B_DIRS if u == 'B' else R_DIRS if u == 'R' else B_DIRS + R_DIRS
                for d in dirs:
                    t = s
                    while True:
                        nt = t + d
                        if not (0 <= nt < 64) or abs(nt % 8 - t % 8) > 1:
                            break
                        t = nt
                        if b[t] is None:
                            out.append((s, t, None))
                        else:
                            if not self.own(b[t], w):
                                out.append((s, t, None))
                            break
        # castling
        if w and b[4] == 'K':
            if 'K' in self.castle and b[5] is None and b[6] is None and b[7] == 'R' and not any(self.attacked(x, False) for x in (4, 5, 6)):
                out.append((4, 6, None))
            if 'Q' in self.castle and b[3] is None and b[2] is None and b[1] is None and b[0] == 'R' and not any(self.attacked(x, False) for x in (4, 3, 2)):
                out.append((4, 2, None))
        if not w and b[60] == 'k':
            if 'k' in self.castle and b[61] is None and b[62] is None and b[63] == 'r' and not any(self.attacked(x, True) for x in (60, 61, 62)):
                out.append((60, 62, None))
            if 'q' in self.castle and b[59] is None and b[58] is None and b[57] is None and b[56] == 'r' and not any(self.attacked(x, True) for x in (60, 59, 58)):
                out.append((60, 58, None))
        return out

    @staticmethod
    def _pawn_add(out, s, t, last):
        if t // 8 == last:
            for pr in 'qrbn':
                out.append((s, t, pr))
        else:
            out.append((s, t, None))

    def push(self, m):
        s, t, pr = m
        b = self.b
        p = b[s]
        undo = (list(b), self.white, self.castle, self.ep, self.half, self.full)
        cap = b[t] is not None
        if p.upper() == 'P' and t == self.ep:
            b[t + (-8 if self.white else 8)] = None
            cap = True
        b[t] = (pr.upper() if self.white else pr) if pr else p
        b[s] = None
        if p.upper() == 'K' and abs(t - s) == 2:
            if t == s + 2:
                b[s + 1], b[s + 3] = b[s + 3], None
            else:
                b[s - 1], b[s - 4] = b[s - 4], None
        self.ep = (s + t) // 2 if p.upper() == 'P' and abs(t - s) == 16 else None
        for x, c in ((0, 'Q'), (7, 'K'), (56, 'q'), (63, 'k')):
            if s == x or t == x:
                self.castle = self.castle.replace(c, '')
        if s == 4:
            self.castle = self.castle.replace('K', '').replace('Q', '')
        if s == 60:
            self.castle = self.castle.replace('k', '').replace('q', '')
        self.half = 0 if cap or p.upper() == 'P' else self.half + 1
        if not self.white:
            self.full += 1
        self.white = not self.white
        return undo

    def pop(self, undo):
        self.b, self.white, self.castle, self.ep, self.half, self.full = undo
        self.b = list(self.b)

    def legal(self):
        res = []
        for m in self.pseudo():
            u = self.push(m)
            ok = not self.attacked(self.king_sq(not self.white), self.white)
            self.pop(u)
            if ok:
                res.append(m)
        return res

    @staticmethod
    def uci(m):
        return name(m[0]) + name(m[1]) + (m[2] or '')

    def parse_san(self, san):
        s = san.rstrip('+#!?')
        legal = self.legal()
        if s in ('O-O', '0-0'):
            cand = [m for m in legal if self.b[m[0]].upper() == 'K' and m[1] - m[0] == 2]
        elif s in ('O-O-O', '0-0-0'):
            cand = [m for m in legal if self.b[m[0]].upper() == 'K' and m[0] - m[1] == 2]
        else:
            pr = None
            if '=' in s:
                s, pr = s.split('=')
                pr = pr.lower()
            piece = s[0] if s[0] in 'NBRQK' else 'P'
            body = s[1:] if piece != 'P' else s
            body = body.replace('x', '')
            dest = sq(body[-2:])
            dis = body[:-2]
            cand = []
            for m in legal:
                if m[1] != dest or self.b[m[0]].upper() != piece or (m[2] or None) != pr:
                    continue
                fr = name(m[0])
                if all((c in FILES and fr[0] == c) or (c.isdigit() and fr[1] == c) for c in dis):
                    cand.append(m)
        if len(cand) != 1:
            raise ValueError(f'bad SAN {san} in {self.fen()}: {cand}')
        return cand[0]

    def from_uci(self, u):
        for m in self.legal():
            if self.uci(m) == u:
                return m
        raise ValueError(f'illegal {u} in {self.fen()}')

    def material_phase(self):
        """Non-pawn material phase 0..24 (N,B=1, R=2, Q=4)."""
        w = {'n': 1, 'b': 1, 'r': 2, 'q': 4}
        return min(24, sum(w.get(p.lower(), 0) for p in self.b if p))
