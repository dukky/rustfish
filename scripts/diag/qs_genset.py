"""Build an exchange-heavy position set from 'fen F moves m1 m2 ...' lines.

Every position right after a capture (class R: a recapture may be pending) and
the final position of each line (class F) is emitted as
'<class> <id> fen <F> moves <prefix>' so the move history (repetitions) is kept.
"""
import sys

def parse_fen(fen):
    parts = fen.split()
    sq = {}
    rows = parts[0].split('/')
    for r, row in enumerate(rows):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                sq[(f, 7 - r)] = ch
                f += 1
    return sq, parts[1]

def apply(sq, mv):
    ff, fr = ord(mv[0]) - 97, int(mv[1]) - 1
    tf, tr = ord(mv[2]) - 97, int(mv[3]) - 1
    p = sq.pop((ff, fr))
    cap = (tf, tr) in sq
    if p in 'Pp' and ff != tf and not cap:  # en passant
        sq.pop((tf, fr), None)
        cap = True
    if p in 'Kk' and abs(tf - ff) == 2:
        if tf == 6:
            sq[(5, fr)] = sq.pop((7, fr))
        else:
            sq[(3, fr)] = sq.pop((0, fr))
    if len(mv) == 5:
        p = mv[4].upper() if p.isupper() else mv[4]
    sq[(tf, tr)] = p
    return cap

out = []
for path in sys.argv[1:]:
    tag = path.rsplit('/', 1)[-1].split('.')[0]
    for li, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        fen = line[4:].split(' moves ')[0]
        moves = line.split(' moves ')[1].split() if ' moves ' in line else []
        sq, _ = parse_fen(fen)
        for i, m in enumerate(moves):
            cap = apply(sq, m)
            if cap and i + 1 < len(moves) and i >= 3:
                out.append(f"R {tag}-{li}-{i+1} fen {fen} moves {' '.join(moves[:i+1])}")
        out.append(f"F {tag}-{li}-{len(moves)} fen {fen} moves {' '.join(moves)}")
print('\n'.join(out))
