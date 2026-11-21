"""Tiny UCI driver. One fresh engine process per search (no state carried between
positions, so a fixed-node search of one binary is reproducible)."""
import re, subprocess

MATE = 3000  # mate scores are clamped to +-3000 cp


def score_cp(line):
    m = re.search(r' score (cp|mate) (-?\d+)', line)
    if not m:
        return None
    v = int(m.group(2))
    if m.group(1) == 'mate':
        return MATE if v > 0 else -MATE
    return max(-MATE, min(MATE, v))


class Engine:
    def __init__(self, path, hash_mb=64, options=(), nice=19):
        self.path, self.hash_mb, self.options, self.nice = path, hash_mb, list(options), nice

    def search(self, fen, go, moves=()):
        """Returns {'bestmove': str, 'rows': {depth: (pv move, cp, nodes)}}."""
        cmd = ['nice', '-n', str(self.nice), self.path] if self.nice else [self.path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        opts = ''.join(f'setoption name {k} value {v}\n' for k, v in
                       [('Hash', self.hash_mb), ('OwnBook', 'false'), ('Threads', 1)] + self.options)
        pos = f'position fen {fen}' + (' moves ' + ' '.join(moves) if moves else '')
        p.stdin.write(f'{opts}isready\n{pos}\ngo {go}\n')
        p.stdin.flush()
        rows, best = {}, None
        for line in p.stdout:
            if line.startswith('info') and ' pv ' in line and ' depth ' in line:
                d = int(re.search(r' depth (\d+)', line).group(1))
                n = int(re.search(r' nodes (\d+)', line).group(1))
                rows[d] = (line.split(' pv ')[1].split()[0], score_cp(line), n)
            elif line.startswith('bestmove'):
                best = line.split()[1]
                break
        try:
            p.stdin.write('quit\n')
            p.stdin.flush()
        except BrokenPipeError:
            pass
        p.wait()
        return {'bestmove': best, 'rows': rows}

    def close(self):
        pass
