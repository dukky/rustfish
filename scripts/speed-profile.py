#!/usr/bin/env python3
"""Source-level time map of a rustfish perf profile.

    perf record -F 4999 -o perf.data -- <non-PIE binary with debug line info> ...
    scripts/speed-profile.py <binary> perf.data [--top 60]

lto + codegen-units=1 inline almost everything into negamax/quiescence, so
the real symbol perf sees says little. This resolves each sampled address
with `addr2line -i` (the whole inline chain) and reports:
  * self time by innermost rustfish source function (skipping core/std frames),
  * self time by rustfish source file and by file:line,
  * "inclusive" time per rustfish function over the inline chain (a function
    counts once per sample if it appears anywhere in the chain; real calls
    between out-of-line functions are not followed),
  * self time by the real (out-of-line) symbol.
The binary must not be PIE (build with -C relocation-model=static) so that
perf's sample addresses are ELF virtual addresses.
"""
import argparse
import collections
import re
import subprocess


def short(fn):
    fn = re.sub(r"::h[0-9a-f]{16}$", "", fn)
    fn = re.sub(r"<[^<>]*>", "<>", fn)
    fn = re.sub(r"<[^<>]*>", "<>", fn)
    return fn


def ours(path):
    return "/src/" in path and "/rustc/" not in path and "/.cargo/" not in path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("perfdata")
    ap.add_argument("--top", type=int, default=60)
    a = ap.parse_args()

    script = subprocess.run(["perf", "script", "-i", a.perfdata, "-F", "ip,sym", "--no-demangle"],
                            capture_output=True, text=True).stdout
    ips = collections.Counter()
    syms = collections.Counter()
    for line in script.splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            ip = int(parts[0], 16)
        except ValueError:
            continue
        ips[ip] += 1
        syms[short(parts[1]) if len(parts) > 1 else "?"] += 1
    total = sum(ips.values())
    uniq = sorted(ips)
    out = subprocess.run(["addr2line", "-a", "-f", "-i", "-C", "-e", a.binary],
                         input="".join(f"{ip:#x}\n" for ip in uniq), capture_output=True, text=True).stdout
    chains = {}
    cur = None
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if re.fullmatch(r"0x[0-9a-f]+", l):
            cur = int(l, 16)
            chains[cur] = []
            i += 1
            continue
        fn = l
        loc = lines[i + 1] if i + 1 < len(lines) else "??:0"
        loc = re.sub(r" \(discriminator \d+\)", "", loc)
        chains.setdefault(cur, []).append((short(fn), loc))
        i += 2

    self_fn = collections.Counter()
    self_file = collections.Counter()
    self_line = collections.Counter()
    incl = collections.Counter()
    for ip, n in ips.items():
        ch = chains.get(ip, [])
        mine = [(f, loc) for f, loc in ch if ours(loc.split(":")[0])]
        if mine:
            f, loc = mine[0]
            path, _, ln = loc.rpartition(":")
            fname = path[path.rfind("/src/") + 1:]
            self_fn[f] += n
            self_file[fname] += n
            self_line[f"{fname}:{ln} ({f.split('::')[-1]})"] += n
        else:
            self_fn["<no rustfish frame> " + (ch[0][0] if ch else "?")] += n
        for f in {f for f, _ in mine}:
            incl[f] += n

    def table(title, c, top):
        print(f"\n## {title}\n")
        for k, n in c.most_common(top):
            print(f"{100 * n / total:6.2f}%  {k}")

    print(f"samples: {total}")
    table("self by rustfish source function (innermost src frame)", self_fn, a.top)
    table("self by source file", self_file, 30)
    table("self by file:line", self_line, a.top)
    table("inclusive over the inline chain", incl, a.top)
    table("self by real symbol", syms, 25)


if __name__ == "__main__":
    main()
