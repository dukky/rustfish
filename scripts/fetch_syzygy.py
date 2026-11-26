#!/usr/bin/env python3
"""Download and verify the Syzygy 3-4-5 tablebases (WDL + DTZ, 290 files, ~945 MB).

  python3 scripts/fetch_syzygy.py <dir>

Files come from the lichess mirror and are checked against
scripts/syzygy-345.md5 (made from a verified copy that matched the mirror's
own md5 list). Files already present with the right md5 are kept, so a rerun
only fetches what is missing or broken. A few parallel connections, retries
with backoff; exits nonzero unless every file verifies. Used by the CI
workflows (sprt.yml, ccrl-ladder.yml) on a cache miss; stdlib only.
"""
import concurrent.futures
import hashlib
import os
import sys
import time
import urllib.request

MIRROR = "https://tablebase.lichess.ovh/tables/standard"
SUMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "syzygy-345.md5")
WORKERS = 4
ATTEMPTS = 6


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, want, out_dir):
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest) and md5_of(dest) == want:
        return name, "kept"
    sub = "3-4-5-wdl" if name.endswith(".rtbw") else "3-4-5-dtz"
    url = f"{MIRROR}/{sub}/{name}"
    part = dest + ".part"
    err = None
    for attempt in range(ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rustfish-ci-fetch"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            got = md5_of(part)
            if got == want:
                os.replace(part, dest)
                return name, "fetched"
            err = f"md5 {got} != {want}"
        except Exception as exc:  # network errors, timeouts, HTTP errors
            err = str(exc)
        time.sleep(5 * 2 ** attempt)
    if os.path.exists(part):
        os.unlink(part)
    return name, f"FAILED: {err}"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    sums = []
    with open(SUMS) as fh:
        for line in fh:
            if line.strip():
                want, name = line.split()
                sums.append((name, want))
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda s: fetch(s[0], s[1], out_dir), sums))
    bad = [f"{n}: {r}" for n, r in results if r.startswith("FAILED")]
    fetched = sum(r == "fetched" for _, r in results)
    total = sum(os.path.getsize(os.path.join(out_dir, n)) for n, r in results if not r.startswith("FAILED"))
    print(f"syzygy 3-4-5: {len(sums) - len(bad)}/{len(sums)} files ok ({fetched} fetched), "
          f"{total / 1e6:.0f} MB, {time.time() - t0:.0f} s")
    if bad:
        sys.exit("failed:\n" + "\n".join(bad))


if __name__ == "__main__":
    main()
