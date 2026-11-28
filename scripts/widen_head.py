#!/usr/bin/env python3
"""Widen a pairwise-head NNUE net file from H1 = 16 to H1 = 32 (stdlib only).

The new net computes exactly the same output as the old one, so an H1 = 32 trainer
(`nnue-trainer --features h1-32`, `nnue-trainer-cuda-cap32`) can warm-start from any
H1 = 16 stage (`--init widened.bin`) instead of retraining the whole chain:

- W1 (H1 x n per output bucket, neuron-major): the 16 old rows are kept; new row k is
  old row k with every entry's sign flipped at random (same magnitudes and grid,
  uncorrelated direction), so the new neurons are alive from step 0;
- b1: the new neurons get the old biases;
- W2 (H2 x H1, `W2[m * H1 + k]`): columns 16..31 are 0, so the new neurons do not
  change the output until training gives them weight;
- everything else (FT, the linear output layer, b2, W3, b3) is copied.

File layout (tools/nnue-trainer/src/lib.rs `Shape`, `head_offs`): i16 words
[FT W: kb*768*H][FT b: H][out W: 2H*buckets][out b: buckets] then per output bucket
[W1: H1*n][b1: H1][W2: H2*H1][b2: H2][W3: H2][b3: 1], n = H (pairwise).
Pairwise-skip nets (neuron 15 is a skip) are not supported.

usage: widen_head.py IN OUT [--hidden 1024] [--buckets 8] [--king-buckets 1] [--seed 1]
(--king-buckets is the number of FT king-bucket blocks in the FILE: 1 for plain and m1
nets, 8 for kb8m, 16 for kb16m.)
"""
import argparse
import random
import struct
import sys

INPUTS, H1_OLD, H1_NEW, H2 = 768, 16, 32, 32


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--king-buckets", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    h, nb, kb = a.hidden, a.buckets, a.king_buckets
    raw = open(a.inp, "rb").read()
    words = list(struct.unpack(f"<{len(raw) // 2}h", raw))
    head_off = kb * INPUTS * h + h + 2 * h * nb + nb
    n = h  # pairwise head input width

    def head_len(h1):
        return h1 * n + h1 + H2 * h1 + H2 + H2 + 1

    want = head_off + nb * head_len(H1_OLD)
    if len(words) != want:
        sys.exit(f"{a.inp}: {len(words)} words, expected {want} for an H1=16 pairwise net "
                 f"(H={h}, buckets={nb}, king buckets in file={kb})")
    rng = random.Random(a.seed)
    out = words[:head_off]
    for b in range(nb):
        o = head_off + b * head_len(H1_OLD)
        w1 = words[o:o + H1_OLD * n]
        b1 = words[o + H1_OLD * n:o + H1_OLD * n + H1_OLD]
        o2 = o + H1_OLD * n + H1_OLD
        w2 = words[o2:o2 + H2 * H1_OLD]
        rest = words[o2 + H2 * H1_OLD:o + head_len(H1_OLD)]  # b2, W3, b3
        new_w1 = [(-x if rng.random() < 0.5 else x) for x in w1]
        out += w1 + new_w1
        out += b1 + b1
        for m in range(H2):
            out += w2[m * H1_OLD:(m + 1) * H1_OLD] + [0] * (H1_NEW - H1_OLD)
        out += rest
    assert len(out) == head_off + nb * head_len(H1_NEW)
    with open(a.out, "wb") as f:
        f.write(struct.pack(f"<{len(out)}h", *out))
    print(f"{a.inp} -> {a.out}: H1 {H1_OLD} -> {H1_NEW}, {len(words)} -> {len(out)} words")


if __name__ == "__main__":
    main()
