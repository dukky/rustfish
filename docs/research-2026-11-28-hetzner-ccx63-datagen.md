# Hetzner CCX63 datagen trial — 2026-11-28

**Date:** 2026-11-28
**Status:** completed; data packed and downloaded; instance deleted.
**Main at writing:** `bd8e38b`; measured ladder 3640.5 (95% CI 3624.6–3656.5).
**Purpose:** measure whether rented cloud CPU is a cost-effective way to grow
the 8.54B pool toward 30B, and at what wall-clock and cost.

## Setup

- Hetzner Cloud **CCX63**, Helsinki (`hel1`): 48 vCPU (24 physical AMD
  EPYC-Milan cores at ~2.2 GHz), 184 GB RAM, 901 GB NVMe.
- Hourly billing, no setup fee or monthly commitment: **EUR 1.16–1.37/h**.
- Prebuilt `rustfish` with `RUSTFLAGS="-C target-cpu=x86-64-v3"` (the host is
  AVX2, not AVX-512), plus the `nnue-trainer` binary for packing. No Rust
  toolchain or repo checkout was needed on the server.
- Run: 48 threads, 5,000 nodes, `softnodes=5000`, seed 12345:

```bash
datagen out-12345.txt 100000000 5000 48 12345 0 softnodes=5000
```

The engine was stopped after 5,044 s (84m04s).

## Result

- 587,500 games, **55,085,699 positions**
- **10,924 pos/s**, **116.47 games/s**
- Depth **7.77** at **4,998 nodes/move**
- W/D/L **182,663 / 220,786 / 184,051**
- Text intermediate: 3.4 GB
- Packed `.rfp`: **174 MB** (181,618,368 bytes), **3.297 bytes/position**,
  276 chunks, 1.07% keyframes; pack time 66.9 s, 0 unparsed lines, 0
  invalid records.
- SHA-256:
  `e81b5e9c47205cec9f2d3af6f7b87da749d182ea804e36793d2e6e8ff837704e`

The downloaded artifact was kept at
`/home/duk/datagen/hetzner-2026-11-28/out-12345.rfp` on `dev`; the Hetzner
instance was deleted after the hash was verified on both ends.

## Findings

- One CCX63 is only **2.2–2.8x the 9600X**, not 10–20x. The nominal 48 vCPU
  are 24 low-clock physical cores; SMT adds little for this workload.
- 55.1M positions is **3.4–3.9h of one desktop** (8–12 threads), ~1.7h of two,
  or ~1.1h of three. The rental bought a couple of hours of wall-clock, not a
  compute-class change.
- Cost: **~EUR 31–36 per 1B positions**. Desktop marginal electricity is
  ~GBP 2.50/1B; a fully amortised desktop is ~GBP 4–6/1B. Cloud is ~5–7x the
  loaded desktop cost and ~12x the marginal electricity cost.
- A 30B total pool (3.5x the current 8.54B) is ~**EUR 640** and 3–6 days on
  4–7 CCX63s. Based on gen12's +17% data → +2.87 LTC and the expected decay,
  that is only **+5–12 LTC** on the current 1024/8 net: a bad trade against
  label depth, DFRC/diversity or capacity.
- A few seeds is a cheap top-up (~1M games / ~93M positions, ~2.4h and ~EUR 3
  per seed on one CCX63). Treat 30B as a background reservoir for a later
  capacity net, not a sprint.
- For a genuine 10–20x burst, **2 high-clock 64-vCPU spot instances**
  (AWS `c7a.16xlarge` / GCP `c3-highcpu-64`) are a better price/performance
  bet than 4–7 low-clock CCX63s; datagen is shardable and restartable, so
  spot interruption is tolerable.

## Decision

No further CCX63 rental for volume. Spend the next data effort on **20k
labels / a 5k+20k mix, DFRC and opening diversity, and a capacity net**
before growing the pool again. The 55.1M positions are a small top-up, not a
generation's worth of data.

## Reproduction notes

- Build: `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --offline`
  and `cargo build --release --manifest-path tools/nnue-trainer/Cargo.toml`.
- Pack: `nnue-trainer pack --chunk 200000 out.rfp out.txt`.
- Verify with `sha256sum` on both ends before deleting the instance.
- Full operational detail (access, cleanup, watcher pitfalls) is in the
  `rustfish-merge-queue` skill's `references/rented-datagen.md`.
