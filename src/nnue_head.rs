//! Pairwise NNUE head (docs/nnue-deeper-head.md, `nnue-trainer --head
//! pairwise` / `--head pairwise-skip`): per output bucket,
//! `H pairwise products -> 16 -> 32 -> 1`, added to the SCReLU linear
//! output (or, with `SKIP`, replacing it). Compiled in with
//! `nnue::HEAD = true`; with the default (`false`) nothing here is reachable
//! and the net format is unchanged.
//!
//! Trainer semantics (all float, `a` = the accumulator in units of QA,
//! the trainer's forward is quantisation-aware, so it computes exactly the
//! integer function below up to float rounding):
//!
//! ```text
//! per perspective (stm first), i < H/2:
//!   x[i]  = round(clamp(a[i], 0, 1) * clamp(a[i + H/2], 0, 1) * QX)   0..=127, QX = 65025/512
//! z1 = b1 + (W1q · x) / (QX * 64)   (16)      W1q = round(64 W1) in -127..=127
//! z2 = b2 + W2 · sq(z1[..k])        (32)      sq(z) = clamp(z, 0, 1)^2, k = 16 (15 with SKIP)
//! out_head = b3 + W3 · sq(z2)  (+ z1[15] with SKIP)
//! eval_cp = 400 * (linear output + out_head)  (SKIP: the output bias only)
//! ```
//!
//! File: after the linear output biases, per bucket `W1 [16][H]`, b1 [16],
//! `W2 [32][16]`, b2 [32], W3 [32], b3, as i16 at `HEAD_Q` = 8192 (W1 is
//! a multiple of 1/64 there: `W1q * 128`).
//!
//! Inference:
//!
//! - L1 input: `x = (c[i] * c[i + H/2] + 256) >> 9`, `c` = the i16
//!   accumulator clamped to 0..=255 (u8, at most 127). SIMD: `mulhrs` /
//!   `sqrdmulh` of `c << 6` and `c`, which is that rounding exactly.
//! - Sparse L1 in i32: the input is scanned in 4-byte chunks and only
//!   nonzero chunks are multiplied: NEON `sdot`, AVX2 `maddubs` + `madd`
//!   (pair sums <= 2 * 127 * 127 fit i16: no saturation), AVX-VNNI /
//!   AVX512-VNNI `vpdpbusd` (runtime detected). All exact, so every target
//!   computes the same z1 bit for bit.
//! - L2 / L3 in f32 (like the trainer), fixed operation order (no fma
//!   contraction in Rust): the same on every target.
//! - Final: `round(400 * (linear + head))` to the nearest centipawn.

use super::{BUCKETS, H, QA, QB, SCALE};

/// Nets from `--head pairwise-skip` (no linear output; L1 neuron 15 is
/// the skip). Must match the net: the file sizes are the same.
pub(crate) const SKIP: bool = false;

pub(crate) const H1: usize = 16;
pub(crate) const H2: usize = 32;
/// Export scale of every head parameter in the trainer's net file.
const HEAD_Q: f32 = 8192.0;
/// Pairwise inputs per perspective, and in all.
const HALF: usize = H / 2;
const N_IN: usize = H;
/// 4-byte chunks of the L1 input.
pub(crate) const CHUNKS: usize = N_IN / 4;
/// i16 words per bucket in the file.
const BUCKET_WORDS: usize = H1 * N_IN + H1 + H2 * H1 + H2 + H2 + 1;
/// i16 words the head adds to the net file.
pub(crate) const HEAD_WORDS: usize = BUCKET_WORDS * BUCKETS;
/// z1 = Σ x w / Z1_DIV (x at 65025/512, w at 64).
const Z1_DIV: f32 = 65025.0 / 512.0 * 64.0;
/// L1 neurons that feed L2.
const KA: usize = if SKIP { H1 - 1 } else { H1 };

/// Storage position of L1 input byte `i` (and of its W1 column). On x86
/// the AVX2 kernels store `packus` output without the lane-fixing
/// permute, so within every 32 bytes the 4-byte chunks sit in the order
/// 0, 1, 4, 5, 2, 3, 6, 7 (an involution); W1 is laid out to match and the
/// scalar twins use the same layout. The L1 sums are integer, so the order
/// of the chunks changes nothing. Identity elsewhere.
#[inline(always)]
pub(crate) const fn slot(i: usize) -> usize {
    if cfg!(target_arch = "x86_64") {
        const P: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
        i / 32 * 32 + P[i / 4 % 8] * 4 + i % 4
    } else {
        i
    }
}

/// W1 of one bucket, chunk-major: `.0[c*64 + k*4 + j]` = W1q[k][4c + j],
/// so a chunk's weights are one 64-byte line.
#[repr(C, align(64))]
#[derive(Clone)]
struct W1Bucket([i8; CHUNKS * H1 * 4]);

#[repr(C, align(64))]
pub(crate) struct Input(pub(crate) [u8; N_IN]);

/// The L1 input and its nonzero-chunk list (not zeroed per eval).
pub(crate) struct Scratch {
    pub(crate) x: Input,
    pub(crate) nnz: [u16; NNZ_BUF],
    pub(crate) n: usize,
}

impl Scratch {
    pub(crate) const fn new() -> Self {
        Scratch { x: Input([0; N_IN]), nnz: [0; NNZ_BUF], n: 0 }
    }
}

thread_local! {
    static SCRATCH: std::cell::UnsafeCell<Scratch> = const { std::cell::UnsafeCell::new(Scratch::new()) };
}

#[derive(Clone)]
pub(crate) struct HeadNet {
    w1: Vec<W1Bucket>,
    b1: [[f32; H1]; BUCKETS],
    /// Transposed, `w2[b][k][m]` = W2[m][k], so L2 is 16 broadcast-multiply-adds.
    w2: [[[f32; H2]; H1]; BUCKETS],
    b2: [[f32; H2]; BUCKETS],
    w3: [[f32; H2]; BUCKETS],
    b3: [f32; BUCKETS],
    /// The file words, for `to_bytes`.
    raw: Vec<i16>,
}

impl HeadNet {
    /// From the trainer's `HEAD_WORDS` i16 words (at `HEAD_Q`).
    pub(crate) fn from_words(raw: Vec<i16>) -> Self {
        assert_eq!(raw.len(), HEAD_WORDS);
        let mut h = Self {
            w1: vec![W1Bucket([0; CHUNKS * H1 * 4]); BUCKETS],
            b1: [[0.0; H1]; BUCKETS],
            w2: [[[0.0; H2]; H1]; BUCKETS],
            b2: [[0.0; H2]; BUCKETS],
            w3: [[0.0; H2]; BUCKETS],
            b3: [0.0; BUCKETS],
            raw: Vec::new(),
        };
        let f = |v: i16| v as f32 / HEAD_Q;
        for b in 0..BUCKETS {
            let mut it = raw[b * BUCKET_WORDS..(b + 1) * BUCKET_WORDS].iter().copied();
            for k in 0..H1 {
                for i in 0..N_IN {
                    // W1q * 128 in the file (the trainer exports quantised W1);
                    // round anyway, so a non-QAT net still loads.
                    let v = it.next().unwrap() as i32;
                    let w = ((v + if v >= 0 { 64 } else { -64 }) / 128).clamp(-127, 127);
                    let p = slot(i);
                    h.w1[b].0[(p / 4) * H1 * 4 + k * 4 + p % 4] = w as i8;
                }
            }
            for k in 0..H1 {
                h.b1[b][k] = f(it.next().unwrap());
            }
            for m in 0..H2 {
                for k in 0..H1 {
                    h.w2[b][k][m] = f(it.next().unwrap());
                }
            }
            for m in 0..H2 {
                h.b2[b][m] = f(it.next().unwrap());
            }
            for m in 0..H2 {
                h.w3[b][m] = f(it.next().unwrap());
            }
            h.b3[b] = f(it.next().unwrap());
            debug_assert!(it.next().is_none());
        }
        h.raw = raw;
        h
    }

    #[allow(dead_code)] // to_bytes (tests)
    pub(crate) fn words(&self) -> &[i16] {
        &self.raw
    }

    /// The head's output for bucket `b`, in trainer units (1.0 = 400 cp).
    #[inline]
    pub(crate) fn forward(&self, b: usize, us: &[i16; H], them: &[i16; H]) -> f32 {
        SCRATCH.with(|s| {
            // SAFETY: the only borrow of this thread's scratch; forward
            // does not re-enter.
            let s = unsafe { &mut *s.get() };
            let n = kernels::activate(us, &mut s.x.0[..HALF], 0, &mut s.nnz, 0);
            let n = kernels::activate(them, &mut s.x.0[HALF..], HALF / 4, &mut s.nnz, n);
            self.l1_rest(b, &s.x.0, &s.nnz[..n])
        })
    }

    /// `forward` on an input already in `s` (the fused update's output).
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn forward_scratch(&self, b: usize, s: &Scratch) -> f32 {
        self.l1_rest(b, &s.x.0, &s.nnz[..s.n])
    }

    /// L1 on `x` (nonzero chunks `nnz`), then `rest`. On x86 one kernel
    /// per L1 flavour does both, so z1 stays in registers.
    #[inline(always)]
    fn l1_rest(&self, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        return kernels::l1_rest(self, b, x, nnz);
        #[cfg(not(target_arch = "x86_64"))]
        {
            let z1 = kernels::l1(self, b, x, nnz);
            self.rest_fast(b, &z1)
        }
    }

    /// Scalar twin of `forward` (tests).
    #[cfg(test)]
    fn forward_scalar(&self, b: usize, us: &[i16; H], them: &[i16; H]) -> f32 {
        let mut x = Input([0; N_IN]);
        let mut nnz = [0u16; NNZ_BUF];
        let n = scalar::activate(us, &mut x.0[..HALF], 0, &mut nnz, 0);
        let n = scalar::activate(them, &mut x.0[HALF..], HALF / 4, &mut nnz, n);
        let z1 = scalar::l1(self, b, &x.0, &nnz[..n]);
        self.rest(b, &z1)
    }

    /// `rest`, explicitly vectorised where available (bit-identical; the
    /// fixture test checks `forward` against `forward_scalar`).
    #[cfg(not(target_arch = "x86_64"))]
    #[inline(always)]
    fn rest_fast(&self, b: usize, z1: &[i32; H1]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        return self.rest_neon(b, z1);
        #[cfg(not(target_arch = "aarch64"))]
        return self.rest(b, z1);
    }

    /// `rest` with explicit AVX2: the same f32 operations in the same
    /// order (separate mul and add, never fused), so bit-identical.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)] // tests
    unsafe fn rest_avx2(&self, b: usize, z1: &[i32; H1]) -> f32 {
        use std::arch::x86_64::*;
        // SAFETY: AVX2 is enabled; in-bounds unaligned loads.
        unsafe {
            let p = z1.as_ptr() as *const __m256i;
            self.rest_avx2_v(b, [_mm256_loadu_si256(p), _mm256_loadu_si256(p.add(1))])
        }
    }

    /// `rest_avx2` on z1 in two registers (callers enable AVX2).
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn rest_avx2_v(&self, b: usize, z1: [std::arch::x86_64::__m256i; 2]) -> f32 {
        use std::arch::x86_64::*;
        // SAFETY: fixed-size arrays, all accesses in bounds (unaligned loads).
        unsafe {
            let (zero, one) = (_mm256_setzero_ps(), _mm256_set1_ps(1.0));
            let inv = _mm256_set1_ps(1.0 / Z1_DIV);
            let mut u = [zero; H1 / 8];
            for (q, u) in u.iter_mut().enumerate() {
                let zi = z1[q];
                let z = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(zi), inv), _mm256_loadu_ps(self.b1[b].as_ptr().add(8 * q)));
                let c = _mm256_min_ps(_mm256_max_ps(z, zero), one);
                *u = _mm256_mul_ps(c, c);
            }
            // u[k] broadcast in registers (128-bit half duplicated, then an
            // in-lane shuffle), not a store and 16 reloads: shorter latency
            // from u to the first L2 product.
            let d = [
                _mm256_permute2f128_ps::<0x00>(u[0], u[0]),
                _mm256_permute2f128_ps::<0x11>(u[0], u[0]),
                _mm256_permute2f128_ps::<0x00>(u[1], u[1]),
                _mm256_permute2f128_ps::<0x11>(u[1], u[1]),
            ];
            let mut uk = [zero; H1];
            for (q, &h) in d.iter().enumerate() {
                uk[4 * q] = _mm256_shuffle_ps::<0x00>(h, h);
                uk[4 * q + 1] = _mm256_shuffle_ps::<0x55>(h, h);
                uk[4 * q + 2] = _mm256_shuffle_ps::<0xAA>(h, h);
                uk[4 * q + 3] = _mm256_shuffle_ps::<0xFF>(h, h);
            }
            let w2 = &self.w2[b];
            let mut z2 = [zero; H2 / 8];
            for (r, z) in z2.iter_mut().enumerate() {
                *z = _mm256_loadu_ps(self.b2[b].as_ptr().add(8 * r));
            }
            for k in 0..KA {
                for (r, z) in z2.iter_mut().enumerate() {
                    *z = _mm256_add_ps(*z, _mm256_mul_ps(_mm256_loadu_ps(w2[k].as_ptr().add(8 * r)), uk[k]));
                }
            }
            let mut t = [zero; H2 / 8];
            for r in 0..H2 / 8 {
                let c = _mm256_min_ps(_mm256_max_ps(z2[r], zero), one);
                t[r] = _mm256_mul_ps(_mm256_loadu_ps(self.w3[b].as_ptr().add(8 * r)), _mm256_mul_ps(c, c));
            }
            // The scalar tree: t[m] += t[m + w] for w = 16, 8 (whole
            // vectors), then 4, 2, 1 inside the first vector.
            let v = _mm256_add_ps(_mm256_add_ps(t[0], t[2]), _mm256_add_ps(t[1], t[3]));
            let q = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let h = _mm_add_ps(q, _mm_movehl_ps(q, q));
            let s = _mm_cvtss_f32(h) + _mm_cvtss_f32(_mm_shuffle_ps::<1>(h, h));
            let mut out = self.b3[b] + s;
            if SKIP {
                out += _mm256_extract_epi32::<7>(z1[1]) as f32 * (1.0 / Z1_DIV) + self.b1[b][H1 - 1];
            }
            out
        }
    }

    /// `rest` with explicit NEON: the same f32 operations in the same
    /// order (no fma), so bit-identical.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn rest_neon(&self, b: usize, z1: &[i32; H1]) -> f32 {
        use std::arch::aarch64::*;
        // SAFETY: fixed-size arrays, all accesses in bounds.
        unsafe {
            let (zero, one) = (vdupq_n_f32(0.0), vdupq_n_f32(1.0));
            let mut u = [0.0f32; H1];
            let inv = vdupq_n_f32(1.0 / Z1_DIV);
            for q in 0..H1 / 4 {
                let z = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(z1.as_ptr().add(4 * q))), inv), vld1q_f32(self.b1[b].as_ptr().add(4 * q)));
                let c = vminq_f32(vmaxq_f32(z, zero), one);
                vst1q_f32(u.as_mut_ptr().add(4 * q), vmulq_f32(c, c));
            }
            let w2 = &self.w2[b];
            let mut z2 = [vdupq_n_f32(0.0); H2 / 4];
            for (r, z) in z2.iter_mut().enumerate() {
                *z = vld1q_f32(self.b2[b].as_ptr().add(4 * r));
            }
            for k in 0..KA {
                let uk = vdupq_n_f32(u[k]);
                for (r, z) in z2.iter_mut().enumerate() {
                    *z = vaddq_f32(*z, vmulq_f32(vld1q_f32(w2[k].as_ptr().add(4 * r)), uk));
                }
            }
            let mut t = [vdupq_n_f32(0.0); H2 / 4];
            for r in 0..H2 / 4 {
                let c = vminq_f32(vmaxq_f32(z2[r], zero), one);
                t[r] = vmulq_f32(vld1q_f32(self.w3[b].as_ptr().add(4 * r)), vmulq_f32(c, c));
            }
            // The scalar tree: t[m] += t[m + w] for w = 16, 8, 4 (whole
            // vectors), then 2 and 1 inside the first vector.
            for r in 0..4 {
                t[r] = vaddq_f32(t[r], t[r + 4]);
            }
            for r in 0..2 {
                t[r] = vaddq_f32(t[r], t[r + 2]);
            }
            let v = vaddq_f32(t[0], t[1]);
            let v2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
            let s = vget_lane_f32::<0>(v2) + vget_lane_f32::<1>(v2);
            let mut out = self.b3[b] + s;
            if SKIP {
                out += z1[H1 - 1] as f32 * (1.0 / Z1_DIV) + self.b1[b][H1 - 1];
            }
            out
        }
    }

    /// b1, the L1 activation, L2 and L3 in f32 (fixed operation order).
    #[inline(always)]
    #[allow(dead_code)] // the reference on aarch64 (forward_scalar)
    fn rest(&self, b: usize, z1: &[i32; H1]) -> f32 {
        let mut z = [0.0f32; H1];
        for k in 0..H1 {
            z[k] = z1[k] as f32 * (1.0 / Z1_DIV) + self.b1[b][k];
        }
        let w2 = &self.w2[b];
        let mut z2 = self.b2[b];
        // (Skipping inactive neurons, ~8 of 16 on trained nets, measured no
        // faster: the dense loop vectorises.)
        for k in 0..KA {
            let c = z[k].clamp(0.0, 1.0);
            let u = c * c;
            for m in 0..H2 {
                z2[m] += w2[k][m] * u;
            }
        }
        let mut t = [0.0f32; H2];
        for m in 0..H2 {
            let c = z2[m].clamp(0.0, 1.0);
            t[m] = self.w3[b][m] * (c * c);
        }
        // Fixed pairwise tree (vectorises; same result everywhere).
        let mut w = H2 / 2;
        while w > 0 {
            for m in 0..w {
                t[m] += t[m + w];
            }
            w /= 2;
        }
        let mut out = self.b3[b] + t[0];
        if SKIP {
            out += z[H1 - 1];
        }
        out
    }
}

/// The nonzero-chunk index list, with room for an 8-index store past the end.
pub(crate) const NNZ_BUF: usize = CHUNKS + 8;

/// For an 8-bit mask of nonzero chunks, their positions (0..8) packed
/// first (unused slots 0), and how many there are.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))] // x86 uses kernels::push8z
static NNZ_LUT8: [[u16; 8]; 256] = {
    let mut t = [[0u16; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut j, mut n) = (0, 0);
        while j < 8 {
            if m >> j & 1 == 1 {
                t[m][n] = j as u16;
                n += 1;
            }
            j += 1;
        }
        m += 1;
    }
    t
};
#[cfg_attr(target_arch = "x86_64", allow(dead_code))] // x86 uses kernels::push8z
static POP8: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut m = 0;
    while m < 256 {
        t[m] = (m as u32).count_ones() as u8;
        m += 1;
    }
    t
};

/// Append the nonzero chunks among `mask` (8 bits, chunks `c0..c0 + 8`):
/// one table load, one 8-lane add and one 16-byte store, no branches.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))] // x86 uses kernels::push8z
#[inline(always)]
pub(crate) fn push8(nnz: &mut [u16; NNZ_BUF], n: usize, c0: usize, mask: u32) -> usize {
    let m = (mask & 0xFF) as usize;
    let mut v = NNZ_LUT8[m];
    for x in v.iter_mut() {
        *x += c0 as u16;
    }
    debug_assert!(n + 8 <= NNZ_BUF);
    // SAFETY: n <= CHUNKS, so n + 8 <= NNZ_BUF.
    unsafe { (nnz.as_mut_ptr().add(n) as *mut [u16; 8]).write_unaligned(v) };
    n + POP8[m] as usize
}

/// Linear output sum (`output_sum`, 0 with `SKIP`) + head output ->
/// clamped centipawns, replacing `nnue::finish` when the head is on,
/// rounded to the nearest centipawn like the trainer's float eval.
#[inline(always)]
pub(crate) fn finish(lin_b: i16, sum: i64, head: f32, limit: i32) -> i32 {
    let lin = (sum / QA as i64 + lin_b as i64) as f64 / (QA as i64 * QB as i64) as f64;
    let y = (lin + head as f64) * SCALE as f64;
    // y.round() (half away from zero) as trunc(y + copysign(0.5 - 2^-54, y)),
    // truncated by the conversion itself, then clamped in integers: the
    // same integer as round + f64 clamp + convert (this is how x86 lowers
    // `round` anyway; `finish_matches_reference` checks ties and near-ties),
    // with a shorter dependency chain from the head output to the score.
    let t = y + (0.5f64 - f64::EPSILON / 4.0).copysign(y);
    debug_assert!(t.is_finite() && t.abs() < 1e15);
    // SAFETY: y is finite and small (|lin| < 2^47, |head| is bounded by
    // the L3 weights), so t is in i64 range.
    let r = unsafe { t.to_int_unchecked::<i64>() };
    r.clamp(-limit as i64, limit as i64) as i32
}

/// The original `finish` (tests).
#[cfg(test)]
fn finish_reference(lin_b: i16, sum: i64, head: f32, limit: i32) -> i32 {
    let lin = (sum / QA as i64 + lin_b as i64) as f64 / (QA as i64 * QB as i64) as f64;
    let raw = ((lin + head as f64) * SCALE as f64).round();
    raw.clamp(-limit as f64, limit as f64) as i32
}

/// Scalar twins (the reference; the only kernels on other targets).
#[allow(dead_code)]
pub(crate) mod scalar {
    use super::*;

    /// One perspective's pairwise input into `out` (H/2 bytes), and its
    /// nonzero 4-byte chunks (numbered from `base`) appended to `nnz[n..]`;
    /// returns the new count.
    #[inline(always)]
    pub(crate) fn activate(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], mut n: usize) -> usize {
        for i in 0..HALF {
            let a = (acc[i] as i32).clamp(0, 255);
            let b = (acc[i + HALF] as i32).clamp(0, 255);
            out[slot(i)] = ((a * b + 256) >> 9) as u8;
        }
        for c in 0..HALF / 4 {
            if out[4 * c..4 * c + 4] != [0; 4] {
                nnz[n] = (base + c) as u16;
                n += 1;
            }
        }
        n
    }

    /// The incremental update `out = src + Σ adds - Σ subs` (wrapping), with
    /// `DOT` its SCReLU output sum against `w` (|w| <= 127, as
    /// `nnue::simd::update_dot`), and the pairwise input of `out` (as
    /// `activate`) in the same pass; returns the sum (0 without `DOT`) and
    /// the new nonzero-chunk count.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn update_pw<const DOT: bool>(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
        x: &mut [u8],
        base: usize,
        nnz: &mut [u16; NNZ_BUF],
        n: usize,
    ) -> (i64, usize) {
        let mut sum = 0i64;
        for i in 0..H {
            let mut v = src[i];
            for a in adds {
                v = v.wrapping_add(a[i]);
            }
            for s in subs {
                v = v.wrapping_sub(s[i]);
            }
            out[i] = v;
            if DOT {
                let c = (v as i32).clamp(0, 255);
                sum += (c * w[i] as i32 * c) as i64;
            }
        }
        (sum, activate(out, x, base, nnz, n))
    }

    #[inline(always)]
    pub(crate) fn l1(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        let w = &net.w1[b].0;
        let mut z = [0i32; H1];
        for &c in nnz {
            let c = c as usize;
            for k in 0..H1 {
                let mut s = 0i32;
                for j in 0..4 {
                    s += x[4 * c + j] as i32 * w[c * H1 * 4 + k * 4 + j] as i32;
                }
                z[k] += s;
            }
        }
        z
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod kernels {
    use super::*;
    use std::arch::aarch64::*;

    #[inline(always)]
    pub(crate) fn activate(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], mut n: usize) -> usize {
        assert!(out.len() == HALF && HALF % 64 == 0);
        // SAFETY: HALF % 64 == 0, all loads/stores in bounds.
        unsafe {
            let p = acc.as_ptr();
            // u8 clamp of 16 lanes (saturating narrow = clamp(0, 255)).
            let c = |o: usize| vqmovun_high_s16(vqmovun_s16(vld1q_s16(p.add(o))), vld1q_s16(p.add(o + 8)));
            // Chunk-nonzero bits enter at the top (lane l: bit 28 + l) and
            // move down 4 per group (sri), so 4 groups end at bit 16 + 4g + l.
            let bits = vld1q_u32([1u32 << 28, 2 << 28, 4 << 28, 8 << 28].as_ptr());
            // 64 inputs (16 chunks) per step; one horizontal add for the mask.
            for i in (0..HALF).step_by(64) {
                let mut m = vdupq_n_u32(0);
                for g in 0..4 {
                    let at = i + 16 * g;
                    let (a, b) = (c(at), c(at + HALF));
                    // (ab + 256) >> 9 = rounding ((ab >> 1) >> 8): ab + 256 is odd when ab is.
                    let lo = vrshrn_n_u16::<8>(vshrq_n_u16::<1>(vmull_u8(vget_low_u8(a), vget_low_u8(b))));
                    let v = vrshrn_high_n_u16::<8>(lo, vshrq_n_u16::<1>(vmull_high_u8(a, b)));
                    vst1q_u8(out.as_mut_ptr().add(at), v);
                    let v32 = vreinterpretq_u32_u8(v);
                    m = vsriq_n_u32::<4>(vandq_u32(vtstq_u32(v32, v32), bits), m);
                }
                let mask = vaddvq_u32(m) >> 16;
                n = push8(nnz, n, base + i / 4, mask);
                n = push8(nnz, n, base + i / 4 + 8, mask >> 8);
            }
        }
        n
    }

    /// `scalar::update_pw` in one pass over lanes `i` and `i + H/2` (the
    /// pairs), with the update's values still in registers.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn update_pw<const DOT: bool>(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
        x: &mut [u8],
        base: usize,
        nnz: &mut [u16; NNZ_BUF],
        n: usize,
    ) -> (i64, usize) {
        // Fixed row counts, so the row loops unroll.
        match (adds.len(), subs.len()) {
            (1, 1) => update_pw_n::<DOT, 1, 1>(out, src, adds, subs, w, x, base, nnz, n),
            (1, 2) => update_pw_n::<DOT, 1, 2>(out, src, adds, subs, w, x, base, nnz, n),
            (2, 2) => update_pw_n::<DOT, 2, 2>(out, src, adds, subs, w, x, base, nnz, n),
            _ => scalar::update_pw::<DOT>(out, src, adds, subs, w, x, base, nnz, n),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn update_pw_n<const DOT: bool, const NA: usize, const NS: usize>(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
        x: &mut [u8],
        base: usize,
        nnz: &mut [u16; NNZ_BUF],
        mut n: usize,
    ) -> (i64, usize) {
        assert!(x.len() == HALF && HALF % 64 == 0 && adds.len() == NA && subs.len() == NS);
        let (adds, subs) = (&adds[..NA], &subs[..NS]);
        // SAFETY: every offset is < H - 32 (lanes) or < HALF - 16 (bytes).
        unsafe {
            let (zero, qa) = (vdupq_n_s16(0), vdupq_n_s16(255));
            // Chunk-nonzero bits enter at the top (lane l: bit 28 + l) and
            // move down 4 per group (sri), so 4 groups end at bit 16 + 4g + l.
            let bits = vld1q_u32([1u32 << 28, 2 << 28, 4 << 28, 8 << 28].as_ptr());
            let mut s = [vdupq_n_s32(0); 8];
            let (o, sp) = (out.as_mut_ptr(), src.as_ptr());
            let mut masks = [vdupq_n_u32(0); HALF / 64];
            // 16 pairs per step (two vectors of each half: few live registers).
            // Single-register loads/stores, not the `_x2` forms: `ld1 {a, b}`
            // has no immediate offset, so the unrolled loop computed every
            // address separately; these become `ldp`/`stp` off one base.
            let ld2 = |p: *const i16| [vld1q_s16(p), vld1q_s16(p.add(8))];
            let delta2 = |at: usize| {
                let mut v = ld2(sp.add(at));
                for a in adds {
                    let x = ld2(a.as_ptr().add(at));
                    v = [vaddq_s16(v[0], x[0]), vaddq_s16(v[1], x[1])];
                }
                for r in subs {
                    let x = ld2(r.as_ptr().add(at));
                    v = [vsubq_s16(v[0], x[0]), vsubq_s16(v[1], x[1])];
                }
                vst1q_s16(o.add(at), v[0]);
                vst1q_s16(o.add(at + 8), v[1]);
                v
            };
            for i in (0..HALF).step_by(64) {
                let mut m = vdupq_n_u32(0);
                for g in 0..4 {
                    let at = i + 16 * g;
                    let lo = delta2(at);
                    let hi = delta2(at + HALF);
                    let v = if DOT {
                        let (wl, wh) = (ld2(w.as_ptr().add(at)), ld2(w.as_ptr().add(at + HALF)));
                        let mut cs = [zero; 4];
                        for (j, (v, wv)) in [(lo[0], wl[0]), (lo[1], wl[1]), (hi[0], wh[0]), (hi[1], wh[1])].into_iter().enumerate() {
                            let c = vminq_s16(vmaxq_s16(v, zero), qa);
                            cs[j] = c;
                            let cw = vmulq_s16(c, wv);
                            s[2 * j] = vmlal_s16(s[2 * j], vget_low_s16(cw), vget_low_s16(c));
                            s[2 * j + 1] = vmlal_high_s16(s[2 * j + 1], cw, c);
                        }
                        // The pairwise input from the same clamped lanes:
                        // sqrdmulh(c << 6, c') = (128 c c' + 2^15) >> 16
                        // = (c c' + 256) >> 9, no saturation (c, c' <= 255).
                        let p0 = vqrdmulhq_s16(vshlq_n_s16::<6>(cs[0]), cs[2]);
                        let p1 = vqrdmulhq_s16(vshlq_n_s16::<6>(cs[1]), cs[3]);
                        vuzp1q_u8(vreinterpretq_u8_s16(p0), vreinterpretq_u8_s16(p1))
                    } else {
                        let a = vqmovun_high_s16(vqmovun_s16(lo[0]), lo[1]);
                        let b = vqmovun_high_s16(vqmovun_s16(hi[0]), hi[1]);
                        let l = vrshrn_n_u16::<8>(vshrq_n_u16::<1>(vmull_u8(vget_low_u8(a), vget_low_u8(b))));
                        vrshrn_high_n_u16::<8>(l, vshrq_n_u16::<1>(vmull_high_u8(a, b)))
                    };
                    vst1q_u8(x.as_mut_ptr().add(at), v);
                    let v32 = vreinterpretq_u32_u8(v);
                    m = vsriq_n_u32::<4>(vandq_u32(vtstq_u32(v32, v32), bits), m);
                }
                masks[i / 64] = m;
            }
            // The chunk list after the lane loop, off its critical path.
            for (j, &m) in masks.iter().enumerate() {
                let mask = vaddvq_u32(m) >> 16;
                n = push8(nnz, n, base + 16 * j, mask);
                n = push8(nnz, n, base + 16 * j + 8, mask >> 8);
            }
            let sum = if DOT { s.iter().map(|&v| vaddlvq_s32(v)).sum() } else { 0 };
            (sum, n)
        }
    }

    #[inline(always)]
    fn dotprod() -> bool {
        if cfg!(target_feature = "dotprod") {
            return true;
        }
        static HAVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *HAVE.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
    }

    #[inline(always)]
    pub(crate) fn l1(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        if dotprod() {
            // SAFETY: FEAT_DotProd is available.
            unsafe { l1_sdot(&net.w1[b].0, x, nnz) }
        } else {
            scalar::l1(net, b, x, nnz)
        }
    }

    /// `acc += Σ_j x[4l+j] * w[4l+j]` per i32 lane (x <= 127, so signed).
    #[inline(always)]
    unsafe fn sdot(mut acc: int32x4_t, x: int8x16_t, w: int8x16_t) -> int32x4_t {
        // SAFETY: register-only instruction; callers enable dotprod.
        unsafe {
            std::arch::asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) acc, in(vreg) x, in(vreg) w, options(pure, nomem, nostack));
        }
        acc
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn l1_sdot(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: every chunk index is < CHUNKS, so all loads are in bounds.
        unsafe {
            // Two chunks in flight: 8 accumulators hide the sdot latency.
            let z = vdupq_n_s32(0);
            let mut acc = [[z; 4]; 2];
            let xp = x.as_ptr() as *const u32;
            let wp = w.as_ptr();
            let mut i = 0;
            while i + 1 < nnz.len() {
                for s in 0..2 {
                    let c = nnz[i + s] as usize;
                    let xv = vreinterpretq_s8_u32(vdupq_n_u32(xp.add(c).read_unaligned()));
                    let wv = vld1q_s8_x4(wp.add(c * 64));
                    let wv = [wv.0, wv.1, wv.2, wv.3];
                    for r in 0..4 {
                        acc[s][r] = sdot(acc[s][r], xv, wv[r]);
                    }
                }
                i += 2;
            }
            if i < nnz.len() {
                let c = nnz[i] as usize;
                let xv = vreinterpretq_s8_u32(vdupq_n_u32(xp.add(c).read_unaligned()));
                for r in 0..4 {
                    acc[0][r] = sdot(acc[0][r], xv, vld1q_s8(wp.add(c * 64 + 16 * r)));
                }
            }
            let mut out = [0i32; H1];
            for r in 0..4 {
                vst1q_s32(out.as_mut_ptr().add(4 * r), vaddq_s32(acc[0][r], acc[1][r]));
            }
            out
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod kernels {
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    pub(crate) fn avx2() -> bool {
        if cfg!(target_feature = "avx2") {
            return true;
        }
        static HAVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *HAVE.get_or_init(|| is_x86_feature_detected!("avx2"))
    }

    /// 0 = AVX2 maddubs, 1 = AVX-VNNI, 2 = AVX512-VNNI (+VL).
    #[inline(always)]
    fn vnni() -> u8 {
        static HAVE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        *HAVE.get_or_init(|| {
            if std::env::var("RUSTFISH_NO_VNNI").is_ok() {
                0
            } else if is_x86_feature_detected!("avxvnni") {
                1
            } else if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") {
                2
            } else {
                0
            }
        })
    }

    #[inline(always)]
    pub(crate) fn activate(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], n: usize) -> usize {
        if avx2() {
            // SAFETY: AVX2 is available.
            unsafe { activate_avx2(acc, out, base, nnz, n) }
        } else {
            scalar::activate(acc, out, base, nnz, n)
        }
    }

    #[allow(dead_code)] // tests (the engine uses l1_rest)
    #[inline(always)]
    pub(crate) fn l1(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        if !avx2() {
            return scalar::l1(net, b, x, nnz);
        }
        let w = &net.w1[b].0;
        // SAFETY: AVX2 (and, per branch, the VNNI flavour) is available.
        unsafe {
            match vnni() {
                1 => l1_v1(w, x, nnz),
                2 => l1_v2(w, x, nnz),
                _ => l1_v0(w, x, nnz),
            }
        }
    }

    /// `net.rest(b, &l1(net, b, x, nnz))`, one kernel per L1 flavour.
    #[inline(always)]
    pub(crate) fn l1_rest(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> f32 {
        if !avx2() {
            return net.rest(b, &scalar::l1(net, b, x, nnz));
        }
        // SAFETY: AVX2 (and, per branch, the VNNI flavour) is available.
        unsafe {
            match vnni() {
                1 => l1_rest_v1(net, b, x, nnz),
                2 => l1_rest_v2(net, b, x, nnz),
                _ => l1_rest_v0(net, b, x, nnz),
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn l1_rest_v0(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> f32 {
        // SAFETY: AVX2 is enabled.
        unsafe { net.rest_avx2_v(b, l1_avx2::<0>(&net.w1[b].0, x, nnz)) }
    }

    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn l1_rest_v1(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> f32 {
        // SAFETY: AVX-VNNI is enabled.
        unsafe { net.rest_avx2_v(b, l1_avx2::<1>(&net.w1[b].0, x, nnz)) }
    }

    #[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl")]
    unsafe fn l1_rest_v2(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> f32 {
        // SAFETY: AVX512-VNNI + VL are enabled.
        unsafe { net.rest_avx2_v(b, l1_avx2::<2>(&net.w1[b].0, x, nnz)) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn activate_avx2(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], mut n: usize) -> usize {
        assert!(out.len() == HALF && HALF % 32 == 0);
        // SAFETY: HALF % 32 == 0, all loads/stores in bounds.
        unsafe {
            let (lo, hi) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
            let p = acc.as_ptr();
            let c = |o: usize| _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(o) as *const __m256i), lo), hi);
            let mut bv = _mm_set1_epi16(base as i16);
            for i in (0..HALF).step_by(32) {
                // mulhrs(a << 6, b) = (64ab + 2^14) >> 15 = (ab + 256) >> 9.
                let x0 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(c(i)), c(i + HALF));
                let x1 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(c(i + 16)), c(i + 16 + HALF));
                let v = _mm256_packus_epi16(x0, x1); // chunks in `slot` order
                _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, v);
                n = push8z(nnz, n, bv, zero_mask(v));
                bv = _mm_add_epi16(bv, _mm_set1_epi16(8));
            }
        }
        n
    }

    /// `scalar::update_pw` in one pass over lanes `i` and `i + H/2`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn update_pw<const DOT: bool>(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
        x: &mut [u8],
        base: usize,
        nnz: &mut [u16; NNZ_BUF],
        n: usize,
    ) -> (i64, usize) {
        if !avx2() {
            return scalar::update_pw::<DOT>(out, src, adds, subs, w, x, base, nnz, n);
        }
        match pw512() {
            // SAFETY: pw512() checked AVX-512F/BW (and VNNI for flavour 2).
            1 | 2 if HALF % 64 == 0 => unsafe {
                macro_rules! go512 {
                    ($f:ident) => {
                        match (adds.len(), subs.len()) {
                            (1, 1) => return $f::<DOT, 1, 1>(out, src, adds, subs, w, x, base, nnz, n),
                            (1, 2) => return $f::<DOT, 1, 2>(out, src, adds, subs, w, x, base, nnz, n),
                            (2, 2) => return $f::<DOT, 2, 2>(out, src, adds, subs, w, x, base, nnz, n),
                            _ => {}
                        }
                    };
                }
                if pw512() == 2 {
                    go512!(update_pw512_vnni)
                } else {
                    go512!(update_pw512)
                }
            },
            _ => {}
        }
        // SAFETY: AVX2 is available.
        unsafe {
            macro_rules! go {
                ($f:ident) => {
                    match (adds.len(), subs.len()) {
                        (1, 1) => $f::<DOT, 1, 1>(out, src, adds, subs, w, x, base, nnz, n),
                        (1, 2) => $f::<DOT, 1, 2>(out, src, adds, subs, w, x, base, nnz, n),
                        (2, 2) => $f::<DOT, 2, 2>(out, src, adds, subs, w, x, base, nnz, n),
                        _ => scalar::update_pw::<DOT>(out, src, adds, subs, w, x, base, nnz, n),
                    }
                };
            }
            // The output sum's multiply-accumulate is one vpdpwssd with VNNI.
            match if DOT { vnni() } else { 0 } {
                1 => go!(update_pw_v1),
                2 => go!(update_pw_v2),
                _ => go!(update_pw_avx2),
            }
        }
    }

    macro_rules! update_pw_flavour {
        ($name:ident, $feat:literal, $v:literal) => {
            #[allow(clippy::too_many_arguments)]
            #[target_feature(enable = $feat)]
            unsafe fn $name<const DOT: bool, const NA: usize, const NS: usize>(
                out: &mut [i16; H],
                src: &[i16; H],
                adds: &[&[i16; H]],
                subs: &[&[i16; H]],
                w: &[i16; H],
                x: &mut [u8],
                base: usize,
                nnz: &mut [u16; NNZ_BUF],
                n: usize,
            ) -> (i64, usize) {
                // SAFETY: the features of flavour $v are enabled.
                unsafe { update_pw_body::<DOT, NA, NS, $v>(out, src, adds, subs, w, x, base, nnz, n) }
            }
        };
    }
    update_pw_flavour!(update_pw_avx2, "avx2", 0);
    update_pw_flavour!(update_pw_v1, "avx2,avxvnni", 1);
    update_pw_flavour!(update_pw_v2, "avx2,avx512f,avx512vnni,avx512vl", 2);

    /// Which 512-bit fused update to use: 0 = none (the 256-bit kernels),
    /// 1 = AVX-512BW, 2 = AVX-512BW + VNNI. On only where 512-bit vectors
    /// are full width and measured faster: AMD Zen 5 and later (family
    /// >= 0x1A; Zen 4 splits them in two and measured -2.9%). Intel is
    /// left on the 256-bit kernels (+0.4% on Granite Rapids, and older
    /// AVX-512 Xeons lower the clock for 512-bit work).
    /// `RUSTFISH_PW512=0` / `=1` overrides the CPU choice (A/B tests);
    /// `RUSTFISH_NO_AVX512` turns it off with the other AVX-512 kernels.
    #[inline(always)]
    pub(crate) fn pw512() -> u8 {
        static HAVE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        *HAVE.get_or_init(|| {
            if !crate::nnue::avx512::enabled() {
                return 0;
            }
            let on = match std::env::var("RUSTFISH_PW512").ok().as_deref() {
                Some("0") => false,
                Some("1") => true,
                _ => amd_family() >= 0x1A,
            };
            if !on {
                0
            } else if is_x86_feature_detected!("avx512vnni") && std::env::var("RUSTFISH_NO_VNNI").is_err() {
                2
            } else {
                1
            }
        })
    }

    /// The CPU family on AMD (0x19 = Zen 3/4, 0x1A = Zen 5), else 0.
    fn amd_family() -> u32 {
        // SAFETY: cpuid is available on every x86-64 CPU.
        let (v, f) = unsafe { (__cpuid(0), __cpuid(1)) };
        let amd = (v.ebx, v.edx, v.ecx) == (0x6874_7541, 0x6974_6e65, 0x444d_4163); // "AuthenticAMD"
        let base = (f.eax >> 8) & 0xF;
        let fam = if base == 0xF { base + ((f.eax >> 20) & 0xFF) } else { base };
        if amd {
            fam
        } else {
            0
        }
    }

    update_pw512_flavour!(update_pw512, "avx2,avx512f,avx512bw", false);
    update_pw512_flavour!(update_pw512_vnni, "avx2,avx512f,avx512bw,avx512vnni", true);

    /// `sum += madd(a, b)` (exact i32 pair sums): AVX2, AVX-VNNI or AVX512-VNNI vpdpwssd.
    #[inline(always)]
    unsafe fn dpwssd<const V: u8>(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: the update_pw flavour of V enables its features.
        unsafe {
            match V {
                1 => dw1(acc, a, b),
                2 => dw2(acc, a, b),
                _ => _mm256_add_epi32(acc, _mm256_madd_epi16(a, b)),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn dw1(mut acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: register-only instruction; AVX-VNNI is enabled.
        unsafe { std::arch::asm!("{{vex}} vpdpwssd {0}, {1}, {2}", inout(ymm_reg) acc, in(ymm_reg) a, in(ymm_reg) b, options(pure, nomem, nostack)) };
        acc
    }

    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl")]
    unsafe fn dw2(mut acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: register-only instruction; AVX512-VNNI + VL are enabled.
        unsafe { std::arch::asm!("{{evex}} vpdpwssd {0}, {1}, {2}", inout(ymm_reg) acc, in(ymm_reg) a, in(ymm_reg) b, options(pure, nomem, nostack)) };
        acc
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn update_pw_body<const DOT: bool, const NA: usize, const NS: usize, const V: u8>(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
        x: &mut [u8],
        base: usize,
        nnz: &mut [u16; NNZ_BUF],
        mut n: usize,
    ) -> (i64, usize) {
        assert!(x.len() == HALF && HALF % 32 == 0 && adds.len() == NA && subs.len() == NS);
        // SAFETY: AVX2 per caller; offsets are < H - 16 (lanes) / HALF - 32 (bytes).
        unsafe {
            let (lo, hi) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
            let ld = |p: *const i16| _mm256_loadu_si256(p as *const __m256i);
            let mut delta = |o: usize| {
                let mut v = ld(src.as_ptr().add(o));
                for a in &adds[..NA] {
                    v = _mm256_add_epi16(v, ld(a.as_ptr().add(o)));
                }
                for s in &subs[..NS] {
                    v = _mm256_sub_epi16(v, ld(s.as_ptr().add(o)));
                }
                _mm256_storeu_si256(out.as_mut_ptr().add(o) as *mut __m256i, v);
                _mm256_min_epi16(_mm256_max_epi16(v, lo), hi)
            };
            let mut sum = [_mm256_setzero_si256(); 4];
            let mut bv = _mm_set1_epi16(base as i16);
            for i in (0..HALF).step_by(32) {
                let (a0, a1) = (delta(i), delta(i + 16));
                let (b0, b1) = (delta(i + HALF), delta(i + HALF + 16));
                if DOT {
                    // c in [0, 255], |w| <= 127: c * w fits i16, madd sums two exact i32 products.
                    for (k, (c, o)) in [(a0, i), (a1, i + 16), (b0, i + HALF), (b1, i + HALF + 16)].into_iter().enumerate() {
                        let cw = _mm256_mullo_epi16(c, ld(w.as_ptr().add(o)));
                        // Four sums: with VNNI each is a dependent vpdpwssd chain.
                        sum[k] = dpwssd::<V>(sum[k], cw, c);
                    }
                }
                // mulhrs(a << 6, b) = (64ab + 2^14) >> 15 = (ab + 256) >> 9.
                let x0 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(a0), b0);
                let x1 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(a1), b1);
                let v = _mm256_packus_epi16(x0, x1); // chunks in `slot` order
                _mm256_storeu_si256(x.as_mut_ptr().add(i) as *mut __m256i, v);
                n = push8z(nnz, n, bv, zero_mask(v));
                bv = _mm_add_epi16(bv, _mm_set1_epi16(8));
            }
            let sum = [_mm256_add_epi32(sum[0], sum[2]), _mm256_add_epi32(sum[1], sum[3])];
            let total = if DOT {
                // i32 lanes -> i64 (each lane sums at most 2 * H / 16 products of <= 255^2 * 127).
                let s = _mm256_add_epi64(
                    _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum[0])),
                    _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(sum[0])),
                );
                let t = _mm256_add_epi64(
                    _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum[1])),
                    _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(sum[1])),
                );
                let mut l = [0i64; 4];
                _mm256_storeu_si256(l.as_mut_ptr() as *mut __m256i, _mm256_add_epi64(s, t));
                l.iter().sum()
            } else {
                0
            };
            (total, n)
        }
    }

    /// `update_pw_body` with 512-bit vectors: the same integer operations
    /// lane for lane (wrapping i16 update, exact i32 products, the `mulhrs`
    /// pairwise rounding), 64 lanes and 16 chunks per step, stored in the
    /// same `slot` order, so the output, the nonzero-chunk list (same
    /// order) and the sum are identical to the 256-bit kernels.
    macro_rules! update_pw512_flavour {
        ($name:ident, $feat:literal, $vnni:literal) => {
            #[allow(clippy::too_many_arguments)]
            #[target_feature(enable = $feat)]
            unsafe fn $name<const DOT: bool, const NA: usize, const NS: usize>(
                out: &mut [i16; H],
                src: &[i16; H],
                adds: &[&[i16; H]],
                subs: &[&[i16; H]],
                w: &[i16; H],
                x: &mut [u8],
                base: usize,
                nnz: &mut [u16; NNZ_BUF],
                mut n: usize,
            ) -> (i64, usize) {
                assert!(x.len() == HALF && HALF % 64 == 0 && adds.len() == NA && subs.len() == NS);
                // SAFETY: the flavour's features are enabled; offsets are
                // <= H - 32 (lanes) / HALF - 64 (bytes).
                unsafe {
                    let (lo, hi) = (_mm512_setzero_si512(), _mm512_set1_epi16(255));
                    let ld = |p: *const i16| _mm512_loadu_si512(p as *const __m512i);
                    let mut delta = |o: usize| {
                        let mut v = ld(src.as_ptr().add(o));
                        for a in &adds[..NA] {
                            v = _mm512_add_epi16(v, ld(a.as_ptr().add(o)));
                        }
                        for s in &subs[..NS] {
                            v = _mm512_sub_epi16(v, ld(s.as_ptr().add(o)));
                        }
                        _mm512_storeu_si512(out.as_mut_ptr().add(o) as *mut __m512i, v);
                        _mm512_min_epi16(_mm512_max_epi16(v, lo), hi)
                    };
                    // packus interleaves its two sources per 128-bit lane (8
                    // bytes each): qwords x0.0 x1.0 x0.1 x1.1 ... This order
                    // gives x0's 32 bytes, then x1's, each in the 256-bit
                    // kernels' `slot` order (qwords 0, 2, 1, 3 of a source).
                    let order = _mm512_set_epi64(7, 3, 5, 1, 6, 2, 4, 0);
                    let mut sum = [_mm512_setzero_si512(); 4];
                    let mut bv = _mm_set1_epi16(base as i16);
                    for i in (0..HALF).step_by(64) {
                        let (a0, a1) = (delta(i), delta(i + 32));
                        let (b0, b1) = (delta(i + HALF), delta(i + HALF + 32));
                        if DOT {
                            // c in [0, 255], |w| <= 127: c * w fits i16, madd
                            // sums two exact i32 products; each i32 lane sums
                            // at most 2 * HALF / 64 products of <= 255^2 * 127.
                            for (k, (c, o)) in [(a0, i), (a1, i + 32), (b0, i + HALF), (b1, i + HALF + 32)].into_iter().enumerate() {
                                let cw = _mm512_mullo_epi16(c, ld(w.as_ptr().add(o)));
                                sum[k] = if $vnni {
                                    _mm512_dpwssd_epi32(sum[k], cw, c)
                                } else {
                                    _mm512_add_epi32(sum[k], _mm512_madd_epi16(cw, c))
                                };
                            }
                        }
                        // mulhrs(a << 6, b) = (64ab + 2^14) >> 15 = (ab + 256) >> 9.
                        let x0 = _mm512_mulhrs_epi16(_mm512_slli_epi16::<6>(a0), b0);
                        let x1 = _mm512_mulhrs_epi16(_mm512_slli_epi16::<6>(a1), b1);
                        let v = _mm512_permutexvar_epi64(order, _mm512_packus_epi16(x0, x1));
                        _mm512_storeu_si512(x.as_mut_ptr().add(i) as *mut __m512i, v);
                        // Bit j: chunk j of these 16 is all zero.
                        let z = _mm512_cmpeq_epi32_mask(v, lo) as usize;
                        n = push8z(nnz, n, bv, z & 0xFF);
                        bv = _mm_add_epi16(bv, _mm_set1_epi16(8));
                        n = push8z(nnz, n, bv, z >> 8);
                        bv = _mm_add_epi16(bv, _mm_set1_epi16(8));
                    }
                    let total = if DOT {
                        let wide = |v: __m512i| {
                            _mm512_add_epi64(
                                _mm512_cvtepi32_epi64(_mm512_castsi512_si256(v)),
                                _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(v)),
                            )
                        };
                        let s = _mm512_add_epi64(
                            _mm512_add_epi64(wide(sum[0]), wide(sum[1])),
                            _mm512_add_epi64(wide(sum[2]), wide(sum[3])),
                        );
                        _mm512_reduce_add_epi64(s)
                    } else {
                        0
                    };
                    (total, n)
                }
            }
        };
    }

    /// For an 8-bit mask of ZERO chunks, the positions (0..8) of the
    /// nonzero ones packed first (as u8, so one `pmovzxbw` loads them), and
    /// how many there are. Indexing by the zero mask saves the inversion.
    static ZLUT8: [[u8; 8]; 256] = {
        let mut t = [[0u8; 8]; 256];
        let mut m = 0;
        while m < 256 {
            let (mut j, mut n) = (0, 0);
            while j < 8 {
                if m >> j & 1 == 0 {
                    t[m][n] = j as u8;
                    n += 1;
                }
                j += 1;
            }
            m += 1;
        }
        t
    };

    /// The zero-chunk mask of 32 input bytes (bit j: chunk j all zero).
    #[inline(always)]
    unsafe fn zero_mask(v: __m256i) -> usize {
        // SAFETY: register-only AVX2; callers enable it.
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, _mm256_setzero_si256()))) as usize }
    }

    /// `push8` for a zero mask (< 256), with the 8 chunk numbers of this
    /// group already broadcast in `base` (kept in a register by the caller,
    /// instead of rebuilding it from a scalar every group).
    #[inline(always)]
    unsafe fn push8z(nnz: &mut [u16; NNZ_BUF], n: usize, base: __m128i, zmask: usize) -> usize {
        debug_assert!(zmask < 256 && n + 8 <= NNZ_BUF);
        // SAFETY: zmask < 256 (an 8-lane movemask); n <= CHUNKS - 8 before
        // the last group, so the 16-byte store stays inside NNZ_BUF.
        unsafe {
            let idx = _mm_cvtepu8_epi16(_mm_loadl_epi64(ZLUT8.get_unchecked(zmask).as_ptr() as *const __m128i));
            _mm_storeu_si128(nnz.as_mut_ptr().add(n) as *mut __m128i, _mm_add_epi16(idx, base));
            // popcnt (x86-64-v2+) instead of a table load: fewer load-port uops.
            n + 8 - (zmask as u32).count_ones() as usize
        }
    }

    /// `acc += Σ_j x[4l+j] * w[4l+j]` per i32 lane (x u8 <= 127, w i8):
    /// AVX2 (pair sums <= 2 * 127 * 127 < 2^15, so maddubs never saturates),
    /// AVX-VNNI, or AVX512-VNNI + VL. All exact.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dp0(acc: __m256i, x: __m256i, w: __m256i) -> __m256i {
        _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), _mm256_set1_epi16(1)))
    }

    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn dp1(mut acc: __m256i, x: __m256i, w: __m256i) -> __m256i {
        // SAFETY: register-only instruction; AVX-VNNI is enabled.
        unsafe { std::arch::asm!("{{vex}} vpdpbusd {0}, {1}, {2}", inout(ymm_reg) acc, in(ymm_reg) x, in(ymm_reg) w, options(pure, nomem, nostack)) };
        acc
    }

    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl")]
    unsafe fn dp2(mut acc: __m256i, x: __m256i, w: __m256i) -> __m256i {
        // SAFETY: register-only instruction; AVX512-VNNI + VL are enabled.
        unsafe { std::arch::asm!("{{evex}} vpdpbusd {0}, {1}, {2}", inout(ymm_reg) acc, in(ymm_reg) x, in(ymm_reg) w, options(pure, nomem, nostack)) };
        acc
    }

    #[inline(always)]
    unsafe fn dpbusd<const V: u8>(acc: __m256i, x: __m256i, w: __m256i) -> __m256i {
        // SAFETY: the l1_v* wrapper of flavour V enables its features.
        unsafe {
            match V {
                1 => dp1(acc, x, w),
                2 => dp2(acc, x, w),
                _ => dp0(acc, x, w),
            }
        }
    }

    #[allow(dead_code)] // tests (the engine uses l1_rest)
    #[target_feature(enable = "avx2")]
    unsafe fn l1_v0(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: AVX2 is enabled.
        unsafe { store16(l1_avx2::<0>(w, x, nnz)) }
    }

    #[allow(dead_code)] // tests (the engine uses l1_rest)
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn l1_v1(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: AVX-VNNI is enabled.
        unsafe { store16(l1_avx2::<1>(w, x, nnz)) }
    }

    #[allow(dead_code)] // tests (the engine uses l1_rest)
    #[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl")]
    unsafe fn l1_v2(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: AVX512-VNNI + VL are enabled.
        unsafe { store16(l1_avx2::<2>(w, x, nnz)) }
    }

    #[allow(dead_code)] // tests (the engine uses l1_rest)
    #[inline(always)]
    unsafe fn store16(z: [__m256i; 2]) -> [i32; H1] {
        let mut out = [0i32; H1];
        // SAFETY: two in-bounds unaligned stores; callers enable AVX2.
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, z[0]);
            _mm256_storeu_si256(out.as_mut_ptr().add(8) as *mut __m256i, z[1]);
        }
        out
    }

    /// z1 of neurons 0..8 and 8..16.
    #[inline(always)]
    unsafe fn l1_avx2<const V: u8>(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [__m256i; 2] {
        // SAFETY: every chunk index is < CHUNKS, so all loads are in bounds.
        // A chunk's 4 input bytes repeat in every i32 lane; its 64 weight
        // bytes are neurons 0..8 (first vector) and 8..16 (second).
        unsafe {
            let mut a = [_mm256_setzero_si256(); 4];
            let xp = x.as_ptr() as *const i32;
            let wp = w.as_ptr() as *const __m256i;
            let mut i = 0;
            while i + 1 < nnz.len() {
                let (c0, c1) = (nnz[i] as usize, nnz[i + 1] as usize);
                let x0 = _mm256_set1_epi32(xp.add(c0).read_unaligned());
                let x1 = _mm256_set1_epi32(xp.add(c1).read_unaligned());
                a[0] = dpbusd::<V>(a[0], x0, _mm256_load_si256(wp.add(2 * c0)));
                a[1] = dpbusd::<V>(a[1], x0, _mm256_load_si256(wp.add(2 * c0 + 1)));
                a[2] = dpbusd::<V>(a[2], x1, _mm256_load_si256(wp.add(2 * c1)));
                a[3] = dpbusd::<V>(a[3], x1, _mm256_load_si256(wp.add(2 * c1 + 1)));
                i += 2;
            }
            if i < nnz.len() {
                let c0 = nnz[i] as usize;
                let x0 = _mm256_set1_epi32(xp.add(c0).read_unaligned());
                a[0] = dpbusd::<V>(a[0], x0, _mm256_load_si256(wp.add(2 * c0)));
                a[1] = dpbusd::<V>(a[1], x0, _mm256_load_si256(wp.add(2 * c0 + 1)));
            }
            [_mm256_add_epi32(a[0], a[2]), _mm256_add_epi32(a[1], a[3])]
        }
    }

    /// Every fused-update flavour this machine can run, as
    /// (out, x, sum, nonzero list) (tests).
    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn update_pw_all<const DOT: bool>(
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
        base: usize,
        n0: usize,
    ) -> Vec<(&'static str, Vec<i16>, Vec<u8>, i64, Vec<u16>)> {
        type K = unsafe fn(&mut [i16; H], &[i16; H], &[&[i16; H]], &[&[i16; H]], &[i16; H], &mut [u8], usize, &mut [u16; NNZ_BUF], usize) -> (i64, usize);
        macro_rules! pick {
            ($f:ident) => {
                match (adds.len(), subs.len()) {
                    (1, 1) => $f::<DOT, 1, 1> as K,
                    (1, 2) => $f::<DOT, 1, 2> as K,
                    _ => $f::<DOT, 2, 2> as K,
                }
            };
        }
        let mut ks: Vec<(&'static str, K)> = Vec::new();
        if avx2() {
            ks.push(("avx2", pick!(update_pw_avx2)));
        }
        if is_x86_feature_detected!("avxvnni") {
            ks.push(("avxvnni", pick!(update_pw_v1)));
        }
        let f = |s: &str| is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && (s != "vnni" || is_x86_feature_detected!("avx512vnni"));
        if f("vnni") && is_x86_feature_detected!("avx512vl") {
            ks.push(("avx512vnni-256", pick!(update_pw_v2)));
        }
        if f("") {
            ks.push(("avx512bw-512", pick!(update_pw512)));
        }
        if f("vnni") {
            ks.push(("avx512vnni-512", pick!(update_pw512_vnni)));
        }
        ks.into_iter()
            .map(|(name, k)| {
                let mut o = [0i16; H];
                let mut x = [0u8; HALF];
                let mut nz = [0u16; NNZ_BUF];
                // SAFETY: each kernel's features were detected above.
                let (sum, n) = unsafe { k(&mut o, src, adds, subs, w, &mut x, base, &mut nz, n0) };
                (name, o.to_vec(), x.to_vec(), sum, nz[n0..n].to_vec())
            })
            .collect()
    }

    /// The L1 kernel flavours on this machine (tests).
    #[cfg(test)]
    pub(crate) fn l1_all(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> Vec<[i32; H1]> {
        let w = &net.w1[b].0;
        let mut v = vec![scalar::l1(net, b, x, nnz)];
        if avx2() {
            // SAFETY: as in `l1`.
            unsafe {
                v.push(l1_v0(w, x, nnz));
                if is_x86_feature_detected!("avxvnni") {
                    v.push(l1_v1(w, x, nnz));
                }
                if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") {
                    v.push(l1_v2(w, x, nnz));
                }
            }
        }
        v
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) mod kernels {
    pub(crate) use super::scalar::{activate, l1, update_pw};
}

/// This thread's head scratch (the fused update writes the L1 input there).
#[inline(always)]
pub(crate) fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    SCRATCH.with(|s| {
        // SAFETY: the only live borrow of this thread's scratch (`forward`
        // and `with_scratch` do not nest).
        f(unsafe { &mut *s.get() })
    })
}

#[cfg(test)]
/// Head words drawn like trained nets, at HEAD_Q (W1 on the 1/64 grid).
pub(crate) fn random_words(seed: u64) -> Vec<i16> {
    let mut s = seed | 1;
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut u = move |range: f64| -> f64 { ((r() % 20001) as f64 / 10000.0 - 1.0) * range };
    let mut v = Vec::with_capacity(HEAD_WORDS);
    for _ in 0..BUCKETS {
        (0..H1 * N_IN).for_each(|_| v.push(((u(0.5) * 64.0).round() * 128.0) as i16));
        (0..H1).for_each(|_| v.push((u(0.3) * 8192.0).round() as i16));
        (0..H2 * H1).for_each(|_| v.push((u(0.5) * 8192.0).round() as i16));
        (0..H2).for_each(|_| v.push((u(0.3) * 8192.0).round() as i16));
        (0..H2).for_each(|_| v.push((u(0.5) * 8192.0).round() as i16));
        v.push((u(0.1) * 8192.0).round() as i16);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn sq(x: f64) -> f64 {
        let c = x.clamp(0.0, 1.0);
        c * c
    }

    /// The trainer's float head (quantisation-aware forward) on the
    /// (already quantised) accumulators.
    fn float_head(raw: &[i16], b: usize, us: &[i16; H], them: &[i16; H]) -> f64 {
        let p: Vec<f64> = raw[b * BUCKET_WORDS..(b + 1) * BUCKET_WORDS].iter().map(|&x| x as f64 / HEAD_Q as f64).collect();
        let qx = 65025.0 / 512.0;
        let pw = |a: &[i16; H], i: usize| {
            let v = (a[i] as f64 / QA as f64).clamp(0.0, 1.0) * (a[i + HALF] as f64 / QA as f64).clamp(0.0, 1.0);
            (v * qx + (0.5 + 1.0 / 1024.0)).floor() / qx
        };
        let x: Vec<f64> = (0..HALF).map(|i| pw(us, i)).chain((0..HALF).map(|i| pw(them, i))).collect();
        let (w1, rest) = p.split_at(H1 * N_IN);
        let (b1, rest) = rest.split_at(H1);
        let (w2, rest) = rest.split_at(H2 * H1);
        let (b2, rest) = rest.split_at(H2);
        let (w3, b3) = rest.split_at(H2);
        let z1: Vec<f64> = (0..H1).map(|k| b1[k] + (0..N_IN).map(|i| w1[k * N_IN + i] * x[i]).sum::<f64>()).collect();
        let mut out = b3[0];
        for m in 0..H2 {
            let z2 = b2[m] + (0..KA).map(|k| w2[m * H1 + k] * sq(z1[k])).sum::<f64>();
            out += w3[m] * sq(z2);
        }
        if SKIP {
            out += z1[H1 - 1];
        }
        out
    }

    fn random_acc(r: &mut impl FnMut() -> u64, zero_frac: u64) -> [i16; H] {
        let mut a = [0i16; H];
        for x in a.iter_mut() {
            *x = if r() % 100 < zero_frac { -((r() % 200) as i16) } else { (r() % 320) as i16 - 20 };
        }
        a
    }

    #[test]
    fn simd_matches_scalar_and_float() {
        let head = HeadNet::from_words(random_words(3));
        let mut r = rng(17);
        let mut worst = 0.0f64;
        for t in 0..200 {
            let us = random_acc(&mut r, [0, 50, 90, 100][t % 4]);
            let them = random_acc(&mut r, [0, 50, 90, 100][(t / 4) % 4]);
            let b = t % BUCKETS;
            let q = head.forward(b, &us, &them);
            assert_eq!(q.to_bits(), head.forward_scalar(b, &us, &them).to_bits(), "case {t}");
            let f = float_head(&head.raw, b, &us, &them);
            let d = (q as f64 - f).abs() * SCALE as f64;
            worst = worst.max(d);
        }
        // Float rounding only (the integer L1 is exact): far below a centipawn.
        assert!(worst < 0.01, "worst head error {worst:.4} cp");
    }

    #[test]
    fn activation_kernel_matches_scalar() {
        let mut acc = [0i16; H];
        for (i, a) in acc.iter_mut().enumerate() {
            *a = (i as i32 * 37 % 700 - 200) as i16;
        }
        acc[0] = i16::MIN;
        acc[1] = i16::MAX;
        acc[HALF + 1] = i16::MAX;
        let (mut s, mut v) = ([0u8; HALF], [0u8; HALF]);
        let (mut ns, mut nv) = ([0u16; NNZ_BUF], [0u16; NNZ_BUF]);
        let a = scalar::activate(&acc, &mut s, 7, &mut ns, 3);
        let b = kernels::activate(&acc, &mut v, 7, &mut nv, 3);
        assert_eq!(s, v);
        assert_eq!(s[1], 127);
        assert_eq!((a, &ns[3..a]), (b, &nv[3..b]));
        assert!(a > 3 && a < 3 + HALF / 4);
        // Every product of clamped values rounds like (a b + 256) >> 9.
        for x in 0..=255i16 {
            let mut acc = [0i16; H];
            for i in 0..HALF {
                acc[i] = x;
                acc[i + HALF] = (i % 256) as i16;
            }
            let (mut s, mut v) = ([0u8; HALF], [0u8; HALF]);
            scalar::activate(&acc, &mut s, 0, &mut ns, 0);
            kernels::activate(&acc, &mut v, 0, &mut nv, 0);
            assert_eq!(s, v, "x = {x}");
        }
    }

    #[test]
    fn fused_update_matches_separate_passes() {
        let mut r = rng(23);
        let mut row = || -> Box<[i16; H]> {
            let mut a = Box::new([0i16; H]);
            for x in a.iter_mut() {
                *x = (r() % 160) as i16 - 80;
            }
            a
        };
        let mut wr = rng(4);
        let mut w = [0i16; H];
        for x in w.iter_mut() {
            *x = (wr() % 255) as i16 - 127;
        }
        for t in 0..60 {
            let mut src = [0i16; H];
            for (i, x) in src.iter_mut().enumerate() {
                *x = ((i * 7919 + t * 104729) % 700) as i16 - 250;
            }
            let (a0, a1, s0, s1) = (row(), row(), row(), row());
            let (adds, subs): (Vec<&[i16; H]>, Vec<&[i16; H]>) = match t % 3 {
                0 => (vec![&*a0], vec![&*s0]),
                1 => (vec![&*a0], vec![&*s0, &*s1]),
                _ => (vec![&*a0, &*a1], vec![&*s0, &*s1]),
            };
            let (mut o1, mut o2) = ([0i16; H], [0i16; H]);
            let (mut x1, mut x2) = ([0u8; HALF], [0u8; HALF]);
            let (mut n1, mut n2) = ([0u16; NNZ_BUF], [0u16; NNZ_BUF]);
            let r1 = scalar::update_pw::<true>(&mut o1, &src, &adds, &subs, &w, &mut x1, 5, &mut n1, 2);
            let r2 = kernels::update_pw::<true>(&mut o2, &src, &adds, &subs, &w, &mut x2, 5, &mut n2, 2);
            assert_eq!(o1, o2);
            assert_eq!(x1, x2);
            assert_eq!((r1.0, r1.1, &n1[2..r1.1]), (r2.0, r2.1, &n2[2..r2.1]), "case {t}");
            #[cfg(target_arch = "x86_64")]
            for (name, o, x, sum, nz) in kernels::update_pw_all::<true>(&src, &adds, &subs, &w, 5, 2) {
                assert_eq!((&o[..], &x[..], sum, &nz[..]), (&o1[..], &x1[..], r1.0, &n1[2..r1.1]), "{name} case {t}");
            }
            #[cfg(target_arch = "x86_64")]
            for (name, o, x, sum, nz) in kernels::update_pw_all::<false>(&src, &adds, &subs, &w, 5, 2) {
                assert_eq!((&o[..], &x[..], sum, &nz[..]), (&o1[..], &x1[..], 0, &n1[2..r1.1]), "{name} nodot case {t}");
            }
            let r3 = kernels::update_pw::<false>(&mut o2, &src, &adds, &subs, &w, &mut x2, 5, &mut n2, 2);
            assert_eq!((r3.0, r3.1), (0, r1.1));
            // The scalar twin is update + screlu_dot + activate.
            let mut x3 = [0u8; HALF];
            let mut n3 = [0u16; NNZ_BUF];
            let k = scalar::activate(&o1, &mut x3, 5, &mut n3, 2);
            assert_eq!((k, x3), (r1.1, x1));
            assert_eq!(r1.0, crate::nnue::scalar::screlu_dot(&o1, &w));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn l1_flavours_agree() {
        let head = HeadNet::from_words(random_words(9));
        let mut r = rng(5);
        for t in 0..50 {
            let (us, them) = (random_acc(&mut r, 60), random_acc(&mut r, 80));
            let mut x = Input([0; N_IN]);
            let mut nnz = [0u16; NNZ_BUF];
            let n = scalar::activate(&us, &mut x.0[..HALF], 0, &mut nnz, 0);
            let n = scalar::activate(&them, &mut x.0[HALF..], HALF / 4, &mut nnz, n);
            let all = kernels::l1_all(&head, t % BUCKETS, &x.0, &nnz[..n]);
            assert!(all.iter().all(|z| *z == all[0]), "case {t}: {} flavours disagree", all.len());
        }
    }

    #[test]
    fn finish_rounds_to_nearest() {
        assert_eq!(finish(0, 0, 1.0, 20000), 400);
        assert_eq!(finish(0, 0, -1.0, 20000), -400);
        let sum = 123_456_789i64;
        let lin = (sum / QA as i64 + 77) as f64 * SCALE as f64 / (QA as i64 * QB as i64) as f64;
        assert_eq!(finish(77, sum, 0.0, 20000) as i64, lin.round() as i64);
        assert_eq!(finish(0, 0, 1.0 / 800.0 + 1e-6, 20000), 1);
        assert_eq!(finish(0, 0, 100.0, 20000), 20000);
    }

    #[test]
    fn finish_matches_reference() {
        let mut r = rng(99);
        let d = (QA as i64 * QB as i64) as f64;
        let mut check = |lb: i16, sum: i64, h: f32| assert_eq!(finish(lb, sum, h, 20000), finish_reference(lb, sum, h, 20000), "{lb} {sum} {h}");
        for _ in 0..2_000_000 {
            let sum = (r() % 2_000_000_000) as i64 - 1_000_000_000;
            let lb = (r() % 2001) as i16 - 1000;
            let h = ((r() % 4_000_001) as f64 / 1_000_000.0 - 2.0) as f32;
            check(lb, sum, h);
            // Heads that put y on or next to a tie k + 0.5.
            let lin = (sum / QA as i64 + lb as i64) as f64 / d;
            let k = (r() % 4001) as f64 - 2000.0;
            let h0 = ((k + 0.5) / SCALE as f64 - lin) as f32;
            check(lb, sum, h0);
            check(lb, sum, f32::from_bits(h0.to_bits() + 1));
            check(lb, sum, f32::from_bits(h0.to_bits().wrapping_sub(1)));
        }
        for sum in -100_000i64..100_000 {
            check(0, sum * QA as i64, 0.0);
            check(3, sum, 0.0);
        }
        // Exact ties: h = (2i + 1) / 32 gives y = 12.5 (2i + 1).
        for i in -200..200 {
            let h = (2 * i + 1) as f32 / 32.0;
            check(0, 0, h);
            assert_eq!(finish(0, 0, h, 20000), finish_reference(0, 0, h, 20000));
            assert_eq!(finish(0, 0, h, 20000) as f64, (12.5 * (2 * i + 1) as f64).round());
        }
        check(0, 0, 0.0);
        check(0, 0, -0.0);
        check(0, 0, 1e9);
        check(0, 0, -1e9);
    }

    /// `cargo test --release head_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn head_throughput() {
        use std::time::Instant;
        let head = HeadNet::from_words(random_words(5));
        let mut r = rng(9);
        for zero in [50u64, 70, 90] {
            let accs: Vec<([i16; H], [i16; H])> = (0..64).map(|_| (random_acc(&mut r, zero), random_acc(&mut r, zero))).collect();
            let mut nz = 0usize;
            let mut idx = [0u16; NNZ_BUF];
            for (u, t) in &accs {
                let mut x = Input([0; N_IN]);
                let n = scalar::activate(u, &mut x.0[..HALF], 0, &mut idx, 0);
                nz += scalar::activate(t, &mut x.0[HALF..], HALF / 4, &mut idx, n);
            }
            let n = 2_000_000;
            let mut s = 0.0f64;
            let t = Instant::now();
            for i in 0..n {
                let (u, th) = &accs[i % accs.len()];
                s += head.forward(i / 64 % BUCKETS, std::hint::black_box(u), std::hint::black_box(th)) as f64;
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            let mut x = Input([0; N_IN]);
            let t = Instant::now();
            for i in 0..n {
                let (u, th) = &accs[i % accs.len()];
                let k = kernels::activate(std::hint::black_box(u), &mut x.0[..HALF], 0, &mut idx, 0);
                let k = kernels::activate(std::hint::black_box(th), &mut x.0[HALF..], HALF / 4, &mut idx, k);
                s += std::hint::black_box(&x).0[i % N_IN] as f64 + k as f64;
            }
            let ns_act = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            let t = Instant::now();
            let mut z = [0i32; H1];
            for i in 0..n {
                z[i % H1] = (i as i32 * 977) & 0x3fff;
                s += head.rest(i % BUCKETS, std::hint::black_box(&z)) as f64;
            }
            let ns_rest = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            println!("head: {zero}% zero acc, {:.0}/{CHUNKS} nonzero chunks: forward {ns:.1} ns (activate + nnz {ns_act:.1}, b1 + L2 + L3 {ns_rest:.1})", nz as f64 / accs.len() as f64);
            std::hint::black_box(s);
        }
    }
}

/// Engine vs trainer on a real pairwise net. The gate for head nets: with
/// `HEAD = true`, `cargo test` (CI, the queue, `netloop.py embed`) fails
/// unless `net/head_fixture.txt` holds the trainer's float evals of the
/// embedded net (`<FEN> | <cp>` lines from `nnue-trainer fen-eval`, 2000
/// validation positions; `scripts/head_check.py --write-fixture` makes it)
/// and the engine agrees with them: mean |diff| < 1 cp, p95 < 3 cp, max
/// < 10 cp. `RUSTFISH_HEAD_FIXTURE` / `RUSTFISH_HEAD_NET` override the
/// fixture / net.
#[cfg(test)]
mod fixture {
    use super::super::{Accumulator, Active, Network, NnueState, HEAD};
    use super::{scalar, CHUNKS, HALF, NNZ_BUF};
    use crate::Board;
    use std::sync::Arc;

    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/net/head_fixture.txt");

    #[test]
    fn head_matches_trainer_fixture() {
        if !HEAD {
            return;
        }
        let path = std::env::var("RUSTFISH_HEAD_FIXTURE").unwrap_or_else(|_| FIXTURE.to_string());
        let net = match std::env::var("RUSTFISH_HEAD_NET") {
            Ok(p) => Network::load(&p).unwrap(),
            Err(_) => Network::from_bytes(crate::EMBEDDED_NET).unwrap(),
        };
        let net = Arc::new(net);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("{path}: {e}. A HEAD net cannot be gated without the trainer's evals of it: scripts/head_check.py --write-fixture")
        });
        let head = net.head.as_ref().unwrap();
        let (mut sum, mut worst, mut worst_fen) = (0.0f64, 0.0f64, String::new());
        let mut diffs = Vec::new();
        let mut nnz_sum = 0usize;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (fen, cp) = line.rsplit_once('|').unwrap();
            let want: f64 = cp.trim().parse().unwrap();
            let mut board = Board::new();
            board.set_fen(fen.trim());
            let st = NnueState::<Active>::new(net.clone(), &board);
            let got = st.evaluate(board.side_to_move, board.bb.occ_all().count_ones()) as f64;
            let acc = Accumulator::refresh(&net, &board);
            let (mut x, mut idx) = ([0u8; HALF], [0u16; NNZ_BUF]);
            nnz_sum += scalar::activate(&acc.v[0], &mut x, 0, &mut idx, 0) + scalar::activate(&acc.v[1], &mut x, 0, &mut idx, 0);
            // The SIMD kernels match the scalar reference on real accumulators.
            let b = super::super::bucket(board.bb.occ_all().count_ones());
            let (u, t) = (&acc.v[0], &acc.v[1]);
            assert_eq!(head.forward(b, u, t).to_bits(), head.forward_scalar(b, u, t).to_bits(), "{fen}");
            assert_eq!(head.forward(b, t, u).to_bits(), head.forward_scalar(b, t, u).to_bits(), "{fen}");
            // The engine clamps at EVAL_LIMIT; compare the unclamped range.
            if want.abs() < 15_000.0 {
                let d = got - want;
                diffs.push(d);
                sum += d.abs();
                if d.abs() > worst {
                    worst = d.abs();
                    worst_fen = format!("{} (engine {got}, trainer {want})", fen.trim());
                }
            }
        }
        let n = diffs.len();
        assert!(n >= 1000, "fixture has {n} usable positions (want 2000)");
        let mean = sum / n as f64;
        let bias = diffs.iter().sum::<f64>() / n as f64;
        diffs.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        let p95 = diffs[n * 95 / 100].abs();
        let nnz = nnz_sum as f64 / diffs.len() as f64;
        eprintln!("head fixture: nonzero L1 input chunks {nnz:.1} of {CHUNKS} ({:.1}%)", 100.0 * nnz / CHUNKS as f64);
        eprintln!("head fixture: {n} positions, |engine - trainer| mean {mean:.3} cp, p95 {p95:.2}, max {worst:.2}, bias {bias:+.3}; worst {worst_fen}");
        assert!(mean < 1.0 && p95 < 3.0 && worst < 10.0, "engine and trainer disagree beyond quantisation error");
    }
}

/// Offline FT neuron permutation for pairwise-head nets (exact: the eval
/// is unchanged bit for bit, only the order of the L1 input changes).
///
/// The L1 input is scanned in 4-byte chunks and only nonzero chunks cost
/// work, so grouping pairs that are zero together into the same chunks
/// cuts the nonzero-chunk count. Pair `i` (accumulator neurons `i` and
/// `i + H/2`) moves to slot `p` together with its FT columns, FT bias,
/// linear output weights and W1 columns in both halves of the L1 input;
/// every sum involved is an exact integer sum, so the result is identical.
///
/// Run (ignored test; FENs one per line, e.g. a slice of a validation set
/// converted with scripts/head_check.py's `rec_to_fen`):
/// `RUSTFISH_PERM_FENS=fens.txt RUSTFISH_PERM_OUT=out.bin \
///  [RUSTFISH_PERM_TEST=heldout.txt] [RUSTFISH_HEAD_NET=in.bin] \
///  cargo test --release head_permute -- --ignored --nocapture`
#[cfg(test)]
mod permute {
    use super::super::{net_bytes, Accumulator, Network, BUCKETS, HEAD, INPUTS, KB};
    use super::{BUCKET_WORDS, H, H1, HALF, N_IN};
    use crate::Board;

    /// Per-pair activity over samples (position x perspective) as bitsets.
    fn activity(net: &Network, path: &str) -> (Vec<Vec<u64>>, usize) {
        let text = std::fs::read_to_string(path).unwrap();
        let fens: Vec<&str> = text.lines().map(|l| l.split('|').next().unwrap().trim()).filter(|l| !l.is_empty()).collect();
        let samples = fens.len() * 2;
        let mut bits = vec![vec![0u64; samples.div_ceil(64)]; HALF];
        for (n, fen) in fens.iter().enumerate() {
            let mut board = Board::new();
            board.set_fen(fen);
            let acc = Accumulator::refresh(net, &board);
            for p in 0..2 {
                let s = 2 * n + p;
                let a = &acc.v[p];
                for i in 0..HALF {
                    let x = ((a[i] as i32).clamp(0, 255) * (a[i + HALF] as i32).clamp(0, 255) + 256) >> 9;
                    if x != 0 {
                        bits[i][s / 64] |= 1 << (s % 64);
                    }
                }
            }
        }
        (bits, samples)
    }

    /// Nonzero chunks (summed over samples) of slot order `perm`.
    fn cost(bits: &[Vec<u64>], perm: &[usize]) -> u64 {
        perm.chunks(4).map(|c| chunk_cost(bits, c)).sum()
    }

    fn chunk_cost(bits: &[Vec<u64>], c: &[usize]) -> u64 {
        (0..bits[0].len()).map(|k| c.iter().fold(0u64, |m, &i| m | bits[i][k]).count_ones() as u64).sum()
    }

    /// Order the pairs: by activity, then pair swaps between chunks while
    /// they reduce the nonzero-chunk total (deterministic xorshift).
    fn optimise(bits: &[Vec<u64>]) -> Vec<usize> {
        let freq: Vec<u32> = bits.iter().map(|b| b.iter().map(|w| w.count_ones()).sum()).collect();
        let mut perm: Vec<usize> = (0..HALF).collect();
        perm.sort_by_key(|&i| (std::cmp::Reverse(freq[i]), i));
        let mut cc: Vec<u64> = perm.chunks(4).map(|c| chunk_cost(bits, c)).collect();
        eprintln!("  sorted by activity: {}", cc.iter().sum::<u64>());
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let iters: usize = std::env::var("RUSTFISH_PERM_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(400_000);
        for t in 0..iters {
            let (a, b) = ((next() % HALF as u64) as usize, (next() % HALF as u64) as usize);
            let (ca, cb) = (a / 4, b / 4);
            if ca == cb {
                continue;
            }
            perm.swap(a, b);
            let na = chunk_cost(bits, &perm[4 * ca..4 * ca + 4]);
            let nb = chunk_cost(bits, &perm[4 * cb..4 * cb + 4]);
            if na + nb < cc[ca] + cc[cb] {
                cc[ca] = na;
                cc[cb] = nb;
            } else {
                perm.swap(a, b);
            }
            if t % 100_000 == 0 {
                eprintln!("  iter {t}: {}", cc.iter().sum::<u64>());
            }
        }
        perm
    }

    /// Slot `p` of a row of H gets pair `perm[p]` (both halves).
    fn permute_row(r: &mut [i16], perm: &[usize]) {
        let old = r.to_vec();
        for (p, &i) in perm.iter().enumerate() {
            r[p] = old[i];
            r[p + HALF] = old[i + HALF];
        }
    }

    /// The net file with pair `perm[p]` moved to slot `p`.
    fn permute_bytes(bytes: &[u8], perm: &[usize]) -> Vec<u8> {
        let mut w: Vec<i16> = bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        // FT weight rows, FT bias, then the linear output rows (us, them per bucket).
        let rows = KB * INPUTS + 1 + BUCKETS * 2;
        for r in w[..rows * H].chunks_exact_mut(H) {
            permute_row(r, perm);
        }
        // W1 rows: the L1 input is the stm's pairs 0..HALF, then the other side's.
        let head = rows * H + BUCKETS;
        for b in 0..BUCKETS {
            for k in 0..H1 {
                let o = head + b * BUCKET_WORDS + k * N_IN;
                permute_row(&mut w[o..o + N_IN], perm);
            }
        }
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    #[ignore]
    fn head_permute() {
        assert!(HEAD && N_IN == H);
        let bytes = match std::env::var("RUSTFISH_HEAD_NET") {
            Ok(p) => std::fs::read(p).unwrap(),
            Err(_) => crate::EMBEDDED_NET.to_vec(),
        };
        assert_eq!(bytes.len(), net_bytes(KB));
        let net = Network::from_bytes(&bytes).unwrap();
        let (bits, samples) = activity(&net, &std::env::var("RUSTFISH_PERM_FENS").expect("RUSTFISH_PERM_FENS"));
        let id: Vec<usize> = (0..HALF).collect();
        // Nonzero chunks per eval (two perspectives).
        let per = |c: u64, s: usize| 2.0 * c as f64 / s as f64;
        eprintln!("train ({samples} samples): identity {:.1} nonzero chunks / eval", per(cost(&bits, &id), samples));
        let perm = optimise(&bits);
        eprintln!("train: permuted {:.1}", per(cost(&bits, &perm), samples));
        let out = permute_bytes(&bytes, &perm);
        let pnet = Network::from_bytes(&out).unwrap();
        if let Ok(t) = std::env::var("RUSTFISH_PERM_TEST") {
            let (tb, ts) = activity(&net, &t);
            let (pb, _) = activity(&pnet, &t);
            eprintln!(
                "test ({ts} samples): identity {:.1}, permuted {:.1} (permuted net, measured: {:.1})",
                per(cost(&tb, &id), ts),
                per(cost(&tb, &perm), ts),
                per(cost(&pb, &id), ts)
            );
        }
        if let Ok(o) = std::env::var("RUSTFISH_PERM_OUT") {
            std::fs::write(&o, &out).unwrap();
            eprintln!("wrote {o}");
        }
    }

    /// Microbench of the head's parts on real accumulators (ns per eval).
    #[test]
    #[ignore]
    fn head_microbench() {
        use super::super::{out_half, simd};
        use super::{kernels, Scratch, CHUNKS, KA, Z1_DIV};
        let net = Network::from_bytes(crate::EMBEDDED_NET).unwrap();
        let accs: Vec<Accumulator> = if let Ok(path) = std::env::var("RUSTFISH_PERM_FENS") {
            let text = std::fs::read_to_string(path).unwrap();
            text.lines()
                .take(2000)
                .map(|fen| {
                    let mut board = Board::new();
                    board.set_fen(fen.split('|').next().unwrap().trim());
                    Accumulator::refresh(&net, &board)
                })
                .collect()
        } else {
            // Random walks from the bench positions.
            let mut seed = 0x1234_5678_u64;
            let mut v = Vec::new();
            let mut moves = Vec::new();
            for fen in crate::uci::BENCH_FENS {
                for _ in 0..16 {
                    let mut board = Board::new();
                    board.set_fen(fen);
                    for _ in 0..10 {
                        board.generate_legal_moves(false, &mut moves);
                        if moves.is_empty() {
                            break;
                        }
                        let mv = moves[(crate::tt::splitmix64(&mut seed) % moves.len() as u64) as usize];
                        board.make_move(mv);
                        v.push(Accumulator::refresh(&net, &board));
                    }
                }
            }
            v
        };
        let (add, sub) = (net.row(100), net.row(200));
        let w0 = out_half(&net, 3, true);
        let head = net.head.as_ref().unwrap();
        let mut out = [0i16; H];
        let mut sc = Scratch::new();
        // RUSTFISH_MB_HOT: 8 accumulators only (L1-resident, like the
        // parent ply's in search).
        let (accs, reps) = if std::env::var("RUSTFISH_MB_HOT").is_ok() { (accs[..8].to_vec(), 20 * accs.len() / 8) } else { (accs, 20) };
        let t = std::time::Instant::now();
        let mut sink = 0i64;
        for _ in 0..reps {
            for a in &accs {
                for p in 0..2 {
                    sink += simd::update_dot(&mut out, &a.v[p], &[add], &[sub], w0);
                }
            }
        }
        let n = (reps * accs.len()) as f64;
        eprintln!("update_dot x2: {:.1} ns", t.elapsed().as_nanos() as f64 / n);
        let t = std::time::Instant::now();
        let mut nz = 0usize;
        for _ in 0..reps {
            for a in &accs {
                sc.n = 0;
                for p in 0..2 {
                    let hx = CHUNKS * 2;
                    let (s, n) = kernels::update_pw::<true>(&mut out, &a.v[p], &[add], &[sub], w0, &mut sc.x.0[p * hx..(p + 1) * hx], p * hx / 4, &mut sc.nnz, sc.n);
                    sink += s;
                    sc.n = n;
                }
                nz += sc.n;
            }
        }
        eprintln!("update_pw x2: {:.1} ns ({:.1} nnz)", t.elapsed().as_nanos() as f64 / n, nz as f64 / n);
        // L1 + rest on each position's input.
        let scs: Vec<Box<Scratch>> = accs
            .iter()
            .map(|a| {
                let mut s = Box::new(Scratch::new());
                let n = kernels::activate(&a.v[0], &mut s.x.0[..HALF], 0, &mut s.nnz, 0);
                s.n = kernels::activate(&a.v[1], &mut s.x.0[HALF..], HALF / 4, &mut s.nnz, n);
                s
            })
            .collect();
        let mut f = 0.0f32;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for (j, s) in scs.iter().enumerate() {
                let z = kernels::l1(head, j % BUCKETS, &s.x.0, &s.nnz[..s.n]);
                sink += z[0] as i64;
            }
        }
        eprintln!("l1: {:.1} ns", t.elapsed().as_nanos() as f64 / n);
        let z1s: Vec<[i32; H1]> = scs.iter().enumerate().map(|(j, s)| kernels::l1(head, j % BUCKETS, &s.x.0, &s.nnz[..s.n])).collect();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for (j, z) in z1s.iter().enumerate() {
                f += head.rest(j % BUCKETS, std::hint::black_box(z));
            }
        }
        eprintln!("rest: {:.1} ns  ({sink} {f})", t.elapsed().as_nanos() as f64 / n);
        let active: usize = z1s
            .iter()
            .enumerate()
            .map(|(j, z)| (0..KA).filter(|&k| z[k] as f32 * (1.0 / Z1_DIV) + head.b1[j % BUCKETS][k] > 0.0).count())
            .sum();
        eprintln!("active L1 neurons: {:.2} of {KA}", active as f64 / z1s.len() as f64);
        {
            // Largest |w| pair sum (maddubs lanes) over chunks, per bucket.
            let m = (0..BUCKETS)
                .map(|b| {
                    let w = &head.w1[b].0;
                    (0..w.len() / 2).map(|p| (w[2 * p] as i32).abs() + (w[2 * p + 1] as i32).abs()).max().unwrap()
                })
                .max()
                .unwrap();
            eprintln!("max |w| pair sum: {m}");
        }
        #[cfg(target_arch = "x86_64")]
        {
            for (j, z) in z1s.iter().enumerate() {
                assert_eq!(head.rest(j % BUCKETS, z).to_bits(), unsafe { head.rest_avx2(j % BUCKETS, z) }.to_bits());
            }
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for (j, z) in z1s.iter().enumerate() {
                    f += unsafe { head.rest_avx2(j % BUCKETS, std::hint::black_box(z)) };
                }
            }
            eprintln!("rest_avx2: {:.1} ns  ({f})", t.elapsed().as_nanos() as f64 / n);
        }
        #[cfg(target_arch = "aarch64")]
        {
            for (j, z) in z1s.iter().enumerate() {
                assert_eq!(head.rest(j % BUCKETS, z).to_bits(), head.rest_neon(j % BUCKETS, z).to_bits());
            }
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for (j, z) in z1s.iter().enumerate() {
                    f += head.rest_neon(j % BUCKETS, std::hint::black_box(z));
                }
            }
            eprintln!("rest_neon: {:.1} ns  ({f})", t.elapsed().as_nanos() as f64 / n);
        }
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for (j, s) in scs.iter().enumerate() {
                let z = kernels::l1(head, 5 + 0 * j, &s.x.0, &s.nnz[..s.n]);
                sink += z[0] as i64;
            }
        }
        eprintln!("l1 (one bucket): {:.1} ns {sink}", t.elapsed().as_nanos() as f64 / n);
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for (j, s) in scs.iter().enumerate() {
                f += head.l1_rest(j % BUCKETS, std::hint::black_box(&s.x.0), &s.nnz[..s.n]);
            }
        }
        eprintln!("l1_rest: {:.1} ns {f}", t.elapsed().as_nanos() as f64 / n);
        #[cfg(target_arch = "x86_64")]
        {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for (j, s) in scs.iter().enumerate() {
                    let z = kernels::l1(head, j % BUCKETS, std::hint::black_box(&s.x.0), &s.nnz[..s.n]);
                    f += unsafe { head.rest_avx2(j % BUCKETS, &z) };
                }
            }
            eprintln!("l1 + rest_avx2: {:.1} ns {f}", t.elapsed().as_nanos() as f64 / n);
        }
    }
}
