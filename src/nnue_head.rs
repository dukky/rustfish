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
                    h.w1[b].0[(i / 4) * H1 * 4 + k * 4 + i % 4] = w as i8;
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
            let z1 = kernels::l1(self, b, &s.x.0, &s.nnz[..n]);
            self.rest(b, &z1)
        })
    }

    /// `forward` on an input already in `s` (the fused update's output).
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn forward_scratch(&self, b: usize, s: &Scratch) -> f32 {
        let z1 = kernels::l1(self, b, &s.x.0, &s.nnz[..s.n]);
        self.rest(b, &z1)
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

    /// b1, the L1 activation, L2 and L3 in f32 (fixed operation order).
    #[inline(always)]
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
            out[i] = ((a * b + 256) >> 9) as u8;
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
            let bits = vld1q_u32([1u32, 2, 4, 8].as_ptr());
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
                    m = vorrq_u32(m, vshlq_u32(vandq_u32(vtstq_u32(v32, v32), bits), vdupq_n_s32(4 * g as i32)));
                }
                let mask = vaddvq_u32(m);
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
            let bits = vld1q_u32([1u32, 2, 4, 8].as_ptr());
            let mut s = [vdupq_n_s32(0); 8];
            let (o, sp) = (out.as_mut_ptr(), src.as_ptr());
            // 16 pairs per step (two vectors of each half: few live registers).
            let delta2 = |at: usize| {
                let v = vld1q_s16_x2(sp.add(at));
                let mut v = [v.0, v.1];
                for a in adds {
                    let x = vld1q_s16_x2(a.as_ptr().add(at));
                    v = [vaddq_s16(v[0], x.0), vaddq_s16(v[1], x.1)];
                }
                for r in subs {
                    let x = vld1q_s16_x2(r.as_ptr().add(at));
                    v = [vsubq_s16(v[0], x.0), vsubq_s16(v[1], x.1)];
                }
                vst1q_s16_x2(o.add(at), int16x8x2_t(v[0], v[1]));
                v
            };
            for i in (0..HALF).step_by(64) {
                let mut m = vdupq_n_u32(0);
                for g in 0..4 {
                    let at = i + 16 * g;
                    let lo = delta2(at);
                    let hi = delta2(at + HALF);
                    if DOT {
                        let (wl, wh) = (vld1q_s16_x2(w.as_ptr().add(at)), vld1q_s16_x2(w.as_ptr().add(at + HALF)));
                        for (j, (v, wv)) in [(lo[0], wl.0), (lo[1], wl.1), (hi[0], wh.0), (hi[1], wh.1)].into_iter().enumerate() {
                            let c = vminq_s16(vmaxq_s16(v, zero), qa);
                            let cw = vmulq_s16(c, wv);
                            s[2 * j] = vmlal_s16(s[2 * j], vget_low_s16(cw), vget_low_s16(c));
                            s[2 * j + 1] = vmlal_high_s16(s[2 * j + 1], cw, c);
                        }
                    }
                    let a = vqmovun_high_s16(vqmovun_s16(lo[0]), lo[1]);
                    let b = vqmovun_high_s16(vqmovun_s16(hi[0]), hi[1]);
                    let l = vrshrn_n_u16::<8>(vshrq_n_u16::<1>(vmull_u8(vget_low_u8(a), vget_low_u8(b))));
                    let v = vrshrn_high_n_u16::<8>(l, vshrq_n_u16::<1>(vmull_high_u8(a, b)));
                    vst1q_u8(x.as_mut_ptr().add(at), v);
                    let v32 = vreinterpretq_u32_u8(v);
                    m = vorrq_u32(m, vshlq_u32(vandq_u32(vtstq_u32(v32, v32), bits), vdupq_n_s32(4 * g as i32)));
                }
                let mask = vaddvq_u32(m);
                n = push8(nnz, n, base + i / 4, mask);
                n = push8(nnz, n, base + i / 4 + 8, mask >> 8);
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
    fn avx2() -> bool {
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

    #[target_feature(enable = "avx2")]
    unsafe fn activate_avx2(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], mut n: usize) -> usize {
        assert!(out.len() == HALF && HALF % 32 == 0);
        // SAFETY: HALF % 32 == 0, all loads/stores in bounds.
        unsafe {
            let (lo, hi) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
            let p = acc.as_ptr();
            let c = |o: usize| _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(o) as *const __m256i), lo), hi);
            for i in (0..HALF).step_by(32) {
                // mulhrs(a << 6, b) = (64ab + 2^14) >> 15 = (ab + 256) >> 9.
                let x0 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(c(i)), c(i + HALF));
                let x1 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(c(i + 16)), c(i + 16 + HALF));
                // packus interleaves 128-bit lanes; restore the order.
                let v = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(x0, x1));
                _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, v);
                let zero = _mm256_cmpeq_epi32(v, lo);
                n = push8(nnz, n, base + i / 4, !(_mm256_movemask_ps(_mm256_castsi256_ps(zero)) as u32));
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
        // SAFETY: AVX2 is available.
        unsafe {
            match (adds.len(), subs.len()) {
                (1, 1) => update_pw_avx2::<DOT, 1, 1>(out, src, adds, subs, w, x, base, nnz, n),
                (1, 2) => update_pw_avx2::<DOT, 1, 2>(out, src, adds, subs, w, x, base, nnz, n),
                (2, 2) => update_pw_avx2::<DOT, 2, 2>(out, src, adds, subs, w, x, base, nnz, n),
                _ => scalar::update_pw::<DOT>(out, src, adds, subs, w, x, base, nnz, n),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    unsafe fn update_pw_avx2<const DOT: bool, const NA: usize, const NS: usize>(
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
            let mut sum = [_mm256_setzero_si256(); 2];
            for i in (0..HALF).step_by(32) {
                let (a0, a1) = (delta(i), delta(i + 16));
                let (b0, b1) = (delta(i + HALF), delta(i + HALF + 16));
                if DOT {
                    // c in [0, 255], |w| <= 127: c * w fits i16, madd sums two exact i32 products.
                    for (k, (c, o)) in [(a0, i), (a1, i + 16), (b0, i + HALF), (b1, i + HALF + 16)].into_iter().enumerate() {
                        let cw = _mm256_mullo_epi16(c, ld(w.as_ptr().add(o)));
                        sum[k & 1] = _mm256_add_epi32(sum[k & 1], _mm256_madd_epi16(cw, c));
                    }
                }
                // mulhrs(a << 6, b) = (64ab + 2^14) >> 15 = (ab + 256) >> 9.
                let x0 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(a0), b0);
                let x1 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(a1), b1);
                let v = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(x0, x1));
                _mm256_storeu_si256(x.as_mut_ptr().add(i) as *mut __m256i, v);
                let zero = _mm256_cmpeq_epi32(v, lo);
                n = push8(nnz, n, base + i / 4, !(_mm256_movemask_ps(_mm256_castsi256_ps(zero)) as u32));
            }
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

    #[target_feature(enable = "avx2")]
    unsafe fn l1_v0(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: AVX2 is enabled.
        unsafe { l1_avx2::<0>(w, x, nnz) }
    }

    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn l1_v1(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: AVX-VNNI is enabled.
        unsafe { l1_avx2::<1>(w, x, nnz) }
    }

    #[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl")]
    unsafe fn l1_v2(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: AVX512-VNNI + VL are enabled.
        unsafe { l1_avx2::<2>(w, x, nnz) }
    }

    #[inline(always)]
    unsafe fn l1_avx2<const V: u8>(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
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
            let mut out = [0i32; H1];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, _mm256_add_epi32(a[0], a[2]));
            _mm256_storeu_si256(out.as_mut_ptr().add(8) as *mut __m256i, _mm256_add_epi32(a[1], a[3]));
            out
        }
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
