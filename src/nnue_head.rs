//! Deeper NNUE head (docs/nnue-deeper-head.md): per output bucket,
//! `2H -> 16 -> 32 -> 1` on the SCReLU'd accumulators, added to the linear
//! output. Compiled in with `nnue::HEAD = true`; with the default (`false`)
//! nothing here is reachable and the net format is unchanged.
//!
//! Trainer semantics (`tools/nnue-trainer --head`, all float, `a` = the
//! accumulator in units of QA, stm half first):
//!
//! ```text
//! z1 = b1 + W1 · sq(a)        (16)     sq(x) = clamp(x, 0, 1)^2
//! z2 = b2 + W2 · sq(z1)       (32)
//! out_head = b3 + W3 · sq(z2)
//! eval_cp = 400 * (linear output + out_head)
//! ```
//!
//! File: after the linear output biases, per bucket `W1 [16][2H]`, b1 [16],
//! `W2 [32][16]`, b2 [32], W3 [32], b3, as i16 at `HEAD_Q` = 8192.
//!
//! Quantisation (integer throughout, so every SIMD kernel matches its
//! scalar twin bit for bit):
//!
//! - L1 input: `x = (clamp(acc, 0, 255)^2 + 128) >> 8` in 0..=254 (u8),
//!   i.e. `sq(a) * 65025 / 256` rounded.
//! - W1: i16 at `2^s1 * 65536 / 65025`, so `Σ x w` is `z1` at `2^p1`,
//!   `p1 = 8 + s1`; b1 in i32 at `2^p1`. `s1` is the largest value <= 14
//!   with every |w| <= 127 * 256 + 127 (so the NEON hi/lo split fits i8)
//!   and, per neuron, `255 * Σ_i (|w_i| + 128) < 2^30` (no partial sum of
//!   any kernel, nor z1 + b1, can wrap). Trained nets clip at 1.98, which
//!   gives `s1` = 13: a step of ~1/8256, as fine as the file's 1/8192.
//!   (The first version used an i8 W1 at a per-net power-of-two scale:
//!   fully trained W1 reach the 1.98 clip but have a median |w| of ~0.04,
//!   so that scale was 2^6 and most weights kept 2-3 levels; it cost
//!   8.2 cp mean / 88 cp max against the trainer on g9h1024kb8m-head.)
//! - Kernels: NEON splits W1 into `hi * 256 + lo` (both i8) and uses two
//!   `usdot` (u8 x i8, FEAT_I8MM); AVX2 widens the 4 input bytes to i16
//!   and uses `madd` on the i16 W1; both are exact, so all targets agree.
//! - Sparse L1: the input is scanned in 4-byte chunks and only nonzero
//!   chunks are multiplied (most SCReLU outputs are 0).
//! - Final: linear + head meet at a common denominator and are rounded
//!   to the nearest centipawn.
//! - L2 / L3 in i32: activations at 2^12, weights at 2^12, sums (z2, the
//!   head output) at 2^24. Bounds: |W| <= 1.98 (trainer clip), so
//!   16 * 2^12 * 8110 and 32 * 2^12 * 8110 stay below 2^31.

use super::{BUCKETS, H, QA, QB, SCALE};

pub(crate) const H1: usize = 16;
pub(crate) const H2: usize = 32;
/// Export scale of every head parameter in the trainer's net file.
const HEAD_Q: i64 = 8192;
/// Trainer weight clip, in export units (1.98 * 8192).
const CLIP_Q: i64 = 16220;
/// L1 inputs (both perspectives) and their 4-byte chunks.
pub(crate) const N_IN: usize = 2 * H;
const CHUNKS: usize = N_IN / 4;
/// i16 words per bucket in the file.
const BUCKET_WORDS: usize = H1 * N_IN + H1 + H2 * H1 + H2 + H2 + 1;
/// i16 words the head adds to the net file.
pub(crate) const HEAD_WORDS: usize = BUCKETS * BUCKET_WORDS;
/// Activation scale (shift) of the L2 and L3 inputs.
const QS: u32 = 12;
/// Scale (shift) of z2 and the head output.
const QO: u32 = 24;

#[repr(C, align(64))]
#[derive(Clone)]
struct W1Bucket([i16; CHUNKS * H1 * 4]);

/// `W1Bucket` split into `hi * 256 + lo` (NEON `usdot`), interleaved per
/// chunk so that one chunk's weights are one 128-byte line:
/// `.0[c*128 + k*4 + j]` = hi, `.0[c*128 + 64 + k*4 + j]` = lo.
#[repr(C, align(128))]
#[derive(Clone)]
struct W1Bytes([i8; CHUNKS * H1 * 8]);

#[repr(C, align(64))]
#[derive(Clone)]
pub(crate) struct Input(pub(crate) [u8; N_IN]);

/// The L1 input (stm half first) and its nonzero-chunk list: filled by
/// `activate` or, fused into the accumulator update, by
/// `nnue::update_dot_act`, then consumed by `forward_scratch`. Not zeroed
/// per eval; only `nnz[..n]` and the chunks it lists are read.
#[derive(Clone)]
pub(crate) struct Scratch {
    pub(crate) x: Input,
    pub(crate) nnz: [u16; NNZ_BUF],
    pub(crate) n: usize,
}

impl Scratch {
    pub(crate) const fn new() -> Self {
        Self { x: Input([0; N_IN]), nnz: [0; NNZ_BUF], n: 0 }
    }
}

thread_local! {
    static SCRATCH: std::cell::UnsafeCell<Scratch> = const { std::cell::UnsafeCell::new(Scratch::new()) };
}

/// Largest |W1| in engine units for which `hi = round(w / 256)` fits i8.
const W1_MAX: i64 = 127 * 256 + 127;

#[derive(Clone)]
pub(crate) struct HeadNet {
    /// Per bucket, chunk-major: `w1[b].0[c*64 + k*4 + j]` = W1[k][4c + j].
    w1: Vec<W1Bucket>,
    /// The same, as `w1 = 256 * hi + lo` (see `W1Bytes`).
    #[allow(dead_code)] // aarch64 only
    w1hl: Vec<W1Bytes>,
    b1: [[i32; H1]; BUCKETS],
    /// z1 scale is `2^p1`.
    p1: u32,
    /// Transposed, `w2[b][k][m]` = W2[m][k], so L2 is 16 broadcast-multiply-adds.
    w2: [[[i32; H2]; H1]; BUCKETS],
    b2: [[i32; H2]; BUCKETS],
    w3: [[i32; H2]; BUCKETS],
    b3: [i32; BUCKETS],
    /// The file words, for `to_bytes`.
    raw: Vec<i16>,
}

/// `round(n / d)`, halves away from zero (d > 0).
fn round_div(n: i64, d: i64) -> i64 {
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

impl HeadNet {
    /// From the trainer's `HEAD_WORDS` i16 words (at `HEAD_Q`).
    pub(crate) fn from_words(raw: Vec<i16>) -> Self {
        Self::from_words_permuted(raw, &[])
    }

    /// `from_words` for accumulators whose lane `j` is the file's lane
    /// `perm[j]` (`nnue::lane_order`; empty = file order): W1's input
    /// columns move with the lanes, in both halves.
    pub(crate) fn from_words_permuted(raw: Vec<i16>, perm: &[usize]) -> Self {
        assert_eq!(raw.len(), HEAD_WORDS);
        assert!(perm.is_empty() || perm.len() == H);
        let mut inv: Vec<usize> = (0..H).collect();
        for (j, &old) in perm.iter().enumerate() {
            inv[old] = j;
        }
        let q = |x: i16| (x as i64).clamp(-CLIP_Q, CLIP_Q);
        // W1 file value v -> v * 2^s1 * 65536 / (65025 * 8192).
        let w1_of = |v: i64, s1: u32| round_div(v << (s1 + 16), 65025 * HEAD_Q);
        // The largest s1 <= 14 with every |w| <= W1_MAX and, per neuron,
        // 255 * Σ (|w| + 128) < 2^31 (see the module docs).
        let fits = |s1: u32| {
            (0..BUCKETS * H1).all(|n| {
                let row = &raw[(n / H1) * BUCKET_WORDS + (n % H1) * N_IN..][..N_IN];
                let w: Vec<i64> = row.iter().map(|&x| w1_of(q(x), s1).abs()).collect();
                w.iter().all(|&a| a <= W1_MAX) && 255 * w.iter().map(|a| a + 128).sum::<i64>() < 1 << 30
            })
        };
        let mut s1 = 14u32;
        while s1 > 4 && !fits(s1) {
            s1 -= 1;
        }
        assert!(fits(s1), "head W1 too large for the i32 L1");
        let p1 = 8 + s1;
        let mut h = Self {
            w1: vec![W1Bucket([0; CHUNKS * H1 * 4]); BUCKETS],
            w1hl: vec![W1Bytes([0; CHUNKS * H1 * 8]); BUCKETS],
            b1: [[0; H1]; BUCKETS],
            p1,
            w2: [[[0; H2]; H1]; BUCKETS],
            b2: [[0; H2]; BUCKETS],
            w3: [[0; H2]; BUCKETS],
            b3: [0; BUCKETS],
            raw: Vec::new(),
        };
        // 2^p1 / 8192 and 2^12 / 8192, 2^24 / 8192 conversions.
        let to = |v: i64, shift: u32| -> i32 {
            if shift >= 13 {
                (v << (shift - 13)) as i32
            } else {
                round_div(v, 1 << (13 - shift)) as i32
            }
        };
        for b in 0..BUCKETS {
            let mut it = raw[b * BUCKET_WORDS..(b + 1) * BUCKET_WORDS].iter().map(|&x| q(x));
            for k in 0..H1 {
                for f in 0..N_IN {
                    let w = w1_of(it.next().unwrap(), s1);
                    let i = f / H * H + inv[f % H];
                    let at = (i / 4) * H1 * 4 + k * 4 + i % 4;
                    // lo in -128..=127; |w| <= W1_MAX keeps |hi| <= 127.
                    let hi = (w + 128) >> 8;
                    h.w1[b].0[at] = w as i16;
                    let at2 = (i / 4) * H1 * 8 + k * 4 + i % 4;
                    h.w1hl[b].0[at2] = hi as i8;
                    h.w1hl[b].0[at2 + H1 * 4] = (w - 256 * hi) as i8;
                }
            }
            for k in 0..H1 {
                h.b1[b][k] = to(it.next().unwrap(), p1);
            }
            for m in 0..H2 {
                for k in 0..H1 {
                    h.w2[b][k][m] = to(it.next().unwrap(), QS);
                }
            }
            for m in 0..H2 {
                h.b2[b][m] = to(it.next().unwrap(), QO);
            }
            for m in 0..H2 {
                h.w3[b][m] = to(it.next().unwrap(), QS);
            }
            h.b3[b] = to(it.next().unwrap(), QO);
            debug_assert!(it.next().is_none());
        }
        h.raw = raw;
        h
    }

    #[allow(dead_code)] // to_bytes (tests)
    pub(crate) fn words(&self) -> &[i16] {
        &self.raw
    }

    /// The head's output for bucket `b` at `2^QO` (= 1.0 in trainer units).
    #[inline]
    pub(crate) fn forward(&self, b: usize, us: &[i16; H], them: &[i16; H]) -> i32 {
        SCRATCH.with(|s| {
            // SAFETY: the only borrow of this thread's scratch; forward
            // does not re-enter.
            let s = unsafe { &mut *s.get() };
            let n = kernels::activate(us, &mut s.x.0[..H], 0, &mut s.nnz, 0);
            s.n = kernels::activate(them, &mut s.x.0[H..], H / 4, &mut s.nnz, n);
            self.forward_scratch(b, s)
        })
    }

    /// The head's output from an already filled `Scratch`.
    #[inline(always)]
    pub(crate) fn forward_scratch(&self, b: usize, s: &Scratch) -> i32 {
        let mut z1 = kernels::l1(self, b, &s.x.0, &s.nnz[..s.n]);
        self.rest(b, &mut z1)
    }

    /// Scalar twin of `forward` (tests).
    #[cfg(test)]
    fn forward_scalar(&self, b: usize, us: &[i16; H], them: &[i16; H]) -> i32 {
        let mut x = Input([0; N_IN]);
        let mut nnz = [0u16; NNZ_BUF];
        let n = scalar::activate(us, &mut x.0[..H], 0, &mut nnz, 0);
        let n = scalar::activate(them, &mut x.0[H..], H / 4, &mut nnz, n);
        let mut z1 = scalar::l1(self, b, &x.0, &nnz[..n]);
        self.rest(b, &mut z1)
    }

    /// b1, the L1 activation, L2 and L3 (fixed-size i32 loops; LLVM
    /// vectorises them, and integer results do not depend on how).
    #[inline(always)]
    fn rest(&self, b: usize, z1: &mut [i32; H1]) -> i32 {
        let top = 1i32 << self.p1;
        let down = self.p1 - QS;
        let mut u1 = [0i32; H1];
        for k in 0..H1 {
            let c = (z1[k] + self.b1[b][k]).clamp(0, top) >> down;
            u1[k] = (c * c) >> QS;
        }
        let w2 = &self.w2[b];
        let mut z = self.b2[b];
        for k in 0..H1 {
            for m in 0..H2 {
                z[m] += w2[k][m] * u1[k];
            }
        }
        let mut out = [0i32; H2];
        for m in 0..H2 {
            let c = z[m].clamp(0, 1 << QO) >> (QO - QS);
            out[m] = self.w3[b][m] * ((c * c) >> QS);
        }
        self.b3[b] + out.iter().sum::<i32>()
    }
}

/// The nonzero-chunk index list, with room for an 8-index store past the end.
pub(crate) const NNZ_BUF: usize = CHUNKS + 8;

/// For an 8-bit mask of nonzero chunks, their positions (unused slots 0)
/// and their count: the kernels add the group's first chunk index to all
/// eight lanes and store 16 bytes, so appending 8 chunks is one load, add
/// and store, and the count's dependency chain is one add per 8 chunks.
#[repr(C, align(64))]
#[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(dead_code))]
struct Lut8([[u16; 8]; 256], [u8; 256]);

#[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(dead_code))]
static NNZ_LUT8: Lut8 = {
    let mut t = Lut8([[0u16; 8]; 256], [0u8; 256]);
    let mut m = 0;
    while m < 256 {
        let (mut j, mut n) = (0, 0);
        while j < 8 {
            if m >> j & 1 == 1 {
                t.0[m][n] = j as u16;
                n += 1;
            }
            j += 1;
        }
        t.1[m] = n as u8;
        m += 1;
    }
    t
};

/// The L1 input of one perspective into `out` (H bytes), its nonzero chunks
/// (numbered from `base`) appended to `nnz[n..]`; returns the new count.
#[inline(always)]
pub(crate) fn activate(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], n: usize) -> usize {
    kernels::activate(acc, out, base, nnz, n)
}

/// Append the nonzero chunks among `mask` (8 bits, chunks `c0..c0 + 8`)
/// to `nnz[n..]`; returns the new count.
#[inline(always)]
#[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(dead_code))]
pub(crate) fn push_nnz8(nnz: &mut [u16; NNZ_BUF], n: usize, c0: usize, mask: u32) -> usize {
    debug_assert!(mask < 256 && n + 8 <= NNZ_BUF);
    let m = mask as usize & 255;
    // SAFETY: at most CHUNKS - 8 chunks precede these 8, so n + 8 <= NNZ_BUF.
    unsafe {
        let dst = nnz.as_mut_ptr().add(n);
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            vst1q_u16(dst, vaddq_u16(vld1q_u16(NNZ_LUT8.0[m].as_ptr()), vdupq_n_u16(c0 as u16)));
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let v = _mm_add_epi16(_mm_loadu_si128(NNZ_LUT8.0[m].as_ptr() as *const __m128i), _mm_set1_epi16(c0 as i16));
            _mm_storeu_si128(dst as *mut __m128i, v);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        for j in 0..8 {
            *dst.add(j) = NNZ_LUT8.0[m][j] + c0 as u16;
        }
    }
    n + NNZ_LUT8.1[m] as usize
}

/// Linear output sum (`output_sum`) + head output -> clamped centipawns,
/// replacing `nnue::finish` when the head is on: the two terms meet at a
/// common denominator before the one rounding division (the trainer's
/// float eval is matched to the nearest centipawn, not truncated).
#[inline(always)]
pub(crate) fn finish(lin_b: i16, sum: i64, head: i32, limit: i32) -> i32 {
    let qq = QA as i64 * QB as i64;
    let num = (sum / QA as i64 + lin_b as i64) * (1i64 << QO) + head as i64 * qq;
    let raw = round_div(num * SCALE as i64, qq << QO);
    raw.clamp(-limit as i64, limit as i64) as i32
}

/// Scalar twins (the reference; the only kernels on other targets).
#[allow(dead_code)]
pub(crate) mod scalar {
    use super::*;

    /// The L1 input of one perspective into `out` (H bytes), and its
    /// nonzero 4-byte chunks (numbered from `base`) appended to `nnz[n..]`;
    /// returns the new count.
    #[inline(always)]
    pub(crate) fn activate(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], mut n: usize) -> usize {
        for (o, &a) in out.iter_mut().zip(acc.iter()) {
            let c = (a as i32).clamp(0, 255);
            *o = ((c * c + 128) >> 8) as u8;
        }
        for c in 0..H / 4 {
            if out[4 * c..4 * c + 4] != [0; 4] {
                nnz[n] = (base + c) as u16;
                n += 1;
            }
        }
        n
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
mod kernels {
    use super::*;
    use std::arch::aarch64::*;

    #[inline(always)]
    pub(super) fn activate(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], mut n: usize) -> usize {
        assert!(out.len() == H && H % 64 == 0);
        // SAFETY: H % 64 == 0, all loads/stores in bounds.
        unsafe {
            let (lo, hi) = (vdupq_n_s16(0), vdupq_n_s16(255));
            let bits = vld1q_u32([1u32, 2, 4, 8].as_ptr());
            // 64 inputs (16 chunks) per step: one horizontal add for the mask.
            for i in (0..H).step_by(64) {
                let mut m = vdupq_n_u32(0);
                for g in 0..4 {
                    let p = acc.as_ptr().add(i + 16 * g);
                    // c^2 <= 65025 fits u16; rshrn = (c^2 + 128) >> 8.
                    let a = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(p), lo), hi));
                    let b = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(p.add(8)), lo), hi));
                    let v = vrshrn_high_n_u16::<8>(vrshrn_n_u16::<8>(vmulq_u16(a, a)), vmulq_u16(b, b));
                    vst1q_u8(out.as_mut_ptr().add(i + 16 * g), v);
                    let v32 = vreinterpretq_u32_u8(v);
                    m = vorrq_u32(m, vshlq_u32(vandq_u32(vtstq_u32(v32, v32), bits), vdupq_n_s32(4 * g as i32)));
                }
                let mask = vaddvq_u32(m);
                n = push_nnz8(nnz, n, base + i / 4, mask & 255);
                n = push_nnz8(nnz, n, base + i / 4 + 8, mask >> 8);
            }
        }
        n
    }

    #[inline(always)]
    fn i8mm() -> bool {
        if cfg!(target_feature = "i8mm") {
            return true;
        }
        static HAVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *HAVE.get_or_init(|| std::arch::is_aarch64_feature_detected!("i8mm"))
    }

    #[inline(always)]
    pub(super) fn l1(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        if i8mm() {
            // SAFETY: FEAT_I8MM is available.
            unsafe { l1_usdot(&net.w1hl[b].0, x, nnz) }
        } else {
            scalar::l1(net, b, x, nnz)
        }
    }

    /// `acc += Σ_j x[4l+j] * w[4l+j]` per i32 lane, x unsigned (`usdot`).
    #[inline(always)]
    unsafe fn usdot(mut acc: int32x4_t, x: uint8x16_t, w: int8x16_t) -> int32x4_t {
        // SAFETY: register-only instruction; callers enable i8mm.
        unsafe {
            std::arch::asm!("usdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) acc, in(vreg) x, in(vreg) w, options(pure, nomem, nostack));
        }
        acc
    }

    /// z = 256 * Σ x hi + Σ x lo; both partial sums are bounded by the
    /// load-time check, so neither wraps.
    #[target_feature(enable = "i8mm")]
    unsafe fn l1_usdot(w: &[i8; CHUNKS * H1 * 8], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: every chunk index is < CHUNKS, so all loads are in bounds.
        unsafe {
            // Two chunks in flight: 16 accumulators hide the usdot latency.
            let z = vdupq_n_s32(0);
            let (mut h, mut l) = ([[z; 4]; 2], [[z; 4]; 2]);
            let xp = x.as_ptr() as *const u32;
            let wp = w.as_ptr();
            let mut i = 0;
            while i + 1 < nnz.len() {
                for s in 0..2 {
                    let c = nnz[i + s] as usize;
                    let xv = vreinterpretq_u8_u32(vdupq_n_u32(xp.add(c).read_unaligned()));
                    let (hc, lc) = (wp.add(c * 128), wp.add(c * 128 + 64));
                    let (wh, wl) = (vld1q_s8_x4(hc), vld1q_s8_x4(lc));
                    let (wh, wl) = ([wh.0, wh.1, wh.2, wh.3], [wl.0, wl.1, wl.2, wl.3]);
                    for r in 0..4 {
                        h[s][r] = usdot(h[s][r], xv, wh[r]);
                        l[s][r] = usdot(l[s][r], xv, wl[r]);
                    }
                }
                i += 2;
            }
            if i < nnz.len() {
                let c = nnz[i] as usize;
                let xv = vreinterpretq_u8_u32(vdupq_n_u32(xp.add(c).read_unaligned()));
                let (hc, lc) = (wp.add(c * 128), wp.add(c * 128 + 64));
                for r in 0..4 {
                    h[0][r] = usdot(h[0][r], xv, vld1q_s8(hc.add(16 * r)));
                    l[0][r] = usdot(l[0][r], xv, vld1q_s8(lc.add(16 * r)));
                }
            }
            let mut out = [0i32; H1];
            for r in 0..4 {
                let hv = vaddq_s32(h[0][r], h[1][r]);
                let lv = vaddq_s32(l[0][r], l[1][r]);
                vst1q_s32(out.as_mut_ptr().add(4 * r), vaddq_s32(vshlq_n_s32::<8>(hv), lv));
            }
            out
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod kernels {
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

    #[inline(always)]
    pub(super) fn activate(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], n: usize) -> usize {
        if avx2() {
            // SAFETY: AVX2 is available.
            unsafe { activate_avx2(acc, out, base, nnz, n) }
        } else {
            scalar::activate(acc, out, base, nnz, n)
        }
    }

    #[inline(always)]
    pub(super) fn l1(net: &HeadNet, b: usize, x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        if avx2() {
            // SAFETY: AVX2 is available.
            unsafe { l1_avx2(&net.w1[b].0, x, nnz) }
        } else {
            scalar::l1(net, b, x, nnz)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn activate_avx2(acc: &[i16; H], out: &mut [u8], base: usize, nnz: &mut [u16; NNZ_BUF], mut n: usize) -> usize {
        assert!(out.len() == H && H % 32 == 0);
        // SAFETY: H % 32 == 0, all loads/stores in bounds.
        unsafe {
            let (lo, hi) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
            for i in (0..H).step_by(32) {
                let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
                let b = _mm256_loadu_si256(acc.as_ptr().add(i + 16) as *const __m256i);
                // c^2 <= 65025 is exact as u16 (mullo), + 128 <= 65153: (c^2 + 128) >> 8.
                let a = _mm256_min_epi16(_mm256_max_epi16(a, lo), hi);
                let b = _mm256_min_epi16(_mm256_max_epi16(b, lo), hi);
                let r = _mm256_set1_epi16(128);
                let a = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(a, a), r));
                let b = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(b, b), r));
                // packus interleaves 128-bit lanes; restore the order.
                let p = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(a, b));
                _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, p);
                let zero = _mm256_cmpeq_epi32(p, lo);
                let mask = !(_mm256_movemask_ps(_mm256_castsi256_ps(zero)) as u32) & 0xFF;
                n = push_nnz8(nnz, n, base + i / 4, mask);
            }
        }
        n
    }

    #[target_feature(enable = "avx2")]
    unsafe fn l1_avx2(w: &[i16; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: every chunk index is < CHUNKS, so all loads are in bounds.
        // The chunk's 4 input bytes widen to i16 and repeat in every 64-bit
        // lane; madd against W1[k][4c..4c+4] (4 neurons per vector) leaves
        // neuron 4r + l/2's two half sums in lanes l of a[r]. Products
        // <= 254 * 32767, pair sums fit i32.
        unsafe {
            let mut a = [_mm256_setzero_si256(); 4];
            let mut b = [_mm256_setzero_si256(); 4];
            let xp = x.as_ptr() as *const i32;
            let wp = w.as_ptr() as *const __m256i;
            let widen = |c: usize| _mm256_broadcastq_epi64(_mm_cvtepu8_epi16(_mm_cvtsi32_si128(xp.add(c).read_unaligned())));
            let mut i = 0;
            while i + 1 < nnz.len() {
                let (c0, c1) = (nnz[i] as usize, nnz[i + 1] as usize);
                let (x0, x1) = (widen(c0), widen(c1));
                for r in 0..4 {
                    a[r] = _mm256_add_epi32(a[r], _mm256_madd_epi16(x0, _mm256_load_si256(wp.add(c0 * 4 + r))));
                    b[r] = _mm256_add_epi32(b[r], _mm256_madd_epi16(x1, _mm256_load_si256(wp.add(c1 * 4 + r))));
                }
                i += 2;
            }
            if i < nnz.len() {
                let c0 = nnz[i] as usize;
                let x0 = widen(c0);
                for r in 0..4 {
                    a[r] = _mm256_add_epi32(a[r], _mm256_madd_epi16(x0, _mm256_load_si256(wp.add(c0 * 4 + r))));
                }
            }
            let mut t = [0i32; 2 * H1];
            for r in 0..4 {
                _mm256_storeu_si256(t.as_mut_ptr().add(8 * r) as *mut __m256i, _mm256_add_epi32(a[r], b[r]));
            }
            std::array::from_fn(|k| t[2 * k] + t[2 * k + 1])
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
mod kernels {
    pub(super) use super::scalar::{activate, l1};
}

#[cfg(test)]
/// Head words drawn like trained nets (W1 small), at HEAD_Q.
pub(crate) fn random_words(seed: u64) -> Vec<i16> {
    let mut s = seed | 1;
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut u = move |range: f64| -> i16 { (((r() % 20001) as f64 / 10000.0 - 1.0) * range * 8192.0).round() as i16 };
    let mut v = Vec::with_capacity(HEAD_WORDS);
    for _ in 0..BUCKETS {
        (0..H1 * N_IN).for_each(|_| v.push(u(0.07)));
        (0..H1).for_each(|_| v.push(u(0.3)));
        (0..H2 * H1).for_each(|_| v.push(u(0.3)));
        (0..H2).for_each(|_| v.push(u(0.3)));
        (0..H2).for_each(|_| v.push(u(0.3)));
        v.push(u(0.1));
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

    /// The trainer's float head on the (already quantised) accumulators.
    fn float_head(raw: &[i16], b: usize, us: &[i16; H], them: &[i16; H]) -> f64 {
        let p: Vec<f64> = raw[b * BUCKET_WORDS..(b + 1) * BUCKET_WORDS].iter().map(|&x| x as f64 / HEAD_Q as f64).collect();
        let a: Vec<f64> = us.iter().chain(them.iter()).map(|&x| sq(x as f64 / QA as f64)).collect();
        let (w1, rest) = p.split_at(H1 * N_IN);
        let (b1, rest) = rest.split_at(H1);
        let (w2, rest) = rest.split_at(H2 * H1);
        let (b2, rest) = rest.split_at(H2);
        let (w3, b3) = rest.split_at(H2);
        let z1: Vec<f64> = (0..H1).map(|k| b1[k] + (0..N_IN).map(|i| w1[k * N_IN + i] * a[i]).sum::<f64>()).collect();
        let mut out = b3[0];
        for m in 0..H2 {
            let z2 = b2[m] + (0..H1).map(|k| w2[m * H1 + k] * sq(z1[k])).sum::<f64>();
            out += w3[m] * sq(z2);
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
            assert_eq!(q, head.forward_scalar(b, &us, &them), "case {t}");
            let f = float_head(&head.raw, b, &us, &them);
            let d = (q as f64 / (1u64 << QO) as f64 - f).abs() * SCALE as f64;
            worst = worst.max(d);
        }
        // Quantisation error in centipawns (the head output * 400).
        assert!(worst < 10.0, "worst head error {worst:.2} cp");
    }

    #[test]
    fn activation_kernel_matches_scalar() {
        let mut acc = [0i16; H];
        for (i, a) in acc.iter_mut().enumerate() {
            *a = (i as i32 * 37 % 700 - 200) as i16;
        }
        acc[0] = i16::MIN;
        acc[1] = i16::MAX;
        let (mut s, mut v) = ([0u8; H], [0u8; H]);
        let (mut ns, mut nv) = ([0u16; NNZ_BUF], [0u16; NNZ_BUF]);
        let a = scalar::activate(&acc, &mut s, 7, &mut ns, 3);
        let b = kernels::activate(&acc, &mut v, 7, &mut nv, 3);
        assert_eq!(s, v);
        assert_eq!(s[1], 254);
        assert_eq!((a, &ns[3..a]), (b, &nv[3..b]));
        assert!(a > 3 + H / 8 && a < 3 + H / 4);
    }

    #[test]
    fn finish_combines_exactly() {
        // Head of exactly 1.0 = 400 cp; linear part as nnue::finish, but
        // rounded to the nearest centipawn.
        assert_eq!(finish(0, 0, 1 << QO, 20000), 400);
        assert_eq!(finish(0, 0, -(1 << QO), 20000), -400);
        let sum = 123_456_789i64;
        let lin = (sum / QA as i64 + 77) as f64 * SCALE as f64 / (QA as i64 * QB as i64) as f64;
        assert_eq!(finish(77, sum, 0, 20000) as i64, lin.round() as i64);
        assert_eq!(finish(77, -sum, 0, 20000) as i64, ((-sum / QA as i64 + 77) as f64 * SCALE as f64 / (QA * QB) as f64).round() as i64);
        // Half a centipawn of head rounds away from zero.
        let half = ((1i64 << QO) + 799) / 800;
        assert_eq!(finish(0, 0, half as i32, 20000), 1);
    }

    /// How sparse the L1 input would be on the embedded net's accumulators
    /// (`cargo test --release head_input_density -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn head_input_density() {
        use super::super::{Accumulator, Network};
        let net = Network::from_bytes(crate::EMBEDDED_NET).unwrap();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        ];
        let (mut el, mut ch, mut tot) = (0usize, 0usize, 0usize);
        for f in fens {
            let mut b = crate::Board::new();
            b.set_fen(f);
            let acc = Accumulator::refresh(&net, &b);
            for v in &acc.v {
                let (mut x, mut idx) = ([0u8; H], [0u16; NNZ_BUF]);
                ch += scalar::activate(v, &mut x, 0, &mut idx, 0);
                el += x.iter().filter(|&&q| q != 0).count();
                tot += H;
            }
        }
        println!("embedded net: nonzero L1 inputs {:.0}%, nonzero 4-byte chunks {:.0}%", 100.0 * el as f64 / tot as f64, 400.0 * ch as f64 / tot as f64);
    }

    /// `cargo test --release head_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn head_throughput() {
        use std::time::Instant;
        let head = HeadNet::from_words(random_words(5));
        let mut r = rng(9);
        let t = Instant::now();
        let mut z = [0i32; H1];
        let mut acc_r = 0i64;
        for i in 0..2_000_000 {
            z[i % H1] = (i as i32 * 977) & 0x3ffff;
            acc_r += head.rest(i % BUCKETS, std::hint::black_box(&mut z)) as i64;
        }
        println!("head: b1 + L2 + L3 alone {:.1} ns ({acc_r})", t.elapsed().as_secs_f64() * 1e9 / 2e6);
        for zero in [50u64, 70, 90] {
            let accs: Vec<([i16; H], [i16; H])> = (0..64).map(|_| (random_acc(&mut r, zero), random_acc(&mut r, zero))).collect();
            let mut nz = 0usize;
            let mut idx = [0u16; NNZ_BUF];
            for (u, t) in &accs {
                let mut x = Input([0; N_IN]);
                let n = scalar::activate(u, &mut x.0[..H], 0, &mut idx, 0);
                nz += scalar::activate(t, &mut x.0[H..], H / 4, &mut idx, n);
            }
            let n = 2_000_000;
            let mut s = 0i64;
            let t = Instant::now();
            for i in 0..n {
                let (u, th) = &accs[i % accs.len()];
                s += head.forward(i / 64 % BUCKETS, std::hint::black_box(u), std::hint::black_box(th)) as i64;
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            let mut x = Input([0; N_IN]);
            let t = Instant::now();
            for i in 0..n {
                let (u, th) = &accs[i % accs.len()];
                let k = kernels::activate(std::hint::black_box(u), &mut x.0[..H], 0, &mut idx, 0);
                let k = kernels::activate(std::hint::black_box(th), &mut x.0[H..], H / 4, &mut idx, k);
                s += std::hint::black_box(&x).0[i % N_IN] as i64 + k as i64;
            }
            let ns_act = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            println!("head: {zero}% zero inputs, {:.0}/{CHUNKS} nonzero chunks: forward {ns:.1} ns (activate + nnz {ns_act:.1})", nz as f64 / accs.len() as f64);
            std::hint::black_box(s);
        }
    }
}

/// Engine vs trainer on a real `--head` net. The gate for head nets: with
/// `HEAD = true`, `cargo test` (CI, the queue, `netloop.py embed`) fails
/// unless `net/head_fixture.txt` holds the trainer's float evals of the
/// embedded net (`<FEN> | <cp>` lines from `nnue-trainer fen-eval`, 2000
/// validation positions; `scripts/head_check.py --write-fixture` makes it)
/// and the engine agrees with them: mean |diff| < 1 cp, p95 < 3 cp, max
/// < 10 cp. `RUSTFISH_HEAD_FIXTURE` / `RUSTFISH_HEAD_NET` override the
/// fixture / net.
#[cfg(test)]
mod fixture {
    use super::super::{Accumulator, Active, Network, NnueState, HEAD, H};
    use super::{scalar, CHUNKS, NNZ_BUF};
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
            let (mut x, mut idx) = ([0u8; H], [0u16; NNZ_BUF]);
            nnz_sum += scalar::activate(&acc.v[0], &mut x, 0, &mut idx, 0) + scalar::activate(&acc.v[1], &mut x, 0, &mut idx, 0);
            // The SIMD kernels match the scalar reference on real accumulators.
            let b = super::super::bucket(board.bb.occ_all().count_ones());
            let (u, t) = (&acc.v[0], &acc.v[1]);
            assert_eq!(head.forward(b, u, t), head.forward_scalar(b, u, t), "{fen}");
            assert_eq!(head.forward(b, t, u), head.forward_scalar(b, t, u), "{fen}");
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
        eprintln!("head fixture: nonzero L1 input chunks {nnz:.0} of {CHUNKS} ({:.0}%)", 100.0 * nnz / CHUNKS as f64);
        eprintln!("head fixture: {n} positions, |engine - trainer| mean {mean:.2} cp, p95 {p95:.2}, max {worst:.2}, bias {bias:+.2}; worst {worst_fen}");
        // Expected: ~0.5 cp mean (half of it the rounding to whole cp, most
        // of the rest the 8-bit L1 input), p95 ~1.3, max ~6 (g9h1024kb8m-head).
        assert!(mean < 1.0 && p95 < 3.0 && worst < 10.0, "engine and trainer disagree beyond quantisation error");
    }

    /// Experiment: nonzero-chunk density under lane permutations.
    #[test]
    #[ignore]
    fn head_perm_density() {
        let net = Arc::new(Network::from_bytes(crate::EMBEDDED_NET).unwrap());
        let text = std::fs::read_to_string(FIXTURE).unwrap();
        let mut samples: Vec<Vec<bool>> = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (fen, _) = line.rsplit_once('|').unwrap();
            let mut board = Board::new();
            board.set_fen(fen.trim());
            let acc = Accumulator::refresh(&net, &board);
            for v in &acc.v {
                samples.push(v.iter().map(|&a| a >= 12).collect());
            }
        }
        let (train, test): (Vec<_>, Vec<_>) = samples.iter().enumerate().partition(|(i, _)| (i / 2) % 2 == 0);
        let train: Vec<&Vec<bool>> = train.into_iter().map(|x| x.1).collect();
        let test: Vec<&Vec<bool>> = test.into_iter().map(|x| x.1).collect();
        let density = |perm: &[usize], set: &[&Vec<bool>]| -> f64 {
            let mut ch = 0usize;
            for s in set {
                for c in 0..H / 4 {
                    if (0..4).any(|j| s[perm[4 * c + j]]) {
                        ch += 1;
                    }
                }
            }
            ch as f64 / (set.len() * H / 4) as f64
        };
        let ident: Vec<usize> = (0..H).collect();
        let freq: Vec<usize> = (0..H).map(|i| train.iter().filter(|s| s[i]).count()).collect();
        let mut by_freq = ident.clone();
        by_freq.sort_by_key(|&i| std::cmp::Reverse(freq[i]));
        // Greedy co-activation: seed with the most frequent remaining lane,
        // add the lanes that grow the chunk's active set least.
        let words = train.len().div_ceil(64);
        let bits: Vec<Vec<u64>> = (0..H)
            .map(|i| {
                let mut b = vec![0u64; words];
                for (k, s) in train.iter().enumerate() {
                    if s[i] {
                        b[k / 64] |= 1 << (k % 64);
                    }
                }
                b
            })
            .collect();
        let mut left: Vec<usize> = by_freq.clone();
        let mut greedy = Vec::new();
        while !left.is_empty() {
            let seed = left.remove(0);
            let mut set = bits[seed].clone();
            greedy.push(seed);
            for _ in 0..3 {
                let (bi, _) = left
                    .iter()
                    .enumerate()
                    .map(|(k, &i)| (k, set.iter().zip(&bits[i]).map(|(a, b)| (a | b).count_ones()).sum::<u32>()))
                    .min_by_key(|x| x.1)
                    .unwrap();
                let i = left.remove(bi);
                for (a, b) in set.iter_mut().zip(&bits[i]) {
                    *a |= b;
                }
                greedy.push(i);
            }
        }
        // Jaccard matching, twice: lanes -> pairs -> quads.
        let jac = |a: &[u64], b: &[u64]| -> f64 {
            let (i, u) = a.iter().zip(b).fold((0u32, 0u32), |(i, u), (x, y)| (i + (x & y).count_ones(), u + (x | y).count_ones()));
            if u == 0 { 1.0 } else { i as f64 / u as f64 }
        };
        let mut groups: Vec<(Vec<usize>, Vec<u64>)> = by_freq.iter().map(|&i| (vec![i], bits[i].clone())).collect();
        for _ in 0..2 {
            let mut next = Vec::new();
            while !groups.is_empty() {
                let (g, b) = groups.remove(0);
                let (k, _) = groups.iter().enumerate().map(|(k, (_, b2))| (k, jac(&b, b2))).max_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap();
                let (g2, b2) = groups.remove(k);
                let u: Vec<u64> = b.iter().zip(&b2).map(|(x, y)| x | y).collect();
                next.push(([g, g2].concat(), u));
            }
            next.sort_by_key(|(_, b)| std::cmp::Reverse(b.iter().map(|x| x.count_ones()).sum::<u32>()));
            groups = next;
        }
        let jperm: Vec<usize> = groups.into_iter().flat_map(|(g, _)| g).collect();
        let el = test.iter().map(|s| s.iter().filter(|&&b| b).count()).sum::<usize>() as f64 / (test.len() * H) as f64;
        eprintln!("elements {:.3}; chunks: identity {:.3}, by-freq {:.3}, greedy {:.3}, jaccard {:.3} (train: {:.3} {:.3} {:.3} {:.3})",
            el, density(&ident, &test), density(&by_freq, &test), density(&greedy, &test), density(&jperm, &test),
            density(&ident, &train), density(&by_freq, &train), density(&greedy, &train), density(&jperm, &train));
        let dead = freq.iter().filter(|&&f| f == 0).count();
        eprintln!("dead lanes (never active in train): {dead}");
    }

    /// Per-stage error budget on the fixture's positions (diagnostic):
    /// `cargo test --release head_error_budget -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn head_error_budget() {
        use super::super::{QA, QB, SCALE};
        use super::{HeadNet, BUCKET_WORDS, H1, H2, HEAD_Q, N_IN, QO};
        let path = std::env::var("RUSTFISH_HEAD_FIXTURE").unwrap_or_else(|_| FIXTURE.to_string());
        let net = Arc::new(Network::from_bytes(crate::EMBEDDED_NET).unwrap());
        let head: &HeadNet = net.head.as_ref().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let sq = |x: f64| x.clamp(0.0, 1.0).powi(2);
        let names = ["float-vs-trainer", "lin int-vs-float", "in-quant only", "W1-engine only", "L2/L3 int only", "engine head", "engine total"];
        let mut errs: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
        let mut extra: Vec<Vec<f64>> = vec![Vec::new(); 4];
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (fen, cp) = line.rsplit_once('|').unwrap();
            let want: f64 = cp.trim().parse().unwrap();
            if want.abs() > 15000.0 {
                continue;
            }
            let mut board = Board::new();
            board.set_fen(fen.trim());
            let acc = Accumulator::refresh(&net, &board);
            let (us, them) = match board.side_to_move {
                crate::Color::White => (&acc.v[0], &acc.v[1]),
                crate::Color::Black => (&acc.v[1], &acc.v[0]),
            };
            let b = super::super::bucket(board.bb.occ_all().count_ones());
            let ow = &net.out_w[b * 2 * H..(b + 1) * 2 * H];
            let a_lin: Vec<f64> = us.iter().chain(them.iter()).map(|&x| sq(x as f64 / QA as f64)).collect();
            let lin_f = net.out_b[b] as f64 / (QA * QB) as f64 + a_lin.iter().zip(ow.iter()).map(|(x, &w)| x * w as f64 / QB as f64).sum::<f64>();
            // The accumulators in file order, for the head's file weights.
            let (mut usf, mut thf) = ([0i16; H], [0i16; H]);
            for j in 0..H {
                usf[net.file_lane(j)] = us[j];
                thf[net.file_lane(j)] = them[j];
            }
            let a: Vec<f64> = usf.iter().chain(thf.iter()).map(|&x| sq(x as f64 / QA as f64)).collect();
            let sum = super::super::output_sum(&net, b, us, them);
            let lin_i = ((sum / QA as i64 + net.out_b[b] as i64) * SCALE as i64 / (QA as i64 * QB as i64)) as f64;
            let p: Vec<f64> = head.raw[b * BUCKET_WORDS..(b + 1) * BUCKET_WORDS].iter().map(|&x| x as f64 / HEAD_Q as f64).collect();
            let (w1, rest) = p.split_at(H1 * N_IN);
            let (b1, rest) = rest.split_at(H1);
            let (w2, rest) = rest.split_at(H2 * H1);
            let (b2, rest) = rest.split_at(H2);
            let (w3, b3) = rest.split_at(H2);
            let s1 = head.p1 as i32 - 8;
            let k1 = 2f64.powi(s1) * 65536.0 / 65025.0;
            let w1q = |w: f64| (w * k1).round() / k1;
            let aq: Vec<f64> = usf
                .iter()
                .chain(thf.iter())
                .map(|&x| {
                    let c = (x as i32).clamp(0, 255);
                    ((c * c + 128) >> 8) as f64 * 256.0 / 65025.0
                })
                .collect();
            let z1 = |inp: &[f64], wq: bool| -> Vec<f64> {
                (0..H1).map(|k| b1[k] + (0..N_IN).map(|i| (if wq { w1q(w1[k * N_IN + i]) } else { w1[k * N_IN + i] }) * inp[i]).sum::<f64>()).collect()
            };
            let tail = |z1: &[f64]| -> f64 {
                let mut out = b3[0];
                for m in 0..H2 {
                    let z2 = b2[m] + (0..H1).map(|k| w2[m * H1 + k] * sq(z1[k])).sum::<f64>();
                    out += w3[m] * sq(z2);
                }
                out
            };
            let tail_int = |z1: &[f64]| -> f64 {
                let mut zi = [0i32; H1];
                for k in 0..H1 {
                    zi[k] = ((z1[k] - b1[k]) * 2f64.powi(head.p1 as i32)).round() as i32;
                }
                head.rest(b, &mut zi) as f64 / (1u64 << QO) as f64
            };
            let zf = z1(&a, false);
            let hf = tail(&zf);
            let s = SCALE as f64;
            let hq = head.forward(b, us, them) as f64 / (1u64 << QO) as f64;
            let got = NnueState::<Active>::new(net.clone(), &board).evaluate(board.side_to_move, board.bb.occ_all().count_ones()) as f64;
            // Candidate designs: W1 exact, input c^2 >> k (rounded), engine
            // L2/L3, linear integer sum, final round-to-nearest.
            for (j, k) in [9u32, 8, 7, 6].into_iter().enumerate() {
                let ak: Vec<f64> = usf
                    .iter()
                    .chain(thf.iter())
                    .map(|&x| {
                        let c = (x as i32).clamp(0, 255);
                        ((c * c + (1 << (k - 1))) >> k) as f64 * (1 << k) as f64 / 65025.0
                    })
                    .collect();
                let h = tail_int(&z1(&ak, false));
                let lin = (sum / QA as i64 + net.out_b[b] as i64) as f64 / (QA * QB) as f64;
                let tot = ((lin + h) * s).round();
                extra[j].push(tot - want);
            }
            let v = [
                (lin_f + hf) * s - want,
                lin_i - lin_f * s,
                (tail(&z1(&aq, false)) - hf) * s,
                (tail(&z1(&a, true)) - hf) * s,
                (tail_int(&zf) - hf) * s,
                (hq - hf) * s,
                got - want,
            ];
            for (e, x) in errs.iter_mut().zip(v) {
                e.push(x);
            }
        }
        let xn = ["W1 exact in>>9 round", "W1 exact in>>8 round", "W1 exact in>>7 round", "W1 exact in>>6 round"];
        for (n, e) in xn.iter().zip(extra.iter_mut()) {
            let k = e.len();
            let bias = e.iter().sum::<f64>() / k as f64;
            e.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
            let mean = e.iter().map(|x| x.abs()).sum::<f64>() / k as f64;
            eprintln!("budget {n:>18}: mean {mean:7.3} p95 {:7.3} max {:7.3} bias {bias:+.3}", e[k * 95 / 100].abs(), e[k - 1].abs());
        }
        for (n, e) in names.iter().zip(errs.iter_mut()) {
            let k = e.len();
            let bias = e.iter().sum::<f64>() / k as f64;
            e.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
            let mean = e.iter().map(|x| x.abs()).sum::<f64>() / k as f64;
            eprintln!("budget {n:>18}: mean {mean:7.3} p95 {:7.3} max {:7.3} bias {bias:+.3}", e[k * 95 / 100].abs(), e[k - 1].abs());
        }
    }
}
