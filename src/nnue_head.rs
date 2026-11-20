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
//! - L1 input: `x = (clamp(acc, 0, 255)^2 + 256) >> 9` in 0..=127 (u8),
//!   i.e. `sq(a) * 65025 / 512` rounded. 0..127 keeps `maddubs` from saturating and
//!   lets NEON use a signed `sdot`.
//! - W1: i8 at `2^s1 * 65536 / 65025`, `s1` chosen per net at load (the
//!   largest power of two that keeps every |W1| <= 127: trained W1 are
//!   small, ~0.02 mean, ~0.07 max), so `Σ x w` is `z1` at `2^p1`,
//!   `p1 = 7 + s1`. b1 in i32 at `2^p1`.
//! - Sparse L1: the input is scanned in 4-byte chunks and only nonzero
//!   chunks are multiplied (most SCReLU outputs are 0).
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
const N_IN: usize = 2 * H;
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
struct W1Bucket([i8; CHUNKS * H1 * 4]);

#[repr(C, align(64))]
struct Input([u8; N_IN]);

#[derive(Clone)]
pub(crate) struct HeadNet {
    /// Per bucket, chunk-major: `w1[b].0[c*64 + k*4 + j]` = W1[k][4c + j].
    w1: Vec<W1Bucket>,
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
        assert_eq!(raw.len(), HEAD_WORDS);
        let q = |x: i16| (x as i64).clamp(-CLIP_Q, CLIP_Q);
        let mut max_w1 = 0i64;
        for b in 0..BUCKETS {
            for &x in &raw[b * BUCKET_WORDS..b * BUCKET_WORDS + H1 * N_IN] {
                max_w1 = max_w1.max(q(x).abs());
            }
        }
        // W1 file value v -> v * 2^s1 * 65536 / (65025 * 8192).
        let w1_of = |v: i64, s1: u32| round_div(v << (s1 + 16), 65025 * HEAD_Q);
        let mut s1 = 14u32;
        while s1 > 6 && w1_of(max_w1, s1) > 127 {
            s1 -= 1;
        }
        let p1 = 7 + s1;
        let mut h = Self {
            w1: vec![W1Bucket([0; CHUNKS * H1 * 4]); BUCKETS],
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
                for i in 0..N_IN {
                    let w = w1_of(it.next().unwrap(), s1).clamp(-127, 127);
                    h.w1[b].0[(i / 4) * H1 * 4 + k * 4 + i % 4] = w as i8;
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
        let mut x = Input([0; N_IN]);
        let mut nnz = [0u16; NNZ_BUF];
        let n = kernels::activate(us, &mut x.0[..H], 0, &mut nnz, 0);
        let n = kernels::activate(them, &mut x.0[H..], H / 4, &mut nnz, n);
        let mut z1 = kernels::l1(&self.w1[b].0, &x.0, &nnz[..n]);
        self.rest(b, &mut z1)
    }

    /// Scalar twin of `forward` (tests).
    #[cfg(test)]
    fn forward_scalar(&self, b: usize, us: &[i16; H], them: &[i16; H]) -> i32 {
        let mut x = Input([0; N_IN]);
        let mut nnz = [0u16; NNZ_BUF];
        let n = scalar::activate(us, &mut x.0[..H], 0, &mut nnz, 0);
        let n = scalar::activate(them, &mut x.0[H..], H / 4, &mut nnz, n);
        let mut z1 = scalar::l1(&self.w1[b].0, &x.0, &nnz[..n]);
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

/// The nonzero-chunk index list, with room for a 4-index store past the end.
const NNZ_BUF: usize = CHUNKS + 8;

/// For a 4-bit mask of nonzero chunks, their positions as four packed u16
/// (unused slots 0): the SIMD kernels add the group's first chunk index to
/// all four lanes at once (no carries: indices < 2^16) and store 8 bytes.
const NNZ_LUT: [u64; 16] = {
    let mut t = [0u64; 16];
    let mut m = 0;
    while m < 16 {
        let (mut j, mut n) = (0, 0);
        while j < 4 {
            if m >> j & 1 == 1 {
                t[m] |= (j as u64) << (16 * n);
                n += 1;
            }
            j += 1;
        }
        m += 1;
    }
    t
};

/// Append the nonzero chunks among `mask` (4 bits, chunks `c0..c0 + 4`).
#[inline(always)]
fn push_nnz(nnz: &mut [u16; NNZ_BUF], n: usize, c0: usize, mask: u32) -> usize {
    let v = NNZ_LUT[mask as usize] + (c0 as u64) * 0x0001_0001_0001_0001;
    debug_assert!(n + 4 <= NNZ_BUF);
    // SAFETY: n <= CHUNKS, so n + 4 <= NNZ_BUF.
    unsafe { (nnz.as_mut_ptr().add(n) as *mut u64).write_unaligned(v.to_le()) };
    // A nibble table, not count_ones (which goes through the vector unit on aarch64).
    n + (0x4332_3221_3221_2110u64 >> (4 * mask) & 15) as usize
}

/// Linear output sum (`output_sum`) + head output -> clamped centipawns,
/// replacing `nnue::finish` when the head is on: the two terms meet at a
/// common denominator before the one truncating division.
#[inline(always)]
pub(crate) fn finish(lin_b: i16, sum: i64, head: i32, limit: i32) -> i32 {
    let qq = QA as i64 * QB as i64;
    let num = (sum / QA as i64 + lin_b as i64) * (1i64 << QO) + head as i64 * qq;
    let raw = num * SCALE as i64 / (qq << QO);
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
            *o = ((c * c + 256) >> 9) as u8;
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
    pub(crate) fn l1(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
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
                    // t = 8c: sqrdmulh(t, t) = (128 c^2 + 2^15) >> 16 = (c^2 + 256) >> 9.
                    let a = vshlq_n_s16::<3>(vminq_s16(vmaxq_s16(vld1q_s16(p), lo), hi));
                    let b = vshlq_n_s16::<3>(vminq_s16(vmaxq_s16(vld1q_s16(p.add(8)), lo), hi));
                    let (a, b) = (vqrdmulhq_s16(a, a), vqrdmulhq_s16(b, b));
                    let v = vuzp1q_u8(vreinterpretq_u8_s16(a), vreinterpretq_u8_s16(b));
                    vst1q_u8(out.as_mut_ptr().add(i + 16 * g), v);
                    let v32 = vreinterpretq_u32_u8(v);
                    m = vorrq_u32(m, vshlq_u32(vandq_u32(vtstq_u32(v32, v32), bits), vdupq_n_s32(4 * g as i32)));
                }
                let mask = vaddvq_u32(m);
                for g in 0..4 {
                    n = push_nnz(nnz, n, base + i / 4 + 4 * g, mask >> (4 * g) & 15);
                }
            }
        }
        n
    }

    /// `acc += Σ_j w[4l+j] * x[4l+j]` per i32 lane (`sdot`).
    #[cfg(target_feature = "dotprod")]
    #[inline(always)]
    unsafe fn dot4(mut acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
        // SAFETY: register-only instruction; dotprod is enabled.
        unsafe {
            std::arch::asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) acc, in(vreg) w, in(vreg) x, options(pure, nomem, nostack));
        }
        acc
    }

    #[cfg(not(target_feature = "dotprod"))]
    #[inline(always)]
    unsafe fn dot4(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
        // |products| <= 127 * 127, pair sums fit i16.
        unsafe {
            let lo = vmull_s8(vget_low_s8(w), vget_low_s8(x));
            let hi = vmull_high_s8(w, x);
            vpadalq_s16(acc, vpaddq_s16(lo, hi))
        }
    }

    #[inline(always)]
    pub(super) fn l1(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: every chunk index is < CHUNKS, so all loads are in bounds.
        unsafe {
            // Four independent accumulator sets (16 registers) hide sdot latency.
            let z = vdupq_n_s32(0);
            let mut a = [[z; 4]; 4];
            let xp = x.as_ptr() as *const u32;
            let wp = w.as_ptr();
            let mut i = 0;
            while i + 3 < nnz.len() {
                for s in 0..4 {
                    let c = nnz[i + s] as usize;
                    let xv = vreinterpretq_s8_u32(vdupq_n_u32(xp.add(c).read_unaligned()));
                    let wc = wp.add(c * 64);
                    for r in 0..4 {
                        a[s][r] = dot4(a[s][r], vld1q_s8(wc.add(16 * r)), xv);
                    }
                }
                i += 4;
            }
            while i < nnz.len() {
                let c = nnz[i] as usize;
                let xv = vreinterpretq_s8_u32(vdupq_n_u32(xp.add(c).read_unaligned()));
                let wc = wp.add(c * 64);
                for r in 0..4 {
                    a[0][r] = dot4(a[0][r], vld1q_s8(wc.add(16 * r)), xv);
                }
                i += 1;
            }
            let mut out = [0i32; H1];
            for r in 0..4 {
                let v = vaddq_s32(vaddq_s32(a[0][r], a[1][r]), vaddq_s32(a[2][r], a[3][r]));
                vst1q_s32(out.as_mut_ptr().add(4 * r), v);
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
    pub(super) fn l1(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        if avx2() {
            // SAFETY: AVX2 is available.
            unsafe { l1_avx2(w, x, nnz) }
        } else {
            scalar::l1(w, x, nnz)
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
                // t = 8c: mulhrs(t, t) = (64 c^2 + 2^14) >> 15 = (c^2 + 256) >> 9.
                let a = _mm256_slli_epi16::<3>(_mm256_min_epi16(_mm256_max_epi16(a, lo), hi));
                let b = _mm256_slli_epi16::<3>(_mm256_min_epi16(_mm256_max_epi16(b, lo), hi));
                let (a, b) = (_mm256_mulhrs_epi16(a, a), _mm256_mulhrs_epi16(b, b));
                // packus interleaves 128-bit lanes; restore the order.
                let p = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(a, b));
                _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, p);
                let zero = _mm256_cmpeq_epi32(p, lo);
                let mask = !(_mm256_movemask_ps(_mm256_castsi256_ps(zero)) as u32) & 0xFF;
                n = push_nnz(nnz, n, base + i / 4, mask & 15);
                n = push_nnz(nnz, n, base + i / 4 + 4, mask >> 4);
            }
        }
        n
    }

    #[target_feature(enable = "avx2")]
    unsafe fn l1_avx2(w: &[i8; CHUNKS * H1 * 4], x: &[u8; N_IN], nnz: &[u16]) -> [i32; H1] {
        // SAFETY: every chunk index is < CHUNKS, so all loads are in bounds.
        // maddubs: u8 (<= 127) x i8 pairs <= 2 * 127 * 127, no saturation.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut a = [_mm256_setzero_si256(); 2];
            let mut b = [_mm256_setzero_si256(); 2];
            let xp = x.as_ptr() as *const i32;
            let wp = w.as_ptr() as *const __m256i;
            let mut i = 0;
            while i + 1 < nnz.len() {
                let (c0, c1) = (nnz[i] as usize, nnz[i + 1] as usize);
                let x0 = _mm256_set1_epi32(xp.add(c0).read_unaligned());
                let x1 = _mm256_set1_epi32(xp.add(c1).read_unaligned());
                for r in 0..2 {
                    let p0 = _mm256_maddubs_epi16(x0, _mm256_load_si256(wp.add(c0 * 2 + r)));
                    let p1 = _mm256_maddubs_epi16(x1, _mm256_load_si256(wp.add(c1 * 2 + r)));
                    a[r] = _mm256_add_epi32(a[r], _mm256_madd_epi16(p0, ones));
                    b[r] = _mm256_add_epi32(b[r], _mm256_madd_epi16(p1, ones));
                }
                i += 2;
            }
            if i < nnz.len() {
                let c0 = nnz[i] as usize;
                let x0 = _mm256_set1_epi32(xp.add(c0).read_unaligned());
                for r in 0..2 {
                    let p0 = _mm256_maddubs_epi16(x0, _mm256_load_si256(wp.add(c0 * 2 + r)));
                    a[r] = _mm256_add_epi32(a[r], _mm256_madd_epi16(p0, ones));
                }
            }
            let mut out = [0i32; H1];
            for r in 0..2 {
                _mm256_storeu_si256(out.as_mut_ptr().add(8 * r) as *mut __m256i, _mm256_add_epi32(a[r], b[r]));
            }
            out
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
        assert_eq!(s[1], 127);
        assert_eq!((a, &ns[3..a]), (b, &nv[3..b]));
        assert!(a > 3 + H / 8 && a < 3 + H / 4);
    }

    #[test]
    fn finish_combines_exactly() {
        // Head of exactly 1.0 = 400 cp; linear part as nnue::finish.
        assert_eq!(finish(0, 0, 1 << QO, 20000), 400);
        let sum = 123_456_789i64;
        let lin = (sum / QA as i64 + 77) * SCALE as i64 / (QA as i64 * QB as i64);
        assert_eq!(finish(77, sum, 0, 20000) as i64, lin);
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

/// Engine vs trainer on a real `--head` net: needs a `HEAD = true` build
/// (the net embedded, or `RUSTFISH_HEAD_NET`) and a fixture of
/// `<FEN> | <trainer cp>` lines from `nnue-trainer fen-eval`;
/// `scripts/head_check.py` sets all of that up. Skipped otherwise.
#[cfg(test)]
mod fixture {
    use super::super::{Accumulator, Active, Network, NnueState, HEAD, H};
    use super::{scalar, CHUNKS, NNZ_BUF};
    use crate::Board;
    use std::sync::Arc;

    #[test]
    fn head_matches_trainer_fixture() {
        let Ok(path) = std::env::var("RUSTFISH_HEAD_FIXTURE") else {
            eprintln!("RUSTFISH_HEAD_FIXTURE not set: skipped");
            return;
        };
        assert!(HEAD, "the fixture check needs a HEAD = true build");
        let net = match std::env::var("RUSTFISH_HEAD_NET") {
            Ok(p) => Network::load(&p).unwrap(),
            Err(_) => Network::from_bytes(crate::EMBEDDED_NET).unwrap(),
        };
        let net = Arc::new(net);
        let text = std::fs::read_to_string(&path).unwrap();
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
        assert!(n >= 10, "fixture has {n} usable positions");
        let mean = sum / n as f64;
        let bias = diffs.iter().sum::<f64>() / n as f64;
        diffs.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        let p95 = diffs[n * 95 / 100].abs();
        let nnz = nnz_sum as f64 / diffs.len() as f64;
        eprintln!("head fixture: nonzero L1 input chunks {nnz:.0} of {CHUNKS} ({:.0}%)", 100.0 * nnz / CHUNKS as f64);
        eprintln!("head fixture: {n} positions, |engine - trainer| mean {mean:.2} cp, p95 {p95:.2}, max {worst:.2}, bias {bias:+.2}; worst {worst_fen}");
        // The engine truncates to whole centipawns (bias ~ -0.5) and
        // quantises FT-free head inputs to 7 bits.
        assert!(mean < 1.5 && p95 < 4.0 && worst < 12.0, "engine and trainer disagree beyond quantisation error");
    }
}
