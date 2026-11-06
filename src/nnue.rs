//! NNUE inference: (768 -> H) x 2 -> 1 with SCReLU, as specified in the
//! shared spec (see `docs`/build brief):
//!
//! - feature for perspective p, piece (colour c, type t) on square s:
//!   `(c == p ? 0 : 384) + t*64 + (p == White ? s : s ^ 56)`
//! - net file: little-endian i16, feature weights [768][H], feature biases
//!   [H], output weights [2H] (stm half first), output bias [1]; no header.
//! - `eval = ((Σ screlu(us)·w[i] + Σ screlu(them)·w[H+i]) / QA + b) · SCALE / (QA·QB)`
//!   with `screlu(x) = clamp(x, 0, QA)²`, integer division truncating to zero.
//!
//! Accumulators are updated incrementally and lazily: `Board::make_move`
//! records the move's feature delta through `NnueState::push_move` (one stack
//! entry per ply), `undo_move` pops it, and `evaluate` materialises pending
//! entries (copy-and-update fused in one pass) only when an eval is needed.
//! Every SIMD kernel has a scalar twin and both are exact (wrapping i16 adds;
//! output sums proven not to overflow i32, otherwise the i64 scalar path is
//! used), so results never depend on the target.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use crate::{Board, Color, Move, Piece, PieceType};

pub(crate) const INPUTS: usize = 768;
pub(crate) const H: usize = 256;
pub(crate) const QA: i32 = 255;
pub(crate) const QB: i32 = 64;
pub(crate) const SCALE: i32 = 400;
/// Output buckets, selected by piece count: `(pieces - 2) * BUCKETS / 32`.
pub(crate) const BUCKETS: usize = 1;

/// Bytes in a net file: (768*H + H + BUCKETS*2H + BUCKETS) little-endian i16.
pub(crate) const NET_BYTES: usize = (INPUTS * H + H + BUCKETS * (2 * H + 1)) * 2;

/// NNUE scores are clamped well inside the mate range.
const EVAL_LIMIT: i32 = 20_000;

#[derive(Clone)]
pub(crate) struct Network {
    /// Feature-major: weights of feature f are `ft_w[f*H .. (f+1)*H]`.
    ft_w: Vec<i16>,
    ft_b: [i16; H],
    /// Bucket-major: bucket b's weights are `out_w[b*2H .. (b+1)*2H]`, stm half first.
    out_w: Vec<i16>,
    out_b: [i16; BUCKETS],
    /// All |out_w| <= 127: the SIMD output kernel's i16 products and i32
    /// per-perspective sums cannot overflow (256 * 255^2 * 127 < 2^31).
    small_out: bool,
}

impl Network {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != NET_BYTES {
            return Err(format!("net is {} bytes, expected {}", bytes.len(), NET_BYTES));
        }
        let mut words = bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]));
        let ft_w: Vec<i16> = words.by_ref().take(INPUTS * H).collect();
        let mut ft_b = [0i16; H];
        for v in ft_b.iter_mut() {
            *v = words.next().unwrap();
        }
        let out_w: Vec<i16> = words.by_ref().take(BUCKETS * 2 * H).collect();
        let mut out_b = [0i16; BUCKETS];
        for v in out_b.iter_mut() {
            *v = words.next().unwrap();
        }
        Ok(Self::from_parts(ft_w, ft_b, out_w, out_b))
    }

    fn from_parts(ft_w: Vec<i16>, ft_b: [i16; H], out_w: Vec<i16>, out_b: [i16; BUCKETS]) -> Self {
        assert_eq!(ft_w.len(), INPUTS * H);
        assert_eq!(out_w.len(), BUCKETS * 2 * H);
        let small_out = out_w.iter().all(|w| (-127..=127).contains(w));
        Self { ft_w, ft_b, out_w, out_b, small_out }
    }

    pub(crate) fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Self::from_bytes(&bytes)
    }

    #[cfg(test)]
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(NET_BYTES);
        let all = self.ft_w.iter().chain(self.ft_b.iter()).chain(self.out_w.iter());
        for w in all.chain(self.out_b.iter()) {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Deterministic pseudo-random net (xorshift) for tests and benchmarks.
    /// `out_range` bounds |out_w|; > 127 exercises the wide (i64) output path.
    #[cfg(test)]
    pub(crate) fn random(seed: u64, ft_range: i16, out_range: i16) -> Self {
        let mut s = seed | 1;
        let mut next = move |range: i16| -> i16 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let span = 2 * range as i64 + 1;
            ((s % span as u64) as i64 - range as i64) as i16
        };
        let ft_w = (0..INPUTS * H).map(|_| next(ft_range)).collect();
        let mut ft_b = [0i16; H];
        ft_b.iter_mut().for_each(|v| *v = next(ft_range));
        let out_w = (0..BUCKETS * 2 * H).map(|_| next(out_range)).collect();
        let mut out_b = [0i16; BUCKETS];
        out_b.iter_mut().for_each(|v| *v = next(out_range));
        Self::from_parts(ft_w, ft_b, out_w, out_b)
    }

    #[inline(always)]
    fn row(&self, f: usize) -> &[i16; H] {
        self.ft_w[f * H..(f + 1) * H].try_into().unwrap()
    }
}

#[inline(always)]
pub(crate) fn feature(persp: Color, p: Piece, sq: u8) -> usize {
    let side = if p.color == persp { 0 } else { 384 };
    let s = if persp == Color::White { sq } else { sq ^ 56 };
    side + p.kind as usize * 64 + s as usize
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct Accumulator {
    /// Indexed by perspective: 0 = White, 1 = Black.
    pub(crate) v: [[i16; H]; 2],
}

impl Accumulator {
    const ZERO: Self = Self { v: [[0; H]; 2] };

    pub(crate) fn refresh(net: &Network, board: &Board) -> Self {
        let mut acc = Self { v: [net.ft_b; 2] };
        for sq in 0..64u8 {
            if let Some(p) = board.squares[sq as usize] {
                for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
                    let row = net.row(feature(persp, p, sq));
                    for (a, w) in acc.v[pi].iter_mut().zip(row.iter()) {
                        *a = a.wrapping_add(*w);
                    }
                }
            }
        }
        acc
    }
}

/// Feature changes of one move: at most two additions and two removals
/// (castling: king + rook; capture-promotion: pawn + victim removed).
#[derive(Clone, Copy, Default)]
struct Delta {
    add: [(u8, Option<Piece>); 2],
    sub: [(u8, Option<Piece>); 2],
    n_add: usize,
    n_sub: usize,
}

impl Delta {
    fn add(&mut self, sq: u8, p: Piece) {
        self.add[self.n_add] = (sq, Some(p));
        self.n_add += 1;
    }
    fn sub(&mut self, sq: u8, p: Piece) {
        self.sub[self.n_sub] = (sq, Some(p));
        self.n_sub += 1;
    }
}

fn move_delta(board: &Board, mv: Move) -> Delta {
    let mut d = Delta::default();
    let moving = board.squares[mv.from as usize].expect("nnue: no piece on from square");
    let mut placed = moving;
    if let Some(promo) = mv.promotion {
        placed.kind = promo;
    }
    d.sub(mv.from, moving);
    d.add(mv.to, placed);
    if mv.is_en_passant {
        let cap_sq = if moving.color == Color::White { mv.to - 8 } else { mv.to + 8 };
        if let Some(cp) = board.squares[cap_sq as usize] {
            d.sub(cap_sq, cp);
        }
    } else if let Some(cp) = board.squares[mv.to as usize] {
        d.sub(mv.to, cp);
    }
    if mv.is_castle {
        let rook_shift = match mv.to {
            6 => Some((7u8, 5u8)),
            2 => Some((0, 3)),
            62 => Some((63, 61)),
            58 => Some((56, 59)),
            _ => None,
        };
        if let Some((rf, rt)) = rook_shift {
            let rook = Piece { color: moving.color, kind: PieceType::Rook };
            d.sub(rf, rook);
            d.add(rt, rook);
        }
    }
    d
}

/// One ply of the accumulator stack: the move's feature delta from the
/// entry below, and the accumulator it yields once materialised.
#[derive(Clone, Copy)]
struct Slot {
    acc: Accumulator,
    delta: Delta,
}

/// Per-board NNUE state: the net and one accumulator per ply.
///
/// Updates are lazy: `push_move` only records the move's feature delta, and
/// `evaluate` materialises the missing accumulators bottom-up from the last
/// computed one. Nodes that never evaluate (TT cutoffs, eval-cache hits,
/// checks, repetitions) then cost no accumulator work at all. The values
/// are identical to eager updating: the same wrapping adds in the same chain.
#[derive(Clone)]
pub(crate) struct NnueState {
    net: Arc<Network>,
    /// `stack[..=computed]` hold valid accumulators for the current line;
    /// `computed <= top` always. Behind a RefCell so `evaluate(&self)` can
    /// fill in the pending entries.
    stack: RefCell<Vec<Slot>>,
    computed: Cell<usize>,
    top: usize,
}

impl PartialEq for NnueState {
    /// Board equality is about the position; the accumulator is derived data.
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for NnueState {}

impl NnueState {
    pub(crate) fn new(net: Arc<Network>, board: &Board) -> Self {
        let mut v = Vec::with_capacity(256);
        v.push(Slot { acc: Accumulator::ZERO, delta: Delta::default() });
        let mut s = Self { net, stack: RefCell::new(v), computed: Cell::new(0), top: 0 };
        s.refresh(board);
        s
    }

    /// Rebuild from scratch and reset the stack to one entry.
    pub(crate) fn refresh(&mut self, board: &Board) {
        self.top = 0;
        self.computed.set(0);
        self.stack.get_mut()[0].acc = Accumulator::refresh(&self.net, board);
    }

    /// The (materialised) accumulator of the current position.
    #[cfg(test)]
    pub(crate) fn current(&self) -> Accumulator {
        self.materialise();
        self.stack.borrow()[self.top].acc
    }

    /// Called by `make_move` before the board changes.
    #[inline]
    pub(crate) fn push_move(&mut self, board: &Board, mv: Move) {
        let delta = move_delta(board, mv);
        let stack = self.stack.get_mut();
        self.top += 1;
        if self.top == stack.len() {
            stack.push(Slot { acc: Accumulator::ZERO, delta });
        } else {
            stack[self.top].delta = delta;
        }
    }

    /// Called by `undo_move`.
    #[inline]
    pub(crate) fn pop(&mut self) {
        self.top -= 1;
        if self.computed.get() > self.top {
            self.computed.set(self.top);
        }
    }

    /// Bring `stack[..=top]` up to date.
    #[inline]
    fn materialise(&self) {
        let mut k = self.computed.get();
        if k == self.top {
            return;
        }
        let mut stack = self.stack.borrow_mut();
        let net = &*self.net;
        while k < self.top {
            let (lo, hi) = stack.split_at_mut(k + 1);
            apply_delta(net, &lo[k].acc, &mut hi[0]);
            k += 1;
        }
        self.computed.set(k);
    }

    /// Score from the side to move's point of view, in centipawns, for a
    /// position with `pieces` pieces on the board (selects the output bucket).
    #[inline]
    pub(crate) fn evaluate(&self, stm: Color, pieces: u32) -> i32 {
        self.materialise();
        let stack = self.stack.borrow();
        evaluate_acc(&self.net, &stack[self.top].acc, stm, pieces)
    }
}

/// `dst.acc = src + dst.delta`, both perspectives in one copy-and-update pass.
#[inline(always)]
fn apply_delta(net: &Network, src: &Accumulator, dst: &mut Slot) {
    let d = dst.delta;
    for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
        let f = |e: (u8, Option<Piece>)| net.row(feature(persp, e.1.unwrap(), e.0));
        let a0 = f(d.add[0]);
        let s0 = f(d.sub[0]);
        let (s, o) = (&src.v[pi], &mut dst.acc.v[pi]);
        match (d.n_add, d.n_sub) {
            (1, 1) => update(o, s, &[a0], &[s0]),
            (1, 2) => update(o, s, &[a0], &[s0, f(d.sub[1])]),
            (2, 2) => update(o, s, &[a0, f(d.add[1])], &[s0, f(d.sub[1])]),
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub(crate) fn bucket(pieces: u32) -> usize {
    if BUCKETS == 1 {
        0
    } else {
        (pieces.saturating_sub(2) as usize * BUCKETS / 32).min(BUCKETS - 1)
    }
}

pub(crate) fn evaluate_acc(net: &Network, acc: &Accumulator, stm: Color, pieces: u32) -> i32 {
    let (us, them) = match stm {
        Color::White => (&acc.v[0], &acc.v[1]),
        Color::Black => (&acc.v[1], &acc.v[0]),
    };
    let b = bucket(pieces);
    let sum = output_sum(net, b, us, them);
    let raw = (sum / QA as i64 + net.out_b[b] as i64) * SCALE as i64 / (QA as i64 * QB as i64);
    raw.clamp(-EVAL_LIMIT as i64, EVAL_LIMIT as i64) as i32
}

#[inline(always)]
fn output_sum(net: &Network, b: usize, us: &[i16; H], them: &[i16; H]) -> i64 {
    let (w_us, w_them) = net.out_w[b * 2 * H..(b + 1) * 2 * H].split_at(H);
    let w_us: &[i16; H] = w_us.try_into().unwrap();
    let w_them: &[i16; H] = w_them.try_into().unwrap();
    if net.small_out {
        simd::screlu_dot(us, w_us) + simd::screlu_dot(them, w_them)
    } else {
        simd::screlu_dot_wide(us, w_us) + simd::screlu_dot_wide(them, w_them)
    }
}

/// `out = src + Σ adds - Σ subs` (wrapping i16, so lane order is irrelevant).
#[inline(always)]
fn update(out: &mut [i16; H], src: &[i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]) {
    simd::update(out, src, adds, subs)
}

pub(crate) mod scalar {
    use super::{H, QA};

    // Reference for the SIMD kernels; the live path on non-SIMD targets.
    #[allow(dead_code)]
    pub(crate) fn update(out: &mut [i16; H], src: &[i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        for i in 0..H {
            let mut v = src[i];
            for a in adds {
                v = v.wrapping_add(a[i]);
            }
            for s in subs {
                v = v.wrapping_sub(s[i]);
            }
            out[i] = v;
        }
    }

    /// Exact for any i16 weights.
    pub(crate) fn screlu_dot_wide(acc: &[i16; H], w: &[i16; H]) -> i64 {
        let mut sum = 0i64;
        for i in 0..H {
            let c = (acc[i] as i32).clamp(0, QA) as i64;
            sum += c * c * w[i] as i64;
        }
        sum
    }

    /// Requires |w| <= 127 (see `Network::small_out`): each product fits i32.
    #[allow(dead_code)]
    pub(crate) fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i64 {
        let mut sum = 0i64;
        for i in 0..H {
            let c = (acc[i] as i32).clamp(0, QA);
            sum += (c * w[i] as i32 * c) as i64;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod simd {
    use super::{H, QA};
    use std::arch::aarch64::*;
    pub(crate) use super::scalar::screlu_dot_wide;

    #[inline(always)]
    pub(crate) fn update(out: &mut [i16; H], src: &[i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        // SAFETY: NEON is baseline on aarch64; all pointers cover H i16s and
        // the loop reads/writes 8 lanes at offsets < H.
        unsafe {
            let mut i = 0;
            while i < H {
                let mut v = vld1q_s16(src.as_ptr().add(i));
                for a in adds {
                    v = vaddq_s16(v, vld1q_s16(a.as_ptr().add(i)));
                }
                for s in subs {
                    v = vsubq_s16(v, vld1q_s16(s.as_ptr().add(i)));
                }
                vst1q_s16(out.as_mut_ptr().add(i), v);
                i += 8;
            }
        }
    }

    #[inline(always)]
    pub(crate) fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i64 {
        // SAFETY: as above. c in [0,255], |w| <= 127 so c*w fits i16; each
        // i32 lane sums H/32 products of at most 255^2*127 (exact for H <= 8192)
        // and the lanes are reduced in i64. Eight independent accumulators
        // keep the multiply-accumulate chains from bounding throughput.
        const { assert!(H % 32 == 0) };
        unsafe {
            let zero = vdupq_n_s16(0);
            let qa = vdupq_n_s16(QA as i16);
            let mut s = [vdupq_n_s32(0); 8];
            let mut i = 0;
            while i < H {
                for j in 0..4 {
                    let c = vminq_s16(vmaxq_s16(vld1q_s16(acc.as_ptr().add(i + 8 * j)), zero), qa);
                    let cw = vmulq_s16(c, vld1q_s16(w.as_ptr().add(i + 8 * j)));
                    s[2 * j] = vmlal_s16(s[2 * j], vget_low_s16(cw), vget_low_s16(c));
                    s[2 * j + 1] = vmlal_high_s16(s[2 * j + 1], cw, c);
                }
                i += 32;
            }
            s.iter().map(|&x| vaddlvq_s32(x)).sum()
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) mod simd {
    use super::{H, QA};
    use std::arch::x86_64::*;

    #[inline(always)]
    pub(crate) fn update(out: &mut [i16; H], src: &[i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        // SAFETY: AVX2 is enabled at compile time (cfg); unaligned 256-bit
        // loads/stores at offsets < H.
        unsafe {
            let mut i = 0;
            while i < H {
                let mut v = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                for a in adds {
                    v = _mm256_add_epi16(v, _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i));
                }
                for s in subs {
                    v = _mm256_sub_epi16(v, _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i));
                }
                _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, v);
                i += 16;
            }
        }
    }

    #[inline(always)]
    pub(crate) fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i64 {
        // SAFETY: as above. c*w fits i16 (|w| <= 127), madd(c*w, c) sums two
        // exact i32 products; total bounded by `Network::small_out`.
        unsafe {
            let zero = _mm256_setzero_si256();
            let qa = _mm256_set1_epi16(QA as i16);
            let mut sum = _mm256_setzero_si256();
            let mut i = 0;
            while i < H {
                let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
                let c = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
                let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
                let cw = _mm256_mullo_epi16(c, wv);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(cw, c));
                i += 16;
            }
            // Each i32 lane holds H/8 products (exact for H <= 2048); reduce in i64.
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum);
            lanes.iter().map(|&x| x as i64).sum()
        }
    }

    /// Exact i64 screlu dot for any i16 weights (the `!small_out` path).
    /// c^2 <= 65025 and |w| <= 32767, so each 32-bit product fits signed
    /// i32; it is widened to i64 by `_mm256_mul_epi32` before accumulating.
    #[inline(always)]
    pub(crate) fn screlu_dot_wide(acc: &[i16; H], w: &[i16; H]) -> i64 {
        // SAFETY: AVX2 is enabled at compile time (cfg); unaligned loads at
        // offsets < H, and the accumulator is 4 i64 lanes.
        unsafe {
            let zero = _mm256_setzero_si256();
            let qa = _mm256_set1_epi16(QA as i16);
            let mut sum = _mm256_setzero_si256();
            let mut i = 0;
            while i < H {
                let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
                let c16 = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
                let w16 = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
                let c_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c16));
                let c_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c16, 1));
                let w_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w16));
                let w_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w16, 1));
                let cc_lo = _mm256_mullo_epi32(c_lo, c_lo);
                let cc_hi = _mm256_mullo_epi32(c_hi, c_hi);
                // Even/odd 32-bit lanes -> 64-bit products, two vectors each.
                let p_lo = _mm256_add_epi64(
                    _mm256_mul_epi32(cc_lo, w_lo),
                    _mm256_mul_epi32(_mm256_srli_epi64(cc_lo, 32), _mm256_srli_epi64(w_lo, 32)),
                );
                let p_hi = _mm256_add_epi64(
                    _mm256_mul_epi32(cc_hi, w_hi),
                    _mm256_mul_epi32(_mm256_srli_epi64(cc_hi, 32), _mm256_srli_epi64(w_hi, 32)),
                );
                sum = _mm256_add_epi64(sum, _mm256_add_epi64(p_lo, p_hi));
                i += 16;
            }
            let lo = _mm256_castsi256_si128(sum);
            let hi = _mm256_extracti128_si256(sum, 1);
            let s = _mm_add_epi64(lo, hi);
            let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
            _mm_cvtsi128_si64(s)
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) mod simd {
    pub(crate) use super::scalar::{screlu_dot, screlu_dot_wide, update};
}

/// x86-64 builds without AVX2 enabled at compile time (the default
/// `cargo build`): detect AVX2 once and use the same kernels at runtime,
/// falling back to scalar on pre-Haswell CPUs.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub(crate) mod simd {
    use super::{H, QA};
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    #[inline]
    fn avx2() -> bool {
        static HAVE: OnceLock<bool> = OnceLock::new();
        *HAVE.get_or_init(|| is_x86_feature_detected!("avx2"))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn update_avx2(out: &mut [i16; H], src: &[i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        let mut i = 0;
        while i < H {
            let mut v = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            for a in adds {
                v = _mm256_add_epi16(v, _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i));
            }
            for s in subs {
                v = _mm256_sub_epi16(v, _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i));
            }
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, v);
            i += 16;
        }
    }

    #[inline(always)]
    pub(crate) fn update(out: &mut [i16; H], src: &[i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        if avx2() {
            // SAFETY: the feature was detected above and all slices cover H i16s.
            unsafe { update_avx2(out, src, adds, subs) }
        } else {
            super::scalar::update(out, src, adds, subs)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn screlu_dot_avx2(acc: &[i16; H], w: &[i16; H]) -> i64 {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut sum = _mm256_setzero_si256();
        let mut i = 0;
        while i < H {
            let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let c = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
            let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            let cw = _mm256_mullo_epi16(c, wv);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(cw, c));
            i += 16;
        }
        // Each i32 lane holds H/8 products (exact for H <= 2048); reduce in i64.
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum);
        lanes.iter().map(|&x| x as i64).sum()
    }

    #[inline(always)]
    pub(crate) fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i64 {
        if avx2() {
            // SAFETY: detected above; |w| <= 127 bound as in the scalar path.
            unsafe { screlu_dot_avx2(acc, w) }
        } else {
            super::scalar::screlu_dot(acc, w)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn screlu_dot_wide_avx2(acc: &[i16; H], w: &[i16; H]) -> i64 {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut sum = _mm256_setzero_si256();
        let mut i = 0;
        while i < H {
            let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let c16 = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
            let w16 = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            let c_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c16));
            let c_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c16, 1));
            let w_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w16));
            let w_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w16, 1));
            let cc_lo = _mm256_mullo_epi32(c_lo, c_lo);
            let cc_hi = _mm256_mullo_epi32(c_hi, c_hi);
            let p_lo = _mm256_add_epi64(
                _mm256_mul_epi32(cc_lo, w_lo),
                _mm256_mul_epi32(_mm256_srli_epi64(cc_lo, 32), _mm256_srli_epi64(w_lo, 32)),
            );
            let p_hi = _mm256_add_epi64(
                _mm256_mul_epi32(cc_hi, w_hi),
                _mm256_mul_epi32(_mm256_srli_epi64(cc_hi, 32), _mm256_srli_epi64(w_hi, 32)),
            );
            sum = _mm256_add_epi64(sum, _mm256_add_epi64(p_lo, p_hi));
            i += 16;
        }
        let lo = _mm256_castsi256_si128(sum);
        let hi = _mm256_extracti128_si256(sum, 1);
        let s = _mm_add_epi64(lo, hi);
        let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
        _mm_cvtsi128_si64(s)
    }

    #[inline(always)]
    pub(crate) fn screlu_dot_wide(acc: &[i16; H], w: &[i16; H]) -> i64 {
        if avx2() {
            // SAFETY: detected above; products fit i32 as documented.
            unsafe { screlu_dot_wide_avx2(acc, w) }
        } else {
            super::scalar::screlu_dot_wide(acc, w)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::move_to_uci;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    const FENS: [&str; 6] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "4k3/1P6/8/2pP4/8/8/6p1/4K3 w - c6 0 1",
    ];

    /// Hand-computed: one nonzero bias, one nonzero weight, empty-ish board.
    #[test]
    fn tiny_net_matches_formula() {
        let mut ft_w = vec![0i16; INPUTS * H];
        let mut ft_b = [0i16; H];
        let mut out_w = vec![0i16; BUCKETS * 2 * H];
        ft_b[0] = 100;
        ft_b[1] = 300; // clamps to QA
        // White king on e1 (sq 4): white persp feature 5*64+4 = 324,
        // black persp feature 384+5*64+(4^56)=764.
        ft_w[324 * H + 0] = 50; // white acc[0] = 150
        ft_w[764 * H + 0] = -200; // black acc[0] = -100 -> 0
        ft_w[764 * H + 2] = 20; // black acc[2] = 20
        out_w[0] = 3; // us[0]
        out_w[1] = -2; // us[1]
        out_w[H + 2] = 7; // them[2]
        let out_b = [11; BUCKETS];
        let net = Network::from_parts(ft_w, ft_b, out_w, out_b);
        let net2 = Network::from_bytes(&net.to_bytes()).unwrap();
        let mut board = Board::new();
        board.set_fen("8/8/8/8/8/8/8/4K3 w - - 0 1");
        let acc = Accumulator::refresh(&net2, &board);
        assert_eq!(acc.v[0][0], 150);
        assert_eq!(acc.v[1][0], -100);
        assert_eq!(acc.v[1][2], 20);
        // White to move: us = white.
        let sum_w: i64 = 150 * 150 * 3 + 255 * 255 * -2 + 20 * 20 * 7;
        let want_w = (sum_w / 255 + 11) * 400 / (255 * 64);
        assert_eq!(evaluate_acc(&net2, &acc, Color::White, 1) as i64, want_w);
        // Black to move: us = black acc (0, 255, 20...), them = white acc.
        // us[0]=0 (clamped), us[1]=255 -> -2*255^2; them[2]=white acc[2]=0.
        let sum_b: i64 = 255 * 255 * -2;
        let want_b = (sum_b / 255 + 11) * 400 / (255 * 64);
        assert_eq!(evaluate_acc(&net2, &acc, Color::Black, 1) as i64, want_b);
        // ((22500*3 - 65025*2 + 400*7)/255 + 11)*400/16320 = (-234+11)*400/16320
        assert_eq!(want_w, -5);
        assert_eq!(want_b, -12);
    }

    #[test]
    fn incremental_matches_refresh() {
        for (seed, out_range) in [(7u64, 100i16), (99, 2000)] {
            let net = Arc::new(Network::random(seed, 64, out_range));
            let mut next = rng(seed * 31 + 5);
            let mut legal = Vec::new();
            let (mut caps, mut castles, mut eps, mut promos) = (0, 0, 0, 0);
            for game in 0..60 {
                let mut board = Board::new();
                board.set_fen(FENS[game % FENS.len()]);
                board.attach_nnue(Some(net.clone()));
                let mut history = Vec::new();
                for _ in 0..120 {
                    board.generate_legal_moves(false, &mut legal);
                    if legal.is_empty() {
                        break;
                    }
                    let mv = legal[(next() as usize) % legal.len()];
                    if board.squares[mv.to as usize].is_some() || mv.is_en_passant {
                        caps += 1;
                    }
                    castles += mv.is_castle as usize;
                    eps += mv.is_en_passant as usize;
                    promos += mv.promotion.is_some() as usize;
                    let undo = board.make_move(mv);
                    history.push((mv, undo));
                    let st = board.nnue.as_ref().unwrap();
                    let full = Accumulator::refresh(&net, &board);
                    assert!(st.current().v == full.v, "after {}", move_to_uci(mv));
                    for stm in [Color::White, Color::Black] {
                        let n = board.bb.occ_all().count_ones();
                        assert_eq!(st.evaluate(stm, n), evaluate_acc(&net, &full, stm, n));
                    }
                }
                // Unwind and check the root accumulator is intact.
                while let Some((mv, undo)) = history.pop() {
                    board.undo_move(mv, undo);
                    let full = Accumulator::refresh(&net, &board);
                    assert!(board.nnue.as_ref().unwrap().current().v == full.v);
                }
            }
            assert!(caps > 100 && castles > 0 && eps > 0 && promos > 0, "{caps} {castles} {eps} {promos}");
        }
    }

    #[test]
    fn simd_matches_scalar() {
        let mut next = rng(12345);
        for _ in 0..2000 {
            let mut rand_vec = |range: i32| -> [i16; H] {
                let mut v = [0i16; H];
                for x in v.iter_mut() {
                    *x = ((next() % (2 * range as u64 + 1)) as i32 - range) as i16;
                }
                v
            };
            let src = rand_vec(32767);
            let a0 = rand_vec(32767);
            let a1 = rand_vec(32767);
            let s0 = rand_vec(32767);
            let s1 = rand_vec(32767);
            for (adds, subs) in [(vec![&a0], vec![&s0]), (vec![&a0], vec![&s0, &s1]), (vec![&a0, &a1], vec![&s0, &s1])] {
                let mut o1 = [0i16; H];
                let mut o2 = [0i16; H];
                simd::update(&mut o1, &src, &adds, &subs);
                scalar::update(&mut o2, &src, &adds, &subs);
                assert_eq!(o1, o2);
            }
            // Accumulator values around and beyond the clamp range.
            let acc = rand_vec(400);
            let w = rand_vec(127);
            let s = simd::screlu_dot(&acc, &w);
            assert_eq!(s, scalar::screlu_dot(&acc, &w));
            assert_eq!(s as i64, scalar::screlu_dot_wide(&acc, &w));
        }
        // Worst case for the i32 bound: all lanes saturated, |w| = 127.
        let acc = [1000i16; H];
        for w in [[127i16; H], [-127i16; H]] {
            let s = simd::screlu_dot(&acc, &w);
            assert_eq!(s as i64, scalar::screlu_dot_wide(&acc, &w));
            assert_eq!(s, H as i64 * 255 * 255 * w[0] as i64);
        }
    }

    #[test]
    fn rejects_wrong_size() {
        assert!(Network::from_bytes(&[0u8; 10]).is_err());
        assert!(Network::from_bytes(&vec![0u8; NET_BYTES]).is_ok());
    }

    /// Throughput report: `cargo test --release nnue_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn nnue_throughput() {
        use std::time::Instant;
        let net = Arc::new(Network::random(42, 64, 100));
        let mut board = Board::new();
        board.set_fen(FENS[1]);
        let mut st = NnueState::new(net.clone(), &board);
        let n = 200_000;
        let t = Instant::now();
        let mut x = 0i64;
        for _ in 0..n {
            let acc = Accumulator::refresh(&net, &board);
            x += evaluate_acc(&net, std::hint::black_box(&acc), Color::White, 32) as i64;
        }
        let full = n as f64 / t.elapsed().as_secs_f64();
        let mut legal = Vec::new();
        board.generate_legal_moves(false, &mut legal);
        let n = 5_000_000;
        let t = Instant::now();
        for i in 0..n {
            let mv = legal[i % legal.len()];
            st.push_move(&board, mv);
            x += st.evaluate(std::hint::black_box(Color::Black), 32) as i64;
            st.pop();
        }
        let inc = n as f64 / t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 0..n {
            x += st.evaluate(std::hint::black_box(Color::White), 32) as i64;
        }
        let out_only = n as f64 / t.elapsed().as_secs_f64();
        let t = Instant::now();
        for i in 0..n {
            st.push_move(&board, legal[i % legal.len()]);
            st.materialise();
            st.pop();
        }
        let upd_only = n as f64 / t.elapsed().as_secs_f64();
        println!(
            "nnue: full refresh+eval {:.0}/s, incremental update+eval {:.0}/s, eval only {:.0}/s, update only {:.0}/s ({x})",
            full, inc, out_only, upd_only
        );
    }
}
