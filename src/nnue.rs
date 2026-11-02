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
//! Accumulators are updated incrementally by `Board::make_move` through
//! `NnueState::push_move` (one stack entry per ply, copy-and-update fused in
//! one pass) and popped by `undo_move`. Every SIMD kernel has a scalar twin
//! and both are exact (wrapping i16 adds; output sums proven not to overflow
//! i32, otherwise the i64 scalar path is used), so results never depend on
//! the target.

use std::sync::Arc;

use crate::{Board, Color, Move, Piece, PieceType};

pub(crate) const INPUTS: usize = 768;
pub(crate) const H: usize = 256;
pub(crate) const QA: i32 = 255;
pub(crate) const QB: i32 = 64;
pub(crate) const SCALE: i32 = 400;

/// Bytes in a net file: (768*H + H + 2H + 1) little-endian i16.
pub(crate) const NET_BYTES: usize = (INPUTS * H + H + 2 * H + 1) * 2;

/// NNUE scores are clamped well inside the mate range.
const EVAL_LIMIT: i32 = 20_000;

#[derive(Clone)]
pub(crate) struct Network {
    /// Feature-major: weights of feature f are `ft_w[f*H .. (f+1)*H]`.
    ft_w: Vec<i16>,
    ft_b: [i16; H],
    out_w: [i16; 2 * H],
    out_b: i16,
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
        let mut out_w = [0i16; 2 * H];
        for v in out_w.iter_mut() {
            *v = words.next().unwrap();
        }
        let out_b = words.next().unwrap();
        Ok(Self::from_parts(ft_w, ft_b, out_w, out_b))
    }

    fn from_parts(ft_w: Vec<i16>, ft_b: [i16; H], out_w: [i16; 2 * H], out_b: i16) -> Self {
        assert_eq!(ft_w.len(), INPUTS * H);
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
        for w in all.chain(std::iter::once(&self.out_b)) {
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
        let mut out_w = [0i16; 2 * H];
        out_w.iter_mut().for_each(|v| *v = next(out_range));
        let out_b = next(out_range);
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

/// Per-board NNUE state: the net and one accumulator per ply.
#[derive(Clone)]
pub(crate) struct NnueState {
    net: Arc<Network>,
    stack: Vec<Accumulator>,
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
        let mut s = Self { net, stack: Vec::with_capacity(256), top: 0 };
        s.stack.push(Accumulator::ZERO);
        s.refresh(board);
        s
    }

    /// Rebuild from scratch and reset the stack to one entry.
    pub(crate) fn refresh(&mut self, board: &Board) {
        self.top = 0;
        self.stack[0] = Accumulator::refresh(&self.net, board);
    }

    pub(crate) fn current(&self) -> &Accumulator {
        &self.stack[self.top]
    }

    /// Called by `make_move` before the board changes.
    #[inline]
    pub(crate) fn push_move(&mut self, board: &Board, mv: Move) {
        let d = move_delta(board, mv);
        if self.top + 1 == self.stack.len() {
            self.stack.push(Accumulator::ZERO);
        }
        let (lo, hi) = self.stack.split_at_mut(self.top + 1);
        let src = &lo[self.top];
        let dst = &mut hi[0];
        self.top += 1;
        let net = &*self.net;
        for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
            let f = |e: (u8, Option<Piece>)| net.row(feature(persp, e.1.unwrap(), e.0));
            let a0 = f(d.add[0]);
            let s0 = f(d.sub[0]);
            let (s, o) = (&src.v[pi], &mut dst.v[pi]);
            match (d.n_add, d.n_sub) {
                (1, 1) => update(o, s, &[a0], &[s0]),
                (1, 2) => update(o, s, &[a0], &[s0, f(d.sub[1])]),
                (2, 2) => update(o, s, &[a0, f(d.add[1])], &[s0, f(d.sub[1])]),
                _ => unreachable!(),
            }
        }
    }

    /// Called by `undo_move`.
    #[inline]
    pub(crate) fn pop(&mut self) {
        self.top -= 1;
    }

    /// Score from the side to move's point of view, in centipawns.
    #[inline]
    pub(crate) fn evaluate(&self, stm: Color) -> i32 {
        evaluate_acc(&self.net, self.current(), stm)
    }
}

pub(crate) fn evaluate_acc(net: &Network, acc: &Accumulator, stm: Color) -> i32 {
    let (us, them) = match stm {
        Color::White => (&acc.v[0], &acc.v[1]),
        Color::Black => (&acc.v[1], &acc.v[0]),
    };
    let sum = output_sum(net, us, them);
    let raw = (sum / QA as i64 + net.out_b as i64) * SCALE as i64 / (QA as i64 * QB as i64);
    raw.clamp(-EVAL_LIMIT as i64, EVAL_LIMIT as i64) as i32
}

#[inline(always)]
fn output_sum(net: &Network, us: &[i16; H], them: &[i16; H]) -> i64 {
    let (w_us, w_them) = net.out_w.split_at(H);
    let w_us: &[i16; H] = w_us.try_into().unwrap();
    let w_them: &[i16; H] = w_them.try_into().unwrap();
    if net.small_out {
        simd::screlu_dot(us, w_us) as i64 + simd::screlu_dot(them, w_them) as i64
    } else {
        scalar::screlu_dot_wide(us, w_us) + scalar::screlu_dot_wide(them, w_them)
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

    /// Requires |w| <= 127 (see `Network::small_out`): no overflow in i32.
    #[allow(dead_code)]
    pub(crate) fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i32 {
        let mut sum = 0i32;
        for i in 0..H {
            let c = (acc[i] as i32).clamp(0, QA);
            sum += c * w[i] as i32 * c;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod simd {
    use super::{H, QA};
    use std::arch::aarch64::*;

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
    pub(crate) fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i32 {
        // SAFETY: as above. c in [0,255], |w| <= 127 so c*w fits i16; the
        // widening multiply-accumulate by c is exact in i32 (bound in
        // `Network::small_out`).
        unsafe {
            let zero = vdupq_n_s16(0);
            let qa = vdupq_n_s16(QA as i16);
            let mut s0 = vdupq_n_s32(0);
            let mut s1 = vdupq_n_s32(0);
            let mut i = 0;
            while i < H {
                let c = vminq_s16(vmaxq_s16(vld1q_s16(acc.as_ptr().add(i)), zero), qa);
                let cw = vmulq_s16(c, vld1q_s16(w.as_ptr().add(i)));
                s0 = vmlal_s16(s0, vget_low_s16(cw), vget_low_s16(c));
                s1 = vmlal_high_s16(s1, cw, c);
                i += 8;
            }
            vaddvq_s32(vaddq_s32(s0, s1))
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
    pub(crate) fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i32 {
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
            let lo = _mm256_castsi256_si128(sum);
            let hi = _mm256_extracti128_si256(sum, 1);
            let s = _mm_add_epi32(lo, hi);
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
            _mm_cvtsi128_si32(s)
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2"))))]
pub(crate) mod simd {
    pub(crate) use super::scalar::{screlu_dot, update};
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
        let mut out_w = [0i16; 2 * H];
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
        let out_b = 11;
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
        assert_eq!(evaluate_acc(&net2, &acc, Color::White) as i64, want_w);
        // Black to move: us = black acc (0, 255, 20...), them = white acc.
        // us[0]=0 (clamped), us[1]=255 -> -2*255^2; them[2]=white acc[2]=0.
        let sum_b: i64 = 255 * 255 * -2;
        let want_b = (sum_b / 255 + 11) * 400 / (255 * 64);
        assert_eq!(evaluate_acc(&net2, &acc, Color::Black) as i64, want_b);
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
                        assert_eq!(st.evaluate(stm), evaluate_acc(&net, &full, stm));
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
            assert_eq!(s as i64, 256 * 255 * 255 * w[0] as i64);
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
            x += evaluate_acc(&net, std::hint::black_box(&acc), Color::White) as i64;
        }
        let full = n as f64 / t.elapsed().as_secs_f64();
        let mut legal = Vec::new();
        board.generate_legal_moves(false, &mut legal);
        let n = 5_000_000;
        let t = Instant::now();
        for i in 0..n {
            let mv = legal[i % legal.len()];
            st.push_move(&board, mv);
            x += st.evaluate(std::hint::black_box(Color::Black)) as i64;
            st.pop();
        }
        let inc = n as f64 / t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 0..n {
            x += st.evaluate(std::hint::black_box(Color::White)) as i64;
        }
        let out_only = n as f64 / t.elapsed().as_secs_f64();
        let t = Instant::now();
        for i in 0..n {
            st.push_move(&board, legal[i % legal.len()]);
            st.pop();
        }
        let upd_only = n as f64 / t.elapsed().as_secs_f64();
        println!(
            "nnue: full refresh+eval {:.0}/s, incremental update+eval {:.0}/s, eval only {:.0}/s, update only {:.0}/s ({x})",
            full, inc, out_only, upd_only
        );
    }
}
