//! NNUE inference: (768 -> H) x 2 -> 1 with SCReLU, as specified in the
//! shared spec (see `docs`/build brief):
//!
//! - feature for perspective p, piece (colour c, type t) on square s:
//!   `kb*768 + (c == p ? 0 : 384) + t*64 + (o(s) ^ m)` with
//!   `o(s) = (p == White ? s : s ^ 56)` (the perspective's own orientation),
//!   `kb = KING_LAYOUT.buckets[o(king_p)]` (king bucket of p's own king) and
//!   `m = 7` when the layout mirrors and p's king is on files e-h, else 0.
//!   With the default layout (`LAYOUT_NONE`: one bucket, no mirror) this is
//!   the original `(c == p ? 0 : 384) + t*64 + o(s)`.
//! - net file: little-endian i16, feature weights [KB][768][H], feature
//!   biases [H], output weights [B][2H] (stm half first), output biases [B];
//!   no header.
//! - `eval = ((Σ screlu(us)·w[i] + Σ screlu(them)·w[H+i]) / QA + b) · SCALE / (QA·QB)`
//!   with `screlu(x) = clamp(x, 0, QA)²`, integer division truncating to zero.
//!
//! Accumulators are updated incrementally and lazily: `Board::make_move`
//! records the move's feature delta through `NnueState::push_move` (one stack
//! entry per ply), `undo_move` pops it, and `evaluate` materialises pending
//! entries (copy-and-update fused in one pass) only when an eval is needed.
//! A king move that changes its own perspective's bucket (or crosses the
//! mirror line) marks that perspective for a refresh instead, done at
//! materialisation from a per-(perspective, bucket) accumulator cache
//! ("finny table") with the post-move piece bitboards saved in the slot;
//! with one unmirrored bucket that path compiles away.
//! Every SIMD kernel has a scalar twin and both are exact (wrapping i16 adds;
//! output sums proven not to overflow i32, otherwise the i64 scalar path is
//! used), so results never depend on the target.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::{Board, Color, Move, Piece, PieceType};

pub(crate) const INPUTS: usize = 768;
pub(crate) const H: usize = 1024;
pub(crate) const QA: i32 = 255;
pub(crate) const QB: i32 = 64;
pub(crate) const SCALE: i32 = 400;
/// Output buckets, selected by piece count: `(pieces - 2) * BUCKETS / 32`.
pub(crate) const BUCKETS: usize = 8;

// ------------------------------------------------------------ king buckets

/// Input (king) buckets: each perspective picks its feature-transformer
/// block from its own king square, seen in its own orientation (a Black
/// perspective flips ranks first). With `mirror`, a king on files e-h also
/// flips every square's file (`sq ^ 7`) for that perspective, so the table
/// only needs to distinguish files a-d and must be left-right symmetric.
/// The layouts (names and tables) match `tools/nnue-trainer --king-buckets`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct KingLayout {
    /// Bucket of each king square (a1 = 0, h8 = 63, perspective-relative).
    pub(crate) buckets: [u8; 64],
    pub(crate) mirror: bool,
}

/// Rank-major half-board (files a-d, rank 1 first) -> left-right symmetric table.
const fn mirrored(half: [u8; 32]) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 32 {
        let (r, f) = (i / 4, i % 4);
        t[r * 8 + f] = half[i];
        t[r * 8 + 7 - f] = half[i];
        i += 1;
    }
    t
}

/// One bucket, no mirroring: the original 768-input net.
#[allow(dead_code)]
pub(crate) const LAYOUT_NONE: KingLayout = KingLayout { buckets: [0; 64], mirror: false };
/// One bucket with horizontal mirroring (`m1`).
#[allow(dead_code)]
pub(crate) const LAYOUT_M1: KingLayout = KingLayout { buckets: [0; 64], mirror: true };
/// Four buckets, no mirroring (`kb4`): back rank queenside / kingside,
/// second rank, everything further up.
#[allow(dead_code)]
#[rustfmt::skip]
pub(crate) const LAYOUT_KB4: KingLayout = KingLayout {
    buckets: [
        0, 0, 0, 0, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
    ],
    mirror: false,
};
/// Four buckets with mirroring (`kb4m`): back rank a-b / c-d, second rank, rest.
#[allow(dead_code)]
#[rustfmt::skip]
pub(crate) const LAYOUT_KB4M: KingLayout = KingLayout {
    buckets: mirrored([
        0, 0, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
    ]),
    mirror: true,
};
/// Eight buckets with mirroring (`kb8m`), the common half-board layout:
/// every back-rank file on its own, second rank in pairs, ranks 3-4, 5-8.
#[allow(dead_code)]
#[rustfmt::skip]
pub(crate) const LAYOUT_KB8M: KingLayout = KingLayout {
    buckets: mirrored([
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
    ]),
    mirror: true,
};

/// The layout the engine runs. Switching it changes the net format (KB
/// blocks of 768 feature rows): embed a net trained with the same layout.
pub(crate) const KING_LAYOUT: KingLayout = LAYOUT_KB8M;

pub(crate) const fn layout_buckets(l: &KingLayout) -> usize {
    let mut m = 0;
    let mut i = 0;
    while i < 64 {
        if l.buckets[i] as usize > m {
            m = l.buckets[i] as usize;
        }
        i += 1;
    }
    m + 1
}

/// A king layout as a type, so the accumulator code is monomorphised per
/// layout: the active one compiles to exactly the old code when it has one
/// unmirrored bucket, and tests can drive other layouts side by side.
pub(crate) trait Layout: Clone {
    const L: KingLayout;
    const KB: usize = layout_buckets(&Self::L);
    /// Whether a king move can change its perspective's feature block.
    const REFRESH: bool = Self::KB > 1 || Self::L.mirror;
}

#[derive(Clone, Copy)]
pub(crate) struct Active;
impl Layout for Active {
    const L: KingLayout = KING_LAYOUT;
}

/// Input buckets of the active layout.
pub(crate) const KB: usize = layout_buckets(&KING_LAYOUT);

/// Bytes in a net file with `kb` king buckets:
/// (kb*768*H + H + BUCKETS*2H + BUCKETS) little-endian i16.
pub(crate) const fn net_bytes(kb: usize) -> usize {
    (kb * INPUTS * H + H + BUCKETS * (2 * H + 1)) * 2
}

/// Bytes in a net file for the active layout.
#[allow(dead_code)]
pub(crate) const NET_BYTES: usize = net_bytes(KB);

/// A perspective's king context: feature block offset (`bucket * 768`) and
/// the square xor (7 when mirrored).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct KingCtx {
    base: usize,
    flip: u8,
}

impl KingCtx {
    const ZERO: Self = Self { base: 0, flip: 0 };

    /// Finny-table slot within one perspective: bucket * 2 + mirrored.
    fn slot(self) -> usize {
        self.base / INPUTS * 2 + (self.flip != 0) as usize
    }
}

#[inline(always)]
pub(crate) fn king_ctx<L: Layout>(persp: Color, ksq: u8) -> KingCtx {
    if !L::REFRESH {
        return KingCtx::ZERO;
    }
    let s = if persp == Color::White { ksq } else { ksq ^ 56 };
    let flip = if L::L.mirror && s & 7 >= 4 { 7 } else { 0 };
    KingCtx { base: L::L.buckets[s as usize] as usize * INPUTS, flip }
}

#[inline(always)]
fn board_ctx<L: Layout>(board: &Board, persp: Color) -> KingCtx {
    if !L::REFRESH {
        return KingCtx::ZERO;
    }
    king_ctx::<L>(persp, board.king_square(persp).unwrap_or(0))
}

/// NNUE scores are clamped well inside the mate range.
const EVAL_LIMIT: i32 = 20_000;

/// One 64-byte cache line of i16 weights.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([i16; 32]);

/// i16 weights stored 64-byte aligned, so every H-row (H * 2 bytes, a
/// multiple of 64) starts on a cache line and no SIMD load of a row splits
/// one. A plain `Vec<i16>` of this size is mmapped by glibc 16 bytes past a
/// page boundary, which made half of the 256-bit row loads on Linux x86
/// straddle two lines (macOS happens to page-align it).
#[derive(Clone)]
struct AlignedI16 {
    lines: Vec<Line>,
    len: usize,
}

impl AlignedI16 {
    fn new(v: &[i16]) -> Self {
        let mut lines = vec![Line([0; 32]); v.len().div_ceil(32)];
        for (l, c) in lines.iter_mut().zip(v.chunks(32)) {
            l.0[..c.len()].copy_from_slice(c);
        }
        Self { lines, len: v.len() }
    }
}

impl std::ops::Deref for AlignedI16 {
    type Target = [i16];
    #[inline(always)]
    fn deref(&self) -> &[i16] {
        // SAFETY: `Line` is repr(C) [i16; 32] with no padding, so `lines` is
        // `32 * lines.len() >= len` contiguous, initialised i16s.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr() as *const i16, self.len) }
    }
}

#[derive(Clone)]
pub(crate) struct Network {
    /// Feature-major: weights of feature f are `ft_w[f*H .. (f+1)*H]`,
    /// `kb * 768` features.
    ft_w: AlignedI16,
    kb: usize,
    ft_b: [i16; H],
    /// Bucket-major: bucket b's weights are `out_w[b*2H .. (b+1)*2H]`, stm half first.
    out_w: AlignedI16,
    out_b: [i16; BUCKETS],
    /// All |out_w| <= 127: the SIMD output kernel's i16 products and i32
    /// per-perspective sums cannot overflow (256 * 255^2 * 127 < 2^31).
    small_out: bool,
}

impl Network {
    /// A net for the active king layout.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::from_bytes_kb(bytes, KB)
    }

    pub(crate) fn from_bytes_kb(bytes: &[u8], kb: usize) -> Result<Self, String> {
        if bytes.len() != net_bytes(kb) {
            return Err(format!(
                "net is {} bytes, expected {} (H={H}, {kb} king bucket(s), {BUCKETS} output bucket(s))",
                bytes.len(),
                net_bytes(kb)
            ));
        }
        let mut words = bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]));
        let ft_w: Vec<i16> = words.by_ref().take(kb * INPUTS * H).collect();
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
        assert!(!ft_w.is_empty() && ft_w.len() % (INPUTS * H) == 0);
        assert_eq!(out_w.len(), BUCKETS * 2 * H);
        let kb = ft_w.len() / (INPUTS * H);
        let small_out = out_w.iter().all(|w| (-127..=127).contains(w));
        Self { ft_w: AlignedI16::new(&ft_w), kb, ft_b, out_w: AlignedI16::new(&out_w), out_b, small_out }
    }

    pub(crate) fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Self::from_bytes(&bytes)
    }

    #[cfg(test)]
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(net_bytes(self.kb));
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
        Self::random_kb(seed, ft_range, out_range, KB)
    }

    #[cfg(test)]
    pub(crate) fn random_kb(seed: u64, ft_range: i16, out_range: i16, kb: usize) -> Self {
        let mut s = seed | 1;
        let mut next = move |range: i16| -> i16 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let span = 2 * range as i64 + 1;
            ((s % span as u64) as i64 - range as i64) as i16
        };
        let ft_w = (0..kb * INPUTS * H).map(|_| next(ft_range)).collect();
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

/// Feature of piece `p` on `sq` for perspective `persp`, whose king context is `k`.
#[inline(always)]
pub(crate) fn feature(persp: Color, p: Piece, sq: u8, k: KingCtx) -> usize {
    feature_ck(persp, p.color as usize, p.kind as usize, sq, k)
}

/// `feature` with the piece as (colour index, kind index).
#[inline(always)]
fn feature_ck(persp: Color, color: usize, kind: usize, sq: u8, k: KingCtx) -> usize {
    let side = if color == persp as usize { 0 } else { 384 };
    let s = if persp == Color::White { sq } else { sq ^ 56 };
    k.base + side + kind * 64 + (s ^ k.flip) as usize
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct Accumulator {
    /// Indexed by perspective: 0 = White, 1 = Black.
    pub(crate) v: [[i16; H]; 2],
}

impl Accumulator {
    const ZERO: Self = Self { v: [[0; H]; 2] };

    /// From scratch, for the active layout.
    #[cfg(test)]
    pub(crate) fn refresh(net: &Network, board: &Board) -> Self {
        Self::refresh_with::<Active>(net, board)
    }

    pub(crate) fn refresh_with<L: Layout>(net: &Network, board: &Board) -> Self {
        debug_assert_eq!(net.kb, L::KB);
        let ctx = [board_ctx::<L>(board, Color::White), board_ctx::<L>(board, Color::Black)];
        let mut acc = Self { v: [net.ft_b; 2] };
        for sq in 0..64u8 {
            if let Some(p) = board.squares[sq as usize] {
                for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
                    let row = net.row(feature(persp, p, sq, ctx[pi]));
                    for (a, w) in acc.v[pi].iter_mut().zip(row.iter()) {
                        *a = a.wrapping_add(*w);
                    }
                }
            }
        }
        acc
    }
}

/// Piece bitboards, `[colour][kind]` (as `BoardBB::pieces`).
type PieceBBs = [[u64; 6]; 2];

/// Finny-table entry: the accumulator of one perspective in one king
/// context for the pieces in `bb`. Refreshing applies only the difference
/// between `bb` and the new position, then stores the result back.
#[derive(Clone)]
#[repr(C, align(64))]
struct CacheEntry {
    acc: [i16; H],
    bb: PieceBBs,
}

/// Refresh perspective `persp` in king context `k` for the position `bb`
/// through its cache entry; the result goes to `out` and the entry.
/// Exact: wrapping i16 sums do not depend on the order of the rows.
fn refresh_cached(net: &Network, e: &mut CacheEntry, persp: Color, k: KingCtx, bb: &PieceBBs, out: &mut [i16; H]) {
    let mut adds = [0usize; 32];
    let mut subs = [0usize; 32];
    let (mut na, mut ns) = (0, 0);
    for c in 0..2 {
        for t in 0..6 {
            let (new, old) = (bb[c][t], e.bb[c][t]);
            let (mut a, mut s) = (new & !old, old & !new);
            while a != 0 && na < 32 {
                adds[na] = feature_ck(persp, c, t, a.trailing_zeros() as u8, k);
                na += 1;
                a &= a - 1;
            }
            while s != 0 && ns < 32 {
                subs[ns] = feature_ck(persp, c, t, s.trailing_zeros() as u8, k);
                ns += 1;
                s &= s - 1;
            }
            debug_assert!(a == 0 && s == 0, "more than 32 pieces");
        }
    }
    let (mut i, mut j) = (0, 0);
    while i < na || j < ns {
        let ka = (na - i).min(4);
        let ks = (ns - j).min(4);
        let ar: [&[i16; H]; 4] = std::array::from_fn(|x| net.row(adds[(i + x.min(ka.saturating_sub(1))).min(31)]));
        let sr: [&[i16; H]; 4] = std::array::from_fn(|x| net.row(subs[(j + x.min(ks.saturating_sub(1))).min(31)]));
        // In place on the entry, every chunk also stored to `out` (the last
        // one's stores leave it final), so neither side needs a copy.
        let acc = e.acc.as_mut_ptr();
        // SAFETY: both outputs and the source cover H i16s; the kernel
        // reads each lane of `acc` before storing it.
        unsafe { simd::update_dual(acc, out.as_mut_ptr(), acc, &ar[..ka], &sr[..ks]) };
        i += ka;
        j += ks;
    }
    if na == 0 && ns == 0 {
        *out = e.acc;
    }
    e.bb = *bb;
}

/// Cache line size the prefetches step by (Apple cores: 128 bytes).
const LINE_I16: usize = if cfg!(target_arch = "aarch64") { 64 } else { 32 };

#[inline(always)]
fn prefetch_row(row: &[i16; H]) {
    let mut i = 0;
    while i < H {
        prefetch(row[i..].as_ptr());
        i += LINE_I16;
    }
}

#[inline(always)]
fn prefetch(p: *const i16) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch is only a hint; SSE is baseline on x86-64.
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a prefetch is only a hint and never faults.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags))
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
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
    const EMPTY: Self = Self { add: [(0, None); 2], sub: [(0, None); 2], n_add: 0, n_sub: 0 };

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

/// Piece bitboards after `d` is applied to `board`.
fn bb_after(board: &Board, d: &Delta) -> PieceBBs {
    let mut bb = board.bb.pieces;
    for &(sq, p) in d.sub[..d.n_sub].iter().chain(d.add[..d.n_add].iter()) {
        let p = p.unwrap();
        bb[p.color as usize][p.kind as usize] ^= 1u64 << sq;
    }
    bb
}

/// One ply of the accumulator stack: the move's feature delta from the
/// entry below, and the accumulator it yields once materialised. With king
/// buckets (`L::REFRESH`) also each perspective's king context after the
/// move and whether it needs a refresh; `bb` (the post-move pieces) is only
/// written when one does.
#[derive(Clone, Copy)]
struct Slot {
    acc: Accumulator,
    delta: Delta,
    ctx: [KingCtx; 2],
    refresh: [bool; 2],
    bb: PieceBBs,
}

impl Slot {
    const EMPTY: Self = Self {
        acc: Accumulator::ZERO,
        delta: Delta::EMPTY,
        ctx: [KingCtx::ZERO; 2],
        refresh: [false; 2],
        bb: [[0; 6]; 2],
    };
}

/// Per-board NNUE state: the net and one accumulator per ply, for king
/// layout `L` (the active one unless a test picks another).
///
/// Updates are lazy: `push_move` only records the move's feature delta, and
/// `evaluate` materialises the missing accumulators bottom-up from the last
/// computed one. Nodes that never evaluate (TT cutoffs, eval-cache hits,
/// checks, repetitions) then cost no accumulator work at all. The values
/// are identical to eager updating: the same wrapping adds in the same chain.
#[derive(Clone)]
pub(crate) struct NnueState<L: Layout = Active> {
    net: Arc<Network>,
    /// `stack[..=computed]` hold valid accumulators for the current line;
    /// `computed <= top` always. Behind a RefCell so `evaluate(&self)` can
    /// fill in the pending entries.
    stack: RefCell<Vec<Slot>>,
    computed: Cell<usize>,
    top: usize,
    /// Finny table, `[perspective][bucket * 2 + mirrored]`; empty unless
    /// `L::REFRESH`.
    cache: RefCell<Vec<CacheEntry>>,
    _layout: PhantomData<L>,
}

impl<L: Layout> PartialEq for NnueState<L> {
    /// Board equality is about the position; the accumulator is derived data.
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl<L: Layout> Eq for NnueState<L> {}

impl<L: Layout> NnueState<L> {
    pub(crate) fn new(net: Arc<Network>, board: &Board) -> Self {
        assert_eq!(net.kb, L::KB, "net has {} king buckets, layout needs {}", net.kb, L::KB);
        let cache = if L::REFRESH {
            vec![CacheEntry { acc: net.ft_b, bb: [[0; 6]; 2] }; 2 * L::KB * 2]
        } else {
            Vec::new()
        };
        let mut v = Vec::with_capacity(256);
        v.push(Slot::EMPTY);
        let mut s = Self {
            net,
            stack: RefCell::new(v),
            computed: Cell::new(0),
            top: 0,
            cache: RefCell::new(cache),
            _layout: PhantomData,
        };
        s.refresh(board);
        s
    }

    /// Rebuild from scratch and reset the stack to one entry.
    pub(crate) fn refresh(&mut self, board: &Board) {
        self.top = 0;
        self.computed.set(0);
        let stack = self.stack.get_mut();
        stack[0].acc = Accumulator::refresh_with::<L>(&self.net, board);
        if L::REFRESH {
            stack[0].ctx = [board_ctx::<L>(board, Color::White), board_ctx::<L>(board, Color::Black)];
        }
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
            stack.push(Slot { delta, ..Slot::EMPTY });
        } else {
            stack[self.top].delta = delta;
        }
        if L::REFRESH {
            let prev = stack[self.top - 1].ctx;
            let slot = &mut stack[self.top];
            slot.ctx = prev;
            slot.refresh = [false; 2];
            let moving = delta.sub[0].1.unwrap();
            if moving.kind == PieceType::King {
                let pi = moving.color as usize;
                let k = king_ctx::<L>(moving.color, mv.to);
                if k != prev[pi] {
                    slot.ctx[pi] = k;
                    slot.refresh[pi] = true;
                    slot.bb = bb_after(board, &delta);
                }
            }
        }
        // Start fetching the feature rows this ply will need: the update
        // runs at the next evaluate, typically after the child's TT probe
        // and move generation, by which time the rows are on their way in.
        let slot = &stack[self.top];
        let net = &*self.net;
        for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
            if L::REFRESH && slot.refresh[pi] {
                continue;
            }
            let k = if L::REFRESH { slot.ctx[pi] } else { KingCtx::ZERO };
            for &(sq, p) in delta.sub[..delta.n_sub].iter().chain(delta.add[..delta.n_add].iter()) {
                prefetch_row(net.row(feature(persp, p.unwrap(), sq, k)));
            }
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
        self.materialise_to(self.top);
    }

    /// Bring `stack[..=target]` up to date (`target <= top`).
    #[inline]
    fn materialise_to(&self, target: usize) {
        let mut k = self.computed.get();
        if k >= target {
            return;
        }
        let mut stack = self.stack.borrow_mut();
        let net = &*self.net;
        while k < target {
            let (lo, hi) = stack.split_at_mut(k + 1);
            apply_delta::<L>(net, &lo[k].acc, &mut hi[0], &self.cache);
            k += 1;
        }
        self.computed.set(k);
    }

    /// Score from the side to move's point of view, in centipawns, for a
    /// position with `pieces` pieces on the board (selects the output bucket).
    #[inline]
    pub(crate) fn evaluate(&self, stm: Color, pieces: u32) -> i32 {
        let net = &*self.net;
        if self.computed.get() < self.top && net.small_out {
            // Bring all but the last pending ply up to date, then update
            // both perspectives for that ply and take their output sums in
            // the same pass over the lanes.
            self.materialise_to(self.top - 1);
            let k = self.top - 1;
            let b = bucket(pieces);
            let mut stack = self.stack.borrow_mut();
            let (lo, hi) = stack.split_at_mut(self.top);
            let sum = apply_delta_with::<L, true>(net, &lo[k].acc, &mut hi[0], &self.cache, stm, b);
            self.computed.set(self.top);
            return finish(net, b, sum);
        }
        self.materialise();
        let stack = self.stack.borrow();
        evaluate_acc(net, &stack[self.top].acc, stm, pieces)
    }
}

/// `dst.acc = src + dst.delta`, both perspectives in one copy-and-update
/// pass; a perspective marked for refresh is rebuilt from the finny table.
#[inline(always)]
fn apply_delta<L: Layout>(net: &Network, src: &Accumulator, dst: &mut Slot, cache: &RefCell<Vec<CacheEntry>>) {
    apply_delta_with::<L, false>(net, src, dst, cache, Color::White, 0);
}

/// `apply_delta`, and with `EVAL` also the output sum of the result for
/// side to move `stm` and output bucket `b` (requires `net.small_out`),
/// fused into the update pass. Without `EVAL` it returns 0.
#[inline(always)]
fn apply_delta_with<L: Layout, const EVAL: bool>(
    net: &Network,
    src: &Accumulator,
    dst: &mut Slot,
    cache: &RefCell<Vec<CacheEntry>>,
    stm: Color,
    b: usize,
) -> i64 {
    let d = dst.delta;
    let mut sum = 0i64;
    for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
        let w = if EVAL { out_half(net, b, persp == stm) } else { &net.ft_b };
        let k = if L::REFRESH { dst.ctx[pi] } else { KingCtx::ZERO };
        if L::REFRESH && dst.refresh[pi] {
            let mut c = cache.borrow_mut();
            let bb = dst.bb;
            refresh_cached(net, &mut c[pi * 2 * L::KB + k.slot()], persp, k, &bb, &mut dst.acc.v[pi]);
            if EVAL {
                sum += simd::screlu_dot(&dst.acc.v[pi], w);
            }
            continue;
        }
        let f = |e: (u8, Option<Piece>)| net.row(feature(persp, e.1.unwrap(), e.0, k));
        let a0 = f(d.add[0]);
        let s0 = f(d.sub[0]);
        let (s, o) = (&src.v[pi], &mut dst.acc.v[pi]);
        let run = |o: &mut [i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]| -> i64 {
            if EVAL {
                simd::update_dot(o, s, adds, subs, w)
            } else {
                update(o, s, adds, subs);
                0
            }
        };
        sum += match (d.n_add, d.n_sub) {
            (1, 1) => run(o, &[a0], &[s0]),
            (1, 2) => run(o, &[a0], &[s0, f(d.sub[1])]),
            (2, 2) => run(o, &[a0, f(d.add[1])], &[s0, f(d.sub[1])]),
            _ => unreachable!(),
        };
    }
    sum
}

/// Output weights of bucket `b` for the side to move's half (`us`) or the other.
#[inline(always)]
fn out_half(net: &Network, b: usize, us: bool) -> &[i16; H] {
    let o = b * 2 * H + if us { 0 } else { H };
    net.out_w[o..o + H].try_into().unwrap()
}

/// Output sum -> clamped centipawn score for output bucket `b`.
#[inline(always)]
fn finish(net: &Network, b: usize, sum: i64) -> i32 {
    let raw = (sum / QA as i64 + net.out_b[b] as i64) * SCALE as i64 / (QA as i64 * QB as i64);
    raw.clamp(-EVAL_LIMIT as i64, EVAL_LIMIT as i64) as i32
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
    finish(net, b, output_sum(net, b, us, them))
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

    /// `out = out2 = src + Σ adds - Σ subs`; `src` may alias either output.
    ///
    /// # Safety
    /// `out`, `out2` and `src` must each cover H i16s.
    #[allow(dead_code)]
    pub(crate) unsafe fn update_dual(out: *mut i16, out2: *mut i16, src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        for i in 0..H {
            let mut v = *src.add(i);
            for a in adds {
                v = v.wrapping_add(a[i]);
            }
            for s in subs {
                v = v.wrapping_sub(s[i]);
            }
            *out.add(i) = v;
            *out2.add(i) = v;
        }
    }

    /// `update` followed by `screlu_dot(out, w)`; requires |w| <= 127.
    #[allow(dead_code)]
    pub(crate) fn update_dot(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
    ) -> i64 {
        update(out, src, adds, subs);
        screlu_dot(out, w)
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

    // The kernels move 32 lanes (64 bytes) per step with the multi-register
    // ld1/st1 forms: fewer load/store instructions than one q register at a
    // time, measured ~8% faster on the update kernel alone (M3).
    const _: () = assert!(H % 32 == 0);

    #[inline(always)]
    unsafe fn ld4(p: *const i16) -> [int16x8_t; 4] {
        let v = vld1q_s16_x4(p);
        [v.0, v.1, v.2, v.3]
    }

    #[inline(always)]
    unsafe fn st4(p: *mut i16, v: [int16x8_t; 4]) {
        vst1q_s16_x4(p, int16x8x4_t(v[0], v[1], v[2], v[3]))
    }

    /// `src[i..i+32] + Σ adds - Σ subs` as four vectors.
    #[inline(always)]
    unsafe fn delta4(src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]], i: usize) -> [int16x8_t; 4] {
        let mut v = ld4(src.add(i));
        for a in adds {
            let x = ld4(a.as_ptr().add(i));
            for j in 0..4 {
                v[j] = vaddq_s16(v[j], x[j]);
            }
        }
        for s in subs {
            let x = ld4(s.as_ptr().add(i));
            for j in 0..4 {
                v[j] = vsubq_s16(v[j], x[j]);
            }
        }
        v
    }

    #[inline(always)]
    pub(crate) fn update(out: &mut [i16; H], src: &[i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        // SAFETY: NEON is baseline on aarch64; all pointers cover H i16s and
        // the loop reads/writes 32 lanes at offsets i with i + 32 <= H.
        unsafe {
            let mut i = 0;
            while i < H {
                st4(out.as_mut_ptr().add(i), delta4(src.as_ptr(), adds, subs, i));
                i += 32;
            }
        }
    }

    /// `out = out2 = src + Σ adds - Σ subs`; `src` may alias either output.
    ///
    /// # Safety
    /// `out`, `out2` and `src` must each cover H i16s.
    #[inline(always)]
    pub(crate) unsafe fn update_dual(out: *mut i16, out2: *mut i16, src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        let mut i = 0;
        while i < H {
            let v = delta4(src, adds, subs, i);
            st4(out.add(i), v);
            st4(out2.add(i), v);
            i += 32;
        }
    }

    /// `update` and `screlu_dot(out, w)` in one pass (|w| <= 127): the
    /// output arithmetic overlaps the update's loads and stores.
    #[inline(always)]
    pub(crate) fn update_dot(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
    ) -> i64 {
        // SAFETY: as in `update` and `screlu_dot`.
        unsafe {
            let zero = vdupq_n_s16(0);
            let qa = vdupq_n_s16(QA as i16);
            let mut s = [vdupq_n_s32(0); 8];
            let mut i = 0;
            while i < H {
                let v = delta4(src.as_ptr(), adds, subs, i);
                st4(out.as_mut_ptr().add(i), v);
                let wv = ld4(w.as_ptr().add(i));
                for j in 0..4 {
                    let c = vminq_s16(vmaxq_s16(v[j], zero), qa);
                    let cw = vmulq_s16(c, wv[j]);
                    s[2 * j] = vmlal_s16(s[2 * j], vget_low_s16(cw), vget_low_s16(c));
                    s[2 * j + 1] = vmlal_high_s16(s[2 * j + 1], cw, c);
                }
                i += 32;
            }
            s.iter().map(|&x| vaddlvq_s32(x)).sum()
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
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::update(out, src, adds, subs) };
        }
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

    /// `out = out2 = src + Σ adds - Σ subs`; `src` may alias either output.
    ///
    /// # Safety
    /// `out`, `out2` and `src` must each cover H i16s.
    #[inline(always)]
    pub(crate) unsafe fn update_dual(out: *mut i16, out2: *mut i16, src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::update_dual(out, out2, src, adds, subs) };
        }
        super::avx2_kernels::update_dual(out, out2, src, adds, subs)
    }

    /// `update` and `screlu_dot(out, w)` in one pass (|w| <= 127).
    #[inline(always)]
    pub(crate) fn update_dot(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
    ) -> i64 {
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::update_dot(out, src, adds, subs, w) };
        }
        // SAFETY: AVX2 is enabled at compile time (cfg).
        unsafe { super::avx2_kernels::update_dot(out, src, adds, subs, w) }
    }

    #[inline(always)]
    pub(crate) fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i64 {
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::screlu_dot(acc, w) };
        }
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
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::screlu_dot_wide(acc, w) };
        }
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

/// AVX-512 (F + BW) kernels, chosen at runtime on top of either x86-64
/// `simd` module (CI builds x86-64-v3, which has no AVX-512). Same integer
/// arithmetic as the AVX2/scalar kernels, so results are bit-identical:
/// wrapping i16 adds/subs, exact i32 products and an i64 final reduction.
/// `RUSTFISH_NO_AVX512=1` (read once) forces the AVX2 path, for A/B checks
/// on the same machine. The (adds, subs) counts of the incremental update
/// are dispatched to fixed-size instances so the row loops unroll.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    use super::{H, QA};
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    const _: () = assert!(H % 64 == 0);

    /// AVX-512F and AVX-512BW both present (and not disabled by env var).
    #[inline]
    pub(crate) fn enabled() -> bool {
        static HAVE: OnceLock<bool> = OnceLock::new();
        *HAVE.get_or_init(|| {
            let off = std::env::var_os("RUSTFISH_NO_AVX512").is_some_and(|v| !v.is_empty() && v != "0");
            !off && supported()
        })
    }

    pub(crate) fn supported() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }

    #[inline(always)]
    unsafe fn delta<const NA: usize, const NS: usize>(v: __m512i, adds: &[&[i16; H]], subs: &[&[i16; H]], i: usize) -> __m512i {
        let mut v = v;
        for a in &adds[..NA] {
            v = _mm512_add_epi16(v, _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i));
        }
        for s in &subs[..NS] {
            v = _mm512_sub_epi16(v, _mm512_loadu_si512(s.as_ptr().add(i) as *const __m512i));
        }
        v
    }

    #[inline(always)]
    unsafe fn delta_dyn(v: __m512i, adds: &[&[i16; H]], subs: &[&[i16; H]], i: usize) -> __m512i {
        let mut v = v;
        for a in adds {
            v = _mm512_add_epi16(v, _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i));
        }
        for s in subs {
            v = _mm512_sub_epi16(v, _mm512_loadu_si512(s.as_ptr().add(i) as *const __m512i));
        }
        v
    }

    #[inline(always)]
    unsafe fn update_n<const NA: usize, const NS: usize>(out: *mut i16, src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        let mut i = 0;
        while i < H {
            let v = delta::<NA, NS>(_mm512_loadu_si512(src.add(i) as *const __m512i), adds, subs, i);
            _mm512_storeu_si512(out.add(i) as *mut __m512i, v);
            i += 32;
        }
    }

    /// # Safety
    /// The CPU must support AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn update(out: &mut [i16; H], src: &[i16; H], adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        let (o, sp) = (out.as_mut_ptr(), src.as_ptr());
        match (adds.len(), subs.len()) {
            (1, 1) => update_n::<1, 1>(o, sp, adds, subs),
            (1, 2) => update_n::<1, 2>(o, sp, adds, subs),
            (2, 2) => update_n::<2, 2>(o, sp, adds, subs),
            _ => {
                let mut i = 0;
                while i < H {
                    let v = delta_dyn(_mm512_loadu_si512(sp.add(i) as *const __m512i), adds, subs, i);
                    _mm512_storeu_si512(o.add(i) as *mut __m512i, v);
                    i += 32;
                }
            }
        }
    }

    /// `out = out2 = src + Σ adds - Σ subs`; `src` may alias either output.
    /// # Safety
    /// The CPU must support AVX-512F and AVX-512BW; all pointers cover H i16s.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn update_dual(out: *mut i16, out2: *mut i16, src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        let mut i = 0;
        while i < H {
            let v = delta_dyn(_mm512_loadu_si512(src.add(i) as *const __m512i), adds, subs, i);
            _mm512_storeu_si512(out.add(i) as *mut __m512i, v);
            _mm512_storeu_si512(out2.add(i) as *mut __m512i, v);
            i += 32;
        }
    }

    #[inline(always)]
    unsafe fn update_dot_n<const NA: usize, const NS: usize>(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
    ) -> i64 {
        // c in [0,255], |w| <= 127 so c*w fits i16; madd(c*w, c) sums two
        // exact i32 products. Each i32 lane collects 2*H/64 products of at
        // most 255^2*127 (exact for H <= 8192); reduced in i64.
        let zero = _mm512_setzero_si512();
        let qa = _mm512_set1_epi16(QA as i16);
        let mut s = [_mm512_setzero_si512(); 2];
        let mut i = 0;
        while i < H {
            for j in 0..2 {
                let o = i + 32 * j;
                let v = _mm512_loadu_si512(src.as_ptr().add(o) as *const __m512i);
                let v = if NA == 0 && NS == 0 { delta_dyn(v, adds, subs, o) } else { delta::<NA, NS>(v, adds, subs, o) };
                _mm512_storeu_si512(out.as_mut_ptr().add(o) as *mut __m512i, v);
                let c = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
                let wv = _mm512_loadu_si512(w.as_ptr().add(o) as *const __m512i);
                s[j] = _mm512_add_epi32(s[j], _mm512_madd_epi16(_mm512_mullo_epi16(c, wv), c));
            }
            i += 64;
        }
        reduce2(s[0], s[1])
    }

    /// `update` then `screlu_dot(out, w)` in one pass; |w| <= 127.
    /// # Safety
    /// The CPU must support AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn update_dot(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
    ) -> i64 {
        match (adds.len(), subs.len()) {
            (1, 1) => update_dot_n::<1, 1>(out, src, adds, subs, w),
            (1, 2) => update_dot_n::<1, 2>(out, src, adds, subs, w),
            (2, 2) => update_dot_n::<2, 2>(out, src, adds, subs, w),
            _ => update_dot_n::<0, 0>(out, src, adds, subs, w),
        }
    }

    #[inline(always)]
    unsafe fn reduce2(a: __m512i, b: __m512i) -> i64 {
        let wide = |v: __m512i| {
            _mm512_add_epi64(
                _mm512_cvtepi32_epi64(_mm512_castsi512_si256(v)),
                _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(v)),
            )
        };
        _mm512_reduce_add_epi64(_mm512_add_epi64(wide(a), wide(b)))
    }

    /// Requires |w| <= 127 (see `Network::small_out`).
    /// # Safety
    /// The CPU must support AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn screlu_dot(acc: &[i16; H], w: &[i16; H]) -> i64 {
        // Bounds as in `update_dot_n`.
        let zero = _mm512_setzero_si512();
        let qa = _mm512_set1_epi16(QA as i16);
        let mut s0 = _mm512_setzero_si512();
        let mut s1 = _mm512_setzero_si512();
        let mut i = 0;
        while i < H {
            let a0 = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
            let a1 = _mm512_loadu_si512(acc.as_ptr().add(i + 32) as *const __m512i);
            let c0 = _mm512_min_epi16(_mm512_max_epi16(a0, zero), qa);
            let c1 = _mm512_min_epi16(_mm512_max_epi16(a1, zero), qa);
            let w0 = _mm512_loadu_si512(w.as_ptr().add(i) as *const __m512i);
            let w1 = _mm512_loadu_si512(w.as_ptr().add(i + 32) as *const __m512i);
            s0 = _mm512_add_epi32(s0, _mm512_madd_epi16(_mm512_mullo_epi16(c0, w0), c0));
            s1 = _mm512_add_epi32(s1, _mm512_madd_epi16(_mm512_mullo_epi16(c1, w1), c1));
            i += 64;
        }
        reduce2(s0, s1)
    }

    /// Exact i64 screlu dot for any i16 weights (the `!small_out` path).
    /// # Safety
    /// The CPU must support AVX-512F and AVX-512BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn screlu_dot_wide(acc: &[i16; H], w: &[i16; H]) -> i64 {
        // c^2 <= 65025 and |w| <= 32767: each product fits signed i32 before
        // being widened to i64 by `_mm512_mul_epi32` (even lanes; odd via shift).
        let zero = _mm512_setzero_si512();
        let qa = _mm512_set1_epi16(QA as i16);
        let mut sum = _mm512_setzero_si512();
        let mut i = 0;
        while i < H {
            let a = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
            let c16 = _mm512_min_epi16(_mm512_max_epi16(a, zero), qa);
            let w16 = _mm512_loadu_si512(w.as_ptr().add(i) as *const __m512i);
            let c_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(c16));
            let c_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(c16));
            let w_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(w16));
            let w_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(w16));
            let cc_lo = _mm512_mullo_epi32(c_lo, c_lo);
            let cc_hi = _mm512_mullo_epi32(c_hi, c_hi);
            let p_lo = _mm512_add_epi64(
                _mm512_mul_epi32(cc_lo, w_lo),
                _mm512_mul_epi32(_mm512_srli_epi64::<32>(cc_lo), _mm512_srli_epi64::<32>(w_lo)),
            );
            let p_hi = _mm512_add_epi64(
                _mm512_mul_epi32(cc_hi, w_hi),
                _mm512_mul_epi32(_mm512_srli_epi64::<32>(cc_hi), _mm512_srli_epi64::<32>(w_hi)),
            );
            sum = _mm512_add_epi64(sum, _mm512_add_epi64(p_lo, p_hi));
            i += 32;
        }
        _mm512_reduce_add_epi64(sum)
    }
}

/// Which NNUE kernel set this process uses (for logs / CI).
pub(crate) fn simd_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512::enabled() {
            "avx512"
        } else if is_x86_feature_detected!("avx2") {
            "avx2"
        } else {
            "scalar"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        "scalar"
    }
}

/// AVX2 kernels shared by the compile-time and the runtime-detected x86 paths.
#[cfg(target_arch = "x86_64")]
mod avx2_kernels {
    use super::{H, QA};
    use std::arch::x86_64::*;

    /// `out = out2 = src + Σ adds - Σ subs`; `src` may alias either output.
    ///
    /// # Safety
    /// The CPU must support AVX2; `out`, `out2` and `src` must each cover H i16s.
    #[inline(always)]
    pub(super) unsafe fn update_dual(out: *mut i16, out2: *mut i16, src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        let mut i = 0;
        while i < H {
            let mut v = _mm256_loadu_si256(src.add(i) as *const __m256i);
            for a in adds {
                v = _mm256_add_epi16(v, _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i));
            }
            for s in subs {
                v = _mm256_sub_epi16(v, _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i));
            }
            _mm256_storeu_si256(out.add(i) as *mut __m256i, v);
            _mm256_storeu_si256(out2.add(i) as *mut __m256i, v);
            i += 16;
        }
    }

    /// `update` then `screlu_dot(out, w)` in one pass; |w| <= 127.
    ///
    /// # Safety
    /// The CPU must support AVX2.
    #[inline(always)]
    pub(super) unsafe fn update_dot(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
    ) -> i64 {
        const { assert!(H % 32 == 0) };
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut sum = [_mm256_setzero_si256(); 2];
        let mut i = 0;
        while i < H {
            for j in 0..2 {
                let o = i + 16 * j;
                let mut v = _mm256_loadu_si256(src.as_ptr().add(o) as *const __m256i);
                for a in adds {
                    v = _mm256_add_epi16(v, _mm256_loadu_si256(a.as_ptr().add(o) as *const __m256i));
                }
                for s in subs {
                    v = _mm256_sub_epi16(v, _mm256_loadu_si256(s.as_ptr().add(o) as *const __m256i));
                }
                _mm256_storeu_si256(out.as_mut_ptr().add(o) as *mut __m256i, v);
                let c = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
                let wv = _mm256_loadu_si256(w.as_ptr().add(o) as *const __m256i);
                let cw = _mm256_mullo_epi16(c, wv);
                sum[j] = _mm256_add_epi32(sum[j], _mm256_madd_epi16(cw, c));
            }
            i += 32;
        }
        // Each i32 lane holds H/16 products (exact for H <= 4096); reduce in i64.
        let mut lanes = [0i32; 16];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum[0]);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(8) as *mut __m256i, sum[1]);
        lanes.iter().map(|&x| x as i64).sum()
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) mod simd {
    pub(crate) use super::scalar::{screlu_dot, screlu_dot_wide, update, update_dot, update_dual};
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
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::update(out, src, adds, subs) };
        }
        if avx2() {
            // SAFETY: the feature was detected above and all slices cover H i16s.
            unsafe { update_avx2(out, src, adds, subs) }
        } else {
            super::scalar::update(out, src, adds, subs)
        }
    }

    #[inline(always)]
    pub(crate) fn update_dot(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
    ) -> i64 {
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::update_dot(out, src, adds, subs, w) };
        }
        if avx2() {
            // SAFETY: detected above; |w| <= 127 bound as in the scalar path.
            unsafe { update_dot_avx2(out, src, adds, subs, w) }
        } else {
            super::scalar::update_dot(out, src, adds, subs, w)
        }
    }

    /// # Safety
    /// `out`, `out2` and `src` must each cover H i16s.
    #[inline(always)]
    pub(crate) unsafe fn update_dual(out: *mut i16, out2: *mut i16, src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::update_dual(out, out2, src, adds, subs) };
        }
        if avx2() {
            update_dual_avx2(out, out2, src, adds, subs)
        } else {
            super::scalar::update_dual(out, out2, src, adds, subs)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn update_dual_avx2(out: *mut i16, out2: *mut i16, src: *const i16, adds: &[&[i16; H]], subs: &[&[i16; H]]) {
        super::avx2_kernels::update_dual(out, out2, src, adds, subs)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn update_dot_avx2(
        out: &mut [i16; H],
        src: &[i16; H],
        adds: &[&[i16; H]],
        subs: &[&[i16; H]],
        w: &[i16; H],
    ) -> i64 {
        super::avx2_kernels::update_dot(out, src, adds, subs, w)
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
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::screlu_dot(acc, w) };
        }
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
        if super::avx512::enabled() {
            // SAFETY: AVX-512F/BW detected at runtime.
            return unsafe { super::avx512::screlu_dot_wide(acc, w) };
        }
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
        let net2 = Network::from_bytes_kb(&net.to_bytes(), 1).unwrap();
        let mut board = Board::new();
        board.set_fen("8/8/8/8/8/8/8/4K3 w - - 0 1");
        let acc = Accumulator::refresh_with::<None1>(&net2, &board);
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

    macro_rules! layout {
        ($name:ident, $l:expr) => {
            #[derive(Clone, Copy)]
            struct $name;
            impl Layout for $name {
                const L: KingLayout = $l;
            }
        };
    }
    layout!(None1, LAYOUT_NONE);
    layout!(M1, LAYOUT_M1);
    layout!(Kb4, LAYOUT_KB4);
    layout!(Kb4m, LAYOUT_KB4M);
    layout!(Kb8m, LAYOUT_KB8M);

    #[test]
    fn layouts_are_well_formed() {
        assert_eq!((None1::KB, None1::REFRESH), (1, false));
        assert_eq!((M1::KB, M1::REFRESH), (1, true));
        assert_eq!((Kb4::KB, Kb4m::KB, Kb8m::KB), (4, 4, 8));
        for l in [LAYOUT_NONE, LAYOUT_M1, LAYOUT_KB4, LAYOUT_KB4M, LAYOUT_KB8M] {
            let kb = layout_buckets(&l);
            for b in 0..kb as u8 {
                assert!(l.buckets.contains(&b), "bucket {b} unused");
            }
            if l.mirror {
                for sq in 0..64 {
                    assert_eq!(l.buckets[sq], l.buckets[sq ^ 7], "mirrored layout must be symmetric");
                }
            }
        }
        // The kb8m table, rank 1 and 2 (files a-h).
        assert_eq!(&LAYOUT_KB8M.buckets[..16], &[0, 1, 2, 3, 3, 2, 1, 0, 4, 4, 5, 5, 5, 5, 4, 4]);
    }

    /// KB = 1 without mirroring is exactly the original 768-input net.
    #[test]
    fn kb1_matches_legacy_layout() {
        // Only meaningful while main embeds a KB=1 net; a king-bucket net on main skips it.
        if KING_LAYOUT != LAYOUT_NONE {
            return;
        }
        assert_eq!(NET_BYTES, (768 * H + H + BUCKETS * (2 * H + 1)) * 2);
        assert_eq!(net_bytes(1), NET_BYTES);
        for ksq in 0..64 {
            for persp in [Color::White, Color::Black] {
                assert_eq!(king_ctx::<Active>(persp, ksq), KingCtx::ZERO);
                assert_eq!(king_ctx::<None1>(persp, ksq), KingCtx::ZERO);
            }
        }
        for persp in [Color::White, Color::Black] {
            for color in [Color::White, Color::Black] {
                for kind in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King] {
                    for sq in 0..64u8 {
                        let side = if color == persp { 0 } else { 384 };
                        let s = if persp == Color::White { sq } else { sq ^ 56 };
                        let legacy = side + kind as usize * 64 + s as usize;
                        assert_eq!(feature(persp, Piece { color, kind }, sq, KingCtx::ZERO), legacy);
                    }
                }
            }
        }
        // The embedded net still loads.
        assert!(Network::from_bytes(include_bytes!("../net/nnue.bin")).is_ok());
    }

    /// Hand-checked indices; `tools/nnue-trainer` has the same cases.
    #[test]
    fn bucketed_feature_indices() {
        let wp = Piece { color: Color::White, kind: PieceType::Pawn };
        let bp = Piece { color: Color::Black, kind: PieceType::Pawn };
        // White king g1 (6): kb8m, file g mirrors -> b1, bucket 1, flip 7.
        let kw = king_ctx::<Kb8m>(Color::White, 6);
        assert_eq!(kw, KingCtx { base: 768, flip: 7 });
        // a2 pawn (8) -> h2 (15).
        assert_eq!(feature(Color::White, wp, 8, kw), 768 + 15);
        // Black king e8 (60): oriented e1 (4), file e mirrors -> d1, bucket 3.
        let kb = king_ctx::<Kb8m>(Color::Black, 60);
        assert_eq!(kb, KingCtx { base: 3 * 768, flip: 7 });
        // Own h7 pawn (55): oriented h2 (15), mirrored a2 (8).
        assert_eq!(feature(Color::Black, bp, 55, kb), 3 * 768 + 8);
        // Enemy (white) a2 pawn from Black: oriented a7 (48) -> h7 (55).
        assert_eq!(feature(Color::Black, wp, 8, kb), 3 * 768 + 384 + 55);
        // Unmirrored kb4: white king e1 -> bucket 1, no flip; black king e8 -> e1 -> 1.
        assert_eq!(king_ctx::<Kb4>(Color::White, 4), KingCtx { base: 768, flip: 0 });
        assert_eq!(king_ctx::<Kb4>(Color::Black, 60), KingCtx { base: 768, flip: 0 });
        // Black king on d7 (51): oriented d2 (11) -> bucket 2.
        assert_eq!(king_ctx::<Kb4>(Color::Black, 51), KingCtx { base: 2 * 768, flip: 0 });
        // m1: one bucket, only the flip.
        assert_eq!(king_ctx::<M1>(Color::White, 3), KingCtx::ZERO);
        assert_eq!(king_ctx::<M1>(Color::White, 4), KingCtx { base: 0, flip: 7 });
    }

    /// Incremental updates (including finny-table refreshes on king moves
    /// across buckets and the mirror line) match a from-scratch refresh
    /// over random games, for every layout.
    fn incremental_matches_refresh_for<L: Layout>(seed: u64) {
        let net = Arc::new(Network::random_kb(seed, 64, 100, L::KB));
        let mut next = rng(seed * 31 + 5);
        let mut legal = Vec::new();
        let (mut king_moves, mut refreshes, mut flips) = (0, 0, 0);
        let fens = [
            FENS[0],
            FENS[2],
            FENS[3],
            "8/8/3k4/8/8/4K3/8/8 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
        ];
        for game in 0..80 {
            let mut board = Board::new();
            board.set_fen(fens[game % fens.len()]);
            let mut st = NnueState::<L>::new(net.clone(), &board);
            let mut history = Vec::new();
            for _ in 0..120 {
                board.generate_legal_moves(false, &mut legal);
                if legal.is_empty() {
                    break;
                }
                // Prefer king moves half the time so buckets change often.
                let kings: Vec<Move> = legal
                    .iter()
                    .copied()
                    .filter(|m| board.squares[m.from as usize].unwrap().kind == PieceType::King)
                    .collect();
                let mv = if !kings.is_empty() && next() % 2 == 0 {
                    kings[(next() as usize) % kings.len()]
                } else {
                    legal[(next() as usize) % legal.len()]
                };
                let mover = board.side_to_move;
                if board.squares[mv.from as usize].unwrap().kind == PieceType::King {
                    king_moves += 1;
                    let (a, b) = (king_ctx::<L>(mover, mv.from), king_ctx::<L>(mover, mv.to));
                    refreshes += (a != b) as usize;
                    flips += (a.flip != b.flip) as usize;
                }
                st.push_move(&board, mv);
                let undo = board.make_move(mv);
                history.push((mv, undo));
                // Leave some plies pending (multi-ply materialisation), and
                // evaluate before reading the accumulator on others (the
                // fused update + output path, refreshes included).
                match next() % 3 {
                    0 => continue,
                    1 => {
                        let full = Accumulator::refresh_with::<L>(&net, &board);
                        let (stm, n) = (board.side_to_move, board.bb.occ_all().count_ones());
                        assert_eq!(st.evaluate(stm, n), evaluate_acc(&net, &full, stm, n), "after {}", move_to_uci(mv));
                        assert!(st.current().v == full.v, "after {}", move_to_uci(mv));
                    }
                    _ => {
                        let full = Accumulator::refresh_with::<L>(&net, &board);
                        assert!(st.current().v == full.v, "after {}", move_to_uci(mv));
                    }
                }
            }
            while let Some((mv, undo)) = history.pop() {
                board.undo_move(mv, undo);
                st.pop();
                assert!(st.current().v == Accumulator::refresh_with::<L>(&net, &board).v);
            }
        }
        assert!(king_moves > 1000, "{king_moves}");
        if L::REFRESH {
            assert!(refreshes > 200, "{refreshes}");
        }
        if L::L.mirror {
            assert!(flips > 100, "{flips}");
        }
    }

    #[test]
    fn incremental_matches_refresh_all_layouts() {
        incremental_matches_refresh_for::<Active>(3);
        incremental_matches_refresh_for::<None1>(4);
        incremental_matches_refresh_for::<M1>(5);
        incremental_matches_refresh_for::<Kb4>(6);
        incremental_matches_refresh_for::<Kb4m>(7);
        incremental_matches_refresh_for::<Kb8m>(8);
    }

    #[test]
    fn bucketed_net_roundtrip_and_size() {
        let net = Network::random_kb(11, 64, 100, 8);
        let bytes = net.to_bytes();
        assert_eq!(bytes.len(), net_bytes(8));
        assert_eq!(bytes.len(), (8 * 768 * H + H + BUCKETS * (2 * H + 1)) * 2);
        assert!(Network::from_bytes_kb(&bytes, 8).is_ok());
        assert!(Network::from_bytes_kb(&bytes, 1).is_err());
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
                // Fused update + output sum (output weights |w| <= 127).
                let w = rand_vec(127);
                let mut o3 = [0i16; H];
                let d = simd::update_dot(&mut o3, &src, &adds, &subs, &w);
                assert_eq!(o3, o2);
                assert_eq!(d, scalar::screlu_dot_wide(&o2, &w));
                // Dual-output update, in place on the first output.
                let (mut o4, mut o5) = (src, [0i16; H]);
                let p = o4.as_mut_ptr();
                // SAFETY: all three pointers cover H i16s.
                unsafe { simd::update_dual(p, o5.as_mut_ptr(), p, &adds, &subs) };
                assert!(o4 == o2 && o5 == o2);
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

    /// The AVX-512 kernels against scalar, called directly (so it holds even
    /// with `RUSTFISH_NO_AVX512` set); skipped on hosts without AVX-512.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar() {
        if !avx512::supported() {
            eprintln!("avx512_matches_scalar: host lacks AVX-512F/BW, skipped");
            return;
        }
        let mut next = rng(777);
        for iter in 0..4000 {
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
            let w = rand_vec(127);
            for (adds, subs) in [
                (vec![], vec![]),
                (vec![&a0], vec![&s0]),
                (vec![&a0], vec![&s0, &s1]),
                (vec![&a0, &a1], vec![&s0, &s1]),
                (vec![&a0, &a1, &s0, &s1], vec![&a1, &a0, &src, &s1]),
            ] {
                let mut o1 = [0i16; H];
                let mut o2 = [0i16; H];
                let mut o3 = [0i16; H];
                let (mut o4, mut o5) = (src, [0i16; H]);
                scalar::update(&mut o2, &src, &adds, &subs);
                // SAFETY: AVX-512F/BW checked above; all pointers cover H i16s.
                unsafe {
                    avx512::update(&mut o1, &src, &adds, &subs);
                    let d = avx512::update_dot(&mut o3, &src, &adds, &subs, &w);
                    assert_eq!(d, scalar::screlu_dot_wide(&o2, &w));
                    let p = o4.as_mut_ptr();
                    avx512::update_dual(p, o5.as_mut_ptr(), p, &adds, &subs);
                }
                assert!(o1 == o2 && o3 == o2 && o4 == o2 && o5 == o2);
            }
            let acc = rand_vec(if iter % 2 == 0 { 400 } else { 32767 });
            let ww = rand_vec(32767);
            // SAFETY: as above.
            unsafe {
                assert_eq!(avx512::screlu_dot(&acc, &w), scalar::screlu_dot(&acc, &w));
                assert_eq!(avx512::screlu_dot_wide(&acc, &w), scalar::screlu_dot_wide(&acc, &w));
                assert_eq!(avx512::screlu_dot_wide(&acc, &ww), scalar::screlu_dot_wide(&acc, &ww));
            }
        }
        for v in [1000i16, -1000, 255, 0] {
            let acc = [v; H];
            for w in [[127i16; H], [-127i16; H], [32767i16; H], [-32768i16; H]] {
                // SAFETY: as above.
                let wide = unsafe { avx512::screlu_dot_wide(&acc, &w) };
                assert_eq!(wide, scalar::screlu_dot_wide(&acc, &w));
                if (w[0] as i32).abs() <= 127 {
                    assert_eq!(unsafe { avx512::screlu_dot(&acc, &w) }, scalar::screlu_dot(&acc, &w));
                }
            }
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
        let mut st = NnueState::<Active>::new(net.clone(), &board);
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
