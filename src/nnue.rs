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

// ------------------------------------------------------------ threat inputs

/// Optional threat input features (`tools/nnue-trainer --threats <name>`).
/// Each mode adds `blocks()` blocks of 768 feature rows after the king
/// buckets' FT rows, indexed exactly like the base features
/// (`side*384 + kind*64 + (o(sq) ^ m)`, same orientation and mirror flip as
/// the perspective, but not king-bucketed). A row is active when the piece
/// on that square has the mode's property; kings never do.
///
/// - `EnPrise` (1 block): the piece is attacked by an enemy piece of lower
///   value (P < N = B < R < Q), or attacked and not defended by its own side.
///   This is the first-order static-exchange signal ("this piece can be
///   won"), at most 30 active rows per perspective.
/// - `AttDef` (2 blocks): block 0 = attacked by any enemy piece, block 1 =
///   defended by an own piece (both from plain attack sets, no x-rays).
///
/// The property of a piece does not depend on the perspective, so both
/// perspectives activate the same pieces (in their own orientation).
/// Threats change non-locally, but few of them per move (about 1 row per
/// ply for `EnPrise`, 3.5 for `AttDef`), so they ride in the normal lazy
/// accumulator stack: `push_move` saves the post-move piece bitboards, and
/// materialising a ply computes its threat bitboards and adds the changed
/// threat rows in the same fused update pass as the move's piece rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub(crate) enum ThreatMode {
    None,
    EnPrise,
    AttDef,
}

impl ThreatMode {
    pub(crate) const fn blocks(self) -> usize {
        match self {
            ThreatMode::None => 0,
            ThreatMode::EnPrise => 1,
            ThreatMode::AttDef => 2,
        }
    }
}

/// The threat inputs the engine runs. Switching it changes the net format
/// (`blocks()` x 768 extra FT rows after the king buckets' rows): embed a
/// net trained with the same `--threats`.
pub(crate) const THREATS: ThreatMode = ThreatMode::EnPrise;

/// Threat bitboards per block, `[block][colour][kind]`: the pieces whose
/// threat row of that block is active.
pub(crate) type ThreatBBs = [[[u64; 6]; 2]; 2];

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = FILE_A << 7;

/// Squares attacked by colour `c`'s pieces, per attacker kind.
#[inline(always)]
fn attacks_by_kind(bb: &PieceBBs, c: usize, occ: u64) -> [u64; 6] {
    use crate::bitboard::{bishop_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS};
    let p = &bb[c];
    let pawns = if c == 0 {
        ((p[0] & !FILE_A) << 7) | ((p[0] & !FILE_H) << 9)
    } else {
        ((p[0] & !FILE_A) >> 9) | ((p[0] & !FILE_H) >> 7)
    };
    let mut a = [pawns, 0, 0, 0, 0, 0];
    let mut x = p[1];
    while x != 0 {
        a[1] |= KNIGHT_ATTACKS[x.trailing_zeros() as usize];
        x &= x - 1;
    }
    let mut x = p[2];
    while x != 0 {
        a[2] |= bishop_attacks(x.trailing_zeros() as usize, occ);
        x &= x - 1;
    }
    let mut x = p[3];
    while x != 0 {
        a[3] |= rook_attacks(x.trailing_zeros() as usize, occ);
        x &= x - 1;
    }
    let mut x = p[4];
    while x != 0 {
        let s = x.trailing_zeros() as usize;
        a[4] |= bishop_attacks(s, occ) | rook_attacks(s, occ);
        x &= x - 1;
    }
    let mut x = p[5];
    while x != 0 {
        a[5] |= KING_ATTACKS[x.trailing_zeros() as usize];
        x &= x - 1;
    }
    a
}

/// The threat bitboards of a position (only the first `mode.blocks()` are
/// meaningful). The same definition as `threat_masks` in the trainers.
#[inline]
pub(crate) fn threat_bbs(bb: &PieceBBs, mode: ThreatMode) -> ThreatBBs {
    let mut t = [[[0u64; 6]; 2]; 2];
    if mode == ThreatMode::None {
        return t;
    }
    let mut occ = 0;
    for c in 0..2 {
        for k in 0..6 {
            occ |= bb[c][k];
        }
    }
    let att = [attacks_by_kind(bb, 0, occ), attacks_by_kind(bb, 1, occ)];
    let all = |a: &[u64; 6]| a[0] | a[1] | a[2] | a[3] | a[4] | a[5];
    let any = [all(&att[0]), all(&att[1])];
    for c in 0..2 {
        let (e, by, def) = (&att[1 - c], any[1 - c], any[c]);
        match mode {
            ThreatMode::EnPrise => {
                let hanging = by & !def;
                let minor = e[0];
                let rook = minor | e[1] | e[2];
                let queen = rook | e[3];
                let lower = [0, minor, minor, rook, queen];
                for k in 0..5 {
                    t[0][c][k] = bb[c][k] & (lower[k] | hanging);
                }
            }
            ThreatMode::AttDef => {
                for k in 0..5 {
                    t[0][c][k] = bb[c][k] & by;
                    t[1][c][k] = bb[c][k] & def;
                }
            }
            ThreatMode::None => {}
        }
    }
    t
}

/// A king layout as a type, so the accumulator code is monomorphised per
/// layout: the active one compiles to exactly the old code when it has one
/// unmirrored bucket, and tests can drive other layouts side by side. The
/// threat mode rides along (`TH`, off unless a type sets it).
pub(crate) trait Layout: Clone {
    const L: KingLayout;
    const KB: usize = layout_buckets(&Self::L);
    /// Whether a king move can change its perspective's feature block.
    const REFRESH: bool = Self::KB > 1 || Self::L.mirror;
    const TH: ThreatMode = ThreatMode::None;
    /// Threat blocks of 768 rows after the `KB * 768` king-bucket rows.
    const TB: usize = Self::TH.blocks();
}

#[derive(Clone, Copy)]
pub(crate) struct Active;
impl Layout for Active {
    const L: KingLayout = KING_LAYOUT;
    const TH: ThreatMode = THREATS;
}

/// Input buckets of the active layout.
pub(crate) const KB: usize = layout_buckets(&KING_LAYOUT);
/// Threat blocks of the active net.
pub(crate) const TB: usize = THREATS.blocks();

/// Bytes in a net file with `blocks` blocks of 768 FT rows (king buckets
/// plus threat blocks): (blocks*768*H + H + BUCKETS*2H + BUCKETS) little-endian i16.
pub(crate) const fn net_bytes(blocks: usize) -> usize {
    (blocks * INPUTS * H + H + BUCKETS * (2 * H + 1)) * 2
}

/// Bytes in a net file for the active layout.
#[allow(dead_code)]
pub(crate) const NET_BYTES: usize = net_bytes(KB + TB);

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

#[derive(Clone)]
pub(crate) struct Network {
    /// Feature-major: weights of feature f are `ft_w[f*H .. (f+1)*H]`,
    /// `kb * 768` features (`blocks = kb + threat blocks`, threat rows last).
    ft_w: Vec<i16>,
    kb: usize,
    ft_b: [i16; H],
    /// Bucket-major: bucket b's weights are `out_w[b*2H .. (b+1)*2H]`, stm half first.
    out_w: Vec<i16>,
    out_b: [i16; BUCKETS],
    /// All |out_w| <= 127: the SIMD output kernel's i16 products and i32
    /// per-perspective sums cannot overflow (256 * 255^2 * 127 < 2^31).
    small_out: bool,
}

impl Network {
    /// A net for the active king layout (and threat inputs).
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::from_bytes_kb(bytes, KB + TB)
    }

    pub(crate) fn from_bytes_kb(bytes: &[u8], kb: usize) -> Result<Self, String> {
        if bytes.len() != net_bytes(kb) {
            return Err(format!(
                "net is {} bytes, expected {} (H={H}, {kb} block(s) of 768 FT rows = {KB} king bucket(s) + {TB} threat block(s), {BUCKETS} output bucket(s))",
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
        Self { ft_w, kb, ft_b, out_w, out_b, small_out }
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
        Self::random_kb(seed, ft_range, out_range, KB + TB)
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

    /// From scratch: piece rows, and threat rows when `L::TB > 0`.
    pub(crate) fn refresh_with<L: Layout>(net: &Network, board: &Board) -> Self {
        debug_assert_eq!(net.kb, L::KB + L::TB);
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
        if L::TB > 0 {
            let tb = threat_bbs(&board.bb.pieces, L::TH);
            for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
                add_threat_rows::<L>(net, &mut acc.v[pi], persp, ctx[pi].flip, &tb);
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
        let src = e.acc;
        update(&mut e.acc, &src, &ar[..ka], &sr[..ks]);
        i += ka;
        j += ks;
    }
    e.bb = *bb;
    *out = e.acc;
}

/// `out = src + Σ rows(adds) - Σ rows(subs)`: up to two adds and two subs
/// per pass through the SIMD `update` with fixed-length row lists (dynamic
/// lists compile to a much slower inner loop); further passes run in place.
fn update_rows(net: &Network, out: &mut [i16; H], src: &[i16; H], adds: &[u16], subs: &[u16]) {
    let (mut i, mut j) = (0, 0);
    let mut first = true;
    loop {
        let (ka, ks) = ((adds.len() - i).min(2), (subs.len() - j).min(2));
        if !first && ka == 0 && ks == 0 {
            return;
        }
        let a = |k: usize| net.row(adds[i + k] as usize);
        let s = |k: usize| net.row(subs[j + k] as usize);
        let tmp;
        let from: &[i16; H] = if first {
            src
        } else {
            tmp = *out;
            &tmp
        };
        match (ka, ks) {
            (0, 0) => *out = *from,
            (1, 0) => update(out, from, &[a(0)], &[]),
            (0, 1) => update(out, from, &[], &[s(0)]),
            (1, 1) => update(out, from, &[a(0)], &[s(0)]),
            (2, 0) => update(out, from, &[a(0), a(1)], &[]),
            (0, 2) => update(out, from, &[], &[s(0), s(1)]),
            (2, 1) => update(out, from, &[a(0), a(1)], &[s(0)]),
            (1, 2) => update(out, from, &[a(0)], &[s(0), s(1)]),
            _ => update(out, from, &[a(0), a(1)], &[s(0), s(1)]),
        }
        first = false;
        i += ka;
        j += ks;
    }
}

/// Threat feature of block `t` for piece (colour index `c`, kind) on `sq`,
/// perspective `persp` with square xor `flip`: after the king-bucket rows,
/// not bucketed.
#[inline(always)]
fn threat_feature<L: Layout>(persp: Color, t: usize, c: usize, kind: usize, sq: u8, flip: u8) -> usize {
    (L::KB + t) * INPUTS + feature_ck(persp, c, kind, sq, KingCtx { base: 0, flip })
}

/// Indices of the threat rows of perspective `persp` (square xor `flip`)
/// that are set in `new` but not in `old` (into `adds`) and the reverse
/// (into `subs`); returns the counts. Kings have no threat rows.
#[inline(always)]
fn threat_diff<L: Layout>(persp: Color, flip: u8, old: &ThreatBBs, new: &ThreatBBs, adds: &mut [u16], subs: &mut [u16]) -> (usize, usize) {
    let (mut na, mut ns) = (0, 0);
    for t in 0..L::TB {
        for c in 0..2 {
            for k in 0..5 {
                let (n, o) = (new[t][c][k], old[t][c][k]);
                if n == o {
                    continue;
                }
                let (mut a, mut r) = (n & !o, o & !n);
                while a != 0 && na < adds.len() {
                    adds[na] = threat_feature::<L>(persp, t, c, k, a.trailing_zeros() as u8, flip) as u16;
                    na += 1;
                    a &= a - 1;
                }
                while r != 0 && ns < subs.len() {
                    subs[ns] = threat_feature::<L>(persp, t, c, k, r.trailing_zeros() as u8, flip) as u16;
                    ns += 1;
                    r &= r - 1;
                }
                debug_assert!(a == 0 && r == 0, "too many threat rows");
            }
        }
    }
    (na, ns)
}

/// `acc += ` every threat row of `tb` for perspective `persp`.
fn add_threat_rows<L: Layout>(net: &Network, acc: &mut [i16; H], persp: Color, flip: u8, tb: &ThreatBBs) {
    let mut adds = [0u16; 64];
    let (na, _) = threat_diff::<L>(persp, flip, &[[[0; 6]; 2]; 2], tb, &mut adds, &mut []);
    let src = *acc;
    update_rows(net, acc, &src, &adds[..na], &[]);
}

/// Threat accumulators of `board` from scratch (tests).
#[cfg(test)]
fn threat_scratch<L: Layout>(net: &Network, board: &Board) -> [[i16; H]; 2] {
    let bbs = threat_bbs(&board.bb.pieces, L::TH);
    let mut out = [[0i16; H]; 2];
    for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
        let flip = board_ctx::<L>(board, persp).flip;
        for t in 0..L::TB {
            for c in 0..2 {
                for k in 0..6 {
                    let mut x = bbs[t][c][k];
                    while x != 0 {
                        let row = net.row(threat_feature::<L>(persp, t, c, k, x.trailing_zeros() as u8, flip));
                        for (a, w) in out[pi].iter_mut().zip(row.iter()) {
                            *a = a.wrapping_add(*w);
                        }
                        x &= x - 1;
                    }
                }
            }
        }
    }
    out
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
    /// Threat bitboards of `stack[..=computed]`'s positions (`L::TB > 0`
    /// only, else empty; see `ThreatMode`).
    tbbs: RefCell<Vec<ThreatBBs>>,
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
        assert_eq!(net.kb, L::KB + L::TB, "net has {} FT blocks, layout needs {} + {} threat", net.kb, L::KB, L::TB);
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
            tbbs: RefCell::new(if L::TB > 0 { Vec::with_capacity(256) } else { Vec::new() }),
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
        if L::TB > 0 {
            let t = self.tbbs.get_mut();
            t.clear();
            t.push(threat_bbs(&board.bb.pieces, L::TH));
        }
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
        if L::TB > 0 {
            // Materialising this ply needs its piece bitboards (threats).
            stack[self.top].bb = bb_after(board, &delta);
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
                    if L::TB == 0 {
                        slot.bb = bb_after(board, &delta);
                    }
                }
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
        let mut k = self.computed.get();
        if k == self.top {
            return;
        }
        let mut stack = self.stack.borrow_mut();
        let mut tbbs = self.tbbs.borrow_mut();
        let net = &*self.net;
        while k < self.top {
            let (lo, hi) = stack.split_at_mut(k + 1);
            if L::TB > 0 {
                // tbbs[k] is ply k's; compute ply k + 1's.
                tbbs.truncate(k + 1);
                let next = threat_bbs(&hi[0].bb, L::TH);
                apply_delta::<L>(net, &lo[k].acc, &mut hi[0], &self.cache, (&tbbs[k], &next));
                tbbs.push(next);
            } else {
                apply_delta::<L>(net, &lo[k].acc, &mut hi[0], &self.cache, (&[[[0; 6]; 2]; 2], &[[[0; 6]; 2]; 2]));
            }
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

/// `dst.acc = src + dst.delta`, both perspectives in one copy-and-update
/// pass; a perspective marked for refresh is rebuilt from the finny table.
/// With threat inputs, `thr` = (threat bitboards of `src`'s position, of
/// `dst`'s): the changed threat rows join the same pass (a refreshed
/// perspective gets all of `dst`'s threat rows after the finny refresh).
#[inline(always)]
fn apply_delta<L: Layout>(net: &Network, src: &Accumulator, dst: &mut Slot, cache: &RefCell<Vec<CacheEntry>>, thr: (&ThreatBBs, &ThreatBBs)) {
    let d = dst.delta;
    for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
        let k = if L::REFRESH { dst.ctx[pi] } else { KingCtx::ZERO };
        if L::REFRESH && dst.refresh[pi] {
            let mut c = cache.borrow_mut();
            let bb = dst.bb;
            refresh_cached(net, &mut c[pi * 2 * L::KB + k.slot()], persp, k, &bb, &mut dst.acc.v[pi]);
            if L::TB > 0 {
                add_threat_rows::<L>(net, &mut dst.acc.v[pi], persp, k.flip, thr.1);
            }
            continue;
        }
        // Threat rows unchanged (about half the plies with `EnPrise`): the
        // plain piece-row pass below.
        if L::TB > 0 && thr.0[..L::TB] != thr.1[..L::TB] {
            // Piece rows (at most 2 + 2), then the changed threat rows.
            let mut adds = [0u16; 64 + 2];
            let mut subs = [0u16; 64 + 2];
            for x in 0..d.n_add {
                adds[x] = feature(persp, d.add[x].1.unwrap(), d.add[x].0, k) as u16;
            }
            for x in 0..d.n_sub {
                subs[x] = feature(persp, d.sub[x].1.unwrap(), d.sub[x].0, k) as u16;
            }
            let (ta, ts) = threat_diff::<L>(persp, k.flip, thr.0, thr.1, &mut adds[d.n_add..], &mut subs[d.n_sub..]);
            let (na, ns) = (d.n_add + ta, d.n_sub + ts);
            update_rows(net, &mut dst.acc.v[pi], &src.v[pi], &adds[..na], &subs[..ns]);
            continue;
        }
        let f = |e: (u8, Option<Piece>)| net.row(feature(persp, e.1.unwrap(), e.0, k));
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
        ($name:ident, $l:expr, $th:expr) => {
            #[derive(Clone, Copy)]
            struct $name;
            impl Layout for $name {
                const L: KingLayout = $l;
                const TH: ThreatMode = $th;
            }
        };
    }
    layout!(None1, LAYOUT_NONE);
    layout!(M1, LAYOUT_M1);
    layout!(Kb4, LAYOUT_KB4);
    layout!(Kb4m, LAYOUT_KB4M);
    layout!(Kb8m, LAYOUT_KB8M);
    layout!(None1Ep, LAYOUT_NONE, ThreatMode::EnPrise);
    layout!(None1Ad, LAYOUT_NONE, ThreatMode::AttDef);
    layout!(M1Ep, LAYOUT_M1, ThreatMode::EnPrise);
    layout!(Kb8mAd, LAYOUT_KB8M, ThreatMode::AttDef);
    layout!(Kb4Ep, LAYOUT_KB4, ThreatMode::EnPrise);

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
        if KING_LAYOUT != LAYOUT_NONE || THREATS != ThreatMode::None {
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
        let net = Arc::new(Network::random_kb(seed, 64, 100, L::KB + L::TB));
        let mut next = rng(seed * 31 + 5);
        let mut legal = Vec::new();
        let (mut king_moves, mut refreshes, mut flips) = (0, 0, 0);
        // Piece + threat rows of `board`, independently of `refresh_with`.
        let scratch = |board: &Board| {
            let ctx = [board_ctx::<L>(board, Color::White), board_ctx::<L>(board, Color::Black)];
            let mut a = Accumulator { v: [net.ft_b; 2] };
            let t = threat_scratch::<L>(&net, board);
            for (pi, persp) in [Color::White, Color::Black].into_iter().enumerate() {
                for sq in 0..64u8 {
                    if let Some(p) = board.squares[sq as usize] {
                        for (x, w) in a.v[pi].iter_mut().zip(net.row(feature(persp, p, sq, ctx[pi])).iter()) {
                            *x = x.wrapping_add(*w);
                        }
                    }
                }
                for (x, y) in a.v[pi].iter_mut().zip(t[pi].iter()) {
                    *x = x.wrapping_add(*y);
                }
            }
            a
        };
        let mut threat_rows = 0usize;
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
                // With threats, materialise only sometimes, so several
                // plies (and their threat diffs) are applied in one go.
                if L::TB == 0 || next() % 3 == 0 {
                    let full = Accumulator::refresh_with::<L>(&net, &board);
                    assert!(st.current().v == full.v, "after {}", move_to_uci(mv));
                }
                if L::TB > 0 && next() % 3 == 0 {
                    let want = scratch(&board);
                    assert!(st.current().v == want.v, "threats after {}", move_to_uci(mv));
                    assert!(Accumulator::refresh_with::<L>(&net, &board).v == want.v);
                    let n = board.bb.occ_all().count_ones();
                    assert_eq!(st.evaluate(board.side_to_move, n), evaluate_acc(&net, &want, board.side_to_move, n));
                    let tb = threat_bbs(&board.bb.pieces, L::TH);
                    threat_rows += tb[..L::TB].iter().flatten().flatten().map(|x| x.count_ones() as usize).sum::<usize>();
                }
            }
            while let Some((mv, undo)) = history.pop() {
                board.undo_move(mv, undo);
                st.pop();
                if L::TB == 0 || next() % 2 == 0 {
                    assert!(st.current().v == Accumulator::refresh_with::<L>(&net, &board).v);
                }
                if L::TB > 0 && next() % 4 == 0 {
                    assert!(st.current().v == scratch(&board).v);
                }
            }
        }
        if L::TB > 0 {
            assert!(threat_rows > 2_000, "{threat_rows}");
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

    /// Threat accumulators (diffed per eval, rebuilt on mirror flips) match
    /// a from-scratch build, with and without king buckets and mirroring.
    #[test]
    fn threat_inputs_incremental_matches_scratch() {
        incremental_matches_refresh_for::<None1Ep>(21);
        incremental_matches_refresh_for::<None1Ad>(22);
        incremental_matches_refresh_for::<M1Ep>(23);
        incremental_matches_refresh_for::<Kb8mAd>(24);
        incremental_matches_refresh_for::<Kb4Ep>(25);
    }

    /// Hand-checked threat bitboards; `tools/nnue-trainer` has the same cases.
    #[test]
    fn threat_bbs_hand_checked() {
        let bit = |s: u32| 1u64 << s;
        let mut b = Board::new();
        // W Ke1 Rh1 Ne4 Pc4, B Ke8 Pd5 Bc6 Rf6.
        b.set_fen("4k3/8/2b2r2/3p4/2P1N3/8/8/4K2R w - - 0 1");
        let (c4, e4, f6, d5, c6) = (26, 28, 45, 35, 42);
        let ep = threat_bbs(&b.bb.pieces, ThreatMode::EnPrise);
        // White: c4 pawn attacked by d5 and undefended; e4 knight attacked by a pawn.
        assert_eq!(ep[0][0], [bit(c4), bit(e4), 0, 0, 0, 0]);
        // Black: f6 rook attacked by a knight; d5 pawn attacked by a pawn but defended by c6.
        assert_eq!(ep[0][1], [0, 0, 0, bit(f6), 0, 0]);
        assert_eq!(ep[1], [[0; 6]; 2]);
        let ad = threat_bbs(&b.bb.pieces, ThreatMode::AttDef);
        assert_eq!(ad[0][0], [bit(c4), bit(e4), 0, 0, 0, 0]);
        assert_eq!(ad[0][1], [bit(d5), 0, 0, bit(f6), 0, 0]);
        // Defended: no White piece; Black d5 (by c6) and c6 (by f6).
        assert_eq!(ad[1][0], [0; 6]);
        assert_eq!(ad[1][1], [bit(d5), 0, bit(c6), 0, 0, 0]);
        assert_eq!(threat_bbs(&b.bb.pieces, ThreatMode::None), [[[0; 6]; 2]; 2]);
        // Feature indices (as the trainer's `threat_masks_hand_checked`):
        // block 0 follows the king buckets; same orientation and flip as pieces.
        let (w, bl) = (Color::White, Color::Black);
        assert_eq!(threat_feature::<None1Ep>(w, 0, 0, 0, c4 as u8, 0), 768 + 26);
        assert_eq!(threat_feature::<None1Ep>(w, 0, 1, 3, f6 as u8, 0), 768 + 384 + 192 + 45);
        assert_eq!(threat_feature::<None1Ep>(bl, 0, 1, 3, f6 as u8, 0), 768 + 192 + 21);
        assert_eq!(threat_feature::<None1Ep>(bl, 0, 0, 1, e4 as u8, 0), 768 + 384 + 64 + 36);
        // kb8m (kings e1 / e8 -> flip 7): rows 8*768 + block*768 + ...
        assert_eq!(threat_feature::<Kb8mAd>(w, 0, 1, 3, f6 as u8, 7), 8 * 768 + 618);
        assert_eq!(threat_feature::<Kb8mAd>(w, 1, 1, 2, c6 as u8, 7), 9 * 768 + 557);
        assert_eq!(threat_feature::<Kb8mAd>(bl, 1, 1, 0, d5 as u8, 7), 9 * 768 + 28);
        // W Kg1 Pc3 Nd4 Qh5 Ra1, B Ra8 Rd8 Ke8 Bb6 Rh8.
        b.set_fen("r2rk2r/8/1b6/7Q/3N4/2P5/8/R5K1 w - - 0 1");
        let ep = threat_bbs(&b.bb.pieces, ThreatMode::EnPrise);
        // Nd4: attackers Bb6 (equal) and Rd8 (higher), defended by c3 -> no.
        // Qh5: attacked by Rh8 (lower) -> yes. Ra1: attacked by Ra8 (equal),
        // undefended -> yes. Kings never.
        assert_eq!(ep[0][0], [0, 0, 0, bit(0), bit(39), 0]);
        // Ra8: attacked by Ra1 (equal) but defended by Rd8 -> no. Rh8:
        // attacked by the queen (higher), undefended -> yes.
        assert_eq!(ep[0][1], [0, 0, 0, bit(63), 0, 0]);
        let ad = threat_bbs(&b.bb.pieces, ThreatMode::AttDef);
        assert_eq!(ad[0][0], [0, bit(27), 0, bit(0), bit(39), 0]);
        assert_eq!(ad[0][1], [0, 0, 0, bit(56) | bit(63), 0, 0]);
        assert_eq!(ad[1][0], [0, bit(27), 0, 0, 0, 0]);
        assert_eq!(ad[1][1], [0, 0, 0, bit(56) | bit(59), 0, 0]);
    }

    /// `threat_bbs` against a naive definition (attacker lists from ray
    /// walks over the mailbox) over random game positions.
    #[test]
    fn threat_bbs_match_naive() {
        // Kinds of colour `by`'s pieces attacking `sq` (no x-rays).
        fn attackers(board: &Board, sq: i32, by: Color) -> Vec<usize> {
            let mut out = Vec::new();
            for from in 0..64i32 {
                let Some(p) = board.squares[from as usize] else { continue };
                if p.color != by || from == sq {
                    continue;
                }
                let (df, dr) = (sq % 8 - from % 8, sq / 8 - from / 8);
                let hit = match p.kind {
                    PieceType::Pawn => df.abs() == 1 && dr == if by == Color::White { 1 } else { -1 },
                    PieceType::Knight => (df.abs(), dr.abs()) == (1, 2) || (df.abs(), dr.abs()) == (2, 1),
                    PieceType::King => df.abs() <= 1 && dr.abs() <= 1,
                    _ => {
                        let diag = df.abs() == dr.abs();
                        let ortho = df == 0 || dr == 0;
                        let ok = match p.kind {
                            PieceType::Bishop => diag,
                            PieceType::Rook => ortho,
                            _ => diag || ortho,
                        };
                        ok && {
                            let step = df.signum() + 8 * dr.signum();
                            let mut s = from + step;
                            let mut clear = true;
                            while s != sq {
                                clear &= board.squares[s as usize].is_none();
                                s += step;
                            }
                            clear
                        }
                    }
                };
                if hit {
                    out.push(p.kind as usize);
                }
            }
            out
        }
        let value = [1, 3, 3, 5, 9, 100];
        let mut next = rng(77);
        let mut legal = Vec::new();
        let mut set = [0usize; 3];
        for game in 0..40 {
            let mut board = Board::new();
            board.set_fen(FENS[game % FENS.len()]);
            for _ in 0..80 {
                board.generate_legal_moves(false, &mut legal);
                if legal.is_empty() {
                    break;
                }
                board.make_move(legal[(next() as usize) % legal.len()]);
                let ep = threat_bbs(&board.bb.pieces, ThreatMode::EnPrise);
                let ad = threat_bbs(&board.bb.pieces, ThreatMode::AttDef);
                for sq in 0..64 {
                    let Some(p) = board.squares[sq as usize] else { continue };
                    let (c, k) = (p.color as usize, p.kind as usize);
                    let by = attackers(&board, sq, p.color.opposite());
                    let def = attackers(&board, sq, p.color);
                    let king = p.kind == PieceType::King;
                    let att = !king && !by.is_empty();
                    let dfd = !king && !def.is_empty();
                    let lower = by.iter().any(|&a| value[a] < value[k]);
                    let want_ep = !king && (lower || (att && !dfd));
                    let got = |t: &ThreatBBs, b: usize| t[b][c][k] >> sq & 1 == 1;
                    assert_eq!(got(&ep, 0), want_ep, "en prise sq {sq} game {game}");
                    assert_eq!(got(&ad, 0), att, "attacked sq {sq} game {game}");
                    assert_eq!(got(&ad, 1), dfd, "defended sq {sq} game {game}");
                    set[0] += want_ep as usize;
                    set[1] += att as usize;
                    set[2] += dfd as usize;
                }
            }
        }
        assert!(set.iter().all(|&n| n > 1000), "{set:?}");
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

    /// End-to-end check against the trainers: evals of the FENs in
    /// `$THREAT_FENS` (one per line) with the net `$THREAT_NET` trained with
    /// `--threats $THREAT_MODE` (ep|ad, no king buckets), printed as
    /// `|cp|<TAB>fen` for comparison with `nnue-trainer eval` on the same
    /// positions. `cargo test --release threat_net_eval_dump -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn threat_net_eval_dump() {
        fn dump<L: Layout>(bytes: &[u8], fens: &str) {
            let net = Arc::new(Network::from_bytes_kb(bytes, L::KB + L::TB).unwrap());
            for fen in fens.lines().filter(|l| !l.trim().is_empty()) {
                let mut b = Board::new();
                b.set_fen(fen);
                let st = NnueState::<L>::new(net.clone(), &b);
                println!("{}\t{fen}", st.evaluate(b.side_to_move, b.bb.occ_all().count_ones()).abs());
            }
        }
        let net = std::fs::read(std::env::var("THREAT_NET").unwrap()).unwrap();
        let fens = std::fs::read_to_string(std::env::var("THREAT_FENS").unwrap()).unwrap();
        match std::env::var("THREAT_MODE").unwrap().as_str() {
            "ep" => dump::<None1Ep>(&net, &fens),
            "ad" => dump::<None1Ad>(&net, &fens),
            m => panic!("unknown THREAT_MODE {m}"),
        }
    }

    /// Threat-input cost: `cargo test --release threat_throughput -- --ignored --nocapture`.
    /// Walks random games, evaluating each position (threat diff of one ply).
    #[test]
    #[ignore]
    fn threat_throughput() {
        use std::time::Instant;
        fn run<L: Layout>(name: &str) {
            let net = Arc::new(Network::random_kb(42, 64, 100, L::KB + L::TB));
            let mut next = rng(5);
            let mut legal = Vec::new();
            // A fixed list of positions along random games.
            let mut boards = Vec::new();
            for g in 0..20 {
                let mut b = Board::new();
                b.set_fen(FENS[g % FENS.len()]);
                for _ in 0..60 {
                    b.generate_legal_moves(false, &mut legal);
                    if legal.is_empty() {
                        break;
                    }
                    b.make_move(legal[(next() as usize) % legal.len()]);
                    boards.push(b.clone());
                }
            }
            let reps = 2000;
            let t = Instant::now();
            let mut x = 0u64;
            for _ in 0..reps {
                for b in &boards {
                    let bb = threat_bbs(&std::hint::black_box(b).bb.pieces, L::TH);
                    x = x.wrapping_add(bb[0][0][1] ^ bb[1][1][3]);
                }
            }
            let n = (reps * boards.len()) as f64;
            let masks = t.elapsed().as_secs_f64() * 1e9 / n;
            // One ply: push a move and materialise, vs the same layout's
            // piece rows only (L::TB = 0 twin) for the plain cost.
            let mut st = NnueState::<L>::new(net.clone(), &boards[0]);
            let mut b0 = boards[0].clone();
            b0.generate_legal_moves(false, &mut legal);
            let t = Instant::now();
            for i in 0..reps * boards.len() {
                let mv = legal[i % legal.len()];
                st.push_move(&b0, mv);
                st.materialise();
                st.pop();
            }
            let upd = t.elapsed().as_secs_f64() * 1e9 / n;
            x = x.wrapping_add(st.current().v[0][1] as u64);
            // Rows changed per step (per perspective) and active rows.
            let (mut changed, mut active) = (0usize, 0usize);
            for w in boards.windows(2) {
                let (a, b) = (threat_bbs(&w[0].bb.pieces, L::TH), threat_bbs(&w[1].bb.pieces, L::TH));
                for t in 0..L::TB {
                    for c in 0..2 {
                        for k in 0..6 {
                            changed += (a[t][c][k] ^ b[t][c][k]).count_ones() as usize;
                            active += b[t][c][k].count_ones() as usize;
                        }
                    }
                }
            }
            let m = (boards.len() - 1) as f64;
            println!(
                "{name}: threat_bbs {masks:.0} ns, push+materialise (one ply) {upd:.0} ns; {:.1} rows changed, {:.1} active ({x})",
                changed as f64 / m,
                active as f64 / m
            );
        }
        run::<None1Ep>("ep");
        run::<None1Ad>("ad");
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
