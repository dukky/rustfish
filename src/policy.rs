//! Policy upper bounds, shape 2 cost side: a learned quiet-move scorer that
//! runs only at deep nodes (depth >= MIN_DEPTH), called once per node from
//! negamax_body after the picker is built (a cold, out-of-line call: no hook
//! in MovePicker::next or the move loop). Same dummy shape as gate B. Weights are random (fixed seed): this measures cost only. The
//! result reaches the move score as `black_box(s) & 0`, so the search tree
//! (bench nodes) is identical to main and the NPS difference is the cost.
//!
//! Shape: a per-node context vector (bias + one row per piece on the board,
//! 768 rows, side-to-move relative) and, per quiet move, a from-row and a
//! to-row indexed by the moving piece; hidden = clamp(ctx + from + to, 0,
//! 255), score = hidden . out. Parameters: (768 + 2 * 384 + 2) * H
//! (H = 64: ~98k; H = 128: ~197k).

use super::*;
use std::sync::OnceLock;

/// Hidden width; set per measurement branch.
pub(crate) const H: usize = 32;
/// Score only at nodes of at least this depth.
pub(crate) const MIN_DEPTH: i32 = 3;
/// Plumbing control: the hook is compiled in but never scores.
pub(crate) const NEVER: bool = false;

#[repr(C, align(64))]
pub(crate) struct PolicyNet {
    bias: [i16; H],
    ctx: [[i16; H]; 768],
    from: [[i16; H]; 384],
    to: [[i16; H]; 384],
    out: [i16; H],
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Ctx {
    v: [i16; H],
}

static NET: OnceLock<Box<PolicyNet>> = OnceLock::new();

fn build() -> Box<PolicyNet> {
    let mut seed = 0x5eed_1234_u64;
    let mut r = || (crate::tt::splitmix64(&mut seed) % 41) as i16 - 20;
    let mut n = Box::new(PolicyNet { bias: [0; H], ctx: [[0; H]; 768], from: [[0; H]; 384], to: [[0; H]; 384], out: [0; H] });
    for x in n.bias.iter_mut() {
        *x = r() + 40;
    }
    for row in n.ctx.iter_mut().chain(n.from.iter_mut()).chain(n.to.iter_mut()) {
        for x in row.iter_mut() {
            *x = r();
        }
    }
    for x in n.out.iter_mut() {
        *x = r();
    }
    n
}

#[inline(always)]
pub(crate) fn net() -> &'static PolicyNet {
    NET.get_or_init(build)
}

/// Context vector of the position, side-to-move relative.
#[inline]
pub(crate) fn context(board: &Board) -> Ctx {
    let n = net();
    let mut v = n.bias;
    let flip = if board.side_to_move == Color::White { 0 } else { 56 };
    let us = color_index(board.side_to_move);
    for c in 0..2 {
        let rel = (c != us) as usize;
        for pt in 0..6 {
            let mut bb = board.bb.pieces[c][pt];
            while bb != 0 {
                let sq = (bb.trailing_zeros() as usize) ^ flip;
                bb &= bb - 1;
                let row = &n.ctx[rel * 384 + pt * 64 + sq];
                for i in 0..H {
                    v[i] = v[i].wrapping_add(row[i]);
                }
            }
        }
    }
    Ctx { v }
}

/// Score of one quiet move (piece type index `pt`, squares from/to).
#[inline]
pub(crate) fn score(ctx: &Ctx, board: &Board, pt: usize, from: u8, to: u8) -> i32 {
    let n = net();
    let flip = if board.side_to_move == Color::White { 0 } else { 56 };
    let f = &n.from[pt * 64 + (from as usize ^ flip)];
    let t = &n.to[pt * 64 + (to as usize ^ flip)];
    let mut s = 0i32;
    for i in 0..H {
        let a = ctx.v[i].wrapping_add(f[i]).wrapping_add(t[i]).clamp(0, 255);
        s += a as i32 * n.out[i] as i32;
    }
    s
}

#[inline(always)]
pub(crate) fn piece_kind(board: &Board, sq: u8) -> usize {
    board.squares[sq as usize].map(|p| crate::bitboard::kind_index(p.kind)).unwrap_or(0)
}
