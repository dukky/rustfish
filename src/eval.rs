use super::*;
use crate::eval_params::{self as ep, P, *};

#[derive(Clone, Copy)]
struct SqList {
    data: [u8; 16],
    len: usize,
}

impl SqList {
    fn new() -> Self {
        Self {
            data: [0; 16],
            len: 0,
        }
    }

    fn push(&mut self, sq: u8) {
        if self.len < self.data.len() {
            self.data[self.len] = sq;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }

    /// The squares of `bits` in ascending order (first 16).
    #[inline(always)]
    fn from_bits(mut bits: u64) -> Self {
        let mut list = Self::new();
        let mut len = 0;
        while bits != 0 && len < list.data.len() {
            list.data[len] = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            len += 1;
        }
        list.len = len;
        list
    }
}

#[derive(Clone, Copy)]
struct PieceList {
    data: [(u8, Piece); 16],
    len: usize,
}

impl PieceList {
    fn new() -> Self {
        Self {
            data: [(0, Piece { color: Color::White, kind: PieceType::Pawn }); 16],
            len: 0,
        }
    }

    fn as_slice(&self) -> &[(u8, Piece)] {
        &self.data[..self.len]
    }
}

pub(crate) fn eval_for_side(board: &Board) -> i32 {
    let s = evaluate(board);
    if board.side_to_move == Color::White {
        s
    } else {
        -s
    }
}

/// Zone-attack units per attack pair, by piece kind (pawn..king), feeding the
/// attacker-count buckets.
const ZONE_UNITS: [i32; 6] = [0, 2, 2, 3, 5, 0];
/// Tropism weight per piece kind (pawn..king).
const TROPISM_WEIGHT: [i32; 6] = [0, 2, 2, 2, 3, 0];
const FILE_A: u64 = 0x0101_0101_0101_0101;

/// King safety of `color`'s king, traced as linear terms; `sign` is +1 for
/// White, -1 for Black. `enemy_atts[i]` is the attack set of
/// `enemy_pieces[i]`; `own_all` / `own_no_king` are the union of own attacks
/// with and without the king, `enemy_all` the union of enemy attacks.
#[allow(clippy::too_many_arguments)]
fn king_safety<T: Tracer>(
    s: &mut Score<'_, T>,
    sign: i32,
    board: &Board,
    color: Color,
    own_pawn_files: &[i32; 8],
    enemy_pawn_files: &[i32; 8],
    enemy_pieces: &[(u8, Piece)],
    enemy_atts: &[u64],
    own_all: u64,
    own_no_king: u64,
    enemy_all: u64,
) {
    use crate::bitboard::{bishop_attacks, kind_index, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS};

    let Some(ksq) = board.king_square(color) else {
        return;
    };
    let ci = color_index(color);
    let zone = KING_ATTACKS[ksq as usize] | (1u64 << ksq);

    // Attacks on the king zone, per attacker kind, and the per-kind attack
    // unions used for safe checks.
    let mut pairs = [0i32; 6];
    let mut kind_att = [0u64; 6];
    let mut attackers = 0;
    let mut units = 0;
    let mut tropism = 0;
    // Pawns (listed first in `enemy_pieces`) only feed `pairs[0]`; each pawn
    // has at most one attack per diagonal direction, so the per-pawn sum of
    // zone hits equals the zone hits of the two shifted pawn sets.
    let enemy_pawns_bb = board.bb.pieces[1 - ci][0];
    let (east, west) = if color == Color::White {
        ((enemy_pawns_bb & !(FILE_A << 7)) >> 7, (enemy_pawns_bb & !FILE_A) >> 9)
    } else {
        ((enemy_pawns_bb & !(FILE_A << 7)) << 9, (enemy_pawns_bb & !FILE_A) << 7)
    };
    pairs[0] = ((east & zone).count_ones() + (west & zone).count_ones()) as i32;
    let npawns = (enemy_pawns_bb.count_ones() as usize).min(enemy_pieces.len());
    debug_assert!(enemy_pieces[..npawns].iter().all(|&(_, p)| p.kind == PieceType::Pawn));
    for (&(from, piece), &att) in enemy_pieces[npawns..].iter().zip(&enemy_atts[npawns..]) {
        let k = kind_index(piece.kind);
        if k == 5 {
            // The king feeds no king-safety term.
            continue;
        }
        kind_att[k] |= att;
        let hits = (att & zone).count_ones() as i32;
        pairs[k] += hits;
        if hits > 0 && ZONE_UNITS[k] > 0 {
            attackers += 1;
            units += ZONE_UNITS[k] * hits;
        }
        if TROPISM_WEIGHT[k] > 0 {
            tropism += (7 - king_distance(from, ksq)).max(0) * TROPISM_WEIGHT[k];
        }
    }
    s.add(P::KsAttackPawn, -sign * pairs[0]);
    s.add(P::KsAttackKnight, -sign * pairs[1]);
    s.add(P::KsAttackBishop, -sign * pairs[2]);
    s.add(P::KsAttackRook, -sign * pairs[3]);
    s.add(P::KsAttackQueen, -sign * pairs[4]);
    if attackers >= 2 {
        let bucket = match attackers {
            2 => P::KsAttackers2,
            3 => P::KsAttackers3,
            _ => P::KsAttackers4,
        };
        s.add(bucket, -sign * units);
    }
    s.add(P::KsTropism, -sign * tropism);

    // Safe checks: squares the enemy piece reaches, from which it would check
    // the king, not defended by us and not occupied by the enemy.
    let occ = board.bb.occ_all();
    let safe = !own_all & !board.bb.occ[1 - ci];
    let diag = bishop_attacks(ksq as usize, occ);
    let ortho = rook_attacks(ksq as usize, occ);
    let checks = |bb: u64| -sign * (bb & safe).count_ones() as i32;
    s.add(P::KsSafeCheckKnight, checks(KNIGHT_ATTACKS[ksq as usize] & kind_att[1]));
    s.add(P::KsSafeCheckBishop, checks(diag & kind_att[2]));
    s.add(P::KsSafeCheckRook, checks(ortho & kind_att[3]));
    s.add(P::KsSafeCheckQueen, checks((diag | ortho) & kind_att[4]));

    // Zone squares the enemy attacks, and those only the king defends.
    let hit = zone & enemy_all;
    s.add(P::KsZoneAttacked, -sign * hit.count_ones() as i32);
    s.add(P::KsZoneWeak, -sign * (hit & !own_no_king).count_ones() as i32);

    // Pawn shield / storm and open files on the king file and its neighbours.
    let kf = (ksq % 8) as i32;
    let kr = (ksq / 8) as i32;
    let white = color == Color::White;
    let ahead = if white {
        if kr < 7 { !0u64 << (8 * (kr + 1)) } else { 0 }
    } else {
        (1u64 << (8 * kr)) - 1
    };
    // Rank distance from the king to the nearest pawn of `bb` ahead of it.
    let dist = |bb: u64| -> i32 {
        let sq = if white { bb.trailing_zeros() as i32 } else { 63 - bb.leading_zeros() as i32 };
        (sq / 8 - kr).abs()
    };
    let own_pawns = board.bb.pieces[ci][0];
    let enemy_pawns = board.bb.pieces[1 - ci][0];
    for f in (kf - 1).max(0)..=(kf + 1).min(7) {
        let file = (FILE_A << f) & ahead;
        let own = own_pawns & file;
        let own_d = if own != 0 { dist(own) } else { 0 };
        let (d1, d2, missing) = if f == kf {
            (P::KsShieldKingFileD1, P::KsShieldKingFileD2, P::KsShieldKingFileMissing)
        } else {
            (P::KsShieldSideFileD1, P::KsShieldSideFileD2, P::KsShieldSideFileMissing)
        };
        match own_d {
            1 => s.add(d1, sign),
            2 => s.add(d2, sign),
            _ => s.add(missing, -sign),
        }

        let enemy = enemy_pawns & file;
        if enemy != 0 {
            let enemy_d = dist(enemy);
            if own != 0 && enemy_d == own_d + 1 {
                s.add(P::KsStormBlocked, -sign);
            } else {
                match enemy_d {
                    1 => s.add(P::KsStormD1, -sign),
                    2 => s.add(P::KsStormD2, -sign),
                    3 => s.add(P::KsStormD3, -sign),
                    4 => s.add(P::KsStormD4, -sign),
                    _ => {}
                }
            }
        }

        let fi = f as usize;
        if own_pawn_files[fi] == 0 {
            let p = if enemy_pawn_files[fi] == 0 { P::KsOpenFile } else { P::KsHalfOpenFile };
            s.add(p, -sign);
        }
    }
}

fn has_pawn_bits(mask: &[u8; 8], file: usize, bits: u8) -> bool {
    (mask[file] & bits) != 0
}

fn is_supported_by_pawn(board: &Board, sq: u8, color: Color) -> bool {
    // Own pawns diagonally behind == own-colored pawn attackers of this square.
    crate::bitboard::PAWN_ATTACKERS[color_index(color)][sq as usize]
        & board.bb.pieces[color_index(color)][0]
        != 0
}

fn is_connected_pawn(pawn_masks: &[u8; 8], sq: u8, color: Color) -> bool {
    let f = (sq % 8) as usize;
    let r = (sq / 8) as u8;
    let mut rank_bits = 1u8 << r;
    match color {
        Color::White if r > 0 => rank_bits |= 1u8 << (r - 1),
        Color::Black if r < 7 => rank_bits |= 1u8 << (r + 1),
        _ => {}
    }
    (f > 0 && has_pawn_bits(pawn_masks, f - 1, rank_bits))
        || (f < 7 && has_pawn_bits(pawn_masks, f + 1, rank_bits))
}

fn is_backward_pawn(board: &Board, pawn_masks: &[u8; 8], sq: u8, color: Color, passed: bool) -> bool {
    if passed {
        return false;
    }

    let f = (sq % 8) as usize;
    let r = (sq / 8) as i32;
    let advance_rank = match color {
        Color::White => r + 1,
        Color::Black => r - 1,
    };
    if !(0..8).contains(&advance_rank) {
        return false;
    }

    let support_bits = match color {
        Color::White => ((1u16 << (r as u32 + 1)) - 1) as u8,
        Color::Black => !((1u8 << (r as u32)) - 1),
    };
    let has_adjacent_support = (f > 0 && has_pawn_bits(pawn_masks, f - 1, support_bits))
        || (f < 7 && has_pawn_bits(pawn_masks, f + 1, support_bits));
    if has_adjacent_support {
        return false;
    }

    let enemy = color.opposite();
    let advance_sq = (advance_rank as u8) * 8 + (f as u8);
    // Enemy pawns attacking the advance square block it: same set as the
    // enemy-colored pawn attackers of that square.
    crate::bitboard::PAWN_ATTACKERS[color_index(enemy)][advance_sq as usize]
        & board.bb.pieces[color_index(enemy)][0]
        != 0
}

fn pawn_has_lever(board: &Board, sq: u8, color: Color) -> bool {
    // Enemy pawns this pawn attacks == enemy pawns on our forward diagonals;
    // the mirrored attacker table gives exactly those squares.
    let enemy = color.opposite();
    crate::bitboard::PAWN_ATTACKERS[color_index(enemy)][sq as usize]
        & board.bb.pieces[color_index(enemy)][0]
        != 0
}

// Per-rank passed-pawn tables, indexed by relative rank - 1 (ranks 2..7).
const PASSED_BASE: [P; 6] =
    [P::PassedR2, P::PassedR3, P::PassedR4, P::PassedR5, P::PassedR6, P::PassedR7];
const PASSED_FREE: [P; 6] = [
    P::PassedFreeR2,
    P::PassedFreeR3,
    P::PassedFreeR4,
    P::PassedFreeR5,
    P::PassedFreeR6,
    P::PassedFreeR7,
];
const PASSED_SAFE_PUSH: [P; 6] = [
    P::PassedSafePushR2,
    P::PassedSafePushR3,
    P::PassedSafePushR4,
    P::PassedSafePushR5,
    P::PassedSafePushR6,
    P::PassedSafePushR7,
];
const PASSED_BLOCKED: [P; 6] = [
    P::PassedBlockedR2,
    P::PassedBlockedR3,
    P::PassedBlockedR4,
    P::PassedBlockedR5,
    P::PassedBlockedR6,
    P::PassedBlockedR7,
];
const PASSED_OWN_KING_DIST: [P; 6] = [
    P::PassedOwnKingDistR2,
    P::PassedOwnKingDistR3,
    P::PassedOwnKingDistR4,
    P::PassedOwnKingDistR5,
    P::PassedOwnKingDistR6,
    P::PassedOwnKingDistR7,
];
const PASSED_ENEMY_KING_DIST: [P; 6] = [
    P::PassedEnemyKingDistR2,
    P::PassedEnemyKingDistR3,
    P::PassedEnemyKingDistR4,
    P::PassedEnemyKingDistR5,
    P::PassedEnemyKingDistR6,
    P::PassedEnemyKingDistR7,
];
const PASSED_CONNECTED: [P; 6] = [
    P::PassedConnectedR2,
    P::PassedConnectedR3,
    P::PassedConnectedR4,
    P::PassedConnectedR5,
    P::PassedConnectedR6,
    P::PassedConnectedR7,
];

/// Squares strictly in front of `sq` on its file, toward promotion.
#[inline(always)]
fn front_span(sq: u8, color: Color) -> u64 {
    let file = 0x0101_0101_0101_0101u64 << (sq % 8);
    match color {
        Color::White => file & (!0u64 << sq << 1),
        Color::Black => file & ((1u64 << sq) - 1),
    }
}

/// Passed-pawn terms; `sign` is +1 for White, -1 for Black. `enemy_all` is
/// the union of the enemy's attacks (king included).
#[allow(clippy::too_many_arguments)]
fn passed_pawn_score<T: Tracer>(
    s: &mut Score<'_, T>,
    sign: i32,
    board: &Board,
    pawn_masks: &[u8; 8],
    enemy_all: u64,
    sq: u8,
    color: Color,
    own_king: Option<u8>,
    enemy_king: Option<u8>,
    own_rooks: &[u8],
) {
    let rr = relative_rank(color, sq);
    let i = (rr - 1).clamp(0, 5) as usize;
    s.add(PASSED_BASE[i], sign);

    let stop = match color {
        Color::White => sq + 8,
        Color::Black => sq - 8,
    };
    let occ = board.bb.occ_all();
    let enemy = color.opposite();
    if front_span(sq, color) & occ == 0 {
        s.add(PASSED_FREE[i], sign);
    }
    if occ & (1u64 << stop) == 0 {
        if enemy_all & (1u64 << stop) == 0 {
            s.add(PASSED_SAFE_PUSH[i], sign);
        }
    } else if board.bb.occ[color_index(enemy)] & (1u64 << stop) != 0 {
        s.add(PASSED_BLOCKED[i], sign);
    }
    if let Some(king) = own_king {
        s.add(PASSED_OWN_KING_DIST[i], sign * king_distance(king, stop).min(5));
    }
    if let Some(king) = enemy_king {
        s.add(PASSED_ENEMY_KING_DIST[i], sign * king_distance(king, stop).min(5));
    }
    if is_connected_pawn(pawn_masks, sq, color) {
        s.add(PASSED_CONNECTED[i], sign);
    }

    let promo_sq = match color {
        Color::White => 56 + (sq % 8),
        Color::Black => sq % 8,
    };
    let remaining = match color {
        Color::White => 7 - (sq / 8) as i32,
        Color::Black => (sq / 8) as i32,
    };
    if let Some(king) = enemy_king {
        let promo_dist = king_distance(king, promo_sq);
        if promo_dist > remaining {
            s.add(P::PasserEnemyKingFar, sign * (promo_dist - remaining));
        }
    }

    for &rook_sq in own_rooks {
        if rook_sq % 8 != sq % 8 {
            continue;
        }
        let rook_rank = rook_sq / 8;
        let pawn_rank = sq / 8;
        let behind = match color {
            Color::White => rook_rank < pawn_rank,
            Color::Black => rook_rank > pawn_rank,
        };
        if behind {
            s.add(P::PasserRookBehind, sign);
        } else {
            s.add(P::PasserRookFront, -sign);
        }
    }
}

/// Trapped-piece penalty; `sign` is +1 for White, -1 for Black.
#[inline(always)]
fn trapped_piece_penalty<T: Tracer>(
    s: &mut Score<'_, T>,
    sign: i32,
    sq: u8,
    piece: Piece,
    mobility: i32,
    own_king: Option<u8>,
) {
    const EDGE: u64 = 0xFF81_8181_8181_81FF;
    const CORNER: u64 = 0x8100_0000_0000_0081;
    // Files a, b, g, h on ranks 1, 2, 7, 8.
    const ROOK_TRAP: u64 = 0xC3C3_0000_0000_C3C3;
    let bit = 1u64 << sq;

    match piece.kind {
        PieceType::Knight => {
            if mobility <= 2 && bit & EDGE != 0 {
                let p = if bit & CORNER != 0 { P::TrappedKnightCorner } else { P::TrappedKnightEdge };
                s.add(p, -sign * (3 - mobility));
            }
        }
        PieceType::Bishop => {
            if mobility <= 2 && bit & EDGE != 0 {
                s.add(P::TrappedBishopEdge, -sign * (3 - mobility));
                if bit & CORNER != 0 {
                    s.add(P::TrappedBishopCornerExtra, -sign);
                }
            }
        }
        PieceType::Rook => {
            if mobility <= 3 && bit & ROOK_TRAP != 0 {
                if let Some(king_sq) = own_king {
                    // The king sits on the rook's rank, between it and the centre.
                    let f = sq % 8;
                    let king_file = king_sq % 8;
                    if king_sq / 8 == sq / 8 && ((f <= 1 && king_file > f) || (f >= 6 && king_file < f))
                    {
                        s.add(P::TrappedRookByKing, -sign * (4 - mobility));
                    }
                }
            }
        }
        _ => {}
    }
}

fn is_outpost_square(
    sq: u8,
    color: Color,
    white_min_pawn_rank: &[i8; 8],
    black_max_pawn_rank: &[i8; 8],
) -> bool {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i8;
    for df in [-1, 1] {
        let ef = f + df;
        if !(0..8).contains(&ef) {
            continue;
        }
        let enemy_pawn_blocks = match color {
            Color::White => black_max_pawn_rank[ef as usize] > r,
            Color::Black => white_min_pawn_rank[ef as usize] < r,
        };
        if enemy_pawn_blocks {
            return false;
        }
    }
    true
}

fn king_edge_pressure(sq: u8) -> i32 {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let edge_dist = f.min(7 - f) + r.min(7 - r);
    14 - edge_dist * 4
}

fn king_distance(a: u8, b: u8) -> i32 {
    let af = (a % 8) as i32;
    let ar = (a / 8) as i32;
    let bf = (b % 8) as i32;
    let br = (b / 8) as i32;
    (af - bf).abs().max((ar - br).abs())
}

fn has_direct_opposition(white_king: u8, black_king: u8) -> bool {
    let same_file = white_king % 8 == black_king % 8
        && ((white_king / 8) as i32 - (black_king / 8) as i32).abs() == 2;
    let same_rank = white_king / 8 == black_king / 8
        && ((white_king % 8) as i32 - (black_king % 8) as i32).abs() == 2;
    same_file || same_rank
}

fn promotion_distance(board: &Board, sq: u8, color: Color, side_to_move: Color) -> Option<i32> {
    let file = sq % 8;
    let rank = (sq / 8) as i32;
    let dir = if color == Color::White { 1 } else { -1 };
    let mut r = rank + dir;
    while (0..8).contains(&r) {
        let forward_sq = (r as u8) * 8 + file;
        if board.squares[forward_sq as usize].is_some() {
            return None;
        }
        r += dir;
    }

    let mut moves = match color {
        Color::White => 7 - rank,
        Color::Black => rank,
    };
    let start_rank = match color {
        Color::White => 1,
        Color::Black => 6,
    };
    if rank == start_rank {
        let one_step = ((rank + dir) as u8) * 8 + file;
        let two_step = ((rank + dir * 2) as u8) * 8 + file;
        if board.squares[one_step as usize].is_none() && board.squares[two_step as usize].is_none()
        {
            moves -= 1;
        }
    }
    if color != side_to_move {
        moves += 1;
    }
    Some(moves)
}

fn passer_is_unstoppable(board: &Board, sq: u8, color: Color, enemy_king: u8) -> bool {
    let Some(moves) = promotion_distance(board, sq, color, board.side_to_move) else {
        return false;
    };
    let promo_sq = match color {
        Color::White => 56 + (sq % 8),
        Color::Black => sq % 8,
    };
    king_distance(enemy_king, promo_sq) > moves
}

/// Single-rook endgame terms (eg only); `sign` is +1 for White, -1 for Black.
fn rook_endgame_bonus<T: Tracer>(
    s: &mut Score<'_, T>,
    sign: i32,
    color: Color,
    rooks: &[u8],
    own_passed: &[u8],
    enemy_passed: &[u8],
    enemy_king: Option<u8>,
) {
    for &rook_sq in rooks {
        if let Some(king_sq) = enemy_king {
            s.add_eg(P::RookEgEnemyKingClose, sign * (7 - king_distance(rook_sq, king_sq)).max(0));
        }

        for &pawn_sq in own_passed {
            if rook_sq % 8 != pawn_sq % 8 {
                continue;
            }
            let rook_rank = rook_sq / 8;
            let pawn_rank = pawn_sq / 8;
            let behind = match color {
                Color::White => rook_rank < pawn_rank,
                Color::Black => rook_rank > pawn_rank,
            };
            if behind {
                s.add_eg(P::RookEgBehindPasser, sign);
            } else {
                s.add_eg(P::RookEgFrontOfPasser, -sign);
            }
        }

        for &pawn_sq in enemy_passed {
            if rook_sq % 8 != pawn_sq % 8 {
                continue;
            }
            let rook_rank = rook_sq / 8;
            let pawn_rank = pawn_sq / 8;
            let behind_enemy = match color {
                Color::White => rook_rank > pawn_rank,
                Color::Black => rook_rank < pawn_rank,
            };
            if behind_enemy {
                s.add_eg(P::RookEgBehindEnemyPasser, sign);
            }
        }
    }
}

fn relative_rank(color: Color, sq: u8) -> i32 {
    let r = (sq / 8) as i32;
    match color {
        Color::White => r,
        Color::Black => 7 - r,
    }
}

/// Appends the pieces of side `c` and kind index `k` (pawn..king) to
/// `pieces` and their attack sets to `atts` (both capped at 16). Returns the
/// union of the appended attack sets.
#[inline(always)]
fn push_kind(
    pieces: &mut PieceList,
    atts: &mut [u64; 16],
    board: &Board,
    c: usize,
    k: usize,
    occ: u64,
) -> u64 {
    const KINDS: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];
    let color = if c == 0 { Color::White } else { Color::Black };
    let piece = Piece { color, kind: KINDS[k] };
    let mut union = 0;
    let mut bits = board.bb.pieces[c][k];
    let mut len = pieces.len;
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        if len < pieces.data.len() {
            let a = piece_attacks(sq, piece, occ);
            pieces.data[len] = (sq, piece);
            atts[len] = a;
            len += 1;
            union |= a;
        }
    }
    pieces.len = len;
    union
}

/// Squares attacked by one side, split by attacker class. Shared by king
/// safety, mobility and threats.
#[derive(Clone, Copy, Default)]
struct AttackMaps {
    pawn: u64,
    minor: u64,
    rook: u64,
    /// Every attacker except the king.
    no_king: u64,
    /// Every attacker, king included.
    all: u64,
}

/// Threats by the side with attack maps `us` against the pieces of `them_idx`
/// (defended by `them`): pieces attacked by a lower-value attacker, and
/// undefended attacked pieces.
#[inline(always)]
fn threats<T: Tracer>(
    s: &mut Score<'_, T>,
    sign: i32,
    board: &Board,
    us: &AttackMaps,
    them: &AttackMaps,
    them_idx: usize,
) {
    let p = &board.bb.pieces[them_idx];
    let minors = p[1] | p[2];
    let rooks = p[3];
    let queens = p[4];
    let cnt = |b: u64| b.count_ones() as i32;
    let n = cnt(us.pawn & minors);
    if n != 0 {
        s.add(P::ThreatPawnOnMinor, sign * n);
    }
    let n = cnt(us.pawn & rooks);
    if n != 0 {
        s.add(P::ThreatPawnOnRook, sign * n);
    }
    let n = cnt(us.pawn & queens);
    if n != 0 {
        s.add(P::ThreatPawnOnQueen, sign * n);
    }
    let n = cnt(us.minor & rooks);
    if n != 0 {
        s.add(P::ThreatMinorOnRook, sign * n);
    }
    let n = cnt(us.minor & queens);
    if n != 0 {
        s.add(P::ThreatMinorOnQueen, sign * n);
    }
    let n = cnt(us.rook & queens);
    if n != 0 {
        s.add(P::ThreatRookOnQueen, sign * n);
    }
    let n = cnt(us.all & !them.all & (minors | rooks | queens));
    if n != 0 {
        s.add(P::Hanging, sign * n);
    }
}

pub(crate) fn taper_score(mg: i32, eg: i32, phase: i32) -> i32 {
    (mg * phase + eg * (24 - phase)) / 24
}

/// Development terms (mg only); `sign` is +1 for White, -1 for Black.
#[inline(always)]
fn development_score<T: Tracer>(s: &mut Score<'_, T>, sign: i32, board: &Board, color: Color) {
    let (minor_starts, king_home, king_castles) = match color {
        Color::White => (&[1u8, 2, 5, 6][..], 4u8, [2u8, 6u8]),
        Color::Black => (&[57u8, 58, 61, 62][..], 60u8, [58u8, 62u8]),
    };

    let ci = color_index(color);
    let start_mask = minor_starts.iter().fold(0u64, |m, &sq| m | (1u64 << sq));
    let minors = board.bb.pieces[ci][1] | board.bb.pieces[ci][2];
    let undeveloped = (minors & start_mask).count_ones() as i32;

    let king_sq = board.king_square(color);
    s.add_mg(P::DevUndevelopedMinor, -sign * undeveloped);
    if king_sq == Some(king_home) && undeveloped >= 2 {
        s.add_mg(P::DevKingHomeUndeveloped, -sign);
    }
    if king_sq.is_some_and(|sq| king_castles.contains(&sq)) {
        s.add_mg(P::DevKingCastled, sign);
    }
}

/// Full-strength endgame scale factor (scores are multiplied by `scale / SCALE_NORMAL`).
pub(crate) const SCALE_NORMAL: i32 = 128;
/// Stronger side has no pawns and too little extra material to mate.
const SCALE_DRAW: i32 = 0;
/// No pawns, small material edge, defender has at most a minor (e.g. KR vs KB).
const SCALE_NO_PAWNS_VS_MINOR: i32 = 8;
/// No pawns, small material edge, defender has more than a minor (e.g. KRB vs KR).
const SCALE_NO_PAWNS_VS_MAJOR: i32 = 28;
/// One pawn and the defender can give up a piece for it.
const SCALE_ONE_PAWN: i32 = 96;
/// Pure opposite-coloured bishop ending with a pawn edge of at most one.
const SCALE_OCB_CLOSE: i32 = 40;
/// Pure opposite-coloured bishop ending with a larger pawn edge.
const SCALE_OCB: i32 = 64;
/// Single-rook ending with a one-pawn edge.
const SCALE_ROOK_ONE_PAWN: i32 = 96;
/// Light squares (b1, a2, ...); a1 is dark.
const LIGHT_SQUARES: u64 = 0x55AA_55AA_55AA_55AA;
const FILE_A_BB: u64 = 0x0101_0101_0101_0101;
const FILE_H_BB: u64 = FILE_A_BB << 7;

/// Drawishness scale factor in `0..=SCALE_NORMAL` for the side the tapered
/// `score` favours (white if positive). Applied as `score * scale / SCALE_NORMAL`.
fn endgame_scale(board: &Board, score: i32) -> i32 {
    if score == 0 {
        return SCALE_NORMAL;
    }
    let acc = &board.eval_acc;
    let (s, w) = if score > 0 { (0usize, 1usize) } else { (1usize, 0usize) };
    let pawns = [
        board.bb.pieces[0][0].count_ones() as i32,
        board.bb.pieces[1][0].count_ones() as i32,
    ];
    let npm_s = acc.npm[s];
    let npm_w = acc.npm[w];
    let bishop = search::piece_value(PieceType::Bishop);
    let rook = search::piece_value(PieceType::Rook);

    if pawns[s] == 0 && npm_s - npm_w <= bishop {
        return if npm_s < rook {
            SCALE_DRAW
        } else if npm_w <= bishop {
            SCALE_NO_PAWNS_VS_MINOR
        } else {
            SCALE_NO_PAWNS_VS_MAJOR
        };
    }
    // KNN vs bare king cannot be forced.
    if pawns[s] == 0
        && pawns[w] == 0
        && npm_w == 0
        && npm_s == 2 * search::piece_value(PieceType::Knight)
        && board.bb.pieces[s][1].count_ones() == 2
    {
        return SCALE_DRAW;
    }

    // KB + rook pawn(s) vs K with the wrong-coloured bishop and the defending
    // king on the queening corner.
    if npm_s == bishop && acc.bishops[s] == 1 && npm_w == 0 && pawns[s] > 0 {
        let sp = board.bb.pieces[s][0];
        let file_bb = if sp & !FILE_A_BB == 0 {
            Some(0u8)
        } else if sp & !FILE_H_BB == 0 {
            Some(7u8)
        } else {
            None
        };
        let weak_king = board.king_square(if w == 0 { Color::White } else { Color::Black });
        if let (Some(file), Some(wk)) = (file_bb, weak_king) {
            let queen_sq = if s == 0 { 56 + file } else { file };
            let bishop_light = board.bb.pieces[s][2] & LIGHT_SQUARES != 0;
            let corner_light = (1u64 << queen_sq) & LIGHT_SQUARES != 0;
            let dist = ((wk % 8) as i32 - (queen_sq % 8) as i32)
                .abs()
                .max(((wk / 8) as i32 - (queen_sq / 8) as i32).abs());
            if bishop_light != corner_light && dist <= 1 {
                return SCALE_DRAW;
            }
        }
    }

    if pawns[s] == 1 && npm_w > 0 && npm_s - npm_w <= bishop {
        return SCALE_ONE_PAWN;
    }

    let pawn_diff = pawns[s] - pawns[w];
    // Pure opposite-coloured bishops.
    if npm_s == bishop && npm_w == bishop && acc.bishops[0] == 1 && acc.bishops[1] == 1 {
        let s_light = board.bb.pieces[s][2] & LIGHT_SQUARES != 0;
        let w_light = board.bb.pieces[w][2] & LIGHT_SQUARES != 0;
        if s_light != w_light {
            return if pawn_diff <= 1 { SCALE_OCB_CLOSE } else { SCALE_OCB };
        }
    }
    // Single-rook endings with a one-pawn edge.
    if npm_s == rook
        && npm_w == rook
        && board.bb.pieces[0][3].count_ones() == 1
        && board.bb.pieces[1][3].count_ones() == 1
        && pawn_diff <= 1
    {
        return SCALE_ROOK_ONE_PAWN;
    }
    SCALE_NORMAL
}

/// Observer of the linear structure of `evaluate`: receives, for every
/// eval_params term, the White-minus-Black coefficient that multiplies its mg
/// and eg weight (before tapering), plus the nonlinear remainder and the
/// endgame scale factor applied to the tapered score. The no-op `NoTrace` is
/// what search uses; it compiles away completely.
pub(crate) trait Tracer {
    fn add(&mut self, p: P, mg: i32, eg: i32);
    fn nonlinear(&mut self, mg: i32, eg: i32);
    /// Endgame scale factor: the final score is
    /// `taper_score(mg, eg, phase) * scale / SCALE_NORMAL + tempo`.
    fn scale(&mut self, scale: i32);
}

pub(crate) struct NoTrace;

impl Tracer for NoTrace {
    #[inline(always)]
    fn add(&mut self, _p: P, _mg: i32, _eg: i32) {}
    #[inline(always)]
    fn nonlinear(&mut self, _mg: i32, _eg: i32) {}
    #[inline(always)]
    fn scale(&mut self, _scale: i32) {}
}

/// Recording tracer used by the tuner.
#[derive(Clone)]
pub(crate) struct EvalTrace {
    pub(crate) mg: [i32; PARAM_COUNT],
    pub(crate) eg: [i32; PARAM_COUNT],
    /// Nonlinear (untuned) part of the untapered mg / eg score: mop-up edge pressure.
    pub(crate) nonlinear_mg: i32,
    pub(crate) nonlinear_eg: i32,
    /// Endgame scale factor (SCALE_NORMAL = unscaled) applied to the tapered score.
    pub(crate) scale: i32,
}

impl EvalTrace {
    pub(crate) fn new() -> Self {
        Self {
            mg: [0; PARAM_COUNT],
            eg: [0; PARAM_COUNT],
            nonlinear_mg: 0,
            nonlinear_eg: 0,
            scale: SCALE_NORMAL,
        }
    }
}

impl Tracer for EvalTrace {
    fn add(&mut self, p: P, mg: i32, eg: i32) {
        self.mg[p as usize] += mg;
        self.eg[p as usize] += eg;
    }
    fn nonlinear(&mut self, mg: i32, eg: i32) {
        self.nonlinear_mg += mg;
        self.nonlinear_eg += eg;
    }
    fn scale(&mut self, scale: i32) {
        self.scale = scale;
    }
}

/// Untapered (mg, eg) score under construction, forwarding every term to the tracer.
pub(crate) struct Score<'a, T: Tracer> {
    mg: i32,
    eg: i32,
    t: &'a mut T,
}

impl<T: Tracer> Score<'_, T> {
    /// `n` times parameter `p`, in both mg and eg.
    #[inline(always)]
    fn add(&mut self, p: P, n: i32) {
        self.mg += n * ep::mg(p);
        self.eg += n * ep::eg(p);
        self.t.add(p, n, n);
    }

    #[inline(always)]
    fn add_mg(&mut self, p: P, n: i32) {
        self.mg += n * ep::mg(p);
        self.t.add(p, n, 0);
    }

    #[inline(always)]
    fn add_eg(&mut self, p: P, n: i32) {
        self.eg += n * ep::eg(p);
        self.t.add(p, 0, n);
    }

    #[inline(always)]
    fn nonlinear(&mut self, mg: i32, eg: i32) {
        self.mg += mg;
        self.eg += eg;
        self.t.nonlinear(mg, eg);
    }
}

/// Mobility (returned), outposts, bishop mobility and trapped pieces for one
/// side; `sign` is +1 for White, -1 for Black. `pieces` / `atts` are in kind
/// order with `ends[k]` the list length after kind `k`; `own_occ` is the
/// side's occupancy.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn mobility_side<T: Tracer>(
    s: &mut Score<'_, T>,
    sign: i32,
    color: Color,
    pieces: &[(u8, Piece)],
    atts: &[u64; 16],
    ends: &[usize; 6],
    own_occ: u64,
    white_min_pawn_rank: &[i8; 8],
    black_max_pawn_rank: &[i8; 8],
    own_king: Option<u8>,
) -> i32 {
    let white = color == Color::White;
    let mob = |a: u64| (a & !own_occ).count_ones() as i32;
    // Pieces of kinds `from..to` (indices into `ends`, 0 = pawn) with attacks.
    let kinds = |from: usize, to: usize| {
        let lo = if from == 0 { 0 } else { ends[from - 1] };
        let hi = ends[to - 1];
        pieces[lo..hi].iter().zip(&atts[lo..hi])
    };
    let mut total = 0;
    // Pawns, queens and king: mobility only (no outpost or trapped terms).
    for (_, &a) in kinds(0, 1).chain(kinds(4, 6)) {
        total += mob(a);
    }
    let outpost = |sq: u8| {
        let r = sq / 8;
        (if white { r >= 3 } else { r <= 4 })
            && is_outpost_square(sq, color, white_min_pawn_rank, black_max_pawn_rank)
    };
    for (&(sq, piece), &a) in kinds(1, 2) {
        let mobility = mob(a);
        total += mobility;
        if outpost(sq) {
            s.add(P::KnightOutpost, sign);
        }
        trapped_piece_penalty(s, sign, sq, piece, mobility, own_king);
    }
    for (&(sq, piece), &a) in kinds(2, 3) {
        let mobility = mob(a);
        total += mobility;
        if outpost(sq) {
            s.add(P::BishopOutpost, sign);
        }
        s.add(P::BishopMobilityExtra, sign * mobility);
        trapped_piece_penalty(s, sign, sq, piece, mobility, own_king);
    }
    for (&(sq, piece), &a) in kinds(3, 4) {
        let mobility = mob(a);
        total += mobility;
        trapped_piece_penalty(s, sign, sq, piece, mobility, own_king);
    }
    total
}

pub(crate) fn evaluate(board: &Board) -> i32 {
    evaluate_with(board, &mut NoTrace)
}

/// `evaluate`, additionally reporting its linear structure to `t`.
#[inline(always)]
pub(crate) fn evaluate_with<T: Tracer>(board: &Board, t: &mut T) -> i32 {
    let acc = &board.eval_acc;
    let mut s = Score { mg: acc.mg, eg: acc.eg, t };

    let white_pawns = SqList::from_bits(board.bb.pieces[0][0]);
    let black_pawns = SqList::from_bits(board.bb.pieces[1][0]);
    let mut white_pieces = PieceList::new();
    let mut black_pieces = PieceList::new();
    let white_rooks = SqList::from_bits(board.bb.pieces[0][3]);
    let black_rooks = SqList::from_bits(board.bb.pieces[1][3]);
    let mut white_passed = SqList::new();
    let mut black_passed = SqList::new();
    let mut white_pawn_files = [0i32; 8];
    let mut black_pawn_files = [0i32; 8];
    let mut white_pawn_masks = [0u8; 8];
    let mut black_pawn_masks = [0u8; 8];
    let mut white_min_pawn_rank = [8i8; 8];
    let mut black_max_pawn_rank = [-1i8; 8];
    let white_non_pawn_material = acc.npm[0];
    let black_non_pawn_material = acc.npm[1];
    let white_has_queen = acc.queens[0] > 0;
    let black_has_queen = acc.queens[1] > 0;
    let phase = acc.phase;

    // Piece lists in kind order (pawns..king) with per-piece attack sets and
    // per-side attack maps, shared by passed pawns, king safety, mobility and
    // threats. `ends[c][k]` is the list length after kind `k`.
    let occ = board.bb.occ_all();
    let mut white_atts = [0u64; 16];
    let mut black_atts = [0u64; 16];
    let mut att = [AttackMaps::default(); 2];
    let mut ends = [[0usize; 6]; 2];
    for c in 0..2usize {
        let (pieces, atts) = if c == 0 {
            (&mut white_pieces, &mut white_atts)
        } else {
            (&mut black_pieces, &mut black_atts)
        };
        let pawn = push_kind(pieces, atts, board, c, 0, occ);
        ends[c][0] = pieces.len;
        let knight = push_kind(pieces, atts, board, c, 1, occ);
        ends[c][1] = pieces.len;
        let bishop = push_kind(pieces, atts, board, c, 2, occ);
        ends[c][2] = pieces.len;
        let rook = push_kind(pieces, atts, board, c, 3, occ);
        ends[c][3] = pieces.len;
        let queen = push_kind(pieces, atts, board, c, 4, occ);
        ends[c][4] = pieces.len;
        let king = push_kind(pieces, atts, board, c, 5, occ);
        ends[c][5] = pieces.len;
        let no_king = pawn | knight | bishop | rook | queen;
        att[c] = AttackMaps { pawn, minor: knight | bishop, rook, no_king, all: no_king | king };
    }
    // Per-file pawn counts, rank masks (bit r = pawn on rank r) and the
    // White-most-backward / Black-most-advanced pawn ranks (8 / -1 if none).
    let (wp, bp) = (board.bb.pieces[0][0], board.bb.pieces[1][0]);
    for f in 0..8 {
        let w = (wp >> f) & FILE_A;
        let b = (bp >> f) & FILE_A;
        white_pawn_files[f] = w.count_ones() as i32;
        black_pawn_files[f] = b.count_ones() as i32;
        white_pawn_masks[f] = (w.wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8;
        black_pawn_masks[f] = (b.wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8;
        if w != 0 {
            white_min_pawn_rank[f] = (w.trailing_zeros() / 8) as i8;
        }
        if b != 0 {
            black_max_pawn_rank[f] = ((63 - b.leading_zeros()) / 8) as i8;
        }
    }

    if acc.bishops[0] >= 2 {
        s.add(P::BishopPair, 1);
    }
    if acc.bishops[1] >= 2 {
        s.add(P::BishopPair, -1);
    }

    if phase >= 12 {
        development_score(&mut s, 1, board, Color::White);
        development_score(&mut s, -1, board, Color::Black);
    }

    let white_king = board.king_square(Color::White);
    let black_king = board.king_square(Color::Black);

    for &sq in white_pawns.as_slice() {
        let f = (sq % 8) as usize;
        let r = (sq / 8) as i32;

        if white_pawn_files[f] > 1 {
            s.add(P::Doubled, -1);
        }

        let left = if f > 0 { white_pawn_files[f - 1] } else { 0 };
        let right = if f < 7 { white_pawn_files[f + 1] } else { 0 };
        if left == 0 && right == 0 {
            s.add(P::Isolated, -1);
        }

        let mut passed = true;
        for file in f.saturating_sub(1)..=(f + 1).min(7) {
            if black_max_pawn_rank[file] > r as i8 {
                passed = false;
                break;
            }
        }
        if passed {
            white_passed.push(sq);
            passed_pawn_score(
                &mut s,
                1,
                board,
                &white_pawn_masks,
                att[1].all,
                sq,
                Color::White,
                white_king,
                black_king,
                white_rooks.as_slice(),
            );
        }

        if is_connected_pawn(&white_pawn_masks, sq, Color::White) {
            s.add(P::Connected, 1);
        }

        if is_backward_pawn(board, &white_pawn_masks, sq, Color::White, passed) {
            s.add(P::Backward, -1);
        }

        let candidate = black_max_pawn_rank[f] <= r as i8;
        if candidate {
            s.add(P::CandidateBase, 1);
            s.add(P::CandidateRank, r);
            if pawn_has_lever(board, sq, Color::White) {
                s.add(P::CandidateLever, 1);
                s.add(P::CandidateLeverRank, r);
            }
            if is_supported_by_pawn(board, sq, Color::White)
                || is_connected_pawn(&white_pawn_masks, sq, Color::White)
            {
                s.add(P::CandidateSupported, 1);
            }
        }
    }

    for &sq in black_pawns.as_slice() {
        let f = (sq % 8) as usize;
        let r = (sq / 8) as i32;

        if black_pawn_files[f] > 1 {
            s.add(P::Doubled, 1);
        }

        let left = if f > 0 { black_pawn_files[f - 1] } else { 0 };
        let right = if f < 7 { black_pawn_files[f + 1] } else { 0 };
        if left == 0 && right == 0 {
            s.add(P::Isolated, 1);
        }

        let mut passed = true;
        for file in f.saturating_sub(1)..=(f + 1).min(7) {
            if white_min_pawn_rank[file] < r as i8 {
                passed = false;
                break;
            }
        }
        if passed {
            black_passed.push(sq);
            passed_pawn_score(
                &mut s,
                -1,
                board,
                &black_pawn_masks,
                att[0].all,
                sq,
                Color::Black,
                black_king,
                white_king,
                black_rooks.as_slice(),
            );
        }

        if is_connected_pawn(&black_pawn_masks, sq, Color::Black) {
            s.add(P::Connected, -1);
        }

        if is_backward_pawn(board, &black_pawn_masks, sq, Color::Black, passed) {
            s.add(P::Backward, 1);
        }

        let candidate = white_min_pawn_rank[f] >= r as i8;
        if candidate {
            s.add(P::CandidateBase, -1);
            s.add(P::CandidateRank, -(7 - r));
            if pawn_has_lever(board, sq, Color::Black) {
                s.add(P::CandidateLever, -1);
                s.add(P::CandidateLeverRank, -(7 - r));
            }
            if is_supported_by_pawn(board, sq, Color::Black)
                || is_connected_pawn(&black_pawn_masks, sq, Color::Black)
            {
                s.add(P::CandidateSupported, -1);
            }
        }
    }

    for &sq in white_rooks.as_slice() {
        let f = (sq % 8) as usize;
        let r = sq / 8;
        if white_pawn_files[f] == 0 {
            let p = if black_pawn_files[f] == 0 { P::RookOpenFile } else { P::RookSemiOpenFile };
            s.add(p, 1);
        }
        if r == 6 {
            s.add(P::RookSeventh, 1);
        }
    }
    for &sq in black_rooks.as_slice() {
        let f = (sq % 8) as usize;
        let r = sq / 8;
        if black_pawn_files[f] == 0 {
            let p = if white_pawn_files[f] == 0 { P::RookOpenFile } else { P::RookSemiOpenFile };
            s.add(p, -1);
        }
        if r == 1 {
            s.add(P::RookSeventh, -1);
        }
    }

    for &sq in white_pawns.as_slice() {
        let f = (sq % 8) as i32;
        let r = (sq / 8) as i32;
        for df in [-1, 1] {
            let nf = f + df;
            let nr = r + 1;
            if !in_bounds(nf as i8, nr as i8) {
                continue;
            }
            let target = (nr as u8) * 8 + (nf as u8);
            let rel = relative_rank(Color::White, target);
            if (2..=4).contains(&rel) && (2..=5).contains(&nf) {
                s.add(P::PawnCenterControl, 1);
            }
        }
    }
    for &sq in black_pawns.as_slice() {
        let f = (sq % 8) as i32;
        let r = (sq / 8) as i32;
        for df in [-1, 1] {
            let nf = f + df;
            let nr = r - 1;
            if !in_bounds(nf as i8, nr as i8) {
                continue;
            }
            let target = (nr as u8) * 8 + (nf as u8);
            let rel = relative_rank(Color::Black, target);
            if (2..=4).contains(&rel) && (2..=5).contains(&nf) {
                s.add(P::PawnCenterControl, -1);
            }
        }
    }

    let phase = phase.clamp(0, 24);
    let (white_all, white_no_king) = (att[0].all, att[0].no_king);
    let (black_all, black_no_king) = (att[1].all, att[1].no_king);

    king_safety(
        &mut s,
        1,
        board,
        Color::White,
        &white_pawn_files,
        &black_pawn_files,
        black_pieces.as_slice(),
        &black_atts,
        white_all,
        white_no_king,
        black_all,
    );
    king_safety(
        &mut s,
        -1,
        board,
        Color::Black,
        &black_pawn_files,
        &white_pawn_files,
        white_pieces.as_slice(),
        &white_atts,
        black_all,
        black_no_king,
        white_all,
    );

    if phase <= 8 {
        let material_edge = white_non_pawn_material - black_non_pawn_material;
        if material_edge >= 500 {
            if let (Some(wk), Some(bk)) = (white_king, black_king) {
                s.nonlinear(0, king_edge_pressure(bk));
                s.add(P::MopUpKingDistance, 14 - king_distance(wk, bk));
            }
        } else if material_edge <= -500 {
            if let (Some(wk), Some(bk)) = (white_king, black_king) {
                s.nonlinear(0, -king_edge_pressure(wk));
                s.add(P::MopUpKingDistance, -(14 - king_distance(wk, bk)));
            }
        }
    }

    if white_non_pawn_material == 0 && black_non_pawn_material == 0 {
        if let (Some(wk), Some(bk)) = (white_king, black_king) {
            if has_direct_opposition(wk, bk)
                && (!white_pawns.as_slice().is_empty() || !black_pawns.as_slice().is_empty())
            {
                if board.side_to_move == Color::Black {
                    s.add(P::PawnEndgameOpposition, 1);
                } else {
                    s.add(P::PawnEndgameOpposition, -1);
                }
            }
        }

        let white_race = white_passed
            .as_slice()
            .iter()
            .filter_map(|&sq| {
                black_king
                    .filter(|&king| passer_is_unstoppable(board, sq, Color::White, king))
                    .and_then(|_| promotion_distance(board, sq, Color::White, board.side_to_move))
            })
            .min();
        let black_race = black_passed
            .as_slice()
            .iter()
            .filter_map(|&sq| {
                white_king
                    .filter(|&king| passer_is_unstoppable(board, sq, Color::Black, king))
                    .and_then(|_| promotion_distance(board, sq, Color::Black, board.side_to_move))
            })
            .min();

        // (sign, base param, per-unit param, units)
        let race = match (white_race, black_race) {
            (Some(w), None) => Some((1, P::RaceUnstoppable, P::RaceUnstoppableSpeed, (8 - w).max(0))),
            (None, Some(b)) => Some((-1, P::RaceUnstoppable, P::RaceUnstoppableSpeed, (8 - b).max(0))),
            (Some(w), Some(b)) if w + 1 < b => Some((1, P::RaceAhead, P::RaceAheadGap, b - w)),
            (Some(w), Some(b)) if b + 1 < w => Some((-1, P::RaceAhead, P::RaceAheadGap, w - b)),
            _ => None,
        };
        if let Some((sign, base, unit, units)) = race {
            s.add(base, sign);
            s.add(unit, sign * units);
        }
    }

    if white_rooks.as_slice().len() == 1
        && black_rooks.as_slice().len() == 1
        && !white_has_queen
        && !black_has_queen
    {
        rook_endgame_bonus(
            &mut s,
            1,
            Color::White,
            white_rooks.as_slice(),
            white_passed.as_slice(),
            black_passed.as_slice(),
            black_king,
        );
        rook_endgame_bonus(
            &mut s,
            -1,
            Color::Black,
            black_rooks.as_slice(),
            black_passed.as_slice(),
            white_passed.as_slice(),
            white_king,
        );
    }

    let white_mob = mobility_side(
        &mut s,
        1,
        Color::White,
        white_pieces.as_slice(),
        &white_atts,
        &ends[0],
        board.bb.occ[0],
        &white_min_pawn_rank,
        &black_max_pawn_rank,
        white_king,
    );
    let black_mob = mobility_side(
        &mut s,
        -1,
        Color::Black,
        black_pieces.as_slice(),
        &black_atts,
        &ends[1],
        board.bb.occ[1],
        &white_min_pawn_rank,
        &black_max_pawn_rank,
        black_king,
    );
    s.add(P::Mobility, white_mob - black_mob);

    threats(&mut s, 1, board, &att[0], &att[1], 1);
    threats(&mut s, -1, board, &att[1], &att[0], 0);

    let mut score = taper_score(s.mg, s.eg, phase);
    let scale = endgame_scale(board, score);
    s.t.scale(scale);
    score = score * scale / SCALE_NORMAL;
    if board.side_to_move == Color::White {
        score += TEMPO;
    } else {
        score -= TEMPO;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lever_detection_finds_adjacent_enemy_pawn() {
        let mut board = Board::new();
        board.set_fen("4k3/8/3p4/4P3/8/8/8/4K3 w - - 0 1");
        assert!(pawn_has_lever(&board, parse_square("e5").unwrap(), Color::White));
        assert!(!pawn_has_lever(&board, parse_square("e5").unwrap(), Color::Black));
    }

    #[test]
    fn trapped_corner_knight_gets_penalized() {
        let knight = Piece { color: Color::White, kind: PieceType::Knight };
        let (mut t1, mut t2) = (NoTrace, NoTrace);
        let mut trapped = Score { mg: 0, eg: 0, t: &mut t1 };
        trapped_piece_penalty(&mut trapped, 1, parse_square("a1").unwrap(), knight, 1, None);
        let mut free = Score { mg: 0, eg: 0, t: &mut t2 };
        trapped_piece_penalty(&mut free, 1, parse_square("d4").unwrap(), knight, 6, None);
        // Only mg: tuned eg weights may take either sign (the corner PSQT
        // squares already price an eg corner knight).
        assert!(trapped.mg < free.mg);
    }

    #[test]
    fn trace_rebuilds_evaluate_exactly() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
            "8/5k2/8/3P4/8/8/5K2/8 b - - 0 1",
            "6k1/5p2/8/8/8/8/1R3PP1/r5K1 w - - 0 40",
            "4k3/8/8/8/8/8/8/1Q2K3 b - - 0 1",
            "2n1k3/1P6/8/8/8/8/8/N3K2B w - - 0 1",
            "r2qk2r/ppp2ppp/2n5/3pP1B1/1b1n4/2N2N2/PPP2PPP/R2QKB1R b KQkq - 0 8",
            // Endgame-scaled positions: OCB, single rooks, wrong bishop, KB vs K.
            "4k3/5pp1/4b3/8/8/2B5/5PPP/4K3 w - - 0 1",
            "4k3/r4pp1/8/8/8/8/R4PPP/4K3 b - - 0 1",
            "7k/8/8/7P/8/8/2B5/4K3 w - - 0 1",
            "8/8/4k3/8/8/2B5/8/4K3 b - - 0 1",
        ];
        let mut scaled = 0;
        let mut board = Board::new();
        for fen in fens {
            board.set_fen(fen);
            let mut t = EvalTrace::new();
            let score = evaluate_with(&board, &mut t);
            assert_eq!(score, evaluate(&board), "{fen}");
            let (mut mg, mut eg) = (board.eval_acc.mg + t.nonlinear_mg, board.eval_acc.eg + t.nonlinear_eg);
            for i in 0..PARAM_COUNT {
                mg += t.mg[i] * ep::PARAMS[i].0;
                eg += t.eg[i] * ep::PARAMS[i].1;
            }
            let tempo = if board.side_to_move == Color::White { TEMPO } else { -TEMPO };
            let phase = board.eval_acc.phase.clamp(0, 24);
            // The endgame scale is applied after tapering, so the trace
            // records the factor and the rebuild applies it the same way.
            if t.scale != SCALE_NORMAL {
                scaled += 1;
            }
            assert_eq!(
                taper_score(mg, eg, phase) * t.scale / SCALE_NORMAL + tempo,
                score,
                "{fen}"
            );
        }
        assert!(scaled >= 4, "scaled positions exercised: {scaled}");
    }

    fn eval_fen(fen: &str) -> i32 {
        let mut board = Board::new();
        board.set_fen(fen);
        evaluate(&board)
    }

    #[test]
    fn insufficient_material_scales_to_near_zero() {
        for fen in [
            "8/8/4k3/8/8/2B5/8/4K3 w - - 0 1",   // KB vs K
            "8/8/4k3/8/8/2N5/8/4K3 b - - 0 1",   // KN vs K
            "8/8/4k3/8/8/2NN4/8/4K3 w - - 0 1",  // KNN vs K
            "8/8/4k3/3n4/8/2B5/8/4K3 w - - 0 1", // KB vs KN
            "8/8/4k3/3p4/8/2B5/8/4K3 w - - 0 1", // KB vs KP
        ] {
            let s = eval_fen(fen);
            assert!(s.abs() <= 2 * TEMPO, "{fen}: {s}");
        }
    }

    #[test]
    fn opposite_bishops_are_scaled_but_pawn_edge_kept() {
        // OCB, white a pawn up.
        let ocb = eval_fen("4k3/5pp1/4b3/8/8/2B5/5PPP/4K3 w - - 0 1");
        // Same-coloured bishops, white a pawn up.
        let scb = eval_fen("4k3/5pp1/3b4/8/8/2B5/5PPP/4K3 w - - 0 1");
        assert!(ocb > 0 && ocb < scb, "ocb {ocb} scb {scb}");
        // KBP vs K with a central pawn stays clearly winning.
        assert!(eval_fen("8/8/4k3/8/3P4/2B5/8/4K3 w - - 0 1") > 300);
        // Wrong-coloured bishop + h-pawn with the defender in the corner: draw.
        let wrong = eval_fen("7k/8/8/7P/8/8/2B5/4K3 w - - 0 1");
        // Right-coloured bishop: still winning.
        assert!(eval_fen("7k/8/8/7P/8/8/1B6/4K3 w - - 0 1") > 300);
        assert!(wrong.abs() <= 2 * TEMPO, "wrong bishop {wrong}");
    }

    #[test]
    fn threats_are_counted() {
        let mut board = Board::new();
        // e4 forks the undefended d5 knight and f5 bishop.
        board.set_fen("4k3/8/8/3n1b2/4P3/8/8/4K3 w - - 0 1");
        let mut t = EvalTrace::new();
        evaluate_with(&board, &mut t);
        assert_eq!(t.mg[P::ThreatPawnOnMinor as usize], 2);
        assert_eq!(t.mg[P::Hanging as usize], 2);
        // Knight on c7 forks the a8 rook and e8 queen, which defend each other.
        board.set_fen("r3q1k1/2N3pp/8/8/8/8/8/4K2R b - - 0 1");
        let mut t = EvalTrace::new();
        evaluate_with(&board, &mut t);
        assert_eq!(t.mg[P::ThreatMinorOnRook as usize], 1);
        assert_eq!(t.mg[P::ThreatMinorOnQueen as usize], 1);
        assert_eq!(t.mg[P::Hanging as usize], 0);
    }

    #[test]
    fn direct_opposition_is_detected() {
        assert!(has_direct_opposition(
            parse_square("c4").unwrap(),
            parse_square("e4").unwrap(),
        ));
        assert!(has_direct_opposition(
            parse_square("e3").unwrap(),
            parse_square("e5").unwrap(),
        ));
        assert!(!has_direct_opposition(
            parse_square("c4").unwrap(),
            parse_square("f4").unwrap(),
        ));
    }

    #[test]
    fn unstoppable_pawn_race_is_detected() {
        let mut board = Board::new();
        board.set_fen("7k/8/3P4/8/8/8/4K3/8 w - - 0 1");
        assert!(passer_is_unstoppable(
            &board,
            parse_square("d6").unwrap(),
            Color::White,
            parse_square("h8").unwrap(),
        ));
    }
}
