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

    fn push(&mut self, sq: u8, piece: Piece) {
        if self.len < self.data.len() {
            self.data[self.len] = (sq, piece);
            self.len += 1;
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
    for (&(from, piece), &att) in enemy_pieces.iter().zip(enemy_atts) {
        let k = kind_index(piece.kind);
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

/// King and rook support of a passed pawn; `sign` is +1 for White, -1 for Black.
fn passed_pawn_support_score<T: Tracer>(
    s: &mut Score<'_, T>,
    sign: i32,
    sq: u8,
    color: Color,
    own_king: Option<u8>,
    enemy_king: Option<u8>,
    own_rooks: &[u8],
) {
    if let Some(king) = own_king {
        let dist = king_distance(king, sq);
        if dist <= 2 {
            s.add(P::PasserOwnKingNear, sign * (3 - dist));
        }
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
        if king_distance(king, sq) <= 1 {
            s.add(P::PasserEnemyKingAdjacent, -sign);
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
fn trapped_piece_penalty<T: Tracer>(
    s: &mut Score<'_, T>,
    sign: i32,
    sq: u8,
    piece: Piece,
    mobility: i32,
    own_king: Option<u8>,
) {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let on_edge = f == 0 || f == 7 || r == 0 || r == 7;
    let in_corner = (f == 0 || f == 7) && (r == 0 || r == 7);

    match piece.kind {
        PieceType::Knight => {
            if in_corner && mobility <= 2 {
                s.add(P::TrappedKnightCorner, -sign * (3 - mobility));
            } else if on_edge && mobility <= 2 {
                s.add(P::TrappedKnightEdge, -sign * (3 - mobility));
            }
        }
        PieceType::Bishop => {
            if on_edge && mobility <= 2 {
                s.add(P::TrappedBishopEdge, -sign * (3 - mobility));
                if in_corner {
                    s.add(P::TrappedBishopCornerExtra, -sign);
                }
            }
        }
        PieceType::Rook => {
            if mobility <= 3 {
                if let Some(king_sq) = own_king {
                    let same_rank = king_sq / 8 == sq / 8;
                    let king_file = (king_sq % 8) as i32;
                    let back_rank = r <= 1 || r >= 6;
                    let king_between = same_rank
                        && ((f <= 1 && king_file > f) || (f >= 6 && king_file < f));
                    if back_rank && king_between {
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

pub(crate) fn taper_score(mg: i32, eg: i32, phase: i32) -> i32 {
    (mg * phase + eg * (24 - phase)) / 24
}

/// Development terms (mg only); `sign` is +1 for White, -1 for Black.
fn development_score<T: Tracer>(s: &mut Score<'_, T>, sign: i32, board: &Board, color: Color) {
    let (minor_starts, king_home, king_castles) = match color {
        Color::White => (&[1u8, 2, 5, 6][..], 4u8, [2u8, 6u8]),
        Color::Black => (&[57u8, 58, 61, 62][..], 60u8, [58u8, 62u8]),
    };

    let undeveloped = minor_starts
        .iter()
        .filter(|&&sq| {
            matches!(
                board.squares[sq as usize],
                Some(Piece {
                    color: c,
                    kind: PieceType::Knight | PieceType::Bishop,
                }) if c == color
            )
        })
        .count() as i32;

    let king_sq = board.king_square(color);
    s.add_mg(P::DevUndevelopedMinor, -sign * undeveloped);
    if king_sq == Some(king_home) && undeveloped >= 2 {
        s.add_mg(P::DevKingHomeUndeveloped, -sign);
    }
    if king_sq.is_some_and(|sq| king_castles.contains(&sq)) {
        s.add_mg(P::DevKingCastled, sign);
    }
}

/// Pawn-hash entry: the full pawn bitboards are the key, so a hit is exact.
/// The all-zero entry is also correct (no pawns -> no terms, no passers).
#[derive(Clone, Copy, Default)]
pub(crate) struct PawnEntry {
    wp: u64,
    bp: u64,
    mg: i32,
    eg: i32,
    passed: u64,
}

const PAWN_TABLE_BITS: u32 = 14;

#[derive(Clone)]
pub(crate) struct PawnTable {
    entries: Vec<PawnEntry>,
}

impl PawnTable {
    pub(crate) fn new() -> Self {
        Self { entries: vec![PawnEntry::default(); 1 << PAWN_TABLE_BITS] }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.fill(PawnEntry::default());
    }

    #[inline(always)]
    fn probe(&mut self, wp: u64, bp: u64) -> &mut PawnEntry {
        let h = (wp ^ bp.rotate_left(29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        &mut self.entries[(h >> (64 - PAWN_TABLE_BITS)) as usize]
    }
}

#[inline(always)]
fn squares(mut bits: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if bits == 0 {
            return None;
        }
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        Some(sq)
    })
}

/// Terms that depend only on the pawn bitboards (doubled/isolated/passed/
/// connected/backward/candidate/center control). Returns the passed-pawn set.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pawn_structure<T: Tracer>(
    s: &mut Score<'_, T>,
    board: &Board,
    wf: &[i32; 8],
    bf: &[i32; 8],
    wm: &[u8; 8],
    bm: &[u8; 8],
    wmin: &[i8; 8],
    bmax: &[i8; 8],
) -> u64 {
    let mut passed_bb = 0u64;
    for sq in squares(board.bb.pieces[0][0]) {
        let f = (sq % 8) as usize;
        let r = (sq / 8) as i32;

        if wf[f] > 1 {
            s.add(P::Doubled, -1);
        }

        let left = if f > 0 { wf[f - 1] } else { 0 };
        let right = if f < 7 { wf[f + 1] } else { 0 };
        if left == 0 && right == 0 {
            s.add(P::Isolated, -1);
        }

        let mut passed = true;
        for file in f.saturating_sub(1)..=(f + 1).min(7) {
            if bmax[file] > r as i8 {
                passed = false;
                break;
            }
        }
        if passed {
            s.add(P::PassedBase, 1);
            s.add(P::PassedRank, r);
            passed_bb |= 1u64 << sq;
            if is_supported_by_pawn(board, sq, Color::White) {
                s.add(P::PassedSupportedBase, 1);
                s.add(P::PassedSupportedRank, r);
            }
        }

        if is_connected_pawn(wm, sq, Color::White) {
            s.add(P::Connected, 1);
        }

        if is_backward_pawn(board, wm, sq, Color::White, passed) {
            s.add(P::Backward, -1);
        }

        let candidate = bmax[f] <= r as i8;
        if candidate {
            s.add(P::CandidateBase, 1);
            s.add(P::CandidateRank, r);
            if pawn_has_lever(board, sq, Color::White) {
                s.add(P::CandidateLever, 1);
                s.add(P::CandidateLeverRank, r);
            }
            if is_supported_by_pawn(board, sq, Color::White)
                || is_connected_pawn(wm, sq, Color::White)
            {
                s.add(P::CandidateSupported, 1);
            }
        }
    }

    for sq in squares(board.bb.pieces[1][0]) {
        let f = (sq % 8) as usize;
        let r = (sq / 8) as i32;

        if bf[f] > 1 {
            s.add(P::Doubled, 1);
        }

        let left = if f > 0 { bf[f - 1] } else { 0 };
        let right = if f < 7 { bf[f + 1] } else { 0 };
        if left == 0 && right == 0 {
            s.add(P::Isolated, 1);
        }

        let mut passed = true;
        for file in f.saturating_sub(1)..=(f + 1).min(7) {
            if wmin[file] < r as i8 {
                passed = false;
                break;
            }
        }
        if passed {
            s.add(P::PassedBase, -1);
            s.add(P::PassedRank, -(7 - r));
            passed_bb |= 1u64 << sq;
            if is_supported_by_pawn(board, sq, Color::Black) {
                s.add(P::PassedSupportedBase, -1);
                s.add(P::PassedSupportedRank, -(7 - r));
            }
        }

        if is_connected_pawn(bm, sq, Color::Black) {
            s.add(P::Connected, -1);
        }

        if is_backward_pawn(board, bm, sq, Color::Black, passed) {
            s.add(P::Backward, 1);
        }

        let candidate = wmin[f] >= r as i8;
        if candidate {
            s.add(P::CandidateBase, -1);
            s.add(P::CandidateRank, -(7 - r));
            if pawn_has_lever(board, sq, Color::Black) {
                s.add(P::CandidateLever, -1);
                s.add(P::CandidateLeverRank, -(7 - r));
            }
            if is_supported_by_pawn(board, sq, Color::Black)
                || is_connected_pawn(bm, sq, Color::Black)
            {
                s.add(P::CandidateSupported, -1);
            }
        }
    }

    for sq in squares(board.bb.pieces[0][0]) {
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
    for sq in squares(board.bb.pieces[1][0]) {
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

    passed_bb
}

/// Observer of the linear structure of `evaluate`: receives, for every
/// eval_params term, the White-minus-Black coefficient that multiplies its mg
/// and eg weight (before tapering), plus the nonlinear remainder. The no-op
/// `NoTrace` is what search uses; it compiles away completely.
pub(crate) trait Tracer {
    fn add(&mut self, p: P, mg: i32, eg: i32);
    fn nonlinear(&mut self, mg: i32, eg: i32);
}

pub(crate) struct NoTrace;

impl Tracer for NoTrace {
    #[inline(always)]
    fn add(&mut self, _p: P, _mg: i32, _eg: i32) {}
    #[inline(always)]
    fn nonlinear(&mut self, _mg: i32, _eg: i32) {}
}

/// Recording tracer used by the tuner.
#[derive(Clone)]
pub(crate) struct EvalTrace {
    pub(crate) mg: [i32; PARAM_COUNT],
    pub(crate) eg: [i32; PARAM_COUNT],
    /// Nonlinear (untuned) part of the untapered mg / eg score: mop-up edge pressure.
    pub(crate) nonlinear_mg: i32,
    pub(crate) nonlinear_eg: i32,
}

impl EvalTrace {
    pub(crate) fn new() -> Self {
        Self { mg: [0; PARAM_COUNT], eg: [0; PARAM_COUNT], nonlinear_mg: 0, nonlinear_eg: 0 }
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

pub(crate) fn evaluate(board: &Board) -> i32 {
    evaluate_impl(board, &mut NoTrace, None)
}

/// `eval_for_side` with pawn-structure terms served from `pawns`; bit-identical result.
pub(crate) fn eval_for_side_cached(board: &Board, pawns: &mut PawnTable) -> i32 {
    let s = evaluate_impl(board, &mut NoTrace, Some(pawns));
    if board.side_to_move == Color::White {
        s
    } else {
        -s
    }
}

/// `evaluate`, additionally reporting its linear structure to `t`.
/// Bypasses the pawn hash so every term reaches the tracer.
pub(crate) fn evaluate_with<T: Tracer>(board: &Board, t: &mut T) -> i32 {
    evaluate_impl(board, t, None)
}

#[inline(always)]
fn evaluate_impl<T: Tracer>(board: &Board, t: &mut T, pawns: Option<&mut PawnTable>) -> i32 {
    let acc = &board.eval_acc;
    let mut s = Score { mg: acc.mg, eg: acc.eg, t };

    let mut white_pawns = SqList::new();
    let mut black_pawns = SqList::new();
    let mut white_pieces = PieceList::new();
    let mut black_pieces = PieceList::new();
    let mut white_rooks = SqList::new();
    let mut black_rooks = SqList::new();
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

    for c in 0..2usize {
        let color = if c == 0 { Color::White } else { Color::Black };
        for kind in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            let piece = Piece { color, kind };
            let mut bits = board.bb.pieces[c][crate::bitboard::kind_index(kind)];
            while bits != 0 {
                let sq = bits.trailing_zeros() as u8;
                bits &= bits - 1;
                if c == 0 {
                    white_pieces.push(sq, piece);
                } else {
                    black_pieces.push(sq, piece);
                }
                match kind {
                    PieceType::Pawn => {
                        let f = (sq % 8) as usize;
                        let rank = (sq / 8) as i8;
                        if c == 0 {
                            white_pawns.push(sq);
                            white_pawn_files[f] += 1;
                            white_pawn_masks[f] |= 1u8 << rank;
                            white_min_pawn_rank[f] = white_min_pawn_rank[f].min(rank);
                        } else {
                            black_pawns.push(sq);
                            black_pawn_files[f] += 1;
                            black_pawn_masks[f] |= 1u8 << rank;
                            black_max_pawn_rank[f] = black_max_pawn_rank[f].max(rank);
                        }
                    }
                    PieceType::Rook => {
                        if c == 0 {
                            white_rooks.push(sq);
                        } else {
                            black_rooks.push(sq);
                        }
                    }
                    _ => {}
                }
            }
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

    // Pawn-structure terms depend only on the two pawn bitboards: serve them
    // (and the passed-pawn set) from the pawn hash when one is supplied.
    let passed_bb = match pawns {
        Some(table) => {
            let wp = board.bb.pieces[0][0];
            let bp = board.bb.pieces[1][0];
            let entry = table.probe(wp, bp);
            if entry.wp != wp || entry.bp != bp {
                let mut ps = Score { mg: 0, eg: 0, t: &mut NoTrace };
                let passed = pawn_structure(
                    &mut ps,
                    board,
                    &white_pawn_files,
                    &black_pawn_files,
                    &white_pawn_masks,
                    &black_pawn_masks,
                    &white_min_pawn_rank,
                    &black_max_pawn_rank,
                );
                *entry = PawnEntry { wp, bp, mg: ps.mg, eg: ps.eg, passed };
            }
            s.mg += entry.mg;
            s.eg += entry.eg;
            entry.passed
        }
        None => pawn_structure(
            &mut s,
            board,
            &white_pawn_files,
            &black_pawn_files,
            &white_pawn_masks,
            &black_pawn_masks,
            &white_min_pawn_rank,
            &black_max_pawn_rank,
        ),
    };

    let mut bits = passed_bb & board.bb.pieces[0][0];
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        white_passed.push(sq);
        passed_pawn_support_score(&mut s, 1, sq, Color::White, white_king, black_king, white_rooks.as_slice());
    }
    let mut bits = passed_bb & board.bb.pieces[1][0];
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        black_passed.push(sq);
        passed_pawn_support_score(&mut s, -1, sq, Color::Black, black_king, white_king, black_rooks.as_slice());
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

    let phase = phase.clamp(0, 24);
    // Per-piece attack sets, shared by king safety and mobility.
    let occ = board.bb.occ_all();
    let mut white_atts = [0u64; 16];
    let mut black_atts = [0u64; 16];
    let (mut white_all, mut white_no_king, mut black_all, mut black_no_king) = (0u64, 0u64, 0u64, 0u64);
    for (i, &(sq, piece)) in white_pieces.as_slice().iter().enumerate() {
        let att = piece_attacks(sq, piece, occ);
        white_atts[i] = att;
        white_all |= att;
        if piece.kind != PieceType::King {
            white_no_king |= att;
        }
    }
    for (i, &(sq, piece)) in black_pieces.as_slice().iter().enumerate() {
        let att = piece_attacks(sq, piece, occ);
        black_atts[i] = att;
        black_all |= att;
        if piece.kind != PieceType::King {
            black_no_king |= att;
        }
    }

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

    let mut white_mob = 0;
    let mut black_mob = 0;
    for (i, &(sq, piece)) in white_pieces.as_slice().iter().enumerate() {
        let mobility = (white_atts[i] & !board.bb.occ[0]).count_ones() as i32;
        white_mob += mobility;
        let r = sq / 8;
        match piece.kind {
            PieceType::Knight => {
                if r >= 3
                    && is_outpost_square(sq, Color::White, &white_min_pawn_rank, &black_max_pawn_rank)
                {
                    s.add(P::KnightOutpost, 1);
                }
            }
            PieceType::Bishop => {
                if r >= 3
                    && is_outpost_square(sq, Color::White, &white_min_pawn_rank, &black_max_pawn_rank)
                {
                    s.add(P::BishopOutpost, 1);
                }
                s.add(P::BishopMobilityExtra, mobility);
            }
            _ => {}
        }
        trapped_piece_penalty(&mut s, 1, sq, piece, mobility, white_king);
    }
    for (i, &(sq, piece)) in black_pieces.as_slice().iter().enumerate() {
        let mobility = (black_atts[i] & !board.bb.occ[1]).count_ones() as i32;
        black_mob += mobility;
        let r = sq / 8;
        match piece.kind {
            PieceType::Knight => {
                if r <= 4
                    && is_outpost_square(sq, Color::Black, &white_min_pawn_rank, &black_max_pawn_rank)
                {
                    s.add(P::KnightOutpost, -1);
                }
            }
            PieceType::Bishop => {
                if r <= 4
                    && is_outpost_square(sq, Color::Black, &white_min_pawn_rank, &black_max_pawn_rank)
                {
                    s.add(P::BishopOutpost, -1);
                }
                s.add(P::BishopMobilityExtra, -mobility);
            }
            _ => {}
        }
        trapped_piece_penalty(&mut s, -1, sq, piece, mobility, black_king);
    }
    s.add(P::Mobility, white_mob - black_mob);

    let mut score = taper_score(s.mg, s.eg, phase);
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
        ];
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
            assert_eq!(taper_score(mg, eg, phase) + tempo, score, "{fen}");
        }
    }

    #[test]
    fn pawn_hash_matches_uncached_eval() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "8/5k2/8/3P4/8/8/5K2/8 b - - 0 1",
            "6k1/5p2/8/8/8/8/1R3PP1/r5K1 w - - 0 40",
            "4k3/8/8/8/8/8/8/1Q2K3 b - - 0 1",
            // Same pawns, different pieces: hit must still add king/rook passer terms.
            "8/5k2/8/3P4/8/8/3R1K2/8 w - - 0 1",
            "8/3k4/8/3P4/8/8/5K2/r7 b - - 0 1",
        ];
        let mut table = PawnTable::new();
        let mut board = Board::new();
        for _ in 0..2 {
            for fen in fens {
                board.set_fen(fen);
                assert_eq!(eval_for_side_cached(&board, &mut table), eval_for_side(&board), "{fen}");
            }
        }
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
