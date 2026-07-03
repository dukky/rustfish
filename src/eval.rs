use super::*;

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

fn attack_unit_weight(kind: PieceType) -> i32 {
    match kind {
        PieceType::Pawn => 1,
        PieceType::Knight | PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 5,
        PieceType::King => 0,
    }
}

fn king_attack_units(
    board: &Board,
    attacker_pieces: &[(u8, Piece)],
    attacker: Color,
    king_sq: u8,
) -> i32 {
    let mut units = 0;
    let kf = (king_sq % 8) as i8;
    let kr = (king_sq / 8) as i8;

    for &(from, piece) in attacker_pieces {
        if piece.color != attacker || piece.kind == PieceType::King {
            continue;
        }

        // Geometric pre-cull: skip pieces that cannot reach any square of the
        // 3x3 king ring even on an empty board, so the expensive per-target
        // ray checks below run only for plausible attackers. The conditions
        // are necessary (blockers only remove attacks), so results are
        // unchanged.
        let ff = (from % 8) as i8;
        let fr = (from / 8) as i8;
        let reachable = match piece.kind {
            PieceType::Pawn => (ff - kf).abs() <= 2 && (fr - kr).abs() <= 2,
            PieceType::Knight => (ff - kf).abs() <= 3 && (fr - kr).abs() <= 3,
            PieceType::Rook => (ff - kf).abs() <= 1 || (fr - kr).abs() <= 1,
            PieceType::Bishop => {
                ((ff - fr) - (kf - kr)).abs() <= 2 || ((ff + fr) - (kf + kr)).abs() <= 2
            }
            PieceType::Queen => {
                (ff - kf).abs() <= 1
                    || (fr - kr).abs() <= 1
                    || ((ff - fr) - (kf - kr)).abs() <= 2
                    || ((ff + fr) - (kf + kr)).abs() <= 2
            }
            PieceType::King => false,
        };
        if !reachable {
            continue;
        }

        let mut hits = 0;
        for df in -1..=1 {
            for dr in -1..=1 {
                let nf = kf + df;
                let nr = kr + dr;
                if !in_bounds(nf, nr) {
                    continue;
                }
                let target = (nr as u8) * 8 + (nf as u8);
                if piece_attacks_square(&board.squares, from, piece, target) {
                    hits += 1;
                }
            }
        }
        if hits > 0 {
            units += attack_unit_weight(piece.kind) * hits;
        }
    }

    units
}

fn king_tropism_from_pieces(attacker_pieces: &[(u8, Piece)], attacker: Color, king_sq: u8) -> i32 {
    let mut score = 0;

    for &(from, piece) in attacker_pieces {
        if piece.color != attacker {
            continue;
        }

        let weight = match piece.kind {
            PieceType::Knight | PieceType::Bishop => 2,
            PieceType::Rook => 2,
            PieceType::Queen => 3,
            PieceType::Pawn | PieceType::King => 0,
        };
        if weight == 0 {
            continue;
        }

        let closeness = (7 - king_distance(from, king_sq)).max(0);
        score += closeness * weight;
    }

    score
}

fn king_attack_scale(enemy_non_pawn_material: i32, enemy_has_queen: bool) -> i32 {
    let mut scale = 12 + enemy_non_pawn_material.min(3200) / 200;
    if enemy_has_queen {
        scale += 8;
    } else {
        scale -= 2;
    }
    scale.max(6)
}

fn king_safety(
    board: &Board,
    color: Color,
    own_pawn_files: &[i32; 8],
    enemy_pawn_files: &[i32; 8],
    enemy_non_pawn_material: i32,
    enemy_has_queen: bool,
    enemy_pieces: &[(u8, Piece)],
) -> i32 {
    let Some(ksq) = board.king_square(color) else {
        return 0;
    };

    let mut score = 0;
    let kf = (ksq % 8) as i32;
    let kr = (ksq / 8) as i32;

    let shield_rank = if color == Color::White { kr + 1 } else { kr - 1 };
    if shield_rank >= 0 && shield_rank < 8 {
        for df in -1..=1 {
            let f = kf + df;
            if f < 0 || f >= 8 {
                continue;
            }
            let sq = (shield_rank as u8) * 8 + (f as u8);
            if board.bb.pieces[color_index(color)][0] & (1u64 << sq) != 0 {
                score += 10;
            } else {
                score -= 8;
            }
        }
    }

    for df in -1..=1 {
        let f = kf + df;
        if f < 0 || f >= 8 {
            continue;
        }
        let fi = f as usize;
        if own_pawn_files[fi] == 0 {
            if enemy_pawn_files[fi] == 0 {
                score -= 14;
            } else {
                score -= 22;
            }
        }
    }

    let enemy = color.opposite();
    let mut attacked = 0;
    let kf = (ksq % 8) as i8;
    let kr = (ksq / 8) as i8;
    for df in -1..=1 {
        for dr in -1..=1 {
            let nf = kf + df;
            let nr = kr + dr;
            if !in_bounds(nf, nr) {
                continue;
            }
            let sq = (nr as u8) * 8 + (nf as u8);
            if is_square_attacked(board, sq, enemy) {
                attacked += 1;
            }
        }
    }
    score -= attacked * 6;

    let attack_units = king_attack_units(board, enemy_pieces, enemy, ksq);
    let attack_scale = king_attack_scale(enemy_non_pawn_material, enemy_has_queen);
    score -= attack_units * attack_scale / 10;

    let tropism = king_tropism_from_pieces(enemy_pieces, enemy, ksq);
    score -= tropism * attack_scale / 16;

    score
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

fn passed_pawn_support_score(
    sq: u8,
    color: Color,
    own_king: Option<u8>,
    enemy_king: Option<u8>,
    own_rooks: &[u8],
) -> (i32, i32) {
    let mut mg = 0;
    let mut eg = 0;

    if let Some(king) = own_king {
        let dist = king_distance(king, sq);
        if dist <= 2 {
            let bonus = 3 - dist;
            mg += bonus * 3;
            eg += bonus * 8;
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
            eg += (promo_dist - remaining) * 2;
        }
        if king_distance(king, sq) <= 1 {
            eg -= 10;
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
            mg += 4;
            eg += 10;
        } else {
            eg -= 4;
        }
    }

    (mg, eg)
}

fn trapped_piece_penalty(
    sq: u8,
    piece: Piece,
    mobility: i32,
    own_king: Option<u8>,
) -> (i32, i32) {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let on_edge = f == 0 || f == 7 || r == 0 || r == 7;
    let in_corner = (f == 0 || f == 7) && (r == 0 || r == 7);
    let mut mg = 0;
    let mut eg = 0;

    match piece.kind {
        PieceType::Knight => {
            if in_corner && mobility <= 2 {
                let severity = 3 - mobility;
                mg += severity * 12;
                eg += severity * 6;
            } else if on_edge && mobility <= 2 {
                let severity = 3 - mobility;
                mg += severity * 6;
                eg += severity * 3;
            }
        }
        PieceType::Bishop => {
            if on_edge && mobility <= 2 {
                let severity = 3 - mobility;
                mg += severity * 10;
                eg += severity * 5;
                if in_corner {
                    mg += 6;
                    eg += 2;
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
                        let severity = 4 - mobility;
                        mg += severity * 8;
                        eg += severity * 3;
                    }
                }
            }
        }
        _ => {}
    }

    (mg.max(0), eg.max(0))
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

fn king_centralization(sq: u8) -> i32 {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let df = (f - 3).abs().min((f - 4).abs());
    let dr = (r - 3).abs().min((r - 4).abs());
    14 - (df + dr) * 4
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

fn rook_endgame_bonus(
    color: Color,
    rooks: &[u8],
    own_passed: &[u8],
    enemy_passed: &[u8],
    enemy_king: Option<u8>,
) -> i32 {
    let mut score = 0;

    for &rook_sq in rooks {
        if let Some(king_sq) = enemy_king {
            score += (7 - king_distance(rook_sq, king_sq)).max(0) * 2;
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
                score += 18;
            } else {
                score -= 8;
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
                score += 14;
            }
        }
    }

    score
}

fn relative_rank(color: Color, sq: u8) -> i32 {
    let r = (sq / 8) as i32;
    match color {
        Color::White => r,
        Color::Black => 7 - r,
    }
}

fn taper_score(mg: i32, eg: i32, phase: i32) -> i32 {
    (mg * phase + eg * (24 - phase)) / 24
}

fn mg_pst_value(kind: PieceType, color: Color, sq: u8) -> i32 {
    let idx = if color == Color::White {
        sq as usize
    } else {
        (sq ^ 56) as usize
    };

    const PAWN: [i32; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const KNIGHT: [i32; 64] = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ];
    const BISHOP: [i32; 64] = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ];
    const ROOK: [i32; 64] = [
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const QUEEN: [i32; 64] = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ];
    const KING: [i32; 64] = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20,
    ];

    match kind {
        PieceType::Pawn => PAWN[idx],
        PieceType::Knight => KNIGHT[idx],
        PieceType::Bishop => BISHOP[idx],
        PieceType::Rook => ROOK[idx],
        PieceType::Queen => QUEEN[idx],
        PieceType::King => KING[idx],
    }
}

fn eg_pst_value(kind: PieceType, color: Color, sq: u8) -> i32 {
    let idx = if color == Color::White {
        sq
    } else {
        sq ^ 56
    };
    let f = (idx % 8) as i32;
    let r = (idx / 8) as i32;
    let center_file = (f - 3).abs().min((f - 4).abs());
    let center_rank = (r - 3).abs().min((r - 4).abs());
    let center = center_file + center_rank;

    match kind {
        PieceType::Pawn => r * 10 - center_file * 2,
        PieceType::Knight => 28 - center * 8,
        PieceType::Bishop => 20 - center * 5,
        PieceType::Rook => r * 4 - center_file * 2,
        PieceType::Queen => 12 - center * 4,
        PieceType::King => king_centralization(idx) * 2,
    }
}

pub(crate) fn pst_values(kind: PieceType, color: Color, sq: u8) -> (i32, i32) {
    (
        mg_pst_value(kind, color, sq),
        eg_pst_value(kind, color, sq),
    )
}

fn development_score(board: &Board, color: Color) -> i32 {
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
    let mut score = -undeveloped * 5;
    if king_sq == Some(king_home) && undeveloped >= 2 {
        score -= 8;
    }
    if king_sq.is_some_and(|sq| king_castles.contains(&sq)) {
        score += 8;
    }
    score
}

pub(crate) fn evaluate(board: &Board) -> i32 {
    let acc = &board.eval_acc;
    let mut mg_score = acc.mg;
    let mut eg_score = acc.eg;

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
        mg_score += 30;
        eg_score += 40;
    }
    if acc.bishops[1] >= 2 {
        mg_score -= 30;
        eg_score -= 40;
    }

    if phase >= 12 {
        mg_score += development_score(board, Color::White);
        mg_score -= development_score(board, Color::Black);
    }

    let white_king = board.king_square(Color::White);
    let black_king = board.king_square(Color::Black);

    for &sq in white_pawns.as_slice() {
        let f = (sq % 8) as usize;
        let r = (sq / 8) as i32;

        if white_pawn_files[f] > 1 {
            mg_score -= 12;
            eg_score -= 16;
        }

        let left = if f > 0 { white_pawn_files[f - 1] } else { 0 };
        let right = if f < 7 { white_pawn_files[f + 1] } else { 0 };
        if left == 0 && right == 0 {
            mg_score -= 14;
            eg_score -= 12;
        }

        let mut passed = true;
        for file in f.saturating_sub(1)..=(f + 1).min(7) {
            if black_max_pawn_rank[file] > r as i8 {
                passed = false;
                break;
            }
        }
        if passed {
            mg_score += 12 + r * 4;
            eg_score += 28 + r * 12;
            white_passed.push(sq);
            if is_supported_by_pawn(board, sq, Color::White) {
                mg_score += 6 + r;
                eg_score += 12 + r * 3;
            }
            let (support_mg, support_eg) = passed_pawn_support_score(
                sq,
                Color::White,
                white_king,
                black_king,
                white_rooks.as_slice(),
            );
            mg_score += support_mg;
            eg_score += support_eg;
        }

        if is_connected_pawn(&white_pawn_masks, sq, Color::White) {
            mg_score += 8;
            eg_score += 10;
        }

        if is_backward_pawn(board, &white_pawn_masks, sq, Color::White, passed) {
            mg_score -= 12;
            eg_score -= 6;
        }

        let candidate = black_max_pawn_rank[f] <= r as i8;
        if candidate {
            let lever = pawn_has_lever(board, sq, Color::White);
            mg_score += 4 + r + if lever { 4 } else { 0 };
            eg_score += 8 + r * 3 + if lever { 6 + r } else { 0 };
            if is_supported_by_pawn(board, sq, Color::White)
                || is_connected_pawn(&white_pawn_masks, sq, Color::White)
            {
                mg_score += 2;
                eg_score += 4;
            }
        }
    }

    for &sq in black_pawns.as_slice() {
        let f = (sq % 8) as usize;
        let r = (sq / 8) as i32;

        if black_pawn_files[f] > 1 {
            mg_score += 12;
            eg_score += 16;
        }

        let left = if f > 0 { black_pawn_files[f - 1] } else { 0 };
        let right = if f < 7 { black_pawn_files[f + 1] } else { 0 };
        if left == 0 && right == 0 {
            mg_score += 14;
            eg_score += 12;
        }

        let mut passed = true;
        for file in f.saturating_sub(1)..=(f + 1).min(7) {
            if white_min_pawn_rank[file] < r as i8 {
                passed = false;
                break;
            }
        }
        if passed {
            mg_score -= 12 + (7 - r) * 4;
            eg_score -= 28 + (7 - r) * 12;
            black_passed.push(sq);
            if is_supported_by_pawn(board, sq, Color::Black) {
                mg_score -= 6 + (7 - r);
                eg_score -= 12 + (7 - r) * 3;
            }
            let (support_mg, support_eg) = passed_pawn_support_score(
                sq,
                Color::Black,
                black_king,
                white_king,
                black_rooks.as_slice(),
            );
            mg_score -= support_mg;
            eg_score -= support_eg;
        }

        if is_connected_pawn(&black_pawn_masks, sq, Color::Black) {
            mg_score -= 8;
            eg_score -= 10;
        }

        if is_backward_pawn(board, &black_pawn_masks, sq, Color::Black, passed) {
            mg_score += 12;
            eg_score += 6;
        }

        let candidate = white_min_pawn_rank[f] >= r as i8;
        if candidate {
            let lever = pawn_has_lever(board, sq, Color::Black);
            mg_score -= 4 + (7 - r) + if lever { 4 } else { 0 };
            eg_score -= 8 + (7 - r) * 3 + if lever { 6 + (7 - r) } else { 0 };
            if is_supported_by_pawn(board, sq, Color::Black)
                || is_connected_pawn(&black_pawn_masks, sq, Color::Black)
            {
                mg_score -= 2;
                eg_score -= 4;
            }
        }
    }

    for &sq in white_rooks.as_slice() {
        let f = (sq % 8) as usize;
        let r = sq / 8;
        if white_pawn_files[f] == 0 {
            mg_score += if black_pawn_files[f] == 0 { 20 } else { 10 };
            eg_score += if black_pawn_files[f] == 0 { 14 } else { 8 };
        }
        if r == 6 {
            mg_score += 16;
            eg_score += 24;
        }
    }
    for &sq in black_rooks.as_slice() {
        let f = (sq % 8) as usize;
        let r = sq / 8;
        if black_pawn_files[f] == 0 {
            mg_score -= if white_pawn_files[f] == 0 { 20 } else { 10 };
            eg_score -= if white_pawn_files[f] == 0 { 14 } else { 8 };
        }
        if r == 1 {
            mg_score -= 16;
            eg_score -= 24;
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
                mg_score += 4;
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
                mg_score -= 4;
            }
        }
    }

    let phase = phase.clamp(0, 24);
    let king_safety_score = king_safety(
        board,
        Color::White,
        &white_pawn_files,
        &black_pawn_files,
        black_non_pawn_material,
        black_has_queen,
        black_pieces.as_slice(),
    ) - king_safety(
        board,
        Color::Black,
        &black_pawn_files,
        &white_pawn_files,
        white_non_pawn_material,
        white_has_queen,
        white_pieces.as_slice(),
    );
    mg_score += king_safety_score;
    eg_score += king_safety_score / 6;

    if phase <= 8 {
        let material_edge = white_non_pawn_material - black_non_pawn_material;
        if material_edge >= 500 {
            if let (Some(wk), Some(bk)) = (white_king, black_king) {
                eg_score += king_edge_pressure(bk);
                eg_score += (14 - king_distance(wk, bk)) * 3;
            }
        } else if material_edge <= -500 {
            if let (Some(wk), Some(bk)) = (white_king, black_king) {
                eg_score -= king_edge_pressure(wk);
                eg_score -= (14 - king_distance(wk, bk)) * 3;
            }
        }
    }

    if white_non_pawn_material == 0 && black_non_pawn_material == 0 {
        if let (Some(wk), Some(bk)) = (white_king, black_king) {
            if has_direct_opposition(wk, bk)
                && (!white_pawns.as_slice().is_empty() || !black_pawns.as_slice().is_empty())
            {
                if board.side_to_move == Color::Black {
                    eg_score += 14;
                } else {
                    eg_score -= 14;
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

        match (white_race, black_race) {
            (Some(w), None) => eg_score += 90 + (8 - w).max(0) * 6,
            (None, Some(b)) => eg_score -= 90 + (8 - b).max(0) * 6,
            (Some(w), Some(b)) if w + 1 < b => eg_score += 40 + (b - w) * 8,
            (Some(w), Some(b)) if b + 1 < w => eg_score -= 40 + (w - b) * 8,
            _ => {}
        }
    }

    if white_rooks.as_slice().len() == 1
        && black_rooks.as_slice().len() == 1
        && !white_has_queen
        && !black_has_queen
    {
        eg_score += rook_endgame_bonus(
            Color::White,
            white_rooks.as_slice(),
            white_passed.as_slice(),
            black_passed.as_slice(),
            black_king,
        );
        eg_score -= rook_endgame_bonus(
            Color::Black,
            black_rooks.as_slice(),
            black_passed.as_slice(),
            white_passed.as_slice(),
            white_king,
        );
    }

    let mut white_mob = 0;
    let mut black_mob = 0;
    for &(sq, piece) in white_pieces.as_slice() {
        let mobility = mobility_count(board, sq, piece);
        white_mob += mobility;
        let r = sq / 8;
        match piece.kind {
            PieceType::Knight => {
                if r >= 3
                    && is_outpost_square(sq, Color::White, &white_min_pawn_rank, &black_max_pawn_rank)
                {
                    mg_score += 18;
                    eg_score += 8;
                }
            }
            PieceType::Bishop => {
                if r >= 3
                    && is_outpost_square(sq, Color::White, &white_min_pawn_rank, &black_max_pawn_rank)
                {
                    mg_score += 10;
                    eg_score += 6;
                }
                mg_score += mobility * 2;
                eg_score += mobility;
            }
            _ => {}
        }
        let (trap_mg, trap_eg) = trapped_piece_penalty(sq, piece, mobility, white_king);
        mg_score -= trap_mg;
        eg_score -= trap_eg;
    }
    for &(sq, piece) in black_pieces.as_slice() {
        let mobility = mobility_count(board, sq, piece);
        black_mob += mobility;
        let r = sq / 8;
        match piece.kind {
            PieceType::Knight => {
                if r <= 4
                    && is_outpost_square(sq, Color::Black, &white_min_pawn_rank, &black_max_pawn_rank)
                {
                    mg_score -= 18;
                    eg_score -= 8;
                }
            }
            PieceType::Bishop => {
                if r <= 4
                    && is_outpost_square(sq, Color::Black, &white_min_pawn_rank, &black_max_pawn_rank)
                {
                    mg_score -= 10;
                    eg_score -= 6;
                }
                mg_score -= mobility * 2;
                eg_score -= mobility;
            }
            _ => {}
        }
        let (trap_mg, trap_eg) = trapped_piece_penalty(sq, piece, mobility, black_king);
        mg_score += trap_mg;
        eg_score += trap_eg;
    }
    let mobility_delta = white_mob - black_mob;
    mg_score += mobility_delta * 2;
    eg_score += mobility_delta;

    let mut score = taper_score(mg_score, eg_score, phase);
    if board.side_to_move == Color::White {
        score += 10;
    } else {
        score -= 10;
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
        let trapped = trapped_piece_penalty(
            parse_square("a1").unwrap(),
            Piece {
                color: Color::White,
                kind: PieceType::Knight,
            },
            1,
            None,
        );
        let free = trapped_piece_penalty(
            parse_square("d4").unwrap(),
            Piece {
                color: Color::White,
                kind: PieceType::Knight,
            },
            6,
            None,
        );
        assert!(trapped.0 > free.0);
        assert!(trapped.1 > free.1);
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
