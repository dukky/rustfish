use super::*;

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
            if board.squares[sq as usize]
                == Some(Piece {
                    color,
                    kind: PieceType::Pawn,
                })
            {
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

fn is_connected_pawn(pawns: &[u8], sq: u8, color: Color) -> bool {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    for &p in pawns {
        if p == sq {
            continue;
        }
        let pf = (p % 8) as i32;
        let pr = (p / 8) as i32;
        if (pf - f).abs() == 1 {
            match color {
                Color::White => {
                    if pr == r || pr == r - 1 {
                        return true;
                    }
                }
                Color::Black => {
                    if pr == r || pr == r + 1 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn is_supported_by_pawn(board: &Board, sq: u8, color: Color) -> bool {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let support_rank = match color {
        Color::White => r - 1,
        Color::Black => r + 1,
    };
    if !(0..8).contains(&support_rank) {
        return false;
    }

    for df in [-1, 1] {
        let pf = f + df;
        if !(0..8).contains(&pf) {
            continue;
        }
        let psq = (support_rank as u8) * 8 + (pf as u8);
        if board.squares[psq as usize]
            == Some(Piece {
                color,
                kind: PieceType::Pawn,
            })
        {
            return true;
        }
    }

    false
}

fn is_backward_pawn(board: &Board, pawns: &[u8], sq: u8, color: Color, passed: bool) -> bool {
    if passed {
        return false;
    }

    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let advance_rank = match color {
        Color::White => r + 1,
        Color::Black => r - 1,
    };
    if !(0..8).contains(&advance_rank) {
        return false;
    }

    let mut has_adjacent_support = false;
    for &p in pawns {
        if p == sq {
            continue;
        }
        let pf = (p % 8) as i32;
        let pr = (p / 8) as i32;
        if (pf - f).abs() != 1 {
            continue;
        }
        match color {
            Color::White if pr <= r => {
                has_adjacent_support = true;
                break;
            }
            Color::Black if pr >= r => {
                has_adjacent_support = true;
                break;
            }
            _ => {}
        }
    }
    if has_adjacent_support {
        return false;
    }

    let enemy = color.opposite();
    for df in [-1, 1] {
        let ef = f + df;
        let er = match color {
            Color::White => advance_rank + 1,
            Color::Black => advance_rank - 1,
        };
        if !(0..8).contains(&ef) || !(0..8).contains(&er) {
            continue;
        }
        let esq = (er as u8) * 8 + (ef as u8);
        if board.squares[esq as usize]
            == Some(Piece {
                color: enemy,
                kind: PieceType::Pawn,
            })
        {
            return true;
        }
    }

    false
}

fn is_outpost_square(board: &Board, sq: u8, color: Color) -> bool {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let enemy = color.opposite();

    for df in [-1, 1] {
        let ef = f + df;
        if ef < 0 || ef >= 8 {
            continue;
        }
        match color {
            Color::White => {
                for er in (r + 1)..8 {
                    let esq = (er as u8) * 8 + (ef as u8);
                    if board.squares[esq as usize]
                        == Some(Piece {
                            color: enemy,
                            kind: PieceType::Pawn,
                        })
                    {
                        return false;
                    }
                }
            }
            Color::Black => {
                for er in (0..r).rev() {
                    let esq = (er as u8) * 8 + (ef as u8);
                    if board.squares[esq as usize]
                        == Some(Piece {
                            color: enemy,
                            kind: PieceType::Pawn,
                        })
                    {
                        return false;
                    }
                }
            }
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

fn relative_rank(color: Color, sq: u8) -> i32 {
    let r = (sq / 8) as i32;
    match color {
        Color::White => r,
        Color::Black => 7 - r,
    }
}

fn add_tapered_score(mg_score: &mut i32, eg_score: &mut i32, color: Color, mg: i32, eg: i32) {
    if color == Color::White {
        *mg_score += mg;
        *eg_score += eg;
    } else {
        *mg_score -= mg;
        *eg_score -= eg;
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

fn pst_values(kind: PieceType, color: Color, sq: u8) -> (i32, i32) {
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
    let mut mg_score = 0;
    let mut eg_score = 0;

    let mut white_pawns: Vec<u8> = Vec::new();
    let mut black_pawns: Vec<u8> = Vec::new();
    let mut white_pieces: Vec<(u8, Piece)> = Vec::new();
    let mut black_pieces: Vec<(u8, Piece)> = Vec::new();
    let mut white_rooks: Vec<u8> = Vec::new();
    let mut black_rooks: Vec<u8> = Vec::new();
    let mut white_knights: Vec<u8> = Vec::new();
    let mut black_knights: Vec<u8> = Vec::new();
    let mut white_bishop_sqs: Vec<u8> = Vec::new();
    let mut black_bishop_sqs: Vec<u8> = Vec::new();
    let mut white_bishops = 0;
    let mut black_bishops = 0;
    let mut white_pawn_files = [0i32; 8];
    let mut black_pawn_files = [0i32; 8];
    let mut white_min_pawn_rank = [8i8; 8];
    let mut black_max_pawn_rank = [-1i8; 8];
    let mut white_non_pawn_material = 0i32;
    let mut black_non_pawn_material = 0i32;
    let mut white_has_queen = false;
    let mut black_has_queen = false;
    let mut phase = 0i32;

    for sq in 0..64_usize {
        if let Some(p) = board.squares[sq] {
            if p.color == Color::White {
                white_pieces.push((sq as u8, p));
            } else {
                black_pieces.push((sq as u8, p));
            }
            let (pst_mg, pst_eg) = pst_values(p.kind, p.color, sq as u8);
            let mut mg = piece_value(p.kind) + pst_mg;
            let mut eg = piece_value(p.kind) + pst_eg;
            if p.kind == PieceType::Bishop {
                mg += 5;
                eg += 8;
                if p.color == Color::White {
                    white_bishops += 1;
                    white_bishop_sqs.push(sq as u8);
                } else {
                    black_bishops += 1;
                    black_bishop_sqs.push(sq as u8);
                }
            }
            if p.kind == PieceType::Pawn {
                let f = sq % 8;
                let rank = (sq / 8) as i8;
                if p.color == Color::White {
                    white_pawns.push(sq as u8);
                    white_pawn_files[f] += 1;
                    white_min_pawn_rank[f] = white_min_pawn_rank[f].min(rank);
                } else {
                    black_pawns.push(sq as u8);
                    black_pawn_files[f] += 1;
                    black_max_pawn_rank[f] = black_max_pawn_rank[f].max(rank);
                }
            }
            if p.kind == PieceType::Knight {
                if p.color == Color::White {
                    white_knights.push(sq as u8);
                } else {
                    black_knights.push(sq as u8);
                }
            }
            if p.kind == PieceType::Rook {
                if p.color == Color::White {
                    white_rooks.push(sq as u8);
                } else {
                    black_rooks.push(sq as u8);
                }
            }
            phase += match p.kind {
                PieceType::Knight | PieceType::Bishop => 1,
                PieceType::Rook => 2,
                PieceType::Queen => 4,
                _ => 0,
            };
            match p.kind {
                PieceType::Knight | PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
                    if p.color == Color::White {
                        white_non_pawn_material += piece_value(p.kind);
                    } else {
                        black_non_pawn_material += piece_value(p.kind);
                    }
                }
                _ => {}
            }
            if p.kind == PieceType::Queen {
                if p.color == Color::White {
                    white_has_queen = true;
                } else {
                    black_has_queen = true;
                }
            }
            add_tapered_score(&mut mg_score, &mut eg_score, p.color, mg, eg);
        }
    }

    if white_bishops >= 2 {
        mg_score += 30;
        eg_score += 40;
    }
    if black_bishops >= 2 {
        mg_score -= 30;
        eg_score -= 40;
    }

    if phase >= 12 {
        mg_score += development_score(board, Color::White);
        mg_score -= development_score(board, Color::Black);
    }

    for &sq in &white_pawns {
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
            if is_supported_by_pawn(board, sq, Color::White) {
                mg_score += 6 + r;
                eg_score += 12 + r * 3;
            }
        }

        if is_connected_pawn(&white_pawns, sq, Color::White) {
            mg_score += 8;
            eg_score += 10;
        }

        if is_backward_pawn(board, &white_pawns, sq, Color::White, passed) {
            mg_score -= 12;
            eg_score -= 6;
        }

        let candidate = black_max_pawn_rank[f] <= r as i8;
        if candidate {
            mg_score += 4 + r;
            eg_score += 8 + r * 3;
        }
    }

    for &sq in &black_pawns {
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
            if is_supported_by_pawn(board, sq, Color::Black) {
                mg_score -= 6 + (7 - r);
                eg_score -= 12 + (7 - r) * 3;
            }
        }

        if is_connected_pawn(&black_pawns, sq, Color::Black) {
            mg_score -= 8;
            eg_score -= 10;
        }

        if is_backward_pawn(board, &black_pawns, sq, Color::Black, passed) {
            mg_score += 12;
            eg_score += 6;
        }

        let candidate = white_min_pawn_rank[f] >= r as i8;
        if candidate {
            mg_score -= 4 + (7 - r);
            eg_score -= 8 + (7 - r) * 3;
        }
    }

    for &sq in &white_rooks {
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
    for &sq in &black_rooks {
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

    for &sq in &white_knights {
        let r = sq / 8;
        if r >= 3 && is_outpost_square(board, sq, Color::White) {
            mg_score += 18;
            eg_score += 8;
        }
    }
    for &sq in &black_knights {
        let r = sq / 8;
        if r <= 4 && is_outpost_square(board, sq, Color::Black) {
            mg_score -= 18;
            eg_score -= 8;
        }
    }
    for &sq in &white_bishop_sqs {
        let r = sq / 8;
        if r >= 3 && is_outpost_square(board, sq, Color::White) {
            mg_score += 10;
            eg_score += 6;
        }
        let scope = mobility_count(
            board,
            sq,
            Piece {
                color: Color::White,
                kind: PieceType::Bishop,
            },
        );
        mg_score += scope * 2;
        eg_score += scope;
    }
    for &sq in &black_bishop_sqs {
        let r = sq / 8;
        if r <= 4 && is_outpost_square(board, sq, Color::Black) {
            mg_score -= 10;
            eg_score -= 6;
        }
        let scope = mobility_count(
            board,
            sq,
            Piece {
                color: Color::Black,
                kind: PieceType::Bishop,
            },
        );
        mg_score -= scope * 2;
        eg_score -= scope;
    }

    for &sq in &white_pawns {
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
    for &sq in &black_pawns {
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
        &black_pieces,
    ) - king_safety(
        board,
        Color::Black,
        &black_pawn_files,
        &white_pawn_files,
        white_non_pawn_material,
        white_has_queen,
        &white_pieces,
    );
    mg_score += king_safety_score;
    eg_score += king_safety_score / 6;

    let white_king = board.king_square(Color::White);
    let black_king = board.king_square(Color::Black);

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

    let mut white_mob = 0;
    let mut black_mob = 0;
    for &(sq, piece) in &white_pieces {
        white_mob += mobility_count(board, sq, piece);
    }
    for &(sq, piece) in &black_pieces {
        black_mob += mobility_count(board, sq, piece);
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
