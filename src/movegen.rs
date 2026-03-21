use super::*;

pub(crate) fn generate_pseudo_moves(board: &Board, captures_only: bool, moves: &mut Vec<Move>) {
    moves.clear();

    let c = color_index(board.side_to_move);
    for pi in 0..board.piece_cnt[c] {
        let sq = board.piece_sqs[c][pi];
        let piece = board.squares[sq as usize].unwrap();

        let f = (sq % 8) as i8;
        let r = (sq / 8) as i8;

        match piece.kind {
            PieceType::Pawn => {
                let dir: i8 = if piece.color == Color::White { 1 } else { -1 };
                let start_rank: i8 = if piece.color == Color::White { 1 } else { 6 };
                let promo_rank: i8 = if piece.color == Color::White { 6 } else { 1 };

                if !captures_only {
                    let one_r = r + dir;
                    if in_bounds(f, one_r) {
                        let to = (one_r as u8) * 8 + (f as u8);
                        if board.squares[to as usize].is_none() {
                            if r == promo_rank {
                                for promo in [
                                    PieceType::Queen,
                                    PieceType::Rook,
                                    PieceType::Bishop,
                                    PieceType::Knight,
                                ] {
                                    moves.push(Move {
                                        from: sq,
                                        to,
                                        promotion: Some(promo),
                                        is_en_passant: false,
                                        is_castle: false,
                                    });
                                }
                            } else {
                                moves.push(Move {
                                    from: sq,
                                    to,
                                    promotion: None,
                                    is_en_passant: false,
                                    is_castle: false,
                                });
                                if r == start_rank {
                                    let two_r = r + dir * 2;
                                    let to2 = (two_r as u8) * 8 + (f as u8);
                                    if board.squares[to2 as usize].is_none() {
                                        moves.push(Move {
                                            from: sq,
                                            to: to2,
                                            promotion: None,
                                            is_en_passant: false,
                                            is_castle: false,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }

                for df in [-1, 1] {
                    let nf = f + df;
                    let nr = r + dir;
                    if !in_bounds(nf, nr) {
                        continue;
                    }
                    let to = (nr as u8) * 8 + (nf as u8);
                    let target = board.squares[to as usize];
                    if let Some(tp) = target {
                        if tp.color != piece.color {
                            if r == promo_rank {
                                for promo in [
                                    PieceType::Queen,
                                    PieceType::Rook,
                                    PieceType::Bishop,
                                    PieceType::Knight,
                                ] {
                                    moves.push(Move {
                                        from: sq,
                                        to,
                                        promotion: Some(promo),
                                        is_en_passant: false,
                                        is_castle: false,
                                    });
                                }
                            } else {
                                moves.push(Move {
                                    from: sq,
                                    to,
                                    promotion: None,
                                    is_en_passant: false,
                                    is_castle: false,
                                });
                            }
                        }
                    } else if let Some(ep) = board.en_passant {
                        if ep == to {
                            moves.push(Move {
                                from: sq,
                                to,
                                promotion: None,
                                is_en_passant: true,
                                is_castle: false,
                            });
                        }
                    }
                }
            }
            PieceType::Knight => {
                for (df, dr) in [
                    (1, 2),
                    (2, 1),
                    (2, -1),
                    (1, -2),
                    (-1, -2),
                    (-2, -1),
                    (-2, 1),
                    (-1, 2),
                ] {
                    let nf = f + df;
                    let nr = r + dr;
                    if !in_bounds(nf, nr) {
                        continue;
                    }
                    let to = (nr as u8) * 8 + (nf as u8);
                    match board.squares[to as usize] {
                        Some(tp) if tp.color == piece.color => {}
                        Some(_) => moves.push(Move {
                            from: sq,
                            to,
                            promotion: None,
                            is_en_passant: false,
                            is_castle: false,
                        }),
                        None if !captures_only => moves.push(Move {
                            from: sq,
                            to,
                            promotion: None,
                            is_en_passant: false,
                            is_castle: false,
                        }),
                        None => {}
                    }
                }
            }
            PieceType::Bishop => {
                gen_sliders(board, sq, piece, captures_only, &[(1, 1), (1, -1), (-1, 1), (-1, -1)], moves);
            }
            PieceType::Rook => {
                gen_sliders(board, sq, piece, captures_only, &[(1, 0), (-1, 0), (0, 1), (0, -1)], moves);
            }
            PieceType::Queen => {
                gen_sliders(
                    board,
                    sq,
                    piece,
                    captures_only,
                    &[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
                    moves,
                );
            }
            PieceType::King => {
                for df in -1..=1 {
                    for dr in -1..=1 {
                        if df == 0 && dr == 0 {
                            continue;
                        }
                        let nf = f + df;
                        let nr = r + dr;
                        if !in_bounds(nf, nr) {
                            continue;
                        }
                        let to = (nr as u8) * 8 + (nf as u8);
                        match board.squares[to as usize] {
                            Some(tp) if tp.color == piece.color => {}
                            Some(_) => moves.push(Move {
                                from: sq,
                                to,
                                promotion: None,
                                is_en_passant: false,
                                is_castle: false,
                            }),
                            None if !captures_only => moves.push(Move {
                                from: sq,
                                to,
                                promotion: None,
                                is_en_passant: false,
                                is_castle: false,
                            }),
                            None => {}
                        }
                    }
                }

                if !captures_only {
                    if piece.color == Color::White && sq == 4 {
                        if board.castling & 1 != 0
                            && board.squares[5].is_none()
                            && board.squares[6].is_none()
                            && !is_square_attacked(board, 4, Color::Black)
                            && !is_square_attacked(board, 5, Color::Black)
                            && !is_square_attacked(board, 6, Color::Black)
                        {
                            moves.push(Move {
                                from: 4,
                                to: 6,
                                promotion: None,
                                is_en_passant: false,
                                is_castle: true,
                            });
                        }
                        if board.castling & 2 != 0
                            && board.squares[3].is_none()
                            && board.squares[2].is_none()
                            && board.squares[1].is_none()
                            && !is_square_attacked(board, 4, Color::Black)
                            && !is_square_attacked(board, 3, Color::Black)
                            && !is_square_attacked(board, 2, Color::Black)
                        {
                            moves.push(Move {
                                from: 4,
                                to: 2,
                                promotion: None,
                                is_en_passant: false,
                                is_castle: true,
                            });
                        }
                    }
                    if piece.color == Color::Black && sq == 60 {
                        if board.castling & 4 != 0
                            && board.squares[61].is_none()
                            && board.squares[62].is_none()
                            && !is_square_attacked(board, 60, Color::White)
                            && !is_square_attacked(board, 61, Color::White)
                            && !is_square_attacked(board, 62, Color::White)
                        {
                            moves.push(Move {
                                from: 60,
                                to: 62,
                                promotion: None,
                                is_en_passant: false,
                                is_castle: true,
                            });
                        }
                        if board.castling & 8 != 0
                            && board.squares[59].is_none()
                            && board.squares[58].is_none()
                            && board.squares[57].is_none()
                            && !is_square_attacked(board, 60, Color::White)
                            && !is_square_attacked(board, 59, Color::White)
                            && !is_square_attacked(board, 58, Color::White)
                        {
                            moves.push(Move {
                                from: 60,
                                to: 58,
                                promotion: None,
                                is_en_passant: false,
                                is_castle: true,
                            });
                        }
                    }
                }
            }
        }
    }

}

pub(crate) fn mobility_count(board: &Board, sq: u8, piece: Piece) -> i32 {
    let f = (sq % 8) as i8;
    let r = (sq / 8) as i8;
    let mut count = 0i32;

    match piece.kind {
        PieceType::Pawn => {
            let dir: i8 = if piece.color == Color::White { 1 } else { -1 };
            for df in [-1, 1] {
                let nf = f + df;
                let nr = r + dir;
                if in_bounds(nf, nr) {
                    let to = (nr as u8) * 8 + (nf as u8);
                    if board.squares[to as usize]
                        .map(|p| p.color != piece.color)
                        .unwrap_or(true)
                    {
                        count += 1;
                    }
                }
            }
        }
        PieceType::Knight => {
            for (df, dr) in [
                (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
            ] {
                let nf = f + df;
                let nr = r + dr;
                if in_bounds(nf, nr) {
                    let to = (nr as u8) * 8 + (nf as u8);
                    if board.squares[to as usize]
                        .map(|p| p.color != piece.color)
                        .unwrap_or(true)
                    {
                        count += 1;
                    }
                }
            }
        }
        PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
            let dirs: &[(i8, i8)] = match piece.kind {
                PieceType::Bishop => &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
                PieceType::Rook => &[(1, 0), (-1, 0), (0, 1), (0, -1)],
                _ => &[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
            };
            for (df, dr) in dirs {
                let mut nf = f + *df;
                let mut nr = r + *dr;
                while in_bounds(nf, nr) {
                    let to = (nr as u8) * 8 + (nf as u8);
                    if let Some(p) = board.squares[to as usize] {
                        if p.color != piece.color {
                            count += 1;
                        }
                        break;
                    }
                    count += 1;
                    nf += *df;
                    nr += *dr;
                }
            }
        }
        PieceType::King => {
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let nf = f + df;
                    let nr = r + dr;
                    if in_bounds(nf, nr) {
                        let to = (nr as u8) * 8 + (nf as u8);
                        if board.squares[to as usize]
                            .map(|p| p.color != piece.color)
                            .unwrap_or(true)
                        {
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    count
}

fn gen_sliders(
    board: &Board,
    sq: u8,
    piece: Piece,
    captures_only: bool,
    dirs: &[(i8, i8)],
    out: &mut Vec<Move>,
) {
    let f = (sq % 8) as i8;
    let r = (sq / 8) as i8;

    for (df, dr) in dirs {
        let mut nf = f + *df;
        let mut nr = r + *dr;
        while in_bounds(nf, nr) {
            let to = (nr as u8) * 8 + (nf as u8);
            if let Some(tp) = board.squares[to as usize] {
                if tp.color != piece.color {
                    out.push(Move {
                        from: sq,
                        to,
                        promotion: None,
                        is_en_passant: false,
                        is_castle: false,
                    });
                }
                break;
            } else if !captures_only {
                out.push(Move {
                    from: sq,
                    to,
                    promotion: None,
                    is_en_passant: false,
                    is_castle: false,
                });
            }
            nf += *df;
            nr += *dr;
        }
    }
}

pub(crate) fn is_square_attacked(board: &Board, sq: u8, by: Color) -> bool {
    let f = (sq % 8) as i8;
    let r = (sq / 8) as i8;

    match by {
        Color::White => {
            for (df, dr) in [(-1, -1), (1, -1)] {
                let nf = f + df;
                let nr = r + dr;
                if in_bounds(nf, nr) {
                    let from = (nr as u8) * 8 + (nf as u8);
                    if board.squares[from as usize]
                        == Some(Piece {
                            color: Color::White,
                            kind: PieceType::Pawn,
                        })
                    {
                        return true;
                    }
                }
            }
        }
        Color::Black => {
            for (df, dr) in [(-1, 1), (1, 1)] {
                let nf = f + df;
                let nr = r + dr;
                if in_bounds(nf, nr) {
                    let from = (nr as u8) * 8 + (nf as u8);
                    if board.squares[from as usize]
                        == Some(Piece {
                            color: Color::Black,
                            kind: PieceType::Pawn,
                        })
                    {
                        return true;
                    }
                }
            }
        }
    }

    for (df, dr) in [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ] {
        let nf = f + df;
        let nr = r + dr;
        if in_bounds(nf, nr) {
            let from = (nr as u8) * 8 + (nf as u8);
            if board.squares[from as usize]
                == Some(Piece {
                    color: by,
                    kind: PieceType::Knight,
                })
            {
                return true;
            }
        }
    }

    for (df, dr) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] {
        let mut nf = f + df;
        let mut nr = r + dr;
        while in_bounds(nf, nr) {
            let from = (nr as u8) * 8 + (nf as u8);
            if let Some(p) = board.squares[from as usize] {
                if p.color == by && (p.kind == PieceType::Bishop || p.kind == PieceType::Queen) {
                    return true;
                }
                break;
            }
            nf += df;
            nr += dr;
        }
    }

    for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let mut nf = f + df;
        let mut nr = r + dr;
        while in_bounds(nf, nr) {
            let from = (nr as u8) * 8 + (nf as u8);
            if let Some(p) = board.squares[from as usize] {
                if p.color == by && (p.kind == PieceType::Rook || p.kind == PieceType::Queen) {
                    return true;
                }
                break;
            }
            nf += df;
            nr += dr;
        }
    }

    for df in -1..=1 {
        for dr in -1..=1 {
            if df == 0 && dr == 0 {
                continue;
            }
            let nf = f + df;
            let nr = r + dr;
            if in_bounds(nf, nr) {
                let from = (nr as u8) * 8 + (nf as u8);
                if board.squares[from as usize]
                    == Some(Piece {
                        color: by,
                        kind: PieceType::King,
                    })
                {
                    return true;
                }
            }
        }
    }

    false
}

pub(crate) fn parse_uci_move(board: &mut Board, s: &str) -> Option<Move> {
    if s.len() < 4 {
        return None;
    }
    let from = parse_square(&s[0..2])?;
    let to = parse_square(&s[2..4])?;
    let promotion = if s.len() >= 5 {
        match s.chars().nth(4)?.to_ascii_lowercase() {
            'q' => Some(PieceType::Queen),
            'r' => Some(PieceType::Rook),
            'b' => Some(PieceType::Bishop),
            'n' => Some(PieceType::Knight),
            _ => None,
        }
    } else {
        None
    };

    let mut legal = Vec::new();
    board.generate_legal_moves(false, &mut legal);
    legal.into_iter().find(|m| m.from == from && m.to == to && m.promotion == promotion)
}

pub(crate) fn parse_square(s: &str) -> Option<u8> {
    if s.len() != 2 {
        return None;
    }
    let bytes = s.as_bytes();
    let file = bytes[0].to_ascii_lowercase();
    let rank = bytes[1];
    if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
        return None;
    }
    let f = file - b'a';
    let r = rank - b'1';
    Some(r * 8 + f)
}

pub(crate) fn move_to_uci(mv: Move) -> String {
    let mut out = format!("{}{}", square_to_string(mv.from), square_to_string(mv.to));
    if let Some(p) = mv.promotion {
        out.push(match p {
            PieceType::Queen => 'q',
            PieceType::Rook => 'r',
            PieceType::Bishop => 'b',
            PieceType::Knight => 'n',
            _ => 'q',
        });
    }
    out
}

pub(crate) fn square_to_string(sq: u8) -> String {
    let f = (sq % 8) as u8;
    let r = (sq / 8) as u8;
    let file = (b'a' + f) as char;
    let rank = (b'1' + r) as char;
    format!("{}{}", file, rank)
}

pub(crate) fn in_bounds(file: i8, rank: i8) -> bool {
    file >= 0 && file < 8 && rank >= 0 && rank < 8
}
