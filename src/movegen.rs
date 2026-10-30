use super::*;

pub(crate) fn generate_pseudo_moves(board: &Board, captures_only: bool, moves: &mut Vec<Move>) {
    use crate::bitboard::{
        bishop_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKERS,
    };

    moves.clear();

    let us = board.side_to_move;
    let c = color_index(us);
    let them_idx = 1 - c;
    let bb = &board.bb;
    let own = bb.occ[c];
    let them = bb.occ[them_idx];
    let occ = own | them;
    let target_mask = if captures_only { them } else { !own };

    let push_move = |moves: &mut Vec<Move>, from: u8, to: u8| {
        moves.push(Move {
            from,
            to,
            promotion: None,
            is_en_passant: false,
            is_castle: false,
        });
    };
    let push_promos = |moves: &mut Vec<Move>, from: u8, to: u8| {
        for promo in [
            PieceType::Queen,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Knight,
        ] {
            moves.push(Move {
                from,
                to,
                promotion: Some(promo),
                is_en_passant: false,
                is_castle: false,
            });
        }
    };

    let ep_bit = board.en_passant.map(|e| 1u64 << e).unwrap_or(0);
    let promo_from_rank: u8 = if us == Color::White { 6 } else { 1 };
    let start_rank: u8 = if us == Color::White { 1 } else { 6 };

    let mut pawns = bb.pieces[c][0];
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        let r = sq / 8;
        let is_promo = r == promo_from_rank;

        if !captures_only {
            let one = if us == Color::White { sq + 8 } else { sq - 8 };
            if occ & (1u64 << one) == 0 {
                if is_promo {
                    push_promos(moves, sq, one);
                } else {
                    push_move(moves, sq, one);
                    if r == start_rank {
                        let two = if us == Color::White { sq + 16 } else { sq - 16 };
                        if occ & (1u64 << two) == 0 {
                            push_move(moves, sq, two);
                        }
                    }
                }
            }
        }

        // A pawn's attack squares equal the enemy pawn-attacker set of its
        // own square (mirrored table).
        let attacks = PAWN_ATTACKERS[them_idx][sq as usize];
        let mut caps = attacks & them;
        while caps != 0 {
            let to = caps.trailing_zeros() as u8;
            caps &= caps - 1;
            if is_promo {
                push_promos(moves, sq, to);
            } else {
                push_move(moves, sq, to);
            }
        }
        if attacks & ep_bit != 0 {
            moves.push(Move {
                from: sq,
                to: ep_bit.trailing_zeros() as u8,
                promotion: None,
                is_en_passant: true,
                is_castle: false,
            });
        }
    }

    let mut knights = bb.pieces[c][1];
    while knights != 0 {
        let sq = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        let mut targets = KNIGHT_ATTACKS[sq as usize] & target_mask;
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            push_move(moves, sq, to);
        }
    }

    for kind in [PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let mut sliders = bb.pieces[c][crate::bitboard::kind_index(kind)];
        while sliders != 0 {
            let sq = sliders.trailing_zeros() as u8;
            sliders &= sliders - 1;
            let attacks = match kind {
                PieceType::Bishop => bishop_attacks(sq as usize, occ),
                PieceType::Rook => rook_attacks(sq as usize, occ),
                _ => bishop_attacks(sq as usize, occ) | rook_attacks(sq as usize, occ),
            };
            let mut targets = attacks & target_mask;
            while targets != 0 {
                let to = targets.trailing_zeros() as u8;
                targets &= targets - 1;
                push_move(moves, sq, to);
            }
        }
    }

    let kings = bb.pieces[c][5];
    if kings != 0 {
        let sq = kings.trailing_zeros() as u8;
        let mut targets = KING_ATTACKS[sq as usize] & target_mask;
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            push_move(moves, sq, to);
        }

        if !captures_only {
            if us == Color::White && sq == 4 {
                if board.castling & 1 != 0
                    && board.squares[7]
                        == Some(Piece {
                            color: Color::White,
                            kind: PieceType::Rook,
                        })
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
                    && board.squares[0]
                        == Some(Piece {
                            color: Color::White,
                            kind: PieceType::Rook,
                        })
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
            if us == Color::Black && sq == 60 {
                if board.castling & 4 != 0
                    && board.squares[63]
                        == Some(Piece {
                            color: Color::Black,
                            kind: PieceType::Rook,
                        })
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
                    && board.squares[56]
                        == Some(Piece {
                            color: Color::Black,
                            kind: PieceType::Rook,
                        })
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

pub(crate) fn mobility_count(board: &Board, sq: u8, piece: Piece) -> i32 {
    use crate::bitboard::{
        bishop_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKERS,
    };

    let s = sq as usize;
    let occ = board.bb.occ_all();
    let attacks = match piece.kind {
        // A pawn's attack squares equal the squares an enemy pawn would
        // attack this square from, so the attacker table works mirrored.
        PieceType::Pawn => PAWN_ATTACKERS[color_index(piece.color.opposite())][s],
        PieceType::Knight => KNIGHT_ATTACKS[s],
        PieceType::Bishop => bishop_attacks(s, occ),
        PieceType::Rook => rook_attacks(s, occ),
        PieceType::Queen => bishop_attacks(s, occ) | rook_attacks(s, occ),
        PieceType::King => KING_ATTACKS[s],
    };
    (attacks & !board.bb.occ[color_index(piece.color)]).count_ones() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn castling_requires_rook_on_rook_square() {
        let mut board = Board::new();
        let mut moves = Vec::new();

        board.set_fen("4k3/8/8/8/8/8/8/4K3 w K - 0 1");
        board.generate_legal_moves(false, &mut moves);
        assert!(moves.iter().all(|mv| !mv.is_castle));

        board.set_fen("4k3/8/8/8/8/8/8/4K3 b k - 0 1");
        board.generate_legal_moves(false, &mut moves);
        assert!(moves.iter().all(|mv| !mv.is_castle));
    }
}
pub(crate) fn is_square_attacked(board: &Board, sq: u8, by: Color) -> bool {
    is_square_attacked_occ(board, sq, by, board.bb.occ_all())
}

/// `is_square_attacked` with slider rays traced over `occ` instead of the
/// board's real occupancy (e.g. with the moving king lifted off).
#[inline(always)]
pub(crate) fn is_square_attacked_occ(board: &Board, sq: u8, by: Color, occ: u64) -> bool {
    use crate::bitboard::{
        bishop_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKERS,
    };

    let c = color_index(by);
    let bb = &board.bb;
    let s = sq as usize;

    if PAWN_ATTACKERS[c][s] & bb.pieces[c][0] != 0 {
        return true;
    }
    if KNIGHT_ATTACKS[s] & bb.pieces[c][1] != 0 {
        return true;
    }
    if KING_ATTACKS[s] & bb.pieces[c][5] != 0 {
        return true;
    }

    let diag_sliders = bb.pieces[c][2] | bb.pieces[c][4];
    if diag_sliders != 0 && bishop_attacks(s, occ) & diag_sliders != 0 {
        return true;
    }
    let ortho_sliders = bb.pieces[c][3] | bb.pieces[c][4];
    if ortho_sliders != 0 && rook_attacks(s, occ) & ortho_sliders != 0 {
        return true;
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
