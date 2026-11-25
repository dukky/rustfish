use super::*;

#[cfg(test)]
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

/// Upper bound on the legal moves of a position (218 is the known maximum).
const MAX_LEGAL_MOVES: usize = 256;

/// Append-only writer into a `Vec<Move>`'s reserved spare capacity, without
/// the per-push capacity check; the generator commits the length at the end.
struct MoveSink {
    ptr: *mut Move,
    len: usize,
}

impl MoveSink {
    #[inline(always)]
    fn push(&mut self, mv: Move) {
        assert!(self.len < MAX_LEGAL_MOVES);
        // SAFETY: the buffer has capacity for MAX_LEGAL_MOVES moves (reserved
        // by the generator) and len < MAX_LEGAL_MOVES.
        unsafe { self.ptr.add(self.len).write(mv) };
        self.len += 1;
    }
}

/// Legal move generation: the same moves, in the same order, as
/// `generate_pseudo_moves` followed by the legality filter, with the
/// check/pin masks applied per piece instead of per move. En passant
/// legality is decided by the caller (`ep_legal_from`: pawns whose en
/// passant capture is legal); king destinations are tested with the king
/// lifted off the board.
pub(crate) fn generate_legal_moves_fused(
    board: &Board,
    captures_only: bool,
    st: &LegalState,
    ep_legal_from: u64,
    moves: &mut Vec<Move>,
) {
    use crate::bitboard::{
        bishop_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKERS,
    };

    moves.clear();
    moves.reserve(MAX_LEGAL_MOVES);
    let mut sink = MoveSink { ptr: moves.as_mut_ptr(), len: 0 };

    let us = board.side_to_move;
    let c = color_index(us);
    let them_idx = 1 - c;
    let bb = &board.bb;
    let own = bb.occ[c];
    let them = bb.occ[them_idx];
    let occ = own | them;
    let target_mask = if captures_only { them } else { !own };

    let push_move = |out: &mut MoveSink, from: u8, to: u8| {
        out.push(Move {
            from,
            to,
            promotion: None,
            is_en_passant: false,
            is_castle: false,
        });
    };
    let push_promos = |out: &mut MoveSink, from: u8, to: u8| {
        for promo in [
            PieceType::Queen,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Knight,
        ] {
            out.push(Move {
                from,
                to,
                promotion: Some(promo),
                is_en_passant: false,
                is_castle: false,
            });
        }
    };

    // Double check: only king moves (and en passant, which the caller has
    // already tested via make/undo) can be legal.
    let non_king = st.check_count < 2;

    let ep_bit = board.en_passant.map(|e| 1u64 << e).unwrap_or(0);
    let promo_from_rank: u8 = if us == Color::White { 6 } else { 1 };
    let start_rank: u8 = if us == Color::White { 1 } else { 6 };

    let mut pawns = bb.pieces[c][0];
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        let r = sq / 8;
        let is_promo = r == promo_from_rank;
        let allowed = if non_king { st.allowed(sq) } else { 0 };

        if !captures_only {
            let one = if us == Color::White { sq + 8 } else { sq - 8 };
            if occ & (1u64 << one) == 0 {
                if is_promo {
                    if allowed & (1u64 << one) != 0 {
                        push_promos(&mut sink, sq, one);
                    }
                } else {
                    if allowed & (1u64 << one) != 0 {
                        push_move(&mut sink, sq, one);
                    }
                    if r == start_rank {
                        let two = if us == Color::White { sq + 16 } else { sq - 16 };
                        if occ & (1u64 << two) == 0 && allowed & (1u64 << two) != 0 {
                            push_move(&mut sink, sq, two);
                        }
                    }
                }
            }
        }

        let attacks = PAWN_ATTACKERS[them_idx][sq as usize];
        let mut caps = attacks & them & allowed;
        while caps != 0 {
            let to = caps.trailing_zeros() as u8;
            caps &= caps - 1;
            if is_promo {
                push_promos(&mut sink, sq, to);
            } else {
                push_move(&mut sink, sq, to);
            }
        }
        if attacks & ep_bit != 0 && ep_legal_from & (1u64 << sq) != 0 {
            sink.push(Move {
                from: sq,
                to: ep_bit.trailing_zeros() as u8,
                promotion: None,
                is_en_passant: true,
                is_castle: false,
            });
        }
    }

    if non_king {
        // A pinned knight never has a legal move.
        let mut knights = bb.pieces[c][1] & !st.pinned;
        while knights != 0 {
            let sq = knights.trailing_zeros() as u8;
            knights &= knights - 1;
            let mut targets = KNIGHT_ATTACKS[sq as usize] & target_mask & st.evasion_mask;
            while targets != 0 {
                let to = targets.trailing_zeros() as u8;
                targets &= targets - 1;
                push_move(&mut sink, sq, to);
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
                let mut targets = attacks & target_mask & st.allowed(sq);
                while targets != 0 {
                    let to = targets.trailing_zeros() as u8;
                    targets &= targets - 1;
                    push_move(&mut sink, sq, to);
                }
            }
        }
    }

    let kings = bb.pieces[c][5];
    if kings != 0 {
        let sq = kings.trailing_zeros() as u8;
        let enemy = us.opposite();
        // Sliders x-ray through the king's origin square.
        let occ_no_king = occ & !(1u64 << sq);
        let mut targets = KING_ATTACKS[sq as usize] & target_mask;
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            if !is_square_attacked_occ(board, to, enemy, occ_no_king) {
                push_move(&mut sink, sq, to);
            }
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
                    && !is_square_attacked_occ(board, 6, enemy, occ_no_king)
                {
                    sink.push(Move {
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
                    && !is_square_attacked_occ(board, 2, enemy, occ_no_king)
                {
                    sink.push(Move {
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
                    && !is_square_attacked_occ(board, 62, enemy, occ_no_king)
                {
                    sink.push(Move {
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
                    && !is_square_attacked_occ(board, 58, enemy, occ_no_king)
                {
                    sink.push(Move {
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
    // SAFETY: the sink wrote `sink.len` initialised moves into the reserved buffer.
    unsafe { moves.set_len(sink.len) };
}

/// Squares attacked by `piece` standing on `sq`, given occupancy `occ`.
#[inline(always)]
pub(crate) fn piece_attacks(sq: u8, piece: Piece, occ: u64) -> u64 {
    use crate::bitboard::{
        bishop_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKERS,
    };

    let s = sq as usize;
    match piece.kind {
        // A pawn's attack squares equal the squares an enemy pawn would
        // attack this square from, so the attacker table works mirrored.
        PieceType::Pawn => PAWN_ATTACKERS[color_index(piece.color.opposite())][s],
        PieceType::Knight => KNIGHT_ATTACKS[s],
        PieceType::Bishop => bishop_attacks(s, occ),
        PieceType::Rook => rook_attacks(s, occ),
        PieceType::Queen => bishop_attacks(s, occ) | rook_attacks(s, occ),
        PieceType::King => KING_ATTACKS[s],
    }
}

/// Squares attacked by one side, accumulated by attacker value: `pawn` holds
/// pawn attacks, `minor` adds knights and bishops, `rook` adds rooks, `all`
/// adds queens and the king.
#[derive(Copy, Clone, Default)]
pub(crate) struct ThreatMaps {
    pub(crate) pawn: u64,
    pub(crate) minor: u64,
    pub(crate) rook: u64,
    pub(crate) all: u64,
}

impl ThreatMaps {
    /// Squares attacked by an enemy piece worth less than `kind` (empty for
    /// pawns and the king).
    #[inline(always)]
    pub(crate) fn by_lesser(&self, kind: PieceType) -> u64 {
        match kind {
            PieceType::Knight | PieceType::Bishop => self.pawn,
            PieceType::Rook => self.minor,
            PieceType::Queen => self.rook,
            PieceType::Pawn | PieceType::King => 0,
        }
    }
}

pub(crate) fn threat_maps(board: &Board, by: Color) -> ThreatMaps {
    use crate::bitboard::{bishop_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS};

    const NOT_A_FILE: u64 = 0xfefe_fefe_fefe_fefe;
    const NOT_H_FILE: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    let c = color_index(by);
    let bb = &board.bb;
    let occ = bb.occ_all();
    let p = bb.pieces[c];

    let pawn = if by == Color::White {
        ((p[0] << 7) & NOT_H_FILE) | ((p[0] << 9) & NOT_A_FILE)
    } else {
        ((p[0] >> 9) & NOT_H_FILE) | ((p[0] >> 7) & NOT_A_FILE)
    };

    let mut minor = pawn;
    let mut b = p[1];
    while b != 0 {
        minor |= KNIGHT_ATTACKS[b.trailing_zeros() as usize];
        b &= b - 1;
    }
    let mut b = p[2];
    while b != 0 {
        minor |= bishop_attacks(b.trailing_zeros() as usize, occ);
        b &= b - 1;
    }

    let mut rook = minor;
    let mut b = p[3];
    while b != 0 {
        rook |= rook_attacks(b.trailing_zeros() as usize, occ);
        b &= b - 1;
    }

    let mut all = rook;
    let mut b = p[4];
    while b != 0 {
        let s = b.trailing_zeros() as usize;
        all |= bishop_attacks(s, occ) | rook_attacks(s, occ);
        b &= b - 1;
    }
    if p[5] != 0 {
        all |= KING_ATTACKS[p[5].trailing_zeros() as usize];
    }

    ThreatMaps { pawn, minor, rook, all }
}

/// Per-node data for a gives-check test without make_move (side to move =
/// the checking side): the squares from which each piece kind would attack
/// the enemy king, and our pieces whose departure would uncover a check by
/// one of our sliders (discovered-check candidates).
pub(crate) struct CheckInfo {
    king: u8,
    /// Indexed by `kind_index`: pawn, knight, bishop, rook, queen, king (0).
    check_squares: [u64; 6],
    discoverers: u64,
    diag: u64,
    ortho: u64,
}

impl CheckInfo {
    pub(crate) fn new(board: &Board) -> Self {
        use crate::bitboard::{bishop_attacks, rook_attacks, KNIGHT_ATTACKS, PAWN_ATTACKERS};

        let c = color_index(board.side_to_move);
        let bb = &board.bb;
        let p = &bb.pieces[c];
        let diag = p[2] | p[4];
        let ortho = p[3] | p[4];
        let Some(king) = board.king_square(board.side_to_move.opposite()) else {
            return Self { king: 0, check_squares: [0; 6], discoverers: 0, diag, ortho };
        };
        let k = king as usize;
        let occ = bb.occ_all();
        let (b_att, r_att) = (bishop_attacks(k, occ), rook_attacks(k, occ));
        let check_squares = [PAWN_ATTACKERS[c][k], KNIGHT_ATTACKS[k], b_att, r_att, b_att | r_att, 0];

        // Our sliders on an empty-board line with the king; a lone own piece
        // between one of them and the king is a discovered-check candidate.
        let king_bit = 1u64 << king;
        let mut snipers = (bishop_attacks(k, 0) & diag) | (rook_attacks(k, 0) & ortho);
        let mut discoverers = 0u64;
        while snipers != 0 {
            let s = snipers.trailing_zeros() as usize;
            snipers &= snipers - 1;
            let s_bit = 1u64 << s;
            let between = if bishop_attacks(k, 0) & s_bit != 0 {
                bishop_attacks(k, s_bit) & bishop_attacks(s, king_bit)
            } else {
                rook_attacks(k, s_bit) & rook_attacks(s, king_bit)
            };
            let blockers = between & occ;
            if blockers.count_ones() == 1 && blockers & bb.occ[c] != 0 {
                discoverers |= blockers;
            }
        }
        Self { king, check_squares, discoverers, diag, ortho }
    }

    /// Whether `mv` (a legal move of the side to move) gives check. Exact for
    /// every move kind; ordinary moves cost a table lookup or two, and only
    /// discovered-check candidates, promotions, castling and en passant
    /// trace slider attacks over the post-move occupancy.
    #[inline]
    pub(crate) fn gives_check(&self, board: &Board, mv: Move) -> bool {
        use crate::bitboard::{bishop_attacks, rook_attacks};

        let Some(piece) = board.squares[mv.from as usize] else {
            return false;
        };
        let k = self.king as usize;
        let (from_bit, to_bit) = (1u64 << mv.from, 1u64 << mv.to);
        let occ = board.bb.occ_all();

        if mv.is_castle {
            let (rook_from, rook_to) = match mv.to {
                6 => (7u8, 5u8),
                2 => (0, 3),
                62 => (63, 61),
                _ => (56, 59),
            };
            let (rf, rt) = (1u64 << rook_from, 1u64 << rook_to);
            let occ2 = occ ^ from_bit ^ to_bit ^ rf ^ rt;
            let ortho = (self.ortho & !rf) | rt;
            return rook_attacks(k, occ2) & ortho != 0 || bishop_attacks(k, occ2) & self.diag != 0;
        }
        if mv.is_en_passant {
            if self.check_squares[0] & to_bit != 0 {
                return true;
            }
            let cap = if board.side_to_move == Color::White { mv.to - 8 } else { mv.to + 8 };
            let occ2 = (occ ^ from_bit ^ (1u64 << cap)) | to_bit;
            return rook_attacks(k, occ2) & self.ortho != 0 || bishop_attacks(k, occ2) & self.diag != 0;
        }

        if let Some(promo) = mv.promotion {
            // The vacated from-square may lie between `to` and the king.
            let promoted = Piece { color: piece.color, kind: promo };
            if piece_attacks(mv.to, promoted, occ ^ from_bit) & (1u64 << self.king) != 0 {
                return true;
            }
        } else if self.check_squares[crate::bitboard::kind_index(piece.kind)] & to_bit != 0 {
            return true;
        }

        if self.discoverers & from_bit != 0 {
            let occ2 = (occ ^ from_bit) | to_bit;
            return rook_attacks(k, occ2) & self.ortho & !from_bit != 0
                || bishop_attacks(k, occ2) & self.diag & !from_bit != 0;
        }
        false
    }
}

impl Board {
    /// Whether `mv` (typically a TT move, which may come from another
    /// position) is one of the legal moves `generate_legal_moves(false, ..)`
    /// would produce here, flags included, decided without generating.
    /// Castling and en passant always answer false (the caller then falls
    /// back to full generation), so a true answer is always exact.
    pub(crate) fn tt_move_legal(&self, mv: Move) -> bool {
        use crate::bitboard::{KING_ATTACKS, PAWN_ATTACKERS};
        if mv.is_castle || mv.is_en_passant || mv.from >= 64 || mv.to >= 64 || mv.from == mv.to {
            return false;
        }
        let us = self.side_to_move;
        let Some(piece) = self.squares[mv.from as usize] else { return false };
        if piece.color != us {
            return false;
        }
        let c = color_index(us);
        let bb = &self.bb;
        let own = bb.occ[c];
        let them = bb.occ[1 - c];
        let occ = own | them;
        let to_bit = 1u64 << mv.to;
        if own & to_bit != 0 {
            return false;
        }
        if matches!(self.squares[mv.to as usize], Some(p) if p.kind == PieceType::King) {
            return false;
        }
        let st = self.legal_state(us);
        if piece.kind == PieceType::King {
            if mv.promotion.is_some() || KING_ATTACKS[mv.from as usize] & to_bit == 0 {
                return false;
            }
            return !is_square_attacked_occ(self, mv.to, us.opposite(), occ & !(1u64 << mv.from));
        }
        if st.check_count >= 2 || st.allowed(mv.from) & to_bit == 0 {
            return false;
        }
        if piece.kind == PieceType::Pawn {
            let r = mv.from / 8;
            let promo_rank = if us == Color::White { 6 } else { 1 };
            let start_rank = if us == Color::White { 1 } else { 6 };
            let is_promo = r == promo_rank;
            match mv.promotion {
                None if is_promo => return false,
                Some(_) if !is_promo => return false,
                Some(PieceType::Pawn | PieceType::King) => return false,
                _ => {}
            }
            let one = if us == Color::White { mv.from + 8 } else { mv.from.wrapping_sub(8) };
            if mv.to == one {
                return occ & to_bit == 0;
            }
            if r == start_rank {
                let two = if us == Color::White { mv.from + 16 } else { mv.from - 16 };
                if mv.to == two {
                    return occ & (1u64 << one) == 0 && occ & to_bit == 0;
                }
            }
            return PAWN_ATTACKERS[1 - c][mv.from as usize] & to_bit & them != 0;
        }
        if mv.promotion.is_some() {
            return false;
        }
        piece_attacks(mv.from, piece, occ) & to_bit != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tt_move_legal` is exact: true only for generated legal moves, and
    /// true for every generated move but castling / en passant.
    #[test]
    fn tt_move_legal_matches_generation() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        ];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let promos = [None, Some(PieceType::Queen), Some(PieceType::Rook), Some(PieceType::Bishop), Some(PieceType::Knight)];
        for fen in fens {
            let mut board = Board::new();
            board.set_fen(fen);
            for _ in 0..60 {
                let mut legal = Vec::new();
                board.generate_legal_moves(false, &mut legal);
                if legal.is_empty() {
                    break;
                }
                for &mv in &legal {
                    assert_eq!(board.tt_move_legal(mv), !mv.is_castle && !mv.is_en_passant, "{fen} {mv:?}");
                }
                for from in 0..64u8 {
                    for to in 0..64u8 {
                        for promotion in promos {
                            let mv = Move { from, to, promotion, is_en_passant: false, is_castle: false };
                            if board.tt_move_legal(mv) {
                                assert!(legal.contains(&mv), "{fen} {mv:?} accepted but not legal");
                            }
                        }
                    }
                }
                let mv = legal[(rnd() % legal.len() as u64) as usize];
                board.make_move(mv);
            }
        }
    }

    #[test]
    fn gives_check_matches_make_move() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/1P4P1/8/2pPp3/8/8/1p4p1/R3K2R w KQkq e6 0 1",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
            "8/2k5/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1",
        ];
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut legal = Vec::new();
        let (mut checks, mut kinds) = (0u32, [0u32; 4]);
        for fen in fens {
            for _ in 0..40 {
                let mut board = Board::new();
                board.set_fen(fen);
                for _ in 0..60 {
                    board.generate_legal_moves(false, &mut legal);
                    if legal.is_empty() {
                        break;
                    }
                    let info = CheckInfo::new(&board);
                    for mv in legal.clone() {
                        let fast = info.gives_check(&board, mv);
                        let undo = board.make_move(mv);
                        let slow = board.is_in_check(board.side_to_move);
                        board.undo_move(mv, undo);
                        assert_eq!(fast, slow, "{mv:?} in a line from {fen}");
                        if slow {
                            checks += 1;
                            let disc = info.discoverers >> mv.from & 1 == 1;
                            let i = if mv.is_castle {
                                0
                            } else if mv.is_en_passant {
                                1
                            } else if mv.promotion.is_some() {
                                2
                            } else {
                                3 * disc as usize
                            };
                            kinds[i] += 1;
                        }
                    }
                    let r = crate::tt::splitmix64(&mut rng);
                    let mv = legal[(r >> 8) as usize % legal.len()];
                    board.make_move(mv);
                }
            }
        }
        assert!(checks > 1000 && kinds.iter().all(|&n| n > 0), "{checks} {kinds:?}");
    }

    #[test]
    fn threat_maps_match_square_attacks() {
        let mut board = Board::new();
        for fen in [
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            board.set_fen(fen);
            for by in [Color::White, Color::Black] {
                let t = threat_maps(&board, by);
                for sq in 0..64u8 {
                    assert_eq!(
                        t.all >> sq & 1 == 1,
                        is_square_attacked(&board, sq, by),
                        "{fen} sq {sq}"
                    );
                }
                assert_eq!(t.pawn & !t.minor, 0);
                assert_eq!(t.minor & !t.rook, 0);
                assert_eq!(t.rook & !t.all, 0);
            }
        }
    }

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
