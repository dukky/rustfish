use super::*;

pub(crate) struct BookEntry {
    pub(crate) key: u64,
    pub(crate) moves: &'static [&'static str],
}

pub(crate) fn build_opening_book(zobrist: &Zobrist) -> Vec<BookEntry> {
    let lines: &[(&[&str], &[&str])] = &[
        (&[], &["e2e4", "d2d4", "g1f3", "c2c4"]),
        (&["e2e4"], &["e7e5", "c7c5", "e7e6", "c7c6", "g8f6"]),
        (&["d2d4"], &["d7d5", "g8f6", "e7e6", "f7f5"]),
        (&["c2c4"], &["e7e5", "g8f6", "c7c5", "e7e6"]),
        (&["g1f3"], &["d7d5", "g8f6", "c7c5", "e7e6"]),
        (&["e2e4", "e7e5"], &["g1f3", "f1c4", "d2d4", "b1c3"]),
        (&["e2e4", "c7c5"], &["g1f3", "c2c3", "d2d4", "b1c3"]),
        (&["e2e4", "e7e6"], &["d2d4", "g1f3"]),
        (&["e2e4", "c7c6"], &["d2d4", "b1c3"]),
        (&["e2e4", "g8f6"], &["e4e5", "b1c3"]),
        (&["d2d4", "d7d5"], &["c2c4", "g1f3", "e2e3"]),
        (&["d2d4", "g8f6"], &["c2c4", "g1f3", "e2e3"]),
        (&["d2d4", "f7f5"], &["g2g3", "c2c4"]),
        (&["c2c4", "e7e5"], &["g2g3", "b1c3", "g1f3"]),
        (&["c2c4", "g8f6"], &["b1c3", "g2g3", "g1f3"]),
        (&["g1f3", "d7d5"], &["d2d4", "c2c4", "g2g3"]),
        (&["g1f3", "g8f6"], &["c2c4", "d2d4", "g2g3"]),
        (&["e2e4", "e7e5", "g1f3"], &["b8c6", "g8f6", "d7d6"]),
        (&["e2e4", "e7e5", "f1c4"], &["g8f6", "b8c6"]),
        (&["e2e4", "c7c5", "g1f3"], &["d7d6", "b8c6", "e7e6"]),
        (&["d2d4", "d7d5", "c2c4"], &["e7e6", "c7c6", "d5c4"]),
        (&["d2d4", "g8f6", "c2c4"], &["e7e6", "g7g6", "c7c5"]),
    ];

    let mut book = Vec::new();
    for (prefix, replies) in lines {
        let mut board = Board::new();
        board.set_startpos();
        let mut ok = true;
        for mv_str in *prefix {
            if let Some(mv) = parse_uci_move(&mut board, mv_str) {
                let _ = board.make_move(mv);
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            book.push(BookEntry {
                key: board_hash(&board, zobrist),
                moves: replies,
            });
        }
    }
    book
}

pub(crate) fn try_book_move(board: &mut Board, engine: &EngineState) -> Option<String> {
    let key = board_hash(board, &engine.zobrist);
    let entry = engine.opening_book.iter().find(|e| e.key == key)?;
    let mut legal_book_moves = Vec::new();
    for mv_str in entry.moves {
        if let Some(mv) = parse_uci_move(board, mv_str) {
            legal_book_moves.push(move_to_uci(mv));
        }
    }
    if legal_book_moves.is_empty() {
        None
    } else {
        let idx = (key as usize) % legal_book_moves.len();
        Some(legal_book_moves[idx].clone())
    }
}
