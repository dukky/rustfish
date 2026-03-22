use super::*;
use std::fs;

#[derive(Clone)]
pub(crate) struct BookEntry {
    pub(crate) key: u64,
    pub(crate) moves: Vec<String>,
}

fn book_lines(style: BookStyle) -> &'static [(&'static [&'static str], &'static [&'static str])] {
    match style {
        BookStyle::Normal => &[
        (&[], &["e2e4", "d2d4", "g1f3", "c2c4"]),
        (&["e2e4"], &["e7e5", "e7e6", "c7c5", "c7c6", "g8f6"]),
        (&["d2d4"], &["d7d5", "e7e6", "f7f5", "g8f6"]),
        (&["c2c4"], &["e7e5", "c7c5", "e7e6", "g8f6"]),
        (&["g1f3"], &["d7d5", "c7c5", "g8f6", "e7e6"]),
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
        (&["g1f3", "g8f6", "c2c4"], &["e7e6", "g7g6", "c7c5"]),
        (&["g1f3", "g8f6", "c2c4", "e7e6"], &["b1c3", "g2g3"]),
        (&["g1f3", "g8f6", "c2c4", "e7e6", "b1c3"], &["d7d5", "f8b4"]),
        (&["g1f3", "g8f6", "c2c4", "e7e6", "b1c3", "d7d5"], &["d2d4", "e2e3"]),
        (&["g1f3", "d7d5", "d2d4"], &["g8f6", "e7e6"]),
        (&["g1f3", "d7d5", "d2d4", "g8f6"], &["c2c4", "e2e3"]),
        (&["g1f3", "d7d5", "d2d4", "g8f6", "c2c4"], &["e7e6", "c7c6"]),
        (&["g1f3", "c7c5"], &["c2c4", "e2e4"]),
        (&["g1f3", "c7c5", "c2c4"], &["b8c6", "g8f6"]),
        (&["g1f3", "c7c5", "c2c4", "b8c6"], &["b1c3", "g2g3"]),
        (&["e2e4", "c7c5", "g1f3", "d7d6"], &["d2d4", "b1c3"]),
        (&["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"], &["c5d4", "g8f6"]),
        (&["d2d4", "g8f6", "c2c4", "e7e6"], &["b1c3", "g1f3"]),
        (&["d2d4", "g8f6", "c2c4", "e7e6", "b1c3"], &["f8b4", "d7d5"]),
        (&["c2c4", "g8f6", "b1c3"], &["e7e6", "c7c5"]),
        (&["c2c4", "g8f6", "b1c3", "e7e6"], &["g1f3", "d2d4"]),
        (&["c2c4", "g8f6", "b1c3", "e7e6", "g1f3"], &["d7d5", "f8b4"]),
        ],
        BookStyle::Grinder => &[
        (&[], &["d2d4", "g1f3", "e2e4", "d2d4"]),
        (&["e2e4"], &["e7e5", "e7e5", "c7c6", "e7e6"]),
        (&["d2d4"], &["d7d5", "d7d5", "g8f6", "e7e6"]),
        (&["c2c4"], &["e7e6", "g8f6", "c7c5"]),
        (&["g1f3"], &["d7d5", "g8f6", "e7e6"]),
        (&["e2e4", "e7e5"], &["g1f3", "f1c4", "f1b5", "g1f3"]),
        (&["e2e4", "c7c5"], &["g1f3", "c2c3", "b1c3"]),
        (&["e2e4", "e7e6"], &["d2d4", "b1c3"]),
        (&["e2e4", "c7c6"], &["d2d4", "b1c3"]),
        (&["d2d4", "d7d5"], &["c2c4", "g1f3", "e2e3", "c2c4"]),
        (&["d2d4", "g8f6"], &["c2c4", "g1f3", "e2e3"]),
        (&["c2c4", "e7e5"], &["g2g3", "b1c3", "g1f3"]),
        (&["c2c4", "g8f6"], &["g1f3", "g2g3", "b1c3"]),
        (&["g1f3", "d7d5"], &["d2d4", "c2c4", "g2g3"]),
        (&["g1f3", "g8f6"], &["d2d4", "c2c4", "g2g3"]),
        (&["e2e4", "e7e5", "g1f3"], &["b8c6", "g8f6", "d7d6"]),
        (&["e2e4", "e7e5", "g1f3", "b8c6"], &["f1b5", "f1c4", "d2d4"]),
        (&["e2e4", "c7c5", "g1f3"], &["d7d6", "b8c6", "e7e6"]),
        (&["e2e4", "c7c5", "g1f3", "d7d6"], &["d2d4", "b1c3", "c2c3"]),
        (&["d2d4", "d7d5", "c2c4"], &["e7e6", "c7c6", "g8f6"]),
        (&["d2d4", "g8f6", "c2c4"], &["e7e6", "g7g6", "c7c5"]),
        (&["c2c4", "g8f6", "b1c3"], &["e7e6", "g7g6", "c7c5"]),
        ],
        BookStyle::Chaos => &[
        (&[], &["e2e4", "c2c4", "f2f4", "b2b4"]),
        (&["e2e4"], &["c7c5", "g8f6", "e7e5", "c7c6"]),
        (&["d2d4"], &["f7f5", "g8f6", "c7c5"]),
        (&["c2c4"], &["c7c5", "e7e5", "g8f6"]),
        (&["g1f3"], &["c7c5", "d7d5", "g8f6"]),
        (&["f2f4"], &["d7d5", "g8f6", "e7e5"]),
        (&["b2b4"], &["e7e5", "d7d5", "g8f6"]),
        (&["e2e4", "c7c5"], &["f2f4", "b1c3", "g1f3"]),
        (&["e2e4", "e7e5"], &["f2f4", "f1c4", "b1c3"]),
        (&["e2e4", "g8f6"], &["e4e5", "b1c3", "f2f4"]),
        (&["d2d4", "f7f5"], &["g2g3", "e2e4", "c2c4"]),
        (&["d2d4", "g8f6"], &["c2c4", "f2f3", "g2g4"]),
        (&["c2c4", "e7e5"], &["b1c3", "g2g3", "e2e4"]),
        (&["c2c4", "c7c5"], &["b1c3", "g2g3", "g1f3"]),
        (&["g1f3", "c7c5"], &["c2c4", "b2b4", "e2e4"]),
        (&["e2e4", "c7c5", "g1f3"], &["b8c6", "d7d6", "e7e6"]),
        (&["e2e4", "c7c5", "g1f3", "d7d6"], &["f1b5", "d2d4", "f2f4"]),
        (&["e2e4", "e7e5", "f2f4"], &["e5f4", "d7d5"]),
        (&["e2e4", "e7e5", "g1f3"], &["b8c6", "g8f6", "f7f5"]),
        (&["d2d4", "d7d5"], &["c2c4", "e2e4", "b1c3"]),
        (&["d2d4", "d7d5", "c2c4"], &["e7e6", "d5c4", "c7c6"]),
        (&["d2d4", "g8f6", "c2c4"], &["c7c5", "g7g6", "e7e6"]),
        ],
    }
}

fn apply_position_spec(board: &mut Board, spec: &str) -> bool {
    let tokens: Vec<&str> = spec.split_whitespace().collect();
    if tokens.is_empty() {
        return false;
    }

    let mut i = 0usize;
    if tokens[i] == "position" {
        i += 1;
    }
    if i >= tokens.len() {
        return false;
    }

    if tokens[i] == "startpos" {
        board.set_startpos();
        i += 1;
    } else if tokens[i] == "fen" {
        i += 1;
        let mut fen_parts = Vec::new();
        while i < tokens.len() && tokens[i] != "moves" {
            fen_parts.push(tokens[i]);
            i += 1;
        }
        if fen_parts.is_empty() {
            return false;
        }
        board.set_fen(&fen_parts.join(" "));
    } else {
        return false;
    }

    if i < tokens.len() && tokens[i] == "moves" {
        i += 1;
        while i < tokens.len() {
            let Some(mv) = parse_uci_move(board, tokens[i]) else {
                return false;
            };
            let _ = board.make_move(mv);
            i += 1;
        }
    }

    true
}

fn push_book_entry(book: &mut Vec<BookEntry>, key: u64, moves: Vec<String>) {
    if moves.is_empty() {
        return;
    }

    if let Some(entry) = book.iter_mut().find(|entry| entry.key == key) {
        for mv in moves {
            if !entry.moves.iter().any(|existing| existing == &mv) {
                entry.moves.push(mv);
            }
        }
        return;
    }

    book.push(BookEntry { key, moves });
}

fn parse_external_book(path: &str, zobrist: &Zobrist) -> Vec<BookEntry> {
    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };

    let mut book = Vec::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((spec, replies)) = line
            .split_once("=>")
            .or_else(|| line.split_once('|'))
        else {
            continue;
        };

        let mut board = Board::new();
        if !apply_position_spec(&mut board, spec.trim()) {
            continue;
        }

        let moves = replies
            .split_whitespace()
            .filter_map(|mv| parse_uci_move(&mut board, mv).map(|_| mv.to_string()))
            .collect::<Vec<_>>();
        push_book_entry(&mut book, board_hash(&board, zobrist), moves);
    }

    book
}

pub(crate) fn build_opening_book(
    zobrist: &Zobrist,
    style: BookStyle,
    external_path: Option<&str>,
) -> Vec<BookEntry> {
    let lines = book_lines(style);

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
            push_book_entry(
                &mut book,
                board_hash(&board, zobrist),
                replies.iter().map(|mv| (*mv).to_string()).collect(),
            );
        }
    }
    if let Some(path) = external_path {
        for entry in parse_external_book(path, zobrist) {
            push_book_entry(&mut book, entry.key, entry.moves);
        }
    }
    book
}

pub(crate) fn try_book_move(board: &mut Board, engine: &EngineState) -> Option<String> {
    let key = board_hash(board, &engine.zobrist);
    let entry = engine.opening_book.iter().find(|e| e.key == key)?;
    let mut legal_book_moves = Vec::new();
    for mv_str in &entry.moves {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_book_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rustfish_{name}_{nanos}.txt"))
    }

    #[test]
    fn external_book_entries_are_loaded() {
        let path = temp_book_path("book");
        fs::write(
            &path,
            "# comment\nstartpos moves e2e4 e7e5 => g1f3 f1c4\n",
        )
        .unwrap();

        let zobrist = Zobrist::new();
        let book = build_opening_book(&zobrist, BookStyle::Normal, Some(path.to_str().unwrap()));
        let mut board = Board::new();
        board.set_startpos();
        let mv = parse_uci_move(&mut board, "e2e4").unwrap();
        let _ = board.make_move(mv);
        let mv = parse_uci_move(&mut board, "e7e5").unwrap();
        let _ = board.make_move(mv);
        let key = board_hash(&board, &zobrist);
        let entry = book.iter().find(|entry| entry.key == key).unwrap();

        assert!(entry.moves.iter().any(|mv| mv == "g1f3"));
        assert!(entry.moves.iter().any(|mv| mv == "f1c4"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn position_spec_parser_accepts_fen_with_moves() {
        let mut board = Board::new();
        assert!(apply_position_spec(
            &mut board,
            "fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4 c7c5"
        ));
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(
            board.squares[28],
            Some(Piece {
                color: Color::White,
                kind: PieceType::Pawn,
            })
        );
        assert_eq!(
            board.squares[34],
            Some(Piece {
                color: Color::Black,
                kind: PieceType::Pawn,
            })
        );
    }
}
