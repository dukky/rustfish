use super::*;

pub(crate) fn flush_stdout() {
    let _ = io::stdout().flush();
}

pub(crate) fn handle_setoption(cmd: &str, engine: &mut EngineState) {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    if tokens.len() < 3 {
        return;
    }

    let mut name = String::new();
    let mut value = String::new();
    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "name" => {
                i += 1;
                while i < tokens.len() && tokens[i] != "value" {
                    if !name.is_empty() {
                        name.push(' ');
                    }
                    name.push_str(tokens[i]);
                    i += 1;
                }
            }
            "value" => {
                i += 1;
                while i < tokens.len() {
                    if !value.is_empty() {
                        value.push(' ');
                    }
                    value.push_str(tokens[i]);
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    let n = name.to_ascii_lowercase();
    if n == "clear hash" {
        engine.tt.clear();
        return;
    }
    if n == "hash" {
        if let Ok(mb) = value.trim().parse::<usize>() {
            engine.set_hash(mb);
        }
        return;
    }
    if n == "threads" {
        if let Ok(threads) = value.trim().parse::<usize>() {
            engine.set_threads(threads);
        }
        return;
    }
    if n == "ownbook" {
        let v = value.trim().to_ascii_lowercase();
        engine.own_book = v == "true" || v == "1" || v == "on";
        return;
    }
    if n == "bookstyle" {
        let style = match value.trim().to_ascii_lowercase().as_str() {
            "grinder" | "solid" => Some(BookStyle::Grinder),
            "chaos" | "sharp" => Some(BookStyle::Chaos),
            "normal" | "" => Some(BookStyle::Normal),
            _ => None,
        };
        if let Some(style) = style {
            engine.set_book_style(style);
        }
        return;
    }
    if n == "bookfile" {
        let path = value.trim();
        if path.is_empty() || path.eq_ignore_ascii_case("none") || path == "-" {
            engine.set_book_file(None);
        } else {
            engine.set_book_file(Some(path.to_string()));
        }
        return;
    }

    if let Ok(v) = value.trim().parse::<i32>() {
        match n.as_str() {
            "futilitymargind1" => engine.search_params.futility_margin_d1 = v.clamp(0, 500),
            "futilitymargind2" => engine.search_params.futility_margin_d2 = v.clamp(0, 700),
            "reversefutilitymargind1" => {
                engine.search_params.reverse_futility_margin_d1 = v.clamp(0, 500)
            }
            "reversefutilitymargind2" => {
                engine.search_params.reverse_futility_margin_d2 = v.clamp(0, 700)
            }
            "reversefutilitymargind3" => {
                engine.search_params.reverse_futility_margin_d3 = v.clamp(0, 900)
            }
            "nullmarginbase" => engine.search_params.null_margin_base = v.clamp(0, 300),
            "nullmargindepthscale" => {
                engine.search_params.null_margin_depth_scale = v.clamp(0, 80)
            }
            "nullreductionbase" => engine.search_params.null_reduction_base = v.clamp(0, 6),
            "nullreductiondepthdiv" => engine.search_params.null_reduction_depth_div = v.clamp(1, 12),
            "lmrmindepth" => engine.search_params.lmr_min_depth = v.clamp(1, 12),
            "lmrextradepth" => engine.search_params.lmr_extra_depth = v.clamp(1, 16),
            _ => {}
        }
    }

    if let Ok(v) = value.trim().parse::<usize>() {
        match n.as_str() {
            "lmplimitd1" => engine.search_params.lmp_limit_d1 = v.clamp(0, 24),
            "lmplimitd2" => engine.search_params.lmp_limit_d2 = v.clamp(0, 32),
            "lmplimitd3" => engine.search_params.lmp_limit_d3 = v.clamp(0, 40),
            "lmrminmoveindex" => engine.search_params.lmr_min_move_index = v.clamp(1, 24),
            "lmrextramoveindex" => engine.search_params.lmr_extra_move_index = v.clamp(1, 32),
            _ => {}
        }
    }
}

pub(crate) fn handle_position(cmd: &str, board: &mut Board, engine: &mut EngineState) {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    if tokens.len() < 2 {
        return;
    }

    let mut i = 1;
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
        let fen = fen_parts.join(" ");
        board.set_fen(&fen);
    }

    engine.position_history.clear();
    engine.position_history.push(board_hash(board, &engine.zobrist));

    if i < tokens.len() && tokens[i] == "moves" {
        i += 1;
        while i < tokens.len() {
            if let Some(mv) = parse_uci_move(board, tokens[i]) {
                let _ = board.make_move(mv);
                engine.position_history.push(board_hash(board, &engine.zobrist));
            }
            i += 1;
        }
    }
}

pub(crate) fn handle_perft(cmd: &str, board: &mut Board) {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let depth: i32 = if tokens.len() >= 2 {
        tokens[1].parse().unwrap_or(1)
    } else {
        1
    };
    let start = Instant::now();
    let nodes = perft(board, depth.max(0));
    let ms = start.elapsed().as_millis() as u64;
    println!("info string perft depth {} nodes {} time {}", depth.max(0), nodes, ms);
    flush_stdout();
}

pub(crate) fn handle_divide(cmd: &str, board: &mut Board) {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let depth: i32 = if tokens.len() >= 2 {
        tokens[1].parse().unwrap_or(1)
    } else {
        1
    };
    let d = depth.max(1);
    let start = Instant::now();
    let mut total = 0u64;

    let mut root = Vec::new();
    board.generate_legal_moves(false, &mut root);
    root.sort_by_key(|m| move_to_uci(*m));
    for mv in root {
        let undo = board.make_move(mv);
        let n = perft(board, d - 1);
        board.undo_move(mv, undo);
        total += n;
        println!("{} {}", move_to_uci(mv), n);
    }

    let ms = start.elapsed().as_millis() as u64;
    println!("info string divide depth {} total {} time {}", d, total, ms);
    flush_stdout();
}

pub(crate) fn perft(board: &mut Board, depth: i32) -> u64 {
    if depth <= 0 {
        return 1;
    }
    let mut moves = Vec::new();
    board.generate_legal_moves(false, &mut moves);
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0u64;
    for mv in moves {
        let undo = board.make_move(mv);
        nodes += perft(board, depth - 1);
        board.undo_move(mv, undo);
    }
    nodes
}

pub(crate) fn handle_perft_test(board: &mut Board) {
    let tests: [(&str, i32, u64); 8] = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            1,
            20,
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            2,
            400,
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            3,
            8902,
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            4,
            197281,
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            1,
            26,
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            2,
            568,
        ),
        (
            "8/8/8/3pP3/8/8/8/8 w - d6 0 1",
            1,
            2,
        ),
        (
            "8/P7/8/8/8/8/7p/7k w - - 0 1",
            1,
            4,
        ),
    ];

    let old = board.clone();
    let mut ok = 0usize;
    let start = Instant::now();
    for (idx, (fen, depth, expected)) in tests.iter().enumerate() {
        board.set_fen(fen);
        let got = perft(board, *depth);
        if got == *expected {
            ok += 1;
            println!("info string perfttest {} ok depth {} nodes {}", idx + 1, depth, got);
        } else {
            println!(
                "info string perfttest {} FAIL depth {} got {} expected {} fen {}",
                idx + 1,
                depth,
                got,
                expected,
                fen
            );
        }
    }
    *board = old;

    let ms = start.elapsed().as_millis() as u64;
    println!(
        "info string perfttest summary {}/{} passed in {} ms",
        ok,
        tests.len(),
        ms
    );
    flush_stdout();
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn check_legal_move_regression(
    board: &mut Board,
    fen: &str,
    required: &[&str],
    forbidden: &[&str],
) -> Result<(), String> {
    board.set_fen(fen);
    let mut legal = Vec::new();
    board.generate_legal_moves(false, &mut legal);
    let legal_uci: Vec<String> = legal.into_iter().map(move_to_uci).collect();

    for mv in required {
        if !legal_uci.iter().any(|legal_mv| legal_mv == mv) {
            return Err(format!(
                "missing required move {} in fen {} legal {:?}",
                mv, fen, legal_uci
            ));
        }
    }

    for mv in forbidden {
        if legal_uci.iter().any(|legal_mv| legal_mv == mv) {
            return Err(format!(
                "found forbidden move {} in fen {} legal {:?}",
                mv, fen, legal_uci
            ));
        }
    }

    Ok(())
}

pub(crate) fn handle_selftest(cmd: &str, board: &mut Board) {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let iters: usize = if tokens.len() >= 2 {
        tokens[1].parse().unwrap_or(200)
    } else {
        200
    };

    let old = board.clone();
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut tests_done = 0usize;
    let edge_fens = [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "8/8/8/3pP3/8/8/8/8 w - d6 0 1",
        "8/P7/8/8/8/8/7p/7k w - - 0 1",
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
    ];
    let regression_suites = [
        (
            "4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1",
            &[][..],
            &["e2c1", "e2c3", "e2d4", "e2f4", "e2g3", "e2g1"][..],
        ),
        (
            "4r1k1/8/8/8/8/8/3B4/4K3 w - - 0 1",
            &["d2e3"][..],
            &["d2c3", "d2b4", "d2a5", "d2f4"][..],
        ),
        (
            "4k3/8/8/8/1b6/8/4r3/3QK3 w - - 0 1",
            &[][..],
            &["d1d8", "d1e2", "d1h5"][..],
        ),
    ];

    let mut legal = Vec::new();
    for fen in edge_fens {
        board.set_fen(fen);
        let baseline = board.clone();
        board.generate_legal_moves(false, &mut legal);
        for mv in legal.iter().copied() {
            let undo = board.make_move(mv);
            let acc_ok = board.eval_acc == board.recompute_eval_acc();
            board.undo_move(mv, undo);
            if *board != baseline || !acc_ok || board.eval_acc != baseline.eval_acc {
                println!(
                    "info string selftest FAIL edge make/undo mismatch at move {} fen {}",
                    move_to_uci(mv),
                    fen
                );
                *board = old;
                flush_stdout();
                return;
            }
            tests_done += 1;
        }
    }

    for (fen, required, forbidden) in regression_suites {
        match check_legal_move_regression(board, fen, required, forbidden) {
            Ok(()) => tests_done += required.len() + forbidden.len(),
            Err(err) => {
                println!("info string selftest FAIL regression {}", err);
                *board = old;
                flush_stdout();
                return;
            }
        }
    }

    for _ in 0..iters {
        board.set_startpos();
        let plies = (xorshift64(&mut rng) % 40) as usize;
        for _ in 0..plies {
            board.generate_legal_moves(false, &mut legal);
            if legal.is_empty() {
                break;
            }
            let idx = (xorshift64(&mut rng) as usize) % legal.len();
            let _ = board.make_move(legal[idx]);
        }

        let baseline = board.clone();
        board.generate_legal_moves(false, &mut legal);
        for mv in legal.iter().copied() {
            let undo = board.make_move(mv);
            if board.eval_acc != board.recompute_eval_acc() {
                println!(
                    "info string selftest FAIL incremental eval_acc mismatch after move {}",
                    move_to_uci(mv)
                );
                *board = old;
                flush_stdout();
                return;
            }
            board.undo_move(mv, undo);
            if *board != baseline || board.eval_acc != baseline.eval_acc {
                println!("info string selftest FAIL make/undo mismatch at move {}", move_to_uci(mv));
                *board = old;
                flush_stdout();
                return;
            }
            tests_done += 1;
        }
    }

    *board = old;
    println!("info string selftest ok positions {} makeundo_checks {}", iters, tests_done);
    flush_stdout();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn position_moves_preserve_repetition_history() {
        thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let mut board = Board::new();
                let mut engine = EngineState::new();
                handle_position(
                    "position startpos moves g1f3 g8f6 f3g1 f6g8",
                    &mut board,
                    &mut engine,
                );

                let key = board_hash(&board, &engine.zobrist);
                assert_eq!(engine.position_history.first().copied(), Some(key));
                assert_eq!(engine.position_history.last().copied(), Some(key));
                assert!(crate::search::is_repetition(
                    &engine.position_history,
                    key,
                    board.halfmove_clock,
                ));
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
