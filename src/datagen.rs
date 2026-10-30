//! Self-play data generation for the Texel tuner.
//!
//! `datagen <out.txt> <games> [nodes] [threads]` plays fixed-node self-play
//! games from randomised openings and writes quiet positions as
//! `fen;result` lines (result from White's side: 1, 0.5, 0), the format the
//! tuner reads. No effect on normal play.

use super::*;
use crate::movegen::square_to_string;
use crate::search::{qsearch_gap, LAST_SEARCH_SCORE, QUIET_SEARCH};
use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const RANDOM_PLIES: usize = 8;
const MAX_OPENING_EVAL: i32 = 400;
const MAX_RECORD_EVAL: i32 = 2000;
const MAX_GAME_PLIES: usize = 300;
const WIN_SCORE: i32 = 1000;
const WIN_PLIES: u32 = 4;
const DRAW_SCORE: i32 = 10;
const DRAW_PLIES: u32 = 10;
const DRAW_MIN_PLY: usize = 80; // move 40
const DATAGEN_HASH_MB: usize = 16;

struct Shared {
    games: u64,
    nodes: u64,
    next_game: AtomicU64,
    games_done: AtomicU64,
    positions: AtomicU64,
    results: [AtomicU64; 3], // white wins, draws, black wins
    out: Mutex<BufWriter<File>>,
    start: Instant,
}

pub(crate) fn handle_datagen(cmd: &str) {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let Some(path) = tokens.get(1) else {
        eprintln!("usage: datagen <out.txt> <games> [nodes] [threads]");
        return;
    };
    let games: u64 = tokens.get(2).and_then(|t| t.parse().ok()).unwrap_or(100);
    let nodes: u64 = tokens.get(3).and_then(|t| t.parse().ok()).unwrap_or(5000).max(1);
    let threads: usize = tokens.get(4).and_then(|t| t.parse().ok()).unwrap_or(1).clamp(1, 64);
    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("datagen: cannot create {}: {}", path, e);
            return;
        }
    };
    // Distinct runs get distinct openings; games within a run are seeded by index.
    let base_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);

    let shared = Arc::new(Shared {
        games,
        nodes,
        next_game: AtomicU64::new(0),
        games_done: AtomicU64::new(0),
        positions: AtomicU64::new(0),
        results: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        out: Mutex::new(BufWriter::new(file)),
        start: Instant::now(),
    });
    eprintln!(
        "datagen: {} games, {} nodes/move, {} threads -> {}",
        games, nodes, threads, path
    );

    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let shared = Arc::clone(&shared);
            let seed = base_seed ^ ((t as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || worker(&shared, seed))
                .expect("spawn datagen thread")
        })
        .collect();
    for h in handles {
        let _ = h.join();
    }
    if let Ok(mut out) = shared.out.lock() {
        let _ = out.flush();
    }
    report(&shared, true);
}

fn report(shared: &Shared, final_line: bool) {
    let done = shared.games_done.load(Ordering::Relaxed);
    let pos = shared.positions.load(Ordering::Relaxed);
    let secs = shared.start.elapsed().as_secs_f64().max(1e-3);
    let r: Vec<u64> = shared.results.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    eprintln!(
        "datagen{}: games {}/{} positions {} ({:.1}/game) {:.2} games/s {:.0} pos/s  W/D/L {}/{}/{}",
        if final_line { " done" } else { "" },
        done,
        shared.games,
        pos,
        pos as f64 / done.max(1) as f64,
        done as f64 / secs,
        pos as f64 / secs,
        r[0],
        r[1],
        r[2]
    );
}

fn worker(shared: &Shared, seed: u64) {
    QUIET_SEARCH.with(|q| q.set(true));
    let mut board = Board::new();
    let mut engine = EngineState::new();
    engine.set_hash(DATAGEN_HASH_MB);
    engine.own_book = false;
    engine.threads = 1;
    // qsearch_gap probes the TT, so it gets its own engine: the search TT would
    // return deep search scores for positions the previous search visited.
    let mut qengine = EngineState::new();
    qengine.set_hash(1);
    qengine.own_book = false;
    let go = format!("go nodes {}", shared.nodes);

    loop {
        let g = shared.next_game.fetch_add(1, Ordering::Relaxed);
        if g >= shared.games {
            break;
        }
        let mut rng = seed ^ g.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut lines = Vec::new();
        let result = play_game(&mut board, &mut engine, &mut qengine, &go, &mut rng, &mut lines);
        let (res_str, idx) = match result {
            r if r > 0 => ("1", 0),
            0 => ("0.5", 1),
            _ => ("0", 2),
        };
        if let Ok(mut out) = shared.out.lock() {
            for fen in &lines {
                let _ = writeln!(out, "{};{}", fen, res_str);
            }
        }
        shared.results[idx].fetch_add(1, Ordering::Relaxed);
        shared.positions.fetch_add(lines.len() as u64, Ordering::Relaxed);
        let done = shared.games_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 50 == 0 {
            report(shared, false);
        }
    }
}

fn xorshift(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 0x2545_F491_4F6C_DD1D;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn push_history(board: &Board, engine: &mut EngineState) {
    let h = board_hash(board, &engine.zobrist);
    engine.position_history.push(h);
}

/// Start position plus RANDOM_PLIES random legal moves, retried until the
/// game is still going and the static eval is not lopsided.
fn random_opening(board: &mut Board, engine: &mut EngineState, rng: &mut u64) {
    let mut moves = Vec::new();
    'retry: loop {
        board.set_startpos();
        engine.clear_for_new_game();
        push_history(board, engine);
        for _ in 0..RANDOM_PLIES {
            board.generate_legal_moves(false, &mut moves);
            if moves.is_empty() {
                continue 'retry;
            }
            let mv = moves[(xorshift(rng) % moves.len() as u64) as usize];
            let _ = board.make_move(mv);
            push_history(board, engine);
        }
        board.generate_legal_moves(false, &mut moves);
        if moves.is_empty() || eval_for_side(board).abs() > MAX_OPENING_EVAL {
            continue;
        }
        return;
    }
}

fn insufficient_material(board: &Board) -> bool {
    let mut minors = 0;
    for p in board.squares.iter().flatten() {
        match p.kind {
            PieceType::King => {}
            PieceType::Knight | PieceType::Bishop => minors += 1,
            _ => return false,
        }
    }
    minors <= 1
}

fn threefold(engine: &EngineState, halfmove_clock: u32) -> bool {
    let hist = &engine.position_history;
    let Some(&key) = hist.last() else { return false };
    let window = (halfmove_clock as usize + 1).min(hist.len());
    hist[hist.len() - window..].iter().filter(|&&h| h == key).count() >= 3
}

/// Plays one game; returns +1 / 0 / -1 from White's side and fills `lines`
/// with the FENs of positions worth recording.
fn play_game(
    board: &mut Board,
    engine: &mut EngineState,
    qengine: &mut EngineState,
    go: &str,
    rng: &mut u64,
    lines: &mut Vec<String>,
) -> i32 {
    random_opening(board, engine, rng);
    qengine.clear_for_new_game();
    let mut moves = Vec::new();
    let mut win_run = 0u32;
    let mut win_sign = 0i32;
    let mut draw_run = 0u32;

    for ply in RANDOM_PLIES..MAX_GAME_PLIES {
        let white = board.side_to_move == Color::White;
        board.generate_legal_moves(false, &mut moves);
        let in_check = board.is_in_check(board.side_to_move);
        if moves.is_empty() {
            if !in_check {
                return 0;
            }
            return if white { -1 } else { 1 };
        }
        if board.halfmove_clock >= 100 || threefold(engine, board.halfmove_clock) || insufficient_material(board) {
            return 0;
        }

        if !in_check && eval_for_side(board).abs() < MAX_RECORD_EVAL && qsearch_gap(board, qengine) == 0 {
            lines.push(board_to_fen(board));
        }

        let Some(uci) = handle_go(go, board, engine) else { return 0 };
        let score = LAST_SEARCH_SCORE.with(|s| s.get());
        let white_score = if white { score } else { -score };

        if white_score.abs() >= MATE_SCORE - 1000 {
            return white_score.signum();
        }
        if white_score.abs() >= WIN_SCORE {
            let sign = white_score.signum();
            if sign == win_sign {
                win_run += 1;
            } else {
                win_sign = sign;
                win_run = 1;
            }
            if win_run >= WIN_PLIES {
                return sign;
            }
        } else {
            win_run = 0;
            win_sign = 0;
        }
        if ply >= DRAW_MIN_PLY && white_score.abs() <= DRAW_SCORE {
            draw_run += 1;
            if draw_run >= DRAW_PLIES {
                return 0;
            }
        } else {
            draw_run = 0;
        }

        let Some(mv) = parse_uci_move(board, &uci) else { return 0 };
        let _ = board.make_move(mv);
        push_history(board, engine);
    }
    0
}

fn board_to_fen(board: &Board) -> String {
    let mut s = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            match board.squares[rank * 8 + file] {
                None => empty += 1,
                Some(p) => {
                    if empty > 0 {
                        s.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let c = match p.kind {
                        PieceType::Pawn => 'p',
                        PieceType::Knight => 'n',
                        PieceType::Bishop => 'b',
                        PieceType::Rook => 'r',
                        PieceType::Queen => 'q',
                        PieceType::King => 'k',
                    };
                    s.push(if p.color == Color::White { c.to_ascii_uppercase() } else { c });
                }
            }
        }
        if empty > 0 {
            s.push_str(&empty.to_string());
        }
        if rank > 0 {
            s.push('/');
        }
    }
    s.push_str(if board.side_to_move == Color::White { " w " } else { " b " });
    let mut castling = String::new();
    for (bit, c) in [(1, 'K'), (2, 'Q'), (4, 'k'), (8, 'q')] {
        if board.castling & bit != 0 {
            castling.push(c);
        }
    }
    s.push_str(if castling.is_empty() { "-" } else { &castling });
    s.push(' ');
    s.push_str(&board.en_passant.map(square_to_string).unwrap_or_else(|| "-".to_string()));
    s.push_str(&format!(" {} {}", board.halfmove_clock, board.fullmove_number));
    s
}
