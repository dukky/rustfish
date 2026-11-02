//! Self-play data generation (training data for NNUE and the Texel tuner).
//!
//! `datagen <out.txt> <games> [nodes] [threads] [seed] [first_game]` plays
//! fixed-node self-play games from random openings and writes one position
//! per line in the shared training format:
//!
//! `<FEN> | <score_cp from White's view> | <result from White's view: 1.0/0.5/0.0>`
//!
//! Games are numbered `first_game .. first_game + games`, and game `g` depends
//! only on (`seed`, `g`, `nodes`): its RNG is derived from both, and the
//! engine's TT and history tables are cleared before it starts. Output is
//! written in game order, so a run is byte-identical for any thread count, and
//! CI shards that split one seed into disjoint game ranges concatenate into
//! exactly the file a single run would have written.
//!
//! Openings are startpos plus 8 or 9 uniformly random legal plies; starts that
//! a short search scores beyond OPENING_MAX_SCORE are rejected. A position is
//! recorded unless the side to move is in check, the search's best move is a
//! capture or promotion, or the score is a mate score. Games end on mate,
//! stalemate, the 50-move rule, threefold repetition or insufficient material,
//! or are adjudicated: a win once |score| >= WIN_SCORE for WIN_PLIES plies in a
//! row (same sign) or on a mate score, a draw once |score| <= DRAW_SCORE for
//! DRAW_PLIES plies in a row after DRAW_MIN_PLY. No effect on normal play.

use super::*;
use crate::movegen::square_to_string;
use crate::search::{LAST_SEARCH_SCORE, QUIET_SEARCH};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const RANDOM_PLIES_MIN: usize = 8; // 8 or 9, so either side can be to move
const OPENING_PROBE_NODES: u64 = 1000;
const OPENING_MAX_SCORE: i32 = 400;
const MAX_GAME_PLIES: usize = 400;
const MATE_THRESHOLD: i32 = MATE_SCORE - 1000;
const WIN_SCORE: i32 = 1000;
const WIN_PLIES: u32 = 4;
const DRAW_SCORE: i32 = 10;
const DRAW_PLIES: u32 = 10;
const DRAW_MIN_PLY: usize = 80; // move 40
const DATAGEN_HASH_MB: usize = 16;
const REPORT_EVERY: u64 = 100;

struct Output {
    out: BufWriter<File>,
    next: u64,                      // next game index to write
    pending: BTreeMap<u64, String>, // finished games waiting for earlier ones
}

struct Shared {
    first_game: u64,
    end_game: u64,
    nodes: u64,
    seed: u64,
    next_game: AtomicU64,
    games_done: AtomicU64,
    positions: AtomicU64,
    plies: AtomicU64,
    adjudicated: AtomicU64,
    results: [AtomicU64; 3], // white wins, draws, black wins
    output: Mutex<Output>,
    start: Instant,
}

pub(crate) fn handle_datagen(cmd: &str) {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let Some(path) = tokens.get(1) else {
        eprintln!("usage: datagen <out.txt> <games> [nodes] [threads] [seed] [first_game]");
        return;
    };
    let arg = |i: usize, default: u64| tokens.get(i).and_then(|t| t.parse::<u64>().ok()).unwrap_or(default);
    let games = arg(2, 100);
    let nodes = arg(3, 5000).max(1);
    let threads = (arg(4, 1) as usize).clamp(1, 256);
    let seed = arg(5, 1);
    let first_game = arg(6, 0);
    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("datagen: cannot create {}: {}", path, e);
            return;
        }
    };

    let shared = Arc::new(Shared {
        first_game,
        end_game: first_game.saturating_add(games),
        nodes,
        seed,
        next_game: AtomicU64::new(first_game),
        games_done: AtomicU64::new(0),
        positions: AtomicU64::new(0),
        plies: AtomicU64::new(0),
        adjudicated: AtomicU64::new(0),
        results: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        output: Mutex::new(Output {
            out: BufWriter::with_capacity(1 << 20, file),
            next: first_game,
            pending: BTreeMap::new(),
        }),
        start: Instant::now(),
    });
    eprintln!(
        "datagen: games {}..{} seed {} nodes {} threads {} -> {}",
        first_game, shared.end_game, seed, nodes, threads, path
    );

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let shared = Arc::clone(&shared);
            thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || worker(&shared))
                .expect("spawn datagen thread")
        })
        .collect();
    for h in handles {
        let _ = h.join();
    }
    if let Ok(mut o) = shared.output.lock() {
        let _ = o.out.flush();
    }
    report(&shared, true);
}

fn report(shared: &Shared, final_line: bool) {
    let done = shared.games_done.load(Ordering::Relaxed);
    let pos = shared.positions.load(Ordering::Relaxed);
    let plies = shared.plies.load(Ordering::Relaxed);
    let adj = shared.adjudicated.load(Ordering::Relaxed);
    let secs = shared.start.elapsed().as_secs_f64().max(1e-3);
    let r: Vec<u64> = shared.results.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    eprintln!(
        "datagen{}: games {}/{} positions {} ({:.1}/game, {:.1} plies/game, {} adjudicated) \
         {:.0}s {:.2} games/s {:.0} pos/s  W/D/L {}/{}/{}",
        if final_line { " done" } else { "" },
        done,
        shared.end_game - shared.first_game,
        pos,
        pos as f64 / done.max(1) as f64,
        plies as f64 / done.max(1) as f64,
        adj,
        secs,
        done as f64 / secs,
        pos as f64 / secs,
        r[0],
        r[1],
        r[2]
    );
}

#[derive(Default)]
struct Game {
    result: i32,                   // +1 / 0 / -1 from White's side
    positions: Vec<(String, i32)>, // FEN, White-POV score
    plies: usize,
    adjudicated: bool,
}

fn worker(shared: &Shared) {
    QUIET_SEARCH.with(|q| q.set(true));
    let mut board = Board::new();
    let mut engine = EngineState::new();
    engine.set_hash(DATAGEN_HASH_MB);
    engine.own_book = false;
    engine.threads = 1;
    // Opening probes get their own small engine, so the game engine is exactly
    // "freshly cleared" when a game's first real search starts.
    let mut probe = EngineState::new();
    probe.set_hash(1);
    probe.own_book = false;
    probe.threads = 1;
    let go = format!("go nodes {}", shared.nodes);
    let go_probe = format!("go nodes {}", OPENING_PROBE_NODES);

    loop {
        let g = shared.next_game.fetch_add(1, Ordering::Relaxed);
        if g >= shared.end_game {
            break;
        }
        let mut rng = splitmix64(splitmix64(shared.seed) ^ g);
        let mut game = Game::default();
        play_game(&mut board, &mut engine, &mut probe, &go, &go_probe, &mut rng, &mut game);

        let (res_str, idx) = match game.result {
            r if r > 0 => ("1.0", 0),
            0 => ("0.5", 1),
            _ => ("0.0", 2),
        };
        let mut text = String::with_capacity(game.positions.len() * 80);
        for (fen, score) in &game.positions {
            text.push_str(fen);
            text.push_str(&format!(" | {} | {}\n", score, res_str));
        }
        if let Ok(mut o) = shared.output.lock() {
            o.pending.insert(g, text);
            loop {
                let next = o.next;
                let Some(t) = o.pending.remove(&next) else { break };
                let _ = o.out.write_all(t.as_bytes());
                o.next += 1;
            }
        }
        shared.results[idx].fetch_add(1, Ordering::Relaxed);
        shared.positions.fetch_add(game.positions.len() as u64, Ordering::Relaxed);
        shared.plies.fetch_add(game.plies as u64, Ordering::Relaxed);
        if game.adjudicated {
            shared.adjudicated.fetch_add(1, Ordering::Relaxed);
        }
        let done = shared.games_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % REPORT_EVERY == 0 {
            report(shared, false);
        }
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn next_rand(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64(*state)
}

fn push_history(board: &Board, engine: &mut EngineState) {
    let h = board_hash(board, &engine.zobrist);
    engine.position_history.push(h);
}

/// Start position plus 8 or 9 random legal plies, retried until the game is
/// still going and a short search does not find it lopsided. Leaves `engine`
/// cleared with the opening in its position history; returns the ply count.
fn random_opening(
    board: &mut Board,
    engine: &mut EngineState,
    probe: &mut EngineState,
    go_probe: &str,
    rng: &mut u64,
) -> usize {
    let mut moves = Vec::new();
    'retry: loop {
        let plies = RANDOM_PLIES_MIN + (next_rand(rng) & 1) as usize;
        board.set_startpos();
        probe.clear_for_new_game();
        push_history(board, probe);
        for _ in 0..plies {
            board.generate_legal_moves(false, &mut moves);
            if moves.is_empty() {
                continue 'retry;
            }
            let mv = moves[(next_rand(rng) % moves.len() as u64) as usize];
            let _ = board.make_move(mv);
            push_history(board, probe);
        }
        board.generate_legal_moves(false, &mut moves);
        if moves.is_empty() || handle_go(go_probe, board, probe).is_none() {
            continue;
        }
        if LAST_SEARCH_SCORE.with(|s| s.get()).abs() > OPENING_MAX_SCORE {
            continue;
        }
        engine.clear_for_new_game();
        engine.position_history.extend_from_slice(&probe.position_history);
        return plies;
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

/// Plays one game into `game` (result, recorded positions, length).
fn play_game(
    board: &mut Board,
    engine: &mut EngineState,
    probe: &mut EngineState,
    go: &str,
    go_probe: &str,
    rng: &mut u64,
    game: &mut Game,
) {
    let start_ply = random_opening(board, engine, probe, go_probe, rng);
    let mut moves = Vec::new();
    let mut win_run = 0u32;
    let mut win_sign = 0i32;
    let mut draw_run = 0u32;
    game.result = 0;

    for ply in start_ply..MAX_GAME_PLIES {
        game.plies = ply;
        let white = board.side_to_move == Color::White;
        board.generate_legal_moves(false, &mut moves);
        let in_check = board.is_in_check(board.side_to_move);
        if moves.is_empty() {
            game.result = match (in_check, white) {
                (false, _) => 0,
                (true, true) => -1,
                (true, false) => 1,
            };
            return;
        }
        if board.halfmove_clock >= 100 || threefold(engine, board.halfmove_clock) || insufficient_material(board) {
            return;
        }

        let Some(uci) = handle_go(go, board, engine) else { return };
        let Some(mv) = parse_uci_move(board, &uci) else { return };
        let score = LAST_SEARCH_SCORE.with(|s| s.get());
        let white_score = if white { score } else { -score };

        let noisy = mv.is_en_passant || mv.promotion.is_some() || board.squares[mv.to as usize].is_some();
        if !in_check && !noisy && white_score.abs() < MATE_THRESHOLD {
            game.positions.push((board_to_fen(board), white_score));
        }

        if white_score.abs() >= MATE_THRESHOLD {
            game.result = white_score.signum();
            game.adjudicated = true;
            return;
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
                game.result = sign;
                game.adjudicated = true;
                return;
            }
        } else {
            win_run = 0;
            win_sign = 0;
        }
        if ply >= DRAW_MIN_PLY && white_score.abs() <= DRAW_SCORE {
            draw_run += 1;
            if draw_run >= DRAW_PLIES {
                game.adjudicated = true;
                return;
            }
        } else {
            draw_run = 0;
        }

        let _ = board.make_move(mv);
        push_history(board, engine);
    }
    // Ply cap: scored as a draw.
    game.plies = MAX_GAME_PLIES;
    game.adjudicated = true;
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
