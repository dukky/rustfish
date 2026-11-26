//! Relabelling: new search scores for existing training positions.
//!
//! `relabel <in> <out> [nodes] [threads] [softnodes=N] [clear=game|pos] [hash=MB] [skip=N] [limit=N]`
//!
//! Reads positions in either training format and writes the same format with
//! each score replaced by a fresh `go nodes <nodes>` search of the position
//! (White's view, net units, exactly as `datagen` records it). The game result
//! and the position are kept, so two relabellings of one file differ only in
//! their labels.
//!
//! * A path ending in `.txt` is the text format `<FEN> | <score> | <result>`.
//! * Anything else is the trainer's 32-byte record (`tools/nnue-trainer`,
//!   `Rec`; `nnue-trainer unpack` turns an `.rfp` into one, `pack` turns the
//!   output back). A record has no castling rights, en passant square or move
//!   counters: castling is inferred from kings and rooks on their home
//!   squares, there is no en passant square and the halfmove clock is 0.
//!   Scores are saturated to the record's i16 range +-32000.
//!
//! State: `clear=game` (default) mirrors datagen, whose searches inherit the
//! TT and history tables from the earlier moves of their game: the engine is
//! cleared at each game start and reused along the game. Input must then be in
//! game order (datagen output, game-ordered `.rfp` sources; not a shuffled
//! pool). A game starts where the result changes, the piece count rises, or
//! (text) the fullmove number falls. `clear=pos` clears before every
//! position (independent labels, for shuffled input; costs a TT clear each).
//! No position history is kept (the skipped plies are unknown), so a search
//! sees no repetitions before its root.
//!
//! Work is split into blocks of whole games, each started from a cleared
//! engine, and written in input order: the output is byte-identical for any
//! thread count. Mate scores are written saturated (+-32000) and counted.

use super::*;
use crate::search::{LAST_SEARCH_DEPTH, LAST_SEARCH_SCORE, QUIET_SEARCH, SOFT_NODE_LIMIT};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const REC: usize = 32;
const BLOCK_MIN: usize = 4096; // positions per work block (then extended to a game boundary)
const SCORE_CAP: i32 = 32000;
const MATE_THRESHOLD: i32 = MATE_SCORE - 1000;
const REPORT_EVERY_SECS: u64 = 30;

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Text,
    Rec,
}

/// One input position: its raw form (a text line without the newline, or a
/// 32-byte record) and the keys used to find game boundaries.
struct Item {
    raw: Vec<u8>,
    result: u8,
    pieces: u32,
    fullmove: u32,
}

struct Input {
    reader: Box<dyn BufRead + Send>,
    format: Format,
    carry: Option<Item>,
    next_block: u64,
    skip: u64,
    left: u64, // positions still to hand out (limit)
    eof: bool,
}

struct Output {
    out: BufWriter<File>,
    next: u64,
    pending: BTreeMap<u64, Vec<u8>>,
}

struct Shared {
    nodes: u64,
    soft_nodes: u64,
    per_position: bool,
    hash_mb: usize,
    format: Format,
    input: Mutex<Input>,
    output: Mutex<Output>,
    positions: AtomicU64,
    games: AtomicU64,
    search_nodes: AtomicU64,
    search_depth: AtomicU64,
    mates: AtomicU64,
    no_move: AtomicU64,
    bad: AtomicU64,
    start: Instant,
    last_report: AtomicU64,
}

pub(crate) fn handle_relabel(cmd: &str) {
    let tokens: Vec<&str> = cmd.split_whitespace().filter(|t| !t.contains('=')).collect();
    let mut soft_nodes = u64::MAX;
    let mut per_position = false;
    let mut hash_mb = 16usize;
    let mut skip = 0u64;
    let mut limit = u64::MAX;
    for opt in cmd.split_whitespace().filter(|t| t.contains('=')) {
        let ok = match opt.split_once('=') {
            Some(("softnodes", v)) => v.parse::<u64>().ok().filter(|&n| n > 0).map(|n| soft_nodes = n).is_some(),
            Some(("clear", "game")) => {
                per_position = false;
                true
            }
            Some(("clear", "pos")) => {
                per_position = true;
                true
            }
            Some(("hash", v)) => v.parse::<usize>().ok().filter(|&n| n > 0).map(|n| hash_mb = n).is_some(),
            Some(("skip", v)) => v.parse::<u64>().ok().map(|n| skip = n).is_some(),
            Some(("limit", v)) => v.parse::<u64>().ok().map(|n| limit = n).is_some(),
            _ => false,
        };
        if !ok {
            eprintln!("relabel: bad option {:?}", opt);
            return;
        }
    }
    let (Some(inp), Some(outp)) = (tokens.get(1), tokens.get(2)) else {
        eprintln!("usage: relabel <in> <out> [nodes] [threads] [softnodes=N] [clear=game|pos] [hash=MB] [skip=N] [limit=N]");
        return;
    };
    let arg = |i: usize, default: u64| tokens.get(i).and_then(|t| t.parse::<u64>().ok()).unwrap_or(default);
    let nodes = arg(3, 5000).max(1);
    let threads = (arg(4, 1) as usize).clamp(1, 256);
    let format = if inp.ends_with(".txt") { Format::Text } else { Format::Rec };
    if (format == Format::Text) != outp.ends_with(".txt") {
        eprintln!("relabel: input and output must both be .txt or both be record files");
        return;
    }
    let reader: Box<dyn BufRead + Send> = match File::open(inp) {
        Ok(f) => Box::new(BufReader::with_capacity(1 << 20, f)),
        Err(e) => {
            eprintln!("relabel: cannot open {}: {}", inp, e);
            return;
        }
    };
    let file = match File::create(outp) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("relabel: cannot create {}: {}", outp, e);
            return;
        }
    };
    let shared = Arc::new(Shared {
        nodes,
        soft_nodes,
        per_position,
        hash_mb,
        format,
        input: Mutex::new(Input { reader, format, carry: None, next_block: 0, skip, left: limit, eof: false }),
        output: Mutex::new(Output { out: BufWriter::with_capacity(1 << 20, file), next: 0, pending: BTreeMap::new() }),
        positions: AtomicU64::new(0),
        games: AtomicU64::new(0),
        search_nodes: AtomicU64::new(0),
        search_depth: AtomicU64::new(0),
        mates: AtomicU64::new(0),
        no_move: AtomicU64::new(0),
        bad: AtomicU64::new(0),
        start: Instant::now(),
        last_report: AtomicU64::new(0),
    });
    eprintln!(
        "relabel: {} -> {} nodes {} softnodes {} clear {} hash {} MB threads {} skip {} limit {}",
        inp,
        outp,
        nodes,
        if soft_nodes == u64::MAX { "off".to_string() } else { soft_nodes.to_string() },
        if per_position { "pos" } else { "game" },
        hash_mb,
        threads,
        skip,
        if limit == u64::MAX { "none".to_string() } else { limit.to_string() },
    );
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let shared = Arc::clone(&shared);
            thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || worker(&shared))
                .expect("spawn relabel thread")
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
    let pos = shared.positions.load(Ordering::Relaxed);
    let secs = shared.start.elapsed().as_secs_f64().max(1e-3);
    let n = pos.max(1) as f64;
    eprintln!(
        "relabel{}: positions {} games {} {:.0}s {:.0} pos/s  {:.0} nodes/pos depth {:.2}  mates {} no-move {} bad {}",
        if final_line { " done" } else { "" },
        pos,
        shared.games.load(Ordering::Relaxed),
        secs,
        pos as f64 / secs,
        shared.search_nodes.load(Ordering::Relaxed) as f64 / n,
        shared.search_depth.load(Ordering::Relaxed) as f64 / n,
        shared.mates.load(Ordering::Relaxed),
        shared.no_move.load(Ordering::Relaxed),
        shared.bad.load(Ordering::Relaxed),
    );
}

impl Input {
    fn read_item(&mut self) -> Option<Item> {
        loop {
            let item = match self.format {
                Format::Rec => {
                    let mut raw = vec![0u8; REC];
                    if self.reader.read_exact(&mut raw).is_err() {
                        return None;
                    }
                    let occ = u64::from_le_bytes(raw[..8].try_into().unwrap());
                    Item { result: raw[26], pieces: occ.count_ones(), fullmove: 0, raw }
                }
                Format::Text => {
                    let mut raw = Vec::with_capacity(96);
                    match self.reader.read_until(b'\n', &mut raw) {
                        Ok(0) | Err(_) => return None,
                        Ok(_) => {}
                    }
                    while matches!(raw.last(), Some(b'\n' | b'\r')) {
                        raw.pop();
                    }
                    if raw.is_empty() {
                        continue;
                    }
                    let line = String::from_utf8_lossy(&raw);
                    let fen = line.split('|').next().unwrap_or("");
                    let mut f = fen.split_whitespace();
                    let pieces = f.next().unwrap_or("").bytes().filter(|b| b.is_ascii_alphabetic()).count() as u32;
                    let fullmove = f.nth(4).and_then(|t| t.parse().ok()).unwrap_or(0);
                    let result = match line.rsplit('|').next().map(str::trim) {
                        Some("1.0") | Some("1") => 2,
                        Some("0.5") => 1,
                        _ => 0,
                    };
                    Item { result, pieces, fullmove, raw }
                }
            };
            if self.skip > 0 {
                self.skip -= 1;
                continue;
            }
            return Some(item);
        }
    }

    fn next_item(&mut self) -> Option<Item> {
        if self.left == 0 {
            return None;
        }
        let it = self.carry.take().or_else(|| if self.eof { None } else { self.read_item() });
        if it.is_none() {
            self.eof = true;
        } else {
            self.left -= 1;
        }
        it
    }

    /// The next block of whole games (any cut point with per-position clearing).
    fn next_block(&mut self, per_position: bool) -> Option<(u64, Vec<Item>)> {
        let mut block: Vec<Item> = Vec::with_capacity(BLOCK_MIN + 256);
        while let Some(it) = self.next_item() {
            if block.len() >= BLOCK_MIN && (per_position || starts_game(block.last().unwrap(), &it)) {
                self.carry = Some(it);
                self.left += 1;
                break;
            }
            block.push(it);
        }
        if block.is_empty() {
            return None;
        }
        let id = self.next_block;
        self.next_block += 1;
        Some((id, block))
    }
}

fn starts_game(prev: &Item, it: &Item) -> bool {
    it.result != prev.result || it.pieces > prev.pieces || it.fullmove < prev.fullmove
}

/// FEN for a 32-byte record (castling inferred from home squares), or None if malformed.
fn rec_to_fen(r: &[u8]) -> Option<String> {
    let occ = u64::from_le_bytes(r[..8].try_into().ok()?);
    if occ.count_ones() > 32 || r[27] > 1 {
        return None;
    }
    let mut sq = [0xFFu8; 64];
    let mut n = 0usize;
    for (s, v) in sq.iter_mut().enumerate() {
        if occ >> s & 1 != 0 {
            let p = (r[8 + n / 2] >> (4 * (n & 1))) & 15;
            if p & 7 > 5 {
                return None;
            }
            *v = p;
            n += 1;
        }
    }
    let mut fen = String::with_capacity(90);
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            let p = sq[rank * 8 + file];
            if p == 0xFF {
                empty += 1;
                continue;
            }
            if empty > 0 {
                fen.push((b'0' + empty) as char);
                empty = 0;
            }
            let c = b"pnbrqk"[(p & 7) as usize];
            fen.push(if p & 8 != 0 { c as char } else { c.to_ascii_uppercase() as char });
        }
        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank > 0 {
            fen.push('/');
        }
    }
    fen.push_str(if r[27] == 0 { " w " } else { " b " });
    // White K = 5, R = 3; Black k = 13, r = 11.
    let mut castling = String::new();
    if sq[4] == 5 && sq[7] == 3 {
        castling.push('K');
    }
    if sq[4] == 5 && sq[0] == 3 {
        castling.push('Q');
    }
    if sq[60] == 13 && sq[63] == 11 {
        castling.push('k');
    }
    if sq[60] == 13 && sq[56] == 11 {
        castling.push('q');
    }
    fen.push_str(if castling.is_empty() { "-" } else { &castling });
    fen.push_str(" - 0 1");
    Some(fen)
}

fn worker(shared: &Shared) {
    QUIET_SEARCH.with(|q| q.set(true));
    SOFT_NODE_LIMIT.with(|s| s.set(shared.soft_nodes));
    let mut board = Board::new();
    let mut engine = EngineState::new();
    engine.set_hash(shared.hash_mb);
    engine.own_book = false;
    engine.threads = 1;
    let go = format!("go nodes {}", shared.nodes);
    let div = |b: &Board| if b.nnue.is_some() { crate::eval::NNUE_EVAL_DIV } else { 1 };

    loop {
        let next = shared.input.lock().ok().and_then(|mut i| i.next_block(shared.per_position));
        let Some((id, block)) = next else { break };
        let mut out: Vec<u8> = Vec::with_capacity(block.len() * if shared.format == Format::Rec { REC } else { 100 });
        let (mut nodes, mut depth, mut mates, mut no_move, mut bad, mut games) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for (i, it) in block.iter().enumerate() {
            if shared.per_position || i == 0 || starts_game(&block[i - 1], it) {
                engine.clear_for_new_game();
                games += 1;
            }
            let text_line = if shared.format == Format::Text { Some(String::from_utf8_lossy(&it.raw)) } else { None };
            let fen = match &text_line {
                Some(l) => Some(l.split('|').next().unwrap_or("").trim().to_string()),
                None => rec_to_fen(&it.raw),
            };
            let new_score = fen.filter(|f| f.split_whitespace().count() >= 4).and_then(|fen| {
                board.set_fen(&fen);
                let white = board.side_to_move == Color::White;
                let searched = handle_go(&go, &mut board, &mut engine);
                nodes += engine.last_search_nodes;
                depth += LAST_SEARCH_DEPTH.with(|d| d.get()) as u64;
                if searched.is_none() {
                    no_move += 1;
                    return None;
                }
                let s = LAST_SEARCH_SCORE.with(|s| s.get());
                if s.abs() >= MATE_THRESHOLD {
                    mates += 1;
                }
                let ws = if white { s } else { -s };
                Some(if ws.abs() >= MATE_THRESHOLD { ws.signum() * SCORE_CAP } else { ws * div(&board) })
            });
            match shared.format {
                Format::Rec => {
                    let mut r = it.raw.clone();
                    match new_score {
                        Some(s) => r[24..26].copy_from_slice(&(s.clamp(-SCORE_CAP, SCORE_CAP) as i16).to_le_bytes()),
                        None => bad += 1,
                    }
                    out.extend_from_slice(&r);
                }
                Format::Text => {
                    let line = text_line.unwrap();
                    let mut parts: Vec<&str> = line.split('|').collect();
                    let s;
                    if let (Some(v), true) = (new_score, parts.len() == 3) {
                        s = format!(" {} ", v);
                        parts[1] = &s;
                    } else {
                        bad += 1;
                    }
                    out.extend_from_slice(parts.join("|").as_bytes());
                    out.push(b'\n');
                }
            }
        }
        shared.positions.fetch_add(block.len() as u64, Ordering::Relaxed);
        shared.games.fetch_add(games, Ordering::Relaxed);
        shared.search_nodes.fetch_add(nodes, Ordering::Relaxed);
        shared.search_depth.fetch_add(depth, Ordering::Relaxed);
        shared.mates.fetch_add(mates, Ordering::Relaxed);
        shared.no_move.fetch_add(no_move, Ordering::Relaxed);
        shared.bad.fetch_add(bad, Ordering::Relaxed);
        if let Ok(mut o) = shared.output.lock() {
            o.pending.insert(id, out);
            loop {
                let next = o.next;
                let Some(t) = o.pending.remove(&next) else { break };
                let _ = o.out.write_all(&t);
                o.next += 1;
            }
        }
        let secs = shared.start.elapsed().as_secs();
        let last = shared.last_report.load(Ordering::Relaxed);
        if secs >= last + REPORT_EVERY_SECS
            && shared.last_report.compare_exchange(last, secs, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            report(shared, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rec_fen_roundtrip_and_castling_inference() {
        // Startpos as a record: occupancy, nibbles in square order, score, result, stm.
        let mut b = Board::new();
        b.set_fen("r3k2r/pppq1ppp/8/8/8/8/PPP2PPP/R3K1R1 b Qkq - 3 12");
        let mut r = [0u8; REC];
        let mut occ = 0u64;
        let mut n = 0;
        for s in 0..64 {
            if let Some(p) = b.squares[s] {
                occ |= 1 << s;
                let code = (p.kind as u8) | if p.color == Color::Black { 8 } else { 0 };
                r[8 + n / 2] |= code << (4 * (n & 1));
                n += 1;
            }
        }
        r[..8].copy_from_slice(&occ.to_le_bytes());
        r[27] = 1;
        // PieceType discriminants must match the trainer's "pnbrqk" order.
        assert_eq!(PieceType::Pawn as u8, 0);
        assert_eq!(PieceType::King as u8, 5);
        let fen = rec_to_fen(&r).unwrap();
        assert_eq!(fen, "r3k2r/pppq1ppp/8/8/8/8/PPP2PPP/R3K1R1 b Qkq - 0 1");
    }
}
