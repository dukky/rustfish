//! policy-order v0, gate A: oracle move ordering (feature `policy-oracle`).
//!
//! `oracletest <deep> <shallow> <arm> [<arm> ...] | position ...`:
//! search `deep` plies, keep that search's TT as a read-only oracle, then for
//! each arm clear all search state and search `shallow` plies with the
//! oracle's best move for each node (if its draft is at least the node's
//! depth + margin) placed in the move order as the arm says. One line of
//! counters per arm. Arm = `mode:margin:pct:noise`:
//!   mode 0 = off (oracle only looked up for the counters)
//!        1 = oracle move first, before the TT move
//!        2 = oracle move right after the TT move
//!        3 = like 2, but only when the oracle move is a quiet
//!        4 = oracle quiet first among the quiets (after good captures,
//!            ahead of killers/counter): what a bounded quiet bonus can do
//!        5 = oracle quiet first among the non-priority quiets only
//!        6 = wrong-oracle control: another quiet takes mode 4's slot
//!        7 = imperfect scorer: mode 4 at pct% of oracle nodes, mode 6 at the rest
//!   margin = draft slack (entry.depth >= node depth + margin)
//!   pct    = apply at this share (0-100) of nodes, chosen by key hash
//!   noise  = history bonus x noise/100 (no-op control seeds: 99/100/101)
//! The root itself is never touched (search_root orders it).

use super::*;
use std::cell::RefCell;

pub(crate) const NB: usize = 3; // depth buckets 1-3, 4-7, 8+
pub(crate) const NT: usize = 3; // node type: pv, cut, all
pub(crate) const NC: usize = 8; // cutter classes

pub(crate) fn dbucket(depth: i32) -> usize {
    if depth <= 3 {
        0
    } else if depth <= 7 {
        1
    } else {
        2
    }
}

#[derive(Default, Clone)]
pub(crate) struct Counters {
    pub loops: [u64; NB],
    pub oracle_avail: [u64; NB],
    pub oracle_is_tt: [u64; NB],
    pub oracle_quiet_not_tt: [u64; NB],
    pub oracle_applied: [u64; NB],
    pub fh: [[u64; NT]; NB],
    pub fh_first: [[u64; NT]; NB],
    pub fh_moves: [[u64; NT]; NB],
    /// fail highs whose cutter was not the first searched move, by class
    pub late: [[[u64; NC]; NT]; NB],
    /// nodes spent on the moves searched before such a cutter
    pub late_waste: [[[u64; NC]; NT]; NB],
    /// late cutters the oracle (looked up, applied or not) named
    pub late_oracle_hit: [[[u64; NC]; NT]; NB],
    /// fail highs with an oracle move: cutter == oracle
    pub fh_with_oracle: [u64; NB],
    pub fh_cutter_is_oracle: [u64; NB],
    /// loop nodes and their total move-loop nodes
    pub loop_nodes: [[u64; NT]; NB],
    pub fh_loop_nodes: [[u64; NT]; NB],
}

pub(crate) struct State {
    pub oracle: Option<TransTable>,
    pub mode: u8,
    pub margin: i32,
    pub pct: u64,
    pub noise: i32,
    pub c: Counters,
}

thread_local! {
    pub(crate) static ST: RefCell<State> = RefCell::new(State {
        oracle: None, mode: 0, margin: 0, pct: 100, noise: 100, c: Counters::default(),
    });
}

pub(crate) fn noise() -> i32 {
    ST.with(|s| s.borrow().noise)
}

/// Oracle move for this node, and whether the arm applies it here.
pub(crate) fn lookup(key: u64, depth: i32) -> (Option<Move>, u8) {
    ST.with(|s| {
        let s = s.borrow();
        let Some(o) = s.oracle.as_ref() else { return (None, 0) };
        let mv = o
            .probe(key)
            .filter(|e| e.depth as i32 >= depth + s.margin)
            .and_then(|e| e.best);
        let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
        if s.mode == 7 {
            // Imperfect scorer: right (mode 4) at pct% of nodes, else wrong (mode 6).
            return (mv, if mv.is_none() { 0 } else if (h % 100) < s.pct { 4 } else { 6 });
        }
        let on = mv.is_some() && (h % 100) < s.pct;
        (mv, if on { s.mode } else { 0 })
    })
}

pub(crate) fn with<R>(f: impl FnOnce(&mut Counters) -> R) -> R {
    ST.with(|s| f(&mut s.borrow_mut().c))
}

fn print_counters(tag: &str, c: &Counters, nodes: u64, bm: &str, score: i32) {
    let j = |v: &[u64]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let j2 = |v: &[[u64; NT]; NB]| v.iter().map(|r| j(r)).collect::<Vec<_>>().join(";");
    let j3 = |v: &[[[u64; NC]; NT]; NB]| {
        v.iter()
            .map(|r| r.iter().map(|x| j(x)).collect::<Vec<_>>().join("/"))
            .collect::<Vec<_>>()
            .join(";")
    };
    println!(
        "ARM {} nodes={} bm={} score={} loops={} avail={} istt={} qnott={} applied={} fh={} fhfirst={} fhmoves={} late={} waste={} latehit={} fhwo={} fhco={} loopnodes={} fhloopnodes={}",
        tag, nodes, bm, score, j(&c.loops), j(&c.oracle_avail), j(&c.oracle_is_tt), j(&c.oracle_quiet_not_tt),
        j(&c.oracle_applied), j2(&c.fh), j2(&c.fh_first), j2(&c.fh_moves), j3(&c.late), j3(&c.late_waste),
        j3(&c.late_oracle_hit), j(&c.fh_with_oracle), j(&c.fh_cutter_is_oracle), j2(&c.loop_nodes), j2(&c.fh_loop_nodes)
    );
}

pub(crate) fn handle_oracletest(cmd: &str, board: &mut Board, engine: &mut EngineState) {
    // `oracletest <deep> <shallow> <arms...> | position fen ...`
    let (head, pos) = cmd.split_once('|').unwrap_or((cmd, "position startpos"));
    let pos = pos.trim().to_string();
    let t: Vec<&str> = head.split_whitespace().collect();
    let deep: i32 = t.get(1).and_then(|x| x.parse().ok()).unwrap_or(17);
    let shallow: i32 = t.get(2).and_then(|x| x.parse().ok()).unwrap_or(13);
    let arms: Vec<&str> = t.iter().skip(3).copied().collect();
    let hash = engine.hash_mb;
    crate::search::QUIET_SEARCH.with(|q| q.set(true));

    // Deep search: its TT becomes the oracle (256 MB so little is evicted).
    ST.with(|s| {
        let mut s = s.borrow_mut();
        s.oracle = None;
        s.mode = 0;
        s.noise = 100;
    });
    engine.set_hash(256);
    engine.clear_for_new_game();
    handle_position(&pos, board, engine);
    let dbm = crate::search::handle_go(&format!("go depth {}", deep), board, engine).unwrap_or_default();
    let dscore = crate::search::LAST_SEARCH_SCORE.with(|s| s.get());
    println!("DEEP depth={} nodes={} bm={} score={}", deep, engine.last_search_nodes, dbm, dscore);
    let oracle = std::mem::replace(&mut engine.tt, TransTable::new(1));
    ST.with(|s| s.borrow_mut().oracle = Some(oracle));

    for arm in arms {
        let p: Vec<i64> = arm.split(':').map(|x| x.parse().unwrap_or(0)).collect();
        ST.with(|s| {
            let mut s = s.borrow_mut();
            s.mode = p.first().copied().unwrap_or(0) as u8;
            s.margin = p.get(1).copied().unwrap_or(0) as i32;
            s.pct = p.get(2).copied().unwrap_or(100) as u64;
            s.noise = p.get(3).copied().unwrap_or(100) as i32;
            s.c = Counters::default();
        });
        engine.set_hash(hash);
        engine.clear_for_new_game();
        handle_position(&pos, board, engine);
        let bm = crate::search::handle_go(&format!("go depth {}", shallow), board, engine).unwrap_or_default();
        let score = crate::search::LAST_SEARCH_SCORE.with(|s| s.get());
        let nodes = engine.last_search_nodes;
        ST.with(|s| print_counters(arm, &s.borrow().c, nodes, &bm, score));
    }
    ST.with(|s| {
        let mut s = s.borrow_mut();
        s.oracle = None;
        s.mode = 0;
        s.noise = 100;
    });
    crate::search::QUIET_SEARCH.with(|q| q.set(false));
    flush_stdout();
}
