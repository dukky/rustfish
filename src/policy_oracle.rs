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
//!   mindepth (optional 5th field) = oracle only at nodes of at least this depth
//!   badcp (optional 6th field) = "bad move" threshold for the LMR modes
//! Reductions / root modes (upper bounds for a policy used outside ordering):
//!        10 = LMR: the oracle best move reduces one ply less
//!        11 = LMR: oracle-bad moves (child's oracle upper bound at least
//!             badcp below the node's oracle lower bound; the child entry's
//!             draft at least the reduced search's depth + margin) reduce
//!             one ply more
//!        12 = 10 + 11
//!        13 = 12 right at pct% of nodes, inverted (best +1, bad -1) at the rest
//!        14 / 15 = control: every LMR-reduced move +1 / -1 (no oracle)
//!        16 = root only: the deep search's best move first at the root
//!        17 / 18 = every LMR-reduced move +1 / +2, the oracle best -1
//!        19 = control: every LMR-reduced move +2 (no oracle)
//!        (17/18: pct = recall, the share of nodes where the best is protected)
//!        20 = control: 18 with a pseudo-random ~0.5% of moves protected
//!        22 / 23 = LMR +2 with a real scorer protecting moves (see scorer_adjust)
//!        21 = base tree, dumps every LMR-reduced quiet (`R` lines) at nodes
//!             where the oracle best is among them, plus 5% of all nodes
//! A `v=<depth>` token verifies every distinct root move chosen (search of
//! the child at that depth) and prints `VER <move> <score>` for value loss.
//! Outside mode 16 the root itself is never touched (search_root orders it).

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
    /// LMR-reduced moves seen; oracle best among them; oracle-bad among them
    pub lmr: [u64; 3],
}

pub(crate) struct State {
    pub oracle: Option<TransTable>,
    pub mode: u8,
    pub margin: i32,
    pub pct: u64,
    pub noise: i32,
    pub min_depth: i32,
    pub bad_cp: i32,
    pub c: Counters,
}

thread_local! {
    pub(crate) static ST: RefCell<State> = RefCell::new(State {
        oracle: None, mode: 0, margin: 0, pct: 100, noise: 100, min_depth: 0, bad_cp: 100, c: Counters::default(),
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
        if depth < s.min_depth {
            return (None, 0);
        }
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

fn usable(score: i32) -> bool {
    score.abs() < 20_000
}

/// Reduction delta for an LMR-reduced quiet (modes 10-15).
pub(crate) fn lmr_adjust(key: u64, child_key: u64, depth: i32, r: i32, po_mv: Option<Move>, mv: Move) -> i32 {
    ST.with(|s| {
        let mut s = s.borrow_mut();
        let mode = s.mode;
        if !(10..=20).contains(&mode) {
            return 0;
        }
        s.c.lmr[0] += 1;
        if mode == 20 {
            // Contamination control for 18: +2 everywhere, but a pseudo-random
            // reduced move (about 0.5%, the oracle best's rate) is protected.
            let r = ((mv.from as u64) << 6 | mv.to as u64) ^ key;
            if (r.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) % 200 == 0 {
                s.c.lmr[1] += 1;
                return -1;
            }
            return 2;
        }
        match mode {
            14 => return 1,
            15 => return -1,
            19 => return 2,
            _ => {}
        }
        if depth < s.min_depth {
            return 0;
        }
        if mode == 17 || mode == 18 {
            // Aggressive LMR made safe by a perfect policy: every reduced
            // move +1 (17) / +2 (18), the oracle best one ply less than main.
            // pct = recall: at the other nodes the policy misses the best move.
            let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
            let best = po_mv == Some(mv) && (h % 100) < s.pct;
            if best {
                s.c.lmr[1] += 1;
                return -1;
            }
            return if mode == 17 { 1 } else { 2 };
        }
        let is_best = po_mv == Some(mv);
        let mut is_bad = false;
        if !is_best {
            if let Some(o) = s.oracle.as_ref() {
                let node = o.probe(key).filter(|e| {
                    e.depth as i32 >= depth + s.margin && matches!(e.bound, Bound::Exact | Bound::Lower) && usable(e.score)
                });
                let child = o.probe(child_key).filter(|e| {
                    e.depth as i32 >= (depth - 1 - r).max(0) + s.margin && matches!(e.bound, Bound::Exact | Bound::Lower) && usable(e.score)
                });
                if let (Some(n), Some(c)) = (node, child) {
                    is_bad = -c.score <= n.score - s.bad_cp;
                }
            }
        }
        if is_best {
            s.c.lmr[1] += 1;
        }
        if is_bad {
            s.c.lmr[2] += 1;
        }
        let (best_d, bad_d) = match mode {
            10 => (-1, 0),
            11 => (0, 1),
            12 => (-1, 1),
            _ => {
                let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
                if (h % 100) < s.pct { (-1, 1) } else { (1, -1) }
            }
        };
        if is_best {
            best_d
        } else if is_bad {
            bad_d
        } else {
            0
        }
    })
}

/// Mode 16: the oracle's best move for the root.
pub(crate) fn root_first(key: u64) -> Option<Move> {
    ST.with(|s| {
        let s = s.borrow();
        if s.mode != 16 {
            return None;
        }
        s.oracle.as_ref().and_then(|o| o.probe(key)).and_then(|e| e.best)
    })
}

// ---- mode 21: LMR recall dump (tree = base arm) ----
thread_local! {
    static DUMP: RefCell<(Vec<Vec<String>>, Vec<bool>, Vec<bool>, u64)> = RefCell::new((Vec::new(), Vec::new(), Vec::new(), 0));
}

pub(crate) fn dumping() -> bool {
    ST.with(|s| matches!(s.borrow().mode, 21 | 22 | 23))
}

thread_local! {
    static LIN_W: Vec<f64> = std::env::var("POLICY_W")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.split_whitespace().filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_default();
}

/// recall.py's linear features of a dump row (f = the `R` fields after the label).
fn lin_feats(f: &[i32]) -> Vec<f64> {
    let clip = |x: i32, a: i32| x.clamp(-a, a) as f64;
    let (depth, nt, idx, r, h, ch, k0, k1, cm, chk, see, kind, fa, ta, se, ce, imp, ttb) =
        (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17]);
    let mut v = vec![
        h as f64 / 4000.0, ch as f64 / 4000.0, k0 as f64, k1 as f64, cm as f64, chk as f64,
        clip(see, 900) / 300.0, (see < 0) as i32 as f64, fa as f64, ta as f64,
        clip(ce, 600) / 200.0, (ce > -25) as i32 as f64,
        clip(if se > -90000 { se } else { -600 }, 600) / 200.0,
        imp as f64, (1.0 + idx as f64).ln(), r as f64 / 3.0, depth as f64 / 10.0,
    ];
    for i in 0..6 {
        v.push((kind == i) as i32 as f64);
    }
    for i in 0..3 {
        v.push((nt == i) as i32 as f64);
    }
    for i in 0..4 {
        v.push((ttb == i) as i32 as f64);
    }
    v
}

/// Modes 22 / 23: LMR +2 everywhere, but a move the scorer protects gets
/// main's reduction - 1. 22 = recall.py's linear scorer (weights from the
/// file in POLICY_W) above badcp/100; 23 = child eval - alpha above badcp.
pub(crate) fn scorer_adjust(f: &[i32]) -> Option<i32> {
    let (mode, t) = ST.with(|s| {
        let s = s.borrow();
        (s.mode, s.bad_cp)
    });
    let protect = match mode {
        22 => {
            let x = lin_feats(f);
            let sc: f64 = LIN_W.with(|w| w.iter().zip(x.iter()).map(|(a, b)| a * b).sum());
            sc > t as f64 / 100.0
        }
        23 => f[15] > t,
        _ => return None,
    };
    ST.with(|s| s.borrow_mut().c.lmr[0] += 1);
    if protect {
        ST.with(|s| s.borrow_mut().c.lmr[1] += 1);
    }
    Some(if protect { -1 } else { 2 })
}

fn flush_ply(d: &mut (Vec<Vec<String>>, Vec<bool>, Vec<bool>, u64), ply: usize) {
    if ply < d.0.len() {
        if d.1[ply] || d.2[ply] {
            for r in d.0[ply].iter() {
                println!("{}", r);
            }
        }
        d.0[ply].clear();
        d.1[ply] = false;
        d.2[ply] = false;
    }
}

/// A new move-loop node at this ply: the previous one there is finished.
pub(crate) fn node_start(ply: usize, key: u64) {
    if !dumping() {
        return;
    }
    DUMP.with(|d| {
        let mut d = d.borrow_mut();
        while d.0.len() <= ply {
            d.0.push(Vec::new());
            d.1.push(false);
            d.2.push(false);
        }
        flush_ply(&mut d, ply);
        d.3 += 1;
        // 5% of nodes kept whatever their labels (negatives for precision).
        d.2[ply] = (key.wrapping_mul(0xD6E8_FEB8_6659_FD93) >> 32) % 20 == 0;
    });
}

pub(crate) fn dump_flush_all() {
    DUMP.with(|d| {
        let mut d = d.borrow_mut();
        for p in 0..d.0.len() {
            flush_ply(&mut d, p);
        }
    });
}

/// One LMR-reduced quiet: `R node sampled label depth nt idx r h ch k0 k1 cm chk see kind fromatt toatt se childeval-alpha improving ttbound hasoracle`.
pub(crate) fn dump_row(ply: usize, label: bool, f: &[i32]) {
    DUMP.with(|d| {
        let mut d = d.borrow_mut();
        if ply >= d.0.len() {
            return;
        }
        let node = d.3;
        let sampled = d.2[ply] as u8;
        let mut line = format!("R {} {} {}", node, sampled, label as u8);
        for x in f {
            line.push(' ');
            line.push_str(&x.to_string());
        }
        d.0[ply].push(line);
        if label {
            d.1[ply] = true;
        }
    });
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
        "ARM {} nodes={} bm={} score={} loops={} avail={} istt={} qnott={} applied={} fh={} fhfirst={} fhmoves={} late={} waste={} latehit={} fhwo={} fhco={} loopnodes={} fhloopnodes={} lmr={}",
        tag, nodes, bm, score, j(&c.loops), j(&c.oracle_avail), j(&c.oracle_is_tt), j(&c.oracle_quiet_not_tt),
        j(&c.oracle_applied), j2(&c.fh), j2(&c.fh_first), j2(&c.fh_moves), j3(&c.late), j3(&c.late_waste),
        j3(&c.late_oracle_hit), j(&c.fh_with_oracle), j(&c.fh_cutter_is_oracle), j2(&c.loop_nodes), j2(&c.fh_loop_nodes), j(&c.lmr)
    );
}

pub(crate) fn handle_oracletest(cmd: &str, board: &mut Board, engine: &mut EngineState) {
    // `oracletest <deep> <shallow> <arms...> | position fen ...`
    let (head, pos) = cmd.split_once('|').unwrap_or((cmd, "position startpos"));
    let pos = pos.trim().to_string();
    let t: Vec<&str> = head.split_whitespace().collect();
    let deep: i32 = t.get(1).and_then(|x| x.parse().ok()).unwrap_or(17);
    let shallow: i32 = t.get(2).and_then(|x| x.parse().ok()).unwrap_or(13);
    let mut verify = 0;
    let arms: Vec<&str> = t
        .iter()
        .skip(3)
        .copied()
        .filter(|a| match a.strip_prefix("v=") {
            Some(v) => {
                verify = v.parse().unwrap_or(0);
                false
            }
            None => true,
        })
        .collect();
    let mut chosen: Vec<String> = Vec::new();
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
            s.min_depth = p.get(4).copied().unwrap_or(0) as i32;
            s.bad_cp = p.get(5).copied().unwrap_or(100) as i32;
            s.c = Counters::default();
        });
        engine.set_hash(hash);
        engine.clear_for_new_game();
        handle_position(&pos, board, engine);
        let bm = crate::search::handle_go(&format!("go depth {}", shallow), board, engine).unwrap_or_default();
        let score = crate::search::LAST_SEARCH_SCORE.with(|s| s.get());
        let nodes = engine.last_search_nodes;
        dump_flush_all();
        ST.with(|s| print_counters(arm, &s.borrow().c, nodes, &bm, score));
        if !chosen.contains(&bm) {
            chosen.push(bm.clone());
        }
    }
    if verify > 0 {
        if !chosen.contains(&dbm) {
            chosen.push(dbm.clone());
        }
        ST.with(|s| {
            let mut s = s.borrow_mut();
            s.mode = 0;
            s.noise = 100;
            s.min_depth = 0;
        });
        for mv in chosen.iter().filter(|m| m.len() >= 4 && m.len() <= 5) {
            let p2 = if pos.contains(" moves ") { format!("{} {}", pos, mv) } else { format!("{} moves {}", pos, mv) };
            engine.set_hash(hash);
            engine.clear_for_new_game();
            handle_position(&p2, board, engine);
            crate::search::handle_go(&format!("go depth {}", verify), board, engine);
            let sc = -crate::search::LAST_SEARCH_SCORE.with(|s| s.get());
            println!("VER {} {}", mv, sc);
        }
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
