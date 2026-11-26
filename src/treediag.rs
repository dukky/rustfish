//! Tree-gap diagnosis (branch diag-tree-gap only, not for merge).
//!
//! Knobs: `setoption name DiagKnobs value ce=0,lmpd=6` changes one search
//! mechanism at run time (defaults reproduce main exactly, bench unchanged).
//! Counters (cargo feature `treestats`): where the nodes of a search go,
//! printed as one `info string treestats ...` line after each `go`.

use std::sync::atomic::{AtomicI32, Ordering};

pub(crate) const K_CE: usize = 0; // check extension: 1 on (main), 0 off
pub(crate) const K_LMPD: usize = 1; // LMP max depth (main 3); d>3 limit = 4 + d*d
pub(crate) const K_LMRCAP: usize = 2; // 1: LMR bad captures, 2: all late captures
pub(crate) const K_LMRS: usize = 3; // LMR table scale, percent (main 100)
pub(crate) const K_NMPE: usize = 4; // NMP extra reduction (se-beta)/NMPE, max 3 (0 off)
pub(crate) const K_NMPR: usize = 5; // NMP additive reduction (0)
pub(crate) const K_SED: usize = 6; // singular extension min depth (8)
pub(crate) const K_ASPW: usize = 7; // aspiration window (35)
pub(crate) const K_RFPD: usize = 8; // RFP max depth (7)
pub(crate) const K_QSD: usize = 9; // qsearch delta-prune margin (0 off)
pub(crate) const K_IIRD: usize = 10; // IIR min depth (4)
pub(crate) const K_FUTD: usize = 11; // per-move futility max depth (8)
pub(crate) const K_HPD: usize = 12; // history prune max depth (0 off)
pub(crate) const K_HPT: usize = 13; // history prune threshold per depth (4000)
pub(crate) const K_SEX: usize = 14; // singular extension size (1)
pub(crate) const K_CEX: usize = 15; // check ext only if SEE(check move) >= 0 / ... (0 plain)
pub(crate) const K_LMRMI: usize = 16; // add to LMR min move index (0)
pub(crate) const K_PVR: usize = 17; // LMR: extra reduction at all nodes (non-PV, non-cut) (0)
pub(crate) const K_TTPV: usize = 18; // 1: allow TT cutoffs at PV nodes when depth >= needed+TTPV-1
pub(crate) const K_LMRD: usize = 19; // re-search after LMR fail-high: 1 = skip if reduced score < alpha+K
pub(crate) const K_SEM: usize = 20; // singular margin per depth (2)
pub(crate) const K_EXL: usize = 21; // extensions only while ply*10 < EXL*root depth (0 off)
pub(crate) const K_SXM: usize = 22; // singular extension only if verify < singular beta - SXM (0)
pub(crate) const K_SENT: usize = 23; // 1: no positive singular extension at expected ALL nodes
pub(crate) const K_EXLC: usize = 24; // 1: EXL also limits the check extension
const NK: usize = 32;

const NAMES: [&str; 25] = [
    "ce", "lmpd", "lmrcap", "lmrs", "nmpe", "nmpr", "sed", "aspw", "rfpd", "qsd", "iird", "futd",
    "hpd", "hpt", "sex", "cex", "lmrmi", "allr", "ttpv", "lmrd", "sem", "exl", "sxm", "sent", "exlc",
];
const DEFAULTS: [i32; 25] = [1, 3, 0, 100, 0, 0, 8, 35, 7, 0, 4, 8, 0, 4000, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0];

static KNOBS: [AtomicI32; NK] = [const { AtomicI32::new(0) }; NK];
static KNOBS_INIT: std::sync::Once = std::sync::Once::new();

fn init() {
    KNOBS_INIT.call_once(|| {
        for (i, d) in DEFAULTS.iter().enumerate() {
            KNOBS[i].store(*d, Ordering::Relaxed);
        }
    });
}

#[inline(always)]
pub(crate) fn k(i: usize) -> i32 {
    init();
    KNOBS[i].load(Ordering::Relaxed)
}

/// `a=1,b=2` (also `;` or spaces). Unknown names are reported and ignored.
pub(crate) fn set_knobs(spec: &str) {
    init();
    for (i, d) in DEFAULTS.iter().enumerate() {
        KNOBS[i].store(*d, Ordering::Relaxed);
    }
    for part in spec.split(|c: char| c == ',' || c == ';' || c == ':' || c.is_whitespace()) {
        let Some((n, v)) = part.split_once(|c: char| c == '=' || c == '_') else { continue };
        match (NAMES.iter().position(|x| *x == n), v.parse::<i32>()) {
            (Some(i), Ok(v)) => KNOBS[i].store(v, Ordering::Relaxed),
            _ => println!("info string DiagKnobs: bad item {part}"),
        }
    }
    let s: Vec<String> = NAMES.iter().enumerate().map(|(i, n)| format!("{n}={}", k(i))).collect();
    println!("info string DiagKnobs {}", s.join(","));
}

// ---------------------------------------------------------------- counters

pub(crate) const DB: usize = 24; // depth buckets 0..23 (23 = 23+)
pub(crate) const NT: usize = 3; // node type: 0 PV, 1 expected cut, 2 expected all

// exit reasons of a negamax node with depth > 0
pub(crate) const X_TT: usize = 0;
pub(crate) const X_RAZOR: usize = 1; // futility d1-2 to qsearch
pub(crate) const X_RFP: usize = 2;
pub(crate) const X_NMP: usize = 3;
pub(crate) const X_PROBCUT: usize = 4;
pub(crate) const X_MULTICUT: usize = 5;
pub(crate) const X_FAILHIGH: usize = 6; // move loop beta cutoff
pub(crate) const X_FAILLOW: usize = 7;
pub(crate) const X_EXACT: usize = 8;
pub(crate) const X_OTHER: usize = 9; // mate/stalemate/excluded/stop
pub(crate) const NX: usize = 10;

// subtree-node attribution (inclusive: nested sub-searches count in each)
pub(crate) const S_NMP: usize = 0;
pub(crate) const S_PROBCUT: usize = 1;
pub(crate) const S_SINGULAR: usize = 2;
pub(crate) const S_FIRST: usize = 3; // first move child (full window)
pub(crate) const S_LATE_RED: usize = 4; // late move, reduced null-window search
pub(crate) const S_LATE_FULL: usize = 5; // late move, unreduced null-window search
pub(crate) const S_LMR_RE: usize = 6; // re-search after reduced fail high
pub(crate) const S_PVS_RE: usize = 7; // full-window re-search
pub(crate) const S_ASP_FAIL: usize = 8; // root iterations that failed the window
pub(crate) const S_CHECKEXT: usize = 9; // subtrees of nodes that took a check extension (at d>=1)
pub(crate) const NS: usize = 10;

pub(crate) const MI: usize = 8; // move index buckets: 0,1,2,3-4,5-7,8-12,13-20,21+
pub(crate) fn mi_bucket(idx: usize) -> usize {
    match idx {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=7 => 4,
        8..=12 => 5,
        13..=20 => 6,
        _ => 7,
    }
}

pub(crate) struct Stats {
    pub nodes_nt_d: [[u64; DB]; NT],
    pub exit: [[[u64; NX]; DB]; NT],
    pub sub: [[u64; DB]; NS],
    pub sub_cnt: [[u64; DB]; NS],
    pub child_nodes_mi: [[[u64; MI]; DB]; NT], // subtree nodes by parent depth and move index
    pub cut_mi: [[u64; MI]; DB], // move-loop fail highs by cutting move index
    pub moves_searched: [[u64; DB]; NT], // moves searched at nodes that ended in the move loop
    pub loop_nodes: [[u64; DB]; NT],
    pub ext_check: [u64; DB],
    pub ext_se: [u64; DB],
    pub ext_neg: [u64; DB],
    pub se_tries: [u64; DB],
    pub se_tries_t: [u64; NT],
    pub ext_se_t: [u64; NT],
    pub nmp_tries: [u64; DB],
    pub nmp_ok: [u64; DB],
    pub lmr_red: [u64; DB],
    pub lmr_re: [u64; DB],
    pub lmr_re_ok: [u64; DB], // re-search also beat alpha
    pub pvs_re: [u64; DB],
    pub tt_probe: [u64; DB],
    pub tt_hit: [u64; DB],
    pub tt_deep: [u64; DB], // hit with enough depth
    pub qs_nodes: u64,
    pub qs_incheck: u64,
    pub qs_standpat_cut: u64,
    pub qs_root: u64, // qsearch entered from negamax (depth<=0 or razor)
    pub seldepth: u64,
    pub asp_fails: u64,
    pub asp_iters: u64,
    pub plies: [u64; 128], // node count by ply from root (negamax + qsearch)
}

#[cfg(feature = "treestats")]
pub(crate) static mut ST: Stats = unsafe { std::mem::zeroed() };

#[cfg(feature = "treestats")]
#[allow(static_mut_refs)]
pub(crate) fn st() -> &'static mut Stats {
    // Single-threaded diagnosis only.
    unsafe { &mut ST }
}

#[cfg(feature = "treestats")]
pub(crate) fn reset() {
    *st() = unsafe { std::mem::zeroed() };
}

#[inline(always)]
pub(crate) fn db(d: i32) -> usize {
    d.clamp(0, DB as i32 - 1) as usize
}

#[cfg(feature = "treestats")]
pub(crate) fn print() {
    let s = st();
    let mut out = String::from("info string treestats");
    fn flat<const N: usize>(out: &mut String, name: &str, v: &[u64; N]) {
        out.push(' ');
        out.push_str(name);
        out.push('=');
        let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
        out.push_str(&parts.join(","));
    }
    for t in 0..NT {
        flat(&mut out, &format!("nodes_t{t}"), &s.nodes_nt_d[t]);
        flat(&mut out, &format!("ms_t{t}"), &s.moves_searched[t]);
        flat(&mut out, &format!("loop_t{t}"), &s.loop_nodes[t]);
        for x in 0..NX {
            let v: [u64; DB] = std::array::from_fn(|d| s.exit[t][d][x]);
            flat(&mut out, &format!("exit_t{t}_x{x}"), &v);
        }
        for m in 0..MI {
            let v: [u64; DB] = std::array::from_fn(|d| s.child_nodes_mi[t][d][m]);
            flat(&mut out, &format!("cmi_t{t}_m{m}"), &v);
        }
    }
    for k in 0..NS {
        flat(&mut out, &format!("sub{k}"), &s.sub[k]);
        flat(&mut out, &format!("subc{k}"), &s.sub_cnt[k]);
    }
    for m in 0..MI {
        let v: [u64; DB] = std::array::from_fn(|d| s.cut_mi[d][m]);
        flat(&mut out, &format!("cut_m{m}"), &v);
    }
    flat(&mut out, "ext_check", &s.ext_check);
    flat(&mut out, "ext_se", &s.ext_se);
    flat(&mut out, "ext_neg", &s.ext_neg);
    flat(&mut out, "se_tries", &s.se_tries);
    flat(&mut out, "se_tries_t", &s.se_tries_t);
    flat(&mut out, "ext_se_t", &s.ext_se_t);
    flat(&mut out, "nmp_tries", &s.nmp_tries);
    flat(&mut out, "nmp_ok", &s.nmp_ok);
    flat(&mut out, "lmr_red", &s.lmr_red);
    flat(&mut out, "lmr_re", &s.lmr_re);
    flat(&mut out, "lmr_re_ok", &s.lmr_re_ok);
    flat(&mut out, "pvs_re", &s.pvs_re);
    flat(&mut out, "tt_probe", &s.tt_probe);
    flat(&mut out, "tt_hit", &s.tt_hit);
    flat(&mut out, "tt_deep", &s.tt_deep);
    flat(&mut out, "plies", &s.plies);
    out.push_str(&format!(
        " qs_nodes={} qs_incheck={} qs_standpat_cut={} qs_root={} seldepth={} asp_fails={} asp_iters={}",
        s.qs_nodes, s.qs_incheck, s.qs_standpat_cut, s.qs_root, s.seldepth, s.asp_fails, s.asp_iters
    ));
    println!("{out}");
}

/// `ts!(s => statements)` runs the statements with `s` = the counters, only
/// in a `treestats` build.
#[macro_export]
macro_rules! ts {
    ($s:ident => $($t:tt)*) => {
        #[cfg(feature = "treestats")]
        {
            let $s = $crate::treediag::st();
            $($t)*
        }
    };
}
