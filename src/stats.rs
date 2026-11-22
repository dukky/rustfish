//! Search statistics for diagnosis (feature `stats`; nothing is compiled
//! without it). Counters are keyed by a static name and a flat index; the
//! `statsdump <path>` UCI command writes them as JSON and `statsreset`
//! clears them. Single-threaded searches only (thread-local counters).
//!
//! Index helpers: `nd(nt, d)` = node type x depth bucket, `ndk(nt, d, k)` adds
//! a third axis of width K. Node types: 0 PV, 1 expected cut, 2 expected all.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

pub const DB: usize = 16;
pub const K: usize = 16;

thread_local! {
    static STATS: RefCell<HashMap<&'static str, Vec<u64>>> = RefCell::new(HashMap::new());
    /// > 0 while a verification search runs: its nodes are not counted.
    pub static SUSPEND: Cell<u32> = const { Cell::new(0) };
    static RNG: Cell<u64> = const { Cell::new(0x9e37_79b9_7f4a_7c15) };
    /// Current iterative-deepening depth (for the leaf ply-excess histogram).
    pub static ROOT_DEPTH: Cell<i32> = const { Cell::new(0) };
}

#[inline]
pub fn db(depth: i32) -> usize {
    (depth.max(0) as usize).min(DB - 1)
}

#[inline]
pub fn nd(nt: usize, depth: i32) -> usize {
    nt * DB + db(depth)
}

#[inline]
pub fn ndk(nt: usize, depth: i32, k: usize) -> usize {
    nd(nt, depth) * K + k.min(K - 1)
}

#[inline]
pub fn suspended() -> bool {
    SUSPEND.with(|s| s.get() > 0)
}

pub fn add(key: &'static str, idx: usize, v: u64) {
    if suspended() {
        return;
    }
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        let e = s.entry(key).or_default();
        if e.len() <= idx {
            e.resize(idx + 1, 0);
        }
        e[idx] += v;
    });
}

#[inline]
pub fn inc(key: &'static str, idx: usize) {
    add(key, idx, 1);
}

/// True once every `n` calls on average (xorshift), for sampled verification.
#[allow(dead_code)]
pub fn sample(n: u64) -> bool {
    RNG.with(|r| {
        let mut x = r.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        r.set(x);
        x % n == 0
    })
}

pub fn reset() {
    STATS.with(|s| s.borrow_mut().clear());
}

pub fn dump(path: &str) {
    let text = STATS.with(|s| {
        let s = s.borrow();
        let mut keys: Vec<_> = s.keys().copied().collect();
        keys.sort();
        let mut out = String::from("{\n");
        for (i, k) in keys.iter().enumerate() {
            let vals: Vec<String> = s[k].iter().map(|v| v.to_string()).collect();
            out.push_str(&format!("  \"{}\": [{}]{}\n", k, vals.join(","), if i + 1 < keys.len() { "," } else { "" }));
        }
        out.push('}');
        out
    });
    if let Err(e) = std::fs::write(path, text) {
        eprintln!("statsdump: {e}");
    }
}
pub use crate::knobs::knob;
