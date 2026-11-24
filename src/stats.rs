//! SEE-scale audit counters (feature `stats`; see-diag branch only).
//! `statsdump <path>` writes the counters as JSON, `statsreset` clears them.
//! Sampled decision records go to the file named by env SEELOG (one line
//! per record, space-separated fields). Single-threaded searches only.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::Write;

thread_local! {
    static STATS: RefCell<HashMap<&'static str, Vec<u64>>> = RefCell::new(HashMap::new());
    /// > 0 while a verification search runs: its nodes and decisions are not counted.
    pub static SUSPEND: Cell<u32> = const { Cell::new(0) };
    static RNG: Cell<u64> = const { Cell::new(0x9e37_79b9_7f4a_7c15) };
    static LOG: RefCell<Option<std::io::BufWriter<std::fs::File>>> = RefCell::new(
        std::env::var("SEELOG").ok().and_then(|p| std::fs::File::create(p).ok()).map(std::io::BufWriter::new)
    );
}

#[inline]
pub fn suspended() -> bool {
    SUSPEND.with(|s| s.get() > 0)
}

pub fn suspend(on: bool) {
    SUSPEND.with(|c| c.set(if on { c.get() + 1 } else { c.get() - 1 }));
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

/// True once every `n` calls on average (xorshift).
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

pub fn log(line: std::fmt::Arguments) {
    LOG.with(|l| {
        if let Some(w) = l.borrow_mut().as_mut() {
            let _ = w.write_fmt(line);
            let _ = w.write_all(b"\n");
        }
    });
}

pub fn reset() {
    STATS.with(|s| s.borrow_mut().clear());
}

pub fn dump(path: &str) {
    LOG.with(|l| {
        if let Some(w) = l.borrow_mut().as_mut() {
            let _ = w.flush();
        }
    });
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
