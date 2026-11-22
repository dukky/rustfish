//! Diagnosis-only integer env knobs (features `stats` or `knobs`).
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static KNOBS: RefCell<HashMap<&'static str, i32>> = RefCell::new(HashMap::new());
}

/// Integer env knob (read once per thread), for diagnosis-only experiments.
pub fn knob(name: &'static str, default: i32) -> i32 {
    KNOBS.with(|k| {
        *k.borrow_mut().entry(name).or_insert_with(|| {
            std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        })
    })
}

thread_local! {
    /// Qsearch nesting depth (0 = the first qsearch node below negamax).
    pub static QPLY: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
