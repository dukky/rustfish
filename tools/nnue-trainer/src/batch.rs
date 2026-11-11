//! Fast batch gradient + AdamW for `train`.
//!
//! The v2 loop gave every thread a full private gradient (zeroed and summed
//! over threads every batch) and scattered each sample's 2H-wide dacc into
//! rows 2 KB apart, i.e. L2/L3 traffic of ~3x the FT per sample. Here:
//!
//! - **Decode** runs first for the whole batch; each sub-batch's
//!   `(sample, side)` pairs are also counting-sorted (stably) by the side's
//!   king bucket.
//! - **Forward** runs over blocks of `FB` sorted pairs: each pair's
//!   accumulators are built 64 floats at a time in registers (bias + active
//!   rows, in feature order, exactly as `forward`). A block reads the rows
//!   of one or two buckets, so with king buckets that slice of the column
//!   tile stays in L2. The **output layer** (dot product, sigmoid, loss,
//!   output gradient) then runs over blocks of samples.
//! - **Backward** is split by *columns*: unit `c` owns columns
//!   `16c..16c+16` (one cache line) of every gradient row, for every sample
//!   of the sub-batch, in sample order. One gradient buffer, no reduction,
//!   no false sharing; its FT part is stored tile-major
//!   (`[tile][row][16]`), so a unit's gradient is one contiguous
//!   `rows * 64` B block that stays in L1/L2.
//! - The batch runs in sub-batches of `sb` samples so the stored
//!   accumulators stay in cache, as one stream of units without barriers:
//!   the forward of sub-batch k, the output layer of k-1 and the backward
//!   of k-2 run side by side, and each unit waits only for the units whose
//!   results it reads (see `run`).
//! - AdamW reads the one gradient, updates and re-zeroes it, in parallel.
//!   It writes the FT's params and moments with non-temporal stores, so
//!   that (on AMD's victim-cache L3) the weights and gradient the next
//!   batch reads are still in L3 (kb8m H 512 does not fit otherwise).
//!
//! Every gradient element is summed in batch order, so results do not
//! depend on the thread count, the scheduling, or the ISA (all SIMD
//! implementations are lane-wise exact). With `exact` the output dot product
//! is the sequential sum of `forward`, and a batch is bit-identical to the
//! v2 trainer run with `--threads 1`; without it the dot product uses 16
//! interleaved partial sums (float reassociation only).

use crate::simd::{Isa, Portable, Simd};
use crate::{decode, sigmoid, Adam, KingLayout, Rec, Sample, Shape, CLIP, INPUTS};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::thread::Thread;


/// Backward column tile (floats): one 64-byte cache line.
pub const TB: usize = 16;
/// Forward register tile (floats).
const TF: usize = 64;
/// Samples per forward unit.
const FB: usize = 32;
/// FT rows per AdamW unit (unfused `adam`).
#[cfg_attr(not(test), allow(dead_code))]
const AB: usize = 16;
/// Fused AdamW units per column tile.
const ADAM_SPLIT: usize = 8;
/// Accumulator buffers (sub-batches in flight).
const NACC: usize = 3;
/// Largest king-bucket count (one bucket per square).
const MAX_KB: usize = 64;

/// King bucket of a sample's perspective (every feature of a side lies in
/// its bucket's block of rows).
#[inline(always)]
fn king_bucket(s: &Sample, side: usize) -> usize {
    if s.n == 0 {
        0
    } else {
        s.f[side][0] as usize / INPUTS
    }
}

// ---------------------------------------------------------------- pool

/// Persistent workers that all run one job per `run`; the caller is worker 0.
/// Idle workers spin briefly, then park.
pub struct Pool {
    n: usize,
    gen: AtomicUsize,
    done: AtomicUsize,
    stop: AtomicBool,
    job: std::cell::UnsafeCell<Option<*const (dyn Fn(usize) + Sync)>>,
    threads: std::sync::Mutex<Vec<Thread>>,
}

unsafe impl Sync for Pool {}

impl Pool {
    /// Runs `body(&pool)` with `n - 1` worker threads alive.
    pub fn with<R>(n: usize, body: impl FnOnce(&Pool) -> R) -> R {
        let pool = Pool {
            n: n.max(1),
            gen: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: std::cell::UnsafeCell::new(None),
            threads: std::sync::Mutex::new(Vec::new()),
        };
        std::thread::scope(|sc| {
            let hs: Vec<_> = (1..pool.n).map(|i| sc.spawn({
                let pool = &pool;
                move || pool.worker(i)
            })).collect();
            *pool.threads.lock().unwrap() = hs.iter().map(|h| h.thread().clone()).collect();
            let r = body(&pool);
            pool.stop.store(true, Ordering::Release);
            pool.gen.fetch_add(1, Ordering::Release);
            for t in pool.threads.lock().unwrap().iter() {
                t.unpark();
            }
            r
        })
    }

    fn worker(&self, id: usize) {
        let mut seen = 0usize;
        loop {
            let mut spins = 0u32;
            loop {
                let g = self.gen.load(Ordering::Acquire);
                if g != seen {
                    seen = g;
                    break;
                }
                spins += 1;
                if spins < 4096 {
                    std::hint::spin_loop();
                } else {
                    std::thread::park();
                }
            }
            if self.stop.load(Ordering::Acquire) {
                return;
            }
            // SAFETY: `run` stored the job before bumping `gen` and keeps it
            // alive until every worker has bumped `done`.
            let job = unsafe { (*self.job.get()).unwrap() };
            unsafe { (*job)(id) };
            self.done.fetch_add(1, Ordering::Release);
        }
    }

    /// Runs `f(worker_id)` on every worker (ids `0..n`) and waits for all.
    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.n == 1 {
            f(0);
            return;
        }
        // SAFETY: only the lifetime is erased; `f` outlives this call, and
        // no worker touches it after the `done` count completes.
        let ptr: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        unsafe { *self.job.get() = Some(ptr) };
        self.done.store(0, Ordering::Relaxed);
        self.gen.fetch_add(1, Ordering::Release);
        for t in self.threads.lock().unwrap().iter() {
            t.unpark();
        }
        f(0);
        let mut spins = 0u32;
        while self.done.load(Ordering::Acquire) < self.n - 1 {
            spins += 1;
            if spins < 1 << 16 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// `units` work items pulled dynamically by every worker.
fn par_units(pool: &Pool, units: usize, f: &(dyn Fn(usize) + Sync)) {
    let next = AtomicUsize::new(0);
    pool.run(&|_| loop {
        let u = next.fetch_add(1, Ordering::Relaxed);
        if u >= units {
            break;
        }
        f(u);
    });
}

// ---------------------------------------------------------------- buffers

/// 64-byte aligned, zeroed f32 buffer.
pub struct ABuf {
    v: Vec<f32>,
    off: usize,
    len: usize,
}

impl ABuf {
    pub fn new(len: usize) -> Self {
        let v = vec![0.0f32; len + 16];
        let off = (16 - (v.as_ptr() as usize / 4) % 16) % 16;
        ABuf { v, off, len }
    }
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn as_slice(&self) -> &[f32] {
        &self.v[self.off..self.off + self.len]
    }
    fn ptr(&mut self) -> Ptr {
        Ptr(self.v[self.off..].as_mut_ptr())
    }
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}
impl Ptr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct APtr(*const AtomicU32);
unsafe impl Send for APtr {}
unsafe impl Sync for APtr {}
impl APtr {
    #[inline(always)]
    fn get(self) -> *const AtomicU32 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct UPtr(*mut u32);
unsafe impl Send for UPtr {}
unsafe impl Sync for UPtr {}
impl UPtr {
    #[inline(always)]
    fn get(self) -> *mut u32 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct SPtr(*mut Sample);
unsafe impl Send for SPtr {}
unsafe impl Sync for SPtr {}
impl SPtr {
    #[inline(always)]
    fn get(self) -> *mut Sample {
        self.0
    }
}

// ---------------------------------------------------------------- trainer

pub struct BatchTrainer {
    pub sh: Shape,
    rows: usize,
    tiles: usize,
    sb: usize,
    pub isa: Isa,
    pub exact: bool,
    /// FT weight gradient, `[tile][row][TB]`.
    gft: ABuf,
    /// Forward tile width (64, or 16 if H is no multiple of 64) and a copy
    /// of the FT weights in `[tile][row][tf]` order, so that a forward tile
    /// pass reads one contiguous block (row-major rows 2 KB apart alias in
    /// the caches). Kept in sync by `adam`.
    tf: usize,
    wt: ABuf,
    wt_ok: bool,
    /// Gradient of params `ft_b()..len()` (FT bias, output weights, output
    /// bias), standard layout.
    grest: ABuf,
    acc: [ABuf; NACC],
    /// The batch's decoded samples.
    samples: Vec<Sample>,
    /// Per sub-batch, its `(sample << 1) | side` pairs stably sorted by the
    /// side's king bucket (`2 * sb` slots per sub-batch).
    pairs: Vec<u32>,
    grad: Vec<f32>,
    loss: Vec<f32>,
    /// `--factorise`: length of the shared block `F` (768 x H) stored after
    /// the net's params in the training vector, 0 without it. The forward
    /// pass then reads the merged rows `clamp(W + F)` from `wt`.
    fl: usize,
    /// `--wd-active-only`: per FT row, the last batch (`gen`) it occurred
    /// in (empty if off).
    active: Vec<AtomicU32>,
    gen: u32,
    /// `step`: the FT's training state, tile-major (`[tile][row][TB]`, the
    /// `rows` FT rows then, with the factoriser, F's 768 rows), so that the
    /// AdamW of a column tile is one sequential stream. It is the master
    /// copy while `ft_live`; `export` writes it back to `p` and the moments.
    ftp: ABuf,
    ftm: ABuf,
    ftv: ABuf,
    ft_live: bool,
}

impl BatchTrainer {
    /// `sb` = sub-batch size (0 = automatic: ~2 MB of accumulators).
    pub fn new(sh: Shape, sb: usize, exact: bool, fl: usize, wd_active: bool) -> Self {
        assert!(fl == 0 || (fl == INPUTS * sh.h && sh.kb > 1), "factoriser needs KB > 1");
        assert!(sh.h % TB == 0 && sh.h > 0, "--hidden must be a multiple of {TB}");
        let sb = if sb == 0 { ((2 << 20) / (8 * sh.h)).clamp(128, 2048) } else { sb };
        let sb = sb.div_ceil(FB) * FB;
        let rows = sh.kb * INPUTS;
        let blank = Sample { f: [[0; 32]; 2], n: 0, bucket: 0, target: 0.0 };
        assert!(sh.kb <= MAX_KB);
        BatchTrainer {
            sh,
            rows,
            tiles: sh.h / TB,
            sb,
            isa: Isa::detect(),
            exact,
            gft: ABuf::new(rows * sh.h),
            tf: if sh.h % TF == 0 { TF } else { TB },
            wt: ABuf::new(rows * sh.h),
            wt_ok: false,
            grest: ABuf::new(sh.len() - sh.ft_b()),
            acc: std::array::from_fn(|_| ABuf::new(sb * 2 * sh.h)),
            samples: vec![blank; sb],
            pairs: Vec::new(),
            grad: Vec::new(),
            loss: Vec::new(),
            fl,
            active: if wd_active { (0..rows).map(|_| AtomicU32::new(0)).collect() } else { Vec::new() },
            gen: 0,
            ftp: ABuf::new(0),
            ftm: ABuf::new(0),
            ftv: ABuf::new(0),
            ft_live: false,
        }
    }

    pub fn sub_batch(&self) -> usize {
        self.sb
    }

    /// Accumulates the gradient of the summed loss over `recs` (record `i`
    /// mirrored iff `mirror[i]`) into the internal buffers; returns the
    /// summed loss. Call `adam` to apply and clear it. Training uses `step`
    /// (the same results); these two are its reference in the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn batch(&mut self, pool: &Pool, p: &[f32], recs: &[Rec], mirror: &[bool], lambda: f32, kl: &KingLayout) -> f64 {
        assert!(!self.ft_live, "batch/adam after step: export first");
        self.run(pool, p, recs, mirror, lambda, kl, None)
    }

    /// `batch` + `adam` in one pass, bit-identical to them: each column
    /// unit of the last sub-batch's backward runs the AdamW of its tile
    /// while the tile's gradient is in L2, on the trainer's tile-major copy
    /// of the FT state (`p`'s and `opt`'s FT parts are stale until
    /// `export`; the FT bias and output layer are updated in `p` and `opt`).
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, pool: &Pool, p: &mut [f32], opt: &mut Adam, recs: &[Rec], mirror: &[bool], lambda: f32, kl: &KingLayout, lr: f32, wd: f32) -> f64 {
        if !self.ft_live {
            self.import(p, opt);
        }
        let c = adam_consts(opt, lr, wd, 1.0 / recs.len() as f32);
        let loss = self.run(pool, p, recs, mirror, lambda, kl, Some(&c));
        // FT bias and output layer (all of their gradient is in by now).
        let ftb = self.sh.ft_b();
        let rest = self.sh.len() - ftb;
        unsafe {
            let a = AdamArgs {
                p: p.as_mut_ptr().add(ftb),
                m: opt.m.as_mut_ptr().add(ftb),
                v: opt.v.as_mut_ptr().add(ftb),
                g: self.grest.ptr().get(),
                wt: std::ptr::null_mut(),
                tf: self.tf,
                h: self.sh.h,
                rows: self.rows,
                tiles: self.tiles,
                f0: 0,
                f1: 0,
                flat: rest,
                n: self.sh.len(),
                fl: self.fl,
                kb: self.sh.kb,
                active: std::ptr::null(),
                gen: 0,
            };
            adam_dispatch(self.isa, &c, &a);
        }
        loss
    }

    fn ft_rows(&self) -> usize {
        self.rows + if self.fl > 0 { INPUTS } else { 0 }
    }

    /// Index in `p` (and the moments) of FT-state row `r`, column `j`.
    fn ft_src(&self, r: usize, j: usize) -> usize {
        let h = self.sh.h;
        if r < self.rows {
            r * h + j
        } else {
            self.sh.len() + (r - self.rows) * h + j
        }
    }

    /// Loads the tile-major FT state from `p` and `opt`.
    fn import(&mut self, p: &[f32], opt: &Adam) {
        let (rows, len) = (self.ft_rows(), self.ft_rows() * self.sh.h);
        if self.ftp.len != len {
            (self.ftp, self.ftm, self.ftv) = (ABuf::new(len), ABuf::new(len), ABuf::new(len));
        }
        let (dp, dm, dv) = (self.ftp.ptr().get(), self.ftm.ptr().get(), self.ftv.ptr().get());
        for t in 0..self.tiles {
            for r in 0..rows {
                let (src, dst) = (self.ft_src(r, t * TB), (t * rows + r) * TB);
                unsafe {
                    std::ptr::copy_nonoverlapping(p.as_ptr().add(src), dp.add(dst), TB);
                    std::ptr::copy_nonoverlapping(opt.m.as_ptr().add(src), dm.add(dst), TB);
                    std::ptr::copy_nonoverlapping(opt.v.as_ptr().add(src), dv.add(dst), TB);
                }
            }
        }
        self.sync_weights(p);
        self.ft_live = true;
    }

    /// Writes the FT state of `step` back to `p` and `opt` (call before
    /// reading `p`'s FT, e.g. to validate or checkpoint). `step` can go on.
    pub fn export(&mut self, p: &mut [f32], opt: &mut Adam) {
        if !self.ft_live {
            return;
        }
        let rows = self.ft_rows();
        let (sp, sm, sv) = (self.ftp.ptr().get(), self.ftm.ptr().get(), self.ftv.ptr().get());
        for t in 0..self.tiles {
            for r in 0..rows {
                let (dst, src) = (self.ft_src(r, t * TB), (t * rows + r) * TB);
                unsafe {
                    std::ptr::copy_nonoverlapping(sp.add(src), p.as_mut_ptr().add(dst), TB);
                    std::ptr::copy_nonoverlapping(sm.add(src), opt.m.as_mut_ptr().add(dst), TB);
                    std::ptr::copy_nonoverlapping(sv.add(src), opt.v.as_mut_ptr().add(dst), TB);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&mut self, pool: &Pool, p: &[f32], recs: &[Rec], mirror: &[bool], lambda: f32, kl: &KingLayout, fused: Option<&AdamC>) -> f64 {
        let n = recs.len();
        self.gen += 1;
        let gen = self.gen;
        assert_eq!(mirror.len(), n);
        assert_eq!(p.len(), self.sh.len() + self.fl);
        self.grad.resize(n, 0.0);
        self.loss.resize(n, 0.0);
        let blank = Sample { f: [[0; 32]; 2], n: 0, bucket: 0, target: 0.0 };
        self.samples.resize(n, blank);
        if !self.wt_ok {
            self.sync_weights(p);
        }
        let (sh, rows, tiles, sb, isa, exact, tf) = (self.sh, self.rows, self.tiles, self.sb, self.isa, self.exact, self.tf);
        let nsb = n.div_ceil(sb);
        self.pairs.resize(2 * nsb * sb, 0);
        let wt = self.wt.ptr();
        let gft = self.gft.ptr();
        let grest = self.grest.ptr();
        let accs: [Ptr; NACC] = std::array::from_fn(|i| self.acc[i].ptr());
        let samples = SPtr(self.samples.as_mut_ptr());
        let pairs = UPtr(self.pairs.as_mut_ptr());
        let grad = Ptr(self.grad.as_mut_ptr());
        let loss = Ptr(self.loss.as_mut_ptr());
        let active = &self.active[..];
        let sb_len = |k: usize| sb.min(n - k * sb);
        let ftr = self.ft_rows();
        let ft = [self.ftp.ptr(), self.ftm.ptr(), self.ftv.ptr()];
        let (fl, kb) = (self.fl > 0, sh.kb);
        let act = APtr(if active.is_empty() { std::ptr::null() } else { active.as_ptr() });
        let decode_one = |r: usize, s: &mut Sample| {
            *s = decode(&recs[r], lambda, sh.buckets, mirror[r], kl);
            if !active.is_empty() {
                // Store only if unset: every thread writing the same hot
                // flags makes their lines bounce between cores.
                for side in 0..2 {
                    for &f in &s.f[side][..s.n] {
                        let a = &active[f as usize];
                        if a.load(Ordering::Relaxed) != gen {
                            a.store(gen, Ordering::Relaxed);
                        }
                    }
                }
            }
        };
        // KB = 1: the sorted order is the identity, so the forward units
        // decode their own samples (no extra phase).
        let sorted = sh.kb > 1;
        // Decode every sub-batch and bucket-sort its pairs (counting sort,
        // stable: within a bucket the pairs keep sample-then-side order).
        par_units(pool, if sorted { nsb } else { 0 }, &|k| unsafe {
            let (base, len) = (k * sb, sb_len(k));
            let ss = std::slice::from_raw_parts_mut(samples.get().add(base), len);
            let mut start = [0u32; MAX_KB + 1];
            for (i, s) in ss.iter_mut().enumerate() {
                decode_one(base + i, s);
                for side in 0..2 {
                    start[king_bucket(s, side) + 1] += 1;
                }
            }
            for b in 1..=MAX_KB {
                start[b] += start[b - 1];
            }
            let out = pairs.get().add(2 * base);
            for (i, s) in ss.iter().enumerate() {
                for side in 0..2 {
                    let b = &mut start[king_bucket(s, side)];
                    *out.add(*b as usize) = (i as u32) << 1 | side as u32;
                    *b += 1;
                }
            }
        });
        // One stream of units, no barriers. "Phase" q lists the backward
        // of sub-batch q - 2 (one unit per column tile), the accumulators of
        // q and the output layer of q - 1 (then, with `fused`, AdamW in
        // `ADAM_SPLIT` row ranges per tile). Units are claimed in list
        // order and wait only for what they read; every wait is on an
        // earlier unit, so all of them have owners and finish.
        //   backward (k, c): output layer of k, and backward (k - 1, c)
        //   accumulators (k): backward of k - NACC (frees acc buffer k % NACC)
        //   output layer (k): accumulators of k
        //   AdamW (c): backward (nsb - 1, c)
        let fwd_units = |k: usize| (2 * sb_len(k)).div_ceil(FB);
        let out_units = |k: usize| sb_len(k).div_ceil(FB);
        let phase = |q: usize| {
            let back = if (2..nsb + 2).contains(&q) { tiles } else { 0 };
            let fwd = if q < nsb { fwd_units(q) } else { 0 };
            let out = if (1..nsb + 1).contains(&q) { out_units(q - 1) } else { 0 };
            let adam = if q == nsb + 1 && fused.is_some() { tiles * ADAM_SPLIT } else { 0 };
            (back, fwd, out, adam)
        };
        let mut starts = vec![0usize; nsb + 3];
        for q in 0..nsb + 2 {
            let (b, f, o, a) = phase(q);
            starts[q + 1] = starts[q] + b + f + o + a;
        }
        let counter = |n: usize| (0..n).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
        let (acc_done, out_done, bwd_done, tile_k) = (counter(nsb), counter(nsb), counter(nsb), counter(tiles));
        let wait = |ready: &dyn Fn() -> bool| {
            let mut spins = 0u32;
            while !ready() {
                spins += 1;
                if spins < 1 << 12 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        };
        let ft_lines = if fl { INPUTS } else { rows };
        {
            par_units(pool, starts[nsb + 2], &|g| unsafe {
                let q = starts.partition_point(|&x| x <= g) - 1;
                let u = g - starts[q];
                let (back, fwd, out, _) = phase(q);
                if u >= back + fwd + out {
                    let (t, part) = ((u - back - fwd - out) / ADAM_SPLIT, (u - back - fwd - out) % ADAM_SPLIT);
                    wait(&|| tile_k[t].load(Ordering::Acquire) == nsb);
                    let o = t * ftr * TB;
                    let a = TileAdamArgs {
                        p: ft[0].get().add(o),
                        m: ft[1].get().add(o),
                        v: ft[2].get().add(o),
                        g: gft.get().add(t * rows * TB),
                        wt: wt.get(),
                        tf,
                        rows,
                        c: t,
                        r0: ft_lines * part / ADAM_SPLIT,
                        r1: ft_lines * (part + 1) / ADAM_SPLIT,
                        fl,
                        kb,
                        active: act.get(),
                        gen,
                    };
                    tile_adam_dispatch(isa, fused.unwrap(), &a);
                    return;
                }
                if u < back {
                    let k = q - 2;
                    wait(&|| out_done[k].load(Ordering::Acquire) == out_units(k) && tile_k[u].load(Ordering::Acquire) == k);
                    let len = sb_len(k);
                    let ss = std::slice::from_raw_parts(samples.get().add(k * sb), len);
                    let gr = std::slice::from_raw_parts(grad.get().add(k * sb), len);
                    let a = BwdArgs { sh, rows, p, ss, acc: accs[k % NACC].get(), grad: gr, gft: gft.get(), grest: grest.get(), c: u };
                    bwd_dispatch(isa, &a);
                    tile_k[u].store(k + 1, Ordering::Release);
                    bwd_done[k].fetch_add(1, Ordering::AcqRel);
                } else if u < back + fwd {
                    let k = q;
                    let j = u - back;
                    if k >= NACC {
                        wait(&|| bwd_done[k - NACC].load(Ordering::Acquire) == tiles);
                    }
                    let len = sb_len(k);
                    let (p0, p1) = (j * FB, ((j + 1) * FB).min(2 * len));
                    if !sorted {
                        // Identity order: pairs p0..p1 are samples p0/2..p1/2.
                        let ss = std::slice::from_raw_parts_mut(samples.get().add(k * sb), len);
                        for i in p0 / 2..p1 / 2 {
                            decode_one(k * sb + i, &mut ss[i]);
                            for side in 0..2 {
                                *pairs.get().add(2 * k * sb + 2 * i + side) = (i as u32) << 1 | side as u32;
                            }
                        }
                    }
                    let ss = std::slice::from_raw_parts(samples.get().add(k * sb), len);
                    let pr = std::slice::from_raw_parts(pairs.get().add(2 * k * sb + p0), p1 - p0);
                    let a = AccArgs { sh, rows, tf, p, wt: wt.get(), ss, pairs: pr, acc: accs[k % NACC].get() };
                    acc_dispatch(isa, &a);
                    acc_done[k].fetch_add(1, Ordering::AcqRel);
                } else {
                    let k = q - 1;
                    let j = u - back - fwd;
                    wait(&|| acc_done[k].load(Ordering::Acquire) == fwd_units(k));
                    let len = sb_len(k);
                    let (s0, s1) = (j * FB, ((j + 1) * FB).min(len));
                    let base = k * sb;
                    let ss = std::slice::from_raw_parts(samples.get().add(base + s0), s1 - s0);
                    let a = OutArgs {
                        sh,
                        tf,
                        p,
                        ss,
                        acc: accs[k % NACC].get().add(s0 * 2 * sh.h),
                        grad: grad.get().add(base + s0),
                        loss: loss.get().add(base + s0),
                    };
                    out_dispatch(isa, exact, &a);
                    out_done[k].fetch_add(1, Ordering::AcqRel);
                }
            });
        }
        self.loss.iter().map(|&l| l as f64).sum()
    }

    /// Rebuilds the tile-major FT copy (merged rows with the factoriser)
    /// from the training vector `p` (call after changing `p` other than
    /// through `adam`).
    pub fn sync_weights(&mut self, p: &[f32]) {
        let (h, rows, tf, n) = (self.sh.h, self.rows, self.tf, self.sh.len());
        let wt = self.wt.ptr().get();
        for r in 0..rows {
            for j in 0..h {
                let mut x = p[r * h + j];
                if self.fl > 0 {
                    x = (x + p[n + (r % INPUTS) * h + j]).clamp(-CLIP, CLIP);
                }
                unsafe { *wt.add((j / tf * rows + r) * tf + j % tf) = x };
            }
        }
        self.wt_ok = true;
    }

    /// One AdamW step (same maths and constants as v2) from the accumulated
    /// gradient times `inv_batch`, with clipping; clears the gradient.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn adam(&mut self, pool: &Pool, p: &mut [f32], opt: &mut Adam, lr: f32, wd: f32, inv_batch: f32) {
        assert!(!self.ft_live, "batch/adam after step: export first");
        let c = adam_consts(opt, lr, wd, inv_batch);
        let gen = self.gen;
        let (h, rows, tiles, ftb) = (self.sh.h, self.rows, self.tiles, self.sh.ft_b());
        let (pp, mp, vp) = (Ptr(p.as_mut_ptr()), Ptr(opt.m.as_mut_ptr()), Ptr(opt.v.as_mut_ptr()));
        if !self.wt_ok {
            self.sync_weights(p);
        }
        let (wt, tf) = (self.wt.ptr(), self.tf);
        let gft = self.gft.ptr();
        let grest = self.grest.ptr();
        let n = self.sh.len();
        assert_eq!(p.len(), n + self.fl);
        let rest = n - ftb;
        let (fl, kb) = (self.fl, self.sh.kb);
        // With the factoriser a unit takes features (all buckets' rows of
        // each, plus its F row); without it, plain FT rows.
        let units_rows = if fl > 0 { INPUTS } else { rows };
        let ft_units = units_rows.div_ceil(AB);
        let isa = self.isa;
        let active = if self.active.is_empty() { std::ptr::null() } else { self.active.as_ptr() };
        let active = APtr(active);
        par_units(pool, ft_units + 1, &|u| unsafe {
            let a = if u < ft_units {
                let (f0, f1) = (u * AB, ((u + 1) * AB).min(units_rows));
                let (p, m, v, g, wt, active) = (pp.get(), mp.get(), vp.get(), gft.get(), wt.get(), active.get());
                AdamArgs { p, m, v, g, wt, tf, h, rows, tiles, f0, f1, flat: 0, n, fl, kb, active, gen }
            } else {
                let (p, m, v) = (pp.get().add(ftb), mp.get().add(ftb), vp.get().add(ftb));
                let (g, wt, active) = (grest.get(), wt.get(), std::ptr::null());
                AdamArgs { p, m, v, g, wt, tf, h, rows, tiles, f0: 0, f1: 0, flat: rest, n, fl, kb, active, gen: 0 }
            };
            adam_dispatch(isa, &c, &a);
        });
    }

    /// The accumulated gradient in the standard parameter layout (tests).
    #[cfg(test)]
    pub fn gradient(&self) -> Vec<f32> {
        let (h, rows) = (self.sh.h, self.rows);
        let mut g = vec![0.0; self.sh.len()];
        let gft = self.gft.as_slice();
        for f in 0..rows {
            for t in 0..self.tiles {
                g[f * h + t * TB..f * h + t * TB + TB].copy_from_slice(&gft[(t * rows + f) * TB..][..TB]);
            }
        }
        g[self.sh.ft_b()..].copy_from_slice(self.grest.as_slice());
        g
    }
}

// ---------------------------------------------------------------- kernels

struct AccArgs<'a> {
    sh: Shape,
    rows: usize,
    tf: usize,
    p: &'a [f32],
    /// Tile-major FT weights (`BatchTrainer::wt`).
    wt: *const f32,
    /// The sub-batch's samples, and the block of its bucket-sorted pairs.
    ss: &'a [Sample],
    pairs: &'a [u32],
    /// The sub-batch's accumulators (`ss.len() * 2h`).
    acc: *mut f32,
}

struct OutArgs<'a> {
    sh: Shape,
    tf: usize,
    p: &'a [f32],
    /// A block of samples and its accumulators, output gradients and losses.
    ss: &'a [Sample],
    acc: *const f32,
    grad: *mut f32,
    loss: *mut f32,
}

struct BwdArgs<'a> {
    sh: Shape,
    rows: usize,
    p: &'a [f32],
    ss: &'a [Sample],
    acc: *const f32,
    grad: &'a [f32],
    gft: *mut f32,
    grest: *mut f32,
    /// Column tile.
    c: usize,
}

/// Bias + active rows for one `TW`-column tile, in registers. `tile` is
/// row 0 of the tile in the tile-major FT copy, `bias` the tile's bias.
#[inline(always)]
unsafe fn fwd_tile<V: Simd, const TW: usize>(tile: *const f32, bias: *const f32, feats: &[u16], out: *mut f32) {
    let n = TW / V::W;
    let mut r = [V::zero(); 16];
    for (k, x) in r[..n].iter_mut().enumerate() {
        *x = V::load(bias.add(k * V::W));
    }
    for &f in feats {
        let row = tile.add(f as usize * TW);
        for (k, x) in r[..n].iter_mut().enumerate() {
            *x = x.add(V::load(row.add(k * V::W)));
        }
    }
    for (k, x) in r[..n].iter().enumerate() {
        x.store(out.add(k * V::W));
    }
}

/// Adds `sum(clamp(a)^2 * w)` over `TW` floats into 16 interleaved lanes
/// (`lanes[j]` sums the elements with index = j mod 16).
#[inline(always)]
unsafe fn dot_tile<V: Simd, const TW: usize>(a: *const f32, w: *const f32, lanes: *mut f32) {
    let per16 = 16 / V::W;
    let mut l = [V::zero(); 16];
    for (k, x) in l[..per16].iter_mut().enumerate() {
        *x = V::load(lanes.add(k * V::W));
    }
    for k in 0..TW / V::W {
        let x = V::load(a.add(k * V::W)).clamp01();
        let l = &mut l[k % per16];
        *l = l.add(x.mul(x).mul(V::load(w.add(k * V::W))));
    }
    for (k, x) in l[..per16].iter().enumerate() {
        x.store(lanes.add(k * V::W));
    }
}

/// One column tile of every pair in the block. The pairs are sorted by
/// king bucket, so a block reads the rows of one or two buckets: that part
/// of the tile stays in L1/L2 while the block reuses it.
#[inline(always)]
unsafe fn acc_cols<V: Simd, const TW: usize>(a: &AccArgs, col: usize) {
    let h = a.sh.h;
    let bias = a.p.as_ptr().add(a.sh.ft_b() + col);
    let tile = a.wt.add(col / TW * a.rows * TW);
    for &pr in a.pairs {
        let (j, side) = ((pr >> 1) as usize, (pr & 1) as usize);
        let s = &a.ss[j];
        fwd_tile::<V, TW>(tile, bias, &s.f[side][..s.n], a.acc.add(j * 2 * h + side * h + col));
    }
}

#[inline(always)]
unsafe fn acc_impl<V: Simd>(a: &AccArgs) {
    for col in (0..a.sh.h).step_by(a.tf) {
        if a.tf == TF {
            acc_cols::<V, TF>(a, col);
        } else {
            acc_cols::<V, TB>(a, col);
        }
    }
}

/// Output layer, loss and output gradient of a block of samples.
#[inline(always)]
unsafe fn out_impl<V: Simd, const EXACT: bool>(a: &OutArgs) {
    let sh = a.sh;
    let h = sh.h;
    for (j, s) in a.ss.iter().enumerate() {
        let acc = a.acc.add(j * 2 * h);
        let ow = a.p.as_ptr().add(sh.out_w() + s.bucket * 2 * h);
        let sum = if EXACT {
            // `forward`'s sequential sum.
            let mut sum = 0.0f32;
            for i in 0..2 * h {
                let x = (*acc.add(i)).clamp(0.0, 1.0);
                sum += x * x * *ow.add(i);
            }
            sum
        } else {
            // 16 interleaved lanes, filled tile by tile (side 0 then 1).
            let mut l = [0.0f32; 16];
            for side in 0..2 {
                for col in (0..h).step_by(a.tf) {
                    let (x, w) = (acc.add(side * h + col), ow.add(side * h + col));
                    if a.tf == TF {
                        dot_tile::<V, TF>(x, w, l.as_mut_ptr());
                    } else {
                        dot_tile::<V, TB>(x, w, l.as_mut_ptr());
                    }
                }
            }
            let mut w = 8;
            while w > 0 {
                for i in 0..w {
                    l[i] += l[i + w];
                }
                w /= 2;
            }
            l[0]
        };
        let mut out = a.p[sh.out_b() + s.bucket];
        out += sum;
        let pred = sigmoid(out);
        let err = pred - s.target;
        *a.grad.add(j) = 2.0 * err * pred * (1.0 - pred);
        *a.loss.add(j) = err * err;
    }
}

/// Column tile `c` of the sub-batch's gradient, in sample order.
#[inline(always)]
unsafe fn bwd_impl<V: Simd>(a: &BwdArgs) {
    let sh = a.sh;
    let h = sh.h;
    let n = TB / V::W;
    let col = a.c * TB;
    let gtile = a.gft.add(a.c * a.rows * TB);
    let gb = a.grest.add(col);
    let gow0 = a.grest.add(sh.out_w() - sh.ft_b());
    let gob = a.grest.add(sh.out_b() - sh.ft_b());
    let ow0 = a.p.as_ptr().add(sh.out_w());
    let two = V::splat(2.0);
    for (j, s) in a.ss.iter().enumerate() {
        let g = a.grad[j];
        if a.c == 0 {
            *gob.add(s.bucket) += g;
        }
        let gs = V::splat(g);
        for side in 0..2 {
            let off = s.bucket * 2 * h + side * h + col;
            let x0 = a.acc.add(j * 2 * h + side * h + col);
            let mut d = [V::zero(); 16];
            for k in 0..n {
                let x = V::load(x0.add(k * V::W));
                let c = x.clamp01();
                let go = gow0.add(off + k * V::W);
                V::load(go).add(gs.mul(c).mul(c)).store(go);
                let w = V::load(ow0.add(off + k * V::W));
                d[k] = V::select_in01(x, gs.mul(w).mul(two).mul(c));
                let b = gb.add(k * V::W);
                V::load(b).add(d[k]).store(b);
            }
            for &f in &s.f[side][..s.n] {
                let row = gtile.add(f as usize * TB);
                for k in 0..n {
                    let q = row.add(k * V::W);
                    V::load(q).add(d[k]).store(q);
                }
            }
        }
    }
}

struct AdamC {
    c1: f32,
    c2: f32,
    lr: f32,
    decay: f32,
    inv_batch: f32,
}

struct AdamArgs {
    p: *mut f32,
    m: *mut f32,
    v: *mut f32,
    g: *mut f32,
    wt: *mut f32,
    tf: usize,
    h: usize,
    rows: usize,
    tiles: usize,
    /// FT rows `f0..f1` (gradient tile-major), or, if `flat > 0`, `flat`
    /// consecutive params with a standard-layout gradient.
    f0: usize,
    f1: usize,
    flat: usize,
    /// Net params (the factoriser block `F` starts here), `F`'s length (0 =
    /// off; then `f0..f1` are features and every bucket's row is updated),
    /// king buckets, and the `--wd-active-only` row flags (null = off).
    n: usize,
    fl: usize,
    kb: usize,
    active: *const AtomicU32,
    /// The batch whose rows count as active.
    gen: u32,
}

/// The AdamW constants of the next step (advances `opt.step`).
fn adam_consts(opt: &mut Adam, lr: f32, wd: f32, inv_batch: f32) -> AdamC {
    opt.step += 1;
    let (b1, b2) = (0.9f32, 0.999f32);
    AdamC {
        c1: 1.0 - b1.powi(opt.step.min(1 << 20) as i32),
        c2: 1.0 - b2.powi(opt.step.min(1 << 20) as i32),
        lr,
        decay: 1.0 - lr * wd,
        inv_batch,
    }
}

/// Column tile `c` of the tile-major FT state (`p`, `m`, `v` and `g` point
/// at the tile).
struct TileAdamArgs {
    p: *mut f32,
    m: *mut f32,
    v: *mut f32,
    g: *mut f32,
    wt: *mut f32,
    tf: usize,
    rows: usize,
    c: usize,
    /// Rows (features with the factoriser) `r0..r1` of the tile.
    r0: usize,
    r1: usize,
    fl: bool,
    kb: usize,
    active: *const AtomicU32,
    gen: u32,
}

/// `adam_rows` for one column tile, row by row in the same per-element
/// order (factoriser: F from every bucket's gradient first, then each
/// bucket's W, then the merged rows).
#[inline(always)]
unsafe fn tile_adam<V: Simd>(c: &AdamC, a: &TileAdamArgs) {
    let col = a.c * TB;
    let wt_at = |r: usize| a.wt.add((col / a.tf * a.rows + r) * a.tf + col % a.tf);
    let decay = |r: usize| if !a.active.is_null() && (*a.active.add(r)).load(Ordering::Relaxed) != a.gen { 1.0 } else { c.decay };
    let line = |r: usize, d: f32, g: *mut f32| adam_line::<V>(c, d, a.p.add(r * TB), a.m.add(r * TB), a.v.add(r * TB), g);
    if !a.fl {
        for r in a.r0..a.r1 {
            let w = line(r, decay(r), a.g.add(r * TB));
            std::ptr::copy_nonoverlapping(w.as_ptr(), wt_at(r), TB);
        }
    } else {
        for f in a.r0..a.r1 {
            let mut gf = [0.0f32; TB];
            for b in 0..a.kb {
                let g = a.g.add((b * INPUTS + f) * TB);
                for (k, x) in gf.iter_mut().enumerate() {
                    *x += *g.add(k);
                }
            }
            let fp = line(a.rows + f, c.decay, gf.as_mut_ptr());
            for b in 0..a.kb {
                let r = b * INPUTS + f;
                let wp = line(r, decay(r), a.g.add(r * TB));
                let w = wt_at(r);
                for k in 0..TB {
                    *w.add(k) = (wp[k] + fp[k]).clamp(-CLIP, CLIP);
                }
            }
        }
    }
    V::fence();
}

/// v2's AdamW expression, element by element (vectorised by LLVM; lane-wise
/// exact), then clears the gradient.
#[inline(always)]
unsafe fn adam_run(c: &AdamC, decay: f32, p: *mut f32, m: *mut f32, v: *mut f32, g: *mut f32, len: usize) {
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let p = std::slice::from_raw_parts_mut(p, len);
    let m = std::slice::from_raw_parts_mut(m, len);
    let v = std::slice::from_raw_parts_mut(v, len);
    let gr = std::slice::from_raw_parts_mut(g, len);
    for i in 0..len {
        let mut g = 0.0f32;
        g += gr[i];
        gr[i] = 0.0;
        g *= c.inv_batch;
        m[i] = b1 * m[i] + (1.0 - b1) * g;
        v[i] = b2 * v[i] + (1.0 - b2) * g * g;
        let upd = (m[i] / c.c1) / ((v[i] / c.c2).sqrt() + eps);
        let x = p[i] * decay - c.lr * upd;
        p[i] = x.clamp(-CLIP, CLIP);
    }
}

/// `adam_run` on one `TB`-float line of the FT; returns the new params.
/// `p`, `m`, `v` are written with non-temporal stores when 64-byte aligned:
/// AdamW streams ~3x the FT through memory, and on CPUs whose L3 is a
/// victim cache (AMD) this keeps those lines out of it, so the tile-major
/// weights and the gradient that the next batch reads stay in L3.
#[inline(always)]
unsafe fn adam_line<V: Simd>(c: &AdamC, decay: f32, p: *mut f32, m: *mut f32, v: *mut f32, g: *mut f32) -> [f32; TB] {
    let mut l = [[0.0f32; TB]; 3];
    for (k, src) in [p, m, v].into_iter().enumerate() {
        std::ptr::copy_nonoverlapping(src, l[k].as_mut_ptr(), TB);
    }
    let old = l;
    let [lp, lm, lv] = &mut l;
    adam_run(c, decay, lp.as_mut_ptr(), lm.as_mut_ptr(), lv.as_mut_ptr(), g, TB);
    // Rows no position can hit (m = v = 0, no decay) do not change: skip
    // the stores, memory writes are the limit (~31 GB/s on a 9600X).
    let same = |a: &[f32; TB], b: &[f32; TB]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    if (0..3).all(|k| same(&l[k], &old[k])) {
        return l[0];
    }
    for (k, dst) in [p, m, v].into_iter().enumerate() {
        if dst as usize % 64 == 0 {
            for i in (0..TB).step_by(V::W) {
                V::load(l[k].as_ptr().add(i)).stream(dst.add(i));
            }
        } else {
            std::ptr::copy_nonoverlapping(l[k].as_ptr(), dst, TB);
        }
    }
    l[0]
}

#[inline(always)]
unsafe fn adam_impl<V: Simd>(c: &AdamC, a: &AdamArgs) {
    adam_rows::<V>(c, a);
    V::fence();
}

#[inline(always)]
unsafe fn adam_rows<V: Simd>(c: &AdamC, a: &AdamArgs) {
    if a.flat > 0 {
        return adam_run(c, c.decay, a.p, a.m, a.v, a.g, a.flat);
    }
    // v2: rows that did not occur in the batch skip the decay (`* 1.0`).
    let row_decay = |r: usize| {
        if !a.active.is_null() && (*a.active.add(r)).load(Ordering::Relaxed) != a.gen {
            1.0
        } else {
            c.decay
        }
    };
    let wt_at = |r: usize, j: usize| a.wt.add((j / a.tf * a.rows + r) * a.tf + j % a.tf);
    if a.fl == 0 {
        for f in a.f0..a.f1 {
            let decay = row_decay(f);
            for t in 0..a.tiles {
                let i = f * a.h + t * TB;
                let w = adam_line::<V>(c, decay, a.p.add(i), a.m.add(i), a.v.add(i), a.g.add((t * a.rows + f) * TB));
                std::ptr::copy_nonoverlapping(w.as_ptr(), wt_at(f, t * TB), TB);
            }
        }
        return;
    }
    for f in a.f0..a.f1 {
        let mut decay = [0.0f32; 8];
        for (b, d) in decay[..a.kb].iter_mut().enumerate() {
            *d = row_decay(b * INPUTS + f);
        }
        for t in 0..a.tiles {
            // F first, from every bucket's gradient (bucket order, as v2),
            // before the W updates clear them.
            let mut gf = [0.0f32; TB];
            for b in 0..a.kb {
                let g = a.g.add((t * a.rows + b * INPUTS + f) * TB);
                for (k, x) in gf.iter_mut().enumerate() {
                    *x += *g.add(k);
                }
            }
            let fi = a.n + f * a.h + t * TB;
            let fp = adam_line::<V>(c, c.decay, a.p.add(fi), a.m.add(fi), a.v.add(fi), gf.as_mut_ptr());
            for (b, &d) in decay[..a.kb].iter().enumerate() {
                let r = b * INPUTS + f;
                let i = r * a.h + t * TB;
                let wp = adam_line::<V>(c, d, a.p.add(i), a.m.add(i), a.v.add(i), a.g.add((t * a.rows + r) * TB));
                let w = wt_at(r, t * TB);
                for k in 0..TB {
                    *w.add(k) = (wp[k] + fp[k]).clamp(-CLIP, CLIP);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- dispatch

macro_rules! entry {
    ($name:ident, $feat:literal, $v:ty) => {
        #[allow(dead_code)]
        mod $name {
            use super::*;
            #[target_feature(enable = $feat)]
            pub unsafe fn acc(a: &AccArgs) {
                acc_impl::<$v>(a)
            }
            #[target_feature(enable = $feat)]
            pub unsafe fn out(a: &OutArgs, exact: bool) {
                if exact {
                    out_impl::<$v, true>(a)
                } else {
                    out_impl::<$v, false>(a)
                }
            }
            #[target_feature(enable = $feat)]
            pub unsafe fn bwd(a: &BwdArgs) {
                bwd_impl::<$v>(a)
            }
            #[target_feature(enable = $feat)]
            pub unsafe fn adam(c: &AdamC, a: &AdamArgs) {
                adam_impl::<$v>(c, a)
            }
            #[target_feature(enable = $feat)]
            pub unsafe fn tile_adam(c: &AdamC, a: &TileAdamArgs) {
                super::tile_adam::<$v>(c, a)
            }
        }
    };
}

mod portable {
    use super::*;
    pub unsafe fn acc(a: &AccArgs) {
        acc_impl::<Portable>(a)
    }
    pub unsafe fn out(a: &OutArgs, exact: bool) {
        if exact {
            out_impl::<Portable, true>(a)
        } else {
            out_impl::<Portable, false>(a)
        }
    }
    pub unsafe fn bwd(a: &BwdArgs) {
        bwd_impl::<Portable>(a)
    }
    pub unsafe fn adam(c: &AdamC, a: &AdamArgs) {
        adam_impl::<Portable>(c, a)
    }
    pub unsafe fn tile_adam(c: &AdamC, a: &TileAdamArgs) {
        super::tile_adam::<Portable>(c, a)
    }
}

#[cfg(target_arch = "x86_64")]
use crate::simd::{Avx2, Avx512};
#[cfg(target_arch = "x86_64")]
entry!(avx2, "avx2,fma", Avx2);
#[cfg(target_arch = "x86_64")]
entry!(avx512, "avx512f,avx512bw,avx512vl,avx2,fma", Avx512);
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
#[cfg(target_arch = "aarch64")]
entry!(neon, "neon", Neon);

unsafe fn acc_dispatch(isa: Isa, a: &AccArgs) {
    match isa {
        Isa::Portable => portable::acc(a),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::acc(a),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::acc(a),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::acc(a),
    }
}

unsafe fn out_dispatch(isa: Isa, exact: bool, a: &OutArgs) {
    match isa {
        Isa::Portable => portable::out(a, exact),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::out(a, exact),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::out(a, exact),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::out(a, exact),
    }
}

unsafe fn bwd_dispatch(isa: Isa, a: &BwdArgs) {
    match isa {
        Isa::Portable => portable::bwd(a),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::bwd(a),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::bwd(a),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::bwd(a),
    }
}

unsafe fn adam_dispatch(isa: Isa, c: &AdamC, a: &AdamArgs) {
    match isa {
        Isa::Portable => portable::adam(c, a),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::adam(c, a),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::adam(c, a),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::adam(c, a),
    }
}

unsafe fn tile_adam_dispatch(isa: Isa, c: &AdamC, a: &TileAdamArgs) {
    match isa {
        Isa::Portable => portable::tile_adam(c, a),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::tile_adam(c, a),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::tile_adam(c, a),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::tile_adam(c, a),
    }
}
