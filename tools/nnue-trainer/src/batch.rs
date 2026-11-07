//! Fast batch gradient + AdamW for `train`.
//!
//! The v2 loop gave every thread a full private gradient (zeroed and summed
//! over threads every batch) and scattered each sample's 2H-wide dacc into
//! rows 2 KB apart, i.e. L2/L3 traffic of ~3x the FT per sample. Here:
//!
//! - **Forward** is split by samples (blocks of `FB`): each sample's
//!   accumulators are built 64 floats at a time in registers (bias + active
//!   rows, in feature order, exactly as `forward`), stored, and the output
//!   dot product, sigmoid, loss and output gradient are computed.
//! - **Backward** is split by *columns*: unit `c` owns columns
//!   `16c..16c+16` (one cache line) of every gradient row, for every sample
//!   of the sub-batch, in sample order. One gradient buffer, no reduction,
//!   no false sharing; its FT part is stored tile-major
//!   (`[tile][row][16]`), so a unit's gradient is one contiguous
//!   `rows * 64` B block that stays in L1/L2.
//! - The batch runs in sub-batches of `sb` samples so the stored
//!   accumulators stay in cache; backward of sub-batch k shares a phase
//!   with forward of sub-batch k+1, which keeps every thread busy while the
//!   column units drain.
//! - AdamW reads the one gradient, updates and re-zeroes it, in parallel.
//!
//! Every gradient element is summed in batch order, so results do not
//! depend on the thread count, the scheduling, or the ISA (all SIMD
//! implementations are lane-wise exact). With `exact` the output dot product
//! is the sequential sum of `forward`, and a batch is bit-identical to the
//! v2 trainer run with `--threads 1`; without it the dot product uses 16
//! interleaved partial sums (float reassociation only).

use crate::simd::{Isa, Portable, Simd};
use crate::{decode, sigmoid, Adam, KingLayout, Rec, Sample, Shape, CLIP, INPUTS};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::Thread;

/// Backward column tile (floats): one 64-byte cache line.
pub const TB: usize = 16;
/// Forward register tile (floats).
const TF: usize = 64;
/// Samples per forward unit.
const FB: usize = 16;
/// FT rows per AdamW unit.
const AB: usize = 16;

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
    #[cfg_attr(not(test), allow(dead_code))]
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
    /// Gradient of params `ft_b()..len()` (FT bias, output weights, output
    /// bias), standard layout.
    grest: ABuf,
    acc: [ABuf; 2],
    samples: [Vec<Sample>; 2],
    grad: Vec<f32>,
    loss: Vec<f32>,
}

impl BatchTrainer {
    /// `sb` = sub-batch size (0 = automatic: ~2 MB of accumulators).
    pub fn new(sh: Shape, sb: usize, exact: bool) -> Self {
        assert!(sh.h % TB == 0 && sh.h > 0, "--hidden must be a multiple of {TB}");
        let sb = if sb == 0 { ((2 << 20) / (8 * sh.h)).clamp(128, 2048) } else { sb };
        let sb = sb.div_ceil(FB) * FB;
        let rows = sh.kb * INPUTS;
        let blank = Sample { f: [[0; 32]; 2], n: 0, bucket: 0, target: 0.0 };
        BatchTrainer {
            sh,
            rows,
            tiles: sh.h / TB,
            sb,
            isa: Isa::detect(),
            exact,
            gft: ABuf::new(rows * sh.h),
            grest: ABuf::new(sh.len() - sh.ft_b()),
            acc: [ABuf::new(sb * 2 * sh.h), ABuf::new(sb * 2 * sh.h)],
            samples: [vec![blank; sb], vec![blank; sb]],
            grad: Vec::new(),
            loss: Vec::new(),
        }
    }

    pub fn sub_batch(&self) -> usize {
        self.sb
    }

    /// Accumulates the gradient of the summed loss over `recs` (record `i`
    /// mirrored iff `mirror[i]`) into the internal buffers; returns the
    /// summed loss. Call `adam` to apply and clear it.
    pub fn batch(&mut self, pool: &Pool, p: &[f32], recs: &[Rec], mirror: &[bool], lambda: f32, kl: &KingLayout) -> f64 {
        let n = recs.len();
        assert_eq!(mirror.len(), n);
        assert_eq!(p.len(), self.sh.len());
        self.grad.resize(n, 0.0);
        self.loss.resize(n, 0.0);
        let (sh, rows, tiles, sb, isa, exact) = (self.sh, self.rows, self.tiles, self.sb, self.isa, self.exact);
        let gft = self.gft.ptr();
        let grest = self.grest.ptr();
        let accs = [self.acc[0].ptr(), self.acc[1].ptr()];
        let samples = [SPtr(self.samples[0].as_mut_ptr()), SPtr(self.samples[1].as_mut_ptr())];
        let grad = Ptr(self.grad.as_mut_ptr());
        let loss = Ptr(self.loss.as_mut_ptr());
        let nsb = n.div_ceil(sb);
        let sb_len = |k: usize| sb.min(n - k * sb);
        for k in 0..=nsb {
            let back = if k >= 1 { tiles } else { 0 };
            let fwd = if k < nsb { sb_len(k).div_ceil(FB) } else { 0 };
            par_units(pool, back + fwd, &|u| unsafe {
                if u < back {
                    let kb = k - 1;
                    let len = sb_len(kb);
                    let ss = std::slice::from_raw_parts(samples[kb % 2].get(), len);
                    let gr = std::slice::from_raw_parts(grad.get().add(kb * sb), len);
                    let a = BwdArgs { sh, rows, p, ss, acc: accs[kb % 2].get(), grad: gr, gft: gft.get(), grest: grest.get(), c: u };
                    bwd_dispatch(isa, &a);
                } else {
                    let j = u - back;
                    let len = sb_len(k);
                    let (s0, s1) = (j * FB, ((j + 1) * FB).min(len));
                    let base = k * sb;
                    let ss = std::slice::from_raw_parts_mut(samples[k % 2].get().add(s0), s1 - s0);
                    for (i, s) in ss.iter_mut().enumerate() {
                        let r = base + s0 + i;
                        *s = decode(&recs[r], lambda, sh.buckets, mirror[r], kl);
                    }
                    let a = FwdArgs {
                        sh,
                        p,
                        ss,
                        acc: accs[k % 2].get().add(s0 * 2 * sh.h),
                        grad: grad.get().add(base + s0),
                        loss: loss.get().add(base + s0),
                    };
                    fwd_dispatch(isa, exact, &a);
                }
            });
        }
        self.loss.iter().map(|&l| l as f64).sum()
    }

    /// One AdamW step (same maths and constants as v2) from the accumulated
    /// gradient times `inv_batch`, with clipping; clears the gradient.
    pub fn adam(&mut self, pool: &Pool, p: &mut [f32], opt: &mut Adam, lr: f32, wd: f32, inv_batch: f32) {
        opt.step += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let c = AdamC {
            c1: 1.0 - b1.powi(opt.step.min(1 << 20) as i32),
            c2: 1.0 - b2.powi(opt.step.min(1 << 20) as i32),
            lr,
            decay: 1.0 - lr * wd,
            inv_batch,
        };
        let (h, rows, tiles, ftb) = (self.sh.h, self.rows, self.tiles, self.sh.ft_b());
        let (pp, mp, vp) = (Ptr(p.as_mut_ptr()), Ptr(opt.m.as_mut_ptr()), Ptr(opt.v.as_mut_ptr()));
        let gft = self.gft.ptr();
        let grest = self.grest.ptr();
        let rest = p.len() - ftb;
        let ft_units = rows.div_ceil(AB);
        let isa = self.isa;
        par_units(pool, ft_units + 1, &|u| unsafe {
            let a = if u < ft_units {
                AdamArgs { p: pp.get(), m: mp.get(), v: vp.get(), g: gft.get(), h, rows, tiles, f0: u * AB, f1: ((u + 1) * AB).min(rows), flat: 0 }
            } else {
                let (p, m, v) = (pp.get().add(ftb), mp.get().add(ftb), vp.get().add(ftb));
                AdamArgs { p, m, v, g: grest.get(), h, rows, tiles, f0: 0, f1: 0, flat: rest }
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

struct FwdArgs<'a> {
    sh: Shape,
    p: &'a [f32],
    ss: &'a [Sample],
    /// `ss.len() * 2h` accumulators out.
    acc: *mut f32,
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

/// Bias + active rows for `TW` columns starting at `col`, in registers.
#[inline(always)]
unsafe fn fwd_tile<V: Simd, const TW: usize>(fw: *const f32, fb: *const f32, h: usize, feats: &[u16], col: usize, out: *mut f32) {
    let n = TW / V::W;
    let mut r = [V::zero(); 16];
    for (k, x) in r[..n].iter_mut().enumerate() {
        *x = V::load(fb.add(col + k * V::W));
    }
    for &f in feats {
        let row = fw.add(f as usize * h + col);
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
unsafe fn dot_tile<V: Simd, const TW: usize>(a: *const f32, w: *const f32, lanes: &mut [V; 16]) {
    let per16 = 16 / V::W;
    for k in 0..TW / V::W {
        let x = V::load(a.add(k * V::W)).clamp01();
        let l = &mut lanes[k % per16];
        *l = l.add(x.mul(x).mul(V::load(w.add(k * V::W))));
    }
}

#[inline(always)]
unsafe fn fwd_impl<V: Simd, const EXACT: bool>(a: &FwdArgs) {
    let sh = a.sh;
    let h = sh.h;
    let fw = a.p.as_ptr();
    let fb = fw.add(sh.ft_b());
    for (j, s) in a.ss.iter().enumerate() {
        let acc = a.acc.add(j * 2 * h);
        let ow = fw.add(sh.out_w() + s.bucket * 2 * h);
        let mut lanes = [V::zero(); 16];
        for side in 0..2 {
            let feats = &s.f[side][..s.n];
            let out = acc.add(side * h);
            let mut col = 0;
            while col + TF <= h {
                fwd_tile::<V, TF>(fw, fb, h, feats, col, out.add(col));
                if !EXACT {
                    dot_tile::<V, TF>(out.add(col), ow.add(side * h + col), &mut lanes);
                }
                col += TF;
            }
            while col < h {
                fwd_tile::<V, TB>(fw, fb, h, feats, col, out.add(col));
                if !EXACT {
                    dot_tile::<V, TB>(out.add(col), ow.add(side * h + col), &mut lanes);
                }
                col += TB;
            }
        }
        let sum = if EXACT {
            // `forward`'s sequential sum.
            let mut sum = 0.0f32;
            for i in 0..2 * h {
                let x = (*acc.add(i)).clamp(0.0, 1.0);
                sum += x * x * *ow.add(i);
            }
            sum
        } else {
            let mut l = [0.0f32; 16];
            let per16 = 16 / V::W;
            for (k, v) in lanes[..per16].iter().enumerate() {
                v.store(l.as_mut_ptr().add(k * V::W));
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
    h: usize,
    rows: usize,
    tiles: usize,
    /// FT rows `f0..f1` (gradient tile-major), or, if `flat > 0`, `flat`
    /// consecutive params with a standard-layout gradient.
    f0: usize,
    f1: usize,
    flat: usize,
}

/// v2's AdamW expression, element by element (vectorised by LLVM; lane-wise
/// exact), then clears the gradient.
#[inline(always)]
unsafe fn adam_run(c: &AdamC, p: *mut f32, m: *mut f32, v: *mut f32, g: *mut f32, len: usize) {
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
        let x = p[i] * c.decay - c.lr * upd;
        p[i] = x.clamp(-CLIP, CLIP);
    }
}

#[inline(always)]
unsafe fn adam_impl(c: &AdamC, a: &AdamArgs) {
    if a.flat > 0 {
        return adam_run(c, a.p, a.m, a.v, a.g, a.flat);
    }
    for f in a.f0..a.f1 {
        for t in 0..a.tiles {
            let i = f * a.h + t * TB;
            adam_run(c, a.p.add(i), a.m.add(i), a.v.add(i), a.g.add((t * a.rows + f) * TB), TB);
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
            pub unsafe fn fwd(a: &FwdArgs, exact: bool) {
                if exact {
                    fwd_impl::<$v, true>(a)
                } else {
                    fwd_impl::<$v, false>(a)
                }
            }
            #[target_feature(enable = $feat)]
            pub unsafe fn bwd(a: &BwdArgs) {
                bwd_impl::<$v>(a)
            }
            #[target_feature(enable = $feat)]
            pub unsafe fn adam(c: &AdamC, a: &AdamArgs) {
                adam_impl(c, a)
            }
        }
    };
}

mod portable {
    use super::*;
    pub unsafe fn fwd(a: &FwdArgs, exact: bool) {
        if exact {
            fwd_impl::<Portable, true>(a)
        } else {
            fwd_impl::<Portable, false>(a)
        }
    }
    pub unsafe fn bwd(a: &BwdArgs) {
        bwd_impl::<Portable>(a)
    }
    pub unsafe fn adam(c: &AdamC, a: &AdamArgs) {
        adam_impl(c, a)
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

unsafe fn fwd_dispatch(isa: Isa, exact: bool, a: &FwdArgs) {
    match isa {
        Isa::Portable => portable::fwd(a, exact),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::fwd(a, exact),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::fwd(a, exact),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::fwd(a, exact),
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
