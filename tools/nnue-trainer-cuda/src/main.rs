//! CUDA backend for `tools/nnue-trainer`: the same command line, data feed,
//! RNG streams, init, validation, export and checkpoints (it is the CPU
//! trainer's `cli` with a GPU `Backend`); forward, backward and AdamW run
//! on the GPU (`src/kernels.cu`). Deterministic (fixed summation orders),
//! but in a different order from the CPU kernels, so nets match the CPU
//! trainer's closely, not bit for bit. Decode and the king-bucket sort run
//! on the CPU pool (`--threads`). Std only: the driver API is called
//! through the small FFI below.

use nnue_trainer::batch::{par_units, BatchTrainer, Pool};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use nnue_trainer::{decode_sh, Adam, Backend, BackendCfg, KingLayout, Rec, Shape, CLIP, INPUTS};
use std::ffi::{c_void, CStr, CString};
use std::sync::atomic::{AtomicU8, Ordering};

// ---------------------------------------------------------------- driver FFI

type CuResult = i32;
type CuDevice = i32;
type CuPtr = u64;
type Handle = *mut c_void;

#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: u32) -> CuResult;
    fn cuDeviceGet(dev: *mut CuDevice, ordinal: i32) -> CuResult;
    fn cuDeviceGetName(name: *mut i8, len: i32, dev: CuDevice) -> CuResult;
    fn cuCtxCreate_v2(ctx: *mut Handle, flags: u32, dev: CuDevice) -> CuResult;
    fn cuModuleLoadData(module: *mut Handle, image: *const c_void) -> CuResult;
    fn cuModuleGetFunction(f: *mut Handle, module: Handle, name: *const i8) -> CuResult;
    fn cuMemAlloc_v2(p: *mut CuPtr, bytes: usize) -> CuResult;
    fn cuMemsetD8_v2(p: CuPtr, v: u8, n: usize) -> CuResult;
    fn cuMemcpyHtoD_v2(dst: CuPtr, src: *const c_void, bytes: usize) -> CuResult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CuPtr, bytes: usize) -> CuResult;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        f: Handle,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shmem: u32,
        stream: Handle,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CuResult;
    fn cuStreamCreate(stream: *mut Handle, flags: u32) -> CuResult;
    fn cuStreamSynchronize(stream: Handle) -> CuResult;
    fn cuMemcpyHtoDAsync_v2(dst: CuPtr, src: *const c_void, bytes: usize, stream: Handle) -> CuResult;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CuPtr, bytes: usize, stream: Handle) -> CuResult;
    fn cuMemAllocHost_v2(p: *mut *mut c_void, bytes: usize) -> CuResult;
    fn cuEventCreate(ev: *mut Handle, flags: u32) -> CuResult;
    fn cuEventRecord(ev: Handle, stream: Handle) -> CuResult;
    fn cuEventSynchronize(ev: Handle) -> CuResult;
    fn cuStreamWaitEvent(stream: Handle, ev: Handle, flags: u32) -> CuResult;
    fn cuGetErrorString(e: CuResult, s: *mut *const i8) -> CuResult;
}

fn ck(r: CuResult, what: &str) {
    if r != 0 {
        let mut s: *const i8 = std::ptr::null();
        unsafe { cuGetErrorString(r, &mut s) };
        let msg = if s.is_null() { String::new() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() };
        panic!("CUDA {what}: error {r} {msg}");
    }
}

/// A device buffer of `n` elements of `T`.
struct DBuf<T> {
    ptr: CuPtr,
    n: usize,
    _t: std::marker::PhantomData<T>,
}

impl<T: Copy> DBuf<T> {
    fn new(n: usize) -> Self {
        let mut ptr = 0;
        let bytes = (n * std::mem::size_of::<T>()).max(16);
        ck(unsafe { cuMemAlloc_v2(&mut ptr, bytes) }, "alloc");
        ck(unsafe { cuMemsetD8_v2(ptr, 0, bytes) }, "memset");
        DBuf { ptr, n, _t: std::marker::PhantomData }
    }
    fn up(&self, src: &[T]) {
        assert!(src.len() <= self.n);
        ck(unsafe { cuMemcpyHtoD_v2(self.ptr, src.as_ptr() as *const c_void, std::mem::size_of_val(src)) }, "upload");
    }
    fn down_at(&self, off: usize, dst: &mut [T]) {
        assert!(off + dst.len() <= self.n);
        let at = self.ptr + (off * std::mem::size_of::<T>()) as u64;
        ck(unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, at, std::mem::size_of_val(dst)) }, "download");
    }
}

impl<T: Copy> DBuf<T> {
    /// Async upload from pinned memory on `stream`.
    fn up_async(&self, src: &HBuf<T>, n: usize, stream: Handle) {
        assert!(n <= self.n && n <= src.n);
        let bytes = n * std::mem::size_of::<T>();
        ck(unsafe { cuMemcpyHtoDAsync_v2(self.ptr, src.ptr as *const c_void, bytes, stream) }, "async upload");
    }
}

/// Pinned (page-locked) host buffer for async copies.
struct HBuf<T> {
    ptr: *mut T,
    n: usize,
}

impl<T: Copy> HBuf<T> {
    fn new(n: usize) -> Self {
        let mut p = std::ptr::null_mut();
        ck(unsafe { cuMemAllocHost_v2(&mut p, (n * std::mem::size_of::<T>()).max(16)) }, "pinned alloc");
        HBuf { ptr: p as *mut T, n }
    }
    fn slice(&mut self) -> &mut [T] {
        // SAFETY: allocated for n elements; pinned memory is zero-initialised
        // by nobody, so callers write before they read.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.n) }
    }
}

/// One batch's inputs: pinned host staging and the device copies (two of
/// these alternate, so batch k is prepared while batch k-1 runs).
struct Slot {
    hs: HBuf<GSample>,
    horder: HBuf<u32>,
    hpairs: HBuf<u32>,
    hitems: HBuf<Item>,
    hilo: HBuf<i32>,
    hrs: HBuf<i32>,
    hrb: HBuf<i32>,
    hact: HBuf<u8>,
    hloss: HBuf<f32>,
    ss: DBuf<GSample>,
    order: DBuf<u32>,
    pairs: DBuf<u32>,
    items: DBuf<Item>,
    ilo: DBuf<i32>,
    /// Ranges: first pair (and one past the last), king bucket.
    rs: DBuf<i32>,
    rb: DBuf<i32>,
    active: DBuf<u8>,
    loss: DBuf<f32>,
    /// Recorded on the compute stream after this slot's batch (and its loss
    /// download), and on the copy stream after its inputs are uploaded.
    done: Handle,
    copied: Handle,
}

impl Slot {
    fn new(b: usize, max_items: usize, max_ranges: usize, kb: usize, rows: usize) -> Self {
        let (mut done, mut copied) = (std::ptr::null_mut(), std::ptr::null_mut());
        // CU_EVENT_DISABLE_TIMING (2): cheaper events.
        ck(unsafe { cuEventCreate(&mut done, 2) }, "event");
        ck(unsafe { cuEventCreate(&mut copied, 2) }, "event");
        Slot {
            hs: HBuf::new(b),
            horder: HBuf::new(b),
            hpairs: HBuf::new(4 * b),
            hitems: HBuf::new(max_items),
            hilo: HBuf::new(kb + 1),
            hrs: HBuf::new(max_ranges + 1),
            hrb: HBuf::new(max_ranges),
            hact: HBuf::new(rows),
            hloss: HBuf::new(b),
            ss: DBuf::new(b),
            order: DBuf::new(b),
            pairs: DBuf::new(4 * b),
            items: DBuf::new(max_items),
            ilo: DBuf::new(kb + 1),
            rs: DBuf::new(max_ranges + 1),
            rb: DBuf::new(max_ranges),
            active: DBuf::new(rows),
            loss: DBuf::new(b),
            done,
            copied,
        }
    }
}

/// Kernel argument (passed by address to cuLaunchKernel).
enum Arg {
    P(CuPtr),
    I(i32),
    C(AdamC),
}

fn launch(f: Handle, grid: (u32, u32), block: u32, stream: Handle, args: &[Arg]) {
    let mut vals: Vec<*mut c_void> = args
        .iter()
        .map(|a| match a {
            Arg::P(p) => p as *const CuPtr as *mut c_void,
            Arg::I(i) => i as *const i32 as *mut c_void,
            Arg::C(c) => c as *const AdamC as *mut c_void,
        })
        .collect();
    let r = unsafe { cuLaunchKernel(f, grid.0, grid.1, 1, block, 1, 1, 0, stream, vals.as_mut_ptr(), std::ptr::null_mut()) };
    ck(r, "launch");
}

// ---------------------------------------------------------------- GPU-side types (kernels.cu)

#[repr(C)]
#[derive(Clone, Copy)]
struct GSample {
    /// `np` piece features, then `n - np` threat features (kernels.cu MAXF).
    f: [[u16; 64]; 2],
    n: u8,
    np: u8,
    bucket: u8,
    pad: u8,
    target: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Item {
    bucket: i32,
    start: i32,
    end: i32,
    /// Index of the item's first range of `R` pairs.
    range: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdamC {
    c1: f32,
    c2: f32,
    lr: f32,
    decay: f32,
    inv_batch: f32,
}

/// Pairs per FT-backward work item (one king bucket each).
const WI: usize = 2048;
/// Pairs per bitmask range (kernels.cu `R`).
const R: usize = 128;
/// Samples per output-layer gradient chunk (kernels.cu `CH`).
const CH: usize = 256;

struct Kernels {
    forward: Handle,
    forward_half: Handle,
    to_half: Handle,
    backward_ft: Handle,
    plan_masks: Handle,
    backward_rest: Handle,
    reduce_rest: Handle,
    adam_flat: Handle,
    adam_ft: Handle,
}

struct GpuTrainer {
    sh: Shape,
    fl: usize,
    wd_active: bool,
    device: String,
    k: Kernels,
    live: bool,
    // Params and Adam moments: FT rows (W), factoriser F, rest.
    pw: DBuf<f32>,
    mw: DBuf<f32>,
    vw: DBuf<f32>,
    pf: DBuf<f32>,
    mf: DBuf<f32>,
    vf: DBuf<f32>,
    pr: DBuf<f32>,
    mr: DBuf<f32>,
    vr: DBuf<f32>,
    gr: DBuf<f32>,
    /// Merged FT rows read by the forward (f32, or fp16 in `wth` with --half).
    wt: DBuf<f32>,
    wth: DBuf<u16>,
    half: bool,
    part: DBuf<f32>,
    prest: DBuf<f32>,
    acc: DBuf<f32>,
    gout: DBuf<f32>,
    slots: [Slot; 2],
    /// Per range of R pairs: per-feature bitmasks of its pairs (plan_masks).
    masks: DBuf<u32>,
    slot: usize,
    /// The last batch launched whose loss `step` has not returned yet:
    /// (slot, positions).
    pending: Option<(usize, usize)>,
    stream: Handle,
    /// Uploads run here, overlapping the previous batch's kernels.
    copy: Handle,
    /// `--wd-active-only` row flags set by the decode.
    act: Vec<AtomicU8>,
}

unsafe impl Send for GpuTrainer {}
unsafe impl Send for Slot {}

impl GpuTrainer {
    fn new(cfg: &BackendCfg) -> Self {
        let sh = cfg.sh;
        assert!(sh.h % 64 == 0 && sh.h <= 2048, "nnue-trainer-cuda: --hidden must be a multiple of 64, at most 2048");
        assert!(sh.buckets <= 8 && sh.kb <= 64, "nnue-trainer-cuda: at most 8 output buckets, 64 king buckets");
        // FT rows: kb king-bucket blocks, then tb threat blocks (docs/nnue.md).
        let (h, tb, b) = (sh.h, sh.tb(), cfg.batch);
        let rows = (sh.kb + tb) * INPUTS;
        let nrest = sh.len() - sh.ft_b();
        ck(unsafe { cuInit(0) }, "init");
        let mut dev = 0;
        ck(unsafe { cuDeviceGet(&mut dev, 0) }, "device");
        let mut name = [0i8; 256];
        ck(unsafe { cuDeviceGetName(name.as_mut_ptr(), 256, dev) }, "device name");
        let device = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        let mut ctx = std::ptr::null_mut();
        ck(unsafe { cuCtxCreate_v2(&mut ctx, 0, dev) }, "context");
        let ptx = CString::new(include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"))).unwrap();
        let mut module = std::ptr::null_mut();
        ck(unsafe { cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void) }, "load PTX");
        let get = |n: &str| {
            let mut f = std::ptr::null_mut();
            let c = CString::new(n).unwrap();
            ck(unsafe { cuModuleGetFunction(&mut f, module, c.as_ptr()) }, n);
            f
        };
        let k = Kernels {
            forward: get("forward"),
            forward_half: get("forward_half"),
            to_half: get("to_half"),
            backward_ft: get("backward_ft"),
            plan_masks: get("plan_masks"),
            backward_rest: get("backward_rest"),
            reduce_rest: get("reduce_rest"),
            adam_flat: get("adam_flat"),
            adam_ft: get("adam_ft"),
        };
        // Work items / ranges: every pair once per king bucket, and once more
        // per threat block.
        let max_items = (1 + tb) * ((2 * b).div_ceil(WI) + 1) + sh.kb;
        let max_ranges = (1 + tb) * (2 * b).div_ceil(R) + max_items;
        let chunks = b.div_ceil(CH);
        let prest_stride = sh.buckets * 2 * h + 2 * h + sh.buckets;
        let nf = INPUTS * h;
        let (mut stream, mut copy) = (std::ptr::null_mut(), std::ptr::null_mut());
        // CU_STREAM_NON_BLOCKING (1): no implicit sync with the null stream.
        ck(unsafe { cuStreamCreate(&mut stream, 1) }, "stream");
        ck(unsafe { cuStreamCreate(&mut copy, 1) }, "stream");
        GpuTrainer {
            sh,
            fl: cfg.fl,
            wd_active: cfg.wd_active,
            device,
            k,
            live: false,
            pw: DBuf::new(rows * h),
            mw: DBuf::new(rows * h),
            vw: DBuf::new(rows * h),
            pf: DBuf::new(if cfg.fl > 0 { nf } else { 0 }),
            mf: DBuf::new(if cfg.fl > 0 { nf } else { 0 }),
            vf: DBuf::new(if cfg.fl > 0 { nf } else { 0 }),
            pr: DBuf::new(nrest),
            mr: DBuf::new(nrest),
            vr: DBuf::new(nrest),
            gr: DBuf::new(nrest),
            wt: DBuf::new(rows * h),
            wth: DBuf::new(if cfg.half { rows * h } else { 0 }),
            half: cfg.half,
            part: DBuf::new(max_items * nf),
            prest: DBuf::new(chunks * prest_stride),
            acc: DBuf::new(b * 2 * h),
            gout: DBuf::new(b),
            slots: [Slot::new(b, max_items, max_ranges, sh.kb + tb, rows), Slot::new(b, max_items, max_ranges, sh.kb + tb, rows)],
            masks: DBuf::new(max_ranges * INPUTS * (R / 32)),
            slot: 0,
            pending: None,
            stream,
            copy,
            act: (0..rows).map(|_| AtomicU8::new(0)).collect(),
        }
    }

    /// Forgets the device state: the next `step` re-imports `p` and `opt`
    /// (call `flush` first).
    fn invalidate(&mut self) {
        self.live = false;
    }

    /// Waits for slot `si`'s batch and sums its `n` losses (sample order).
    fn loss_of(&mut self, si: usize, n: usize) -> f64 {
        let sl = &mut self.slots[si];
        ck(unsafe { cuEventSynchronize(sl.done) }, "wait");
        sl.hloss.slice()[..n].iter().map(|&l| l as f64).sum()
    }

    /// Uploads the training state from the host vectors.
    fn import(&mut self, p: &[f32], opt: &Adam) {
        let (ftb, n) = (self.sh.ft_b(), self.sh.len());
        for (dst, src) in [(&self.pw, &p[..ftb]), (&self.mw, &opt.m[..ftb]), (&self.vw, &opt.v[..ftb])] {
            dst.up(src);
        }
        for (dst, src) in [(&self.pr, &p[ftb..n]), (&self.mr, &opt.m[ftb..n]), (&self.vr, &opt.v[ftb..n])] {
            dst.up(src);
        }
        if self.fl > 0 {
            for (dst, src) in [(&self.pf, &p[n..]), (&self.mf, &opt.m[n..]), (&self.vf, &opt.v[n..])] {
                dst.up(src);
            }
        }
        let h = self.sh.h;
        // The factoriser merges into the bucket rows only, not the threat rows.
        let fkb = self.sh.kb * INPUTS * h;
        let wt: Vec<f32> = (0..ftb)
            .map(|i| if self.fl > 0 && i < fkb { (p[i] + p[n + (i / h % INPUTS) * h + i % h]).clamp(-CLIP, CLIP) } else { p[i] })
            .collect();
        self.wt.up(&wt);
        if self.half {
            let n = wt.len();
            launch(self.k.to_half, (n.div_ceil(256) as u32, 1), 256, std::ptr::null_mut(), &[Arg::P(self.wt.ptr), Arg::P(self.wth.ptr), Arg::I(n as i32)]);
            ck(unsafe { cuStreamSynchronize(std::ptr::null_mut()) }, "to_half");
        }
        self.live = true;
    }
}

impl Backend for GpuTrainer {
    fn describe(&self) -> String {
        format!("kernels cuda ({}), deterministic, work items of {WI} pairs{}", self.device, if self.half { ", fp16 forward weights (--half)" } else { "" })
    }

    fn step(&mut self, pool: &Pool, p: &mut [f32], opt: &mut Adam, recs: &[Rec], mirror: &[bool], lambda: f32, kl: &KingLayout, lr: f32, wd: f32) -> f64 {
        if !self.live {
            self.import(p, opt);
        }
        let (sh, n) = (self.sh, recs.len());
        let (h, kb) = (sh.h, sh.kb);
        let si = self.slot;
        self.slot ^= 1;
        let wd_active = self.wd_active;
        // Prepare this batch on the CPU while the GPU runs the previous one
        // (this slot's last batch, two steps ago, finished: `step` waited
        // for the batch after it).
        let sl = &mut self.slots[si];
        assert!(n <= sl.hs.n, "batch larger than --batch");
        {
            let hs = SyncPtr(sl.hs.ptr);
            let act = &self.act;
            const DB: usize = 512;
            par_units(pool, n.div_ceil(DB), &|u| {
                for i in u * DB..((u + 1) * DB).min(n) {
                    let s = decode_sh(&recs[i], lambda, sh, mirror[i], kl);
                    if wd_active {
                        for side in 0..2 {
                            for &f in &s.f[side][..s.n] {
                                let a = &act[f as usize];
                                if a.load(Ordering::Relaxed) == 0 {
                                    a.store(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                    // SAFETY: every unit writes its own samples.
                    unsafe { *hs.get().add(i) = GSample { f: s.f, n: s.n as u8, np: s.np as u8, bucket: s.bucket as u8, pad: 0, target: s.target } };
                }
            });
        }
        let hs = &sl.hs.slice()[..n];
        let kbk = |s: &GSample, side: usize| if s.np == 0 { 0 } else { s.f[side][0] as usize / INPUTS };
        // Forward order: by (side-to-move bucket, other bucket); pairs and
        // work items by king bucket (stable counting sorts).
        let mut cnt = vec![0usize; kb * kb + 1];
        for s in hs {
            cnt[kbk(s, 0) * kb + kbk(s, 1) + 1] += 1;
        }
        for i in 0..kb * kb {
            cnt[i + 1] += cnt[i];
        }
        let order = sl.horder.slice();
        for (i, s) in hs.iter().enumerate() {
            let c = &mut cnt[kbk(s, 0) * kb + kbk(s, 1)];
            order[*c] = i as u32;
            *c += 1;
        }
        let mut bc = vec![0usize; kb + 1];
        for s in hs {
            for side in 0..2 {
                bc[kbk(s, side) + 1] += 1;
            }
        }
        for b in 0..kb {
            bc[b + 1] += bc[b];
        }
        let mut pos = bc[..kb].to_vec();
        let pairs = sl.hpairs.slice();
        for (i, s) in hs.iter().enumerate() {
            for side in 0..2 {
                let b = kbk(s, side);
                pairs[pos[b]] = (i as u32) << 1 | side as u32;
                pos[b] += 1;
            }
        }
        let (items, ilo, rs, rb) = (sl.hitems.slice(), sl.hilo.slice(), sl.hrs.slice(), sl.hrb.slice());
        let (mut ni, mut nr) = (0, 0);
        for b in 0..kb {
            ilo[b] = ni as i32;
            let mut a = bc[b];
            while a < bc[b + 1] {
                let e = (a + WI).min(bc[b + 1]);
                items[ni] = Item { bucket: b as i32, start: a as i32, end: e as i32, range: nr as i32 };
                ni += 1;
                let mut r = a;
                while r < e {
                    (rs[nr], rb[nr]) = (r as i32, b as i32);
                    nr += 1;
                    r = (r + R).min(e);
                }
                a = e;
            }
        }
        ilo[kb] = ni as i32;
        // Threat block(s): every pair, in sample order, after the bucket-sorted
        // pairs (block index kb + t; plan_masks reads their threat features).
        let tb = sh.tb();
        for t in 0..tb {
            let base = (2 + 2 * t) * n;
            for i in 0..2 * n {
                pairs[base + i] = i as u32;
            }
            let mut a = base;
            while a < base + 2 * n {
                let e = (a + WI).min(base + 2 * n);
                items[ni] = Item { bucket: (kb + t) as i32, start: a as i32, end: e as i32, range: nr as i32 };
                ni += 1;
                let mut r = a;
                while r < e {
                    (rs[nr], rb[nr]) = (r as i32, (kb + t) as i32);
                    nr += 1;
                    r = (r + R).min(e);
                }
                a = e;
            }
            ilo[kb + t + 1] = ni as i32;
        }
        // Ranges tile 0..2n(1 + tb) in order, so range r ends where r + 1 starts.
        rs[nr] = (2 * n * (1 + tb)) as i32;
        if wd_active {
            for (d, a) in sl.hact.slice().iter_mut().zip(&self.act) {
                *d = a.swap(0, Ordering::Relaxed);
            }
        }
        let (st, cs) = (self.stream, self.copy);
        // The slot's device inputs were last read by batch k-2 (done).
        ck(unsafe { cuStreamWaitEvent(cs, sl.done, 0) }, "wait event");
        sl.ss.up_async(&sl.hs, n, cs);
        sl.order.up_async(&sl.horder, n, cs);
        sl.pairs.up_async(&sl.hpairs, 2 * n * (1 + sh.tb()), cs);
        sl.items.up_async(&sl.hitems, ni, cs);
        sl.ilo.up_async(&sl.hilo, kb + sh.tb() + 1, cs);
        sl.rs.up_async(&sl.hrs, nr + 1, cs);
        sl.rb.up_async(&sl.hrb, nr, cs);
        if wd_active {
            sl.active.up_async(&sl.hact, sl.hact.n, cs);
        }
        ck(unsafe { cuEventRecord(sl.copied, cs) }, "event record");
        ck(unsafe { cuStreamWaitEvent(st, sl.copied, 0) }, "wait event");

        opt.step += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let c = AdamC {
            c1: 1.0 - b1.powi(opt.step.min(1 << 20) as i32),
            c2: 1.0 - b2.powi(opt.step.min(1 << 20) as i32),
            lr,
            decay: 1.0 - lr * wd,
            inv_batch: 1.0 / n as f32,
        };
        let (hi, bi, nn) = (h as i32, sh.buckets as i32, n as i32);
        let nrest = sh.len() - sh.ft_b();
        use Arg::*;
        let k = &self.k;
        let (fwd, wt) = if self.half { (k.forward_half, self.wth.ptr) } else { (k.forward, self.wt.ptr) };
        launch(fwd, (n as u32, 1), (h / 2) as u32, st, &[P(sl.ss.ptr), P(sl.order.ptr), P(wt), P(self.pr.ptr), P(self.acc.ptr), P(self.gout.ptr), P(sl.loss.ptr), I(hi), I(bi)]);
        launch(k.plan_masks, (nr as u32, 1), 256, st, &[P(sl.ss.ptr), P(sl.pairs.ptr), P(sl.rs.ptr), P(sl.rb.ptr), P(self.masks.ptr), I(kb as i32)]);
        launch(
            k.backward_ft,
            ((h / 32) as u32, ni as u32),
            1024,
            st,
            &[P(sl.ss.ptr), P(sl.pairs.ptr), P(sl.items.ptr), P(self.masks.ptr), P(self.acc.ptr), P(self.gout.ptr), P(self.pr.ptr), P(self.part.ptr), I(hi)],
        );
        let chunks = n.div_ceil(CH);
        launch(k.backward_rest, ((2 * h).div_ceil(256) as u32, chunks as u32), 256, st, &[P(sl.ss.ptr), P(self.acc.ptr), P(self.gout.ptr), P(self.pr.ptr), P(self.prest.ptr), I(hi), I(bi), I(nn)]);
        launch(k.reduce_rest, (nrest.div_ceil(256) as u32, 1), 256, st, &[P(self.prest.ptr), P(self.gr.ptr), I(hi), I(bi), I(chunks as i32)]);
        let act = if wd_active { sl.active.ptr } else { 0 };
        launch(
            k.adam_ft,
            ((INPUTS * h).div_ceil(256) as u32, 1),
            256,
            st,
            &[C(c), P(self.pw.ptr), P(self.mw.ptr), P(self.vw.ptr), P(self.pf.ptr), P(self.mf.ptr), P(self.vf.ptr), P(self.wt.ptr), P(if self.half { self.wth.ptr } else { 0 }), P(self.part.ptr), P(sl.ilo.ptr), P(act), I(hi), I(kb as i32), I((self.fl > 0) as i32), I(sh.tb() as i32)],
        );
        launch(k.adam_flat, (nrest.div_ceil(256) as u32, 1), 256, st, &[C(c), P(self.pr.ptr), P(self.mr.ptr), P(self.vr.ptr), P(self.gr.ptr), I(nrest as i32)]);
        ck(unsafe { cuMemcpyDtoHAsync_v2(sl.hloss.ptr as *mut c_void, sl.loss.ptr, n * 4, st) }, "loss download");
        ck(unsafe { cuEventRecord(sl.done, st) }, "event record");
        // Return the previous batch's loss (it runs ahead of this one).
        let prev = self.pending.replace((si, n));
        prev.map_or(0.0, |(ps, pn)| self.loss_of(ps, pn))
    }

    fn flush(&mut self, _p: &mut [f32], _opt: &mut Adam) -> f64 {
        self.pending.take().map_or(0.0, |(ps, pn)| self.loss_of(ps, pn))
    }

    fn export(&mut self, p: &mut [f32], opt: &mut Adam) {
        if !self.live {
            return;
        }
        ck(unsafe { cuStreamSynchronize(self.stream) }, "sync");
        let (ftb, n) = (self.sh.ft_b(), self.sh.len());
        self.pw.down_at(0, &mut p[..ftb]);
        self.mw.down_at(0, &mut opt.m[..ftb]);
        self.vw.down_at(0, &mut opt.v[..ftb]);
        self.pr.down_at(0, &mut p[ftb..n]);
        self.mr.down_at(0, &mut opt.m[ftb..n]);
        self.vr.down_at(0, &mut opt.v[ftb..n]);
        if self.fl > 0 {
            self.pf.down_at(0, &mut p[n..]);
            self.mf.down_at(0, &mut opt.m[n..]);
            self.vf.down_at(0, &mut opt.v[n..]);
        }
    }
}

#[derive(Clone, Copy)]
struct SyncPtr(*mut GSample);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}
impl SyncPtr {
    fn get(self) -> *mut GSample {
        self.0
    }
}

// ---------------------------------------------------------------- GPU + CPU pair (experiment)

/// `NNUE_PAIR=K,cpu_threads,gpu_share`: train one net on the GPU and the CPU
/// at once (local SGD / periodic averaging). Each device keeps its own copy
/// of the net and the Adam moments; batch i of every round of K goes to the
/// GPU for a fixed `gpu_share` of them (the pattern depends only on i, so
/// runs are deterministic), the rest to a CPU worker thread with its own
/// pool. After every K batches (and at every epoch end) both wait, average
/// params and moments (equal weights), and continue from the average. Each
/// keeps its own AdamW step count.
struct PairBackend {
    gpu: GpuTrainer,
    k: usize,
    share: usize,
    n: usize,
    jobs: SyncSender<Job>,
    replies: Receiver<Reply>,
    cpu_out: usize,
    cpu_started: bool,
    rounds: u64,
    /// `sum`: merge by adding both devices' parameter changes since the
    /// last merge (`base`), instead of averaging the parameters.
    sum: bool,
    base: Vec<f32>,
}

enum Job {
    State(Vec<f32>, Adam),
    Batch { recs: Vec<Rec>, mirror: Vec<bool>, lambda: f32, kl: KingLayout, lr: f32, wd: f32 },
    Export,
}

enum Reply {
    Loss(f64),
    State(Vec<f32>, Adam),
}

unsafe impl Send for PairBackend {}

impl PairBackend {
    fn new(cfg: &BackendCfg, spec: &str) -> Self {
        let v: Vec<&str> = spec.split(',').collect();
        assert!(v.len() == 3 || v.len() == 4, "NNUE_PAIR=K,cpu_threads,gpu_share[,avg|sum] (e.g. 64,10,0.62)");
        let sum = match v.get(3).copied().unwrap_or("avg") {
            "avg" => false,
            "sum" => true,
            o => panic!("NNUE_PAIR merge mode {o}: avg or sum"),
        };
        let k: usize = v[0].parse().unwrap();
        let cpu_threads: usize = v[1].parse().unwrap();
        let g: f64 = v[2].parse().unwrap();
        let share = ((g * k as f64).round() as usize).clamp(1, k - 1);
        let (jobs, job_rx) = sync_channel::<Job>(8);
        let (reply_tx, replies) = sync_channel::<Reply>(64);
        let (sh, fl, wd_active, half) = (cfg.sh, cfg.fl, cfg.wd_active, cfg.half);
        std::thread::spawn(move || {
            Pool::with(cpu_threads, |pool| {
                let mut bt = BatchTrainer::with_half(sh, 0, false, fl, wd_active, half);
                let mut st: Option<(Vec<f32>, Adam)> = None;
                for job in job_rx {
                    match job {
                        Job::State(p, o) => {
                            st = Some((p, o));
                            bt.invalidate();
                        }
                        Job::Batch { recs, mirror, lambda, kl, lr, wd } => {
                            let (p, o) = st.as_mut().unwrap();
                            let l = bt.step(pool, p, o, &recs, &mirror, lambda, &kl, lr, wd);
                            reply_tx.send(Reply::Loss(l)).unwrap();
                        }
                        Job::Export => {
                            let (mut p, mut o) = st.take().unwrap();
                            bt.export(&mut p, &mut o);
                            reply_tx.send(Reply::State(p, o)).unwrap();
                        }
                    }
                }
            })
        });
        eprintln!("pair: rounds of {k} batches, {share} on the GPU, {} on the CPU ({cpu_threads} threads), merge {}", k - share, if sum { "sum of changes" } else { "average" });
        PairBackend { gpu: GpuTrainer::new(cfg), k, share, n: 0, jobs, replies, cpu_out: 0, cpu_started: false, rounds: 0, sum, base: Vec::new() }
    }

    fn to_gpu(&self, i: usize) -> bool {
        (i + 1) * self.share / self.k > i * self.share / self.k
    }

    fn drain(&mut self, block: bool) -> f64 {
        let mut l = 0.0;
        while self.cpu_out > 0 {
            let r = if block { self.replies.recv().ok() } else { self.replies.try_recv().ok() };
            match r {
                Some(Reply::Loss(x)) => {
                    l += x;
                    self.cpu_out -= 1;
                }
                Some(Reply::State(..)) => unreachable!(),
                None => break,
            }
        }
        l
    }

    /// Waits for both, averages into `p` / `opt` and hands both the result.
    fn sync(&mut self, p: &mut [f32], opt: &mut Adam) -> f64 {
        let mut loss = self.gpu.flush(p, opt);
        self.gpu.export(p, opt);
        loss += self.drain(true);
        if !self.cpu_started {
            return loss;
        }
        self.jobs.send(Job::Export).unwrap();
        let (mut pc, mut oc) = match self.replies.recv().unwrap() {
            Reply::State(pc, oc) => (pc, oc),
            Reply::Loss(_) => unreachable!(),
        };
        if self.sum {
            // p0 + (pg - p0) + (pc - p0), clipped as AdamW clips.
            for ((a, b), &z) in p.iter_mut().zip(pc.iter_mut()).zip(&self.base) {
                *a = (*a + *b - z).clamp(-CLIP, CLIP);
                *b = *a;
            }
            self.base.copy_from_slice(p);
        } else {
            for (a, b) in p.iter_mut().zip(pc.iter_mut()) {
                *a = 0.5 * (*a + *b);
                *b = *a;
            }
        }
        for (a, b) in opt.m.iter_mut().zip(oc.m.iter_mut()) {
            *a = 0.5 * (*a + *b);
            *b = *a;
        }
        for (a, b) in opt.v.iter_mut().zip(oc.v.iter_mut()) {
            *a = 0.5 * (*a + *b);
            *b = *a;
        }
        self.gpu.invalidate();
        self.jobs.send(Job::State(pc, oc)).unwrap();
        self.rounds += 1;
        loss
    }
}

impl Backend for PairBackend {
    fn describe(&self) -> String {
        format!("{}; pair with the CPU: average every {} batches ({} GPU / {} CPU)", self.gpu.describe(), self.k, self.share, self.k - self.share)
    }

    fn step(&mut self, pool: &Pool, p: &mut [f32], opt: &mut Adam, recs: &[Rec], mirror: &[bool], lambda: f32, kl: &KingLayout, lr: f32, wd: f32) -> f64 {
        if !self.cpu_started {
            let o = Adam { m: opt.m.clone(), v: opt.v.clone(), step: opt.step };
            self.jobs.send(Job::State(p.to_vec(), o)).unwrap();
            self.base = p.to_vec();
            self.cpu_started = true;
        }
        let i = self.n % self.k;
        self.n += 1;
        let mut loss = 0.0;
        if self.to_gpu(i) {
            loss += self.gpu.step(pool, p, opt, recs, mirror, lambda, kl, lr, wd);
        } else {
            self.jobs.send(Job::Batch { recs: recs.to_vec(), mirror: mirror.to_vec(), lambda, kl: *kl, lr, wd }).unwrap();
            self.cpu_out += 1;
        }
        loss += self.drain(false);
        if self.n % self.k == 0 {
            loss += self.sync(p, opt);
        }
        loss
    }

    fn export(&mut self, p: &mut [f32], opt: &mut Adam) {
        // After `flush` (epoch ends) p / opt hold the average already.
        self.gpu.export(p, opt);
    }

    fn flush(&mut self, p: &mut [f32], opt: &mut Adam) -> f64 {
        // Epoch end: average now, so the exported / validated net is the average.
        self.sync(p, opt)
    }
}

fn main() {
    if let Ok(spec) = std::env::var("NNUE_PAIR") {
        nnue_trainer::cli(Some(&move |cfg: &BackendCfg| Box::new(PairBackend::new(cfg, &spec)) as Box<dyn Backend>));
    } else {
        nnue_trainer::cli(Some(&|cfg: &BackendCfg| Box::new(GpuTrainer::new(cfg)) as Box<dyn Backend>));
    }
}
