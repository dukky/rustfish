//! CUDA backend for `tools/nnue-trainer`: the same command line, data feed,
//! RNG streams, init, validation, export and checkpoints (it is the CPU
//! trainer's `cli` with a GPU `Backend`); forward, backward and AdamW run
//! on the GPU (`src/kernels.cu`). Deterministic (fixed summation orders),
//! but in a different order from the CPU kernels, so nets match the CPU
//! trainer's closely, not bit for bit. Decode and the king-bucket sort run
//! on the CPU pool (`--threads`). Std only: the driver API is called
//! through the small FFI below.

use nnue_trainer::batch::{par_units, Pool};
use nnue_trainer::{decode, Adam, Backend, BackendCfg, KingLayout, Rec, Shape, CLIP, INPUTS};
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
    fn cuCtxSynchronize() -> CuResult;
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

/// Kernel argument (passed by address to cuLaunchKernel).
enum Arg {
    P(CuPtr),
    I(i32),
    C(AdamC),
}

fn launch(f: Handle, grid: (u32, u32), block: u32, args: &[Arg]) {
    let mut vals: Vec<*mut c_void> = args
        .iter()
        .map(|a| match a {
            Arg::P(p) => p as *const CuPtr as *mut c_void,
            Arg::I(i) => i as *const i32 as *mut c_void,
            Arg::C(c) => c as *const AdamC as *mut c_void,
        })
        .collect();
    let r = unsafe { cuLaunchKernel(f, grid.0, grid.1, 1, block, 1, 1, 0, std::ptr::null_mut(), vals.as_mut_ptr(), std::ptr::null_mut()) };
    ck(r, "launch");
}

// ---------------------------------------------------------------- GPU-side types (kernels.cu)

#[repr(C)]
#[derive(Clone, Copy)]
struct GSample {
    f: [[u16; 32]; 2],
    n: u8,
    bucket: u8,
    pad: u16,
    target: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Item {
    bucket: i32,
    start: i32,
    end: i32,
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
/// Samples per output-layer gradient chunk (kernels.cu `CH`).
const CH: usize = 256;

struct Kernels {
    forward: Handle,
    backward_ft: Handle,
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
    /// Merged FT rows read by the forward.
    wt: DBuf<f32>,
    part: DBuf<f32>,
    prest: DBuf<f32>,
    acc: DBuf<f32>,
    gout: DBuf<f32>,
    loss: DBuf<f32>,
    ss: DBuf<GSample>,
    order: DBuf<u32>,
    pairs: DBuf<u32>,
    items: DBuf<Item>,
    ilo: DBuf<i32>,
    active: DBuf<u8>,
    // Host staging.
    hs: Vec<GSample>,
    hact: Vec<AtomicU8>,
}

unsafe impl Send for GpuTrainer {}

impl GpuTrainer {
    fn new(cfg: &BackendCfg) -> Self {
        let sh = cfg.sh;
        assert!(sh.h % 64 == 0 && sh.h <= 2048, "nnue-trainer-cuda: --hidden must be a multiple of 64, at most 2048");
        assert!(sh.buckets <= 8 && sh.kb <= 64, "nnue-trainer-cuda: at most 8 output buckets, 64 king buckets");
        let (h, rows, b) = (sh.h, sh.kb * INPUTS, cfg.batch);
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
            backward_ft: get("backward_ft"),
            backward_rest: get("backward_rest"),
            reduce_rest: get("reduce_rest"),
            adam_flat: get("adam_flat"),
            adam_ft: get("adam_ft"),
        };
        let max_items = (2 * b).div_ceil(WI) + sh.kb;
        let chunks = b.div_ceil(CH);
        let prest_stride = sh.buckets * 2 * h + 2 * h + sh.buckets;
        let nf = INPUTS * h;
        let blank = GSample { f: [[0; 32]; 2], n: 0, bucket: 0, pad: 0, target: 0.0 };
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
            part: DBuf::new(max_items * nf),
            prest: DBuf::new(chunks * prest_stride),
            acc: DBuf::new(b * 2 * h),
            gout: DBuf::new(b),
            loss: DBuf::new(b),
            ss: DBuf::new(b),
            order: DBuf::new(b),
            pairs: DBuf::new(2 * b),
            items: DBuf::new(max_items),
            ilo: DBuf::new(sh.kb + 1),
            active: DBuf::new(rows),
            hs: vec![blank; b],
            hact: (0..rows).map(|_| AtomicU8::new(0)).collect(),
        }
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
        let wt: Vec<f32> = (0..ftb)
            .map(|i| if self.fl > 0 { (p[i] + p[n + (i / h % INPUTS) * h + i % h]).clamp(-CLIP, CLIP) } else { p[i] })
            .collect();
        self.wt.up(&wt);
        self.live = true;
    }
}

impl Backend for GpuTrainer {
    fn describe(&self) -> String {
        format!("kernels cuda ({}), deterministic, work items of {WI} pairs", self.device)
    }

    fn step(&mut self, pool: &Pool, p: &mut [f32], opt: &mut Adam, recs: &[Rec], mirror: &[bool], lambda: f32, kl: &KingLayout, lr: f32, wd: f32) -> f64 {
        if !self.live {
            self.import(p, opt);
        }
        let (sh, n) = (self.sh, recs.len());
        assert!(n <= self.hs.len(), "batch larger than --batch");
        let (h, kb) = (sh.h, sh.kb);
        // Decode on the CPU pool (and the --wd-active-only row flags).
        {
            let hs = SyncPtr(self.hs.as_mut_ptr());
            let act = &self.hact;
            let wd_active = self.wd_active;
            const DB: usize = 512;
            par_units(pool, n.div_ceil(DB), &|u| {
                for i in u * DB..((u + 1) * DB).min(n) {
                    let s = decode(&recs[i], lambda, sh.buckets, mirror[i], kl);
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
                    unsafe { *hs.get().add(i) = GSample { f: s.f, n: s.n as u8, bucket: s.bucket as u8, pad: 0, target: s.target } };
                }
            });
        }
        let hs = &self.hs[..n];
        let kbk = |s: &GSample, side: usize| if s.n == 0 { 0 } else { s.f[side][0] as usize / INPUTS };
        // Forward order: by (side-to-move bucket, other bucket); pairs and
        // work items by king bucket (stable counting sorts).
        let mut cnt = vec![0usize; kb * kb + 1];
        for s in hs {
            cnt[kbk(s, 0) * kb + kbk(s, 1) + 1] += 1;
        }
        for i in 0..kb * kb {
            cnt[i + 1] += cnt[i];
        }
        let mut order = vec![0u32; n];
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
        let mut pairs = vec![0u32; 2 * n];
        for (i, s) in hs.iter().enumerate() {
            for side in 0..2 {
                let b = kbk(s, side);
                pairs[pos[b]] = (i as u32) << 1 | side as u32;
                pos[b] += 1;
            }
        }
        let mut items = Vec::new();
        let mut ilo = vec![0i32; kb + 1];
        for b in 0..kb {
            ilo[b] = items.len() as i32;
            let mut a = bc[b];
            while a < bc[b + 1] {
                let e = (a + WI).min(bc[b + 1]);
                items.push(Item { bucket: b as i32, start: a as i32, end: e as i32 });
                a = e;
            }
        }
        ilo[kb] = items.len() as i32;
        self.ss.up(hs);
        self.order.up(&order);
        self.pairs.up(&pairs);
        self.items.up(&items);
        self.ilo.up(&ilo);
        if self.wd_active {
            let flags: Vec<u8> = self.hact.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
            self.active.up(&flags);
        }

        opt.step += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let c = AdamC {
            c1: 1.0 - b1.powi(opt.step.min(1 << 20) as i32),
            c2: 1.0 - b2.powi(opt.step.min(1 << 20) as i32),
            lr,
            decay: 1.0 - lr * wd,
            inv_batch: 1.0 / n as f32,
        };
        let (hi, bi, ni) = (h as i32, sh.buckets as i32, n as i32);
        let nrest = sh.len() - sh.ft_b();
        use Arg::*;
        let k = &self.k;
        launch(k.forward, (n as u32, 1), (h / 2) as u32, &[P(self.ss.ptr), P(self.order.ptr), P(self.wt.ptr), P(self.pr.ptr), P(self.acc.ptr), P(self.gout.ptr), P(self.loss.ptr), I(hi), I(bi)]);
        launch(
            k.backward_ft,
            ((h / 32) as u32, items.len() as u32),
            1024,
            &[P(self.ss.ptr), P(self.pairs.ptr), P(self.items.ptr), P(self.acc.ptr), P(self.gout.ptr), P(self.pr.ptr), P(self.part.ptr), I(hi), I(bi)],
        );
        let chunks = n.div_ceil(CH);
        launch(k.backward_rest, ((2 * h).div_ceil(256) as u32, chunks as u32), 256, &[P(self.ss.ptr), P(self.acc.ptr), P(self.gout.ptr), P(self.pr.ptr), P(self.prest.ptr), I(hi), I(bi), I(ni)]);
        launch(k.reduce_rest, (nrest.div_ceil(256) as u32, 1), 256, &[P(self.prest.ptr), P(self.gr.ptr), I(hi), I(bi), I(chunks as i32)]);
        let act = if self.wd_active { self.active.ptr } else { 0 };
        launch(
            k.adam_ft,
            ((INPUTS * h).div_ceil(256) as u32, 1),
            256,
            &[C(c), P(self.pw.ptr), P(self.mw.ptr), P(self.vw.ptr), P(self.pf.ptr), P(self.mf.ptr), P(self.vf.ptr), P(self.wt.ptr), P(self.part.ptr), P(self.ilo.ptr), P(act), I(hi), I(kb as i32), I((self.fl > 0) as i32)],
        );
        launch(k.adam_flat, (nrest.div_ceil(256) as u32, 1), 256, &[C(c), P(self.pr.ptr), P(self.mr.ptr), P(self.vr.ptr), P(self.gr.ptr), I(nrest as i32)]);
        ck(unsafe { cuCtxSynchronize() }, "step");
        let mut loss = vec![0.0f32; n];
        self.loss.down_at(0, &mut loss);
        loss.iter().map(|&l| l as f64).sum()
    }

    fn export(&mut self, p: &mut [f32], opt: &mut Adam) {
        if !self.live {
            return;
        }
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

fn main() {
    nnue_trainer::cli(Some(&|cfg: &BackendCfg| Box::new(GpuTrainer::new(cfg)) as Box<dyn Backend>));
}
