//! NNUE trainer for rustfish (v2).
//!
//! Trains `(KB*768 -> H) x 2 -> B` SCReLU networks (B = output buckets by
//! piece count, KB = king (input) buckets from `--king-buckets`; KB = B = 1
//! is the layout `src/nnue.rs` runs today) from datagen output. Std only,
//! like the engine.
//!
//! King buckets: each perspective's features live in the block of its own
//! king's bucket (`feature = bucket*768 + side*384 + kind*64 + sq`, squares
//! in the perspective's orientation); mirrored layouts also flip every file
//! (`sq ^ 7`) for a perspective whose king is on files e-h. The layouts and
//! their names match `KingLayout` in `src/nnue.rs`. `--init` from a
//! one-bucket net copies its 768 rows into every bucket.
//!
//! Factoriser (`--factorise`, bucketed layouts only): training also keeps a
//! shared 768 x H block `F`, stored after the net's params in the training
//! vector (and checkpoints). Every bucket's effective row is
//! `clamp(W[kb][f] + F[f], +-1.98)`; `F` gets the sum of all buckets'
//! gradients for feature `f`. Exported nets (per epoch and final) hold the
//! merged rows, so the file format and the engine are unchanged. From
//! scratch `W = 0` and `F` = the one-bucket init (same RNG draws as a KB=1
//! net); `--init` from a one-bucket net sets `F` = its rows, `W = 0`, so
//! the start is exactly that net (mirrored layouts: only for an m1 net).
//!
//! `--keep-best` (needs `--val`): `<out>` is the epoch with the lowest
//! quantised val loss, the start net included, instead of the last one.
//!
//! `--wd-active-only` (default on when KB > 1, `--no-wd-active-only` turns
//! it off): AdamW weight decay on the bucketed FT rows (`W`) only for rows
//! whose feature occurs in the batch, so rarely hit buckets are not shrunk
//! toward zero. `F`, the biases and output weights decay as before.
//!
//! Pipeline:
//!
//! ```text
//! gunzip -c datagen-s9-n8000.gz | nnue-trainer convert - s9.bin   # text -> 32-byte records
//! nnue-trainer shuffle --out pool.bin s4.bin s5.bin s9.bin         # one global shuffle
//! nnue-trainer train --data pool.bin --val val.bin --out net.bin --hidden 256 --epochs 20
//! nnue-trainer eval --net net/nnue.bin --val val.bin              # same metric for any i16 net
//! ```
//!
//! Training runs in float with the standard quantisation scheme: feature
//! weights/biases are exported as `round(x * QA)`, output weights as
//! `round(x * QB)`, output bias as `round(x * QA * QB)`. Every parameter is
//! clipped to +-1.98 while training, so output weights export to |w| <= 127
//! (the engine's fast i32 SIMD path) and accumulators cannot overflow i16.
//!
//! Net output `o` is in units of `eval / SCALE`: the engine computes
//! `eval_cp = SCALE * (sum screlu(acc) * w + b)`, the loss is
//! `(sigmoid(o) - target)^2` with
//! `target = lambda * wdl + (1 - lambda) * sigmoid(score_cp / SCALE)`.

mod batch;
mod simd;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const INPUTS: usize = 768;
const QA: f32 = 255.0;
const QB: f32 = 64.0;
const SCALE: f32 = 400.0;
const CLIP: f32 = 1.98;
const REC: usize = 32;
const CKPT_MAGIC: &[u8; 8] = b"RFNNT2\0\0";

/// Packed position: occupancy (8), piece nibbles in square order (16),
/// score cp White POV i16 (2), result 0/1/2 = Black win/draw/White win (1),
/// side to move 0 = White (1), padding (4).
type Rec = [u8; REC];

// ---------------------------------------------------------------- king buckets

/// A king-bucket layout: bucket per king square (perspective-relative,
/// a1 = 0) and horizontal mirroring. Identical to `src/nnue.rs`.
#[derive(Clone, Copy, PartialEq, Debug)]
struct KingLayout {
    name: &'static str,
    buckets: [u8; 64],
    mirror: bool,
}

/// Rank-major half-board (files a-d, rank 1 first) -> left-right symmetric table.
const fn mirrored(half: [u8; 32]) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 32 {
        let (r, f) = (i / 4, i % 4);
        t[r * 8 + f] = half[i];
        t[r * 8 + 7 - f] = half[i];
        i += 1;
    }
    t
}

/// Index = layout id stored in checkpoints (0 = `none`, the original net).
#[rustfmt::skip]
const LAYOUTS: [KingLayout; 5] = [
    KingLayout { name: "none", buckets: [0; 64], mirror: false },
    KingLayout { name: "m1", buckets: [0; 64], mirror: true },
    KingLayout {
        name: "kb4",
        buckets: [
            0, 0, 0, 0, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3,
        ],
        mirror: false,
    },
    KingLayout {
        name: "kb4m",
        buckets: mirrored([
            0, 0, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
        ]),
        mirror: true,
    },
    KingLayout {
        name: "kb8m",
        buckets: mirrored([
            0, 1, 2, 3,
            4, 4, 5, 5,
            6, 6, 6, 6,
            6, 6, 6, 6,
            7, 7, 7, 7,
            7, 7, 7, 7,
            7, 7, 7, 7,
            7, 7, 7, 7,
        ]),
        mirror: true,
    },
];

impl KingLayout {
    fn by_name(name: &str) -> &'static KingLayout {
        LAYOUTS.iter().find(|l| l.name == name).unwrap_or_else(|| {
            let names: Vec<_> = LAYOUTS.iter().map(|l| l.name).collect();
            panic!("unknown --king-buckets {name} (one of {})", names.join(", "))
        })
    }
    fn id(&self) -> usize {
        LAYOUTS.iter().position(|l| l == self).unwrap()
    }
    fn kb(&self) -> usize {
        *self.buckets.iter().max().unwrap() as usize + 1
    }
    /// Feature offset and square xor for a perspective whose own king is on
    /// `ksq_rel` (already in the perspective's orientation).
    #[inline]
    fn ctx(&self, ksq_rel: u16) -> (u16, u16) {
        let flip = if self.mirror && ksq_rel & 7 >= 4 { 7 } else { 0 };
        (self.buckets[ksq_rel as usize] as u16 * INPUTS as u16, flip)
    }
}

const NO_BUCKETS: &KingLayout = &LAYOUTS[0];

// ---------------------------------------------------------------- utilities

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 32) * n as u64 >> 32) as usize
    }
    fn uniform(&mut self, a: f32) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
    }
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn read_recs(path: &Path) -> Vec<Rec> {
    let mut bytes = Vec::new();
    File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).read_to_end(&mut bytes).unwrap();
    assert!(bytes.len() % REC == 0, "{} is not a record file", path.display());
    bytes.chunks_exact(REC).map(|c| c.try_into().unwrap()).collect()
}

fn threads_default() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

// ---------------------------------------------------------------- encoding

/// `<FEN> | <score_cp White POV> | <result White POV>` -> record.
fn encode(line: &str) -> Option<Rec> {
    let mut parts = line.split('|');
    let fen = parts.next()?.trim();
    let score: f32 = parts.next()?.trim().parse().ok()?;
    let result: f32 = parts.next()?.trim().parse().ok()?;
    let mut fields = fen.split_whitespace();
    let board = fields.next()?;
    let stm = match fields.next()? {
        "w" => 0u8,
        "b" => 1,
        _ => return None,
    };
    let mut squares = [0xFFu8; 64];
    let (mut rank, mut file) = (7i32, 0i32);
    for c in board.chars() {
        match c {
            '/' => {
                rank -= 1;
                file = 0;
            }
            '1'..='8' => file += c as i32 - '0' as i32,
            _ => {
                let kind = "pnbrqk".find(c.to_ascii_lowercase())? as u8;
                let black = c.is_ascii_lowercase() as u8;
                if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                    return None;
                }
                squares[(rank * 8 + file) as usize] = black << 3 | kind;
                file += 1;
            }
        }
    }
    let mut rec = [0u8; REC];
    let mut occ = 0u64;
    let mut n = 0usize;
    for (sq, &p) in squares.iter().enumerate() {
        if p != 0xFF {
            if n >= 32 {
                return None;
            }
            occ |= 1 << sq;
            rec[8 + n / 2] |= p << (4 * (n & 1));
            n += 1;
        }
    }
    rec[..8].copy_from_slice(&occ.to_le_bytes());
    let s = score.round().clamp(-32000.0, 32000.0) as i16;
    rec[24..26].copy_from_slice(&s.to_le_bytes());
    rec[26] = if result == 1.0 {
        2
    } else if result == 0.5 {
        1
    } else if result == 0.0 {
        0
    } else {
        return None;
    };
    rec[27] = stm;
    Some(rec)
}

/// Both perspectives' active features: index 0 = side to move.
#[derive(Clone, Copy)]
struct Sample {
    f: [[u16; 32]; 2],
    n: usize,
    bucket: usize,
    target: f32,
}

/// `mirror` is the `--mirror` data augmentation (flip the whole board's
/// files), applied before anything else; `kl` the king-bucket layout.
#[inline]
fn decode(r: &Rec, lambda: f32, buckets: usize, mirror: bool, kl: &KingLayout) -> Sample {
    let occ0 = u64::from_le_bytes(r[..8].try_into().unwrap());
    let score = i16::from_le_bytes([r[24], r[25]]) as f32;
    let stm_black = r[27] == 1;
    let mut s = Sample { f: [[0; 32]; 2], n: 0, bucket: 0, target: 0.0 };
    let flip = if mirror { 7u16 } else { 0 };
    // King contexts of both perspectives (0 = side to move), from their own kings.
    let mut ctx = [(0u16, 0u16); 2];
    if kl.kb() > 1 || kl.mirror {
        let (mut occ, mut n) = (occ0, 0);
        let mut ksq = [0u16; 2]; // by colour: 0 = White
        while occ != 0 {
            let sq = occ.trailing_zeros() as u16 ^ flip;
            occ &= occ - 1;
            let nib = r[8 + n / 2] >> (4 * (n & 1)) & 0xF;
            if nib & 7 == 5 {
                ksq[(nib >> 3) as usize] = sq;
            }
            n += 1;
        }
        for (i, persp_black) in [stm_black, !stm_black].into_iter().enumerate() {
            let k = ksq[persp_black as usize];
            ctx[i] = kl.ctx(if persp_black { k ^ 56 } else { k });
        }
    }
    let mut occ = occ0;
    while occ != 0 {
        let sq = occ.trailing_zeros() as u16 ^ flip;
        occ &= occ - 1;
        let nib = r[8 + s.n / 2] >> (4 * (s.n & 1)) & 0xF;
        let black = nib >> 3 == 1;
        let kind = (nib & 7) as u16;
        // Perspective 0 is the side to move, 1 the opponent.
        for (i, persp_black) in [stm_black, !stm_black].into_iter().enumerate() {
            let side = if black == persp_black { 0 } else { 384 };
            let rel = if persp_black { sq ^ 56 } else { sq };
            s.f[i][s.n] = ctx[i].0 + side + kind * 64 + (rel ^ ctx[i].1);
        }
        s.n += 1;
    }
    s.bucket = if buckets == 1 { 0 } else { ((s.n.saturating_sub(2)) * buckets / 32).min(buckets - 1) };
    let mut wdl = r[26] as f32 / 2.0;
    let mut sc = score;
    if stm_black {
        wdl = 1.0 - wdl;
        sc = -sc;
    }
    s.target = lambda * wdl + (1.0 - lambda) * sigmoid(sc / SCALE);
    s
}

// ---------------------------------------------------------------- network

#[derive(Clone, Copy, PartialEq, Debug)]
struct Shape {
    h: usize,
    buckets: usize,
    /// King (input) buckets: the FT has `kb * 768` rows.
    kb: usize,
}

impl Shape {
    fn ft_b(&self) -> usize {
        self.kb * INPUTS * self.h
    }
    fn out_w(&self) -> usize {
        self.ft_b() + self.h
    }
    fn out_b(&self) -> usize {
        self.out_w() + self.buckets * 2 * self.h
    }
    fn len(&self) -> usize {
        self.out_b() + self.buckets
    }
    /// Size of the exported i16 file.
    fn file_bytes(&self) -> usize {
        self.len() * 2
    }
    fn from_file_bytes(bytes: usize, buckets: usize, kb: usize) -> Option<Self> {
        let words = bytes / 2;
        let h = words.checked_sub(buckets)? / (kb * INPUTS + 1 + 2 * buckets);
        let s = Shape { h, buckets, kb };
        (s.file_bytes() == bytes).then_some(s)
    }
}

/// Factoriser bookkeeping: the training vector is the net's `sh.len()`
/// params followed by `fl` floats of `F` (`fl = 0` without `--factorise`).
#[derive(Clone, Copy, PartialEq, Debug)]
struct Fac {
    fl: usize,
}

impl Fac {
    fn new(sh: Shape, on: bool) -> Self {
        Fac { fl: if on { INPUTS * sh.h } else { 0 } }
    }
    fn on(&self) -> bool {
        self.fl > 0
    }
}

/// Effective net params from the training vector: FT row `f` of bucket `kb`
/// = `clamp(W[kb][f] + F[f % 768])`, everything else copied.
fn merge_factor(p: &[f32], sh: Shape, fac: Fac, out: &mut [f32]) {
    let n = sh.len();
    let (w, f) = (&p[..sh.ft_b()], &p[n..n + fac.fl]);
    std::thread::scope(|sc| {
        let (ow, rest) = out.split_at_mut(sh.ft_b());
        rest.copy_from_slice(&p[sh.ft_b()..n]);
        for (wb, ob) in w.chunks(fac.fl).zip(ow.chunks_mut(fac.fl)) {
            sc.spawn(move || {
                for ((o, &a), &b) in ob.iter_mut().zip(wb).zip(f) {
                    *o = (a + b).clamp(-CLIP, CLIP);
                }
            });
        }
    });
}

/// Training vector for `--factorise` from a net's params: `F` = `rows`
/// (768 x H), and `W = 0` (or the net's own bucket rows when `rows` is None
/// and `F = 0`).
fn factor_vector(p: Vec<f32>, sh: Shape, rows: Option<&[f32]>) -> Vec<f32> {
    let fl = INPUTS * sh.h;
    let mut v = p;
    match rows {
        Some(r) => {
            v[..sh.ft_b()].fill(0.0);
            v.extend_from_slice(r);
        }
        None => v.resize(sh.len() + fl, 0.0),
    }
    v
}

/// Forward pass. `acc` (len 2h) receives the pre-activation accumulators.
#[inline]
fn forward(p: &[f32], sh: Shape, s: &Sample, acc: &mut [f32]) -> f32 {
    let h = sh.h;
    let fw = &p[..sh.ft_b()];
    let fb = &p[sh.ft_b()..sh.ft_b() + h];
    for side in 0..2 {
        let a = &mut acc[side * h..(side + 1) * h];
        a.copy_from_slice(fb);
        for &f in &s.f[side][..s.n] {
            let row = &fw[f as usize * h..f as usize * h + h];
            for (x, w) in a.iter_mut().zip(row) {
                *x += *w;
            }
        }
    }
    let ow = &p[sh.out_w() + s.bucket * 2 * h..sh.out_w() + (s.bucket + 1) * 2 * h];
    let mut out = p[sh.out_b() + s.bucket];
    let mut sum = 0.0f32;
    for (x, w) in acc.iter().zip(ow) {
        let a = x.clamp(0.0, 1.0);
        sum += a * a * *w;
    }
    out += sum;
    out
}

/// Forward + backward for one sample into gradient buffer `g`; returns loss.
/// The reference for `batch.rs` (and the Metal trainer's gradcheck): the v2
/// training loop ran exactly this per sample.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn train_sample(p: &[f32], sh: Shape, s: &Sample, g: &mut [f32], acc: &mut [f32], dacc: &mut [f32]) -> f32 {
    let h = sh.h;
    let out = forward(p, sh, s, acc);
    let pred = sigmoid(out);
    let err = pred - s.target;
    let grad = 2.0 * err * pred * (1.0 - pred);

    let ow0 = sh.out_w() + s.bucket * 2 * h;
    g[sh.out_b() + s.bucket] += grad;
    {
        let ow = &p[ow0..ow0 + 2 * h];
        let gow = &mut g[ow0..ow0 + 2 * h];
        for i in 0..2 * h {
            let x = acc[i];
            let a = x.clamp(0.0, 1.0);
            gow[i] += grad * a * a;
            dacc[i] = if x > 0.0 && x < 1.0 { grad * ow[i] * 2.0 * a } else { 0.0 };
        }
    }
    let (gw, rest) = g.split_at_mut(sh.ft_b());
    let gb = &mut rest[..h];
    for side in 0..2 {
        let d = &dacc[side * h..(side + 1) * h];
        for (x, y) in gb.iter_mut().zip(d) {
            *x += *y;
        }
        for &f in &s.f[side][..s.n] {
            let row = &mut gw[f as usize * h..f as usize * h + h];
            for (x, y) in row.iter_mut().zip(d) {
                *x += *y;
            }
        }
    }
    err * err
}

fn init_params(sh: Shape, rng: &mut Rng) -> Vec<f32> {
    let mut p = vec![0.0f32; sh.len()];
    let a = 1.0 / (32.0f32).sqrt();
    for x in &mut p[..sh.ft_b()] {
        *x = rng.uniform(a);
    }
    let a = 1.0 / ((2 * sh.h) as f32).sqrt();
    for x in &mut p[sh.out_w()..sh.out_b()] {
        *x = rng.uniform(a);
    }
    p
}

fn quantize(p: &[f32], sh: Shape) -> Vec<u8> {
    let mut out = Vec::with_capacity(sh.file_bytes());
    for (i, &x) in p.iter().enumerate() {
        let scale = if i < sh.out_w() {
            QA
        } else if i < sh.out_b() {
            QB
        } else {
            QA * QB
        };
        let q = (x * scale).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

fn dequantize(bytes: &[u8], sh: Shape) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .enumerate()
        .map(|(i, c)| {
            let q = i16::from_le_bytes([c[0], c[1]]) as f32;
            let scale = if i < sh.out_w() {
                QA
            } else if i < sh.out_b() {
                QB
            } else {
                QA * QB
            };
            q / scale
        })
        .collect()
}

fn load_net(path: &Path, buckets: usize, kb: usize) -> (Vec<f32>, Shape) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let sh = Shape::from_file_bytes(bytes.len(), buckets, kb).unwrap_or_else(|| {
        panic!("{}: {} bytes is no ({kb}*768->H)x2->{buckets} net", path.display(), bytes.len())
    });
    (dequantize(&bytes, sh), sh)
}

/// Warm start: a net with `kb` king buckets from `path`, which may also be a
/// one-bucket net; its 768 FT rows are then copied into every bucket.
fn load_init(path: &Path, buckets: usize, kb: usize) -> (Vec<f32>, Shape) {
    let bytes = std::fs::metadata(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).len() as usize;
    if kb == 1 || Shape::from_file_bytes(bytes, buckets, kb).is_some() {
        return load_net(path, buckets, kb);
    }
    let (p1, s1) = load_net(path, buckets, 1);
    println!("init: {} is a one-bucket net; copying its FT rows into all {kb} king buckets", path.display());
    (expand_buckets(&p1, s1, kb), Shape { kb, ..s1 })
}

/// One-bucket params -> `kb` buckets, each a copy of the 768 FT rows.
fn expand_buckets(p1: &[f32], s1: Shape, kb: usize) -> Vec<f32> {
    assert_eq!(s1.kb, 1);
    let ft = &p1[..s1.ft_b()];
    let mut p = Vec::with_capacity(p1.len() + (kb - 1) * ft.len());
    for _ in 0..kb {
        p.extend_from_slice(ft);
    }
    p.extend_from_slice(&p1[s1.ft_b()..]);
    p
}

/// `--factorise` from scratch: `F` = the one-bucket init (the same RNG draws
/// as a KB=1 net of this shape), `W = 0`.
fn init_factor(sh: Shape, rng: &mut Rng) -> Vec<f32> {
    let s1 = Shape { kb: 1, ..sh };
    let p1 = init_params(s1, rng);
    let mut p = vec![0.0f32; sh.ft_b()];
    p.extend_from_slice(&p1[s1.ft_b()..]);
    factor_vector(p, sh, Some(&p1[..s1.ft_b()]))
}

/// `--factorise --init`: a one-bucket net goes into `F` (`W = 0`, so every
/// bucket starts as that net's rows); a net of the full bucketed shape goes
/// into `W` with `F = 0`.
fn load_init_factor(path: &Path, sh: Shape) -> Vec<f32> {
    let bytes = std::fs::metadata(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).len() as usize;
    if Shape::from_file_bytes(bytes, sh.buckets, sh.kb).is_some() {
        let (p, s) = load_net(path, sh.buckets, sh.kb);
        assert_eq!(s, sh, "init net shape differs from --hidden/--buckets/--king-buckets");
        println!("init: {} has all {} king buckets; factoriser starts at 0", path.display(), sh.kb);
        return factor_vector(p, sh, None);
    }
    let (p1, s1) = load_net(path, sh.buckets, 1);
    assert_eq!(Shape { kb: sh.kb, ..s1 }, sh, "init net shape differs from --hidden/--buckets");
    println!("init: {} is a one-bucket net; its FT rows become the factoriser, buckets start at 0", path.display());
    let mut p = vec![0.0f32; sh.ft_b()];
    p.extend_from_slice(&p1[s1.ft_b()..]);
    factor_vector(p, sh, Some(&p1[..s1.ft_b()]))
}

/// A one-bucket `--init` net into a mirrored layout is exact only if that
/// net was trained mirrored (`m1`): a `none` net cannot be expressed (each
/// mirrored bucket serves kings on files x and 7-x with one row per feature),
/// so training starts from its mirrored reading instead.
fn warn_mirrored_init(path: &Path, sh: Shape, kl: &KingLayout) {
    let bytes = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);
    let one_bucket = Shape::from_file_bytes(bytes, sh.buckets, 1).is_some();
    let full = sh.kb > 1 && Shape::from_file_bytes(bytes, sh.buckets, sh.kb).is_some();
    if kl.mirror && one_bucket && !full {
        println!(
            "init: note: {} is read with mirroring ({}); the start equals it exactly only if it is an m1 net, \
             otherwise the start val loss is its `eval --king-buckets m1` loss",
            path.display(),
            kl.name
        );
    }
}

/// Mean validation loss, plus the mean |eval| in cp as a scale check.
fn validate(p: &[f32], sh: Shape, kl: &KingLayout, val: &[Rec], lambda: f32, threads: usize) -> (f64, f64) {
    if val.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let chunk = val.len().div_ceil(threads);
    let parts: Vec<(f64, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = val
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    let mut acc = vec![0.0f32; 2 * sh.h];
                    let (mut l, mut e) = (0.0f64, 0.0f64);
                    for r in part {
                        let s = decode(r, lambda, sh.buckets, false, kl);
                        let o = forward(p, sh, &s, &mut acc);
                        let d = sigmoid(o) - s.target;
                        l += (d * d) as f64;
                        e += (o * SCALE).abs() as f64;
                    }
                    (l, e)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let n = val.len() as f64;
    (parts.iter().map(|x| x.0).sum::<f64>() / n, parts.iter().map(|x| x.1).sum::<f64>() / n)
}

// ---------------------------------------------------------------- optimiser

/// A 64-byte aligned `[f32]` (the batch kernels' AdamW writes whole cache
/// lines of the FT with non-temporal stores).
#[derive(Clone)]
struct AVec {
    buf: Vec<Line>,
    len: usize,
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([f32; 16]);

impl AVec {
    fn zeros(len: usize) -> Self {
        AVec { buf: vec![Line([0.0; 16]); len.div_ceil(16)], len }
    }
}

impl From<Vec<f32>> for AVec {
    fn from(v: Vec<f32>) -> Self {
        let mut a = AVec::zeros(v.len());
        a.copy_from_slice(&v);
        a
    }
}

impl std::ops::Deref for AVec {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // SAFETY: `Line` is 16 f32 with no padding; `len <= 16 * buf.len()`.
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const f32, self.len) }
    }
}

impl std::ops::DerefMut for AVec {
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: as in `deref`.
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut f32, self.len) }
    }
}

struct Adam {
    m: AVec,
    v: AVec,
    step: u64,
}

/// What `adam_step` needs beyond the plain net: the factoriser block (its
/// gradient is the sum of every bucket's FT-row gradient) and, for
/// `--wd-active-only`, which FT rows occurred in the batch.
struct StepCtx<'a> {
    sh: Shape,
    fac: Fac,
    active: Option<&'a [u8]>,
}

/// Sum the per-thread gradients, then one AdamW step with clipping, in parallel.
/// The v2 step, kept as the reference for `batch.rs` (tests).
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn adam_step(p: &mut [f32], opt: &mut Adam, grads: &[Vec<f32>], lr: f32, wd: f32, inv_batch: f32, threads: usize, cx: &StepCtx) {
    opt.step += 1;
    let (n, fl, h, kb, ft_b, active) = (cx.sh.len(), cx.fac.fl, cx.sh.h, cx.sh.kb, cx.sh.ft_b(), cx.active);
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let c1 = 1.0 - b1.powi(opt.step.min(1 << 20) as i32);
    let c2 = 1.0 - b2.powi(opt.step.min(1 << 20) as i32);
    let chunk = p.len().div_ceil(threads);
    std::thread::scope(|sc| {
        for (k, ((pc, mc), vc)) in p.chunks_mut(chunk).zip(opt.m.chunks_mut(chunk)).zip(opt.v.chunks_mut(chunk)).enumerate() {
            let base = k * chunk;
            sc.spawn(move || {
                for i in 0..pc.len() {
                    let gi = base + i;
                    let mut g = 0.0f32;
                    if gi < n {
                        for gr in grads {
                            g += gr[gi];
                        }
                    } else {
                        // Factoriser row element: every bucket's gradient for it.
                        let k = gi - n;
                        for b in 0..kb {
                            for gr in grads {
                                g += gr[b * fl + k];
                            }
                        }
                    }
                    g *= inv_batch;
                    mc[i] = b1 * mc[i] + (1.0 - b1) * g;
                    vc[i] = b2 * vc[i] + (1.0 - b2) * g * g;
                    let upd = (mc[i] / c1) / ((vc[i] / c2).sqrt() + eps);
                    let decay = match active {
                        Some(a) if gi < ft_b && a[gi / h] == 0 => 1.0,
                        _ => 1.0 - lr * wd,
                    };
                    let x = pc[i] * decay - lr * upd;
                    pc[i] = x.clamp(-CLIP, CLIP);
                }
            });
        }
    });
}

/// Header word 1 = output buckets | king layout id << 32 | factoriser << 40
/// (0 for `none`, so one-bucket checkpoints are unchanged). With the
/// factoriser, `p`, `m` and `v` are the training vector (net params + F).
fn ckpt_word1(sh: Shape, kl: &KingLayout, fac: Fac) -> u64 {
    sh.buckets as u64 | (kl.id() as u64) << 32 | (fac.on() as u64) << 40
}

fn save_ckpt(path: &Path, sh: Shape, kl: &KingLayout, fac: Fac, p: &[f32], opt: &Adam, seen: u64) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    w.write_all(CKPT_MAGIC).unwrap();
    for x in [sh.h as u64, ckpt_word1(sh, kl, fac), opt.step, seen] {
        w.write_all(&x.to_le_bytes()).unwrap();
    }
    for arr in [p, &opt.m[..], &opt.v[..]] {
        for x in arr {
            w.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

fn load_ckpt(path: &Path, sh: Shape, kl: &KingLayout, fac: Fac) -> (Vec<f32>, Adam, u64) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], CKPT_MAGIC, "not a v2 checkpoint");
    let word = |i: usize| u64::from_le_bytes(bytes[8 + 8 * i..16 + 8 * i].try_into().unwrap());
    let got = (word(0) as usize, word(1));
    assert_eq!(
        got,
        (sh.h, ckpt_word1(sh, kl, fac)),
        "checkpoint shape (H, buckets | king layout id << 32 | factorise << 40) differs"
    );
    let floats: Vec<f32> = bytes[40..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let n = sh.len() + fac.fl;
    assert_eq!(floats.len(), 3 * n);
    (floats[..n].to_vec(), Adam { m: floats[n..2 * n].to_vec().into(), v: floats[2 * n..].to_vec().into(), step: word(2) }, word(3))
}

// ---------------------------------------------------------------- data feed

/// Streams records from files in order, shuffling within 1M-record chunks.
struct Feed {
    files: Vec<PathBuf>,
    idx: usize,
    reader: Option<BufReader<File>>,
    buf: Vec<Rec>,
    pos: usize,
    rng: Rng,
}

impl Feed {
    const CHUNK: usize = 1 << 20;

    fn new(files: Vec<PathBuf>, seed: u64) -> Self {
        Self { files, idx: 0, reader: None, buf: Vec::new(), pos: 0, rng: Rng::new(seed) }
    }

    /// Next record, wrapping around to the first file at the end (epochs are
    /// counted in positions, not passes).
    fn refill(&mut self) {
        self.buf.clear();
        self.pos = 0;
        let mut raw = vec![0u8; Self::CHUNK * REC];
        let mut filled = 0;
        while filled < raw.len() {
            if self.reader.is_none() {
                let f = &self.files[self.idx % self.files.len()];
                self.reader = Some(BufReader::with_capacity(1 << 22, File::open(f).unwrap()));
            }
            let n = self.reader.as_mut().unwrap().read(&mut raw[filled..]).unwrap();
            if n == 0 {
                self.reader = None;
                self.idx += 1;
                if filled > 0 {
                    break;
                }
                continue;
            }
            filled += n;
        }
        filled -= filled % REC;
        self.buf.extend(raw[..filled].chunks_exact(REC).map(|c| -> Rec { c.try_into().unwrap() }));
        shuffle(&mut self.buf, &mut self.rng);
    }

    fn batch(&mut self, out: &mut Vec<Rec>, n: usize) {
        out.clear();
        while out.len() < n {
            if self.pos == self.buf.len() {
                self.refill();
            }
            let take = (n - out.len()).min(self.buf.len() - self.pos);
            out.extend_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
        }
    }
}

// ---------------------------------------------------------------- commands

fn cmd_convert(args: &[String]) {
    assert!(args.len() == 2, "usage: convert <in.txt|-> <out.bin>");
    let input: Box<dyn BufRead> = if args[0] == "-" {
        Box::new(BufReader::with_capacity(1 << 22, std::io::stdin()))
    } else {
        Box::new(BufReader::with_capacity(1 << 22, File::open(&args[0]).unwrap()))
    };
    let mut out = BufWriter::with_capacity(1 << 22, File::create(&args[1]).unwrap());
    let (mut ok, mut bad) = (0u64, 0u64);
    for line in input.lines() {
        let line = line.unwrap();
        match encode(&line) {
            Some(r) => {
                out.write_all(&r).unwrap();
                ok += 1;
            }
            None => bad += 1,
        }
    }
    out.flush().unwrap();
    eprintln!("{}: {ok} positions, {bad} skipped", args[1]);
}

fn cmd_shuffle(args: &[String]) {
    let mut out_path = None;
    let mut inputs = Vec::new();
    // `--consume <file>`: an input deleted as soon as it has been scattered, so a
    // new pool can be built from the previous one without room for both.
    let mut consume = Vec::new();
    let mut seed = 1u64;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out_path = Some(PathBuf::from(it.next().unwrap())),
            "--seed" => seed = it.next().unwrap().parse().unwrap(),
            "--consume" => {
                let p = PathBuf::from(it.next().unwrap());
                consume.push(p.clone());
                inputs.push(p);
            }
            _ => inputs.push(PathBuf::from(a)),
        }
    }
    let out_path = out_path.expect("--out required");
    let total: u64 = inputs.iter().map(|p| std::fs::metadata(p).unwrap().len() / REC as u64).sum();
    let buckets = (total / (16 << 20) + 1) as usize; // ~512 MB per bucket in RAM
    let tmp: Vec<PathBuf> = (0..buckets).map(|i| out_path.with_extension(format!("tmp{i}"))).collect();
    let mut rng = Rng::new(seed);
    {
        let mut ws: Vec<BufWriter<File>> =
            tmp.iter().map(|p| BufWriter::with_capacity(1 << 20, File::create(p).unwrap())).collect();
        for inp in &inputs {
            let mut r = BufReader::with_capacity(1 << 22, File::open(inp).unwrap());
            let mut rec = [0u8; REC];
            while r.read_exact(&mut rec).is_ok() {
                ws[rng.below(buckets)].write_all(&rec).unwrap();
            }
            if consume.contains(inp) {
                for w in ws.iter_mut() {
                    w.flush().unwrap();
                }
                std::fs::remove_file(inp).unwrap();
                eprintln!("{}: scattered and removed (--consume)", inp.display());
            }
        }
    }
    let mut out = BufWriter::with_capacity(1 << 22, File::create(&out_path).unwrap());
    for p in &tmp {
        let mut recs = read_recs(p);
        shuffle(&mut recs, &mut rng);
        for r in &recs {
            out.write_all(r).unwrap();
        }
        std::fs::remove_file(p).unwrap();
    }
    out.flush().unwrap();
    eprintln!("{}: {total} positions in {buckets} buckets", out_path.display());
}

fn cmd_eval(args: &[String]) {
    let (mut net, mut val, mut buckets, mut lambda) = (None, None, 1usize, 0.5f32);
    let mut kl = NO_BUCKETS;
    let mut threads = threads_default();
    let mut best_scale = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--net" => net = Some(PathBuf::from(it.next().unwrap())),
            "--val" => val = Some(PathBuf::from(it.next().unwrap())),
            "--buckets" => buckets = it.next().unwrap().parse().unwrap(),
            "--king-buckets" => kl = KingLayout::by_name(it.next().unwrap()),
            "--lambda" => lambda = it.next().unwrap().parse().unwrap(),
            "--threads" => threads = it.next().unwrap().parse().unwrap(),
            "--best-scale" => best_scale = true,
            other => panic!("unknown option {other}"),
        }
    }
    let (p, sh) = load_net(&net.expect("--net"), buckets, kl.kb());
    let val = read_recs(&val.expect("--val"));
    let (l, e) = validate(&p, sh, kl, &val, lambda, threads);
    print!("H={} buckets={} ", sh.h, sh.buckets);
    if kl.id() != 0 {
        print!("king-buckets={} ", kl.name);
    }
    println!("val positions {} loss {:.6} mean|eval| {:.1}cp", val.len(), l, e);
    if best_scale {
        let (f, lf) = best_scale_loss(&p, sh, kl, &val, lambda, threads);
        println!("best scale x{f:.4}: loss {lf:.6} mean|eval| {:.1}cp", e * f);
    }
}

/// The output scale `f` minimising validation loss (evals multiplied by `f`,
/// as `rescale --factor f` does up to rounding), and that loss. Comparing
/// nets at their own best scale measures quality with scale profiled out.
fn best_scale_loss(p: &[f32], sh: Shape, kl: &KingLayout, val: &[Rec], lambda: f32, threads: usize) -> (f64, f64) {
    let chunk = val.len().div_ceil(threads.max(1));
    let ot: Vec<(f32, f32)> = std::thread::scope(|sc| {
        let hs: Vec<_> = val
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    let mut acc = vec![0.0f32; 2 * sh.h];
                    part.iter()
                        .map(|r| {
                            let s = decode(r, lambda, sh.buckets, false, kl);
                            (forward(p, sh, &s, &mut acc), s.target)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let loss = |f: f64| {
        ot.iter().map(|&(o, t)| (sigmoid((o as f64 * f) as f32) - t) as f64).map(|d| d * d).sum::<f64>() / ot.len() as f64
    };
    // Golden-section search on [0.5, 2]; the loss is unimodal in f in practice.
    let (mut a, mut b) = (0.5f64, 2.0f64);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut x1, mut x2) = (b - g * (b - a), a + g * (b - a));
    let (mut f1, mut f2) = (loss(x1), loss(x2));
    while b - a > 1e-4 {
        if f1 < f2 {
            (b, x2, f2) = (x2, x1, f1);
            x1 = b - g * (b - a);
            f1 = loss(x1);
        } else {
            (a, x1, f1) = (x1, x2, f2);
            x2 = a + g * (b - a);
            f2 = loss(x2);
        }
    }
    let f = (a + b) / 2.0;
    (f, loss(f))
}

/// Least-squares slope of `score` on `eval` through the origin
/// (`sum(s*e) / sum(e^2)`), Pearson r, and mean |score|.
fn scale_stats(pairs: &[(f64, f64)]) -> (f64, f64, f64) {
    let n = pairs.len() as f64;
    let (mut ss, mut se, mut sss, mut see, mut sse, mut sa) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &(s, e) in pairs {
        ss += s;
        se += e;
        sss += s * s;
        see += e * e;
        sse += s * e;
        sa += s.abs();
    }
    let slope = sse / see;
    let cov = sse / n - ss / n * (se / n);
    let r = cov / ((sss / n - (ss / n).powi(2)).sqrt() * (see / n - (se / n).powi(2)).sqrt());
    (slope, r, sa / n)
}

/// (record score, net eval) in cp, both White POV, for each record.
fn score_eval_pairs(p: &[f32], sh: Shape, kl: &KingLayout, recs: &[Rec], threads: usize) -> Vec<(f64, f64)> {
    if recs.is_empty() {
        return Vec::new();
    }
    let chunk = recs.len().div_ceil(threads.max(1));
    std::thread::scope(|sc| {
        let hs: Vec<_> = recs
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    let mut acc = vec![0.0f32; 2 * sh.h];
                    part.iter()
                        .map(|r| {
                            let s = decode(r, 0.0, sh.buckets, false, kl);
                            let o = (forward(p, sh, &s, &mut acc) * SCALE) as f64; // side to move POV
                            let e = if r[27] == 1 { -o } else { o };
                            (i16::from_le_bytes([r[24], r[25]]) as f64, e)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// `scalecheck`: is the data's score scale the same as the net's? Prints
/// `slope` = least-squares slope of record score on net eval (cp, White POV).
/// ~1.0 = net-scale data; ~0.5 = data written at the engine's halved scale.
/// Records with |score| >= --max-score (mates, default 10000) are skipped.
fn cmd_scalecheck(args: &[String]) {
    let (mut net, mut data, mut buckets, mut kl) = (None, None, 1usize, NO_BUCKETS);
    let (mut n, mut threads, mut max_score) = (200_000usize, 2usize, 10_000.0f64);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("{a} needs a value")).clone();
        match a.as_str() {
            "--net" => net = Some(PathBuf::from(v())),
            "--data" => data = Some(PathBuf::from(v())),
            "--buckets" => buckets = v().parse().unwrap(),
            "--king-buckets" => kl = KingLayout::by_name(&v()),
            "--n" => n = v().parse().unwrap(),
            "--threads" => threads = v().parse().unwrap(),
            "--max-score" => max_score = v().parse().unwrap(),
            other => panic!("unknown option {other}"),
        }
    }
    let (p, sh) = load_net(&net.expect("--net"), buckets, kl.kb());
    let data = data.expect("--data");
    let mut bytes = Vec::new();
    File::open(&data)
        .unwrap_or_else(|e| panic!("{}: {e}", data.display()))
        .take((n * REC) as u64)
        .read_to_end(&mut bytes)
        .unwrap();
    let recs: Vec<Rec> = bytes.chunks_exact(REC).map(|c| c.try_into().unwrap()).collect();
    let all = score_eval_pairs(&p, sh, kl, &recs, threads);
    let pairs: Vec<_> = all.iter().copied().filter(|(s, _)| s.abs() < max_score).collect();
    let (slope, r, mean_abs) = scale_stats(&pairs);
    let mean_eval = pairs.iter().map(|x| x.1.abs()).sum::<f64>() / pairs.len() as f64;
    println!(
        "scalecheck {} positions {} (skipped {} with |score| >= {max_score}) slope {slope:.4} r {r:.4} mean|score| {mean_abs:.1}cp mean|eval| {mean_eval:.1}cp",
        data.display(),
        pairs.len(),
        all.len() - pairs.len(),
    );
}

// ---------------------------------------------------------------- rescale

/// Largest |output weight| the engine's fast i32 output kernel accepts
/// (`Network::small_out` in `src/nnue.rs`); larger weights fall back to the
/// slow i64 path, so `rescale` clips to it.
const OUT_W_FAST: i32 = 127;

/// How `rescale --match` turns (reference, candidate) evals into a factor.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Norm {
    /// `1 / k` with `k` the least-squares slope through 0 of candidate on
    /// reference eval: `k = sum(c*r) / sum(r*r)`. The default; see the README.
    Slope,
    /// `mean|r| / mean|c|`.
    MeanAbs,
}

/// Scale factor that brings the candidate to the reference's scale, from
/// `(reference, candidate)` eval pairs (cp, any common POV). Pairs with
/// |reference| > `max_eval` are left out (selecting on the regressor keeps
/// the slope unbiased); `max_eval <= 0` keeps all.
fn match_factor(pairs: &[(f64, f64)], max_eval: f64, norm: Norm) -> f64 {
    let keep = |r: f64| max_eval <= 0.0 || r.abs() <= max_eval;
    let (mut cr, mut rr, mut ar, mut ac) = (0.0, 0.0, 0.0, 0.0);
    for &(r, c) in pairs.iter().filter(|p| keep(p.0)) {
        cr += c * r;
        rr += r * r;
        ar += r.abs();
        ac += c.abs();
    }
    let f = match norm {
        Norm::Slope => rr / cr,
        Norm::MeanAbs => ar / ac,
    };
    assert!(f.is_finite() && f > 0.0, "cannot match scales (factor {f}): nets uncorrelated or val set empty?");
    f
}

/// Multiplies the output weights and biases of the i16 net in `bytes` by
/// `f` in place. Weights clip to +-`OUT_W_FAST`, biases to i16. Returns the
/// number of clipped (weights, biases).
fn rescale_output(bytes: &mut [u8], sh: Shape, f: f64) -> (usize, usize) {
    let (mut cw, mut cb) = (0, 0);
    for i in sh.out_w()..sh.len() {
        let q = i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) as f64;
        let lim = if i < sh.out_b() { OUT_W_FAST as f64 } else { i16::MAX as f64 };
        let x = (q * f).round();
        let y = x.clamp(-lim, lim);
        if y != x {
            if i < sh.out_b() {
                cw += 1;
            } else {
                cb += 1;
            }
        }
        bytes[2 * i..2 * i + 2].copy_from_slice(&(y as i16).to_le_bytes());
    }
    (cw, cb)
}

/// Sensitivities of the least-squares fit "new eval vs this net's eval" to
/// each i16 output parameter. With `c` the net's eval (cp, side to move) and
/// `c'` the eval after replacing the output layer by `q'`,
/// `sum(c' * c) = sum_j gain[j] * q'[j]` exactly (the output is linear in
/// its parameters), so `achieved factor = sum(gain*q') / sum(gain*q)` over
/// the positions used. Index `j` = parameter `out_w() + j`. Positions with
/// |c| > `max_eval` (when > 0) are left out.
fn output_gains(p: &[f32], sh: Shape, kl: &KingLayout, recs: &[Rec], max_eval: f64, threads: usize) -> Vec<f64> {
    let n_out = sh.len() - sh.out_w();
    if recs.is_empty() {
        return vec![0.0; n_out];
    }
    let chunk = recs.len().div_ceil(threads.max(1));
    let parts: Vec<Vec<f64>> = std::thread::scope(|sc| {
        let hs: Vec<_> = recs
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    let h2 = 2 * sh.h;
                    let mut g = vec![0.0f64; n_out];
                    let mut acc = vec![0.0f32; h2];
                    for r in part {
                        let s = decode(r, 0.0, sh.buckets, false, kl);
                        let c = (forward(p, sh, &s, &mut acc) * SCALE) as f64;
                        if max_eval > 0.0 && c.abs() > max_eval {
                            continue;
                        }
                        let gw = &mut g[s.bucket * h2..(s.bucket + 1) * h2];
                        for (x, a) in gw.iter_mut().zip(&acc) {
                            let a = a.clamp(0.0, 1.0) as f64;
                            *x += a * a * c;
                        }
                        g[sh.buckets * h2 + s.bucket] += c;
                    }
                    g
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut g = vec![0.0f64; n_out];
    for part in parts {
        for (x, y) in g.iter_mut().zip(part) {
            *x += y;
        }
    }
    let nw = sh.out_b() - sh.out_w();
    for (j, x) in g.iter_mut().enumerate() {
        *x *= SCALE as f64 / if j < nw { QB as f64 } else { (QA * QB) as f64 };
    }
    g
}

/// Output layer of `bytes` as i16.
fn out_params(bytes: &[u8], sh: Shape) -> Vec<i16> {
    bytes[2 * sh.out_w()..2 * sh.len()].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Rounds `f * q` (the original output layer `q`) to integers so that the
/// net's evals come out at exactly `f` times the original in the
/// least-squares sense of `output_gains`, instead of what plain rounding
/// gives. Plain rounding is biased because output weights are small (g3f:
/// mean |w| 11.5, mode 6; 6 * 0.92 rounds back to 6), so x0.92 by plain
/// rounding gave g3f an effective x0.935. Starting from plain rounding, the
/// cheapest moves (the parameters whose `f*q` lies closest to the rounding
/// midpoint, per unit of gain) are rounded the other way until the
/// achieved factor is as close to `f` as it gets; no parameter ends up more
/// than one unit from `f*q`. Returns (new layer, plain-rounding factor,
/// achieved factor, parameters re-rounded).
fn round_to_factor(q: &[i16], gains: &[f64], f: f64, n_weights: usize) -> (Vec<i16>, f64, f64, usize) {
    let lim = |j: usize| if j < n_weights { OUT_W_FAST as f64 } else { i16::MAX as f64 };
    let t: Vec<f64> = q.iter().map(|&x| x as f64 * f).collect();
    let mut out: Vec<f64> = t.iter().enumerate().map(|(j, x)| x.round().clamp(-lim(j), lim(j))).collect();
    let dot = |v: &[f64]| v.iter().zip(gains).map(|(a, b)| a * b).sum::<f64>();
    let base = q.iter().zip(gains).map(|(&a, b)| a as f64 * b).sum::<f64>();
    let plain = dot(&out) / base;
    let mut d = f * base - dot(&out);
    // (cost per unit gain, index, step)
    let mut moves: Vec<(f64, usize, f64)> = (0..q.len())
        .filter_map(|j| {
            let e = t[j] - out[j];
            let step = if e > 0.0 { 1.0 } else { -1.0 };
            let ok = e != 0.0 && (step * gains[j]).signum() == d.signum() && (out[j] + step).abs() <= lim(j);
            ok.then(|| ((1.0 - 2.0 * e.abs()) / gains[j].abs(), j, step))
        })
        .collect();
    moves.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut flips = 0;
    for (_, j, step) in moves {
        let nd = d - step * gains[j];
        if nd.abs() < d.abs() {
            out[j] += step;
            d = nd;
            flips += 1;
        }
    }
    let achieved = dot(&out) / base;
    (out.into_iter().map(|x| x as i16).collect(), plain, achieved, flips)
}

/// `rescale`: multiply a net's output layer (weights + bias) by `--factor`,
/// or by the factor that matches `--match ref.bin`'s scale on `--val`.
/// Scaling the output layer scales every eval by the same factor (SCReLU
/// feeds the output linearly), up to i16 rounding.
fn cmd_rescale(args: &[String]) {
    let (mut net, mut out, mut buckets, mut kl) = (None::<PathBuf>, None::<PathBuf>, 1usize, NO_BUCKETS);
    let (mut factor, mut reference, mut val) = (None::<f64>, None::<PathBuf>, None::<PathBuf>);
    let (mut ref_buckets, mut ref_kl) = (None::<usize>, None::<&KingLayout>);
    let (mut norm, mut max_eval, mut n, mut threads, mut lambda) = (Norm::Slope, 1500.0f64, 0usize, 2usize, 0.5f32);
    let mut plain = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("{a} needs a value")).clone();
        match a.as_str() {
            "--net" => net = Some(PathBuf::from(v())),
            "--out" => out = Some(PathBuf::from(v())),
            "--buckets" => buckets = v().parse().unwrap(),
            "--king-buckets" => kl = KingLayout::by_name(&v()),
            "--factor" => factor = Some(v().parse().unwrap()),
            "--match" => reference = Some(PathBuf::from(v())),
            "--val" => val = Some(PathBuf::from(v())),
            "--ref-buckets" => ref_buckets = Some(v().parse().unwrap()),
            "--ref-king-buckets" => ref_kl = Some(KingLayout::by_name(&v())),
            "--norm" => {
                norm = match v().as_str() {
                    "slope" => Norm::Slope,
                    "mean-abs" => Norm::MeanAbs,
                    o => panic!("--norm {o}: one of slope, mean-abs"),
                }
            }
            "--max-eval" => max_eval = v().parse().unwrap(),
            "--n" => n = v().parse().unwrap(),
            "--threads" => threads = v().parse().unwrap(),
            "--lambda" => lambda = v().parse().unwrap(),
            "--plain" => plain = true,
            other => panic!("unknown option {other}"),
        }
    }
    let net = net.expect("--net");
    let out = out.expect("--out");
    assert!(factor.is_some() != reference.is_some(), "pass exactly one of --factor F, --match ref.bin");
    let mut bytes = std::fs::read(&net).unwrap_or_else(|e| panic!("{}: {e}", net.display()));
    let sh = Shape::from_file_bytes(bytes.len(), buckets, kl.kb())
        .unwrap_or_else(|| panic!("{}: {} bytes is no ({}*768->H)x2->{buckets} net", net.display(), bytes.len(), kl.kb()));
    let val = val.map(|path| {
        let mut recs = read_recs(&path);
        if n > 0 {
            recs.truncate(n);
        }
        (path, recs)
    });
    let p = dequantize(&bytes, sh);
    let mut ref_eval = None;
    let f = match (factor, &reference) {
        (Some(f), _) => f,
        (None, Some(rp)) => {
            let (path, recs) = val.as_ref().expect("--match needs --val");
            let rkl = ref_kl.unwrap_or(kl);
            let (rp_, rsh) = load_net(rp, ref_buckets.unwrap_or(buckets), rkl.kb());
            let r = score_eval_pairs(&rp_, rsh, rkl, recs, threads);
            let c = score_eval_pairs(&p, sh, kl, recs, threads);
            let pairs: Vec<(f64, f64)> = r.iter().zip(&c).map(|(r, c)| (r.1, c.1)).collect();
            let (fs, fm) = (match_factor(&pairs, max_eval, Norm::Slope), match_factor(&pairs, max_eval, Norm::MeanAbs));
            let (_, corr, _) = scale_stats(&pairs);
            let kept = pairs.iter().filter(|p| max_eval <= 0.0 || p.0.abs() <= max_eval).count();
            println!(
                "match {} on {} ({kept} of {} positions with |ref| <= {max_eval}cp): r {corr:.4}, slope factor {fs:.4}, mean-abs factor {fm:.4}",
                rp.display(),
                path.display(),
                pairs.len(),
            );
            ref_eval = Some(r);
            if norm == Norm::Slope {
                fs
            } else {
                fm
            }
        }
        _ => unreachable!(),
    };
    let q0 = out_params(&bytes, sh);
    let nw = sh.out_b() - sh.out_w();
    let (cw, cb) = rescale_output(&mut bytes, sh, f);
    match &val {
        Some((path, recs)) if !plain => {
            let gains = output_gains(&p, sh, kl, recs, max_eval, threads);
            let (q1, fp, fa, flips) = round_to_factor(&q0, &gains, f, nw);
            for (j, x) in q1.iter().enumerate() {
                let i = sh.out_w() + j;
                bytes[2 * i..2 * i + 2].copy_from_slice(&x.to_le_bytes());
            }
            println!(
                "rounding on {}: plain rounding gives x{fp:.4}, {flips} parameters re-rounded -> x{fa:.4}",
                path.display()
            );
        }
        _ => println!("plain rounding (no --val, or --plain): the achieved factor can be off by ~1-2%"),
    }
    let maxw = out_params(&bytes, sh)[..nw].iter().map(|w| (*w as i32).abs()).max().unwrap_or(0);
    File::create(&out).unwrap().write_all(&bytes).unwrap();
    println!(
        "wrote {} factor {f:.4}: clipped {cw} of {nw} output weights to +-{OUT_W_FAST} (max |w| now {maxw}), {cb} biases",
        out.display(),
    );
    if cw > 0 {
        println!("warning: {cw} output weights clipped; the achieved scale is below {f:.4}");
    }
    if let Some((path, recs)) = &val {
        let q = dequantize(&bytes, sh);
        let (l0, e0) = validate(&p, sh, kl, recs, lambda, threads);
        let (l1, e1) = validate(&q, sh, kl, recs, lambda, threads);
        print!("{}: loss {l0:.6} -> {l1:.6}, mean|eval| {e0:.1} -> {e1:.1}cp", path.display());
        if let Some(r) = &ref_eval {
            let c = score_eval_pairs(&q, sh, kl, recs, threads);
            let pairs: Vec<(f64, f64)> = r.iter().zip(&c).map(|(r, c)| (r.1, c.1)).collect();
            let er = r.iter().map(|x| x.1.abs()).sum::<f64>() / r.len() as f64;
            // Residual factor after rescaling: 1.0 = scales matched.
            let rs = match_factor(&pairs, max_eval, Norm::Slope);
            let rm = match_factor(&pairs, max_eval, Norm::MeanAbs);
            print!(" (reference {er:.1}cp); residual factor slope {rs:.4} mean-abs {rm:.4}");
        }
        println!();
    }
}

fn cmd_train(args: &[String]) {
    let mut data = Vec::new();
    let mut val_path = None;
    let mut out = PathBuf::from("net.bin");
    let mut sh = Shape { h: 256, buckets: 1, kb: 1 };
    let mut kl = NO_BUCKETS;
    let (mut epochs, mut epoch_size) = (10u64, 0u64);
    let (mut batch, mut lr, mut lr_end, mut wd, mut lambda) = (16384usize, 1e-3f32, 1e-5f32, 0.0f32, 0.5f32);
    let (mut threads, mut seed, mut mirror) = (threads_default(), 1u64, false);
    let (mut init, mut resume, mut ckpt) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
    let (mut exact, mut sub_batch, mut max_batches) = (false, 0usize, u64::MAX);
    let (mut factorise, mut wd_active, mut keep_best) = (false, None::<bool>, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("missing value for {a}")).clone();
        match a.as_str() {
            "--data" => data.push(PathBuf::from(v())),
            "--val" => val_path = Some(PathBuf::from(v())),
            "--out" => out = PathBuf::from(v()),
            "--hidden" => sh.h = v().parse().unwrap(),
            "--buckets" => sh.buckets = v().parse().unwrap(),
            "--king-buckets" => kl = KingLayout::by_name(&v()),
            "--epochs" => epochs = v().parse().unwrap(),
            "--epoch-size" => epoch_size = v().parse().unwrap(),
            "--batch" => batch = v().parse().unwrap(),
            "--lr" => lr = v().parse().unwrap(),
            "--lr-end" => lr_end = v().parse().unwrap(),
            "--wd" => wd = v().parse().unwrap(),
            "--lambda" => lambda = v().parse().unwrap(),
            "--threads" => threads = v().parse().unwrap(),
            "--seed" => seed = v().parse().unwrap(),
            "--mirror" => mirror = true,
            "--init" => init = Some(PathBuf::from(v())),
            "--resume" => resume = Some(PathBuf::from(v())),
            "--checkpoint" => ckpt = Some(PathBuf::from(v())),
            "--exact" => exact = true,
            "--sub-batch" => sub_batch = v().parse().unwrap(),
            "--max-batches" => max_batches = v().parse().unwrap(),
            "--factorise" | "--factorize" => factorise = true,
            "--wd-active-only" => wd_active = Some(true),
            "--no-wd-active-only" => wd_active = Some(false),
            "--keep-best" => keep_best = true,
            other => panic!("unknown option {other}"),
        }
    }
    assert!(!data.is_empty(), "--data required");
    sh.kb = kl.kb();
    assert!(!factorise || sh.kb > 1, "--factorise needs a king-bucket layout with KB > 1");
    let fac = Fac::new(sh, factorise);
    let wd_active = wd_active.unwrap_or(sh.kb > 1);
    if epoch_size == 0 {
        epoch_size = data.iter().map(|p| std::fs::metadata(p).unwrap().len() / REC as u64).sum();
    }
    let total = epochs * epoch_size;
    let ckpt = ckpt.unwrap_or_else(|| out.with_extension("ckpt"));

    let mut rng = Rng::new(seed);
    if let (None, Some(i)) = (&resume, &init) {
        warn_mirrored_init(i, sh, kl);
    }
    let (p, mut opt, mut seen) = match (&resume, &init) {
        (Some(r), _) => load_ckpt(r, sh, kl, fac),
        (None, Some(i)) => {
            let p = if fac.on() {
                load_init_factor(i, sh)
            } else {
                let (p, s) = load_init(i, sh.buckets, sh.kb);
                assert_eq!(s, sh, "init net shape differs from --hidden/--buckets/--king-buckets");
                p
            };
            let n = p.len();
            (p.into_iter().map(|x| x.clamp(-CLIP, CLIP)).collect(), Adam { m: AVec::zeros(n), v: AVec::zeros(n), step: 0 }, 0)
        }
        (None, None) => {
            let p = if fac.on() { init_factor(sh, &mut rng) } else { init_params(sh, &mut rng) };
            let n = p.len();
            (p, Adam { m: AVec::zeros(n), v: AVec::zeros(n), step: 0 }, 0)
        }
    };
    let mut p = AVec::from(p);
    // Effective (merged) net params; only a separate buffer with the factoriser.
    let mut merged = if fac.on() { vec![0.0f32; sh.len()] } else { Vec::new() };
    if fac.on() {
        merge_factor(&p, sh, fac, &mut merged);
    }
    let val = val_path.as_deref().map(read_recs).unwrap_or_default();
    assert!(!keep_best || !val.is_empty(), "--keep-best needs --val");
    let mut kb_note = if kl.id() != 0 { format!(" king-buckets={} (KB={})", kl.name, sh.kb) } else { String::new() };
    if fac.on() {
        kb_note += " factorised";
    }
    if wd_active {
        kb_note += " wd-active-only";
    }
    println!(
        "train H={} buckets={}{kb_note} params={} epochs={epochs} x {epoch_size} batch={batch} lr {lr}->{lr_end} wd={wd} lambda={lambda} mirror={mirror} threads={threads} val={}",
        sh.h,
        sh.buckets,
        sh.len(),
        val.len()
    );
    let pm0 = if fac.on() { &merged[..] } else { &p[..] };
    let (vl, ve) = validate(pm0, sh, kl, &val, lambda, threads);
    let q0 = quantize(pm0, sh);
    let (ql0, _) = validate(&dequantize(&q0, sh), sh, kl, &val, lambda, threads);
    println!("start val loss {vl:.6} (quantized {ql0:.6}) mean|eval| {ve:.1}cp");
    // --keep-best: (quantized val loss, epoch, net bytes) of the best net so
    // far; the start net (epoch 0: the --init / --resume net) competes too.
    let mut best = keep_best.then(|| (ql0, seen / epoch_size, q0));

    let mut bt = batch::BatchTrainer::new(sh, sub_batch, exact, fac.fl, wd_active);
    println!(
        "kernels {} sub-batch {}{}",
        bt.isa.name(),
        bt.sub_batch(),
        if exact { " exact (v2 --threads 1 summation order)" } else { "" }
    );
    // Loader thread: feed + shuffle + the per-batch mirror flags (the same
    // RNG streams as before: `seed_b` from `rng`, one RNG per 256-record
    // chunk), a few batches ahead of the trainer.
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<Rec>, Vec<bool>)>(4);
    let loader = {
        let mut feed = Feed::new(data, seed ^ 0xF00D ^ seen);
        let mut rng = Rng(rng.0);
        std::thread::spawn(move || loop {
            let mut recs = Vec::with_capacity(batch);
            feed.batch(&mut recs, batch);
            let seed_b = rng.next_u64();
            let mut flags = vec![false; recs.len()];
            if mirror {
                for (c, fl) in flags.chunks_mut(256).enumerate() {
                    let mut mr = Rng::new(seed_b ^ c as u64);
                    for x in fl {
                        *x = mr.next_u64() & 1 == 1;
                    }
                }
            }
            if tx.send((recs, flags)).is_err() {
                return;
            }
        })
    };
    let started = Instant::now();
    let start_seen = seen;
    let mut epoch_loss = 0.0f64;
    let mut epoch_n = 0u64;
    let mut ep = seen / epoch_size;
    let mut batches = 0u64;
    batch::Pool::with(threads, |pool| {
        while seen < total && batches < max_batches {
            let t = seen as f64 / total as f64;
            let cur_lr = lr_end + 0.5 * (lr - lr_end) * (1.0 + (std::f64::consts::PI * t).cos() as f32);
            let (recs, flags) = rx.recv().unwrap();
            let loss = bt.batch(pool, &p, &recs, &flags, lambda, kl);
            bt.adam(pool, &mut p, &mut opt, cur_lr, wd, 1.0 / recs.len() as f32);
            seen += recs.len() as u64;
            batches += 1;
            epoch_loss += loss;
            epoch_n += recs.len() as u64;

            if seen / epoch_size != (seen - recs.len() as u64) / epoch_size || seen >= total || batches == max_batches {
                ep += 1;
                let secs = started.elapsed().as_secs_f64();
                if fac.on() {
                    merge_factor(&p, sh, fac, &mut merged);
                }
                let pm = if fac.on() { &merged[..] } else { &p[..] };
                let (vl, ve) = validate(pm, sh, kl, &val, lambda, threads);
                let q = quantize(pm, sh);
                let (ql, _) = validate(&dequantize(&q, sh), sh, kl, &val, lambda, threads);
                let path = out.with_extension(format!("e{ep}.bin"));
                File::create(&path).unwrap().write_all(&q).unwrap();
                if let Some(b) = best.as_mut().filter(|b| ql < b.0) {
                    *b = (ql, ep, q.clone());
                }
                save_ckpt(&ckpt, sh, kl, fac, &p, &opt, seen);
                println!(
                    "epoch {ep} lr {cur_lr:.2e} train {:.6} val {vl:.6} (quantized {ql:.6}) mean|eval| {ve:.1}cp {:.0} pos/s -> {}",
                    epoch_loss / epoch_n as f64,
                    (seen - start_seen) as f64 / secs,
                    path.display()
                );
                epoch_loss = 0.0;
                epoch_n = 0;
            }
        }
    });
    drop(rx);
    loader.join().unwrap();
    if fac.on() {
        merge_factor(&p, sh, fac, &mut merged);
    }
    let mut q = quantize(if fac.on() { &merged } else { &p }, sh);
    if let Some((bl, be, bq)) = best {
        println!("keep-best: epoch {be} (quantized val {bl:.6}) is the output");
        q = bq;
    }
    File::create(&out).unwrap().write_all(&q).unwrap();
    println!("wrote {} ({} bytes, H={} buckets={}{kb_note})", out.display(), q.len(), sh.h, sh.buckets);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: nnue-trainer <convert|shuffle|train|eval|scalecheck|rescale> ... (see the module docs)";
    let (cmd, rest) = args.split_first().unwrap_or_else(|| panic!("{usage}"));
    match cmd.as_str() {
        "convert" => cmd_convert(rest),
        "shuffle" => cmd_shuffle(rest),
        "train" => cmd_train(rest),
        "eval" => cmd_eval(rest),
        "scalecheck" => cmd_scalecheck(rest),
        "rescale" => cmd_rescale(rest),
        _ => panic!("{usage}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_match_runtime_layout() {
        // Startpos, White to move: perspective 0 = White.
        let r = encode("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 10 | 0.5").unwrap();
        let s = decode(&r, 0.5, 1, false, NO_BUCKETS);
        assert_eq!(s.n, 32);
        for side in 0..2 {
            let f = &s.f[side][..32];
            assert!(f.contains(&324), "own king e1/e8 -> 5*64+4");
            assert!(f.contains(&764), "enemy king -> 384+5*64+60");
        }
        // a2 pawn: White persp 8, Black persp 384 + (8^56).
        assert!(s.f[0].contains(&8));
        assert!(s.f[1].contains(&(384 + 48)));
    }

    #[test]
    fn black_to_move_flips_targets_and_perspectives() {
        let r = encode("4k3/8/8/8/8/8/8/4K2R b - - 0 1 | 400 | 1.0").unwrap();
        let s = decode(&r, 1.0, 1, false, NO_BUCKETS);
        assert_eq!(s.target, 0.0, "White won, Black to move");
        // Perspective 0 is Black: its own king on e8 -> 60^56 = 4 -> 324.
        assert!(s.f[0][..s.n].contains(&324));
        // White rook h1 seen from Black: enemy 384 + 3*64 + (7^56=63).
        assert!(s.f[0][..s.n].contains(&(384 + 192 + 63)));
        let s = decode(&r, 0.0, 1, false, NO_BUCKETS);
        assert!((s.target - sigmoid(-1.0)).abs() < 1e-6);
    }

    /// Same cases as `bucketed_feature_indices` in `src/nnue.rs`.
    #[test]
    fn king_bucket_features() {
        // kb8m; White king g1, Black king e8; pawns a2 (White), h7 (Black).
        let r = encode("4k3/7p/8/8/8/8/P7/6K1 w - - 0 1 | 0 | 0.5").unwrap();
        let kl = KingLayout::by_name("kb8m");
        assert_eq!((kl.kb(), kl.mirror), (8, true));
        let s = decode(&r, 0.5, 1, false, kl);
        // Perspective 0 = White: king g1 -> mirrored b1, bucket 1.
        assert!(s.f[0][..s.n].contains(&(768 + 15)), "a2 pawn -> h2 in bucket 1");
        // Perspective 1 = Black: king e8 -> oriented e1 -> mirrored d1, bucket 3.
        assert!(s.f[1][..s.n].contains(&(3 * 768 + 8)), "own h7 pawn -> a2");
        assert!(s.f[1][..s.n].contains(&(3 * 768 + 384 + 55)), "enemy a2 pawn -> a7 -> h7");
        // Own king: White g1 -> b1 (1) in bucket 1; Black e8 -> d1 (3) in bucket 3.
        assert!(s.f[0][..s.n].contains(&(768 + 5 * 64 + 1)));
        assert!(s.f[1][..s.n].contains(&(3 * 768 + 5 * 64 + 3)));
        // Every feature of a perspective sits in that perspective's block.
        for (side, b) in [(0, 1u16), (1, 3)] {
            assert!(s.f[side][..s.n].iter().all(|&f| f / 768 == b));
        }
        // kb4 (no mirror): Black king d7 -> oriented d2 -> bucket 2, no flip.
        let r = encode("8/3k4/8/8/8/8/8/4K3 b - - 0 1 | 0 | 0.5").unwrap();
        let s = decode(&r, 0.5, 1, false, KingLayout::by_name("kb4"));
        assert!(s.f[0][..s.n].contains(&(2 * 768 + 5 * 64 + 11)), "Black (stm) king d7 -> d2");
        assert!(s.f[1][..s.n].contains(&(768 + 5 * 64 + 4)), "White king e1 -> bucket 1");
    }

    /// `none` is bit-for-bit the original encoding; m1 only flips files.
    #[test]
    fn no_buckets_is_legacy_and_m1_mirrors() {
        let fens = [
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 | 35 | 1.0",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1 | -20 | 0.5",
        ];
        for fen in fens {
            let r = encode(fen).unwrap();
            for aug in [false, true] {
                let a = decode(&r, 0.5, 1, aug, NO_BUCKETS);
                // Legacy formula, recomputed here.
                let mut occ = u64::from_le_bytes(r[..8].try_into().unwrap());
                let stm_black = r[27] == 1;
                let mut n = 0;
                while occ != 0 {
                    let sq = occ.trailing_zeros() as u16 ^ if aug { 7 } else { 0 };
                    occ &= occ - 1;
                    let nib = r[8 + n / 2] >> (4 * (n & 1)) & 0xF;
                    for (i, pb) in [stm_black, !stm_black].into_iter().enumerate() {
                        let side = if (nib >> 3 == 1) == pb { 0 } else { 384 };
                        let rel = if pb { sq ^ 56 } else { sq };
                        assert_eq!(a.f[i][n], side + (nib & 7) as u16 * 64 + rel);
                    }
                    n += 1;
                }
                // m1: a perspective whose king is on files e-h sees every file flipped.
                let m = decode(&r, 0.5, 1, aug, KingLayout::by_name("m1"));
                for i in 0..2 {
                    let king = a.f[i][..a.n].iter().find(|&&f| (320..384).contains(&f)).unwrap();
                    let flip = if king % 8 >= 4 { 7 } else { 0 };
                    for k in 0..a.n {
                        assert_eq!(m.f[i][k], a.f[i][k] ^ flip);
                    }
                }
            }
        }
    }

    #[test]
    fn warm_start_copies_rows_into_every_bucket() {
        let s1 = Shape { h: 8, buckets: 2, kb: 1 };
        let mut rng = Rng::new(5);
        let p1 = init_params(s1, &mut rng);
        let p = expand_buckets(&p1, s1, 8);
        let s8 = Shape { kb: 8, ..s1 };
        assert_eq!(p.len(), s8.len());
        for b in 0..8 {
            assert_eq!(&p[b * s1.ft_b()..(b + 1) * s1.ft_b()], &p1[..s1.ft_b()]);
        }
        assert_eq!(&p[s8.ft_b()..], &p1[s1.ft_b()..]);
        // A bucketed net built that way evaluates exactly like the original.
        let r = encode("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 | 35 | 1.0").unwrap();
        let (mut a1, mut a8) = (vec![0.0; 2 * s1.h], vec![0.0; 2 * s1.h]);
        let o1 = forward(&p1, s1, &decode(&r, 0.5, 2, false, NO_BUCKETS), &mut a1);
        let o8 = forward(&p, s8, &decode(&r, 0.5, 2, false, KingLayout::by_name("kb4")), &mut a8);
        assert_eq!(o1, o8);
        // File sizes: KB=1 unchanged, KB=8 has 8x the FT rows.
        assert_eq!(s1.file_bytes(), (768 * 8 + 8 + 2 * 17) * 2);
        assert_eq!(Shape::from_file_bytes(s8.file_bytes(), 2, 8), Some(s8));
        assert_eq!(Shape::from_file_bytes(s8.file_bytes(), 2, 1), None);
    }

    /// Factoriser: merged rows = W + F, F's gradient = the sum of every
    /// bucket's (checked against finite differences on the training vector),
    /// from-scratch and one-bucket warm starts reproduce the one-bucket net,
    /// and active-only weight decay leaves rows the batch never touched alone.
    #[test]
    fn factoriser_merge_gradient_and_init() {
        let kl = KingLayout::by_name("kb4");
        let sh = Shape { h: 8, buckets: 2, kb: kl.kb() };
        let fac = Fac::new(sh, true);
        assert_eq!(fac.fl, 768 * 8);
        let mut rng = Rng::new(11);
        // Random W and F, a positive bias so the SCReLU is live.
        let mut p: Vec<f32> = (0..sh.len() + fac.fl).map(|_| rng.uniform(0.1)).collect();
        for x in &mut p[sh.ft_b()..sh.out_w()] {
            *x = 0.3;
        }
        let mut m = vec![0.0f32; sh.len()];
        merge_factor(&p, sh, fac, &mut m);
        assert_eq!(m[3 * fac.fl + 17], (p[3 * fac.fl + 17] + p[sh.len() + 17]).clamp(-CLIP, CLIP));
        assert_eq!(&m[sh.ft_b()..], &p[sh.ft_b()..sh.len()]);

        let fens = [
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 | 35 | 1.0",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1 | -20 | 0.5",
        ];
        let samples: Vec<Sample> = fens.iter().map(|f| decode(&encode(f).unwrap(), 0.5, 2, false, kl)).collect();
        let mut g = vec![0.0f32; sh.len()];
        let (mut acc, mut dacc) = (vec![0.0; 2 * sh.h], vec![0.0; 2 * sh.h]);
        for s in &samples {
            train_sample(&m, sh, s, &mut g, &mut acc, &mut dacc);
        }
        let loss = |q: &[f32]| {
            let mut mm = vec![0.0f32; sh.len()];
            merge_factor(q, sh, fac, &mut mm);
            let mut acc = vec![0.0; 2 * sh.h];
            samples.iter().map(|s| (sigmoid(forward(&mm, sh, s, &mut acc)) - s.target).powi(2) as f64).sum::<f64>()
        };
        // An F element used by both samples' features (own king row of perspective 0).
        let fk = (samples[0].f[0][0] as usize % 768) * sh.h + 2;
        let gf: f32 = (0..sh.kb).map(|b| g[b * fac.fl + fk]).sum();
        let wi = samples[1].f[1][2] as usize * sh.h + 5;
        for (i, analytic) in [(sh.len() + fk, gf), (wi, g[wi])] {
            let eps = 1e-3;
            let (mut a, mut b) = (p.clone(), p.clone());
            a[i] += eps;
            b[i] -= eps;
            let num = (loss(&a) - loss(&b)) / (2.0 * eps as f64);
            assert!((num - analytic as f64).abs() < 1e-4 + 1e-2 * num.abs(), "param {i}: numeric {num} vs {analytic}");
        }

        // One AdamW step, wd active-only: an untouched bucket row keeps its value exactly.
        let mut active = vec![0u8; sh.kb * 768];
        for s in &samples {
            for side in 0..2 {
                for &f in &s.f[side][..s.n] {
                    active[f as usize] = 1;
                }
            }
        }
        let idle = (0..sh.kb * 768).find(|&r| active[r] == 0).unwrap();
        let n = p.len();
        let mut opt = Adam { m: AVec::zeros(n), v: AVec::zeros(n), step: 0 };
        let mut q = p.clone();
        let cx = StepCtx { sh, fac, active: Some(&active) };
        adam_step(&mut q, &mut opt, std::slice::from_ref(&g), 1e-3, 0.1, 0.5, 2, &cx);
        assert_eq!(&q[idle * sh.h..(idle + 1) * sh.h], &p[idle * sh.h..(idle + 1) * sh.h]);
        assert!(q[sh.len() + fk] != p[sh.len() + fk]);
        let mut q2 = p.clone();
        let mut opt = Adam { m: AVec::zeros(n), v: AVec::zeros(n), step: 0 };
        adam_step(&mut q2, &mut opt, std::slice::from_ref(&g), 1e-3, 0.1, 0.5, 2, &StepCtx { sh, fac, active: None });
        assert_eq!(q2[idle * sh.h], p[idle * sh.h] * (1.0 - 1e-3 * 0.1));

        // From scratch: merged net == the KB=1 init with the same seed, in every bucket.
        let s1 = Shape { kb: 1, ..sh };
        let p1 = init_params(s1, &mut Rng::new(5));
        let pf = init_factor(sh, &mut Rng::new(5));
        assert_eq!(pf.len(), sh.len() + fac.fl);
        merge_factor(&pf, sh, fac, &mut m);
        assert_eq!(m, expand_buckets(&p1, s1, sh.kb));

        // Checkpoint round trip keeps F and the factoriser flag.
        let dir = std::env::temp_dir().join(format!("nnue-fac-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.ckpt");
        let opt = Adam { m: pf.clone().into(), v: pf.clone().into(), step: 9 };
        save_ckpt(&path, sh, kl, fac, &pf, &opt, 1234);
        let (p2, o2, seen) = load_ckpt(&path, sh, kl, fac);
        assert_eq!((p2, o2.step, seen), (pf, 9, 1234));

        // `--init` from a one-bucket net file: merged net == that net (dequantized)
        // in every bucket, so the start is exactly the plain net.
        let init = dir.join("plain.bin");
        File::create(&init).unwrap().write_all(&quantize(&p1, s1)).unwrap();
        let q1 = dequantize(&quantize(&p1, s1), s1);
        let pi = load_init_factor(&init, sh);
        merge_factor(&pi, sh, fac, &mut m);
        assert_eq!(m, expand_buckets(&q1, s1, sh.kb));
        assert_eq!(&pi[sh.len()..], &q1[..s1.ft_b()]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn layouts_match_engine_tables() {
        assert_eq!(LAYOUTS[0].name, "none");
        let kb8m = KingLayout::by_name("kb8m");
        assert_eq!(&kb8m.buckets[..16], &[0, 1, 2, 3, 3, 2, 1, 0, 4, 4, 5, 5, 5, 5, 4, 4]);
        for l in &LAYOUTS {
            if l.mirror {
                assert!((0..64).all(|s| l.buckets[s] == l.buckets[s ^ 7]));
            }
        }
        // The engine's tables, parsed from its source, must be identical.
        let src = include_str!("../../../src/nnue.rs");
        for (engine, name) in [("LAYOUT_KB4", "kb4"), ("LAYOUT_KB4M", "kb4m"), ("LAYOUT_KB8M", "kb8m")] {
            let start = src.find(&format!("const {engine}: KingLayout")).unwrap();
            let body = &src[start..start + src[start..].find("};").unwrap()];
            let nums: Vec<u8> = body
                .split(|c: char| !c.is_ascii_digit())
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().unwrap())
                .collect();
            let l = KingLayout::by_name(name);
            // Skip the digits in the constant's name (e.g. "4" of KB4).
            let nums = &nums[nums.len() - if l.mirror { 32 } else { 64 }..];
            let want: Vec<u8> = if l.mirror {
                (0..32).map(|i| l.buckets[(i / 4) * 8 + i % 4]).collect()
            } else {
                l.buckets.to_vec()
            };
            assert_eq!(nums, &want[..], "{name}");
            assert_eq!(body.contains("mirror: true"), l.mirror, "{name}");
        }
    }

    #[test]
    fn quantize_roundtrip_and_shape() {
        let sh = Shape { h: 32, buckets: 8, kb: 1 };
        let mut rng = Rng::new(3);
        let p = init_params(sh, &mut rng);
        let q = quantize(&p, sh);
        assert_eq!(Shape::from_file_bytes(q.len(), 8, 1), Some(sh));
        let d = dequantize(&q, sh);
        for i in 0..p.len() {
            let tol = if i < sh.out_w() { 0.5 / QA } else { 0.5 / QB };
            assert!((p[i] - d[i]).abs() <= tol + 1e-6);
        }
        assert_eq!(Shape::from_file_bytes(394_753 * 2, 1, 1), Some(Shape { h: 512, buckets: 1, kb: 1 }));
    }

    #[test]
    fn scalecheck_slope_detects_halved_scores() {
        let sh = Shape { h: 32, buckets: 1, kb: 1 };
        let mut rng = Rng::new(11);
        let mut p = init_params(sh, &mut rng);
        for x in &mut p[sh.out_w()..] {
            *x *= 8.0; // spread the evals so whole-cp rounding is small
        }
        let p = dequantize(&quantize(&p, sh), sh);
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "4k3/8/8/8/8/8/8/4K2R b - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
            "8/8/4k3/8/2Q5/8/4K3/8 b - - 0 1",
        ];
        // Net evals (White POV), then records scored at k times the net's scale.
        let evals: Vec<f64> = fens
            .iter()
            .map(|f| score_eval_pairs(&p, sh, NO_BUCKETS, &[encode(&format!("{f} | 0 | 0.5")).unwrap()], 1)[0].1)
            .collect();
        assert!(evals.iter().any(|e| e.abs() > 50.0), "test net too flat: {evals:?}");
        for k in [0.5, 1.0, 2.0] {
            let recs: Vec<Rec> = fens
                .iter()
                .zip(&evals)
                .map(|(f, e)| encode(&format!("{f} | {} | 0.5", e * k)).unwrap())
                .collect();
            // Scores are stored as whole cp, so the slope is close, not exact.
            let (slope, r, _) = scale_stats(&score_eval_pairs(&p, sh, NO_BUCKETS, &recs, 2));
            assert!((slope - k).abs() < 0.03 * k, "k {k}: slope {slope} evals {evals:?}");
            assert!(r > 0.99, "k {k}: r {r}");
        }
        let (slope, r, mean) = scale_stats(&[(2.0, 1.0), (-4.0, -2.0), (6.0, 3.0)]);
        assert!((slope - 2.0).abs() < 1e-12 && (r - 1.0).abs() < 1e-12 && (mean - 4.0).abs() < 1e-12);
    }

    const RESCALE_FENS: [&str; 8] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "4k3/8/8/8/8/8/8/4K2R b - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
        "8/8/4k3/8/2Q5/8/4K3/8 b - - 0 1",
    ];

    /// A quantised test net with spread-out evals (`out` scales the output layer).
    fn rescale_test_net(sh: Shape, out: f32) -> Vec<u8> {
        let mut rng = Rng::new(11);
        let mut p = init_params(sh, &mut rng);
        for x in &mut p[sh.out_w()..] {
            *x *= 8.0 * out;
        }
        quantize(&p, sh)
    }

    fn evals(bytes: &[u8], sh: Shape) -> Vec<f64> {
        let recs: Vec<Rec> = RESCALE_FENS.iter().map(|f| encode(&format!("{f} | 0 | 0.5")).unwrap()).collect();
        score_eval_pairs(&dequantize(bytes, sh), sh, NO_BUCKETS, &recs, 1).iter().map(|x| x.1).collect()
    }

    #[test]
    fn rescale_scales_evals_and_leaves_ft_alone() {
        let sh = Shape { h: 32, buckets: 2, kb: 1 };
        let net = rescale_test_net(sh, 1.0);
        let mut out = net.clone();
        let (cw, cb) = rescale_output(&mut out, sh, 0.9);
        assert_eq!((cw, cb), (0, 0));
        assert_eq!(&out[..2 * sh.out_w()], &net[..2 * sh.out_w()], "FT untouched");
        let (e0, e1) = (evals(&net, sh), evals(&out, sh));
        assert!(e0.iter().any(|e| e.abs() > 50.0), "test net too flat: {e0:?}");
        for (a, b) in e0.iter().zip(&e1) {
            // i16 rounding of each output weight: a few cp at most.
            assert!((b - 0.9 * a).abs() < 0.01 * a.abs() + 3.0, "{a} -> {b}");
        }
    }

    #[test]
    fn rescale_match_recovers_the_scale() {
        let sh = Shape { h: 32, buckets: 1, kb: 1 };
        let reference = rescale_test_net(sh, 1.0);
        let wide = rescale_test_net(sh, 1.25);
        let pairs: Vec<(f64, f64)> = evals(&reference, sh).into_iter().zip(evals(&wide, sh)).collect();
        for norm in [Norm::Slope, Norm::MeanAbs] {
            let f = match_factor(&pairs, 0.0, norm);
            assert!((f - 0.8).abs() < 0.01, "{norm:?}: {f}");
        }
        let mut out = wide.clone();
        rescale_output(&mut out, sh, match_factor(&pairs, 0.0, Norm::Slope));
        let pairs: Vec<(f64, f64)> = evals(&reference, sh).into_iter().zip(evals(&out, sh)).collect();
        let r = match_factor(&pairs, 0.0, Norm::Slope);
        assert!((r - 1.0).abs() < 0.02, "residual {r} (i16 rounding of a tiny net)");
    }

    /// The reason `slope` is the default: a candidate that knows more than
    /// the reference (`c = k*r + u`, `u` uncorrelated with `r`) is measured
    /// as `k` by the slope, but as wider than `k` by mean|eval|, so
    /// mean-abs matching would shrink the extra signal too.
    #[test]
    fn slope_ignores_signal_the_reference_lacks() {
        let mut rng = Rng::new(3);
        let k = 1.1;
        let pairs: Vec<(f64, f64)> = (0..20000)
            .map(|i| {
                let r = rng.uniform(600.0) as f64;
                let u = if i % 2 == 0 { 150.0 } else { -150.0 }; // independent of r
                (r, k * r + u)
            })
            .collect();
        let fs = match_factor(&pairs, 0.0, Norm::Slope);
        let fm = match_factor(&pairs, 0.0, Norm::MeanAbs);
        assert!((fs - 1.0 / k).abs() < 0.01, "slope {fs}");
        assert!(fm < 1.0 / k - 0.03, "mean-abs {fm} over-shrinks");
        // Filtering on |reference| (the regressor) keeps the slope unbiased.
        assert!((match_factor(&pairs, 300.0, Norm::Slope) - 1.0 / k).abs() < 0.02);
    }

    /// Small weights defeat plain rounding (6 * 0.92 = 5.52 -> 6); re-rounding
    /// hits the factor with every parameter within one unit of `f*q`.
    #[test]
    fn round_to_factor_fixes_plain_rounding_bias() {
        let q: Vec<i16> = (0..1000).map(|i| [-6, 6, 5, -5][i % 4]).collect();
        let mut rng = Rng::new(9);
        let gains: Vec<f64> = q.iter().map(|&w| w as f64 * (1.0 + rng.uniform(0.5) as f64)).collect();
        let (out, plain, achieved, flips) = round_to_factor(&q, &gains, 0.92, q.len());
        assert!((plain - 0.92).abs() > 0.02, "plain {plain}");
        assert!((achieved - 0.92).abs() < 1e-3, "achieved {achieved}");
        assert!(flips > 0);
        for (a, b) in q.iter().zip(&out) {
            assert!((*a as f64 * 0.92 - *b as f64).abs() < 1.0);
        }
        // Exact on a real net's evals: achieved factor == slope of new evals on old ones.
        let sh = Shape { h: 32, buckets: 2, kb: 1 };
        let net = rescale_test_net(sh, 1.0);
        let p = dequantize(&net, sh);
        let recs: Vec<Rec> = RESCALE_FENS.iter().map(|f| encode(&format!("{f} | 0 | 0.5")).unwrap()).collect();
        let gains = output_gains(&p, sh, NO_BUCKETS, &recs, 0.0, 1);
        let (q1, _, achieved, _) = round_to_factor(&out_params(&net, sh), &gains, 0.8, sh.out_b() - sh.out_w());
        let mut net1 = net.clone();
        for (j, x) in q1.iter().enumerate() {
            let i = sh.out_w() + j;
            net1[2 * i..2 * i + 2].copy_from_slice(&x.to_le_bytes());
        }
        let pairs: Vec<(f64, f64)> = evals(&net, sh).into_iter().zip(evals(&net1, sh)).collect();
        let slope = pairs.iter().map(|(a, b)| a * b).sum::<f64>() / pairs.iter().map(|(a, _)| a * a).sum::<f64>();
        assert!((slope - achieved).abs() < 1e-4, "slope {slope} vs achieved {achieved}");
        assert!((achieved - 0.8).abs() < 0.01, "achieved {achieved}");
    }

    #[test]
    fn rescale_clips_output_weights_to_the_fast_path() {
        let sh = Shape { h: 32, buckets: 1, kb: 1 };
        let mut net = rescale_test_net(sh, 1.0);
        net[2 * sh.out_w()..2 * sh.out_w() + 2].copy_from_slice(&120i16.to_le_bytes());
        net[2 * sh.out_w() + 2..2 * sh.out_w() + 4].copy_from_slice(&(-100i16).to_le_bytes());
        let (cw, _) = rescale_output(&mut net, sh, 1.2);
        assert!(cw >= 1);
        let w: Vec<i16> =
            net[2 * sh.out_w()..2 * sh.out_b()].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!((w[0], w[1]), (127, -120));
        assert!(w.iter().all(|w| w.abs() <= 127));
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let sh = Shape { h: 16, buckets: 2, kb: 1 };
        let mut rng = Rng::new(7);
        let mut p = init_params(sh, &mut rng);
        for x in &mut p[sh.ft_b()..sh.out_w()] {
            *x = 0.3;
        }
        let r = encode("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 | 35 | 1.0").unwrap();
        let s = decode(&r, 0.5, sh.buckets, false, NO_BUCKETS);
        let mut g = vec![0.0f32; sh.len()];
        let (mut acc, mut dacc) = (vec![0.0; 2 * sh.h], vec![0.0; 2 * sh.h]);
        train_sample(&p, sh, &s, &mut g, &mut acc, &mut dacc);
        let loss = |p: &[f32]| {
            let mut acc = vec![0.0; 2 * sh.h];
            let d = sigmoid(forward(p, sh, &s, &mut acc)) - s.target;
            (d * d) as f64
        };
        let f0 = s.f[0][0] as usize * sh.h;
        for &i in &[f0, f0 + 3, sh.ft_b() + 1, sh.out_w() + s.bucket * 2 * sh.h + 5, sh.out_b() + s.bucket] {
            let eps = 1e-3;
            let mut a = p.clone();
            a[i] += eps;
            let mut b = p.clone();
            b[i] -= eps;
            let num = (loss(&a) - loss(&b)) / (2.0 * eps as f64);
            assert!((num - g[i] as f64).abs() < 1e-4 + 1e-2 * num.abs(), "param {i}: numeric {num} vs analytic {}", g[i]);
        }
    }

    /// Every SIMD implementation this CPU can run.
    fn isas() -> Vec<simd::Isa> {
        let mut v = vec![simd::Isa::Portable];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                v.push(simd::Isa::Avx2);
            }
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
                v.push(simd::Isa::Avx512);
            }
        }
        #[cfg(target_arch = "aarch64")]
        v.push(simd::Isa::Neon);
        v
    }

    /// `n` records cycling through a few positions with varied scores/results.
    fn test_recs(n: usize) -> (Vec<Rec>, Vec<bool>) {
        let mut fens = RESCALE_FENS.to_vec();
        fens.push("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
        fens.push("8/8/8/4k3/8/8/2K5/8 w - - 0 1");
        let mut rng = Rng::new(17);
        let recs = (0..n)
            .map(|i| {
                let score = rng.uniform(900.0) as i32;
                let result = ["0", "0.5", "1"][rng.below(3)];
                encode(&format!("{} | {score} | {result}", fens[i % fens.len()])).unwrap()
            })
            .collect();
        let mirror = (0..n).map(|_| rng.next_u64() & 1 == 1).collect();
        (recs, mirror)
    }

    /// v2's per-sample loop (`train_sample` over the batch in order on the
    /// merged net, one thread), its active-row flags and `adam_step`: the
    /// reference for `batch.rs`. `p` is the training vector (net + F).
    #[allow(clippy::too_many_arguments)]
    fn reference_step(
        p: &[f32],
        sh: Shape,
        kl: &KingLayout,
        fac: Fac,
        wd_active: bool,
        recs: &[Rec],
        mirror: &[bool],
        opt: &mut Adam,
    ) -> (Vec<f32>, f64, Vec<f32>) {
        let mut merged = p[..sh.len()].to_vec();
        if fac.on() {
            merge_factor(p, sh, fac, &mut merged);
        }
        let mut g = vec![0.0f32; sh.len()];
        let (mut acc, mut dacc) = (vec![0.0; 2 * sh.h], vec![0.0; 2 * sh.h]);
        let mut active = vec![0u8; sh.kb * INPUTS];
        let mut loss = 0.0f64;
        for (r, &m) in recs.iter().zip(mirror) {
            let s = decode(r, 0.5, sh.buckets, m, kl);
            loss += train_sample(&merged, sh, &s, &mut g, &mut acc, &mut dacc) as f64;
            for side in 0..2 {
                for &f in &s.f[side][..s.n] {
                    active[f as usize] = 1;
                }
            }
        }
        let mut p = p.to_vec();
        let cx = StepCtx { sh, fac, active: wd_active.then_some(&active[..]) };
        adam_step(&mut p, opt, std::slice::from_ref(&g), 1e-3, 0.01, 1.0 / recs.len() as f32, 3, &cx);
        (g, loss, p)
    }

    /// The batch trainer against v2's loop: bit-identical gradients, loss and
    /// AdamW steps with `exact`, ~1e-6 without; for every ISA, several thread
    /// counts, H not a multiple of the 64-float forward tile, partial
    /// sub-batches, king buckets, the factoriser, active-only decay and
    /// mirroring.
    #[test]
    fn batch_trainer_matches_reference() {
        let cases = [
            (48, 8, NO_BUCKETS, false, false),
            (128, 1, KingLayout::by_name("kb4m"), false, true),
            (64, 8, KingLayout::by_name("kb8m"), false, false),
            (64, 8, KingLayout::by_name("kb8m"), true, true),
            (32, 1, KingLayout::by_name("kb4"), true, false),
        ];
        for (h, buckets, kl, factorise, wd_active) in cases {
            let sh = Shape { h, buckets, kb: kl.kb() };
            let fac = Fac::new(sh, factorise);
            let mut rng = Rng::new(5);
            let mut p = if factorise { init_factor(sh, &mut rng) } else { init_params(sh, &mut rng) };
            if factorise {
                for x in &mut p[..sh.ft_b()] {
                    *x = rng.uniform(0.05); // W != 0, so merging matters
                }
            }
            for x in &mut p[sh.ft_b()..sh.out_w()] {
                *x = 0.2; // keep accumulators inside (0, 1) often
            }
            let (recs, mirror) = test_recs(301);
            let n = p.len();
            let mut ropt = Adam { m: AVec::zeros(n), v: AVec::zeros(n), step: 0 };
            let (gref, lref, pref) = reference_step(&p, sh, kl, fac, wd_active, &recs, &mirror, &mut ropt);
            for isa in isas() {
                for (threads, exact) in [(1, true), (3, true), (4, false), (2, false)] {
                    let mut bt = batch::BatchTrainer::new(sh, 64, exact, fac.fl, wd_active);
                    bt.isa = isa;
                    let mut p2 = p.clone();
                    let mut opt = Adam { m: AVec::zeros(n), v: AVec::zeros(n), step: 0 };
                    batch::Pool::with(threads, |pool| {
                        let loss = bt.batch(pool, &p2, &recs, &mirror, 0.5, kl);
                        let g = bt.gradient();
                        let what = format!("H={h} {} fac={factorise} wda={wd_active} {isa:?} threads={threads} exact={exact}", kl.name);
                        if exact {
                            assert_eq!(loss, lref, "{what}");
                            assert!(g.iter().zip(&gref).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: gradient differs");
                        } else {
                            assert!((loss - lref).abs() <= 1e-6 * lref, "{what}: loss {loss} vs {lref}");
                            let scale = gref.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                            for (i, (a, b)) in g.iter().zip(&gref).enumerate() {
                                assert!((a - b).abs() <= 1e-5 * scale, "{what}: param {i}: {a} vs {b}");
                            }
                        }
                        bt.adam(pool, &mut p2, &mut opt, 1e-3, 0.01, 1.0 / recs.len() as f32);
                        assert!(bt.gradient().iter().all(|&x| x == 0.0), "{what}: gradient not cleared");
                        if exact {
                            assert!(p2.iter().zip(&pref).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: Adam step differs");
                            assert!(opt.m.iter().zip(ropt.m.iter()).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: Adam m");
                            assert!(opt.v.iter().zip(ropt.v.iter()).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: Adam v");
                            // Second step: the trainer's (merged) FT copy and the
                            // active-row flags must follow the update.
                            let mut ropt2 = Adam { m: ropt.m.clone(), v: ropt.v.clone(), step: ropt.step };
                            let (g2, l2, p3) = reference_step(&pref, sh, kl, fac, wd_active, &recs[..200], &mirror[..200], &mut ropt2);
                            assert_eq!(bt.batch(pool, &p2, &recs[..200], &mirror[..200], 0.5, kl), l2, "{what}: second step loss");
                            assert!(bt.gradient().iter().zip(&g2).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: second step gradient");
                            bt.adam(pool, &mut p2, &mut opt, 1e-3, 0.01, 1.0 / 200.0);
                            assert!(p2.iter().zip(&p3).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: second Adam step");
                        }
                    });
                }
            }
        }
    }

    /// Without `exact`, results do not depend on threads or ISA at all.
    #[test]
    fn batch_trainer_is_deterministic() {
        let sh = Shape { h: 96, buckets: 8, kb: 1 };
        let mut rng = Rng::new(9);
        let p = init_params(sh, &mut rng);
        let (recs, mirror) = test_recs(500);
        let mut first: Option<(f64, Vec<f32>)> = None;
        for isa in isas() {
            for threads in [1, 2, 5] {
                let mut bt = batch::BatchTrainer::new(sh, 0, false, 0, false);
                bt.isa = isa;
                let got = batch::Pool::with(threads, |pool| (bt.batch(pool, &p, &recs, &mirror, 0.5, NO_BUCKETS), bt.gradient()));
                match &first {
                    None => first = Some(got),
                    Some(f) => {
                        assert_eq!(f.0, got.0, "{isa:?} threads={threads}");
                        assert!(f.1.iter().zip(&got.1).all(|(a, b)| a.to_bits() == b.to_bits()), "{isa:?} threads={threads}");
                    }
                }
            }
        }
    }
}
