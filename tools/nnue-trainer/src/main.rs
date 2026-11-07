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
//! net); `--init` from a one-bucket net sets `F` = its rows, `W = 0`.
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

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
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
    (floats[..n].to_vec(), Adam { m: floats[n..2 * n].to_vec(), v: floats[2 * n..].to_vec(), step: word(2) }, word(3))
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
    let mut seed = 1u64;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out_path = Some(PathBuf::from(it.next().unwrap())),
            "--seed" => seed = it.next().unwrap().parse().unwrap(),
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
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--net" => net = Some(PathBuf::from(it.next().unwrap())),
            "--val" => val = Some(PathBuf::from(it.next().unwrap())),
            "--buckets" => buckets = it.next().unwrap().parse().unwrap(),
            "--king-buckets" => kl = KingLayout::by_name(it.next().unwrap()),
            "--lambda" => lambda = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown option {other}"),
        }
    }
    let (p, sh) = load_net(&net.expect("--net"), buckets, kl.kb());
    let val = read_recs(&val.expect("--val"));
    let (l, e) = validate(&p, sh, kl, &val, lambda, threads_default());
    print!("H={} buckets={} ", sh.h, sh.buckets);
    if kl.id() != 0 {
        print!("king-buckets={} ", kl.name);
    }
    println!("val positions {} loss {:.6} mean|eval| {:.1}cp", val.len(), l, e);
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
    let (mut factorise, mut wd_active) = (false, None::<bool>);
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
            "--factorise" | "--factorize" => factorise = true,
            "--wd-active-only" => wd_active = Some(true),
            "--no-wd-active-only" => wd_active = Some(false),
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
    let (mut p, mut opt, mut seen) = match (&resume, &init) {
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
            (p.into_iter().map(|x| x.clamp(-CLIP, CLIP)).collect(), Adam { m: vec![0.0; n], v: vec![0.0; n], step: 0 }, 0)
        }
        (None, None) => {
            let p = if fac.on() { init_factor(sh, &mut rng) } else { init_params(sh, &mut rng) };
            let n = p.len();
            (p, Adam { m: vec![0.0; n], v: vec![0.0; n], step: 0 }, 0)
        }
    };
    // Effective (merged) net params; only a separate buffer with the factoriser.
    let mut merged = if fac.on() { vec![0.0f32; sh.len()] } else { Vec::new() };
    if fac.on() {
        merge_factor(&p, sh, fac, &mut merged);
    }
    let val = val_path.as_deref().map(read_recs).unwrap_or_default();
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
    let (vl, ve) = validate(if fac.on() { &merged } else { &p }, sh, kl, &val, lambda, threads);
    println!("start val loss {vl:.6} mean|eval| {ve:.1}cp");

    let mut feed = Feed::new(data, seed ^ 0xF00D ^ seen);
    let mut grads: Vec<Vec<f32>> = (0..threads).map(|_| vec![0.0f32; sh.len()]).collect();
    // --wd-active-only: per thread, which FT rows occurred in the batch.
    let n_rows = if wd_active { sh.kb * INPUTS } else { 0 };
    let mut acts: Vec<Vec<u8>> = (0..threads).map(|_| vec![0u8; n_rows]).collect();
    let mut active = vec![0u8; n_rows];
    let mut recs: Vec<Rec> = Vec::with_capacity(batch);
    let started = Instant::now();
    let start_seen = seen;
    let mut epoch_loss = 0.0f64;
    let mut epoch_n = 0u64;
    let mut ep = seen / epoch_size;
    while seen < total {
        let t = seen as f64 / total as f64;
        let cur_lr = lr_end + 0.5 * (lr - lr_end) * (1.0 + (std::f64::consts::PI * t).cos() as f32);
        feed.batch(&mut recs, batch);
        // Small chunks pulled from a shared counter: the M3's efficiency
        // cores take fewer of them instead of holding up the whole batch.
        let chunk = 256usize;
        let n_chunks = recs.len().div_ceil(chunk);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let seed_b = rng.next_u64();
        let loss: f64 = std::thread::scope(|sc| {
            let (p, recs, next) = (if fac.on() { &merged[..] } else { &p[..] }, &recs, &next);
            let hs: Vec<_> = grads
                .iter_mut()
                .zip(acts.iter_mut())
                .map(|(g, act)| {
                    sc.spawn(move || {
                        g.fill(0.0);
                        act.fill(0);
                        let mut acc = vec![0.0f32; 2 * sh.h];
                        let mut dacc = vec![0.0f32; 2 * sh.h];
                        let mut l = 0.0f64;
                        loop {
                            let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if c >= n_chunks {
                                break;
                            }
                            let mut mr = Rng::new(seed_b ^ c as u64);
                            for r in &recs[c * chunk..((c + 1) * chunk).min(recs.len())] {
                                let m = mirror && mr.next_u64() & 1 == 1;
                                let s = decode(r, lambda, sh.buckets, m, kl);
                                l += train_sample(p, sh, &s, g, &mut acc, &mut dacc) as f64;
                                if wd_active {
                                    for side in 0..2 {
                                        for &f in &s.f[side][..s.n] {
                                            act[f as usize] = 1;
                                        }
                                    }
                                }
                            }
                        }
                        l
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).sum()
        });
        let used = &grads[..];
        if wd_active {
            active.fill(0);
            for a in &acts {
                for (x, y) in active.iter_mut().zip(a) {
                    *x |= *y;
                }
            }
        }
        let cx = StepCtx { sh, fac, active: wd_active.then_some(&active[..]) };
        adam_step(&mut p, &mut opt, used, cur_lr, wd, 1.0 / recs.len() as f32, threads, &cx);
        if fac.on() {
            merge_factor(&p, sh, fac, &mut merged);
        }
        seen += recs.len() as u64;
        epoch_loss += loss;
        epoch_n += recs.len() as u64;

        if seen / epoch_size != (seen - recs.len() as u64) / epoch_size || seen >= total {
            ep += 1;
            let secs = started.elapsed().as_secs_f64();
            let pm = if fac.on() { &merged[..] } else { &p[..] };
            let (vl, ve) = validate(pm, sh, kl, &val, lambda, threads);
            let q = quantize(pm, sh);
            let (ql, _) = validate(&dequantize(&q, sh), sh, kl, &val, lambda, threads);
            let path = out.with_extension(format!("e{ep}.bin"));
            File::create(&path).unwrap().write_all(&q).unwrap();
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
    let q = quantize(if fac.on() { &merged } else { &p }, sh);
    File::create(&out).unwrap().write_all(&q).unwrap();
    println!("wrote {} ({} bytes, H={} buckets={}{kb_note})", out.display(), q.len(), sh.h, sh.buckets);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: nnue-trainer <convert|shuffle|train|eval> ... (see the module docs)";
    let (cmd, rest) = args.split_first().unwrap_or_else(|| panic!("{usage}"));
    match cmd.as_str() {
        "convert" => cmd_convert(rest),
        "shuffle" => cmd_shuffle(rest),
        "train" => cmd_train(rest),
        "eval" => cmd_eval(rest),
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
        let mut opt = Adam { m: vec![0.0; n], v: vec![0.0; n], step: 0 };
        let mut q = p.clone();
        let cx = StepCtx { sh, fac, active: Some(&active) };
        adam_step(&mut q, &mut opt, std::slice::from_ref(&g), 1e-3, 0.1, 0.5, 2, &cx);
        assert_eq!(&q[idle * sh.h..(idle + 1) * sh.h], &p[idle * sh.h..(idle + 1) * sh.h]);
        assert!(q[sh.len() + fk] != p[sh.len() + fk]);
        let mut q2 = p.clone();
        let mut opt = Adam { m: vec![0.0; n], v: vec![0.0; n], step: 0 };
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
        let opt = Adam { m: pf.clone(), v: pf.clone(), step: 9 };
        save_ckpt(&path, sh, kl, fac, &pf, &opt, 1234);
        let (p2, o2, seen) = load_ckpt(&path, sh, kl, fac);
        assert_eq!((p2, o2.step, seen), (pf, 9, 1234));
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
}
