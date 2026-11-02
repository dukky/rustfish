//! NNUE trainer for rustfish.
//!
//! Trains the (768 -> 256) x 2 -> 1 SCReLU network that `src/nnue.rs` runs,
//! from the self-play text format written by `datagen`:
//!
//! ```text
//! <FEN> | <score_cp from White's POV> | <result 1.0 | 0.5 | 0.0>
//! ```
//!
//! The network is trained in the same integer-ish scale the inference code
//! uses (accumulator clamped at QA = 255), so export is a plain round to i16;
//! there is no separate float->quantized rescaling step. Weights are
//! feature-major, exactly like the runtime file:
//!
//! ```text
//! feature weights [768][H], feature bias [H], output weights [2H], output bias
//! ```
//!
//! Std only, like the engine. Run `cargo build --release` in this directory.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

const INPUTS: usize = 768;
const H: usize = 256;
const QA: f32 = 255.0;
const QB: f32 = 64.0;
const SCALE: f32 = 400.0;
/// `(sum / QA + out_b) * SCALE / (QA * QB)`: output units -> centipawns.
const OUT_SCALE: f32 = SCALE / (QA * QB);

const NET_BYTES: usize = (INPUTS * H + H + 2 * H + 1) * 2;
const CHECKPOINT_MAGIC: &[u8; 8] = b"RFNNT1\0\0";

/// A position as two perspective feature lists (max 32 pieces per side).
#[derive(Clone, Copy)]
struct Sample {
    feats: [[u16; 32]; 2],
    counts: [u8; 2],
    stm_white: bool,
    score: f32,
    result: f32,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    #[inline]
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Parse the board and side-to-move fields of a FEN into both perspectives'
/// feature lists. Feature layout must stay in lockstep with `nnue::feature`:
/// `(piece colour != perspective ? 384 : 0) + kind*64 + (persp == White ? sq : sq ^ 56)`,
/// with `kind`: pawn 0 .. king 5 and a1 = 0.
fn parse_fen(fen: &str) -> Option<(Sample, bool)> {
    let mut fields = fen.split_whitespace();
    let board = fields.next()?;
    let stm = fields.next()?;
    let stm_white = match stm {
        "w" => true,
        "b" => false,
        _ => return None,
    };

    let mut sample = Sample {
        feats: [[0; 32]; 2],
        counts: [0; 2],
        stm_white,
        score: 0.0,
        result: 0.0,
    };
    let mut file = 0i32;
    let mut rank = 7i32;
    for c in board.chars() {
        match c {
            '/' => {
                rank -= 1;
                file = 0;
            }
            '1'..='8' => file += c as i32 - '0' as i32,
            _ => {
                let (color_white, kind) = match c {
                    'P' => (true, 0),
                    'N' => (true, 1),
                    'B' => (true, 2),
                    'R' => (true, 3),
                    'Q' => (true, 4),
                    'K' => (true, 5),
                    'p' => (false, 0),
                    'n' => (false, 1),
                    'b' => (false, 2),
                    'r' => (false, 3),
                    'q' => (false, 4),
                    'k' => (false, 5),
                    _ => return None,
                };
                if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                    return None;
                }
                let sq = (rank * 8 + file) as u8;
                for (p, persp_white) in [true, false].into_iter().enumerate() {
                    let side = if color_white == persp_white { 0 } else { 384 };
                    let s = if persp_white { sq } else { sq ^ 56 };
                    let f = side + kind * 64 + s as usize;
                    let n = sample.counts[p] as usize;
                    if n >= 32 {
                        return None;
                    }
                    sample.feats[p][n] = f as u16;
                    sample.counts[p] = (n + 1) as u8;
                }
                file += 1;
            }
        }
    }
    Some((sample, stm_white))
}

/// A line of datagen output: `FEN | score_cp (White POV) | result (White POV)`.
fn parse_line(line: &str) -> Option<Sample> {
    let mut parts = line.split('|');
    let fen = parts.next()?.trim();
    let score: f32 = parts.next()?.trim().parse().ok()?;
    let result: f32 = parts.next()?.trim().parse().ok()?;
    let (mut sample, stm_white) = parse_fen(fen)?;
    if !(0.0..=1.0).contains(&result) {
        return None;
    }
    // Data is White POV; the net learns the side-to-move POV.
    if stm_white {
        sample.score = score;
        sample.result = result;
    } else {
        sample.score = -score;
        sample.result = 1.0 - result;
    }
    Some(sample)
}

/// Sequential reader over one or more plain (uncompressed) data files.
struct LineReader {
    files: Vec<PathBuf>,
    next_file: usize,
    reader: Option<BufReader<File>>,
    line: String,
}

impl LineReader {
    fn new(files: Vec<PathBuf>) -> Self {
        Self { files, next_file: 0, reader: None, line: String::new() }
    }

    fn next_line(&mut self) -> Option<String> {
        loop {
            if self.reader.is_none() {
                if self.next_file >= self.files.len() {
                    return None;
                }
                let f = File::open(&self.files[self.next_file]).ok()?;
                self.next_file += 1;
                self.reader = Some(BufReader::with_capacity(1 << 20, f));
            }
            self.line.clear();
            let read = self.reader.as_mut().unwrap().read_line(&mut self.line).ok()?;
            if read == 0 {
                self.reader = None;
                continue;
            }
            return Some(self.line.clone());
        }
    }
}

struct Shuffler {
    buf: Vec<Sample>,
    reader: LineReader,
    rng: Rng,
    capacity: usize,
    skipped: u64,
}

impl Shuffler {
    fn new(files: Vec<PathBuf>, capacity: usize, seed: u64) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            reader: LineReader::new(files),
            rng: Rng::new(seed),
            capacity,
            skipped: 0,
        }
    }

    fn fill(&mut self) {
        while self.buf.len() < self.capacity {
            match self.reader.next_line() {
                Some(line) => match parse_line(&line) {
                    Some(s) => self.buf.push(s),
                    None => self.skipped += 1,
                },
                None => break,
            }
        }
    }

    /// Start a new pass over the data files (shuffle order continues).
    fn rewind(&mut self) {
        self.buf.clear();
        self.reader = LineReader::new(self.reader.files.clone());
    }

    /// Reservoir-style bounded shuffle: every returned sample is drawn from a
    /// roughly `capacity`-sized rolling window, so memory stays flat on 65M+
    /// position files while consecutive samples are still decorrelated.
    fn next(&mut self) -> Option<Sample> {
        if self.buf.len() < 4096 {
            self.fill();
        }
        if self.buf.is_empty() {
            return None;
        }
        let i = self.rng.below(self.buf.len());
        let sample = self.buf[i];
        let mut line = self.reader.next_line();
        loop {
            match line {
                Some(l) => match parse_line(&l) {
                    Some(s) => {
                        self.buf[i] = s;
                        break;
                    }
                    None => {
                        self.skipped += 1;
                        line = self.reader.next_line();
                    }
                },
                None => {
                    self.buf.swap_remove(i);
                    break;
                }
            }
        }
        Some(sample)
    }
}

struct Net {
    w: Vec<f32>,
    b: Vec<f32>,
    ow: Vec<f32>,
    ob: f32,
}

impl Net {
    fn new(rng: &mut Rng) -> Self {
        let mut w = vec![0.0f32; INPUTS * H];
        for x in w.iter_mut() {
            *x = (rng.f32() - 0.5) * 32.0;
        }
        let mut ow = vec![0.0f32; 2 * H];
        for x in ow.iter_mut() {
            *x = (rng.f32() - 0.5) * 0.1;
        }
        Self { w, b: vec![0.0; H], ow, ob: 0.0 }
    }

    /// Inference in trainer units; used by tests and handy for sanity checks.
    #[allow(dead_code)]
    fn eval(&self, s: &Sample) -> f32 {
        let (us, them) = if s.stm_white { (0, 1) } else { (1, 0) };
        let mut sum = 0.0f32;
        for p in [us, them] {
            let mut acc = [0.0f32; H];
            acc.copy_from_slice(&self.b);
            for i in 0..s.counts[p] as usize {
                let row = s.feats[p][i] as usize * H;
                for (a, w) in acc.iter_mut().zip(&self.w[row..row + H]) {
                    *a += *w;
                }
            }
            let off = if p == us { 0 } else { H };
            for h in 0..H {
                let a = acc[h].clamp(0.0, QA);
                sum += a * a * self.ow[off + h];
            }
        }
        (sum / QA + self.ob) * OUT_SCALE
    }
}

/// Per-thread gradient buffer; one per worker, summed once per batch.
struct Grad {
    w: Vec<f32>,
    b: Vec<f32>,
    ow: Vec<f32>,
    ob: f32,
    loss: f64,
}

impl Grad {
    fn new() -> Self {
        Self { w: vec![0.0; INPUTS * H], b: vec![0.0; H], ow: vec![0.0; 2 * H], ob: 0.0, loss: 0.0 }
    }

    fn clear(&mut self) {
        self.w.fill(0.0);
        self.b.fill(0.0);
        self.ow.fill(0.0);
        self.ob = 0.0;
        self.loss = 0.0;
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Forward, loss and backward for one sample, accumulating into `g`.
fn accumulate(g: &mut Grad, net: &Net, s: &Sample, lambda: f32) {
    let (us, them) = if s.stm_white { (0usize, 1usize) } else { (1, 0) };
    let mut acc = [[0.0f32; H]; 2];
    let mut a = [[0.0f32; H]; 2];
    let mut sq = [[0.0f32; H]; 2];
    let mut sum = 0.0f32;

    for p in [us, them] {
        acc[p].copy_from_slice(&net.b);
        for i in 0..s.counts[p] as usize {
            let row = s.feats[p][i] as usize * H;
            for (x, w) in acc[p].iter_mut().zip(&net.w[row..row + H]) {
                *x += *w;
            }
        }
        for h in 0..H {
            let v = acc[p][h].clamp(0.0, QA);
            a[p][h] = v;
            sq[p][h] = v * v;
            sum += sq[p][h] * net.ow[if p == us { h } else { H + h }];
        }
    }

    let eval = (sum / QA + net.ob) * OUT_SCALE;
    let pred = sigmoid(eval / SCALE);
    let wdl = (pred - s.result) * (pred - s.result);
    let ev = ((eval - s.score) / SCALE) * ((eval - s.score) / SCALE);
    g.loss += (lambda * wdl + (1.0 - lambda) * ev) as f64;

    // d loss / d eval
    let dl_de = lambda * 2.0 * (pred - s.result) * pred * (1.0 - pred) / SCALE
        + (1.0 - lambda) * 2.0 * (eval - s.score) / (SCALE * SCALE);
    let dsum = dl_de * OUT_SCALE / QA;
    g.ob += dl_de * OUT_SCALE;

    for h in 0..H {
        g.ow[h] += dsum * sq[us][h];
        g.ow[H + h] += dsum * sq[them][h];
    }

    // d loss / d accumulator for each perspective, then scatter to feature rows.
    for p in [us, them] {
        let off = if p == us { 0 } else { H };
        let mut dacc = [0.0f32; H];
        for h in 0..H {
            if a[p][h] > 0.0 && a[p][h] < QA {
                dacc[h] = dsum * net.ow[off + h] * 2.0 * a[p][h];
            }
        }
        for i in 0..s.counts[p] as usize {
            let row = s.feats[p][i] as usize * H;
            for (gw, d) in g.w[row..row + H].iter_mut().zip(&dacc) {
                *gw += *d;
            }
        }
        for (gb, d) in g.b.iter_mut().zip(&dacc) {
            *gb += *d;
        }
    }
}

fn run_batch(net: &Net, batch: &[Sample], grads: &mut [Grad], lambda: f32) -> f64 {
    for g in grads.iter_mut() {
        g.clear();
    }
    let chunk = batch.len().div_ceil(grads.len()).max(1);
    std::thread::scope(|scope| {
        for (g, part) in grads.iter_mut().zip(batch.chunks(chunk)) {
            scope.spawn(move || {
                for s in part {
                    accumulate(g, net, s, lambda);
                }
            });
        }
    });
    grads.iter().map(|g| g.loss).sum()
}

fn adam(params: &mut [f32], m: &mut [f32], v: &mut [f32], grad: &[f32], lr: f32, t: u32, scale: f32) {
    let b1 = 0.9f32;
    let b2 = 0.999f32;
    let eps = 1e-8f32;
    let c1 = 1.0 - b1.powi(t as i32);
    let c2 = 1.0 - b2.powi(t as i32);
    for i in 0..params.len() {
        let g = grad[i] * scale;
        m[i] = b1 * m[i] + (1.0 - b1) * g;
        v[i] = b2 * v[i] + (1.0 - b2) * g * g;
        params[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
    }
}

struct Opt {
    mw: Vec<f32>,
    vw: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
    mow: Vec<f32>,
    vow: Vec<f32>,
    mob: f32,
    vob: f32,
    step: u32,
}

impl Opt {
    fn new() -> Self {
        Self {
            mw: vec![0.0; INPUTS * H],
            vw: vec![0.0; INPUTS * H],
            mb: vec![0.0; H],
            vb: vec![0.0; H],
            mow: vec![0.0; 2 * H],
            vow: vec![0.0; 2 * H],
            mob: 0.0,
            vob: 0.0,
            step: 0,
        }
    }

    fn update(&mut self, net: &mut Net, grad: &Grad, count: usize, lr: f32) {
        self.step += 1;
        let inv = 1.0 / count as f32;
        let gob = grad.ob * inv;

        adam(&mut net.w, &mut self.mw, &mut self.vw, &grad.w, lr, self.step, inv);
        adam(&mut net.b, &mut self.mb, &mut self.vb, &grad.b, lr, self.step, inv);
        adam(&mut net.ow, &mut self.mow, &mut self.vow, &grad.ow, lr, self.step, inv);

        // Scalar bias with the same Adam rules.
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps = 1e-8f32;
        let c1 = 1.0 - b1.powi(self.step as i32);
        let c2 = 1.0 - b2.powi(self.step as i32);
        self.mob = b1 * self.mob + (1.0 - b1) * gob;
        self.vob = b2 * self.vob + (1.0 - b2) * gob * gob;
        net.ob -= lr * (self.mob / c1) / ((self.vob / c2).sqrt() + eps);
    }
}

fn save_checkpoint(path: &PathBuf, net: &Net, opt: &Opt) -> std::io::Result<()> {
    let mut out = Vec::with_capacity(8 + 12 + (INPUTS * H + H + 2 * H + 1) * 4 * 3);
    out.extend_from_slice(CHECKPOINT_MAGIC);
    out.extend_from_slice(&(H as u32).to_le_bytes());
    out.extend_from_slice(&opt.step.to_le_bytes());
    for arr in [&net.w, &net.b, &net.ow] {
        for x in arr.iter() {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out.extend_from_slice(&net.ob.to_le_bytes());
    for arr in [&opt.mw, &opt.vw, &opt.mb, &opt.vb, &opt.mow, &opt.vow] {
        for x in arr.iter() {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out.extend_from_slice(&opt.mob.to_le_bytes());
    out.extend_from_slice(&opt.vob.to_le_bytes());
    let mut f = File::create(path)?;
    f.write_all(&out)
}

fn load_checkpoint(path: &PathBuf, net: &mut Net, opt: &mut Opt) -> std::io::Result<()> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 20 || &bytes[..8] != CHECKPOINT_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a trainer checkpoint"));
    }
    let h = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if h as usize != H {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "hidden size differs"));
    }
    if bytes.len() != 16 + (INPUTS * H + H + 2 * H + 1) * 4 * 3 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "checkpoint size differs"));
    }
    opt.step = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let mut pos = 16;
    net.w = take_f32(&bytes, &mut pos, INPUTS * H);
    net.b = take_f32(&bytes, &mut pos, H);
    net.ow = take_f32(&bytes, &mut pos, 2 * H);
    net.ob = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
    pos += 4;
    opt.mw = take_f32(&bytes, &mut pos, INPUTS * H);
    opt.vw = take_f32(&bytes, &mut pos, INPUTS * H);
    opt.mb = take_f32(&bytes, &mut pos, H);
    opt.vb = take_f32(&bytes, &mut pos, H);
    opt.mow = take_f32(&bytes, &mut pos, 2 * H);
    opt.vow = take_f32(&bytes, &mut pos, 2 * H);
    opt.mob = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
    pos += 4;
    opt.vob = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
    Ok(())
}

fn take_f32(bytes: &[u8], pos: &mut usize, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(f32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap()));
        *pos += 4;
    }
    v
}

/// Round to the runtime's little-endian i16 layout and report what got clipped.
fn net_bytes(net: &Net) -> (Vec<u8>, usize, bool) {
    let mut out = Vec::with_capacity(NET_BYTES);
    let mut clipped = 0usize;
    let mut small_out = true;
    let push = |x: f32, out: &mut Vec<u8>, clipped: &mut usize| {
        let r = x.round();
        if r < -32768.0 || r > 32767.0 {
            *clipped += 1;
        }
        let q = r.clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&q.to_le_bytes());
        q
    };
    for &x in &net.w {
        push(x, &mut out, &mut clipped);
    }
    for &x in &net.b {
        push(x, &mut out, &mut clipped);
    }
    for &x in &net.ow {
        let q = push(x, &mut out, &mut clipped);
        if !(-127..=127).contains(&q) {
            small_out = false;
        }
    }
    push(net.ob, &mut out, &mut clipped);
    assert_eq!(out.len(), NET_BYTES);
    (out, clipped, small_out)
}

fn export(net: &Net, path: &PathBuf) -> std::io::Result<()> {
    let (out, clipped, small_out) = net_bytes(net);
    File::create(path)?.write_all(&out)?;
    println!(
        "wrote {} ({} bytes): {} values clipped to i16, output weights {}",
        path.display(),
        out.len(),
        clipped,
        if small_out { "use the fast i32 path" } else { "need the wide i64 path" }
    );
    Ok(())
}

struct Args {
    data: Vec<PathBuf>,
    out: PathBuf,
    epochs: u32,
    batch: usize,
    lr: f32,
    lambda: f32,
    threads: usize,
    buffer: usize,
    max_positions: Option<u64>,
    seed: u64,
    checkpoint_every: u64,
    checkpoint: Option<PathBuf>,
    resume: Option<PathBuf>,
    log_every: u64,
}

impl Args {
    fn parse() -> Self {
        let mut a = Self {
            data: Vec::new(),
            out: PathBuf::from("net.bin"),
            epochs: 1,
            batch: 1024,
            lr: 1e-3,
            lambda: 0.5,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            buffer: 65536,
            max_positions: None,
            seed: 1,
            checkpoint_every: 0,
            checkpoint: None,
            resume: None,
            log_every: 10000,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = || it.next().unwrap_or_else(|| panic!("missing value for {flag}"));
            match flag.as_str() {
                "--data" => a.data.push(PathBuf::from(value())),
                "--out" => a.out = PathBuf::from(value()),
                "--epochs" => a.epochs = value().parse().unwrap(),
                "--batch" => a.batch = value().parse().unwrap(),
                "--lr" => a.lr = value().parse().unwrap(),
                "--lambda" => a.lambda = value().parse().unwrap(),
                "--threads" => a.threads = value().parse().unwrap(),
                "--shuffle-buffer" => a.buffer = value().parse().unwrap(),
                "--max-positions" => a.max_positions = Some(value().parse().unwrap()),
                "--seed" => a.seed = value().parse().unwrap(),
                "--checkpoint-every" => a.checkpoint_every = value().parse().unwrap(),
                "--checkpoint" => a.checkpoint = Some(PathBuf::from(value())),
                "--resume" => a.resume = Some(PathBuf::from(value())),
                "--log-every" => a.log_every = value().parse().unwrap(),
                "--help" | "-h" => {
                    println!(
                        "usage: nnue-trainer --data FILE [--data FILE...] --out net.bin\n\
                         \t[--epochs N] [--batch N] [--lr F] [--lambda F] [--threads N]\n\
                         \t[--shuffle-buffer N] [--max-positions N] [--seed N]\n\
                         \t[--checkpoint-every N] [--checkpoint FILE] [--resume FILE]\n\
                         \t[--log-every N]\n\n\
                         Data is plain text from `gunzip -c datagen-*.gz` in the format\n\
                         `<FEN> | <score_cp White POV> | <result White POV>`."
                    );
                    std::process::exit(0);
                }
                other => panic!("unknown option {other}"),
            }
        }
        assert!(!a.data.is_empty(), "--data is required");
        a
    }
}

fn main() {
    let args = Args::parse();
    let mut rng = Rng::new(args.seed);
    let mut net = Net::new(&mut rng);
    let mut opt = Opt::new();
    if let Some(path) = &args.resume {
        load_checkpoint(path, &mut net, &mut opt).unwrap_or_else(|e| panic!("resume {path:?}: {e}"));
        println!("resumed from {} at step {}", path.display(), opt.step);
    }

    let mut shuffler = Shuffler::new(args.data.clone(), args.buffer, args.seed);
    let mut grads: Vec<Grad> = (0..args.threads).map(|_| Grad::new()).collect();
    let mut batch: Vec<Sample> = Vec::with_capacity(args.batch);
    let started = Instant::now();
    let mut seen: u64 = 0;
    let mut total_loss = 0.0f64;
    let mut loss_count: u64 = 0;
    let mut overall_loss = 0.0f64;
    let mut overall_count: u64 = 0;

    'epochs: for epoch in 1..=args.epochs {
        shuffler.rewind();
        let mut limit = args.max_positions.unwrap_or(u64::MAX);
        loop {
            batch.clear();
            while batch.len() < args.batch && limit > 0 {
                match shuffler.next() {
                    Some(s) => {
                        batch.push(s);
                        limit -= 1;
                    }
                    None => break,
                }
            }
            if batch.is_empty() {
                break;
            }
            let loss = run_batch(&net, &batch, &mut grads, args.lambda);
            let (first, rest) = grads.split_first_mut().unwrap();
            for g in rest {
                add_grad(first, g);
            }
            opt.update(&mut net, first, batch.len(), args.lr);
            seen += batch.len() as u64;
            total_loss += loss;
            loss_count += batch.len() as u64;
            overall_loss += loss;
            overall_count += batch.len() as u64;

            if opt.step as u64 % args.log_every.max(1) == 0 {
                let secs = started.elapsed().as_secs_f64().max(1e-9);
                println!(
                    "epoch {epoch} step {} positions {} loss {:.6} ({:.0} pos/s)",
                    opt.step,
                    seen,
                    total_loss / loss_count as f64,
                    seen as f64 / secs
                );
                total_loss = 0.0;
                loss_count = 0;
            }
            if args.checkpoint_every > 0 && seen % args.checkpoint_every == 0 {
                let path = args.checkpoint.clone().unwrap_or_else(|| PathBuf::from("checkpoint.bin"));
                save_checkpoint(&path, &net, &opt).unwrap();
                println!("checkpoint {}", path.display());
            }
            if args.max_positions.is_some() && limit == 0 {
                continue 'epochs;
            }
        }
    }

    println!(
        "trained {} positions in {:.1}s, average loss {:.6}",
        seen,
        started.elapsed().as_secs_f64(),
        if overall_count > 0 { overall_loss / overall_count as f64 } else { 0.0 }
    );
    export(&net, &args.out).unwrap();
    println!("skipped {} malformed lines", shuffler.skipped);
}

fn add_grad(dst: &mut Grad, src: &Grad) {
    for (a, b) in dst.w.iter_mut().zip(&src.w) {
        *a += *b;
    }
    for (a, b) in dst.b.iter_mut().zip(&src.b) {
        *a += *b;
    }
    for (a, b) in dst.ow.iter_mut().zip(&src.ow) {
        *a += *b;
    }
    dst.ob += src.ob;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_features_match_runtime_layout() {
        let (s, white) =
            parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(white);
        assert_eq!(s.counts, [32, 32]);
        // Both kings are feature 324 in their own perspective (king kind 5,
        // e1/e8 flip to the same relative square); the other king is 764.
        for p in 0..2 {
            let f = &s.feats[p][..s.counts[p] as usize];
            assert!(f.contains(&324), "perspective {p} missing own king");
            assert!(f.contains(&764), "perspective {p} missing enemy king");
            let mut sorted = f.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), f.len(), "duplicate features in perspective {p}");
        }
        // Pawn on a2 (sq 8): white perspective kind 0 square 8; black
        // perspective 384 + (8 ^ 56 = 48) = 432.
        assert!(s.feats[0][..32].contains(&8));
        assert!(s.feats[1][..32].contains(&432));
    }

    #[test]
    fn black_to_move_data_is_mirrored() {
        let s = parse_line("4k3/8/8/8/8/8/8/4K3 b - - 0 1 | 50 | 1.0").unwrap();
        assert!(!s.stm_white);
        assert_eq!(s.score, -50.0);
        assert_eq!(s.result, 0.0);
        let s = parse_line("4k3/8/8/8/8/8/8/4K3 w - - 0 1 | -30 | 0.0").unwrap();
        assert_eq!(s.score, -30.0);
        assert_eq!(s.result, 0.0);
    }

    #[test]
    fn export_rounds_to_little_endian_i16() {
        let mut net = Net {
            w: vec![0.0; INPUTS * H],
            b: vec![0.0; H],
            ow: vec![0.0; 2 * H],
            ob: 0.0,
        };
        net.w[0] = 1.4;
        net.w[1] = -1.6;
        net.w[2] = 40000.0;
        net.b[0] = -300.4;
        net.ow[0] = 200.0; // outside the fast i32 output path
        net.ob = 123.0;
        let (bytes, clipped, small_out) = net_bytes(&net);
        assert_eq!(bytes.len(), NET_BYTES);
        assert_eq!(clipped, 1);
        assert!(!small_out);
        assert_eq!(i16::from_le_bytes([bytes[0], bytes[1]]), 1);
        assert_eq!(i16::from_le_bytes([bytes[2], bytes[3]]), -2);
        assert_eq!(i16::from_le_bytes([bytes[4], bytes[5]]), 32767);
        let b_off = INPUTS * H * 2;
        assert_eq!(i16::from_le_bytes([bytes[b_off], bytes[b_off + 1]]), -300);
    }

    #[test]
    fn eval_matches_manual_formula_for_hand_net() {
        // Same hand net as nnue.rs's tiny_net test, in float/trainer units.
        let mut net = Net {
            w: vec![0.0; INPUTS * H],
            b: vec![0.0; H],
            ow: vec![0.0; 2 * H],
            ob: 0.0,
        };
        net.b[0] = 100.0;
        net.b[1] = 300.0;
        net.w[324 * H] = 50.0;
        net.w[764 * H] = -200.0;
        net.w[764 * H + 2] = 20.0;
        net.ow[0] = 3.0;
        net.ow[1] = -2.0;
        net.ow[H + 2] = 7.0;
        net.ob = 11.0;
        let (s, _) = parse_fen("8/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        // (150^2*3 + 255^2*(-2) + 20^2*7) = 22500*3 - 65025*2 + 400*7
        let sum = 150.0 * 150.0 * 3.0 + 255.0 * 255.0 * -2.0 + 20.0 * 20.0 * 7.0;
        let want = (sum / QA + 11.0) * OUT_SCALE;
        assert!((net.eval(&s) - want).abs() < 1e-3, "{} vs {}", net.eval(&s), want);
    }
}
