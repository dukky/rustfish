//! Streaming shuffle: train straight from game-ordered sources (packed
//! `.rfp` files or 32-byte `.bin` files), no shuffled pool copy on disk.
//!
//! Every pass over the data visits all chunks of all sources in one random
//! order (so files are interleaved at chunk granularity); decoder threads
//! read and decode the chunks (in a fixed round-robin, so the stream is
//! deterministic for a seed); the consumer pushes each record into a
//! shuffle buffer of `cap` records and emits a uniformly random resident
//! (TensorFlow-style shuffle buffer: every record is emitted exactly once,
//! after an exponentially distributed delay of mean `cap`). See
//! `docs/data-format.md` for how large `cap` must be.
//!
//! `ChunkSource` is the interface for any game-ordered record source that
//! can be split into independently readable chunks (a binpack reader can
//! implement it and share the shuffle).

use crate::packed::{is_packed, read_at, PackedFile};
use crate::{shuffle, Rec, Rng, REC};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::time::Instant;

/// Default shuffle buffer (records; 32 bytes each in RAM).
pub const DEFAULT_BUFFER: usize = 16 << 20;
/// Chunk size for 32-byte `.bin` sources.
const RAW_CHUNK: u64 = 1 << 16;

/// A record source split into chunks that can be read independently.
pub trait ChunkSource: Send + Sync {
    fn name(&self) -> String;
    fn chunks(&self) -> usize;
    /// Records in chunk `i`.
    fn chunk_len(&self, i: usize) -> u64;
    /// Appends chunk `i`'s records, in source order, to `out`.
    fn read_chunk(&self, i: usize, out: &mut Vec<Rec>) -> Result<(), String>;
    fn positions(&self) -> u64 {
        (0..self.chunks()).map(|i| self.chunk_len(i)).sum()
    }
}

struct PackedSource {
    pf: PackedFile,
    file: File,
}

impl ChunkSource for PackedSource {
    fn name(&self) -> String {
        self.pf.path.display().to_string()
    }
    fn chunks(&self) -> usize {
        self.pf.chunks.len()
    }
    fn chunk_len(&self, i: usize) -> u64 {
        self.pf.chunks[i].n as u64
    }
    fn read_chunk(&self, i: usize, out: &mut Vec<Rec>) -> Result<(), String> {
        self.pf.read_chunk(&self.file, i, &mut Vec::new(), out)
    }
    fn positions(&self) -> u64 {
        self.pf.positions
    }
}

/// A 32-byte record file read in fixed chunks.
struct RawSource {
    path: PathBuf,
    file: File,
    n: u64,
}

impl ChunkSource for RawSource {
    fn name(&self) -> String {
        self.path.display().to_string()
    }
    fn chunks(&self) -> usize {
        self.n.div_ceil(RAW_CHUNK) as usize
    }
    fn chunk_len(&self, i: usize) -> u64 {
        RAW_CHUNK.min(self.n - i as u64 * RAW_CHUNK)
    }
    fn read_chunk(&self, i: usize, out: &mut Vec<Rec>) -> Result<(), String> {
        let n = self.chunk_len(i) as usize;
        let mut buf = vec![0u8; n * REC];
        read_at(&self.file, &mut buf, i as u64 * RAW_CHUNK * REC as u64).map_err(|e| format!("{}: {e}", self.path.display()))?;
        out.extend(buf.chunks_exact(REC).map(|c| -> Rec { c.try_into().unwrap() }));
        Ok(())
    }
}

/// Expands `--data` arguments: files, directories (their `*.rfp` and
/// `*.bin` files, sorted) and manifests (`*.txt` / `*.list`: one path per
/// line, relative to the manifest, `#` comments).
pub fn expand(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
                .map(|e| e.unwrap().path())
                .filter(|f| f.is_file() && matches!(f.extension().and_then(|e| e.to_str()), Some("rfp" | "bin")))
                .collect();
            v.sort();
            assert!(!v.is_empty(), "{}: no .rfp or .bin files", p.display());
            out.extend(v);
        } else if matches!(p.extension().and_then(|e| e.to_str()), Some("txt" | "list")) {
            let base = p.parent().unwrap_or(Path::new("."));
            let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let listed: Vec<PathBuf> = text
                .lines()
                .map(|l| l.split('#').next().unwrap().trim())
                .filter(|l| !l.is_empty())
                .map(|l| base.join(l))
                .collect();
            out.extend(expand(&listed));
        } else {
            out.push(p.clone());
        }
    }
    out
}

/// True if `data` needs the streaming loader (anything but plain 32-byte files).
pub fn wanted(data: &[PathBuf]) -> bool {
    data.iter().any(|p| p.is_dir() || matches!(p.extension().and_then(|e| e.to_str()), Some("txt" | "list")) || is_packed(p))
}

pub fn open_sources(paths: &[PathBuf]) -> Vec<Arc<dyn ChunkSource>> {
    expand(paths)
        .into_iter()
        .map(|p| -> Arc<dyn ChunkSource> {
            if is_packed(&p) {
                let pf = PackedFile::open(&p).unwrap_or_else(|e| panic!("{e}"));
                Arc::new(PackedSource { file: File::open(&p).unwrap(), pf })
            } else {
                let file = File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
                let len = file.metadata().unwrap().len();
                assert!(len % REC as u64 == 0, "{} is neither a packed nor a 32-byte record file", p.display());
                Arc::new(RawSource { path: p, file, n: len / REC as u64 })
            }
        })
        .collect()
}

/// Chunk visiting order of pass `pass`: every (source, chunk) once.
fn pass_order(sources: &[Arc<dyn ChunkSource>], seed: u64, pass: u64) -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = sources.iter().enumerate().flat_map(|(s, src)| (0..src.chunks()).map(move |c| (s as u32, c as u32))).collect();
    shuffle(&mut v, &mut Rng::new(seed ^ pass.wrapping_mul(0xD6E8_FEB8_6659_FD93)));
    v
}

/// Where decoded chunks come from: decoder threads (read round-robin, so
/// the order does not depend on timing) or the consumer itself.
enum Chunks {
    Threads { rxs: Vec<Receiver<Vec<Rec>>>, next: usize },
    Inline { sources: Arc<Vec<Arc<dyn ChunkSource>>>, seed: u64, pass: u64, order: Vec<(u32, u32)>, pos: usize },
}

fn read(sources: &[Arc<dyn ChunkSource>], (s, c): (u32, u32)) -> Vec<Rec> {
    let src = &sources[s as usize];
    let mut out = Vec::with_capacity(src.chunk_len(c as usize) as usize);
    src.read_chunk(c as usize, &mut out).unwrap_or_else(|e| panic!("{e}"));
    out
}

pub struct Stream {
    chunks: Chunks,
    cur: Vec<Rec>,
    cur_pos: usize,
    buf: Vec<Rec>,
    cap: usize,
    rng: Rng,
    pub positions: u64,
}

impl Stream {
    /// `cap` = shuffle buffer in records (0 = no shuffling, chunk order
    /// only); `threads` decoder threads, 0 = decode in the caller's thread.
    /// The stream is the same for any `threads`.
    pub fn new(sources: Vec<Arc<dyn ChunkSource>>, cap: usize, threads: usize, seed: u64) -> Self {
        let positions: u64 = sources.iter().map(|s| s.positions()).sum();
        assert!(positions > 0, "no training positions");
        let sources = Arc::new(sources);
        if threads == 0 {
            let chunks = Chunks::Inline { sources, seed, pass: 0, order: Vec::new(), pos: 0 };
            return Self { chunks, cur: Vec::new(), cur_pos: 0, buf: Vec::with_capacity(cap), cap, rng: Rng::new(seed ^ 0x5EED_B0F), positions };
        }
        let rxs = (0..threads)
            .map(|t| {
                let (tx, rx) = sync_channel::<Vec<Rec>>(2);
                let sources = Arc::clone(&sources);
                std::thread::spawn(move || {
                    // Thread t decodes items t, t + threads, ... of the
                    // sequence of all passes' chunk orders, concatenated.
                    let per_pass = sources.iter().map(|s| s.chunks()).sum::<usize>() as u64;
                    let (mut pass, mut order) = (u64::MAX, Vec::new());
                    for g in (t as u64..).step_by(threads) {
                        if g / per_pass != pass {
                            pass = g / per_pass;
                            order = pass_order(&sources, seed, pass);
                        }
                        if tx.send(read(&sources, order[(g % per_pass) as usize])).is_err() {
                            return;
                        }
                    }
                });
                rx
            })
            .collect();
        Self { chunks: Chunks::Threads { rxs, next: 0 }, cur: Vec::new(), cur_pos: 0, buf: Vec::with_capacity(cap), cap, rng: Rng::new(seed ^ 0x5EED_B0F), positions }
    }

    pub fn open(data: &[PathBuf], cap: usize, threads: usize, seed: u64) -> Self {
        Self::new(open_sources(data), cap, threads, seed)
    }

    #[inline]
    fn refill(&mut self) {
        while self.cur_pos == self.cur.len() {
            self.cur = match &mut self.chunks {
                Chunks::Threads { rxs, next } => {
                    let c = rxs[*next].recv().expect("data loader thread died");
                    *next = (*next + 1) % rxs.len();
                    c
                }
                Chunks::Inline { sources, seed, pass, order, pos } => {
                    if *pos == order.len() {
                        *order = pass_order(sources, *seed, *pass);
                        *pass += 1;
                        *pos = 0;
                    }
                    *pos += 1;
                    read(sources, order[*pos - 1])
                }
            };
            self.cur_pos = 0;
        }
    }

    pub fn batch(&mut self, out: &mut Vec<Rec>, n: usize) {
        out.clear();
        while out.len() < n {
            self.refill();
            if self.buf.len() < self.cap {
                let take = (self.cap - self.buf.len()).min(self.cur.len() - self.cur_pos);
                self.buf.extend_from_slice(&self.cur[self.cur_pos..self.cur_pos + take]);
                self.cur_pos += take;
                continue;
            }
            let take = (n - out.len()).min(self.cur.len() - self.cur_pos);
            if self.cap == 0 {
                out.extend_from_slice(&self.cur[self.cur_pos..self.cur_pos + take]);
            } else {
                for r in &self.cur[self.cur_pos..self.cur_pos + take] {
                    let j = self.rng.below(self.cap);
                    out.push(std::mem::replace(&mut self.buf[j], *r));
                }
            }
            self.cur_pos += take;
        }
    }
}

impl crate::Feeder for Stream {
    fn batch(&mut self, out: &mut Vec<Rec>, n: usize) {
        Stream::batch(self, out, n)
    }
}

/// `shuffle-check --data <..>... [--shuffle-buffer N] [--batch 16384]
/// [--batches 1000] [--threads 1] [--seed 1] [--skip-batches 0]`: loader
/// throughput, and how close batches are to i.i.d.: the design effect
/// `var(batch mean) * batch / var(sample)` of the White-POV result and of
/// the trainer's target (lambda 0.5); 1 = i.i.d., a fully shuffled pool
/// measures ~1. Plain 32-byte files without `--shuffle-buffer` go through
/// the pool feed (`train`'s default path) for reference.
pub fn cmd_shuffle_check(args: &[String]) {
    let (mut data, mut cap, mut batch, mut batches, mut threads, mut seed, mut skip) =
        (Vec::new(), None::<usize>, 16384usize, 1000u64, 1usize, 1u64, 0u64);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("missing value for {a}")).clone();
        match a.as_str() {
            "--data" => data.push(PathBuf::from(v())),
            "--shuffle-buffer" => cap = Some(v().parse().unwrap()),
            "--batch" => batch = v().parse().unwrap(),
            "--batches" => batches = v().parse().unwrap(),
            "--threads" => threads = v().parse().unwrap(),
            "--seed" => seed = v().parse().unwrap(),
            "--skip-batches" => skip = v().parse().unwrap(),
            other => panic!("unknown option {other}"),
        }
    }
    let t0 = Instant::now();
    let mut feed: Box<dyn crate::Feeder> = if cap.is_none() && !wanted(&data) {
        Box::new(crate::Feed::new(data, seed))
    } else {
        Box::new(Stream::open(&data, cap.unwrap_or(DEFAULT_BUFFER), threads, seed))
    };
    let mut recs = Vec::with_capacity(batch);
    for _ in 0..skip {
        feed.batch(&mut recs, batch);
    }
    let t1 = Instant::now();
    let (mut s1, mut s2, mut m1, mut m2) = ([0f64; 2], [0f64; 2], [0f64; 2], [0f64; 2]);
    for _ in 0..batches {
        feed.batch(&mut recs, batch);
        let mut bm = [0f64; 2];
        for r in &recs {
            let res = r[26] as f64 / 2.0;
            let sc = i16::from_le_bytes([r[24], r[25]]) as f64;
            let (wdl, sc) = if r[27] == 1 { (1.0 - res, -sc) } else { (res, sc) };
            let tgt = 0.5 * wdl + 0.5 / (1.0 + (-sc / 400.0).exp());
            for (i, x) in [res, tgt].into_iter().enumerate() {
                bm[i] += x;
                s1[i] += x;
                s2[i] += x * x;
            }
        }
        for i in 0..2 {
            let m = bm[i] / batch as f64;
            m1[i] += m;
            m2[i] += m * m;
        }
    }
    let secs = t1.elapsed().as_secs_f64();
    let n = (batches * batch as u64) as f64;
    let b = batches as f64;
    let de: Vec<f64> = (0..2)
        .map(|i| {
            let var_pop = s2[i] / n - (s1[i] / n).powi(2);
            let var_bm = (m2[i] - m1[i] * m1[i] / b) / (b - 1.0);
            var_bm * batch as f64 / var_pop
        })
        .collect();
    println!(
        "shuffle-check buffer={} threads={threads} batch={batch} batches={batches}: design effect result {:.3} target {:.3} \
         (i.i.d. = 1 +- {:.3}); {:.2}M positions/s after a {:.1}s start",
        cap.map_or("pool-feed".into(), |c| c.to_string()),
        de[0],
        de[1],
        (2.0 / (b - 1.0)).sqrt(),
        n / secs / 1e6,
        (t1 - t0).as_secs_f64()
    );
}

/// `stream --data <..>... [--shuffle-buffer N] [--threads 1] [--seed 1]`:
/// writes the shuffled stream as 32-byte records to stdout until the reader
/// closes it. For trainers with their own pool reader (Metal): `mkfifo f;
/// nnue-trainer stream --data dir > f & <trainer> train --data f
/// --epoch-size <positions>` (the positions count is printed on stderr).
pub fn cmd_stream(args: &[String]) {
    use std::io::Write;
    let (mut data, mut cap, mut threads, mut seed) = (Vec::new(), DEFAULT_BUFFER, 1usize, 1u64);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("missing value for {a}")).clone();
        match a.as_str() {
            "--data" => data.push(PathBuf::from(v())),
            "--shuffle-buffer" => cap = v().parse().unwrap(),
            "--threads" => threads = v().parse().unwrap(),
            "--seed" => seed = v().parse().unwrap(),
            other => panic!("unknown option {other}"),
        }
    }
    let mut s = Stream::open(&data, cap, threads, seed);
    eprintln!("stream: {} positions per pass, shuffle buffer {cap}", s.positions);
    let mut out = std::io::BufWriter::with_capacity(1 << 22, std::io::stdout().lock());
    let mut recs = Vec::new();
    loop {
        s.batch(&mut recs, 1 << 16);
        if recs.iter().try_for_each(|r| out.write_all(r)).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::Writer;

    struct Mem(Vec<Vec<Rec>>);
    impl ChunkSource for Mem {
        fn name(&self) -> String {
            "mem".into()
        }
        fn chunks(&self) -> usize {
            self.0.len()
        }
        fn chunk_len(&self, i: usize) -> u64 {
            self.0[i].len() as u64
        }
        fn read_chunk(&self, i: usize, out: &mut Vec<Rec>) -> Result<(), String> {
            out.extend_from_slice(&self.0[i]);
            Ok(())
        }
    }

    fn rec(i: u32) -> Rec {
        let mut r = [0u8; REC];
        r[..4].copy_from_slice(&i.to_le_bytes());
        r
    }

    #[test]
    fn every_record_once_per_pass_and_deterministic() {
        let total = 10_000u32;
        let chunks: Vec<Vec<Rec>> = (0..total).collect::<Vec<_>>().chunks(333).map(|c| c.iter().map(|&i| rec(i)).collect()).collect();
        let src = || -> Vec<Arc<dyn ChunkSource>> { vec![Arc::new(Mem(chunks.clone()))] };
        for (cap, threads) in [(0, 1), (0, 0), (1000, 1), (1000, 3), (1000, 0), (7_000, 2)] {
            let mut s = Stream::new(src(), cap, threads, 9);
            let mut out = Vec::new();
            let mut got = Vec::new();
            for _ in 0..3 * total as usize / 100 {
                s.batch(&mut out, 100);
                got.extend(out.iter().map(|r| u32::from_le_bytes(r[..4].try_into().unwrap())));
            }
            // Emitted + buffered + not yet consumed = whole passes read so far,
            // the last one partial: every id is counted p or p + 1 times.
            let id = |r: &Rec| u32::from_le_bytes(r[..4].try_into().unwrap()) as usize;
            let mut seen = vec![0usize; total as usize];
            got.iter().for_each(|&g| seen[g as usize] += 1);
            s.buf.iter().chain(&s.cur[s.cur_pos..]).for_each(|r| seen[id(r)] += 1);
            let read: usize = seen.iter().sum();
            let p = read / total as usize;
            assert!(seen.iter().all(|&c| c == p || c == p + 1), "cap {cap}");
            if cap == 0 {
                assert!(seen.iter().all(|&c| c == 3));
            } else {
                // Shuffled: emitted order is far from the input order.
                assert!(got.windows(2).filter(|w| w[1] == w[0] + 1).count() < got.len() / 10);
            }
            let mut s2 = Stream::new(src(), cap, threads, 9);
            let mut got2 = Vec::new();
            for _ in 0..3 * total as usize / 100 {
                s2.batch(&mut out, 100);
                got2.extend(out.iter().map(|r| u32::from_le_bytes(r[..4].try_into().unwrap())));
            }
            assert_eq!(got, got2, "cap {cap} threads {threads}");
            let mut s4 = Stream::new(src(), cap, if threads == 0 { 2 } else { 0 }, 9);
            s4.batch(&mut out, 3 * total as usize);
            let got4: Vec<u32> = out.iter().map(|r| u32::from_le_bytes(r[..4].try_into().unwrap())).collect();
            let first = got.iter().zip(&got4).position(|(a, b)| a != b);
            assert!(first.is_none() && got.len() == got4.len(), "cap {cap} threads {threads}: thread count changed the stream at {first:?} ({} vs {})", got.len(), got4.len());
            let mut s3 = Stream::new(src(), cap, threads, 10);
            s3.batch(&mut out, 100);
            assert_ne!(&got[..100], &out.iter().map(|r| u32::from_le_bytes(r[..4].try_into().unwrap())).collect::<Vec<_>>()[..]);
        }
    }

    #[test]
    fn packed_and_raw_files_stream_the_same_records() {
        let dir = std::env::temp_dir().join(format!("rfp-stream-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lines = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1 | 30 | 1.0",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2 | 24 | 1.0",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2 | 36 | 0.5",
        ];
        let recs: Vec<Rec> = (0..3000).map(|i| crate::encode(lines[i % 3]).unwrap()).collect();
        let raw: Vec<u8> = recs.iter().flatten().copied().collect();
        std::fs::write(dir.join("a.bin"), &raw).unwrap();
        let mut w = Writer::new(File::create(dir.join("b.rfp")).unwrap(), 100, true);
        recs.iter().for_each(|r| w.push(r));
        w.finish();
        std::fs::write(dir.join("m.list"), "# both\na.bin\nb.rfp\n").unwrap();
        assert!(wanted(&[dir.join("m.list")]) && wanted(&[dir.clone()]) && !wanted(&[dir.join("a.bin")]));
        assert_eq!(expand(&[dir.join("m.list")]), expand(&[dir.clone()]));
        let mut s = Stream::open(&[dir.clone()], 0, 2, 3);
        assert_eq!(s.positions, 6000);
        let mut out = Vec::new();
        s.batch(&mut out, 6000);
        let mut a = out.clone();
        a.sort();
        let mut b: Vec<Rec> = recs.iter().chain(&recs).copied().collect();
        b.sort();
        assert_eq!(a, b);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
