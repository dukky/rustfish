//! Packed training data (`.rfp`): a lossless, chained encoding of the
//! 32-byte records (`crate::Rec`), ~3 bytes per position on game-ordered
//! self-play data. Format and measurements: `docs/data-format.md`.
//!
//! A file is `MAGIC` followed by independent chunks. A chunk is a 20-byte
//! header (`RFCK`, positions, payload bytes, CRC-32 of the payload, score
//! unit u16, Exp-Golomb order u8, reserved u8) and an LSB-first bit stream
//! (whole little-endian u64 words). Each record is either
//!
//! * a keyframe (`11`): occupancy (64 bits), one 4-bit piece per occupied
//!   square in square order, side to move (1), result (2), score (16, raw
//!   i16), or
//! * a delta from the previous record of the chunk, `0` if the side to move
//!   changed, `10` if not: board edits, then the score as a signed
//!   Exp-Golomb code of `(score - previous score) / unit`; the result is
//!   inherited.
//!
//! Board edits are a list ended by `END`: `0` MOVE (index of the moving
//! piece among the occupied squares, `ceil(log2(pieces))` bits, then the
//! target square, 6 bits; a piece on the target is replaced), `10` END,
//! `110` SET (square 6 bits, piece 4 bits), `111` CLEAR (index of the
//! square among the occupied ones). No chess rules are needed: any two
//! records diff exactly, so skipped plies (captures, checks) are just more
//! edits and a shuffled input still round-trips (as keyframes). The encoder
//! picks keyframe or delta by cost; a chunk always starts with a keyframe,
//! so chunks decode independently (the unit of the streaming shuffle).

use crate::{encode, Rec, REC};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 8] = b"RFPACK1\0";
const CHUNK_MAGIC: &[u8; 4] = b"RFCK";
const HEADER: usize = 20;
/// Default positions per chunk (~200 KB packed, ~800 games).
pub const CHUNK_TARGET: usize = 1 << 16;
const EMPTY: u8 = 0xFF;
const MAX_K: u32 = 15;

// ---------------------------------------------------------------- crc32

fn crc_table() -> &'static [u32; 256] {
    static T: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    })
}

pub fn crc32(data: &[u8]) -> u32 {
    let t = crc_table();
    !data.iter().fold(!0u32, |c, &b| t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8))
}

// ---------------------------------------------------------------- bits

#[derive(Default)]
struct BitWriter {
    words: Vec<u64>,
    cur: u64,
    n: u32,
}

impl BitWriter {
    /// Appends the low `bits` (<= 32) bits of `v`.
    #[inline]
    fn put(&mut self, v: u64, bits: u32) {
        debug_assert!(bits <= 32 && (bits == 32 || v >> bits == 0));
        if bits == 0 {
            return;
        }
        self.cur |= v << self.n;
        self.n += bits;
        if self.n >= 64 {
            self.words.push(self.cur);
            self.n -= 64;
            self.cur = if self.n == 0 { 0 } else { v >> (bits - self.n) };
        }
    }
    fn bytes(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.words.push(self.cur);
        }
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

struct BitReader<'a> {
    words: &'a [u64],
    pos: usize,
}

impl BitReader<'_> {
    /// The next 64 bits (zeros past the end).
    #[inline]
    fn peek(&self) -> u64 {
        let (i, o) = (self.pos >> 6, self.pos & 63);
        let lo = self.words.get(i).copied().unwrap_or(0) >> o;
        if o == 0 {
            lo
        } else {
            lo | self.words.get(i + 1).copied().unwrap_or(0) << (64 - o)
        }
    }
    #[inline]
    fn get(&mut self, bits: u32) -> u64 {
        let v = if bits == 0 { 0 } else { self.peek() & (u64::MAX >> (64 - bits)) };
        self.pos += bits as usize;
        v
    }
    fn past_end(&self) -> bool {
        self.pos > self.words.len() * 64
    }
}

#[inline]
fn idx_bits(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

#[inline]
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
fn unzigzag(u: u64) -> i64 {
    (u >> 1) as i64 ^ -((u & 1) as i64)
}

/// Bits of the order-`k` Exp-Golomb code of `u`.
#[inline]
fn eg_bits(u: u64, k: u32) -> u32 {
    let x = u + (1 << k);
    let nb = 64 - x.leading_zeros();
    2 * nb - 1 - k
}

fn eg_put(w: &mut BitWriter, u: u64, k: u32) {
    let x = u + (1 << k);
    let nb = 64 - x.leading_zeros();
    let q = nb - 1 - k;
    // q zeros, a one, then the low nb - 1 bits of x.
    let mut zeros = q;
    while zeros > 32 {
        w.put(0, 32);
        zeros -= 32;
    }
    w.put(0, zeros);
    w.put(1, 1);
    let mut low = nb - 1;
    let mut v = x & ((1u64 << low) - 1);
    while low > 32 {
        w.put(v & 0xFFFF_FFFF, 32);
        v >>= 32;
        low -= 32;
    }
    w.put(v, low);
}

#[inline]
fn eg_get(r: &mut BitReader, k: u32) -> Option<u64> {
    let q = r.peek().trailing_zeros();
    if q > 40 {
        return None;
    }
    r.pos += q as usize + 1;
    let low = q + k;
    let lo = if low > 32 { r.get(32) | r.get(low - 32) << 32 } else { r.get(low) };
    Some(((1u64 << low) | lo) - (1 << k))
}

// ---------------------------------------------------------------- positions

/// A record unpacked to a mailbox (piece nibble or `EMPTY` per square).
#[derive(Clone, Copy, PartialEq, Debug)]
struct Pos {
    board: [u8; 64],
    occ: u64,
    stm: u8,
    result: u8,
    score: i32,
}

impl Pos {
    fn from_rec(r: &Rec) -> Self {
        let occ = u64::from_le_bytes(r[..8].try_into().unwrap());
        let mut board = [EMPTY; 64];
        let (mut o, mut n) = (occ, 0);
        while o != 0 {
            let sq = o.trailing_zeros() as usize;
            o &= o - 1;
            board[sq] = r[8 + n / 2] >> (4 * (n & 1)) & 0xF;
            n += 1;
        }
        Pos { board, occ, stm: r[27], result: r[26], score: i16::from_le_bytes([r[24], r[25]]) as i32 }
    }

    #[inline]
    fn to_rec(&self) -> Rec {
        let mut r = [0u8; REC];
        r[..8].copy_from_slice(&self.occ.to_le_bytes());
        let (mut o, mut n) = (self.occ, 0);
        while o != 0 {
            let sq = o.trailing_zeros() as usize;
            o &= o - 1;
            r[8 + n / 2] |= self.board[sq] << (4 * (n & 1));
            n += 1;
        }
        r[24..26].copy_from_slice(&(self.score as i16).to_le_bytes());
        r[26] = self.result;
        r[27] = self.stm;
        r
    }
}

/// Index of square `sq` among the set bits of `occ`.
#[inline]
fn index_of(occ: u64, sq: usize) -> u32 {
    (occ & ((1u64 << sq) - 1)).count_ones()
}

/// The square of the `i`-th set bit of `occ`.
#[inline]
fn nth_square(mut occ: u64, i: u32) -> Option<usize> {
    for _ in 0..i {
        occ &= occ.wrapping_sub(1);
    }
    (occ != 0).then(|| occ.trailing_zeros() as usize)
}

/// Records the encoder can store: the format of `crate::encode`.
fn valid(r: &Rec) -> bool {
    let occ = u64::from_le_bytes(r[..8].try_into().unwrap());
    let n = occ.count_ones() as usize;
    if n > 32 || r[26] > 2 || r[27] > 1 || r[28..].iter().any(|&b| b != 0) {
        return false;
    }
    // Unused nibbles must be zero and pieces must be valid nibbles.
    (0..32).all(|i| {
        let nib = r[8 + i / 2] >> (4 * (i & 1)) & 0xF;
        if i < n {
            nib & 7 <= 5
        } else {
            nib == 0
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Move(u8, u8),
    Set(u8, u8),
    Clear(u8),
}

/// Board edits that turn `prev` into `next` (pieces only).
fn diff(prev: &Pos, next: &Pos, ops: &mut Vec<Op>) {
    ops.clear();
    let mut w = prev.board;
    let target = &next.board;
    // Moves: a piece on a square it has to leave goes to a square that needs
    // that piece; first onto empty squares, so nothing still needed is
    // overwritten, then any.
    loop {
        let mut progressed = false;
        for allow_occupied in [false, true] {
            for t in 0..64 {
                let want = target[t];
                if want == EMPTY || w[t] == want || (!allow_occupied && w[t] != EMPTY) {
                    continue;
                }
                if let Some(s) = (0..64).find(|&s| w[s] == want && target[s] != want) {
                    ops.push(Op::Move(s as u8, t as u8));
                    w[t] = want;
                    w[s] = EMPTY;
                    progressed = true;
                    break;
                }
            }
            if progressed {
                break;
            }
        }
        if !progressed {
            break;
        }
    }
    for s in 0..64 {
        if w[s] != EMPTY && target[s] == EMPTY {
            ops.push(Op::Clear(s as u8));
            w[s] = EMPTY;
        }
    }
    for t in 0..64 {
        if w[t] != target[t] {
            ops.push(Op::Set(t as u8, target[t]));
            w[t] = target[t];
        }
    }
}

/// Bits of the board edits (with END), applied to `occ`.
fn ops_bits(mut occ: u64, ops: &[Op]) -> u32 {
    let mut bits = 2; // END
    for op in ops {
        match *op {
            Op::Move(s, t) => {
                bits += 1 + idx_bits(occ.count_ones()) + 6;
                occ = (occ & !(1 << s)) | 1 << t;
            }
            Op::Set(t, _) => {
                bits += 3 + 10;
                occ |= 1 << t;
            }
            Op::Clear(s) => {
                bits += 3 + idx_bits(occ.count_ones());
                occ &= !(1 << s);
            }
        }
    }
    bits
}

fn key_bits(p: &Pos) -> u32 {
    2 + 64 + 4 * p.occ.count_ones() + 1 + 2 + 16
}

// ---------------------------------------------------------------- encoder

/// Encoding statistics.
#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub positions: u64,
    pub chunks: u64,
    pub keyframes: u64,
    pub bytes: u64,
    pub move_ops: u64,
    pub set_ops: u64,
    pub clear_ops: u64,
    pub key_bits: u64,
    pub edit_bits: u64,
    pub score_bits: u64,
    pub skipped: u64,
}

impl Stats {
    pub fn report(&self) -> String {
        let n = self.positions.max(1) as f64;
        let d = (self.positions - self.keyframes).max(1) as f64;
        format!(
            "{} positions in {} chunks, {} bytes = {:.3} B/position; keyframes {:.2}% ({:.3} B/pos); \
             deltas: edits {:.2} bits ({:.3} moves, {:.4} sets, {:.4} clears), score {:.2} bits, +header",
            self.positions,
            self.chunks,
            self.bytes,
            self.bytes as f64 / n,
            100.0 * self.keyframes as f64 / n,
            self.key_bits as f64 / 8.0 / n,
            self.edit_bits as f64 / d,
            self.move_ops as f64 / d,
            self.set_ops as f64 / d,
            self.clear_ops as f64 / d,
            self.score_bits as f64 / d,
        )
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Encodes one chunk; `k_hint` is the Exp-Golomb order used for keyframe
/// decisions (the chunk's own best order is stored).
fn encode_chunk(recs: &[Rec], k_hint: u32, st: &mut Stats) -> (Vec<u8>, u32) {
    let unit = recs.iter().fold(0u32, |g, r| gcd(g, i16::from_le_bytes([r[24], r[25]]).unsigned_abs() as u32)).clamp(1, 0xFFFF);
    // Pass 1: keyframe/delta decisions and the score codes.
    let mut plan: Vec<Option<(Vec<Op>, u64)>> = Vec::with_capacity(recs.len());
    let mut ops = Vec::new();
    let mut prev: Option<Pos> = None;
    for r in recs {
        let p = Pos::from_rec(r);
        let mut choice = None;
        if let Some(q) = prev.filter(|q| q.result == p.result) {
            diff(&q, &p, &mut ops);
            let u = zigzag((p.score - q.score) as i64 / unit as i64);
            let cost = if p.stm != q.stm { 1 } else { 2 } + ops_bits(q.occ, &ops) + eg_bits(u, k_hint);
            if cost < key_bits(&p) {
                choice = Some((ops.clone(), u));
            }
        }
        plan.push(choice);
        prev = Some(p);
    }
    let k = (0..=MAX_K)
        .min_by_key(|&k| plan.iter().flatten().map(|(_, u)| eg_bits(*u, k) as u64).sum::<u64>())
        .unwrap();
    // Pass 2: the bit stream.
    let mut w = BitWriter::default();
    let (mut prev_occ, mut prev_stm) = (0u64, 0u8);
    for (r, choice) in recs.iter().zip(&plan) {
        let p = Pos::from_rec(r);
        match choice {
            None => {
                w.put(0b11, 2);
                w.put(p.occ & 0xFFFF_FFFF, 32);
                w.put(p.occ >> 32, 32);
                let (mut o, mut n) = (p.occ, 0);
                while o != 0 {
                    w.put(p.board[o.trailing_zeros() as usize] as u64, 4);
                    o &= o - 1;
                    n += 1;
                }
                w.put(p.stm as u64, 1);
                w.put(p.result as u64, 2);
                w.put(p.score as i16 as u16 as u64, 16);
                st.keyframes += 1;
                st.key_bits += 2 + 64 + 4 * n + 19;
            }
            Some((ops, u)) => {
                if p.stm != prev_stm {
                    w.put(0, 1);
                } else {
                    w.put(0b01, 2);
                }
                let mut occ = prev_occ;
                let start = w.words.len() as u64 * 64 + w.n as u64;
                for op in ops {
                    match *op {
                        Op::Move(s, t) => {
                            w.put(0, 1);
                            w.put(index_of(occ, s as usize) as u64, idx_bits(occ.count_ones()));
                            w.put(t as u64, 6);
                            occ = (occ & !(1 << s)) | 1 << t;
                            st.move_ops += 1;
                        }
                        Op::Set(t, piece) => {
                            w.put(0b011, 3);
                            w.put(t as u64, 6);
                            w.put(piece as u64, 4);
                            occ |= 1 << t;
                            st.set_ops += 1;
                        }
                        Op::Clear(s) => {
                            w.put(0b111, 3);
                            w.put(index_of(occ, s as usize) as u64, idx_bits(occ.count_ones()));
                            occ &= !(1 << s);
                            st.clear_ops += 1;
                        }
                    }
                }
                w.put(0b01, 2);
                debug_assert_eq!(occ, p.occ);
                let mid = w.words.len() as u64 * 64 + w.n as u64;
                eg_put(&mut w, *u, k);
                let end = w.words.len() as u64 * 64 + w.n as u64;
                st.edit_bits += mid - start;
                st.score_bits += end - mid;
            }
        }
        prev_occ = p.occ;
        prev_stm = p.stm;
    }
    st.positions += recs.len() as u64;
    st.chunks += 1;
    let payload = w.bytes();
    let mut out = Vec::with_capacity(HEADER + payload.len());
    out.extend_from_slice(CHUNK_MAGIC);
    out.extend_from_slice(&(recs.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&crc32(&payload).to_le_bytes());
    out.extend_from_slice(&(unit as u16).to_le_bytes());
    out.push(k as u8);
    out.push(0);
    out.extend_from_slice(&payload);
    st.bytes += out.len() as u64;
    (out, k)
}

/// Streams records into a packed file. Chunks end at a keyframe (a new
/// game) once `target` positions are buffered, or at `2 * target`.
pub struct Writer<W: Write> {
    out: W,
    pending: Vec<Rec>,
    target: usize,
    k: u32,
    verify: bool,
    pub stats: Stats,
}

impl<W: Write> Writer<W> {
    pub fn new(mut out: W, target: usize, verify: bool) -> Self {
        out.write_all(MAGIC).unwrap();
        let mut stats = Stats::default();
        stats.bytes = MAGIC.len() as u64;
        Self { out, pending: Vec::with_capacity(2 * target), target: target.max(1), k: 4, verify, stats }
    }

    /// Adds a record (skipped and counted if it is not a valid record).
    pub fn push(&mut self, r: &Rec) {
        if !valid(r) {
            self.stats.skipped += 1;
            return;
        }
        if self.pending.len() >= 2 * self.target
            || (self.pending.len() >= self.target && starts_game(self.pending.last().unwrap(), r))
        {
            self.flush_chunk();
        }
        self.pending.push(*r);
    }

    fn flush_chunk(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let (bytes, k) = encode_chunk(&self.pending, self.k, &mut self.stats);
        self.k = k;
        if self.verify {
            let mut back = Vec::new();
            let h = ChunkHeader::parse(&bytes[..HEADER]).expect("chunk header");
            decode_chunk(&h, &bytes[HEADER..], &mut back).expect("decode of a fresh chunk");
            assert!(back == self.pending, "packed round trip mismatch");
        }
        self.out.write_all(&bytes).unwrap();
        self.pending.clear();
    }

    pub fn finish(mut self) -> Stats {
        self.flush_chunk();
        self.out.flush().unwrap();
        self.stats.clone()
    }
}

/// A new game (or unrelated record) starts at `r`: a delta would not pay.
fn starts_game(prev: &Rec, r: &Rec) -> bool {
    let (q, p) = (Pos::from_rec(prev), Pos::from_rec(r));
    if q.result != p.result {
        return true;
    }
    let mut ops = Vec::new();
    diff(&q, &p, &mut ops);
    2 + ops_bits(q.occ, &ops) + 12 >= key_bits(&p)
}

// ---------------------------------------------------------------- decoder

#[derive(Clone, Copy, Debug)]
pub struct ChunkHeader {
    pub n: u32,
    pub len: u32,
    pub crc: u32,
    pub unit: u16,
    pub k: u8,
}

impl ChunkHeader {
    pub fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER || &b[..4] != CHUNK_MAGIC {
            return Err("bad chunk magic".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let h = ChunkHeader { n: u32_at(4), len: u32_at(8), crc: u32_at(12), unit: u16::from_le_bytes([b[16], b[17]]), k: b[18] };
        if h.len % 8 != 0 || h.k as u32 > MAX_K || h.unit == 0 {
            return Err("bad chunk header".into());
        }
        Ok(h)
    }
}

/// Decodes a chunk's payload, appending its records to `out`.
pub fn decode_chunk(h: &ChunkHeader, payload: &[u8], out: &mut Vec<Rec>) -> Result<(), String> {
    if payload.len() != h.len as usize {
        return Err("truncated chunk".into());
    }
    if crc32(payload) != h.crc {
        return Err("chunk CRC mismatch".into());
    }
    let words: Vec<u64> = payload.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    let mut r = BitReader { words: &words, pos: 0 };
    let (unit, k) = (h.unit as i32, h.k as u32);
    let mut p = Pos { board: [EMPTY; 64], occ: 0, stm: 0, result: 0, score: 0 };
    out.reserve(h.n as usize);
    let bad = || "corrupt chunk".to_string();
    for i in 0..h.n {
        let hdr = r.peek() & 3;
        if hdr == 3 {
            r.pos += 2;
            let occ = r.get(32) | r.get(32) << 32;
            if occ.count_ones() > 32 {
                return Err(bad());
            }
            p.board = [EMPTY; 64];
            let mut o = occ;
            while o != 0 {
                p.board[o.trailing_zeros() as usize] = r.get(4) as u8;
                o &= o - 1;
            }
            p.occ = occ;
            p.stm = r.get(1) as u8;
            p.result = r.get(2) as u8;
            p.score = r.get(16) as u16 as i16 as i32;
        } else {
            if i == 0 {
                return Err(bad());
            }
            if hdr & 1 == 0 {
                r.pos += 1;
                p.stm ^= 1;
            } else {
                r.pos += 2;
            }
            loop {
                let w = r.peek();
                if w & 1 == 0 {
                    // MOVE
                    r.pos += 1;
                    let ib = idx_bits(p.occ.count_ones());
                    let s = nth_square(p.occ, r.get(ib) as u32).ok_or_else(bad)?;
                    let t = r.get(6) as usize;
                    if s == t {
                        return Err(bad());
                    }
                    p.board[t] = p.board[s];
                    p.board[s] = EMPTY;
                    p.occ = (p.occ & !(1 << s)) | 1 << t;
                } else if w & 2 == 0 {
                    r.pos += 2; // END
                    break;
                } else if w & 4 == 0 {
                    r.pos += 3; // SET
                    let t = r.get(6) as usize;
                    p.board[t] = r.get(4) as u8;
                    p.occ |= 1 << t;
                } else {
                    r.pos += 3; // CLEAR
                    let ib = idx_bits(p.occ.count_ones());
                    let s = nth_square(p.occ, r.get(ib) as u32).ok_or_else(bad)?;
                    p.board[s] = EMPTY;
                    p.occ &= !(1 << s);
                }
                if r.past_end() {
                    return Err(bad());
                }
            }
            let d = unzigzag(eg_get(&mut r, k).ok_or_else(bad)?);
            p.score += (d as i32).wrapping_mul(unit);
        }
        if r.past_end() || p.occ.count_ones() > 32 || p.score < i16::MIN as i32 || p.score > i16::MAX as i32 {
            return Err(bad());
        }
        out.push(p.to_rec());
    }
    Ok(())
}

/// A chunk's place in a packed file.
#[derive(Clone, Copy, Debug)]
pub struct ChunkInfo {
    /// Offset of the chunk header.
    pub offset: u64,
    pub n: u32,
    pub len: u32,
}

/// The chunk index of a packed file (built by walking the chunk headers).
pub struct PackedFile {
    pub path: PathBuf,
    pub chunks: Vec<ChunkInfo>,
    pub positions: u64,
}

pub fn is_packed(path: &Path) -> bool {
    let mut m = [0u8; 8];
    File::open(path).and_then(|mut f| f.read_exact(&mut m)).is_ok() && &m == MAGIC
}

impl PackedFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut f = File::open(path).map_err(err)?;
        let size = f.metadata().map_err(err)?.len();
        let mut m = [0u8; 8];
        f.read_exact(&mut m).map_err(err)?;
        if &m != MAGIC {
            return Err(format!("{}: not a packed file", path.display()));
        }
        let (mut chunks, mut positions, mut off) = (Vec::new(), 0u64, MAGIC.len() as u64);
        let mut hb = [0u8; HEADER];
        while off < size {
            read_at(&f, &mut hb, off).map_err(err)?;
            let h = ChunkHeader::parse(&hb).map_err(|e| format!("{} @ {off}: {e}", path.display()))?;
            if off + (HEADER as u64) + h.len as u64 > size {
                return Err(format!("{}: truncated chunk at {off}", path.display()));
            }
            chunks.push(ChunkInfo { offset: off, n: h.n, len: h.len });
            positions += h.n as u64;
            off += HEADER as u64 + h.len as u64;
        }
        Ok(Self { path: path.to_path_buf(), chunks, positions })
    }

    /// Reads and decodes chunk `i` into `out` (appended); `buf` is scratch.
    pub fn read_chunk(&self, f: &File, i: usize, buf: &mut Vec<u8>, out: &mut Vec<Rec>) -> Result<(), String> {
        let c = self.chunks[i];
        buf.resize(HEADER + c.len as usize, 0);
        read_at(f, buf, c.offset).map_err(|e| format!("{}: {e}", self.path.display()))?;
        let h = ChunkHeader::parse(&buf[..HEADER])?;
        decode_chunk(&h, &buf[HEADER..], out).map_err(|e| format!("{} chunk {i}: {e}", self.path.display()))
    }
}

#[cfg(unix)]
pub(crate) fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, off)
}

#[cfg(windows)]
pub(crate) fn read_at(f: &File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

// ---------------------------------------------------------------- commands

/// Calls `f` for every record of a 32-byte file, a packed file or datagen
/// text (`-` = stdin), in file order.
pub fn for_each_record(path: &str, mut f: impl FnMut(&Rec)) -> u64 {
    let mut bad = 0;
    if path != "-" && is_packed(Path::new(path)) {
        let pf = PackedFile::open(Path::new(path)).unwrap_or_else(|e| panic!("{e}"));
        let file = File::open(path).unwrap();
        let (mut buf, mut recs) = (Vec::new(), Vec::new());
        for i in 0..pf.chunks.len() {
            recs.clear();
            pf.read_chunk(&file, i, &mut buf, &mut recs).unwrap_or_else(|e| panic!("{e}"));
            recs.iter().for_each(&mut f);
        }
    } else if path.ends_with(".bin") {
        let mut r = BufReader::with_capacity(1 << 22, File::open(path).unwrap_or_else(|e| panic!("{path}: {e}")));
        let mut rec = [0u8; REC];
        while r.read_exact(&mut rec).is_ok() {
            f(&rec);
        }
    } else {
        let input: Box<dyn BufRead> = if path == "-" {
            Box::new(BufReader::with_capacity(1 << 22, std::io::stdin()))
        } else {
            Box::new(BufReader::with_capacity(1 << 22, File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"))))
        };
        for line in input.lines() {
            match encode(&line.unwrap()) {
                Some(r) => f(&r),
                None => bad += 1,
            }
        }
    }
    bad
}

/// `pack [--chunk N] [--no-verify] <out.rfp> <in>...`: inputs are 32-byte
/// `.bin` files (game-ordered sources pack to ~3 B/position, shuffled pools
/// still round-trip but stay large), datagen text, `-` (stdin) or packed
/// files, concatenated in the order given.
pub fn cmd_pack(args: &[String]) {
    let (mut target, mut verify, mut files) = (CHUNK_TARGET, true, Vec::new());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--chunk" => target = it.next().expect("--chunk N").parse().unwrap(),
            "--no-verify" => verify = false,
            _ => files.push(a.clone()),
        }
    }
    assert!(files.len() >= 2, "usage: pack [--chunk N] [--no-verify] <out.rfp> <in.bin|in.txt|-|in.rfp>...");
    let out = &files[0];
    let t = std::time::Instant::now();
    let mut w = Writer::new(BufWriter::with_capacity(1 << 22, File::create(out).unwrap()), target, verify);
    let mut bad = 0;
    for inp in &files[1..] {
        bad += for_each_record(inp, |r| w.push(r));
    }
    let st = w.finish();
    eprintln!("{out}: {} ({} unparsed lines, {} invalid records) in {:.1}s", st.report(), bad, st.skipped, t.elapsed().as_secs_f64());
}

/// `unpack <in.rfp> <out.bin>`: the 32-byte records in file order.
pub fn cmd_unpack(args: &[String]) {
    assert!(args.len() == 2, "usage: unpack <in.rfp> <out.bin>");
    let mut out = BufWriter::with_capacity(1 << 22, File::create(&args[1]).unwrap());
    let mut n = 0u64;
    for_each_record(&args[0], |r| {
        out.write_all(r).unwrap();
        n += 1;
    });
    out.flush().unwrap();
    eprintln!("{}: {n} positions", args[1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fen_rec(line: &str) -> Rec {
        encode(line).unwrap()
    }

    fn roundtrip(recs: &[Rec], target: usize) -> (Vec<Rec>, Stats) {
        let mut bytes = Vec::new();
        let mut w = Writer::new(&mut bytes, target, true);
        for r in recs {
            w.push(r);
        }
        let st = w.finish();
        let path = std::env::temp_dir().join(format!("rfp-test-{}-{}.rfp", std::process::id(), recs.len()));
        std::fs::write(&path, &bytes).unwrap();
        let mut back = Vec::new();
        for_each_record(path.to_str().unwrap(), |r| back.push(*r));
        std::fs::remove_file(&path).unwrap();
        (back, st)
    }

    #[test]
    fn exp_golomb_roundtrip() {
        for k in 0..=MAX_K {
            let vals: Vec<u64> = (0..300).chain([1000, 65535, 128_000, 1 << 33]).collect();
            let mut w = BitWriter::default();
            let mut bits = 0;
            for &v in &vals {
                eg_put(&mut w, v, k);
                bits += eg_bits(v, k) as usize;
            }
            let words: Vec<u64> = w.bytes().chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
            let mut r = BitReader { words: &words, pos: 0 };
            for &v in &vals {
                assert_eq!(eg_get(&mut r, k), Some(v), "k={k}");
            }
            assert_eq!(r.pos, bits);
        }
        for v in [-70000i64, -3, -1, 0, 1, 2, 32767] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn crc_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn game_with_skipped_plies_castling_promotion_and_shuffled_input() {
        // A game fragment: quiet moves, castling, a capture sequence (gap),
        // en passant, a promotion, and a new game (result change).
        let lines = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1 | 30 | 1.0",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2 | 24 | 1.0",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2 | 36 | 1.0",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 | 30 | 1.0",
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3 | 40 | 1.0",
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4 | 38 | 1.0",
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4 | 42 | 1.0",
            // Bxc6 dxc6 skipped: two plies, a capture and a recapture.
            "r1bqkb1r/ppp2ppp/2p2n2/4p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 6 | -10 | 1.0",
            "8/2P5/8/3pP3/8/8/k6K/8 w - d6 0 50 | 900 | 1.0",
            "8/2P5/3P4/8/8/8/k6K/8 b - - 0 50 | 1200 | 1.0",
            "2Q5/8/3P4/8/8/8/1k5K/8 b - - 0 51 | 2400 | 1.0",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5",
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1 | -32000 | 0.5",
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1 | 32000 | 0.0",
        ];
        let recs: Vec<Rec> = lines.iter().map(|l| fen_rec(l)).collect();
        for target in [1, 2, 3, 1000] {
            let (back, st) = roundtrip(&recs, target);
            assert_eq!(back, recs, "target {target}");
            assert_eq!(st.positions, recs.len() as u64);
        }
        let (_, st) = roundtrip(&recs, 1000);
        assert!(st.keyframes >= 3 && st.keyframes < 8, "{st:?}");
        // Reversed and interleaved orders still round-trip.
        let mut rev = recs.clone();
        rev.reverse();
        assert_eq!(roundtrip(&rev, 4).0, rev);
        let mixed: Vec<Rec> = (0..recs.len()).map(|i| recs[(i * 5) % recs.len()]).collect();
        assert_eq!(roundtrip(&mixed, 1000).0, mixed);
    }

    #[test]
    fn random_boards_roundtrip_and_corruption_is_caught() {
        let mut rng = crate::Rng::new(7);
        let mut recs = Vec::new();
        let mut p = Pos { board: [EMPTY; 64], occ: 0, stm: 0, result: 1, score: 0 };
        for i in 0..5000 {
            // Random walk: a few random edits, sometimes a fresh board.
            if i % 97 == 0 {
                p.board = [EMPTY; 64];
                for sq in 0..64 {
                    if rng.below(3) == 0 && p.board.iter().filter(|&&b| b != EMPTY).count() < 32 {
                        p.board[sq] = [0u8, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13][rng.below(12)];
                    }
                }
                p.result = rng.below(3) as u8;
            }
            for _ in 0..rng.below(4) {
                let (s, t) = (rng.below(64), rng.below(64));
                p.board.swap(s, t);
                if rng.below(10) == 0 {
                    p.board[t] = EMPTY;
                }
            }
            p.occ = (0..64).filter(|&s| p.board[s] != EMPTY).fold(0, |o, s| o | 1 << s);
            p.stm = rng.below(2) as u8;
            p.score = (p.score + rng.below(401) as i32 - 200).clamp(-32000, 32000);
            if i % 13 == 0 {
                p.score = rng.below(64001) as i32 - 32000;
            }
            recs.push(p.to_rec());
        }
        assert!(recs.iter().all(valid));
        assert_eq!(roundtrip(&recs, 300).0, recs);
        // A flipped bit is caught by the CRC.
        let mut bytes = Vec::new();
        let mut w = Writer::new(&mut bytes, 1 << 20, true);
        recs.iter().for_each(|r| w.push(r));
        w.finish();
        let h = ChunkHeader::parse(&bytes[8..8 + HEADER]).unwrap();
        let mut payload = bytes[8 + HEADER..].to_vec();
        payload[100] ^= 4;
        assert!(decode_chunk(&h, &payload, &mut Vec::new()).is_err());
    }

    #[test]
    fn invalid_records_are_skipped() {
        let mut r = fen_rec("8/8/8/8/8/8/k6K/8 w - - 0 1 | 0 | 0.5");
        let mut w = Writer::new(Vec::new(), 10, true);
        w.push(&r);
        r[30] = 1;
        w.push(&r);
        assert_eq!(w.finish().skipped, 1);
    }
}
