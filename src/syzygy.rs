//! Syzygy endgame tablebase probing (WDL + DTZ, up to 5 pieces in practice).
//!
//! Clean-room implementation: the file format was learned from public prose
//! descriptions (Ronald de Man's forum explanations: Re-Pair + canonical
//! Huffman over small blocks, two-sided WDL, one-sided DTZ, "K2"/"111"
//! leading groups, don't-care capture positions) and from inspecting and
//! experimenting with the data files themselves. No probing code was read.
//! The byte-level layout, symmetry rules and index order are written down in
//! `docs/syzygy-format.md`, together with how each part was established.
//!
//! Structure: `TbFile` parses one `.rtbw` / `.rtbz` file into per-table
//! decoding data (`Pairs`) plus the index encoding (`Encoding`). `Tablebases`
//! owns a directory of files (loaded lazily, kept in memory) and offers raw
//! probes on a piece list; the engine-facing functions at the bottom resolve
//! captures, en passant and the side DTZ does not store.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use crate::{Board, Color, Move, PieceType};

/// Score of a tablebase win at the root (minus ply); below every mate score
/// (`MATE_SCORE - 1000` and up) and above any evaluation.
pub(crate) const TB_WIN: i32 = 25_000;

const WDL_MAGIC: [u8; 4] = [0x71, 0xe8, 0x23, 0x5d];
const DTZ_MAGIC: [u8; 4] = [0xd7, 0x66, 0x0c, 0xa5];

// ---------------------------------------------------------------------------
// Combinatorics and fixed square orders
// ---------------------------------------------------------------------------

struct Tables {
    binom: [[u64; 8]; 65],
    /// Pair index of the two leading kings ("K2" leading group), 462 values;
    /// u16::MAX for pairs that are not in canonical form.
    k2: Box<[[u16; 64]; 64]>,
    /// Index of each off-diagonal triangle square b1,c1,d1,c2,d2,d3 (0..5).
    tri: [i8; 64],
    /// Index of each diagonal square a1,b2,c3,d4 within the triangle (0..3).
    diag_t: [i8; 64],
    /// Rank of each strictly-below-diagonal square (file > rank), 0..27.
    below: [i8; 64],
    /// Position of each pawn square in the interleaved order
    /// a2,h2,a3,h3..a7,h7,b2,g2,..,d7,e7 (48 squares; -1 elsewhere).
    pawn_pos: [i8; 64],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut binom = [[0u64; 8]; 65];
        for n in 0..65 {
            binom[n][0] = 1;
            for k in 1..8 {
                binom[n][k] = if n == 0 { 0 } else { binom[n - 1][k - 1] + binom[n - 1][k] };
            }
        }
        let tri_sq = [1u8, 2, 3, 10, 11, 19];
        let mut tri = [-1i8; 64];
        for (i, &s) in tri_sq.iter().enumerate() {
            tri[s as usize] = i as i8;
        }
        let mut diag_t = [-1i8; 64];
        for i in 0..4 {
            diag_t[i * 9] = i as i8;
        }
        let mut below = [-1i8; 64];
        let mut n = 0;
        for s in 0..64 {
            if (s & 7) > (s >> 3) {
                below[s] = n;
                n += 1;
            }
        }
        let adj = |a: u8, b: u8| ((a & 7) as i8 - (b & 7) as i8).abs() <= 1 && ((a >> 3) as i8 - (b >> 3) as i8).abs() <= 1;
        let mut k2 = Box::new([[u16::MAX; 64]; 64]);
        let mut idx = 0u16;
        for &p1 in &tri_sq {
            for p2 in 0..64u8 {
                if p2 != p1 && !adj(p1, p2) {
                    k2[p1 as usize][p2 as usize] = idx;
                    idx += 1;
                }
            }
        }
        for d in 0..4u8 {
            let p1 = d * 9;
            for p2 in 0..64u8 {
                if below[p2 as usize] >= 0 && !adj(p1, p2) {
                    k2[p1 as usize][p2 as usize] = idx;
                    idx += 1;
                }
            }
        }
        for d in 0..4u8 {
            let p1 = d * 9;
            for e in 0..8u8 {
                let p2 = e * 9;
                if p2 != p1 && !adj(p1, p2) {
                    k2[p1 as usize][p2 as usize] = idx;
                    idx += 1;
                }
            }
        }
        debug_assert_eq!(idx, 462);
        let mut pawn_pos = [-1i8; 64];
        let mut i = 0;
        for e in 0..4u8 {
            for r in 1..7u8 {
                for f in [e, 7 - e] {
                    let s = r * 8 + f;
                    pawn_pos[s as usize] = i as i8;
                    i += 1;
                }
            }
        }
        Tables { binom, k2, tri, diag_t, below, pawn_pos }
    })
}

#[inline]
fn binom(n: usize, k: usize) -> u64 {
    tables().binom[n][k]
}

#[inline]
fn on_diag(s: u8) -> bool {
    (s & 7) == (s >> 3)
}

#[inline]
fn above_diag(s: u8) -> bool {
    (s >> 3) > (s & 7)
}

#[inline]
fn transpose(s: u8) -> u8 {
    ((s & 7) << 3) | (s >> 3)
}

// ---------------------------------------------------------------------------
// Compressed table data (Re-Pair symbols + canonical Huffman over blocks)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Pairs {
    /// Some(v): every position stores v (no block data).
    constant: Option<u8>,
    flags: u8,
    block_log: u8,
    idx_bits: u8,
    nblocks: u32,
    min_len: u8,
    /// base[l - min_len]: first canonical code value of length l (longest codes smallest).
    base: Vec<u64>,
    /// offs[l - min_len]: symbol index of the first code of length l.
    offs: Vec<u16>,
    /// (left, right) children; right == 0xfff marks a leaf whose value is left.
    syms: Vec<(u16, u16)>,
    /// Number of values each symbol expands to, minus one.
    symlen: Vec<u16>,
    sparse_off: usize,
    blen_off: usize,
    data_off: usize,
    size: u64,
    /// DTZ value maps (win, loss, cursed win, blessed loss).
    maps: Option<[Vec<u8>; 4]>,
}

fn rd16(d: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([d[p], d[p + 1]])
}

fn rd32(d: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([d[p], d[p + 1], d[p + 2], d[p + 3]])
}

/// Parse one table descriptor at `pos`; returns the new position.
fn parse_pairs(d: &[u8], mut pos: usize) -> Result<(Pairs, usize), String> {
    let mut t = Pairs::default();
    let need = |p: usize| if p > d.len() { Err("truncated table".to_string()) } else { Ok(()) };
    need(pos + 2)?;
    t.flags = d[pos];
    if t.flags & 0x80 != 0 {
        t.constant = Some(d[pos + 1]);
        return Ok((t, pos + 2));
    }
    need(pos + 10)?;
    t.block_log = d[pos + 1];
    t.idx_bits = d[pos + 2];
    let extra = d[pos + 3] as u32;
    t.nblocks = rd32(d, pos + 4);
    let max_len = d[pos + 8];
    t.min_len = d[pos + 9];
    if max_len < t.min_len || max_len > 56 || t.block_log > 12 || t.idx_bits == 0 || t.idx_bits > 40 {
        return Err("bad table header".into());
    }
    pos += 10;
    let nl = (max_len - t.min_len + 1) as usize;
    need(pos + 2 * nl + 2)?;
    t.offs = (0..nl).map(|i| rd16(d, pos + 2 * i)).collect();
    pos += 2 * nl;
    let nsym = rd16(d, pos) as usize;
    pos += 2;
    need(pos + 3 * nsym)?;
    t.syms = (0..nsym)
        .map(|i| {
            let b0 = d[pos + 3 * i] as u16;
            let b1 = d[pos + 3 * i + 1] as u16;
            let b2 = d[pos + 3 * i + 2] as u16;
            (b0 | ((b1 & 0xf) << 8), (b1 >> 4) | (b2 << 4))
        })
        .collect();
    pos += 3 * nsym;
    if pos & 1 == 1 {
        pos += 1;
    }
    // Symbol expansion lengths (children may come after their parent).
    fn len_of(s: usize, syms: &[(u16, u16)], memo: &mut [u16], depth: u32) -> Result<u16, String> {
        if memo[s] != u16::MAX {
            return Ok(memo[s]);
        }
        let (l, r) = syms[s];
        let v = if r == 0xfff {
            0
        } else {
            let (l, r) = (l as usize, r as usize);
            if l >= syms.len() || r >= syms.len() || depth > 512 {
                return Err("bad symbol tree".into());
            }
            let a = len_of(l, syms, memo, depth + 1)?;
            let b = len_of(r, syms, memo, depth + 1)?;
            a.saturating_add(b).saturating_add(1)
        };
        memo[s] = v;
        Ok(v)
    }
    let mut symlen = vec![u16::MAX; nsym];
    for s in 0..nsym {
        len_of(s, &t.syms, &mut symlen, 0)?;
    }
    t.symlen = symlen;
    // Canonical code bases: counts per length come from consecutive offsets;
    // the shortest length takes whatever is left of the code space.
    let mut count = vec![0u64; nl];
    for i in 1..nl {
        count[i] = t.offs[i - 1].wrapping_sub(t.offs[i]) as u64;
    }
    let mut base = vec![0u64; nl];
    for i in (0..nl.saturating_sub(1)).rev() {
        base[i] = (base[i + 1] + count[i + 1]) >> 1;
    }
    t.base = base;
    let _ = extra;
    Ok((t, pos))
}

impl Pairs {
    fn extra_from(d: &[u8], desc_pos: usize) -> u32 {
        d[desc_pos + 3] as u32
    }

    /// Stored value at `idx`.
    fn value(&self, d: &[u8], idx: u64) -> u8 {
        if let Some(c) = self.constant {
            return c;
        }
        let bits = self.idx_bits as u32;
        let k = (idx >> bits) as usize;
        let e = self.sparse_off + 6 * k;
        let mut block = rd32(d, e) as i64;
        let mut off = rd16(d, e + 4) as i64 + (idx as i64 - ((k as i64) << bits) - (1i64 << (bits - 1)));
        let blen = |b: i64| rd16(d, self.blen_off + 2 * b as usize) as i64 + 1;
        while off < 0 {
            block -= 1;
            off += blen(block);
        }
        loop {
            let l = blen(block);
            if off < l {
                break;
            }
            off -= l;
            block += 1;
        }
        // Huffman-decode the block until the symbol covering `off`.
        let bsize = 1usize << self.block_log;
        let start = self.data_off + (block as usize) * bsize;
        let mut buf = [0u8; 1024 + 16];
        buf[..bsize].copy_from_slice(&d[start..start + bsize]);
        let mut bitpos = 0usize;
        let mut remaining = off as u64;
        let min_len = self.min_len as usize;
        let sym = loop {
            let byte = bitpos >> 3;
            let mut w = 0u64;
            for i in 0..8 {
                w = (w << 8) | buf[byte + i] as u64;
            }
            w = (w << (bitpos & 7)) | ((buf[byte + 8] as u64) >> (8 - (bitpos & 7)));
            let mut l = min_len;
            let mut code = w >> (64 - l);
            while code < self.base[l - min_len] {
                l += 1;
                code = w >> (64 - l);
            }
            let s = self.offs[l - min_len] as usize + (code - self.base[l - min_len]) as usize;
            bitpos += l;
            let n = self.symlen[s] as u64 + 1;
            if remaining < n {
                break s;
            }
            remaining -= n;
        };
        // Walk down the pair tree.
        let mut s = sym;
        loop {
            let (l, r) = self.syms[s];
            if r == 0xfff {
                return l as u8;
            }
            let ln = self.symlen[l as usize] as u64 + 1;
            if remaining < ln {
                s = l as usize;
            } else {
                remaining -= ln;
                s = r as usize;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Index encoding
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Encoding {
    /// Piece codes in table order: 1..6 = P N B R Q K white, +8 black.
    pieces: Vec<u8>,
    /// Group sizes; the first is the leading group.
    groups: Vec<u8>,
    /// Radix slot (least significant = 0) of the leading group.
    order: u8,
    /// Radix slot of the second pawn group (tables with pawns of both colours).
    order2: Option<u8>,
    /// Pawn table file 0..3 (None for pawnless).
    file: Option<u8>,
    /// Per-group digit sizes and radix slots (precomputed).
    sizes: Vec<u64>,
    slot_of: Vec<usize>,
}

impl Encoding {
    fn new(pieces: Vec<u8>, order: u8, order2: Option<u8>, file: Option<u8>) -> Self {
        let n = pieces.len();
        let mut groups = Vec::new();
        let mut i;
        if file.is_some() {
            let k = pieces.iter().filter(|&&c| c == pieces[0]).count();
            groups.push(k as u8);
            i = k;
        } else {
            let unique3 = n >= 3 && (0..3).all(|j| pieces.iter().filter(|&&c| c == pieces[j]).count() == 1);
            if unique3 {
                groups.push(3);
                i = 3;
            } else {
                groups.push(2);
                i = 2;
            }
        }
        while i < n {
            let mut j = i;
            while j < n && pieces[j] == pieces[i] {
                j += 1;
            }
            groups.push((j - i) as u8);
            i = j;
        }
        let mut e = Encoding { pieces, groups, order, order2, file, sizes: Vec::new(), slot_of: Vec::new() };
        e.sizes = e.group_sizes();
        e.slot_of = e.slots();
        e
    }

    fn lead_size(&self) -> u64 {
        let k = self.groups[0] as usize;
        match self.file {
            None => {
                if k == 3 {
                    31332
                } else {
                    462
                }
            }
            Some(f) => (0..6).map(|i| binom(47 - 2 * i - 12 * f as usize, k - 1)).sum(),
        }
    }

    /// Group sizes (number of values per radix digit), in group order.
    fn group_sizes(&self) -> Vec<u64> {
        let mut out = vec![self.lead_size()];
        let mut placed = self.groups[0] as usize;
        let mut pawns_placed = if self.file.is_some() { placed } else { 0 };
        let mut i = placed;
        for &k in &self.groups[1..] {
            let k = k as usize;
            if self.pieces[i] & 7 == 1 {
                out.push(binom(48 - pawns_placed, k));
                pawns_placed += k;
            } else {
                out.push(binom(64 - placed, k));
            }
            placed += k;
            i += k;
        }
        out
    }

    fn size(&self) -> u64 {
        self.sizes.iter().product()
    }

    /// Radix slot of each group (least significant first).
    fn slots(&self) -> Vec<usize> {
        let n = self.groups.len();
        let mut slot = vec![usize::MAX; n];
        let mut taken = vec![false; n];
        slot[0] = self.order as usize;
        taken[self.order as usize] = true;
        let mut first_other = 1;
        if let Some(o2) = self.order2 {
            slot[1] = o2 as usize;
            taken[o2 as usize] = true;
            first_other = 2;
        }
        let mut s = 0;
        for g in first_other..n {
            while taken[s] {
                s += 1;
            }
            slot[g] = s;
            taken[s] = true;
        }
        slot
    }

    /// Index of a position given squares in table piece order (like pieces
    /// adjacent). Squares must already be in the table's colour frame.
    fn index(&self, sq_in: &[u8]) -> u64 {
        let t = tables();
        let n = sq_in.len();
        let mut sq = [0u8; 8];
        sq[..n].copy_from_slice(sq_in);
        let k0 = self.groups[0] as usize;
        let lead_idx: u64;
        if let Some(f) = self.file {
            // Leading pawns: sort by interleaved order; mirroring already applied.
            sq[..k0].sort_unstable_by_key(|&s| t.pawn_pos[s as usize]);
            let lead = &sq[..k0];
            let p0 = t.pawn_pos[lead[0] as usize] as usize;
            let q0 = (p0 - 12 * f as usize) / 2;
            let mut idx: u64 = (0..q0).map(|i| binom(47 - 2 * i - 12 * f as usize, k0 - 1)).sum();
            let mut rest = [0usize; 8];
            for (j, &s) in lead[1..].iter().enumerate() {
                rest[j] = 47 - t.pawn_pos[s as usize] as usize;
            }
            let rest = &mut rest[..k0 - 1];
            rest.sort_unstable();
            for (j, &r) in rest.iter().enumerate() {
                idx += binom(r, j + 1);
            }
            lead_idx = idx;
        } else {
            // Symmetry: first piece into the a1-d1-d4 triangle.
            if sq[0] & 7 > 3 {
                for s in &mut sq[..n] {
                    *s ^= 7;
                }
            }
            if sq[0] >> 3 > 3 {
                for s in &mut sq[..n] {
                    *s ^= 56;
                }
            }
            for i in 0..k0 {
                if !on_diag(sq[i]) {
                    if above_diag(sq[i]) {
                        for s in &mut sq[..n] {
                            *s = transpose(*s);
                        }
                    }
                    break;
                }
            }
            if k0 == 3 {
                let (p1, p2, p3) = (sq[0], sq[1], sq[2]);
                let ti = t.tri[p1 as usize];
                lead_idx = if ti >= 0 {
                    let i2 = p2 as u64 - (p2 > p1) as u64;
                    let i3 = p3 as u64 - (p3 > p1) as u64 - (p3 > p2) as u64;
                    ti as u64 * 3906 + i2 * 62 + i3
                } else {
                    let d = t.diag_t[p1 as usize] as u64;
                    if !on_diag(p2) {
                        let i3 = p3 as u64 - (p3 > p1) as u64 - (p3 > p2) as u64;
                        23436 + (d * 28 + t.below[p2 as usize] as u64) * 62 + i3
                    } else {
                        let j2 = (p2 / 9) as u64 - (p2 > p1) as u64;
                        if !on_diag(p3) {
                            30380 + (d * 7 + j2) * 28 + t.below[p3 as usize] as u64
                        } else {
                            let k3 = (p3 / 9) as u64 - (p3 > p1) as u64 - (p3 > p2) as u64;
                            31164 + (d * 7 + j2) * 6 + k3
                        }
                    }
                };
            } else {
                lead_idx = t.k2[sq[0] as usize][sq[1] as usize] as u64;
            }
        }
        // Remaining groups.
        let sizes = &self.sizes;
        let slots = &self.slot_of;
        let mut digits = [0u64; 8];
        digits[0] = lead_idx;
        let mut i = k0;
        for g in 1..self.groups.len() {
            let k = self.groups[g] as usize;
            let mut grp = [0u8; 8];
            grp[..k].copy_from_slice(&sq[i..i + k]);
            let grp = &mut grp[..k];
            grp.sort_unstable();
            let pawn = self.pieces[i] & 7 == 1 && self.file.is_some();
            let mut v = 0u64;
            for (j, &s) in grp.iter().enumerate() {
                let adj = if pawn {
                    // Pawn squares a2..h7 numbered 0..47, skipping placed pawns.
                    let below = sq[..i].iter().enumerate().filter(|&(pi, &p)| self.pieces[pi] & 7 == 1 && p < s).count();
                    s as usize - 8 - below
                } else {
                    s as usize - sq[..i].iter().filter(|&&p| p < s).count()
                };
                v += binom(adj, j + 1);
            }
            digits[g] = v;
            i += k;
        }
        let mut by_slot = [(0u64, 1u64); 8];
        for g in 0..self.groups.len() {
            by_slot[slots[g]] = (digits[g], sizes[g]);
        }
        let mut idx = 0u64;
        for &(v, s) in by_slot[..self.groups.len()].iter().rev() {
            idx = idx * s + v;
        }
        idx
    }
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

struct SubTable {
    enc: Encoding,
    pairs: Pairs,
}

/// One parsed file. Pawnless: `tables[stm]` (1 or 2 tables). Pawns:
/// `tables[file * ntab + stm]`.
struct TbFile {
    data: Vec<u8>,
    wdl: bool,
    has_pawns: bool,
    ntab: usize,
    tables: Vec<SubTable>,
}

fn parse_file(data: Vec<u8>, white_pawns: bool, black_pawns: bool) -> Result<TbFile, String> {
    if data.len() < 32 {
        return Err("file too short".into());
    }
    let wdl = data[..4] == WDL_MAGIC;
    if !wdl && data[..4] != DTZ_MAGIC {
        return Err("bad magic".into());
    }
    let d = &data;
    let n = (d[4] >> 4) as usize;
    let hflags = d[4] & 0xf;
    let has_pawns = hflags & 2 != 0;
    if n < 2 || n > 7 {
        return Err("bad piece count".into());
    }
    let ntab = if wdl && hflags & 1 != 0 { 2 } else { 1 };
    let mut encs: Vec<Encoding> = Vec::new();
    let mut pos;
    let pieces_at = |p: usize, k: usize| -> Vec<u8> { (0..n).map(|i| (d[p + i] >> (4 * k)) & 0xf).collect() };
    if has_pawns {
        pos = 5;
        let both = white_pawns && black_pawns;
        for f in 0..4u8 {
            let o = d[pos];
            pos += 1;
            let o2 = if both {
                pos += 1;
                Some(d[pos - 1])
            } else {
                None
            };
            for k in 0..ntab {
                encs.push(Encoding::new(pieces_at(pos, k), (o >> (4 * k)) & 0xf, o2.map(|x| (x >> (4 * k)) & 0xf), Some(f)));
            }
            pos += n;
        }
    } else {
        let o = d[5];
        pos = 6;
        for k in 0..ntab {
            encs.push(Encoding::new(pieces_at(pos, k), (o >> (4 * k)) & 0xf, None, None));
        }
        pos += n;
    }
    if pos & 1 == 1 {
        pos += 1;
    }
    let mut pairs = Vec::new();
    let mut extras = Vec::new();
    for _ in 0..encs.len() {
        let desc = pos;
        let (p, np) = parse_pairs(d, pos)?;
        extras.push(if p.constant.is_none() { Pairs::extra_from(d, desc) } else { 0 });
        pairs.push(p);
        pos = np;
    }
    if !wdl {
        for p in pairs.iter_mut() {
            if p.constant.is_none() && p.flags & 2 != 0 {
                let mut maps: [Vec<u8>; 4] = Default::default();
                for m in maps.iter_mut() {
                    if pos >= d.len() {
                        return Err("truncated map".into());
                    }
                    let l = d[pos] as usize;
                    *m = d[pos + 1..pos + 1 + l].to_vec();
                    pos += 1 + l;
                }
                p.maps = Some(maps);
            }
        }
        if pos & 1 == 1 {
            pos += 1;
        }
    }
    for (p, e) in pairs.iter_mut().zip(&encs) {
        p.size = e.size();
        if p.constant.is_none() {
            p.sparse_off = pos;
            pos += 6 * ((p.size + (1u64 << p.idx_bits) - 1) >> p.idx_bits) as usize;
        }
    }
    for (p, &x) in pairs.iter_mut().zip(&extras) {
        if p.constant.is_none() {
            p.blen_off = pos;
            pos += 2 * (p.nblocks + x) as usize;
        }
    }
    pos = (pos + 63) & !63;
    for p in pairs.iter_mut() {
        if p.constant.is_none() {
            p.data_off = pos;
            pos += (p.nblocks as usize) << p.block_log;
            pos = (pos + 63) & !63;
        }
    }
    if pos + 16 != d.len() {
        return Err(format!("layout mismatch: computed end {} vs file {}", pos + 16, d.len()));
    }
    let tables = encs.into_iter().zip(pairs).map(|(enc, pairs)| SubTable { enc, pairs }).collect();
    Ok(TbFile { data, wdl, has_pawns, ntab, tables })
}

// ---------------------------------------------------------------------------
// Material keys and the directory of tables
// ---------------------------------------------------------------------------

/// Piece counts per colour: [P, N, B, R, Q] (kings implied).
type Counts = [[u8; 5]; 2];

fn counts_key(c: &Counts) -> u32 {
    let mut k = 0u32;
    for side in c {
        for &x in side {
            k = k * 8 + x as u32;
        }
    }
    k
}

#[cfg(test)]
fn side_name(c: &[u8; 5]) -> String {
    let mut s = String::from("K");
    for (i, ch) in [(4, 'Q'), (3, 'R'), (2, 'B'), (1, 'N'), (0, 'P')] {
        for _ in 0..c[i] {
            s.push(ch);
        }
    }
    s
}

fn parse_side(s: &str) -> Option<[u8; 5]> {
    let mut c = [0u8; 5];
    let mut chars = s.chars();
    if chars.next() != Some('K') {
        return None;
    }
    for ch in chars {
        let i = match ch {
            'P' => 0,
            'N' => 1,
            'B' => 2,
            'R' => 3,
            'Q' => 4,
            _ => return None,
        };
        c[i] += 1;
    }
    Some(c)
}

struct Entry {
    /// File-name material: [white side, black side] as named.
    counts: Counts,
    wdl_path: Option<PathBuf>,
    dtz_path: Option<PathBuf>,
    wdl: OnceLock<Option<Arc<TbFile>>>,
    dtz: OnceLock<Option<Arc<TbFile>>>,
}

pub(crate) struct Tablebases {
    entries: Vec<Entry>,
    /// Material key -> (entry index, colours swapped relative to the file).
    by_key: HashMap<u32, (usize, bool)>,
    pub(crate) max_pieces: usize,
}

impl Tablebases {
    /// Scan `paths` (separated by ':' or ';') for table files.
    pub(crate) fn open(paths: &str) -> Tablebases {
        let mut names: HashMap<String, (Option<PathBuf>, Option<PathBuf>)> = HashMap::new();
        for dir in paths.split(|c| c == ':' || c == ';').filter(|s| !s.is_empty()) {
            let Ok(rd) = std::fs::read_dir(dir) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                let (Some(stem), Some(ext)) = (p.file_stem().and_then(|s| s.to_str()), p.extension().and_then(|s| s.to_str())) else { continue };
                let slot = names.entry(stem.to_string()).or_default();
                match ext {
                    "rtbw" => slot.0 = Some(p.clone()),
                    "rtbz" => slot.1 = Some(p.clone()),
                    _ => {}
                }
            }
        }
        let mut tb = Tablebases { entries: Vec::new(), by_key: HashMap::new(), max_pieces: 0 };
        let mut sorted: Vec<_> = names.into_iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        for (stem, (w, z)) in sorted {
            let Some((a, b)) = stem.split_once('v') else { continue };
            let (Some(ca), Some(cb)) = (parse_side(a), parse_side(b)) else { continue };
            if w.is_none() {
                continue;
            }
            let counts = [ca, cb];
            let pcs = 2 + ca.iter().chain(cb.iter()).map(|&x| x as usize).sum::<usize>();
            tb.max_pieces = tb.max_pieces.max(pcs);
            let idx = tb.entries.len();
            tb.entries.push(Entry { counts, wdl_path: w, dtz_path: z, wdl: OnceLock::new(), dtz: OnceLock::new() });
            tb.by_key.entry(counts_key(&counts)).or_insert((idx, false));
            tb.by_key.entry(counts_key(&[cb, ca])).or_insert((idx, true));
        }
        tb
    }

    pub(crate) fn num_files(&self) -> usize {
        self.entries.len()
    }

    fn file(&self, e: usize, dtz: bool) -> Option<Arc<TbFile>> {
        let ent = &self.entries[e];
        let (cell, path) = if dtz { (&ent.dtz, &ent.dtz_path) } else { (&ent.wdl, &ent.wdl_path) };
        cell.get_or_init(|| {
            let path = path.as_ref()?;
            let data = std::fs::read(path).ok()?;
            let wp = ent.counts[0][0] > 0;
            let bp = ent.counts[1][0] > 0;
            match parse_file(data, wp, bp) {
                Ok(f) => Some(Arc::new(f)),
                Err(e) => {
                    eprintln!("info string syzygy: cannot use {}: {}", path.display(), e);
                    None
                }
            }
        })
        .clone()
    }

    /// Raw table value for `pcs` = (code, square) pairs (codes 1..6 white,
    /// 9..14 black), ignoring en passant. Returns (value, dtz flags) where
    /// value is the stored WDL 0..4 or the stored DTZ symbol; None if the
    /// table is missing or (DTZ) the side to move is not stored.
    fn raw(&self, pcs: &[(u8, u8)], white_to_move: bool, dtz: bool) -> Option<(u32, u8, Option<[Vec<u8>; 4]>)> {
        let mut c: Counts = [[0; 5]; 2];
        for &(code, _) in pcs {
            let k = code & 7;
            if k != 6 {
                c[(code >> 3) as usize][(k - 1) as usize] += 1;
            }
        }
        let &(e, swapped) = self.by_key.get(&counts_key(&c))?;
        let f = self.file(e, dtz)?;
        let symmetric = self.entries[e].counts[0] == self.entries[e].counts[1];
        let mut flip = swapped;
        let mut stm = (!white_to_move) as usize ^ flip as usize;
        // Pick the stored side when the material is symmetric.
        if symmetric {
            let stored = if f.wdl {
                if f.ntab == 2 {
                    None
                } else {
                    Some(0)
                }
            } else {
                None
            };
            if let Some(s) = stored {
                if stm != s {
                    flip = !flip;
                    stm = s;
                }
            }
        }
        let mut sq: Vec<(u8, u8)> = pcs
            .iter()
            .map(|&(code, s)| if flip { (code ^ 8, if f.has_pawns { s ^ 56 } else { s }) } else { (code, s) })
            .collect();
        let pick = |sq: &mut Vec<(u8, u8)>| -> usize {
            if !f.has_pawns {
                return 0;
            }
            let t = tables();
            let lead_code = f.tables[0].enc.pieces[0];
            let first = sq.iter().filter(|x| x.0 == lead_code).map(|x| x.1).min_by_key(|&s| t.pawn_pos[s as usize]).unwrap_or(8);
            let mut fl = first & 7;
            if fl > 3 {
                for x in sq.iter_mut() {
                    x.1 ^= 7;
                }
                fl = 7 - fl;
            }
            fl as usize
        };
        let mut fidx = pick(&mut sq);
        let mut ti = if f.wdl { fidx * f.ntab + stm } else { fidx };
        if !f.wdl {
            let side = (f.tables[ti].pairs.flags & 1) as usize;
            if f.tables[ti].pairs.constant.is_none() && side != stm {
                if !symmetric {
                    return None;
                }
                flip = !flip;
                stm ^= 1;
                sq = pcs
                    .iter()
                    .map(|&(code, s)| if flip { (code ^ 8, if f.has_pawns { s ^ 56 } else { s }) } else { (code, s) })
                    .collect();
                fidx = pick(&mut sq);
                ti = fidx;
                if (f.tables[ti].pairs.flags & 1) as usize != stm && f.tables[ti].pairs.constant.is_none() {
                    return None;
                }
            }
        }
        let st = &f.tables[ti];
        if let Some(v) = st.pairs.constant {
            return Some((v as u32, st.pairs.flags, None));
        }
        // Squares in table piece order.
        let mut ordered = [0u8; 8];
        let mut used = [false; 8];
        for (i, &code) in st.enc.pieces.iter().enumerate() {
            let j = (0..sq.len()).find(|&j| !used[j] && sq[j].0 == code)?;
            used[j] = true;
            ordered[i] = sq[j].1;
        }
        let idx = st.enc.index(&ordered[..sq.len()]);
        if idx >= st.pairs.size {
            return None;
        }
        let maps = if dtz { st.pairs.maps.clone() } else { None };
        Some((st.pairs.value(&f.data, idx) as u32, st.pairs.flags, maps))
    }
}

// ---------------------------------------------------------------------------
// Engine-facing probing
// ---------------------------------------------------------------------------

fn piece_list(board: &Board) -> Vec<(u8, u8)> {
    let mut v = Vec::with_capacity(8);
    for c in 0..2 {
        for k in 0..6 {
            let mut bb = board.bb.pieces[c][k];
            while bb != 0 {
                let s = bb.trailing_zeros() as u8;
                bb &= bb - 1;
                v.push(((k as u8 + 1) | ((c as u8) << 3), s));
            }
        }
    }
    v
}

pub(crate) fn piece_count(board: &Board) -> u32 {
    (board.bb.occ[0] | board.bb.occ[1]).count_ones()
}

/// Stored WDL of the position itself (no capture resolution), -2..2.
fn table_wdl(tb: &Tablebases, board: &Board) -> Option<i32> {
    let pcs = piece_list(board);
    if pcs.len() == 2 {
        return Some(0);
    }
    let (v, _, _) = tb.raw(&pcs, board.side_to_move == Color::White, false)?;
    Some(v as i32 - 2)
}

fn is_capture(board: &Board, mv: &Move) -> bool {
    mv.is_en_passant || board.squares[mv.to as usize].is_some()
}

/// WDL (-2 loss, -1 blessed loss, 0 draw, 1 cursed win, 2 win) for the side
/// to move, assuming the halfmove clock was just reset. None when a table is
/// missing, castling rights exist or the position has too many pieces.
pub(crate) fn probe_wdl(tb: &Tablebases, board: &mut Board) -> Option<i32> {
    if board.castling != 0 || piece_count(board) as usize > tb.max_pieces {
        return None;
    }
    wdl_inner(tb, board)
}

fn wdl_inner(tb: &Tablebases, board: &mut Board) -> Option<i32> {
    if piece_count(board) == 2 {
        return Some(0);
    }
    let mut moves = Vec::with_capacity(64);
    // Captures decide the don't-care positions; en passant also needs to
    // know whether any other move exists.
    let need_all = board.en_passant.is_some();
    board.generate_legal_moves(!need_all, &mut moves);
    let mut best_cap = -3;
    let mut non_ep = false;
    for mv in moves.iter() {
        if !mv.is_en_passant {
            non_ep = true;
        }
        if !is_capture(board, mv) {
            continue;
        }
        let undo = board.make_move(*mv);
        let v = wdl_inner(tb, board).map(|x| -x);
        board.undo_move(*mv, undo);
        let v = v?;
        if v > best_cap {
            best_cap = v;
            if v == 2 {
                return Some(2);
            }
        }
    }
    if need_all && !non_ep {
        // Only en passant captures are legal: the stored value (which
        // assumes no ep right) would describe a stalemate or mate.
        return Some(best_cap);
    }
    let v = table_wdl(tb, board)?;
    Some(v.max(best_cap))
}

/// DTZ for the side to move in plies (sign = result; 0 = draw), assuming a
/// reset halfmove clock. Values from tables stored in moves may be one ply
/// too large or too small (docs/syzygy-format.md); never used inside search.
pub(crate) fn probe_dtz(tb: &Tablebases, board: &mut Board) -> Option<i32> {
    let w = probe_wdl(tb, board)?;
    if w == 0 {
        return Some(0);
    }
    dtz_inner(tb, board, w)
}

fn dtz_inner(tb: &Tablebases, board: &mut Board, w: i32) -> Option<i32> {
    let mut moves = Vec::with_capacity(64);
    board.generate_legal_moves(false, &mut moves);
    if moves.is_empty() {
        return Some(0);
    }
    // A zeroing move that keeps the result: DTZ 1 (stored value is a don't-care).
    if w > 0 {
        for mv in moves.iter() {
            let zeroing = is_capture(board, mv) || board.squares[mv.from as usize].map(|p| p.kind) == Some(PieceType::Pawn);
            if !zeroing {
                continue;
            }
            let undo = board.make_move(*mv);
            let v = wdl_inner(tb, board).map(|x| -x);
            board.undo_move(*mv, undo);
            if v? == w {
                return Some(zero_dtz(w));
            }
        }
    }
    let pcs = piece_list(board);
    if board.en_passant.is_none() {
        if let Some((raw, flags, maps)) = tb.raw(&pcs, board.side_to_move == Color::White, true) {
            return Some(decode_dtz(raw, flags, maps.as_ref(), w));
        }
    }
    // Side not stored (or ep right present): one ply of search.
    let mut best: Option<i32> = None;
    for mv in moves.iter() {
        let zeroing = is_capture(board, mv) || board.squares[mv.from as usize].map(|p| p.kind) == Some(PieceType::Pawn);
        let undo = board.make_move(*mv);
        let cw = wdl_inner(tb, board).map(|x| -x);
        let r = match cw {
            None => None,
            Some(v) if v != w => Some(None),
            Some(_) => {
                if zeroing {
                    Some(Some(zero_dtz(w).abs()))
                } else {
                    let cd = probe_dtz_inner_child(tb, board, -w);
                    cd.map(|d| Some(1 + d.abs()))
                }
            }
        };
        board.undo_move(*mv, undo);
        let r = r?;
        if let Some(d) = r {
            best = Some(match best {
                None => d,
                Some(b) => {
                    if w > 0 {
                        b.min(d)
                    } else {
                        b.max(d)
                    }
                }
            });
        }
    }
    let d = best?;
    Some(if w > 0 { d } else { -d })
}

fn probe_dtz_inner_child(tb: &Tablebases, board: &mut Board, w: i32) -> Option<i32> {
    // Child of a DTZ search: the side to move here is the stored side for
    // non-symmetric tables, so this terminates after at most one more ply.
    if w == 0 {
        return Some(0);
    }
    let pcs = piece_list(board);
    if board.en_passant.is_none() {
        if w > 0 {
            // zeroing winning move?
            let mut moves = Vec::with_capacity(64);
            board.generate_legal_moves(false, &mut moves);
            for mv in moves.iter() {
                let zeroing = is_capture(board, mv) || board.squares[mv.from as usize].map(|p| p.kind) == Some(PieceType::Pawn);
                if !zeroing {
                    continue;
                }
                let undo = board.make_move(*mv);
                let v = wdl_inner(tb, board).map(|x| -x);
                board.undo_move(*mv, undo);
                if v? == w {
                    return Some(zero_dtz(w));
                }
            }
        }
        if let Some((raw, flags, maps)) = tb.raw(&pcs, board.side_to_move == Color::White, true) {
            return Some(decode_dtz(raw, flags, maps.as_ref(), w));
        }
    }
    dtz_inner(tb, board, w)
}

/// DTZ of a position whose best move is zeroing: 1 ply, plus 100 when the
/// result is only a cursed win / blessed loss.
fn zero_dtz(w: i32) -> i32 {
    match w {
        2 => 1,
        1 => 101,
        -1 => -101,
        _ => -1,
    }
}

/// Turn a stored DTZ symbol into plies (signed by `w`).
fn decode_dtz(raw: u32, flags: u8, maps: Option<&[Vec<u8>; 4]>, w: i32) -> i32 {
    let cls = match w {
        2 => 0,
        -2 => 1,
        1 => 2,
        _ => 3,
    };
    let mut v = raw as i32;
    if let Some(m) = maps {
        if let Some(&x) = m[cls].get(raw as usize) {
            v = x as i32;
        }
    }
    // Wins use flag 4, losses flag 8 for "stored in plies"; otherwise
    // values are full moves. Cursed/blessed values are always full moves
    // counted beyond the 100-ply limit.
    let plies = match cls {
        0 => if flags & 4 != 0 { v + 1 } else { 2 * v + 1 },
        1 => if flags & 8 != 0 { v + 1 } else { 2 * v + 2 },
        2 => 101 + 2 * v,
        _ => 100 + 2 * v,
    };
    if w > 0 {
        plies
    } else {
        -plies
    }
}

/// Root move filter: when the root is in the tables, keep only the moves
/// with the best tablebase outcome under the 50-move rule (given the current
/// halfmove clock); after a repetition, keep only DTZ-optimal winning moves.
/// Leaves `moves` untouched when any needed table is missing.
pub(crate) fn filter_root_moves(tb: &Tablebases, board: &mut Board, moves: &mut Vec<Move>, repeated: bool) -> bool {
    if board.castling != 0 || piece_count(board) as usize > tb.max_pieces || moves.is_empty() {
        return false;
    }
    let clock = board.halfmove_clock as i32;
    let mut ranked: Vec<(i32, i32, Move)> = Vec::with_capacity(moves.len());
    for &mv in moves.iter() {
        let zeroing = is_capture(board, &mv) || board.squares[mv.from as usize].map(|p| p.kind) == Some(PieceType::Pawn);
        let undo = board.make_move(mv);
        let res = (|| {
            let mut legal = Vec::new();
            board.generate_legal_moves(false, &mut legal);
            if legal.is_empty() {
                // Mate or stalemate after our move.
                return Some(if board.is_in_check(board.side_to_move) { (2000, 0) } else { (0, 0) });
            }
            let w = -probe_wdl(tb, board)?;
            let d = probe_dtz(tb, board)?.abs();
            let after = if zeroing { 0 } else { clock + 1 };
            let rank = match w {
                2 => {
                    if after + d <= 100 {
                        1000
                    } else {
                        500
                    }
                }
                1 => 500,
                0 => 0,
                -1 => -500,
                _ => {
                    if after + d <= 100 {
                        -1000
                    } else {
                        -500
                    }
                }
            };
            Some((rank, d))
        })();
        board.undo_move(mv, undo);
        let Some((rank, d)) = res else { return false };
        ranked.push((rank, d, mv));
    }
    let best = ranked.iter().map(|r| r.0).max().unwrap_or(0);
    let mut keep: Vec<(i32, i32, Move)> = ranked.into_iter().filter(|r| r.0 == best).collect();
    if repeated && best >= 1000 {
        let dmin = keep.iter().map(|r| r.1).min().unwrap_or(0);
        keep.retain(|r| r.1 == dmin);
    }
    let kept: Vec<Move> = keep.iter().map(|r| r.2).collect();
    moves.retain(|m| kept.contains(m));
    true
}

/// `tbprobe`: print WDL and DTZ of the current position (debug command).
pub(crate) fn handle_tbprobe(board: &mut Board, engine: &crate::EngineState) {
    let Some(tb) = engine.syzygy.as_ref() else {
        println!("info string syzygy: no tables (set SyzygyPath)");
        return;
    };
    let wdl = probe_wdl(tb, board);
    let dtz = probe_dtz(tb, board);
    let fmt = |x: Option<i32>| x.map(|v| v.to_string()).unwrap_or_else(|| "none".into());
    println!("info string syzygy wdl {} dtz {}", fmt(wdl), fmt(dtz));
    crate::uci::flush_stdout();
}

#[cfg(test)]
pub(crate) fn test_tb_path() -> Option<String> {
    let p = std::env::var("RUSTFISH_SYZYGY_TEST_PATH").ok().or_else(|| std::env::var("HOME").ok().map(|h| format!("{}/rustfish-data/syzygy", h)))?;
    if std::path::Path::new(&p).join("KQvK.rtbw").exists() {
        Some(p)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_tables_have_expected_sizes() {
        let t = tables();
        let n = t.k2.iter().flatten().filter(|&&x| x != u16::MAX).count();
        assert_eq!(n, 462);
        assert_eq!(Encoding::new(vec![6, 14, 4], 0, None, None).size(), 31332);
        assert_eq!(Encoding::new(vec![1, 1, 6, 14], 2, None, Some(0)).size(), 252 * 62 * 61);
    }

    #[test]
    fn all_files_parse_with_consistent_layout() {
        let Some(path) = test_tb_path() else {
            eprintln!("syzygy files not found; skipping");
            return;
        };
        let tb = Tablebases::open(&path);
        assert!(tb.num_files() > 0);
        for e in 0..tb.entries.len() {
            for dtz in [false, true] {
                if dtz && tb.entries[e].dtz_path.is_none() {
                    continue;
                }
                let f = tb.file(e, dtz);
                assert!(f.is_some(), "{} dtz={}", side_name(&tb.entries[e].counts[0]), dtz);
                let f = f.unwrap();
                for st in &f.tables {
                    if st.pairs.constant.is_none() {
                        // Block lengths must add up to the index size.
                        let nb = st.pairs.nblocks as usize;
                        let sum: u64 = (0..nb).map(|b| rd16(&f.data, st.pairs.blen_off + 2 * b) as u64 + 1).sum();
                        assert_eq!(sum, st.pairs.size);
                    }
                }
            }
        }
    }

    /// Random legal position for a material string like "KRPvKR" (or None).
    fn random_fen(mat: &str, rng: &mut u64, stm_white: bool) -> Option<String> {
        let mut next = || {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            *rng
        };
        let (w, b) = mat.split_once('v')?;
        let mut board = [None::<char>; 64];
        for (side, white) in [(w, true), (b, false)] {
            for ch in side.chars() {
                loop {
                    let s = (next() % 64) as usize;
                    if board[s].is_some() || (ch == 'P' && (s < 8 || s >= 56)) {
                        continue;
                    }
                    board[s] = Some(if white { ch } else { ch.to_ascii_lowercase() });
                    break;
                }
            }
        }
        let mut fen = String::new();
        for r in (0..8).rev() {
            let mut e = 0;
            for f in 0..8 {
                match board[r * 8 + f] {
                    Some(c) => {
                        if e > 0 {
                            fen.push_str(&e.to_string());
                            e = 0;
                        }
                        fen.push(c);
                    }
                    None => e += 1,
                }
            }
            if e > 0 {
                fen.push_str(&e.to_string());
            }
            if r > 0 {
                fen.push('/');
            }
        }
        fen.push_str(if stm_white { " w - - 0 1" } else { " b - - 0 1" });
        Some(fen)
    }

    /// WDL of every random position equals the best WDL over its children
    /// (the tables must agree with themselves across sides and captures).
    #[test]
    fn wdl_is_self_consistent() {
        let Some(path) = test_tb_path() else {
            eprintln!("syzygy files not found; skipping");
            return;
        };
        let tb = Tablebases::open(&path);
        let mut rng = 0x9e3779b97f4a7c15u64;
        let mut checked = 0;
        let mats: Vec<String> = tb.entries.iter().map(|e| format!("{}v{}", side_name(&e.counts[0]), side_name(&e.counts[1]))).collect();
        for mat in &mats {
            let mut done = 0;
            let mut tries = 0;
            while done < 12 && tries < 2000 {
                tries += 1;
                let stm = rng & 1 == 0;
                let Some(fen) = random_fen(mat, &mut rng, stm) else { continue };
                let mut board = Board::new();
                board.set_fen(&fen);
                if board.is_in_check(board.side_to_move.opposite()) {
                    continue;
                }
                let mut moves = Vec::new();
                board.generate_legal_moves(false, &mut moves);
                if moves.is_empty() {
                    continue;
                }
                let v = probe_wdl(&tb, &mut board).expect(&fen);
                let mut best = -3;
                for mv in moves {
                    let undo = board.make_move(mv);
                    let c = -probe_wdl(&tb, &mut board).expect(&fen);
                    board.undo_move(mv, undo);
                    best = best.max(c);
                }
                assert_eq!(v, best, "{}", fen);
                done += 1;
                checked += 1;
            }
        }
        assert!(checked > 1000);
    }

    /// Probe speed (run with --ignored --nocapture): raw table lookups and
    /// full WDL probes (with capture resolution) on 5-piece positions.
    #[test]
    #[ignore]
    fn probe_speed() {
        let Some(path) = test_tb_path() else { return };
        let tb = Tablebases::open(&path);
        let mut rng = 0x1234567u64;
        let mut boards = Vec::new();
        for mat in ["KQRvKR", "KRPvKR", "KBNvKP", "KPPvKP", "KQvKBN"] {
            while boards.len() % 400 != 0 || boards.len() == 0 || boards.len() < 400 * (["KQRvKR", "KRPvKR", "KBNvKP", "KPPvKP", "KQvKBN"].iter().position(|m| *m == mat).unwrap() + 1) {
                let stm = rng & 1 == 0;
                let fen = random_fen(mat, &mut rng, stm).unwrap();
                let mut b = Board::new();
                b.set_fen(&fen);
                if !b.is_in_check(b.side_to_move.opposite()) {
                    boards.push(b);
                }
            }
        }
        for b in boards.iter_mut() {
            probe_wdl(&tb, b);
        }
        let t = std::time::Instant::now();
        let mut n = 0;
        for _ in 0..5 {
            for b in boards.iter_mut() {
                let pcs = piece_list(b);
                std::hint::black_box(tb.raw(&pcs, b.side_to_move == Color::White, false));
                n += 1;
            }
        }
        let raw_ns = t.elapsed().as_nanos() as f64 / n as f64;
        let t = std::time::Instant::now();
        for _ in 0..5 {
            for b in boards.iter_mut() {
                std::hint::black_box(probe_wdl(&tb, b));
            }
        }
        let wdl_ns = t.elapsed().as_nanos() as f64 / n as f64;
        let t = std::time::Instant::now();
        for b in boards.iter_mut().take(400) {
            std::hint::black_box(probe_dtz(&tb, b));
        }
        let dtz_us = t.elapsed().as_micros() as f64 / 400.0;
        println!("raw table lookup {:.0} ns, full WDL probe {:.0} ns, DTZ probe {:.1} us", raw_ns, wdl_ns, dtz_us);
    }
}

