use super::*;

/// Piece/occupancy bitboards maintained incrementally alongside the mailbox
/// board. Square 0 = a1 maps to bit 0, square 63 = h8 to bit 63.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BoardBB {
    pub(crate) pieces: [[u64; 6]; 2],
    pub(crate) occ: [u64; 2],
}

pub(crate) fn kind_index(kind: PieceType) -> usize {
    match kind {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 5,
    }
}

impl BoardBB {
    pub(crate) fn toggle(&mut self, sq: u8, p: Piece) {
        let bit = 1u64 << sq;
        let c = color_index(p.color);
        self.pieces[c][kind_index(p.kind)] ^= bit;
        self.occ[c] ^= bit;
    }

    pub(crate) fn occ_all(&self) -> u64 {
        self.occ[0] | self.occ[1]
    }
}

const fn knight_attacks_from(sq: usize) -> u64 {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let deltas = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];
    let mut att = 0u64;
    let mut i = 0;
    while i < 8 {
        let nf = f + deltas[i].0;
        let nr = r + deltas[i].1;
        if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
            att |= 1u64 << (nr * 8 + nf);
        }
        i += 1;
    }
    att
}

const fn king_attacks_from(sq: usize) -> u64 {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    let mut att = 0u64;
    let mut df = -1;
    while df <= 1 {
        let mut dr = -1;
        while dr <= 1 {
            if !(df == 0 && dr == 0) {
                let nf = f + df;
                let nr = r + dr;
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    att |= 1u64 << (nr * 8 + nf);
                }
            }
            dr += 1;
        }
        df += 1;
    }
    att
}

/// Squares from which a pawn of `color` attacks `sq`.
const fn pawn_attackers_of(sq: usize, color_idx: usize) -> u64 {
    let f = (sq % 8) as i32;
    let r = (sq / 8) as i32;
    // A white pawn attacks from one rank below the target; black from above.
    let pr = if color_idx == 0 { r - 1 } else { r + 1 };
    let mut att = 0u64;
    if pr >= 0 && pr < 8 {
        if f > 0 {
            att |= 1u64 << (pr * 8 + f - 1);
        }
        if f < 7 {
            att |= 1u64 << (pr * 8 + f + 1);
        }
    }
    att
}

const fn build_knight_table() -> [u64; 64] {
    let mut t = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        t[sq] = knight_attacks_from(sq);
        sq += 1;
    }
    t
}

const fn build_king_table() -> [u64; 64] {
    let mut t = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        t[sq] = king_attacks_from(sq);
        sq += 1;
    }
    t
}

const fn build_pawn_attacker_table() -> [[u64; 64]; 2] {
    let mut t = [[0u64; 64]; 2];
    let mut sq = 0;
    while sq < 64 {
        t[0][sq] = pawn_attackers_of(sq, 0);
        t[1][sq] = pawn_attackers_of(sq, 1);
        sq += 1;
    }
    t
}

const fn build_ray_table(dirs: [(i32, i32); 4]) -> [[u64; 64]; 4] {
    let mut t = [[0u64; 64]; 4];
    let mut d = 0;
    while d < 4 {
        let mut sq = 0;
        while sq < 64 {
            let f0 = (sq % 8) as i32;
            let r0 = (sq / 8) as i32;
            let mut att = 0u64;
            let mut f = f0 + dirs[d].0;
            let mut r = r0 + dirs[d].1;
            while f >= 0 && f < 8 && r >= 0 && r < 8 {
                att |= 1u64 << (r * 8 + f);
                f += dirs[d].0;
                r += dirs[d].1;
            }
            t[d][sq] = att;
            sq += 1;
        }
        d += 1;
    }
    t
}

pub(crate) static KNIGHT_ATTACKS: [u64; 64] = build_knight_table();
pub(crate) static KING_ATTACKS: [u64; 64] = build_king_table();
pub(crate) static PAWN_ATTACKERS: [[u64; 64]; 2] = build_pawn_attacker_table();

/// Directional ray tables. The first two directions of each set run toward
/// higher bit indices (blocker found via trailing_zeros), the last two toward
/// lower ones (leading_zeros).
pub(crate) static DIAG_RAYS: [[u64; 64]; 4] =
    build_ray_table([(1, 1), (-1, 1), (1, -1), (-1, -1)]);
pub(crate) static ORTHO_RAYS: [[u64; 64]; 4] =
    build_ray_table([(0, 1), (1, 0), (0, -1), (-1, 0)]);

/// Magic-bitboard slider lookup ("plain" magics). Multipliers were found
/// offline with a fixed-seed search; the attack table itself is built at
/// compile time, so there is no runtime init and no lazy-init check.
#[derive(Copy, Clone)]
struct Magic {
    mask: u64,
    magic: u64,
    offset: u32,
    shift: u32,
}

#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0x0080008040002018, 0x0040100040002001, 0x09000d0010200040, 0x8080080010008004,
    0xa280080002340080, 0x2500050024000208, 0x0280010000800200, 0x1100144380220100,
    0xa92080048c204002, 0x0802804003200080, 0x0108802000100089, 0x8060800800801002,
    0xa002000822000410, 0x950a001002000824, 0x1202000408010200, 0x02950008408a0100,
    0x40a0a18000814000, 0x0080848020004011, 0x0800828010002000, 0x0500420010200a00,
    0x0008818004000802, 0x0000808004000200, 0x0000040001020810, 0x0081120000442081,
    0x0522400180002090, 0x2800200040401000, 0x402004a180100481, 0x0880420200200810,
    0x000a040080800800, 0x080a000404001020, 0x0000010400820810, 0x4000d08200004c09,
    0x0100804000800020, 0xcc01028026004200, 0x0081002001004010, 0x0000800800801000,
    0x0009800401802800, 0x0002000902000410, 0x0000180144001022, 0x00008100c6002884,
    0x1200800040008024, 0x2000200050044000, 0x0090882200420010, 0x6a1810200a020040,
    0x0480040008008080, 0x0206001108160014, 0xc010040200010100, 0x000001018046000c,
    0x00b0400880042080, 0x0040002010080220, 0x0020004228110100, 0x0150000804004140,
    0x0028008004000980, 0x0801000804000300, 0x2c80210208900400, 0x0000040130408200,
    0x4001409100258001, 0x0005004002883021, 0x019041e003001019, 0x8000200900041001,
    0x4021001002040801, 0x0011000400080201, 0x1600010210408804, 0x2010810084003042,
];
#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0x0032482800818200, 0x0819022820450000, 0x4242108204880008, 0x8044404080410224,
    0x1407104000080100, 0x1b01100210004000, 0x0b0c0888841010c1, 0x08a0108201104020,
    0x00003888654c0410, 0x0010220202520a00, 0x4003048400820001, 0x1007022082000002,
    0x3c22040420040a18, 0x3440508820080030, 0x1204208090082100, 0x0000930501012000,
    0x0010842002d00100, 0x0202000404484208, 0x0008000100410602, 0x0088000404200800,
    0xa004100202020232, 0x0004200100a01002, 0x0201000208020200, 0x2022010022020200,
    0x80100410100410b0, 0x3010704848011100, 0x2000380010048320, 0x8068080000202120,
    0x0001010080104000, 0x4808004120806000, 0x5000821000823008, 0x0004010000504224,
    0x0088044010040808, 0x0812022000032814, 0xa000280800110202, 0x2882200802010105,
    0x0508082400024100, 0x8001010200030800, 0x4001260e00019820, 0x0c02040308507180,
    0x0880884441009000, 0x20a84202a0011000, 0x00010410820c1000, 0x4000002018000108,
    0x0080091124004a00, 0x0409010302010700, 0x0088128802040040, 0x1801010408801100,
    0x0184010190110800, 0x0000404208210000, 0x0000120908880000, 0x9006180020884020,
    0x900000400488410a, 0x0410218401020100, 0x0004d00428008100, 0x0002021404008400,
    0x0911410150222026, 0x0300202088041040, 0x0100a06124122800, 0x001c042000420200,
    0x0004018208030400, 0x0044040405080200, 0x040820200402b680, 0x0204200411020410,
];

const ROOK_DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (-1, 1), (1, -1), (-1, -1)];
/// Sum over squares of 2^popcount(mask): 102400 for rooks, 5248 for bishops.
const ROOK_TABLE_SIZE: usize = 102_400;
const SLIDER_TABLE_SIZE: usize = ROOK_TABLE_SIZE + 5_248;

/// Relevant-occupancy mask (ray squares excluding the board edge) when
/// `mask_only`, otherwise the exact attack set over occupancy `occ`.
const fn slide(sq: usize, occ: u64, dirs: &[(i32, i32); 4], mask_only: bool) -> u64 {
    let f0 = (sq % 8) as i32;
    let r0 = (sq / 8) as i32;
    let mut att = 0u64;
    let mut d = 0;
    while d < 4 {
        let (df, dr) = dirs[d];
        let mut f = f0 + df;
        let mut r = r0 + dr;
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            if mask_only {
                let nf = f + df;
                let nr = r + dr;
                if !(nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                    break;
                }
            }
            let bit = 1u64 << (r * 8 + f);
            att |= bit;
            if occ & bit != 0 {
                break;
            }
            f += df;
            r += dr;
        }
        d += 1;
    }
    att
}

const fn build_magics(magics: &[u64; 64], dirs: &[(i32, i32); 4], base: usize) -> [Magic; 64] {
    let mut out = [Magic { mask: 0, magic: 0, offset: 0, shift: 0 }; 64];
    let mut offset = base;
    let mut sq = 0;
    while sq < 64 {
        let mask = slide(sq, 0, dirs, true);
        let bits = mask.count_ones();
        out[sq] = Magic { mask, magic: magics[sq], offset: offset as u32, shift: 64 - bits };
        offset += 1usize << bits;
        sq += 1;
    }
    out
}

static ROOK_MAGIC: [Magic; 64] = build_magics(&ROOK_MAGICS, &ROOK_DIRS, 0);
static BISHOP_MAGIC: [Magic; 64] = build_magics(&BISHOP_MAGICS, &BISHOP_DIRS, ROOK_TABLE_SIZE);

const fn fill_table(
    t: &mut [u64; SLIDER_TABLE_SIZE],
    magics: &[Magic; 64],
    dirs: &[(i32, i32); 4],
) {
    let mut sq = 0;
    while sq < 64 {
        let m = magics[sq];
        // Carry-rippler enumeration of every subset of the mask.
        let mut sub = 0u64;
        loop {
            let idx = m.offset as usize + (sub.wrapping_mul(m.magic) >> m.shift) as usize;
            t[idx] = slide(sq, sub, dirs, false);
            sub = sub.wrapping_sub(m.mask) & m.mask;
            if sub == 0 {
                break;
            }
        }
        sq += 1;
    }
}

const fn build_slider_table() -> [u64; SLIDER_TABLE_SIZE] {
    let mut t = [0u64; SLIDER_TABLE_SIZE];
    fill_table(&mut t, &ROOK_MAGIC, &ROOK_DIRS);
    fill_table(&mut t, &BISHOP_MAGIC, &BISHOP_DIRS);
    t
}

#[allow(long_running_const_eval)]
static SLIDER_ATTACKS: [u64; SLIDER_TABLE_SIZE] = build_slider_table();

#[inline(always)]
fn magic_lookup(m: &Magic, occ: u64) -> u64 {
    let idx = m.offset as usize + ((occ & m.mask).wrapping_mul(m.magic) >> m.shift) as usize;
    SLIDER_ATTACKS[idx]
}

/// Sliding attacks from `sq` over occupancy `occ` along diagonal rays.
#[inline]
pub(crate) fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    magic_lookup(&BISHOP_MAGIC[sq], occ)
}

/// Sliding attacks from `sq` over occupancy `occ` along orthogonal rays.
#[inline]
pub(crate) fn rook_attacks(sq: usize, occ: u64) -> u64 {
    magic_lookup(&ROOK_MAGIC[sq], occ)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slow_slider(sq: usize, occ: u64, dirs: &[(i32, i32)]) -> u64 {
        let f0 = (sq % 8) as i32;
        let r0 = (sq / 8) as i32;
        let mut att = 0u64;
        for &(df, dr) in dirs {
            let mut f = f0 + df;
            let mut r = r0 + dr;
            while (0..8).contains(&f) && (0..8).contains(&r) {
                let bit = 1u64 << (r * 8 + f);
                att |= bit;
                if occ & bit != 0 {
                    break;
                }
                f += df;
                r += dr;
            }
        }
        att
    }

    #[test]
    fn slider_attacks_exhaustive_over_relevant_occupancies() {
        // Every subset of every square's relevant mask, plus random noise on
        // the edge/irrelevant squares that the magic lookup must ignore.
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        for sq in 0..64 {
            for (magics, dirs, f) in [
                (&ROOK_MAGIC, ROOK_DIRS, rook_attacks as fn(usize, u64) -> u64),
                (&BISHOP_MAGIC, BISHOP_DIRS, bishop_attacks as fn(usize, u64) -> u64),
            ] {
                let mask = magics[sq].mask;
                let mut sub = 0u64;
                loop {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    for occ in [sub, sub | (rng & !mask), sub | !mask] {
                        assert_eq!(f(sq, occ), slow_slider(sq, occ, &dirs), "sq {sq} occ {occ:#x}");
                    }
                    sub = sub.wrapping_sub(mask) & mask;
                    if sub == 0 {
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn slider_attacks_match_reference_scan() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..200_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let occ = rng;
            let sq = (rng >> 32) as usize % 64;
            assert_eq!(
                bishop_attacks(sq, occ),
                slow_slider(sq, occ, &[(1, 1), (-1, 1), (1, -1), (-1, -1)]),
                "bishop sq {sq} occ {occ:#x}"
            );
            assert_eq!(
                rook_attacks(sq, occ),
                slow_slider(sq, occ, &[(0, 1), (1, 0), (0, -1), (-1, 0)]),
                "rook sq {sq} occ {occ:#x}"
            );
        }
    }
}
