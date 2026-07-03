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

/// Sliding attacks from `sq` over occupancy `occ` along diagonal rays.
pub(crate) fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    let mut att = 0u64;
    // Rays toward higher bits: first blocker via trailing_zeros.
    for d in 0..2 {
        let ray = DIAG_RAYS[d][sq];
        let blockers = ray & occ;
        if blockers != 0 {
            let first = blockers.trailing_zeros() as usize;
            att |= ray & !DIAG_RAYS[d][first];
        } else {
            att |= ray;
        }
    }
    // Rays toward lower bits: first blocker via leading_zeros.
    for d in 2..4 {
        let ray = DIAG_RAYS[d][sq];
        let blockers = ray & occ;
        if blockers != 0 {
            let first = 63 - blockers.leading_zeros() as usize;
            att |= ray & !DIAG_RAYS[d][first];
        } else {
            att |= ray;
        }
    }
    att
}

/// Sliding attacks from `sq` over occupancy `occ` along orthogonal rays.
pub(crate) fn rook_attacks(sq: usize, occ: u64) -> u64 {
    let mut att = 0u64;
    for d in 0..2 {
        let ray = ORTHO_RAYS[d][sq];
        let blockers = ray & occ;
        if blockers != 0 {
            let first = blockers.trailing_zeros() as usize;
            att |= ray & !ORTHO_RAYS[d][first];
        } else {
            att |= ray;
        }
    }
    for d in 2..4 {
        let ray = ORTHO_RAYS[d][sq];
        let blockers = ray & occ;
        if blockers != 0 {
            let first = 63 - blockers.leading_zeros() as usize;
            att |= ray & !ORTHO_RAYS[d][first];
        } else {
            att |= ray;
        }
    }
    att
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
    fn slider_attacks_match_reference_scan() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..2000 {
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
