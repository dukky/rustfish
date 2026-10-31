//! Scalar evaluation weights used by `eval::evaluate()` and its helpers.
//!
//! Every entry is an (mg, eg) pair. Most terms enter the score linearly
//! (`score += weight * count`, White minus Black), which is what makes them
//! fittable by a Texel tuner alongside the PSQT. King safety is linear too:
//! its one nonlinearity (attack pressure growing with the number of
//! attackers) is expressed as per-attacker-count buckets whose coefficient is
//! the weighted attack count, so each bucket weight is still a plain slope.
//! Tempo, which is added after tapering, is kept as a plain constant below.
//!
//! Sign convention: the stored value is the magnitude; the call site decides
//! whether it is a bonus or a penalty (e.g. `Doubled` is subtracted). Values
//! are Texel-tuned jointly with src/psqt.rs by the `tune` command, which
//! rewrites only the numbers in the `eval_params!` block; a tuned value may be
//! negative, which simply flips the term.

macro_rules! eval_params {
    ($($id:ident = ($mg:expr, $eg:expr)),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(usize)]
        pub(crate) enum P {
            $($id,)*
        }

        pub(crate) const PARAM_COUNT: usize = [$(stringify!($id)),*].len();

        /// (mg, eg) weight for each `P` variant, indexed by `P as usize`.
        pub(crate) const PARAMS: [(i32, i32); PARAM_COUNT] = [$(($mg, $eg)),*];

        /// Human-readable name for each `P` variant, for tuner output.
        #[allow(dead_code)]
        pub(crate) const PARAM_NAMES: [&str; PARAM_COUNT] = [$(stringify!($id)),*];
    };
}

eval_params! {
    // Bishop pair, per side with >= 2 bishops.
    BishopPair = (14, 67),

    // Development (only applied while phase >= 12; mg only).
    DevUndevelopedMinor = (-6, 0),    // penalty per minor on its start square
    DevKingHomeUndeveloped = (1, 0),  // penalty: king at home with >= 2 undeveloped
    DevKingCastled = (16, 0),         // bonus: king on a castled square

    // Pawn structure.
    Doubled = (2, 7),                 // penalty per pawn on a file with > 1 own pawn
    Isolated = (4, 10),               // penalty per pawn with no own pawns on adjacent files
    Connected = (1, -6),              // bonus per phalanx/supported-chain pawn
    Backward = (7, 5),                // penalty per backward pawn

    // Passed pawns: base + rank * rank_weight (rank is relative 0..7).
    PassedBase = (-22, -2),
    PassedRank = (0, 11),
    PassedSupportedBase = (-13, -69), // extra when defended by own pawn
    PassedSupportedRank = (5, 21),
    PasserOwnKingNear = (6, 7),       // * (3 - king_distance) when distance <= 2
    PasserEnemyKingFar = (2, 28),     // * (enemy king promo distance - remaining)
    PasserEnemyKingAdjacent = (19, 74), // penalty when enemy king touches the passer
    PasserRookBehind = (8, 10),       // own rook behind the passer, same file
    PasserRookFront = (-8, -26),      // penalty: own rook in front of the passer

    // Candidate passers (no enemy pawn ahead on the same file).
    CandidateBase = (-16, -13),
    CandidateRank = (6, 5),
    CandidateLever = (-32, -17),
    CandidateLeverRank = (4, 10),
    CandidateSupported = (16, 11),

    // Pawn attacks on central squares (files c..f, relative ranks 2..4).
    PawnCenterControl = (7, 23),

    // Rooks.
    RookOpenFile = (47, -4),
    RookSemiOpenFile = (11, 20),
    RookSeventh = (6, 33),

    // Minor pieces.
    KnightOutpost = (38, 6),
    BishopOutpost = (34, 11),
    Mobility = (1, 3),                // per pseudo-legal move, all pieces
    BishopMobilityExtra = (2, 0),     // additional per bishop move

    // Trapped pieces: weight * severity, severity = 3 - mobility (rook: 4 - mobility).
    TrappedKnightCorner = (8, -10),
    TrappedKnightEdge = (0, 2),
    TrappedBishopEdge = (1, 7),
    TrappedBishopCornerExtra = (15, 19),
    TrappedRookByKing = (-3, 9),

    // Endgame: mop-up with a >= 500 non-pawn material edge, * (14 - king distance).
    MopUpKingDistance = (45, -9),

    // Pawn endgames (no non-pawn material).
    PawnEndgameOpposition = (0, 25),
    RaceUnstoppable = (0, 81),        // + (8 - moves) * RaceUnstoppableSpeed
    RaceUnstoppableSpeed = (0, 33),
    RaceAhead = (0, 63),              // + (moves gap) * RaceAheadGap
    RaceAheadGap = (0, 66),

    // Single-rook endgames (one rook each, no queens).
    RookEgEnemyKingClose = (0, 0),    // * (7 - distance to enemy king)
    RookEgBehindPasser = (0, 4),
    RookEgFrontOfPasser = (0, 31),    // penalty
    RookEgBehindEnemyPasser = (0, 23),

    // King safety, per king (penalties are applied to the defending side).
    // Zone = the king square and its 8 neighbours.
    KsAttackPawn = (3, 0),            // penalty per enemy pawn attack on a zone square
    KsAttackKnight = (6, 1),          // penalty per (knight, zone square) attack pair
    KsAttackBishop = (6, 1),
    KsAttackRook = (9, 1),
    KsAttackQueen = (15, 2),
    // Interaction: with n >= 2 enemy pieces hitting the zone, an extra penalty
    // of weight * units, units = sum of N/B 2, R 3, Q 5 per attack pair.
    KsAttackers2 = (1, 0),
    KsAttackers3 = (2, 0),
    KsAttackers4 = (3, 0),            // 4 or more attackers
    // Penalty per square from which an enemy piece could give a safe check
    // (reachable by it, not defended by us, not occupied by the enemy).
    KsSafeCheckKnight = (10, 2),
    KsSafeCheckBishop = (6, 1),
    KsSafeCheckRook = (10, 2),
    KsSafeCheckQueen = (8, 2),
    // Pawn shield on the king file / adjacent files: nearest own pawn ahead of
    // the king, 1 or 2 ranks ahead (bonus) or missing / further (penalty).
    KsShieldKingFileD1 = (10, 2),
    KsShieldKingFileD2 = (5, 1),
    KsShieldKingFileMissing = (8, 1),
    KsShieldSideFileD1 = (10, 2),
    KsShieldSideFileD2 = (5, 1),
    KsShieldSideFileMissing = (8, 1),
    // Pawn storm: nearest enemy pawn ahead of the king on the king file or an
    // adjacent file, by rank distance (penalty), or blocked by an own pawn.
    KsStormD1 = (0, 0),
    KsStormD2 = (10, 0),
    KsStormD3 = (6, 0),
    KsStormD4 = (3, 0),
    KsStormBlocked = (2, 0),
    // Files at or next to the king without own pawns (penalty).
    KsOpenFile = (14, 2),             // no pawns of either side
    KsHalfOpenFile = (22, 4),         // enemy pawn present
    // Zone squares attacked by the enemy (penalty), and the subset not
    // defended by any own piece other than the king (extra penalty).
    KsZoneAttacked = (6, 1),
    KsZoneWeak = (4, 1),
    // Enemy piece proximity: sum of (7 - distance) * (N/B/R 2, Q 3) (penalty).
    KsTropism = (2, 0),
}

/// mg weight of a parameter.
#[inline(always)]
pub(crate) const fn mg(p: P) -> i32 {
    PARAMS[p as usize].0
}

/// eg weight of a parameter.
#[inline(always)]
pub(crate) const fn eg(p: P) -> i32 {
    PARAMS[p as usize].1
}

/// Side-to-move bonus, added after tapering.
pub(crate) const TEMPO: i32 = 10;
