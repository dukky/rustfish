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
    BishopPair = (13, 66),

    // Development (only applied while phase >= 12; mg only).
    DevUndevelopedMinor = (-21, 0),   // penalty per minor on its start square
    DevKingHomeUndeveloped = (-1, 0), // penalty: king at home with >= 2 undeveloped
    DevKingCastled = (18, 0),         // bonus: king on a castled square

    // Pawn structure.
    Doubled = (8, 7),                 // penalty per pawn on a file with > 1 own pawn
    Isolated = (5, 7),                // penalty per pawn with no own pawns on adjacent files
    Connected = (1, -2),              // bonus per phalanx/supported-chain pawn
    Backward = (2, 5),                // penalty per backward pawn

    // Passed pawns: per-rank tables, R2..R7 = relative rank of the passer.
    // Base bonus per rank.
    PassedR2 = (13, 37),
    PassedR3 = (8, 17),
    PassedR4 = (-30, 38),
    PassedR5 = (10, 29),
    PassedR6 = (91, -4),
    PassedR7 = (25, -10),
    // Free path: every square from the stop square to promotion is empty.
    PassedFreeR2 = (-7, 11),
    PassedFreeR3 = (-14, 7),
    PassedFreeR4 = (-7, 13),
    PassedFreeR5 = (14, 11),
    PassedFreeR6 = (35, 5),
    PassedFreeR7 = (71, 70),
    // Safe push: stop square empty and not attacked by the enemy.
    PassedSafePushR2 = (-14, 10),
    PassedSafePushR3 = (-2, 7),
    PassedSafePushR4 = (-1, 12),
    PassedSafePushR5 = (-15, 33),
    PassedSafePushR6 = (8, 65),
    PassedSafePushR7 = (-6, 148),
    // Blocked: an enemy piece stands on the stop square (bonus; tunes negative).
    PassedBlockedR2 = (-12, 18),
    PassedBlockedR3 = (-11, -5),
    PassedBlockedR4 = (-40, -1),
    PassedBlockedR5 = (-20, 7),
    PassedBlockedR6 = (-1, -11),
    PassedBlockedR7 = (-20, 51),
    // * min(own king distance to the stop square, 5) (tunes negative).
    PassedOwnKingDistR2 = (-1, 0),
    PassedOwnKingDistR3 = (0, -3),
    PassedOwnKingDistR4 = (9, -13),
    PassedOwnKingDistR5 = (-7, -17),
    PassedOwnKingDistR6 = (-15, -15),
    PassedOwnKingDistR7 = (-14, -23),
    // * min(enemy king distance to the stop square, 5).
    PassedEnemyKingDistR2 = (0, -5),
    PassedEnemyKingDistR3 = (-3, 2),
    PassedEnemyKingDistR4 = (-6, 11),
    PassedEnemyKingDistR5 = (9, 21),
    PassedEnemyKingDistR6 = (-1, 36),
    PassedEnemyKingDistR7 = (21, 31),
    // Connected: defended by an own pawn or with an own pawn beside it.
    PassedConnectedR2 = (-19, -33),
    PassedConnectedR3 = (5, -23),
    PassedConnectedR4 = (-2, -14),
    PassedConnectedR5 = (2, 19),
    PassedConnectedR6 = (40, 46),
    PassedConnectedR7 = (35, 89),
    PasserEnemyKingFar = (-3, 2),     // * (enemy king promo distance - remaining)
    PasserRookBehind = (9, 24),       // own rook behind the passer, same file
    PasserRookFront = (-2, -30),      // penalty: own rook in front of the passer

    // Candidate passers (no enemy pawn ahead on the same file).
    CandidateBase = (-23, -12),
    CandidateRank = (5, 4),
    CandidateLever = (-33, -23),
    CandidateLeverRank = (4, 12),
    CandidateSupported = (18, 12),

    // Pawn attacks on central squares (files c..f, relative ranks 2..4).
    PawnCenterControl = (15, 24),

    // Rooks.
    RookOpenFile = (42, -10),
    RookSemiOpenFile = (11, 12),
    RookSeventh = (-5, 50),

    // Minor pieces.
    KnightOutpost = (27, 2),
    BishopOutpost = (25, 12),

    // Mobility: per attacked "safe" square, i.e. not attacked by an enemy
    // pawn and not occupied by an own pawn or the own king.
    KnightMobility = (5, 4),
    BishopMobility = (6, 3),
    RookMobility = (3, 3),
    QueenMobility = (3, 3),

    // Trapped pieces: weight * severity, severity = 3 - mobility (rook: 4 - mobility).
    TrappedKnightCorner = (19, -6),
    TrappedKnightEdge = (-2, 6),
    TrappedBishopEdge = (-1, 11),
    TrappedBishopCornerExtra = (44, 28),
    TrappedRookByKing = (-2, 8),

    // Threats: per enemy piece attacked by a lower-value piece of ours.
    ThreatPawnOnMinor = (43, 30),     // knight/bishop attacked by a pawn
    ThreatPawnOnRook = (45, 30),
    ThreatPawnOnQueen = (34, 12),
    ThreatMinorOnRook = (45, 17),     // rook attacked by a knight/bishop
    ThreatMinorOnQueen = (44, 0),
    ThreatRookOnQueen = (52, 24),
    Hanging = (14, 24),               // non-pawn piece attacked by us, undefended

    // Endgame: mop-up with a >= 500 non-pawn material edge, * (14 - king distance).
    MopUpKingDistance = (22, -7),

    // Pawn endgames (no non-pawn material).
    PawnEndgameOpposition = (0, 27),
    RaceUnstoppable = (0, 46),        // + (8 - moves) * RaceUnstoppableSpeed
    RaceUnstoppableSpeed = (0, 37),
    RaceAhead = (0, 89),              // + (moves gap) * RaceAheadGap
    RaceAheadGap = (0, 126),

    // Single-rook endgames (one rook each, no queens).
    RookEgEnemyKingClose = (0, 1),    // * (7 - distance to enemy king)
    RookEgBehindPasser = (0, 0),
    RookEgFrontOfPasser = (0, 26),    // penalty
    RookEgBehindEnemyPasser = (0, 42),

    // King safety, per king (penalties are applied to the defending side).
    // Zone = the king square and its 8 neighbours.
    KsAttackPawn = (36, -12),         // penalty per enemy pawn attack on a zone square
    KsAttackKnight = (17, 1),         // penalty per (knight, zone square) attack pair
    KsAttackBishop = (25, 2),
    KsAttackRook = (23, -2),
    KsAttackQueen = (23, 9),
    // Interaction: with n >= 2 enemy pieces hitting the zone, an extra penalty
    // of weight * units, units = sum of N/B 2, R 3, Q 5 per attack pair.
    KsAttackers2 = (3, -2),
    KsAttackers3 = (5, -1),
    KsAttackers4 = (9, -6),           // 4 or more attackers
    // Penalty per square from which an enemy piece could give a safe check
    // (reachable by it, not defended by us, not occupied by the enemy).
    KsSafeCheckKnight = (59, -8),
    KsSafeCheckBishop = (11, 11),
    KsSafeCheckRook = (74, -12),
    KsSafeCheckQueen = (36, -18),
    // Pawn shield on the king file / adjacent files: nearest own pawn ahead of
    // the king, 1 or 2 ranks ahead (bonus) or missing / further (penalty).
    KsShieldKingFileD1 = (7, -9),
    KsShieldKingFileD2 = (7, 3),
    KsShieldKingFileMissing = (7, -8),
    KsShieldSideFileD1 = (-1, 0),
    KsShieldSideFileD2 = (4, 7),
    KsShieldSideFileMissing = (2, -13),
    // Pawn storm: nearest enemy pawn ahead of the king on the king file or an
    // adjacent file, by rank distance (penalty), or blocked by an own pawn.
    KsStormD1 = (-19, 17),
    KsStormD2 = (20, 12),
    KsStormD3 = (11, -1),
    KsStormD4 = (2, 1),
    KsStormBlocked = (-1, 16),
    // Files at or next to the king without own pawns (penalty).
    KsOpenFile = (20, 19),            // no pawns of either side
    KsHalfOpenFile = (9, -3),         // enemy pawn present
    // Zone squares attacked by the enemy (penalty), and the subset not
    // defended by any own piece other than the king (extra penalty).
    KsZoneAttacked = (-28, -2),
    KsZoneWeak = (12, 1),
    // Enemy piece proximity: sum of (7 - distance) * (N/B/R 2, Q 3) (penalty).
    KsTropism = (3, -1),
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
