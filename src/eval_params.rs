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
    BishopPair = (13, 62),

    // Development (only applied while phase >= 12; mg only).
    DevUndevelopedMinor = (-18, 0),   // penalty per minor on its start square
    DevKingHomeUndeveloped = (-6, 0), // penalty: king at home with >= 2 undeveloped
    DevKingCastled = (18, 0),         // bonus: king on a castled square

    // Pawn structure.
    Doubled = (8, 8),                 // penalty per pawn on a file with > 1 own pawn
    Isolated = (5, 8),                // penalty per pawn with no own pawns on adjacent files
    Connected = (1, -3),              // bonus per phalanx/supported-chain pawn
    Backward = (3, 4),                // penalty per backward pawn

    // Passed pawns: per-rank tables, R2..R7 = relative rank of the passer.
    // Base bonus per rank.
    PassedR2 = (0, 35),
    PassedR3 = (-3, 20),
    PassedR4 = (-25, 38),
    PassedR5 = (1, 33),
    PassedR6 = (44, 8),
    PassedR7 = (12, 15),
    // Free path: every square from the stop square to promotion is empty.
    PassedFreeR2 = (-6, 9),
    PassedFreeR3 = (-11, 5),
    PassedFreeR4 = (-6, 12),
    PassedFreeR5 = (11, 11),
    PassedFreeR6 = (24, 7),
    PassedFreeR7 = (43, 55),
    // Safe push: stop square empty and not attacked by the enemy.
    PassedSafePushR2 = (-9, 9),
    PassedSafePushR3 = (1, 7),
    PassedSafePushR4 = (3, 11),
    PassedSafePushR5 = (-9, 31),
    PassedSafePushR6 = (10, 61),
    PassedSafePushR7 = (5, 130),
    // Blocked: an enemy piece stands on the stop square (bonus; tunes negative).
    PassedBlockedR2 = (-6, 12),
    PassedBlockedR3 = (-8, -6),
    PassedBlockedR4 = (-32, -4),
    PassedBlockedR5 = (-16, 5),
    PassedBlockedR6 = (-1, -12),
    PassedBlockedR7 = (-13, 26),
    // * min(own king distance to the stop square, 5) (tunes negative).
    PassedOwnKingDistR2 = (0, 0),
    PassedOwnKingDistR3 = (-1, -3),
    PassedOwnKingDistR4 = (7, -12),
    PassedOwnKingDistR5 = (-5, -17),
    PassedOwnKingDistR6 = (-6, -16),
    PassedOwnKingDistR7 = (-7, -24),
    // * min(enemy king distance to the stop square, 5).
    PassedEnemyKingDistR2 = (2, -4),
    PassedEnemyKingDistR3 = (0, 1),
    PassedEnemyKingDistR4 = (-6, 10),
    PassedEnemyKingDistR5 = (9, 20),
    PassedEnemyKingDistR6 = (1, 34),
    PassedEnemyKingDistR7 = (20, 30),
    // Connected: defended by an own pawn or with an own pawn beside it.
    PassedConnectedR2 = (-17, -36),
    PassedConnectedR3 = (5, -24),
    PassedConnectedR4 = (-1, -15),
    PassedConnectedR5 = (2, 19),
    PassedConnectedR6 = (31, 47),
    PassedConnectedR7 = (26, 83),
    PasserEnemyKingFar = (-3, 4),     // * (enemy king promo distance - remaining)
    PasserRookBehind = (10, 24),      // own rook behind the passer, same file
    PasserRookFront = (-6, -30),      // penalty: own rook in front of the passer

    // Candidate passers (no enemy pawn ahead on the same file).
    CandidateBase = (-23, -13),
    CandidateRank = (6, 3),
    CandidateLever = (-34, -22),
    CandidateLeverRank = (4, 12),
    CandidateSupported = (18, 12),

    // Pawn attacks on central squares (files c..f, relative ranks 2..4).
    PawnCenterControl = (12, 26),

    // Rooks.
    RookOpenFile = (47, -6),
    RookSemiOpenFile = (15, 13),
    RookSeventh = (1, 49),

    // Minor pieces.
    KnightOutpost = (28, 7),
    BishopOutpost = (28, 11),
    Mobility = (0, 2),                // per pseudo-legal move, all pieces
    BishopMobilityExtra = (2, 1),     // additional per bishop move

    // Trapped pieces: weight * severity, severity = 3 - mobility (rook: 4 - mobility).
    TrappedKnightCorner = (23, -6),
    TrappedKnightEdge = (2, 4),
    TrappedBishopEdge = (1, 10),
    TrappedBishopCornerExtra = (38, 28),
    TrappedRookByKing = (-1, 11),

    // Threats: per enemy piece attacked by a lower-value piece of ours.
    ThreatPawnOnMinor = (40, 28),     // knight/bishop attacked by a pawn
    ThreatPawnOnRook = (44, 28),
    ThreatPawnOnQueen = (33, 18),
    ThreatMinorOnRook = (45, 17),     // rook attacked by a knight/bishop
    ThreatMinorOnQueen = (42, 8),
    ThreatRookOnQueen = (48, 22),
    Hanging = (15, 24),               // non-pawn piece attacked by us, undefended

    // Endgame: mop-up with a >= 500 non-pawn material edge, * (14 - king distance).
    MopUpKingDistance = (25, -7),

    // Pawn endgames (no non-pawn material).
    PawnEndgameOpposition = (0, 27),
    RaceUnstoppable = (0, 56),        // + (8 - moves) * RaceUnstoppableSpeed
    RaceUnstoppableSpeed = (0, 35),
    RaceAhead = (0, 84),              // + (moves gap) * RaceAheadGap
    RaceAheadGap = (0, 115),

    // Single-rook endgames (one rook each, no queens).
    RookEgEnemyKingClose = (0, 1),    // * (7 - distance to enemy king)
    RookEgBehindPasser = (0, 3),
    RookEgFrontOfPasser = (0, 28),    // penalty
    RookEgBehindEnemyPasser = (0, 41),

    // King safety, per king (penalties are applied to the defending side).
    // Zone = the king square and its 8 neighbours.
    KsAttackPawn = (32, -12),         // penalty per enemy pawn attack on a zone square
    KsAttackKnight = (17, 2),         // penalty per (knight, zone square) attack pair
    KsAttackBishop = (25, 1),
    KsAttackRook = (22, -1),
    KsAttackQueen = (23, 10),
    // Interaction: with n >= 2 enemy pieces hitting the zone, an extra penalty
    // of weight * units, units = sum of N/B 2, R 3, Q 5 per attack pair.
    KsAttackers2 = (3, -1),
    KsAttackers3 = (5, -1),
    KsAttackers4 = (9, -5),           // 4 or more attackers
    // Penalty per square from which an enemy piece could give a safe check
    // (reachable by it, not defended by us, not occupied by the enemy).
    KsSafeCheckKnight = (61, -7),
    KsSafeCheckBishop = (14, 11),
    KsSafeCheckRook = (73, -9),
    KsSafeCheckQueen = (39, -17),
    // Pawn shield on the king file / adjacent files: nearest own pawn ahead of
    // the king, 1 or 2 ranks ahead (bonus) or missing / further (penalty).
    KsShieldKingFileD1 = (8, -8),
    KsShieldKingFileD2 = (7, 2),
    KsShieldKingFileMissing = (8, -8),
    KsShieldSideFileD1 = (-2, -3),
    KsShieldSideFileD2 = (3, 3),
    KsShieldSideFileMissing = (4, -9),
    // Pawn storm: nearest enemy pawn ahead of the king on the king file or an
    // adjacent file, by rank distance (penalty), or blocked by an own pawn.
    KsStormD1 = (-18, 15),
    KsStormD2 = (22, 11),
    KsStormD3 = (10, -1),
    KsStormD4 = (2, 1),
    KsStormBlocked = (0, 16),
    // Files at or next to the king without own pawns (penalty).
    KsOpenFile = (18, 20),            // no pawns of either side
    KsHalfOpenFile = (8, -3),         // enemy pawn present
    // Zone squares attacked by the enemy (penalty), and the subset not
    // defended by any own piece other than the king (extra penalty).
    KsZoneAttacked = (-26, -2),
    KsZoneWeak = (12, 1),
    // Enemy piece proximity: sum of (7 - distance) * (N/B/R 2, Q 3) (penalty).
    KsTropism = (4, -1),
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
