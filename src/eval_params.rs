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
    BishopPair = (14, 63),

    // Development (only applied while phase >= 12; mg only).
    DevUndevelopedMinor = (-15, 0),   // penalty per minor on its start square
    DevKingHomeUndeveloped = (-6, 0), // penalty: king at home with >= 2 undeveloped
    DevKingCastled = (17, 0),         // bonus: king on a castled square

    // Pawn structure.
    Doubled = (6, 10),                // penalty per pawn on a file with > 1 own pawn
    Isolated = (5, 8),                // penalty per pawn with no own pawns on adjacent files
    Connected = (1, -3),              // bonus per phalanx/supported-chain pawn
    Backward = (3, 4),                // penalty per backward pawn

    // Passed pawns: per-rank tables, R2..R7 = relative rank of the passer.
    // Base bonus per rank.
    PassedR2 = (-24, 5),
    PassedR3 = (-19, 16),
    PassedR4 = (-14, 27),
    PassedR5 = (-9, 38),
    PassedR6 = (-4, 49),
    PassedR7 = (1, 60),
    // Free path: every square from the stop square to promotion is empty.
    PassedFreeR2 = (0, 0),
    PassedFreeR3 = (0, 2),
    PassedFreeR4 = (0, 5),
    PassedFreeR5 = (0, 10),
    PassedFreeR6 = (0, 20),
    PassedFreeR7 = (0, 30),
    // Safe push: stop square empty and not attacked by the enemy.
    PassedSafePushR2 = (0, 0),
    PassedSafePushR3 = (0, 2),
    PassedSafePushR4 = (0, 5),
    PassedSafePushR5 = (0, 10),
    PassedSafePushR6 = (0, 20),
    PassedSafePushR7 = (0, 30),
    // Blocked: an enemy piece stands on the stop square (bonus; tunes negative).
    PassedBlockedR2 = (0, 0),
    PassedBlockedR3 = (0, 0),
    PassedBlockedR4 = (0, -5),
    PassedBlockedR5 = (0, -10),
    PassedBlockedR6 = (0, -15),
    PassedBlockedR7 = (0, -20),
    // * min(own king distance to the stop square, 5) (tunes negative).
    PassedOwnKingDistR2 = (0, 0),
    PassedOwnKingDistR3 = (0, 0),
    PassedOwnKingDistR4 = (0, -2),
    PassedOwnKingDistR5 = (0, -4),
    PassedOwnKingDistR6 = (0, -6),
    PassedOwnKingDistR7 = (0, -8),
    // * min(enemy king distance to the stop square, 5).
    PassedEnemyKingDistR2 = (0, 0),
    PassedEnemyKingDistR3 = (0, 0),
    PassedEnemyKingDistR4 = (0, 5),
    PassedEnemyKingDistR5 = (0, 10),
    PassedEnemyKingDistR6 = (0, 15),
    PassedEnemyKingDistR7 = (0, 20),
    // Connected: defended by an own pawn or with an own pawn beside it.
    PassedConnectedR2 = (-8, -57),
    PassedConnectedR3 = (-4, -34),
    PassedConnectedR4 = (0, -11),
    PassedConnectedR5 = (4, 12),
    PassedConnectedR6 = (8, 35),
    PassedConnectedR7 = (12, 58),
    PasserEnemyKingFar = (0, 25),     // * (enemy king promo distance - remaining)
    PasserRookBehind = (17, 5),       // own rook behind the passer, same file
    PasserRookFront = (-11, -26),     // penalty: own rook in front of the passer

    // Candidate passers (no enemy pawn ahead on the same file).
    CandidateBase = (-19, -15),
    CandidateRank = (5, 6),
    CandidateLever = (-35, -20),
    CandidateLeverRank = (4, 12),
    CandidateSupported = (17, 10),

    // Pawn attacks on central squares (files c..f, relative ranks 2..4).
    PawnCenterControl = (10, 28),

    // Rooks.
    RookOpenFile = (49, -4),
    RookSemiOpenFile = (15, 20),
    RookSeventh = (6, 44),

    // Minor pieces.
    KnightOutpost = (35, 7),
    BishopOutpost = (34, 10),
    Mobility = (0, 2),                // per pseudo-legal move, all pieces
    BishopMobilityExtra = (2, 1),     // additional per bishop move

    // Trapped pieces: weight * severity, severity = 3 - mobility (rook: 4 - mobility).
    TrappedKnightCorner = (23, -12),
    TrappedKnightEdge = (2, 4),
    TrappedBishopEdge = (1, 10),
    TrappedBishopCornerExtra = (30, 25),
    TrappedRookByKing = (-1, 11),

    // Threats: per enemy piece attacked by a lower-value piece of ours.
    ThreatPawnOnMinor = (30, 20),     // knight/bishop attacked by a pawn
    ThreatPawnOnRook = (45, 25),
    ThreatPawnOnQueen = (50, 30),
    ThreatMinorOnRook = (30, 20),     // rook attacked by a knight/bishop
    ThreatMinorOnQueen = (35, 20),
    ThreatRookOnQueen = (30, 15),
    Hanging = (25, 15),               // non-pawn piece attacked by us, undefended

    // Endgame: mop-up with a >= 500 non-pawn material edge, * (14 - king distance).
    MopUpKingDistance = (37, -10),

    // Pawn endgames (no non-pawn material).
    PawnEndgameOpposition = (0, 28),
    RaceUnstoppable = (0, 67),        // + (8 - moves) * RaceUnstoppableSpeed
    RaceUnstoppableSpeed = (0, 36),
    RaceAhead = (0, 77),              // + (moves gap) * RaceAheadGap
    RaceAheadGap = (0, 100),

    // Single-rook endgames (one rook each, no queens).
    RookEgEnemyKingClose = (0, 1),    // * (7 - distance to enemy king)
    RookEgBehindPasser = (0, 3),
    RookEgFrontOfPasser = (0, 37),    // penalty
    RookEgBehindEnemyPasser = (0, 20),

    // King safety, per king (penalties are applied to the defending side).
    // Zone = the king square and its 8 neighbours.
    KsAttackPawn = (24, -10),         // penalty per enemy pawn attack on a zone square
    KsAttackKnight = (14, 1),         // penalty per (knight, zone square) attack pair
    KsAttackBishop = (21, 2),
    KsAttackRook = (17, 0),
    KsAttackQueen = (19, 12),
    // Interaction: with n >= 2 enemy pieces hitting the zone, an extra penalty
    // of weight * units, units = sum of N/B 2, R 3, Q 5 per attack pair.
    KsAttackers2 = (3, -1),
    KsAttackers3 = (5, -1),
    KsAttackers4 = (10, -5),          // 4 or more attackers
    // Penalty per square from which an enemy piece could give a safe check
    // (reachable by it, not defended by us, not occupied by the enemy).
    KsSafeCheckKnight = (61, -7),
    KsSafeCheckBishop = (14, 12),
    KsSafeCheckRook = (69, -7),
    KsSafeCheckQueen = (41, -20),
    // Pawn shield on the king file / adjacent files: nearest own pawn ahead of
    // the king, 1 or 2 ranks ahead (bonus) or missing / further (penalty).
    KsShieldKingFileD1 = (8, -9),
    KsShieldKingFileD2 = (7, 3),
    KsShieldKingFileMissing = (8, -8),
    KsShieldSideFileD1 = (-3, -5),
    KsShieldSideFileD2 = (3, 3),
    KsShieldSideFileMissing = (4, -7),
    // Pawn storm: nearest enemy pawn ahead of the king on the king file or an
    // adjacent file, by rank distance (penalty), or blocked by an own pawn.
    KsStormD1 = (-17, 0),
    KsStormD2 = (25, -5),
    KsStormD3 = (10, -4),
    KsStormD4 = (2, 0),
    KsStormBlocked = (1, 15),
    // Files at or next to the king without own pawns (penalty).
    KsOpenFile = (19, 18),            // no pawns of either side
    KsHalfOpenFile = (9, -7),         // enemy pawn present
    // Zone squares attacked by the enemy (penalty), and the subset not
    // defended by any own piece other than the king (extra penalty).
    KsZoneAttacked = (-22, -4),
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
