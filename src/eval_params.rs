//! Scalar evaluation weights used by `eval::evaluate()` and its helpers.
//!
//! Every entry is an (mg, eg) pair. Most terms enter the score linearly
//! (`score += weight * count`, White minus Black), which is what makes them
//! fittable by a Texel tuner alongside the PSQT. The few weights that feed
//! nonlinear machinery (king safety, which is folded into eg as `/ 6`, and
//! tempo, which is added after tapering) are kept as plain constants below.
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
    BishopPair = (15, 64),

    // Development (only applied while phase >= 12; mg only).
    DevUndevelopedMinor = (-12, 0),   // penalty per minor on its start square
    DevKingHomeUndeveloped = (-4, 0), // penalty: king at home with >= 2 undeveloped
    DevKingCastled = (23, 0),         // bonus: king on a castled square

    // Pawn structure.
    Doubled = (6, 6),                 // penalty per pawn on a file with > 1 own pawn
    Isolated = (4, 10),               // penalty per pawn with no own pawns on adjacent files
    Connected = (0, -6),              // bonus per phalanx/supported-chain pawn
    Backward = (2, 6),                // penalty per backward pawn

    // Passed pawns: per-rank tables, R2..R7 = relative rank of the passer.
    // Base bonus per rank.
    PassedR2 = (3, 38),
    PassedR3 = (-10, 14),
    PassedR4 = (-28, 24),
    PassedR5 = (-8, 19),
    PassedR6 = (28, -9),
    PassedR7 = (-11, 14),
    // Free path: every square from the stop square to promotion is empty.
    PassedFreeR2 = (-6, 13),
    PassedFreeR3 = (-11, 8),
    PassedFreeR4 = (-4, 14),
    PassedFreeR5 = (13, 12),
    PassedFreeR6 = (25, 7),
    PassedFreeR7 = (45, 51),
    // Safe push: stop square empty and not attacked by the enemy.
    PassedSafePushR2 = (-9, 8),
    PassedSafePushR3 = (3, 7),
    PassedSafePushR4 = (0, 12),
    PassedSafePushR5 = (-13, 35),
    PassedSafePushR6 = (6, 68),
    PassedSafePushR7 = (4, 135),
    // Blocked: an enemy piece stands on the stop square (bonus; tunes negative).
    PassedBlockedR2 = (-6, 12),
    PassedBlockedR3 = (-7, -8),
    PassedBlockedR4 = (-31, -6),
    PassedBlockedR5 = (-17, 5),
    PassedBlockedR6 = (-2, -14),
    PassedBlockedR7 = (-15, 19),
    // * min(own king distance to the stop square, 5) (tunes negative).
    PassedOwnKingDistR2 = (1, -1),
    PassedOwnKingDistR3 = (1, -4),
    PassedOwnKingDistR4 = (9, -14),
    PassedOwnKingDistR5 = (-3, -18),
    PassedOwnKingDistR6 = (0, -17),
    PassedOwnKingDistR7 = (-2, -26),
    // * min(enemy king distance to the stop square, 5).
    PassedEnemyKingDistR2 = (0, -3),
    PassedEnemyKingDistR3 = (-1, 4),
    PassedEnemyKingDistR4 = (-8, 17),
    PassedEnemyKingDistR5 = (7, 28),
    PassedEnemyKingDistR6 = (-2, 45),
    PassedEnemyKingDistR7 = (23, 41),
    // Connected: defended by an own pawn or with an own pawn beside it.
    PassedConnectedR2 = (-20, -39),
    PassedConnectedR3 = (4, -22),
    PassedConnectedR4 = (0, -13),
    PassedConnectedR5 = (5, 22),
    PassedConnectedR6 = (35, 51),
    PassedConnectedR7 = (31, 85),
    PasserEnemyKingFar = (-3, 0),     // * (enemy king promo distance - remaining)
    PasserRookBehind = (10, 24),      // own rook behind the passer, same file
    PasserRookFront = (-7, -35),      // penalty: own rook in front of the passer

    // Candidate passers (no enemy pawn ahead on the same file).
    CandidateBase = (-20, -12),
    CandidateRank = (6, 3),
    CandidateLever = (-41, -25),
    CandidateLeverRank = (6, 14),
    CandidateSupported = (18, 13),

    // Pawn attacks on central squares (files c..f, relative ranks 2..4).
    PawnCenterControl = (9, 26),

    // Rooks.
    RookOpenFile = (49, -6),
    RookSemiOpenFile = (16, 13),
    RookSeventh = (4, 43),

    // Minor pieces.
    KnightOutpost = (38, 6),
    BishopOutpost = (36, 12),
    Mobility = (0, 3),                // per pseudo-legal move, all pieces
    BishopMobilityExtra = (2, 0),     // additional per bishop move

    // Trapped pieces: weight * severity, severity = 3 - mobility (rook: 4 - mobility).
    TrappedKnightCorner = (22, -4),
    TrappedKnightEdge = (2, 3),
    TrappedBishopEdge = (0, 11),
    TrappedBishopCornerExtra = (29, 25),
    TrappedRookByKing = (-3, 12),

    // Endgame: mop-up with a >= 500 non-pawn material edge, * (14 - king distance).
    MopUpKingDistance = (32, -8),

    // Pawn endgames (no non-pawn material).
    PawnEndgameOpposition = (0, 26),
    RaceUnstoppable = (0, 70),        // + (8 - moves) * RaceUnstoppableSpeed
    RaceUnstoppableSpeed = (0, 34),
    RaceAhead = (0, 74),              // + (moves gap) * RaceAheadGap
    RaceAheadGap = (0, 92),

    // Single-rook endgames (one rook each, no queens).
    RookEgEnemyKingClose = (0, -1),   // * (7 - distance to enemy king)
    RookEgBehindPasser = (0, 5),
    RookEgFrontOfPasser = (0, 28),    // penalty
    RookEgBehindEnemyPasser = (0, 44),
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

// --- King safety (nonlinear: the per-king sum enters mg fully and eg as / 6,
// and attack units are scaled by enemy material). Single-valued weights. ---
pub(crate) const KS_SHIELD_PAWN: i32 = 10;
pub(crate) const KS_SHIELD_MISSING: i32 = 8;
pub(crate) const KS_OPEN_FILE: i32 = 14; // no pawns of either side on the file
pub(crate) const KS_HALF_OPEN_FILE: i32 = 22; // no own pawn, enemy pawn present
pub(crate) const KS_ATTACKED_SQUARE: i32 = 6;
pub(crate) const KS_EG_DIVISOR: i32 = 6;

/// Side-to-move bonus, added after tapering.
pub(crate) const TEMPO: i32 = 10;
