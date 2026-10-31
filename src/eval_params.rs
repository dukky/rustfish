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

    // Passed pawns: per-rank tables, R2..R7 = relative rank of the passer.
    // Base bonus per rank.
    PassedR2 = (-22, 9),
    PassedR3 = (-22, 20),
    PassedR4 = (-22, 31),
    PassedR5 = (-22, 42),
    PassedR6 = (-22, 53),
    PassedR7 = (-22, 64),
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
    PassedConnectedR2 = (-8, -48),
    PassedConnectedR3 = (-3, -27),
    PassedConnectedR4 = (2, -6),
    PassedConnectedR5 = (7, 15),
    PassedConnectedR6 = (12, 36),
    PassedConnectedR7 = (17, 57),
    PasserEnemyKingFar = (2, 28),     // * (enemy king promo distance - remaining)
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
