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
    BishopPair = (16, 65),

    // Development (only applied while phase >= 12; mg only).
    DevUndevelopedMinor = (-14, 0),   // penalty per minor on its start square
    DevKingHomeUndeveloped = (2, 0),  // penalty: king at home with >= 2 undeveloped
    DevKingCastled = (22, 0),         // bonus: king on a castled square

    // Pawn structure.
    Doubled = (6, 7),                 // penalty per pawn on a file with > 1 own pawn
    Isolated = (5, 10),               // penalty per pawn with no own pawns on adjacent files
    Connected = (0, -6),              // bonus per phalanx/supported-chain pawn
    Backward = (2, 6),                // penalty per backward pawn

    // Passed pawns: base + rank * rank_weight (rank is relative 0..7).
    PassedBase = (-31, 1),
    PassedRank = (6, 10),
    PassedSupportedBase = (-14, -74), // extra when defended by own pawn
    PassedSupportedRank = (5, 22),
    PasserOwnKingNear = (8, 6),       // * (3 - king_distance) when distance <= 2
    PasserEnemyKingFar = (-3, 29),    // * (enemy king promo distance - remaining)
    PasserEnemyKingAdjacent = (27, 74), // penalty when enemy king touches the passer
    PasserRookBehind = (17, 1),       // own rook behind the passer, same file
    PasserRookFront = (-10, -27),     // penalty: own rook in front of the passer

    // Candidate passers (no enemy pawn ahead on the same file).
    CandidateBase = (-17, -18),
    CandidateRank = (4, 5),
    CandidateLever = (-37, -22),
    CandidateLeverRank = (6, 12),
    CandidateSupported = (17, 11),

    // Pawn attacks on central squares (files c..f, relative ranks 2..4).
    PawnCenterControl = (11, 27),

    // Rooks.
    RookOpenFile = (43, -6),
    RookSemiOpenFile = (10, 17),
    RookSeventh = (4, 39),

    // Minor pieces.
    KnightOutpost = (37, -1),
    BishopOutpost = (34, 10),

    // Mobility: per attacked "safe" square, i.e. not attacked by an enemy
    // pawn and not occupied by an own pawn or the own king.
    KnightMobility = (5, 4),
    BishopMobility = (6, 3),
    RookMobility = (3, 2),
    QueenMobility = (3, 3),

    // Trapped pieces: weight * severity, severity = 3 - mobility (rook: 4 - mobility).
    TrappedKnightCorner = (23, -13),
    TrappedKnightEdge = (-1, 7),
    TrappedBishopEdge = (-2, 13),
    TrappedBishopCornerExtra = (30, 24),
    TrappedRookByKing = (-5, 10),

    // Endgame: mop-up with a >= 500 non-pawn material edge, * (14 - king distance).
    MopUpKingDistance = (40, -12),

    // Pawn endgames (no non-pawn material).
    PawnEndgameOpposition = (0, 27),
    RaceUnstoppable = (0, 67),        // + (8 - moves) * RaceUnstoppableSpeed
    RaceUnstoppableSpeed = (0, 36),
    RaceAhead = (0, 77),              // + (moves gap) * RaceAheadGap
    RaceAheadGap = (0, 99),

    // Single-rook endgames (one rook each, no queens).
    RookEgEnemyKingClose = (0, -1),   // * (7 - distance to enemy king)
    RookEgBehindPasser = (0, 4),
    RookEgFrontOfPasser = (0, 36),    // penalty
    RookEgBehindEnemyPasser = (0, 21),
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
