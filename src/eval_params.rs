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
    BishopPair = (17, 61),

    // Development (only applied while phase >= 12; mg only).
    DevUndevelopedMinor = (-12, 0),   // penalty per minor on its start square
    DevKingHomeUndeveloped = (-4, 0), // penalty: king at home with >= 2 undeveloped
    DevKingCastled = (23, 0),         // bonus: king on a castled square

    // Pawn structure.
    Doubled = (6, 8),                 // penalty per pawn on a file with > 1 own pawn
    Isolated = (4, 11),               // penalty per pawn with no own pawns on adjacent files
    Connected = (1, -6),              // bonus per phalanx/supported-chain pawn
    Backward = (3, 5),                // penalty per backward pawn

    // Passed pawns: base + rank * rank_weight (rank is relative 0..7).
    PassedBase = (-28, 0),
    PassedRank = (5, 9),
    PassedSupportedBase = (-15, -72), // extra when defended by own pawn
    PassedSupportedRank = (5, 22),
    PasserOwnKingNear = (7, 7),       // * (3 - king_distance) when distance <= 2
    PasserEnemyKingFar = (-3, 29),    // * (enemy king promo distance - remaining)
    PasserEnemyKingAdjacent = (25, 73), // penalty when enemy king touches the passer
    PasserRookBehind = (17, 2),       // own rook behind the passer, same file
    PasserRookFront = (-12, -26),     // penalty: own rook in front of the passer

    // Candidate passers (no enemy pawn ahead on the same file).
    CandidateBase = (-18, -18),
    CandidateRank = (5, 5),
    CandidateLever = (-35, -21),
    CandidateLeverRank = (5, 11),
    CandidateSupported = (17, 10),

    // Pawn attacks on central squares (files c..f, relative ranks 2..4).
    PawnCenterControl = (9, 27),

    // Rooks.
    RookOpenFile = (47, -5),
    RookSemiOpenFile = (14, 16),
    RookSeventh = (8, 38),

    // Minor pieces.
    KnightOutpost = (33, 4),
    BishopOutpost = (31, 8),
    Mobility = (1, 2),                // per pseudo-legal move, all pieces
    BishopMobilityExtra = (2, 1),     // additional per bishop move

    // Trapped pieces: weight * severity, severity = 3 - mobility (rook: 4 - mobility).
    TrappedKnightCorner = (23, -13),
    TrappedKnightEdge = (2, 5),
    TrappedBishopEdge = (0, 11),
    TrappedBishopCornerExtra = (31, 24),
    TrappedRookByKing = (-3, 13),

    // Threats: per enemy piece attacked by a lower-value piece of ours.
    ThreatPawnOnMinor = (40, 28),     // knight/bishop attacked by a pawn
    ThreatPawnOnRook = (46, 24),
    ThreatPawnOnQueen = (31, 16),
    ThreatMinorOnRook = (46, 15),     // rook attacked by a knight/bishop
    ThreatMinorOnQueen = (43, 6),
    ThreatRookOnQueen = (49, 22),
    Hanging = (19, 26),               // non-pawn piece attacked by us, undefended

    // Endgame: mop-up with a >= 500 non-pawn material edge, * (14 - king distance).
    MopUpKingDistance = (40, -12),

    // Pawn endgames (no non-pawn material).
    PawnEndgameOpposition = (0, 27),
    RaceUnstoppable = (0, 67),        // + (8 - moves) * RaceUnstoppableSpeed
    RaceUnstoppableSpeed = (0, 36),
    RaceAhead = (0, 77),              // + (moves gap) * RaceAheadGap
    RaceAheadGap = (0, 99),

    // Single-rook endgames (one rook each, no queens).
    RookEgEnemyKingClose = (0, 0),    // * (7 - distance to enemy king)
    RookEgBehindPasser = (0, 4),
    RookEgFrontOfPasser = (0, 36),    // penalty
    RookEgBehindEnemyPasser = (0, 22),
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
