//! Scalar evaluation weights used by `eval::evaluate()` and its helpers.
//!
//! Every entry is an (mg, eg) pair. Most terms enter the score linearly
//! (`score += weight * count`, White minus Black), which is what makes them
//! fittable by a Texel tuner alongside the PSQT. The few weights that feed
//! nonlinear machinery (king safety, which is folded into eg as `/ 6`, and
//! tempo, which is added after tapering) are kept as plain constants below.
//!
//! Sign convention: the stored value is the magnitude; the call site decides
//! whether it is a bonus or a penalty (e.g. `Doubled` is subtracted).

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
    BishopPair = (30, 40),

    // Development (only applied while phase >= 12; mg only).
    DevUndevelopedMinor = (5, 0),     // penalty per minor on its start square
    DevKingHomeUndeveloped = (8, 0),  // penalty: king at home with >= 2 undeveloped
    DevKingCastled = (8, 0),          // bonus: king on a castled square

    // Pawn structure.
    Doubled = (12, 16),               // penalty per pawn on a file with > 1 own pawn
    Isolated = (14, 12),              // penalty per pawn with no own pawns on adjacent files
    Connected = (8, 10),              // bonus per phalanx/supported-chain pawn
    Backward = (12, 6),               // penalty per backward pawn

    // Passed pawns: base + rank * rank_weight (rank is relative 0..7).
    PassedBase = (12, 28),
    PassedRank = (4, 12),
    PassedSupportedBase = (6, 12),    // extra when defended by own pawn
    PassedSupportedRank = (1, 3),
    PasserOwnKingNear = (3, 8),       // * (3 - king_distance) when distance <= 2
    PasserEnemyKingFar = (0, 2),      // * (enemy king promo distance - remaining)
    PasserEnemyKingAdjacent = (0, 10), // penalty when enemy king touches the passer
    PasserRookBehind = (4, 10),       // own rook behind the passer, same file
    PasserRookFront = (0, 4),         // penalty: own rook in front of the passer

    // Candidate passers (no enemy pawn ahead on the same file).
    CandidateBase = (4, 8),
    CandidateRank = (1, 3),
    CandidateLever = (4, 6),
    CandidateLeverRank = (0, 1),
    CandidateSupported = (2, 4),

    // Pawn attacks on central squares (files c..f, relative ranks 2..4).
    PawnCenterControl = (4, 0),

    // Rooks.
    RookOpenFile = (20, 14),
    RookSemiOpenFile = (10, 8),
    RookSeventh = (16, 24),

    // Minor pieces.
    KnightOutpost = (18, 8),
    BishopOutpost = (10, 6),
    Mobility = (2, 1),                // per pseudo-legal move, all pieces
    BishopMobilityExtra = (2, 1),     // additional per bishop move

    // Trapped pieces: weight * severity, severity = 3 - mobility (rook: 4 - mobility).
    TrappedKnightCorner = (12, 6),
    TrappedKnightEdge = (6, 3),
    TrappedBishopEdge = (10, 5),
    TrappedBishopCornerExtra = (6, 2),
    TrappedRookByKing = (8, 3),

    // Endgame: mop-up with a >= 500 non-pawn material edge, * (14 - king distance).
    MopUpKingDistance = (0, 3),

    // Pawn endgames (no non-pawn material).
    PawnEndgameOpposition = (0, 14),
    RaceUnstoppable = (0, 90),        // + (8 - moves) * RaceUnstoppableSpeed
    RaceUnstoppableSpeed = (0, 6),
    RaceAhead = (0, 40),              // + (moves gap) * RaceAheadGap
    RaceAheadGap = (0, 8),

    // Single-rook endgames (one rook each, no queens).
    RookEgEnemyKingClose = (0, 2),    // * (7 - distance to enemy king)
    RookEgBehindPasser = (0, 18),
    RookEgFrontOfPasser = (0, 8),     // penalty
    RookEgBehindEnemyPasser = (0, 14),
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
