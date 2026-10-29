use super::*;

const MIN_PARALLEL_DEPTH: i32 = 6;
// Iteration cap for timed searches; leaves MAX_PLY headroom for extensions and qsearch.
const MAX_SEARCH_DEPTH: i32 = 64;
const LMR_MAX_DEPTH: usize = 64;
const LMR_MAX_MOVES: usize = 64;

pub(crate) fn build_lmr_table() -> [[i32; LMR_MAX_MOVES]; LMR_MAX_DEPTH] {
    let mut table = [[0i32; LMR_MAX_MOVES]; LMR_MAX_DEPTH];
    for d in 1..LMR_MAX_DEPTH {
        for m in 1..LMR_MAX_MOVES {
            let ln_d = {
                let mut x = d as f64;
                let mut result = 0.0f64;
                while x >= 2.0 {
                    x /= 2.0;
                    result += 0.693_147_180_559_945_3; // ln(2)
                }
                // ln(1+y) for y in [0,1) via Taylor
                let y = x - 1.0;
                let mut term = y;
                for k in 1..20 {
                    if k % 2 == 1 {
                        result += term / k as f64;
                    } else {
                        result -= term / k as f64;
                    }
                    term *= y;
                }
                result
            };
            let ln_m = {
                let mut x = m as f64;
                let mut result = 0.0f64;
                while x >= 2.0 {
                    x /= 2.0;
                    result += 0.693_147_180_559_945_3;
                }
                let y = x - 1.0;
                let mut term = y;
                for k in 1..20 {
                    if k % 2 == 1 {
                        result += term / k as f64;
                    } else {
                        result -= term / k as f64;
                    }
                    term *= y;
                }
                result
            };
            // Base reduction: ln(d) * ln(m) / 2.0
            table[d][m] = (ln_d * ln_m / 2.0 + 0.5) as i32;
        }
    }
    table
}

pub(crate) struct Search {
    pub(crate) nodes: u64,
    deadline: Option<Instant>,
    pub(crate) stop: bool,
    pub(crate) killers: [[Option<Move>; 2]; MAX_PLY],
    move_buf: Vec<Move>,
}

impl Search {
    pub(crate) fn new(deadline: Option<Instant>) -> Self {
        Self {
            nodes: 0,
            deadline,
            stop: false,
            killers: [[None; 2]; MAX_PLY],
            move_buf: Vec::with_capacity(256),
        }
    }

    pub(crate) fn should_stop(&mut self) -> bool {
        if self.stop {
            return true;
        }
        if self.nodes & 2047 == 0 {
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    self.stop = true;
                    return true;
                }
            }
        }
        false
    }
}

fn history_score(history: &HistoryTable, side: Color, mv: Move) -> i32 {
    history[color_index(side)][mv.from as usize][mv.to as usize]
}

fn continuation_history_score(
    engine: &EngineState,
    side: Color,
    prev_move: Option<Move>,
    mv: Move,
) -> i32 {
    prev_move
        .map(|prev| {
            engine.continuation_history[color_index(side)][prev.to as usize][mv.from as usize]
                [mv.to as usize]
        })
        .unwrap_or(0)
}

fn apply_history_delta(engine: &mut EngineState, side: Color, mv: Move, delta: i32) {
    let h = &mut engine.history[color_index(side)][mv.from as usize][mv.to as usize];
    *h = (*h + delta).clamp(-100_000, 100_000);
}

fn apply_continuation_history_delta(
    engine: &mut EngineState,
    side: Color,
    prev_move: Option<Move>,
    mv: Move,
    delta: i32,
) {
    let Some(prev) = prev_move else {
        return;
    };
    let h = &mut engine.continuation_history[color_index(side)][prev.to as usize][mv.from as usize]
        [mv.to as usize];
    *h = (*h + delta).clamp(-100_000, 100_000);
}

fn counter_move_for(engine: &EngineState, side: Color, prev_move: Option<Move>) -> Option<Move> {
    prev_move.and_then(|mv| engine.counter_moves[color_index(side)][mv.from as usize][mv.to as usize])
}

fn tt_static_eval(entry: crate::tt::TTEntry, ply: usize) -> Option<i32> {
    let mut se = entry.static_eval()?;
    let tt_score = score_from_tt(entry.score, ply);
    if tt_score.abs() < MATE_SCORE - 1000 {
        match entry.bound {
            Bound::Lower => se = se.max(tt_score),
            Bound::Upper => se = se.min(tt_score),
            Bound::Exact => se = tt_score,
        }
    }
    Some(se)
}

struct RootWorkerResult {
    best_score: i32,
    best_move: Option<Move>,
    nodes: u64,
    completed: bool,
    ordered_root_moves: Vec<Move>,
}

#[derive(Default)]
struct PickerBufs {
    good_captures: Vec<(Move, i32)>,
    priority_quiets: Vec<(Move, i32)>,
    quiets: Vec<(Move, i32)>,
    bad_captures: Vec<(Move, i32)>,
}

impl PickerBufs {
    fn clear(&mut self) {
        self.good_captures.clear();
        self.priority_quiets.clear();
        self.quiets.clear();
        self.bad_captures.clear();
    }
}

thread_local! {
    static PICKER_BUF_POOL: std::cell::RefCell<Vec<PickerBufs>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static MOVE_VEC_POOL: std::cell::RefCell<Vec<Vec<Move>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

struct PooledMoveVec {
    vec: Vec<Move>,
}

impl PooledMoveVec {
    fn take() -> Self {
        let mut vec = MOVE_VEC_POOL
            .with(|pool| pool.borrow_mut().pop())
            .unwrap_or_default();
        vec.clear();
        Self { vec }
    }
}

impl Drop for PooledMoveVec {
    fn drop(&mut self) {
        let vec = std::mem::take(&mut self.vec);
        MOVE_VEC_POOL.with(|pool| pool.borrow_mut().push(vec));
    }
}

impl std::ops::Deref for PooledMoveVec {
    type Target = Vec<Move>;
    fn deref(&self) -> &Vec<Move> {
        &self.vec
    }
}

impl std::ops::DerefMut for PooledMoveVec {
    fn deref_mut(&mut self) -> &mut Vec<Move> {
        &mut self.vec
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Stage {
    Tt,
    GoodCapture,
    PriorityQuiet,
    Quiet,
    BadCapture,
}

struct MovePicker {
    tt_move: Option<Move>,
    tt_yielded: bool,
    bufs: PickerBufs,
    total: usize,
}

impl Drop for MovePicker {
    fn drop(&mut self) {
        let bufs = std::mem::take(&mut self.bufs);
        PICKER_BUF_POOL.with(|pool| pool.borrow_mut().push(bufs));
    }
}

impl MovePicker {
    fn new(
        board: &mut Board,
        engine: &EngineState,
        tt_move: Option<Move>,
        counter_move: Option<Move>,
        prev_move: Option<Move>,
        killers: &[Option<Move>; 2],
        history: &HistoryTable,
        captures_only: bool,
        move_buf: &mut Vec<Move>,
    ) -> Self {
        board.generate_legal_moves(captures_only, move_buf);
        let mut bufs = PICKER_BUF_POOL
            .with(|pool| pool.borrow_mut().pop())
            .unwrap_or_default();
        bufs.clear();

        if move_buf.is_empty() {
            return Self {
                tt_move: None,
                tt_yielded: true,
                bufs,
                total: 0,
            };
        }

        let side = board.side_to_move;
        let mut picker = Self {
            tt_move: None,
            tt_yielded: false,
            bufs,
            total: 0,
        };

        for mv in move_buf.iter().copied() {
            picker.total += 1;
            if Some(mv) == tt_move {
                picker.tt_move = Some(mv);
                continue;
            }

            if is_capture_move(board, mv) {
                let see_score = see(board, mv);
                let score = capture_stage_score_from_see(board, mv, see_score);
                if see_score >= 0 {
                    picker.bufs.good_captures.push((mv, score));
                } else {
                    picker.bufs.bad_captures.push((mv, score));
                }
                continue;
            }

            let score = quiet_stage_score(engine, side, mv, counter_move, prev_move, killers);
            if is_priority_quiet(engine, side, mv, counter_move, prev_move, killers) {
                picker.bufs.priority_quiets.push((mv, score));
            } else {
                picker.bufs.quiets.push((mv, score.max(history_score(history, side, mv))));
            }
        }

        picker
    }

    fn is_empty(&self) -> bool {
        self.total == 0
    }

    fn next(&mut self) -> Option<(Move, Stage)> {
        if !self.tt_yielded {
            self.tt_yielded = true;
            if let Some(mv) = self.tt_move {
                return Some((mv, Stage::Tt));
            }
        }

        take_best_scored(&mut self.bufs.good_captures)
            .map(|mv| (mv, Stage::GoodCapture))
            .or_else(|| {
                take_best_scored(&mut self.bufs.priority_quiets)
                    .map(|mv| (mv, Stage::PriorityQuiet))
            })
            .or_else(|| take_best_scored(&mut self.bufs.quiets).map(|mv| (mv, Stage::Quiet)))
            .or_else(|| {
                take_best_scored(&mut self.bufs.bad_captures).map(|mv| (mv, Stage::BadCapture))
            })
    }
}

fn take_best_scored(moves: &mut Vec<(Move, i32)>) -> Option<Move> {
    if moves.is_empty() {
        return None;
    }

    let mut best_idx = 0usize;
    let mut best_score = moves[0].1;
    for (idx, &(_, score)) in moves.iter().enumerate().skip(1) {
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    Some(moves.swap_remove(best_idx).0)
}

pub(crate) fn is_repetition(rep_stack: &[u64], key: u64, halfmove_clock: u32) -> bool {
    if rep_stack.len() < 3 || halfmove_clock < 4 {
        return false;
    }

    let mut idx = rep_stack.len() - 3;
    let mut remaining = halfmove_clock as usize;
    loop {
        if rep_stack[idx] == key {
            return true;
        }
        if idx < 2 || remaining < 2 {
            break;
        }
        idx -= 2;
        remaining -= 2;
    }
    false
}

pub(crate) fn handle_go(cmd: &str, board: &mut Board, engine: &mut EngineState) -> Option<String> {
    if engine.own_book {
        if let Some(book) = try_book_move(board, engine) {
            return Some(book);
        }
    }

    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let mut depth: Option<i32> = None;
    let mut movetime: Option<u64> = None;
    let mut wtime: Option<u64> = None;
    let mut btime: Option<u64> = None;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;

    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "depth" if i + 1 < tokens.len() => {
                depth = Some(tokens[i + 1].parse().unwrap_or(8));
                i += 2;
            }
            "movetime" if i + 1 < tokens.len() => {
                movetime = Some(tokens[i + 1].parse().unwrap_or(0));
                i += 2;
            }
            "wtime" if i + 1 < tokens.len() => {
                wtime = Some(tokens[i + 1].parse().unwrap_or(0));
                i += 2;
            }
            "btime" if i + 1 < tokens.len() => {
                btime = Some(tokens[i + 1].parse().unwrap_or(0));
                i += 2;
            }
            "winc" if i + 1 < tokens.len() => {
                winc = tokens[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            "binc" if i + 1 < tokens.len() => {
                binc = tokens[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            _ => i += 1,
        }
    }

    // Timed searches deepen until the deadline; a bare `go` keeps the old fixed depth.
    let timed = movetime.is_some() || wtime.is_some() || btime.is_some();
    let depth = depth.unwrap_or(if timed { MAX_SEARCH_DEPTH } else { 8 });

    let deadline = if let Some(ms) = movetime {
        Some(Instant::now() + Duration::from_millis(ms.max(1)))
    } else {
        let (time_left, inc) = match board.side_to_move {
            Color::White => (wtime, winc),
            Color::Black => (btime, binc),
        };
        time_left.map(|t| {
            let budget = (t / 28).max(20).saturating_add(inc / 2);
            Instant::now() + Duration::from_millis(budget)
        })
    };

    engine.prepare_for_search();

    let start = Instant::now();
    let mut search = Search::new(deadline);
    let mut best_score = -INF;

    let mut root_moves = Vec::new();
    board.generate_legal_moves(false, &mut root_moves);
    if root_moves.is_empty() {
        return None;
    }

    let hash = board_hash(board, &engine.zobrist);
    let tt_root_move = engine.tt.probe(hash).and_then(|e| e.best);
    let mut rep_stack = if engine.position_history.last().copied() == Some(hash) {
        engine.position_history.clone()
    } else {
        vec![hash]
    };
    order_moves(
        board,
        engine,
        &mut root_moves,
        tt_root_move,
        None,
        None,
        &search.killers[0],
        &engine.history,
    );

    if let Some(entry) = engine.tt.probe(hash) {
        if let Some(best) = entry.best {
            if let Some(pos) = root_moves.iter().position(|m| *m == best) {
                root_moves.swap(0, pos);
            }
        }
    }
    let mut best_move = Some(root_moves[0]);
    let worker_count = engine.threads.min(root_moves.len()).max(1);
    let threaded = worker_count > 1;
    let mut total_threaded_nodes = 0u64;
    let mut worker_engines: Option<Vec<EngineState>> = None;

    let parallel_allowed = movetime
        .map(|ms| ms >= 100)
        .unwrap_or_else(|| depth >= MIN_PARALLEL_DEPTH);

    for d in 1..=depth.max(1) {
        let use_threads = threaded && parallel_allowed && d >= MIN_PARALLEL_DEPTH;
        if !use_threads && search.should_stop() {
            break;
        }

        let mut alpha = -INF;
        let mut beta = INF;
        if d >= 4 {
            alpha = best_score - ASPIRATION_WINDOW;
            beta = best_score + ASPIRATION_WINDOW;
        }

        let (score, mv, completed, depth_nodes) = loop {
            if use_threads {
                if worker_engines.is_none() {
                    worker_engines = Some(
                        (0..worker_count)
                            .map(|_| engine.worker_clone())
                            .collect::<Vec<_>>(),
                    );
                }
                let result = search_root_parallel(
                    board,
                    d,
                    alpha,
                    beta,
                    hash,
                    deadline,
                    worker_engines.as_mut().unwrap(),
                    &mut root_moves,
                    &rep_stack,
                );
                if !result.2 || d < 4 {
                    break (result.0, result.1, result.2, result.3);
                }
                if result.0 <= alpha {
                    alpha = (alpha - ASPIRATION_WINDOW * 2).max(-INF);
                    continue;
                }
                if result.0 >= beta {
                    beta = (beta + ASPIRATION_WINDOW * 2).min(INF);
                    continue;
                }
                break (result.0, result.1, result.2, result.3);
            } else {
                let nodes_before = search.nodes;
                let result = search_root(
                    board,
                    d,
                    alpha,
                    beta,
                    hash,
                    &mut search,
                    engine,
                    &mut root_moves,
                    &mut rep_stack,
                );
                let nodes_after = search.nodes;
                if !result.2 || d < 4 {
                    break (result.0, result.1, result.2, nodes_after - nodes_before);
                }
                if result.0 <= alpha {
                    alpha = (alpha - ASPIRATION_WINDOW * 2).max(-INF);
                    continue;
                }
                if result.0 >= beta {
                    beta = (beta + ASPIRATION_WINDOW * 2).min(INF);
                    continue;
                }
                break (result.0, result.1, result.2, nodes_after - nodes_before);
            }
        };

        if !completed {
            break;
        }

        if use_threads {
            total_threaded_nodes += depth_nodes;
        }
        best_score = score;
        if let Some(m) = mv {
            best_move = Some(m);
            if let Some(pos) = root_moves.iter().position(|x| *x == m) {
                root_moves.swap(0, pos);
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;
        let nodes = search.nodes + total_threaded_nodes;
        print_info(d, best_score, nodes, elapsed, best_move);
    }

    engine.last_search_nodes = search.nodes + total_threaded_nodes;
    best_move.map(move_to_uci)
}

fn search_root_parallel(
    board: &Board,
    depth: i32,
    alpha: i32,
    beta: i32,
    hash: u64,
    deadline: Option<Instant>,
    worker_engines: &mut [EngineState],
    root_moves: &mut [Move],
    rep_stack: &[u64],
) -> (i32, Option<Move>, bool, u64) {
    if root_moves.is_empty() {
        return (-INF, None, true, 0);
    }

    let active_workers = worker_engines.len().max(1);
    let mut results = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (worker_idx, worker) in worker_engines[..active_workers].iter_mut().enumerate() {
            let board_clone = board.clone();
            let rep_stack_vec = rep_stack.to_vec();
            let root_moves_vec = root_moves.to_vec();
            let builder = thread::Builder::new().stack_size(8 * 1024 * 1024);
            handles.push(
                builder
                    .spawn_scoped(scope, move || {
                        search_root_lazy_worker(
                            board_clone,
                            depth,
                            alpha,
                            beta,
                            hash,
                            deadline,
                            worker,
                            root_moves_vec,
                            rep_stack_vec,
                            worker_idx,
                        )
                    })
                    .unwrap(),
            );
        }
        for handle in handles {
            results.push(handle.join().unwrap());
        }
    });

    let mut best_score = -INF;
    let mut best_move = None;
    let mut completed = true;
    let mut total_nodes = 0u64;
    let mut best_root_order = None;

    for result in results {
        total_nodes += result.nodes;
        completed &= result.completed;
        if let Some(mv) = result.best_move {
            if result.best_score > best_score {
                best_score = result.best_score;
                best_move = Some(mv);
                best_root_order = Some(result.ordered_root_moves);
            }
        }
    }

    let alpha_orig = alpha;
    let bound = if best_score <= alpha_orig {
        Bound::Upper
    } else if best_score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    if let Some(mv) = best_move {
        worker_engines[0]
            .tt
            .store(hash, depth, score_to_tt(best_score, 0), None, bound, Some(mv));
        if let Some(ordered) = best_root_order {
            root_moves.clone_from_slice(&ordered);
        }
    }

    (best_score, best_move, completed, total_nodes)
}

fn diversify_root_moves(root_moves: &mut [Move], worker_idx: usize) {
    if worker_idx == 0 || root_moves.len() < 2 {
        return;
    }

    let top = root_moves.len().min(4 + worker_idx);
    let swap_idx = worker_idx.min(top - 1);
    root_moves.swap(0, swap_idx);
}

fn search_root_lazy_worker(
    mut board: Board,
    depth: i32,
    alpha: i32,
    beta: i32,
    hash: u64,
    deadline: Option<Instant>,
    engine: &mut EngineState,
    mut root_moves: Vec<Move>,
    mut rep_stack: Vec<u64>,
    worker_idx: usize,
) -> RootWorkerResult {
    let mut search = Search::new(deadline);
    diversify_root_moves(&mut root_moves, worker_idx);
    let (best_score, best_move, completed) = search_root(
        &mut board,
        depth,
        alpha,
        beta,
        hash,
        &mut search,
        engine,
        &mut root_moves,
        &mut rep_stack,
    );
    RootWorkerResult {
        best_score,
        best_move,
        nodes: search.nodes,
        completed,
        ordered_root_moves: root_moves,
    }
}

fn search_root(
    board: &mut Board,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    key: u64,
    search: &mut Search,
    engine: &mut EngineState,
    root_moves: &mut Vec<Move>,
    rep_stack: &mut Vec<u64>,
) -> (i32, Option<Move>, bool) {
    let alpha_orig = alpha;
    let tt_move = engine.tt.probe(key).and_then(|e| e.best);
    order_moves(
        board,
        engine,
        root_moves,
        tt_move,
        None,
        None,
        &search.killers[0],
        &engine.history,
    );

    let mut best_move = None;
    let mut best_score = -INF;

    for (idx, mv) in root_moves.iter().copied().enumerate() {
        if search.should_stop() {
            return (best_score, best_move, false);
        }

        let undo = board.make_move(mv);
        let child_key = child_hash(key, board, mv, undo, &engine.zobrist);
        rep_stack.push(child_key);

        let mut score;
        if idx == 0 {
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                1,
                search,
                engine,
                rep_stack,
                child_key,
                Some(mv),
                true,
                true,
            );
        } else {
            score = -negamax(
                board,
                depth - 1,
                -alpha - 1,
                -alpha,
                1,
                search,
                engine,
                rep_stack,
                child_key,
                Some(mv),
                true,
                true,
            );
            if score > alpha && score < beta {
                score = -negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                    search,
                    engine,
                    rep_stack,
                    child_key,
                    Some(mv),
                    true,
                    true,
                );
            }
        }

        rep_stack.pop();
        board.undo_move(mv, undo);

        if search.stop {
            return (best_score, best_move, false);
        }

        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }

    let bound = if best_score <= alpha_orig {
        Bound::Upper
    } else if best_score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    engine
        .tt
        .store(key, depth, score_to_tt(best_score, 0), None, bound, best_move);

    (best_score, best_move, true)
}

pub(crate) fn print_info(depth: i32, score: i32, nodes: u64, time_ms: u64, best: Option<Move>) {
    if score.abs() >= MATE_SCORE - 1000 {
        let mate = if score > 0 {
            (MATE_SCORE - score + 1) / 2
        } else {
            -(MATE_SCORE + score + 1) / 2
        };
        println!(
            "info depth {} score mate {} nodes {} time {} pv {}",
            depth,
            mate,
            nodes,
            time_ms,
            best.map(move_to_uci).unwrap_or_else(|| "0000".to_string())
        );
    } else {
        println!(
            "info depth {} score cp {} nodes {} time {} pv {}",
            depth,
            score,
            nodes,
            time_ms,
            best.map(move_to_uci).unwrap_or_else(|| "0000".to_string())
        );
    }
    flush_stdout();
}

fn negamax(
    board: &mut Board,
    mut depth: i32,
    mut alpha: i32,
    mut beta: i32,
    ply: usize,
    search: &mut Search,
    engine: &mut EngineState,
    rep_stack: &mut Vec<u64>,
    key: u64,
    prev_move: Option<Move>,
    allow_null: bool,
    allow_iid: bool,
) -> i32 {
    search.nodes += 1;
    if search.should_stop() {
        return 0;
    }

    if ply >= MAX_PLY - 1 {
        return eval_for_side(board);
    }

    if board.halfmove_clock >= 100 {
        return 0;
    }
    if is_repetition(rep_stack, key, board.halfmove_clock) {
        return 0;
    }

    let in_check = board.is_in_check(board.side_to_move);
    if in_check {
        depth += 1;
    }

    let mate_low = -MATE_SCORE + ply as i32;
    let mate_high = MATE_SCORE - ply as i32 - 1;
    if alpha < mate_low {
        alpha = mate_low;
    }
    if beta > mate_high {
        beta = mate_high;
    }
    if alpha >= beta {
        return alpha;
    }

    if depth <= 0 {
        return quiescence(board, alpha, beta, search, engine, rep_stack, key, ply);
    }

    let alpha_orig = alpha;
    let tt_entry = engine.tt.probe(key);
    let mut tt_move = None;

    if let Some(entry) = tt_entry {
        tt_move = entry.best;
        if entry.depth as i32 >= depth {
            let tt_score = score_from_tt(entry.score, ply);
            match entry.bound {
                Bound::Exact => return tt_score,
                Bound::Lower if tt_score >= beta => return tt_score,
                Bound::Upper if tt_score <= alpha => return tt_score,
                _ => {}
            }
        }
    }

    let raw_static_eval = if in_check {
        None
    } else {
        Some(
            tt_entry
                .and_then(|entry| tt_static_eval(entry, ply))
                .unwrap_or_else(|| eval_for_side(board)),
        )
    };
    let static_eval = raw_static_eval;

    let params = engine.search_params;

    if let Some(se) = static_eval {
        if beta - alpha <= 1 && depth <= 2 {
            let margin = if depth == 1 {
                params.futility_margin_d1
            } else {
                params.futility_margin_d2
            };
            if se + margin <= alpha {
                return quiescence(board, alpha, beta, search, engine, rep_stack, key, ply);
            }
        }

        if beta - alpha <= 1 && depth <= 3 {
            let margin = match depth {
                1 => params.reverse_futility_margin_d1,
                2 => params.reverse_futility_margin_d2,
                _ => params.reverse_futility_margin_d3,
            };
            if se - margin >= beta {
                return se;
            }
        }
    }

    if tt_move.is_none() && allow_iid && depth >= 5 {
        board.generate_legal_moves(false, &mut search.move_buf);
        let mut iid_moves = std::mem::take(&mut search.move_buf);
        if !iid_moves.is_empty() {
            order_moves(
                board,
                engine,
                &mut iid_moves,
                None,
                counter_move_for(engine, board.side_to_move, prev_move),
                prev_move,
                &search.killers[ply],
                &engine.history,
            );
            let mut best_iid = iid_moves[0];
            let mut best_iid_score = -INF;
            for mv in iid_moves.iter().copied().take(8) {
                let undo = board.make_move(mv);
                let child_key = child_hash(key, board, mv, undo, &engine.zobrist);
                rep_stack.push(child_key);
                let s = -negamax(
                    board,
                    depth - 3,
                    -beta,
                    -alpha,
                    ply + 1,
                search,
                engine,
                rep_stack,
                child_key,
                Some(mv),
                true,
                false,
            );
                rep_stack.pop();
                board.undo_move(mv, undo);
                if search.stop {
                    return 0;
                }
                if s > best_iid_score {
                    best_iid_score = s;
                    best_iid = mv;
                }
            }
            tt_move = Some(best_iid);
        }
        search.move_buf = iid_moves;
    }

    if allow_null && depth >= 3 && !in_check && beta - alpha <= 1 {
        let null_margin = params.null_margin_base + depth * params.null_margin_depth_scale;
        let mut has_non_pawn = false;
        let side_c = color_index(board.side_to_move);
        for pi in 0..board.piece_cnt[side_c] {
            let sq = board.piece_sqs[side_c][pi];
            if let Some(p) = board.squares[sq as usize] {
                if p.kind != PieceType::Pawn && p.kind != PieceType::King {
                    has_non_pawn = true;
                    break;
                }
            }
        }
        if has_non_pawn && static_eval.map(|se| se >= beta - null_margin).unwrap_or(false) {
            let old_ep = board.en_passant;
            let old_halfmove = board.halfmove_clock;
            let old_fullmove = board.fullmove_number;
            let old_side = board.side_to_move;

            board.en_passant = None;
            board.side_to_move = board.side_to_move.opposite();
            board.halfmove_clock = board.halfmove_clock.saturating_add(1);
            if old_side == Color::Black {
                board.fullmove_number = board.fullmove_number.saturating_add(1);
            }

            let child_key = null_move_hash(key, old_ep, &engine.zobrist);
            rep_stack.push(child_key);
            let reduction = params.null_reduction_base + depth / params.null_reduction_depth_div;
            let score = -negamax(
                board,
                depth - 1 - reduction,
                -beta,
                -beta + 1,
                ply + 1,
                search,
                engine,
                rep_stack,
                child_key,
                None,
                false,
                false,
            );
            rep_stack.pop();

            board.en_passant = old_ep;
            board.halfmove_clock = old_halfmove;
            board.fullmove_number = old_fullmove;
            board.side_to_move = old_side;

            if search.stop {
                return 0;
            }
            if score >= beta {
                return score;
            }
        }
    }

    let counter_move = counter_move_for(engine, board.side_to_move, prev_move);
    let killers = search.killers[ply];
    let mut picker = MovePicker::new(
        board,
        engine,
        tt_move,
        counter_move,
        prev_move,
        &killers,
        &engine.history,
        false,
        &mut search.move_buf,
    );
    if picker.is_empty() {
        if in_check {
            return -MATE_SCORE + ply as i32;
        }
        return 0;
    }

    let mut best = -INF;
    let mut best_move = None;
    let side = board.side_to_move;
    let mut quiets_tried = PooledMoveVec::take();
    let mut idx = 0usize;

    while let Some((mv, stage)) = picker.next() {
        let quiet = match stage {
            Stage::Tt => !is_capture_move(board, mv),
            Stage::GoodCapture | Stage::BadCapture => false,
            Stage::PriorityQuiet | Stage::Quiet => true,
        };
        let quiet_history = if quiet {
            history_score(&engine.history, side, mv)
        } else {
            0
        };
        let quiet_continuation = if quiet {
            continuation_history_score(engine, side, prev_move, mv)
        } else {
            0
        };
        let priority_quiet = quiet
            && (Some(mv) == killers[0]
                || Some(mv) == killers[1]
                || Some(mv) == counter_move
                || quiet_history > 0
                || quiet_continuation > 0);

        if quiet && !priority_quiet && !in_check && beta - alpha <= 1 && depth <= 3 {
            let limit = match depth {
                1 => params.lmp_limit_d1,
                2 => params.lmp_limit_d2,
                _ => params.lmp_limit_d3,
            };
            if idx >= limit {
                continue;
            }
        }

        if quiet {
            quiets_tried.push(mv);
        }

        let undo = board.make_move(mv);
        let child_key = child_hash(key, board, mv, undo, &engine.zobrist);
        rep_stack.push(child_key);
        let gives_check = board.is_in_check(board.side_to_move);

        let mut score;
        if idx == 0 {
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                ply + 1,
                search,
                engine,
                rep_stack,
                child_key,
                Some(mv),
                true,
                true,
            );
        } else {
            let reduction = if quiet
                && !in_check
                && !gives_check
                && !priority_quiet
                && depth >= params.lmr_min_depth
                && idx >= params.lmr_min_move_index
            {
                let d = (depth as usize).min(LMR_MAX_DEPTH - 1);
                let m = idx.min(LMR_MAX_MOVES - 1);
                let mut r = engine.lmr_table[d][m];
                if depth >= params.lmr_extra_depth && idx >= params.lmr_extra_move_index {
                    r += 1;
                }
                // Reduce more for moves with bad history
                if quiet_history < -2000 {
                    r += 1;
                }
                // Reduce less for moves with good history
                if quiet_history > 2000 {
                    r -= 1;
                }
                r.clamp(0, depth - 2)
            } else {
                0
            };
            let reduced = reduction > 0;
            let reduced_depth = (depth - 1 - reduction).max(0);

            score = -negamax(
                board,
                reduced_depth,
                -alpha - 1,
                -alpha,
                ply + 1,
                search,
                engine,
                rep_stack,
                child_key,
                Some(mv),
                true,
                true,
            );

            if reduced && score > alpha {
                score = -negamax(
                    board,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    search,
                    engine,
                    rep_stack,
                    child_key,
                    Some(mv),
                    true,
                    true,
                );
            }

            if score > alpha && score < beta {
                score = -negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    search,
                    engine,
                    rep_stack,
                    child_key,
                    Some(mv),
                    true,
                    true,
                );
            }
        }

        rep_stack.pop();
        board.undo_move(mv, undo);

        if search.stop {
            return 0;
        }

        if score > best {
            best = score;
            best_move = Some(mv);
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            if quiet {
                let bonus = (depth * depth).min(64 * 64);
                if let Some(prev) = prev_move {
                    engine.counter_moves[color_index(side)][prev.from as usize][prev.to as usize] =
                        Some(mv);
                }
                if search.killers[ply][0] != Some(mv) {
                    search.killers[ply][1] = search.killers[ply][0];
                    search.killers[ply][0] = Some(mv);
                }
                apply_history_delta(engine, side, mv, bonus);
                apply_continuation_history_delta(engine, side, prev_move, mv, bonus);
                for tried in quiets_tried.iter().copied() {
                    if tried != mv {
                        apply_history_delta(engine, side, tried, -bonus);
                        apply_continuation_history_delta(engine, side, prev_move, tried, -bonus);
                    }
                }
            }
            engine
                .tt
                .store(
                    key,
                    depth,
                    score_to_tt(alpha, ply),
                    raw_static_eval,
                    Bound::Lower,
                    best_move,
                );
            return alpha;
        }
        idx += 1;
    }

    let bound = if best <= alpha_orig {
        Bound::Upper
    } else {
        Bound::Exact
    };
    engine
        .tt
        .store(key, depth, score_to_tt(best, ply), raw_static_eval, bound, best_move);

    best
}

fn quiescence(
    board: &mut Board,
    mut alpha: i32,
    beta: i32,
    search: &mut Search,
    engine: &mut EngineState,
    rep_stack: &mut Vec<u64>,
    key: u64,
    ply: usize,
) -> i32 {
    search.nodes += 1;
    if search.should_stop() {
        return 0;
    }

    if board.halfmove_clock >= 100 {
        return 0;
    }

    if is_repetition(rep_stack, key, board.halfmove_clock) {
        return 0;
    }

    let alpha_orig = alpha;
    let tt_entry = engine.tt.probe(key);
    let mut tt_move = None;
    if let Some(entry) = tt_entry {
        tt_move = entry.best;
        if entry.depth >= 0 {
            let tt_score = score_from_tt(entry.score, ply);
            match entry.bound {
                Bound::Exact => return tt_score,
                Bound::Lower if tt_score >= beta => return tt_score,
                Bound::Upper if tt_score <= alpha => return tt_score,
                _ => {}
            }
        }
    }

    let in_check = board.is_in_check(board.side_to_move);
    let mut best_move = None;
    let raw_static_eval = if in_check {
        None
    } else {
        Some(
            tt_entry
                .and_then(|entry| tt_static_eval(entry, ply))
                .unwrap_or_else(|| eval_for_side(board)),
        )
    };
    let static_eval = raw_static_eval;

    if let Some(stand_pat) = static_eval {
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    let mut picker = MovePicker::new(
        board,
        engine,
        tt_move,
        None,
        None,
        &[None, None],
        &engine.history,
        !in_check,
        &mut search.move_buf,
    );
    if picker.is_empty() {
        if in_check {
            return -MATE_SCORE + ply as i32;
        }
        return alpha;
    }

    while let Some((mv, stage)) = picker.next() {
        if !in_check {
            // Stage buckets already encode the SEE sign, so only the TT move
            // (which bypassed bucketing) needs a fresh SEE here.
            if stage == Stage::BadCapture {
                continue;
            }
            if stage == Stage::Tt && see(board, mv) < 0 {
                continue;
            }
            let victim = if mv.is_en_passant {
                piece_value(PieceType::Pawn)
            } else {
                board.squares[mv.to as usize]
                    .map(|p| piece_value(p.kind))
                    .unwrap_or(0)
            };
            if let Some(stand_pat) = static_eval {
                if alpha > -INF / 2 && stand_pat + victim + 150 < alpha {
                    continue;
                }
            }
        }

        let undo = board.make_move(mv);
        let child_key = child_hash(key, board, mv, undo, &engine.zobrist);
        rep_stack.push(child_key);
        let score = -quiescence(board, -beta, -alpha, search, engine, rep_stack, child_key, ply + 1);
        rep_stack.pop();
        board.undo_move(mv, undo);

        if search.stop {
            return 0;
        }

        if score >= beta {
            engine
                .tt
                .store(
                    key,
                    0,
                    score_to_tt(beta, ply),
                    raw_static_eval,
                    Bound::Lower,
                    Some(mv),
                );
            return beta;
        }
        if score > alpha {
            alpha = score;
            best_move = Some(mv);
        }
    }

    let bound = if alpha <= alpha_orig { Bound::Upper } else { Bound::Exact };
    engine
        .tt
        .store(key, 0, score_to_tt(alpha, ply), raw_static_eval, bound, best_move.or(tt_move));
    alpha
}

fn order_moves(
    board: &Board,
    engine: &EngineState,
    moves: &mut Vec<Move>,
    tt_move: Option<Move>,
    counter_move: Option<Move>,
    prev_move: Option<Move>,
    killers: &[Option<Move>; 2],
    history: &HistoryTable,
) {
    moves.sort_by_cached_key(|m| {
        Reverse(move_order_score(
            board,
            engine,
            *m,
            tt_move,
            counter_move,
            prev_move,
            killers,
            history,
        ))
    });
}

fn is_capture_move(board: &Board, mv: Move) -> bool {
    mv.is_en_passant || board.squares[mv.to as usize].is_some()
}

fn is_priority_quiet(
    engine: &EngineState,
    side: Color,
    mv: Move,
    counter_move: Option<Move>,
    prev_move: Option<Move>,
    killers: &[Option<Move>; 2],
) -> bool {
    Some(mv) == killers[0]
        || Some(mv) == killers[1]
        || Some(mv) == counter_move
        || history_score(&engine.history, side, mv) > 0
        || continuation_history_score(engine, side, prev_move, mv) > 0
}

fn quiet_stage_score(
    engine: &EngineState,
    side: Color,
    mv: Move,
    counter_move: Option<Move>,
    prev_move: Option<Move>,
    killers: &[Option<Move>; 2],
) -> i32 {
    let mut score = history_score(&engine.history, side, mv)
        + continuation_history_score(engine, side, prev_move, mv) / 2;
    if Some(mv) == killers[0] {
        score += 30_000;
    } else if Some(mv) == counter_move {
        score += 20_000;
    } else if Some(mv) == killers[1] {
        score += 10_000;
    }
    if mv.is_castle {
        score += 10;
    }
    score += mv
        .promotion
        .map(piece_value)
        .unwrap_or(0)
        .saturating_sub(piece_value(PieceType::Pawn));
    score
}

fn capture_stage_score_from_see(board: &Board, mv: Move, see_score: i32) -> i32 {
    let victim = if mv.is_en_passant {
        piece_value(PieceType::Pawn)
    } else {
        board.squares[mv.to as usize]
            .map(|p| piece_value(p.kind))
            .unwrap_or(0)
    };
    let attacker = board.squares[mv.from as usize]
        .map(|p| piece_value(p.kind))
        .unwrap_or(piece_value(PieceType::Pawn));
    see_score * 16 + victim - attacker
}

pub(crate) fn piece_attacks_square(
    squares: &[Option<Piece>; 64],
    from: u8,
    piece: Piece,
    target: u8,
) -> bool {
    let ff = (from % 8) as i8;
    let fr = (from / 8) as i8;
    let tf = (target % 8) as i8;
    let tr = (target / 8) as i8;
    let df = tf - ff;
    let dr = tr - fr;

    match piece.kind {
        PieceType::Pawn => {
            if piece.color == Color::White {
                dr == 1 && (df == 1 || df == -1)
            } else {
                dr == -1 && (df == 1 || df == -1)
            }
        }
        PieceType::Knight => {
            (df.abs() == 1 && dr.abs() == 2) || (df.abs() == 2 && dr.abs() == 1)
        }
        PieceType::King => df.abs() <= 1 && dr.abs() <= 1 && (df != 0 || dr != 0),
        PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
            let (step_f, step_r) = match piece.kind {
                PieceType::Bishop if df.abs() == dr.abs() && df != 0 => (df.signum(), dr.signum()),
                PieceType::Rook if (df == 0) ^ (dr == 0) => (df.signum(), dr.signum()),
                PieceType::Queen
                    if (df.abs() == dr.abs() && df != 0) || ((df == 0) ^ (dr == 0)) =>
                {
                    (df.signum(), dr.signum())
                }
                _ => return false,
            };

            let mut f = ff + step_f;
            let mut r = fr + step_r;
            while f != tf || r != tr {
                if !in_bounds(f, r) {
                    return false;
                }
                let sq = (r as u8) * 8 + (f as u8);
                if squares[sq as usize].is_some() {
                    return false;
                }
                f += step_f;
                r += step_r;
            }
            true
        }
    }
}

fn piece_on_square(squares: &[Option<Piece>; 64], f: i8, r: i8) -> Option<(u8, Piece)> {
    if !in_bounds(f, r) {
        return None;
    }
    let sq = (r as u8) * 8 + (f as u8);
    squares[sq as usize].map(|p| (sq, p))
}

/// First blocker along `ray` over occupancy `occ`; `toward_high` says
/// whether the ray runs toward higher bit indices from the target.
fn first_blocker(ray: u64, occ: u64, toward_high: bool) -> Option<u8> {
    let blockers = ray & occ;
    if blockers == 0 {
        return None;
    }
    Some(if toward_high {
        blockers.trailing_zeros() as u8
    } else {
        63 - blockers.leading_zeros() as u8
    })
}

/// Directional slider probes in the same order as the old mailbox scan, so
/// equal-valued attacker selection (and therefore x-ray reveal order) is
/// unchanged. Rays: DIAG_RAYS = [(1,1),(-1,1),(1,-1),(-1,-1)],
/// ORTHO_RAYS = [(0,1),(1,0),(0,-1),(-1,0)]; first two of each run toward
/// higher bits.
fn first_slider_attacker(
    squares: &[Option<Piece>; 64],
    occ: u64,
    target: u8,
    side: Color,
    dirs: &[(usize, bool, bool)],
    allowed: &[PieceType],
) -> Option<(u8, Piece)> {
    use crate::bitboard::{DIAG_RAYS, ORTHO_RAYS};
    let t = target as usize;
    for &(idx, diag, toward_high) in dirs {
        let ray = if diag { DIAG_RAYS[idx][t] } else { ORTHO_RAYS[idx][t] };
        if let Some(sq) = first_blocker(ray, occ, toward_high) {
            if let Some(p) = squares[sq as usize] {
                if p.color == side && allowed.contains(&p.kind) {
                    return Some((sq, p));
                }
            }
        }
    }
    None
}

fn least_valuable_attacker(
    squares: &[Option<Piece>; 64],
    occ: u64,
    target: u8,
    side: Color,
) -> Option<(u8, Piece)> {
    let tf = (target % 8) as i8;
    let tr = (target / 8) as i8;

    let pawn_rank = match side {
        Color::White => tr - 1,
        Color::Black => tr + 1,
    };
    for pawn_file in [tf - 1, tf + 1] {
        if let Some((sq, p)) = piece_on_square(squares, pawn_file, pawn_rank) {
            if p.color == side && p.kind == PieceType::Pawn {
                return Some((sq, p));
            }
        }
    }

    for (df, dr) in [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ] {
        if let Some((sq, p)) = piece_on_square(squares, tf + df, tr + dr) {
            if p.color == side && p.kind == PieceType::Knight {
                return Some((sq, p));
            }
        }
    }

    const BISHOP_DIRS: &[(usize, bool, bool)] =
        &[(0, true, true), (2, true, false), (1, true, true), (3, true, false)];
    const ROOK_DIRS: &[(usize, bool, bool)] =
        &[(1, false, true), (3, false, false), (0, false, true), (2, false, false)];
    const QUEEN_DIRS: &[(usize, bool, bool)] = &[
        (0, true, true),
        (2, true, false),
        (1, true, true),
        (3, true, false),
        (1, false, true),
        (3, false, false),
        (0, false, true),
        (2, false, false),
    ];

    if let Some(attacker) =
        first_slider_attacker(squares, occ, target, side, BISHOP_DIRS, &[PieceType::Bishop])
    {
        return Some(attacker);
    }

    if let Some(attacker) =
        first_slider_attacker(squares, occ, target, side, ROOK_DIRS, &[PieceType::Rook])
    {
        return Some(attacker);
    }

    if let Some(attacker) =
        first_slider_attacker(squares, occ, target, side, QUEEN_DIRS, &[PieceType::Queen])
    {
        return Some(attacker);
    }

    for df in -1..=1 {
        for dr in -1..=1 {
            if df == 0 && dr == 0 {
                continue;
            }
            if let Some((sq, p)) = piece_on_square(squares, tf + df, tr + dr) {
                if p.color == side && p.kind == PieceType::King {
                    return Some((sq, p));
                }
            }
        }
    }

    None
}

fn see(board: &Board, mv: Move) -> i32 {
    if !is_capture_move(board, mv) {
        return 0;
    }

    let mut squares = board.squares;
    let moving = match squares[mv.from as usize] {
        Some(p) => p,
        None => return 0,
    };

    let mut occ = board.bb.occ_all();

    let captured = if mv.is_en_passant {
        let cap_sq = if moving.color == Color::White { mv.to - 8 } else { mv.to + 8 };
        let c = squares[cap_sq as usize];
        squares[cap_sq as usize] = None;
        occ &= !(1u64 << cap_sq);
        c
    } else {
        squares[mv.to as usize]
    };

    let captured = match captured {
        Some(c) => c,
        None => return 0,
    };

    squares[mv.from as usize] = None;
    occ &= !(1u64 << mv.from);
    occ |= 1u64 << mv.to;
    let mut placed = moving;
    if let Some(promo) = mv.promotion {
        placed.kind = promo;
    }
    squares[mv.to as usize] = Some(placed);

    let promo_delta = mv
        .promotion
        .map(piece_value)
        .unwrap_or(0)
        .saturating_sub(piece_value(PieceType::Pawn));

    let mut gain = [0i32; 32];
    let mut depth = 0usize;
    gain[0] = piece_value(captured.kind) + promo_delta;

    let mut side = board.side_to_move.opposite();
    let target = mv.to;
    let mut occupant_value = piece_value(placed.kind);

    loop {
        depth += 1;
        if depth >= gain.len() {
            break;
        }

        let Some((from, attacker)) = least_valuable_attacker(&squares, occ, target, side) else {
            break;
        };

        gain[depth] = occupant_value - gain[depth - 1];

        squares[from as usize] = None;
        occ &= !(1u64 << from);
        squares[target as usize] = Some(attacker);
        occupant_value = piece_value(attacker.kind);
        side = side.opposite();

        if gain[depth].max(-gain[depth - 1]) < 0 {
            break;
        }
    }

    while depth > 0 {
        gain[depth - 1] = -std::cmp::max(-gain[depth - 1], gain[depth]);
        depth -= 1;
    }

    gain[0]
}

fn move_order_score(
    board: &Board,
    engine: &EngineState,
    mv: Move,
    tt_move: Option<Move>,
    counter_move: Option<Move>,
    prev_move: Option<Move>,
    killers: &[Option<Move>; 2],
    history: &HistoryTable,
) -> i32 {
    if Some(mv) == tt_move {
        return 2_000_000;
    }

    if is_capture_move(board, mv) {
        let see_score = see(board, mv);
        return 1_000_000 + see_score * 12;
    }

    if Some(mv) == killers[0] {
        return 900_000;
    }
    if Some(mv) == counter_move {
        return 850_000;
    }
    if Some(mv) == killers[1] {
        return 800_000;
    }

    let promo = mv
        .promotion
        .map(piece_value)
        .unwrap_or(0)
        .saturating_sub(piece_value(PieceType::Pawn));
    let castle = if mv.is_castle { 10 } else { 0 };
    let continuation = continuation_history_score(engine, board.side_to_move, prev_move, mv);
    history_score(history, board.side_to_move, mv) + continuation / 2 + promo + castle
}

pub(crate) fn piece_value(kind: PieceType) -> i32 {
    match kind {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King => 20_000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn quiescence_checkmate_score_is_ply_aware() {
        thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let mut board = Board::new();
                board.set_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");

                let mut engine = EngineState::new();
                let key = board_hash(&board, &engine.zobrist);
                let mut rep_stack = vec![key];
                let mut search = Search::new(None);

                let score = quiescence(
                    &mut board,
                    -INF,
                    INF,
                    &mut search,
                    &mut engine,
                    &mut rep_stack,
                    key,
                    3,
                );

                assert_eq!(score, -MATE_SCORE + 3);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn quiescence_tt_probe_is_ply_aware_for_mates() {
        thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let mut board = Board::new();
                board.set_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");

                let mut engine = EngineState::new();
                let key = board_hash(&board, &engine.zobrist);
                engine.tt.store(
                    key,
                    0,
                    score_to_tt(-MATE_SCORE + 5, 5),
                    None,
                    Bound::Exact,
                    None,
                );

                let mut rep_stack = vec![key];
                let mut search = Search::new(None);
                let score = quiescence(
                    &mut board,
                    -INF,
                    INF,
                    &mut search,
                    &mut engine,
                    &mut rep_stack,
                    key,
                    5,
                );

                assert_eq!(score, -MATE_SCORE + 5);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
