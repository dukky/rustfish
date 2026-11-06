//! Uncertainty head experiment (offline only; the engine does not read it).
//!
//! A second output per bucket on the same accumulator,
//! `u = U_SCALE * softplus(w_u . screlu(acc) + b_u)`, trained to predict the
//! net's own error in WDL space: `t = |sigmoid(score / SCALE) - sigmoid(eval / SCALE)|`,
//! where `score` is the record's search score (side to move POV) and `eval`
//! the current net's static eval in engine units (`--eval-mul` x net output,
//! default 0.5 as in src/eval.rs). The target is detached (no gradient through the
//! eval). Squared loss, so `u` estimates the conditional mean of `t`.
//!
//! ```text
//! # frozen: train only the head on a fixed net's accumulators
//! nnue-trainer uncert-train --net net/nnue.bin --data d.bin --val val.bin --out u.head
//! # shared: fine-tune net + head, alpha * head loss added to the main loss;
//! # --no-ft-grad stops the head's gradient at the accumulator (control run)
//! nnue-trainer uncert-train --net net/nnue.bin --head u.head --shared --alpha 1 ...
//! # per-position dump for analysis (scripts/uncert_analysis.py)
//! nnue-trainer uncert-dump --net net/nnue.bin --head u.head --val val.bin --out val.csv
//! ```
//!
//! Head file: little-endian f32, `buckets * 2H` weights (bucket-major, stm half
//! first) then `buckets` biases; no header.

use super::*;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub const U_SCALE: f32 = 0.1;

/// The engine's static eval is `EVAL_MUL * net output` (src/eval.rs halves it
/// since ad53daa), and datagen search scores are in those units, so the
/// residual must compare the score with the *scaled* eval. `--eval-mul`.
static EVAL_MUL: AtomicU32 = AtomicU32::new(0x3F00_0000); // 0.5

fn eval_mul() -> f32 {
    f32::from_bits(EVAL_MUL.load(Ordering::Relaxed))
}

/// The static eval as the search sees it, as a probability.
#[inline]
fn eval_prob(out: f32) -> f32 {
    sigmoid(out * eval_mul())
}

#[inline]
fn softplus(z: f32) -> f32 {
    if z > 20.0 {
        z
    } else {
        z.exp().ln_1p()
    }
}

pub fn head_len(sh: Shape) -> usize {
    sh.buckets * (2 * sh.h + 1)
}

/// Head pre-activation for one sample; `acc` holds the forward pass's accumulators.
#[inline]
fn head_z(q: &[f32], sh: Shape, bucket: usize, acc: &[f32]) -> f32 {
    let h2 = 2 * sh.h;
    let w = &q[bucket * h2..(bucket + 1) * h2];
    let mut z = q[sh.buckets * h2 + bucket];
    for (x, w) in acc.iter().zip(w) {
        let a = x.clamp(0.0, 1.0);
        z += a * a * *w;
    }
    z
}

/// Search score of the record, side-to-move POV, as a probability.
#[inline]
fn score_prob(r: &Rec) -> f32 {
    let mut sc = i16::from_le_bytes([r[24], r[25]]) as f32;
    if r[27] == 1 {
        sc = -sc;
    }
    sigmoid(sc / SCALE)
}

/// One sample. `g` is the gradient of the trained parameters: the head only
/// (frozen) or `[net | head]` (shared). Returns (main loss, head loss).
#[allow(clippy::too_many_arguments)]
#[inline]
fn sample(
    p: &[f32],
    q: &[f32],
    sh: Shape,
    s: &Sample,
    sp: f32,
    g: &mut [f32],
    shared: Option<(f32, bool)>,
    acc: &mut [f32],
    dacc: &mut [f32],
) -> (f32, f32) {
    let h2 = 2 * sh.h;
    let out = forward(p, sh, s, acc);
    let pred = sigmoid(out);
    let t = (sp - eval_prob(out)).abs();
    let z = head_z(q, sh, s.bucket, acc);
    let u = U_SCALE * softplus(z);
    let du = 2.0 * (u - t) * U_SCALE * sigmoid(z);
    let (gp, gq) = if shared.is_some() { g.split_at_mut(sh.len()) } else { g.split_at_mut(0) };
    let alpha = shared.map_or(1.0, |x| x.0);
    let du = alpha * du;
    // Head gradients.
    {
        let gw = &mut gq[s.bucket * h2..(s.bucket + 1) * h2];
        for (gw, x) in gw.iter_mut().zip(acc.iter()) {
            let a = x.clamp(0.0, 1.0);
            *gw += du * a * a;
        }
        gq[sh.buckets * h2 + s.bucket] += du;
    }
    let main_loss = (pred - s.target) * (pred - s.target);
    let Some((_, ft_grad)) = shared else { return (main_loss, (u - t) * (u - t)) };

    // Main loss backward (as train_sample), plus the head's share of dacc.
    let err = pred - s.target;
    let grad = 2.0 * err * pred * (1.0 - pred);
    let ow0 = sh.out_w() + s.bucket * h2;
    gp[sh.out_b() + s.bucket] += grad;
    let wu = &q[s.bucket * h2..(s.bucket + 1) * h2];
    {
        let ow = &p[ow0..ow0 + h2];
        let gow = &mut gp[ow0..ow0 + h2];
        for i in 0..h2 {
            let x = acc[i];
            let a = x.clamp(0.0, 1.0);
            gow[i] += grad * a * a;
            let mut d = grad * ow[i];
            if ft_grad {
                d += du * wu[i];
            }
            dacc[i] = if x > 0.0 && x < 1.0 { d * 2.0 * a } else { 0.0 };
        }
    }
    let (gw, rest) = gp.split_at_mut(INPUTS * sh.h);
    let gb = &mut rest[..sh.h];
    for side in 0..2 {
        let d = &dacc[side * sh.h..(side + 1) * sh.h];
        for (x, y) in gb.iter_mut().zip(d) {
            *x += *y;
        }
        for &f in &s.f[side][..s.n] {
            let row = &mut gw[f as usize * sh.h..f as usize * sh.h + sh.h];
            for (x, y) in row.iter_mut().zip(d) {
                *x += *y;
            }
        }
    }
    (main_loss, (u - t) * (u - t))
}

/// (main val loss, head val loss, mean target, mean u)
fn validate_u(p: &[f32], q: &[f32], sh: Shape, val: &[Rec], lambda: f32, threads: usize) -> [f64; 4] {
    if val.is_empty() {
        return [f64::NAN; 4];
    }
    let chunk = val.len().div_ceil(threads);
    let parts: Vec<[f64; 4]> = std::thread::scope(|sc| {
        let hs: Vec<_> = val
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    let mut acc = vec![0.0f32; 2 * sh.h];
                    let mut r4 = [0.0f64; 4];
                    for r in part {
                        let s = decode(r, lambda, sh.buckets, false);
                        let out = forward(p, sh, &s, &mut acc);
                        let pred = sigmoid(out);
                        let t = (score_prob(r) - eval_prob(out)).abs();
                        let u = U_SCALE * softplus(head_z(q, sh, s.bucket, &acc));
                        r4[0] += ((pred - s.target) * (pred - s.target)) as f64;
                        r4[1] += ((u - t) * (u - t)) as f64;
                        r4[2] += t as f64;
                        r4[3] += u as f64;
                    }
                    r4
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let n = val.len() as f64;
    let mut out = [0.0; 4];
    for x in &parts {
        for i in 0..4 {
            out[i] += x[i] / n;
        }
    }
    out
}

fn load_head(path: &Path, sh: Shape) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    let q: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(q.len(), head_len(sh), "head shape differs from net");
    q
}

fn save_f32(path: &Path, v: &[f32]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    for x in v {
        w.write_all(&x.to_le_bytes()).unwrap();
    }
}

pub fn cmd_uncert_train(args: &[String]) {
    let (mut net, mut head, mut data, mut val_path) = (None::<PathBuf>, None::<PathBuf>, Vec::new(), None::<PathBuf>);
    let mut out = PathBuf::from("u.head");
    let mut buckets = 1usize;
    let (mut epochs, mut epoch_size, mut batch) = (1u64, 0u64, 16384usize);
    let (mut lr, mut lr_end, mut lambda, mut alpha) = (1e-3f32, 1e-5f32, 0.5f32, 1.0f32);
    let (mut threads, mut seed, mut shared, mut ft_grad) = (threads_default(), 1u64, false, true);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("missing value for {a}")).clone();
        match a.as_str() {
            "--net" => net = Some(PathBuf::from(v())),
            "--head" => head = Some(PathBuf::from(v())),
            "--data" => data.push(PathBuf::from(v())),
            "--val" => val_path = Some(PathBuf::from(v())),
            "--out" => out = PathBuf::from(v()),
            "--buckets" => buckets = v().parse().unwrap(),
            "--epochs" => epochs = v().parse().unwrap(),
            "--epoch-size" => epoch_size = v().parse().unwrap(),
            "--batch" => batch = v().parse().unwrap(),
            "--lr" => lr = v().parse().unwrap(),
            "--lr-end" => lr_end = v().parse().unwrap(),
            "--lambda" => lambda = v().parse().unwrap(),
            "--alpha" => alpha = v().parse().unwrap(),
            "--eval-mul" => EVAL_MUL.store(v().parse::<f32>().unwrap().to_bits(), Ordering::Relaxed),
            "--threads" => threads = v().parse().unwrap(),
            "--seed" => seed = v().parse().unwrap(),
            "--shared" => shared = true,
            "--no-ft-grad" => ft_grad = false,
            other => panic!("unknown option {other}"),
        }
    }
    let (mut p, sh) = load_net(&net.expect("--net"), buckets);
    let mut q = match &head {
        Some(hp) => load_head(hp, sh),
        None => vec![0.0f32; head_len(sh)],
    };
    if epoch_size == 0 {
        epoch_size = data.iter().map(|p| std::fs::metadata(p).unwrap().len() / REC as u64).sum();
    }
    let total = epochs * epoch_size;
    let val = val_path.as_deref().map(read_recs).unwrap_or_default();
    println!(
        "uncert-train H={} buckets={} shared={shared} ft_grad={ft_grad} alpha={alpha} eval_mul={} epochs={epochs} x {epoch_size} lr {lr}->{lr_end} threads={threads}",
        sh.h,
        sh.buckets,
        eval_mul()
    );
    let v = validate_u(&p, &q, sh, &val, lambda, threads);
    println!("start val main {:.6} head {:.7} mean t {:.5} mean u {:.5}", v[0], v[1], v[2], v[3]);

    // Trained vector: head only, or [net | head].
    let mut tp: Vec<f32> = if shared { p.iter().chain(q.iter()).copied().collect() } else { q.clone() };
    let n = tp.len();
    let mut opt = Adam { m: vec![0.0; n], v: vec![0.0; n], step: 0 };
    let mut grads: Vec<Vec<f32>> = (0..threads).map(|_| vec![0.0f32; n]).collect();
    let mut feed = Feed::new(data, seed ^ 0xF00D);
    let mut recs: Vec<Rec> = Vec::with_capacity(batch);
    let started = Instant::now();
    let (mut seen, mut ep) = (0u64, 0u64);
    let (mut lm, mut lu, mut ln) = (0.0f64, 0.0f64, 0u64);
    let cfg = shared.then_some((alpha, ft_grad));
    while seen < total {
        let t = seen as f64 / total as f64;
        let cur_lr = lr_end + 0.5 * (lr - lr_end) * (1.0 + (std::f64::consts::PI * t).cos() as f32);
        feed.batch(&mut recs, batch);
        let chunk = 256usize;
        let n_chunks = recs.len().div_ceil(chunk);
        let next = AtomicUsize::new(0);
        let (pp, qq): (&[f32], &[f32]) = if shared { tp.split_at(sh.len()) } else { (&p, &tp) };
        let (a, b): (f64, f64) = std::thread::scope(|sc| {
            let (recs, next) = (&recs, &next);
            let hs: Vec<_> = grads
                .iter_mut()
                .map(|g| {
                    sc.spawn(move || {
                        g.fill(0.0);
                        let mut acc = vec![0.0f32; 2 * sh.h];
                        let mut dacc = vec![0.0f32; 2 * sh.h];
                        let (mut a, mut b) = (0.0f64, 0.0f64);
                        loop {
                            let c = next.fetch_add(1, Ordering::Relaxed);
                            if c >= n_chunks {
                                break;
                            }
                            for r in &recs[c * chunk..((c + 1) * chunk).min(recs.len())] {
                                let s = decode(r, lambda, sh.buckets, false);
                                let (x, y) = sample(pp, qq, sh, &s, score_prob(r), g, cfg, &mut acc, &mut dacc);
                                a += x as f64;
                                b += y as f64;
                            }
                        }
                        (a, b)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).fold((0.0, 0.0), |s, x| (s.0 + x.0, s.1 + x.1))
        });
        adam_step(&mut tp, &mut opt, &grads, cur_lr, 0.0, 1.0 / recs.len() as f32, threads);
        seen += recs.len() as u64;
        lm += a;
        lu += b;
        ln += recs.len() as u64;
        let report = (seen / 1_000_000) != ((seen - recs.len() as u64) / 1_000_000) && (seen / 1_000_000) % 5 == 0;
        let epoch_end = seen / epoch_size != (seen - recs.len() as u64) / epoch_size || seen >= total;
        if report || epoch_end {
            println!(
                "  {}M lr {cur_lr:.2e} train main {:.6} head {:.7} {:.0} pos/s",
                seen / 1_000_000,
                lm / ln as f64,
                lu / ln as f64,
                seen as f64 / started.elapsed().as_secs_f64()
            );
            (lm, lu, ln) = (0.0, 0.0, 0);
        }
        if epoch_end {
            ep += 1;
            if shared {
                p.copy_from_slice(&tp[..sh.len()]);
                q.copy_from_slice(&tp[sh.len()..]);
            } else {
                q.copy_from_slice(&tp);
            }
            let v = validate_u(&p, &q, sh, &val, lambda, threads);
            println!("epoch {ep} val main {:.6} head {:.7} mean t {:.5} mean u {:.5}", v[0], v[1], v[2], v[3]);
        }
    }
    save_f32(&out, &q);
    if shared {
        let np = out.with_extension("net.bin");
        File::create(&np).unwrap().write_all(&quantize(&p, sh)).unwrap();
        println!("wrote {} and {}", out.display(), np.display());
    } else {
        println!("wrote {}", out.display());
    }
}

// ------------------------------------------------------- cheap board features

const VAL: [i32; 6] = [1, 3, 3, 5, 9, 100];

/// (colour, kind) per square from a record.
fn board(r: &Rec) -> [Option<(u8, u8)>; 64] {
    let mut sq = [None; 64];
    let mut occ = u64::from_le_bytes(r[..8].try_into().unwrap());
    let mut n = 0;
    while occ != 0 {
        let s = occ.trailing_zeros() as usize;
        occ &= occ - 1;
        let nib = r[8 + n / 2] >> (4 * (n & 1)) & 0xF;
        sq[s] = Some((nib >> 3, nib & 7));
        n += 1;
    }
    sq
}

/// For each square: lowest value of a piece of each colour attacking it (0 = none).
fn min_attackers(b: &[Option<(u8, u8)>; 64]) -> [[i32; 64]; 2] {
    let mut m = [[0i32; 64]; 2];
    let mut hit = |c: usize, t: i32, v: i32| {
        let x = &mut m[c][t as usize];
        if *x == 0 || v < *x {
            *x = v;
        }
    };
    for s in 0..64i32 {
        let Some((c, k)) = b[s as usize] else { continue };
        let (f, rk) = (s % 8, s / 8);
        let v = VAL[k as usize];
        let ci = c as usize;
        let mut step = |df: i32, dr: i32, slide: bool| {
            let (mut ff, mut rr) = (f + df, rk + dr);
            while (0..8).contains(&ff) && (0..8).contains(&rr) {
                let t = rr * 8 + ff;
                hit(ci, t, v);
                if !slide || b[t as usize].is_some() {
                    break;
                }
                ff += df;
                rr += dr;
            }
        };
        match k {
            0 => {
                let dr = if c == 0 { 1 } else { -1 };
                step(-1, dr, false);
                step(1, dr, false);
            }
            1 => {
                for (df, dr) in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
                    step(df, dr, false);
                }
            }
            5 => {
                for df in -1..=1 {
                    for dr in -1..=1 {
                        if df != 0 || dr != 0 {
                            step(df, dr, false);
                        }
                    }
                }
            }
            _ => {
                if k == 2 || k == 4 {
                    for (df, dr) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                        step(df, dr, true);
                    }
                }
                if k == 3 || k == 4 {
                    for (df, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        step(df, dr, true);
                    }
                }
            }
        }
    }
    m
}

/// (pieces, phase 0..24, material stm POV in pawns, en-prise stm, en-prise opponent,
/// attacked stm, attacked opponent). En prise: attacked and undefended, or attacked
/// by a cheaper piece.
fn features(r: &Rec) -> [i32; 7] {
    let b = board(r);
    let m = min_attackers(&b);
    let stm = r[27] as usize;
    let mut f = [0i32; 7];
    for s in 0..64 {
        let Some((c, k)) = b[s] else { continue };
        let c = c as usize;
        f[0] += 1;
        f[1] += [0, 1, 1, 2, 4, 0][k as usize];
        if k != 5 {
            f[2] += if c == stm { VAL[k as usize] } else { -VAL[k as usize] };
            let enemy = m[1 - c][s];
            if enemy != 0 {
                let own = if c == stm { 5 } else { 6 };
                f[own] += 1;
                if m[c][s] == 0 || enemy < VAL[k as usize] {
                    f[if c == stm { 3 } else { 4 }] += 1;
                }
            }
        }
    }
    f
}

pub fn cmd_uncert_dump(args: &[String]) {
    let (mut net, mut head, mut val, mut out, mut buckets) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>, None::<PathBuf>, 1usize);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--net" => net = Some(PathBuf::from(it.next().unwrap())),
            "--head" => head = Some(PathBuf::from(it.next().unwrap())),
            "--val" => val = Some(PathBuf::from(it.next().unwrap())),
            "--out" => out = Some(PathBuf::from(it.next().unwrap())),
            "--buckets" => buckets = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown option {other}"),
        }
    }
    let (p, sh) = load_net(&net.expect("--net"), buckets);
    let q = load_head(&head.expect("--head"), sh);
    let val = read_recs(&val.expect("--val"));
    let threads = threads_default().min(4);
    let chunk = val.len().div_ceil(threads);
    let rows: Vec<String> = std::thread::scope(|sc| {
        let (p, q) = (&p, &q);
        let hs: Vec<_> = val
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    let mut acc = vec![0.0f32; 2 * sh.h];
                    let mut rows = Vec::with_capacity(part.len());
                    for r in part {
                        let s = decode(r, 0.5, sh.buckets, false);
                        let o = forward(p, sh, &s, &mut acc);
                        let u = U_SCALE * softplus(head_z(q, sh, s.bucket, &acc));
                        let mut sc = i16::from_le_bytes([r[24], r[25]]) as i32;
                        let mut wdl = r[26] as i32;
                        if r[27] == 1 {
                            sc = -sc;
                            wdl = 2 - wdl;
                        }
                        let f = features(r);
                        rows.push(format!(
                            "{:.1},{sc},{wdl},{u:.6},{},{},{},{},{},{},{}",
                            o * SCALE,
                            f[0],
                            f[1],
                            f[2],
                            f[3],
                            f[4],
                            f[5],
                            f[6]
                        ));
                    }
                    rows
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut w = BufWriter::new(File::create(out.expect("--out")).unwrap());
    writeln!(w, "eval,score,wdl,u,pieces,phase,material,enprise_us,enprise_them,attacked_us,attacked_them").unwrap();
    for r in rows {
        writeln!(w, "{r}").unwrap();
    }
    eprintln!("dumped {} rows", val.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_gradient_matches_finite_difference() {
        let sh = Shape { h: 16, buckets: 2 };
        let mut rng = Rng::new(9);
        let mut p = init_params(sh, &mut rng);
        for x in &mut p[sh.ft_b()..sh.out_w()] {
            *x = 0.3;
        }
        let q: Vec<f32> = (0..head_len(sh)).map(|_| rng.uniform(0.5)).collect();
        let r = encode("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 | 135 | 1.0").unwrap();
        let s = decode(&r, 0.5, sh.buckets, false);
        let sp = score_prob(&r);
        let alpha = 0.7f32;
        let mut g = vec![0.0f32; sh.len() + head_len(sh)];
        let (mut acc, mut dacc) = (vec![0.0; 2 * sh.h], vec![0.0; 2 * sh.h]);
        sample(&p, &q, sh, &s, sp, &mut g, Some((alpha, true)), &mut acc, &mut dacc);
        // Total loss with the target t held fixed at its value for the base params.
        let t0 = {
            let mut acc = vec![0.0; 2 * sh.h];
            (sp - eval_prob(forward(&p, sh, &s, &mut acc))).abs()
        };
        let loss = |all: &[f32]| {
            let (p, q) = all.split_at(sh.len());
            let mut acc = vec![0.0; 2 * sh.h];
            let pred = sigmoid(forward(p, sh, &s, &mut acc));
            let u = U_SCALE * softplus(head_z(q, sh, s.bucket, &acc));
            ((pred - s.target).powi(2) + alpha * (u - t0).powi(2)) as f64
        };
        let all: Vec<f32> = p.iter().chain(q.iter()).copied().collect();
        let f0 = s.f[0][0] as usize * sh.h;
        let hb = sh.len();
        for &i in &[f0, f0 + 3, sh.ft_b() + 1, sh.out_b() + s.bucket, hb + s.bucket * 2 * sh.h + 4, hb + 2 * 2 * sh.h + s.bucket] {
            let eps = 1e-3;
            let mut a = all.clone();
            a[i] += eps;
            let mut b = all.clone();
            b[i] -= eps;
            let num = (loss(&a) - loss(&b)) / (2.0 * eps as f64);
            assert!((num - g[i] as f64).abs() < 1e-4 + 1e-2 * num.abs(), "param {i}: numeric {num} vs analytic {}", g[i]);
        }
    }

    #[test]
    fn en_prise_counts() {
        // White to move; black knight d5 attacked by pawn e4 (cheaper) -> en prise them.
        let r = encode("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1 | 0 | 0.5").unwrap();
        let f = features(&r);
        assert_eq!(f[0], 4);
        assert_eq!(f[4], 1, "knight en prise");
        assert_eq!(f[3], 0, "d5 knight does not hit e4");
        assert_eq!(f[2], -2, "material stm POV");
    }
}
