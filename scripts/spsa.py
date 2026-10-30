#!/usr/bin/env python3
"""SPSA tuning of rustfish UCI spin options, in the style of fishtest/OpenBench.

State lives in one JSON file (see tune/spsa-params.json):

  {"iteration": 0, "total_iterations": 200, "alpha": 0.602, "gamma": 0.101,
   "A_ratio": 0.1,
   "params": {"FutilityMarginD1": {"value": 120, "min": 40, "max": 300,
                                    "c_end": 12, "r_end": 0.002}, ...}}

`value` is the float theta; engines get it rounded. The gain schedules are
fishtest's, defined by the end-of-schedule values so they are easy to set:

  N = total_iterations, A = A_ratio * N, k = completed iterations
  c   = c_end * N^gamma              c_k = c / (k+1)^gamma        (c_k -> c_end)
  a   = r_end * c_end^2 * (A+N)^alpha
  a_k = a / (A+k+1)^alpha                                         (a_k -> r_end*c_end^2)

One iteration: draw delta_i = +-1 per param, play `plus` (theta + c_k*delta)
against `minus` (theta - c_k*delta) for a batch of game pairs, then

  theta_i += a_k / c_k_i * (W - L) * delta_i

with W - L counted from plus's side over the batch. This is the standard
SPSA step a_k * (y+ - y-) / (2 c_k delta) with y = game score, written per
game like fishtest; the batch just sums the per-pair updates at fixed k.
theta is clamped to [min, max] after each step.

Integer options: plus/minus values are rounded *stochastically* (seeded),
so their expectation is exactly theta +- c_k even when c_k < 1.

Parallel shards can't share theta mid-run, so the workflow runs independent
SPSA chains from the same theta with different seeds and `merge` averages
their theta deltas weighted by iterations (`--combine sum` adds them
instead). Either is an approximation of sequential SPSA; see merge(). Run the workflow again from the merged file to
continue the schedule.

Commands:
  start   PARAMS --seed S --out STATE          copy params, set seed (shard start)
  perturb STATE [--plus-file F --minus-file F]  pick delta, write fastchess option args
  update  STATE --fastchess-log LOG | --ptnml a,b,c,d,e
  merge   BASE SHARD... --out NEW [--summary MD] [--combine mean|sum]
  toy     [--rounds R --chain-iters N --shards S --combine C]  convergence demo on a fake quadratic
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sprt  # noqa: E402  (pentanomial parsing of fastchess logs)


# ---------------------------------------------------------------- schedules

def schedule(state, name, k=None):
    """Return (c_k, a_k) for one param at iteration k (default: current)."""
    p = state["params"][name]
    n = max(1, int(state["total_iterations"]))
    k = min(state["iteration"] if k is None else k, n - 1)
    alpha, gamma = state.get("alpha", 0.602), state.get("gamma", 0.101)
    big_a = state.get("A_ratio", 0.1) * n
    c = p["c_end"] * n ** gamma
    a = p["r_end"] * p["c_end"] ** 2 * (big_a + n) ** alpha
    return c / (k + 1) ** gamma, a / (big_a + k + 1) ** alpha


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def stochastic_round(x, rng):
    lo = math.floor(x)
    return int(lo + (rng.random() < x - lo))


# ---------------------------------------------------------------- core steps

def perturb(state):
    """Choose delta for the current iteration; return (plus, minus) option dicts."""
    seed, k = state.get("seed", 0), state["iteration"]
    rng = random.Random(f"{seed}:{k}")
    pending = {"k": k, "delta": {}, "c_k": {}}
    plus, minus = {}, {}
    for name, p in state["params"].items():
        c_k, _ = schedule(state, name)
        d = rng.choice((-1, 1))
        pending["delta"][name], pending["c_k"][name] = d, c_k
        hi = clamp(p["value"] + c_k * d, p["min"], p["max"])
        lo = clamp(p["value"] - c_k * d, p["min"], p["max"])
        if p.get("float"):
            plus[name], minus[name] = hi, lo
        else:
            plus[name], minus[name] = stochastic_round(hi, rng), stochastic_round(lo, rng)
    state["pending"] = pending
    return plus, minus


def update(state, ptnml):
    """Apply the pending perturbation's result (pentanomial from plus's side)."""
    pending = state.pop("pending")
    if pending["k"] != state["iteration"]:
        raise ValueError("pending perturbation is from a different iteration")
    # Pair score index i is 0..4 half-points; it contributes i - 2 to W - L.
    result = sum((i - 2) * n for i, n in enumerate(ptnml))
    for name, p in state["params"].items():
        _, a_k = schedule(state, name)
        c_k, d = pending["c_k"][name], pending["delta"][name]
        p["value"] = clamp(p["value"] + a_k / c_k * result * d, p["min"], p["max"])
    state["iteration"] += 1
    state["iterations_run"] = state.get("iterations_run", 0) + 1
    state["ptnml"] = [a + b for a, b in zip(state.get("ptnml", [0] * 5), ptnml)]
    return result


def merge(base, shards, combine="mean"):
    """Combine shards' theta deltas from base.

    mean: iteration-weighted mean delta. SPSA with the gradient averaged over
          shards: lower noise, same step size, so progress per job equals one
          chain's. Safe default.
    sum:  add the deltas, as if the chains' updates had been applied in turn
          to one theta (fishtest workers do this with short staleness). Moves
          about `shards` times further per job; can overshoot early in the
          schedule, so use it only with modest r_end.
    """
    weights = [s.get("iterations_run", s["iteration"] - base["iteration"]) for s in shards]
    total_w = sum(weights)
    new = json.loads(json.dumps(base))
    new.pop("pending", None)
    new.pop("seed", None)
    if total_w == 0:
        return new
    for name, p in new["params"].items():
        delta = sum(w * (s["params"][name]["value"] - p["value"])
                    for s, w in zip(shards, weights)) / total_w
        if combine == "sum":
            delta *= len(shards)  # weighted mean x shards == plain sum when equal
        p["value"] = round(clamp(p["value"] + delta, p["min"], p["max"]), 4)
    # sum applies every shard iteration as an update; mean applies one chain's worth.
    new["iteration"] = base["iteration"] + (
        total_w if combine == "sum" else round(total_w / len(shards)))
    ptnml = [0] * 5
    for s in shards:
        ptnml = [a + b for a, b in zip(ptnml, s.get("ptnml", [0] * 5))]
    new.setdefault("history", []).append({
        "shards": len(shards), "combine": combine, "iterations": weights, "ptnml": ptnml,
        "games": 2 * sum(ptnml), "iteration_after": new["iteration"]})
    new.pop("ptnml", None)
    new.pop("iterations_run", None)
    return new


def engine_values(state):
    return {n: (p["value"] if p.get("float") else round(p["value"]))
            for n, p in state["params"].items()}


def fastchess_args(opts):
    return " ".join(f"option.{k}={v}" for k, v in opts.items())


# ---------------------------------------------------------------- io helpers

def load(path):
    with open(path) as fh:
        return json.load(fh)


def save(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def summary_markdown(base, new):
    lines = ["| param | start | tuned | engine value | change |", "|---|---|---|---|---|"]
    for name, p in new["params"].items():
        old = base["params"][name]["value"]
        lines.append(f"| {name} | {old:g} | {p['value']:.2f} | {round(p['value'])} "
                     f"| {p['value'] - old:+.2f} |")
    h = new.get("history", [{}])[-1]
    head = (f"## SPSA: iteration {base['iteration']} -> {new['iteration']} "
            f"of {new['total_iterations']}\n\n"
            f"{h.get('shards', 0)} shards, {h.get('games', 0)} games, "
            f"pentanomial (plus side) {h.get('ptnml')}\n\n"
            "Shards ran independent chains from the same theta; this is their "
            + ("summed" if h.get("combine") == "sum" else "iteration-weighted mean")
            + " delta (a parallel approximation of SPSA).\n\n")
    return head + "\n".join(lines) + "\n\n```json\n" + json.dumps(new, indent=2) + "\n```\n"


# ---------------------------------------------------------------- toy demo

def toy(rounds, chain_iters, shards, pairs, seed, combine="mean"):
    """Fake engine: Elo = -400 * sum(((x - opt)/range)^2); games from logistic."""
    true_opt = {"Alpha": 150.0, "Beta": 7.0, "Gamma": 40.0}
    per_round = chain_iters * (shards if combine == "sum" else 1)
    state = {"iteration": 0, "total_iterations": rounds * per_round, "alpha": 0.602,
             "gamma": 0.101, "A_ratio": 0.1, "params": {
                 "Alpha": {"value": 90, "min": 0, "max": 300, "c_end": 15, "r_end": 0.002},
                 "Beta": {"value": 3, "min": 1, "max": 12, "c_end": 0.8, "r_end": 0.002},
                 "Gamma": {"value": 70, "min": 0, "max": 100, "c_end": 6, "r_end": 0.002}}}
    ranges = {n: p["max"] - p["min"] for n, p in state["params"].items()}

    def elo(vals):
        return -400.0 * sum(((vals[n] - true_opt[n]) / ranges[n]) ** 2 for n in vals)

    def play(plus, minus, rng):
        p = 1.0 / (1.0 + 10 ** (-(elo(plus) - elo(minus)) / 400.0))
        draw = 0.4
        ptnml = [0] * 5
        for _ in range(pairs):
            pts = 0
            for _ in range(2):
                r = rng.random()
                pts += 2 if r < p * (1 - draw) else (1 if r < p * (1 - draw) + draw else 0)
            ptnml[pts] += 1
        return ptnml

    def report(tag, st):
        vals = {n: p["value"] for n, p in st["params"].items()}
        print(f"{tag:>12}: " + ", ".join(f"{n}={v:7.2f}" for n, v in vals.items())
              + f"   Elo vs optimum {elo(vals):+7.1f}")

    print("optimum      : " + ", ".join(f"{n}={v:7.2f}" for n, v in true_opt.items()))
    report("start", state)
    rng = random.Random(seed)
    # Same scheme as the workflow: each round runs `shards` independent
    # chains of chain_iters from the merged theta, then merges them.
    for rnd in range(rounds):
        chains = []
        for s in range(shards):
            st = json.loads(json.dumps(state))
            st.update(seed=f"{seed}-{rnd}-{s}", iterations_run=0, ptnml=[0] * 5)
            for _ in range(chain_iters):
                plus, minus = perturb(st)
                update(st, play(plus, minus, rng))
            chains.append(st)
        state = merge(state, chains, combine)
        if rnd + 1 == rounds or (rnd + 1) % max(1, rounds // 4) == 0:
            report(f"round {rnd + 1}", state)
    vals = {n: p["value"] for n, p in state["params"].items()}
    return state, elo(vals)


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start")
    s.add_argument("params")
    s.add_argument("--seed", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("perturb")
    s.add_argument("state")
    s.add_argument("--plus-file")
    s.add_argument("--minus-file")

    s = sub.add_parser("update")
    s.add_argument("state")
    s.add_argument("--fastchess-log")
    s.add_argument("--ptnml")

    s = sub.add_parser("merge")
    s.add_argument("base")
    s.add_argument("shards", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--summary")
    s.add_argument("--combine", choices=("mean", "sum"), default="mean")

    s = sub.add_parser("show")
    s.add_argument("state")

    s = sub.add_parser("toy")
    s.add_argument("--rounds", type=int, default=4)
    s.add_argument("--chain-iters", type=int, default=500)
    s.add_argument("--shards", type=int, default=8)
    s.add_argument("--pairs", type=int, default=8)
    s.add_argument("--seed", default="toy")
    s.add_argument("--combine", choices=("mean", "sum"), default="mean")

    args = ap.parse_args()

    if args.cmd == "start":
        state = load(args.params)
        state.pop("pending", None)
        state.update(seed=args.seed, iterations_run=0, ptnml=[0] * 5)
        save(args.out, state)
    elif args.cmd == "perturb":
        state = load(args.state)
        plus, minus = perturb(state)
        save(args.state, state)
        for path, opts in ((args.plus_file, plus), (args.minus_file, minus)):
            if path:
                with open(path, "w") as fh:
                    fh.write(fastchess_args(opts) + "\n")
        print(f"iteration {state['iteration']}\n  plus:  {fastchess_args(plus)}\n"
              f"  minus: {fastchess_args(minus)}")
    elif args.cmd == "update":
        state = load(args.state)
        if "pending" not in state:
            sys.exit("no pending perturbation; run perturb first")
        try:
            ptnml = (sprt.parse_ptnml(args.ptnml) if args.ptnml
                     else sprt.ptnml_from_fastchess_log(args.fastchess_log))
        except (ValueError, OSError, TypeError) as exc:
            # A crashed batch is discarded, not fatal: the next perturb redraws.
            print(f"warning: no result ({exc}); discarding iteration {state['iteration']}")
            state.pop("pending")
            state["skipped"] = state.get("skipped", 0) + 1
            save(args.state, state)
            return
        result = update(state, ptnml)
        save(args.state, state)
        print(f"ptnml {ptnml}, plus W-L {result:+d} -> iteration {state['iteration']}: "
              + ", ".join(f"{n}={p['value']:.2f}" for n, p in state["params"].items()))
    elif args.cmd == "merge":
        base = load(args.base)
        shards = [load(p) for p in args.shards]
        new = merge(base, shards, args.combine)
        save(args.out, new)
        md = summary_markdown(base, new)
        if args.summary:
            with open(args.summary, "a") as fh:
                fh.write(md)
        print(md)
    elif args.cmd == "show":
        print(json.dumps(engine_values(load(args.state)), indent=2))
    elif args.cmd == "toy":
        _, final_elo = toy(args.rounds, args.chain_iters, args.shards, args.pairs, args.seed, args.combine)
        print(f"final Elo deficit vs optimum: {final_elo:+.1f}")


if __name__ == "__main__":
    main()
