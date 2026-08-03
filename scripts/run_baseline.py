"""Produce a frozen baseline (spec principle 10).

Baselines are generated ONCE, by hand, and never regenerated automatically by
the training pipeline; evaluate.py only reads their stored results. This script
is the deliberate producer.

    python -m scripts.run_baseline --baseline BASE001_de_plain \
        --config experiments/EXP002_dqn_restart/config.yaml
    python -m scripts.run_baseline --baseline BASE002_fixed_schedule \
        --config experiments/EXP002_dqn_restart/config.yaml
    python -m scripts.run_baseline --baseline BASE003_random \
        --config experiments/EXP002_dqn_restart/config.yaml

Add --smoke to produce a smoke-scale baseline (D=5, budget=3000, 2 seeds) that
matches a smoke-scale evaluation.

Every baseline uses the IDENTICAL evaluation seed list, function id(s),
dimension, and total FE budget as the agent, all read from the experiment config
(runs_per_function, seed_offset, benchmark.*) — never hardcoded. That parity is
what makes the later paired comparison in evaluate.py legitimate; evaluate.py
re-checks it from baseline_config.yaml and refuses to compare on a mismatch.

Baselines are stored per experiment and per function. --function N produces one
function's baseline; omitting it produces every function in the config. Writes,
for each function, baselines/<experiment>/f<N>/<name>/:
    results.csv           seed, function_id, best_error, evals_used
    baseline_config.yaml  the exact settings used (function/dim/budget/seeds +
                          the baseline's own constants)
    run_metadata.json     same schema as a training/eval job

The three baselines:
  BASE001_de_plain       one full-budget DE segment, no environment, no restarts
                         — "does the multi-restart idea help at all?"
  BASE002_fixed_schedule the real DEEnv episode loop with a CONSTANT action every
                         step — isolates "restarting helps" from "the agent learnt"
  BASE003_random         the real DEEnv episode loop sampling a random action every
                         step — the floor the agent must clear
"""

import argparse
import csv
import os

import numpy as np
import yaml

from derl2 import run_metadata
from derl2.benchmarks import build_benchmark
from derl2.config import Config
from derl2.environments import DEEnv
from derl2.evaluation.evaluate import (baseline_dir, fixed_policy,
                                       random_policy, run_episode)
from derl2.optimizers import run_segment

DE_PLAIN = "BASE001_de_plain"
FIXED_SCHEDULE = "BASE002_fixed_schedule"
RANDOM = "BASE003_random"
KNOWN = (DE_PLAIN, FIXED_SCHEDULE, RANDOM)

RESULT_COLUMNS = ["seed", "function_id", "best_error", "evals_used"]


def seed_list(cfg):
    """The agent's evaluation seed list, derived from the config exactly as
    evaluate.py derives it (offset + i for i in range(runs))."""
    offset = cfg.get("evaluation.seed_offset")
    runs = cfg.get("evaluation.runs_per_function")
    return [offset + i for i in range(runs)]


def segment_seed(episode_seed):
    """Derive a DE segment seed from an episode seed the same way DEEnv derives
    its warm-up seed, so BASE001 is reproducible from the episode seed."""
    return int(np.random.default_rng(episode_seed).integers(0, 2**31 - 1))


def run_de_plain(cfg, functions, seeds, strategy, F, CR):
    """BASE001: one DE segment over the WHOLE budget, sampled from the full
    domain, no restarts. Returns (rows, extra_config)."""
    suite = cfg.get("benchmark.name")
    dim = cfg.get("benchmark.dim")
    budget = int(cfg.get("benchmark.budget"))
    pop_size = cfg.get("episode.pop_size")
    n_checkpoints = cfg.get("environment.n_checkpoints")

    rows = []
    for fid in functions:
        spec = build_benchmark(f"{suite}:f{fid}", dim)
        domain_lo = np.full(dim, spec.lower, dtype=np.float32)
        domain_hi = np.full(dim, spec.upper, dtype=np.float32)
        for s in seeds:
            seg = run_segment(
                spec.evaluate_tf, dim, pop_size,
                domain_lo, domain_hi, domain_lo, domain_hi,   # box == full domain
                strategy, F, CR, budget, n_checkpoints,
                seed=segment_seed(s),
            )
            best_error = max(0.0, float(seg.best_fitness) - spec.optimum)
            rows.append((s, fid, best_error, int(seg.fes_used)))
            print(f"  {DE_PLAIN} f{fid} seed{s}: "
                  f"error {best_error:.4e} in {seg.fes_used} FEs", flush=True)
    extra = {"strategy": strategy, "F": F, "CR": CR,
             "budget_frac": None, "sampling_box": None}
    return rows, extra


def encode_action(action_space, strategy, budget_frac, sampling_box):
    """Raw action index in profiles_budget_area for a constant policy."""
    profiles = [s for (s, _F, _CR) in action_space.profiles]
    if strategy not in profiles:
        raise ValueError(
            f"strategy {strategy!r} is not a configured profile {profiles}.")
    s_idx = profiles.index(strategy)
    fracs = action_space.budget_fracs
    matches = [i for i, fr in enumerate(fracs) if abs(fr - budget_frac) < 1e-9]
    if not matches:
        raise ValueError(
            f"budget_frac {budget_frac} is not in configured budget_fracs "
            f"{fracs}.")
    b_idx = matches[0]
    n_boxes = action_space.N_BOXES
    if not 0 <= sampling_box < n_boxes:
        raise ValueError(f"sampling_box {sampling_box} out of range [0,{n_boxes}).")
    return s_idx * (len(fracs) * n_boxes) + b_idx * n_boxes + sampling_box


def run_env_policy(cfg, functions, seeds, make_policy):
    """Shared driver for BASE002/BASE003: run the real DEEnv episode loop with a
    per-episode policy factory make_policy(env, seed) -> policy."""
    rows = []
    env = DEEnv.from_config(cfg)
    for fid in functions:
        for s in seeds:
            policy = make_policy(env, s)
            ep, _infos, _trajs = run_episode(env, s, fid, policy)
            rows.append((s, fid, float(ep["final_error"]), int(ep["total_FEs"])))
            print(f"  f{fid} seed{s}: error {ep['final_error']:.4e} in "
                  f"{ep['total_FEs']} FEs", flush=True)
    return rows


def run_fixed_schedule(cfg, functions, seeds, strategy, budget_frac,
                       sampling_box):
    """BASE002: constant action every step."""
    env = DEEnv.from_config(cfg)
    raw = encode_action(env.action, strategy, budget_frac, sampling_box)
    decoded = env.action.decode(raw)          # the F/CR the profile carries
    rows = run_env_policy(cfg, functions, seeds,
                          lambda _env, _s: fixed_policy(raw))
    extra = {"strategy": strategy, "F": decoded["F"], "CR": decoded["CR"],
             "budget_frac": budget_frac, "sampling_box": sampling_box,
             "raw_action": raw}
    return rows, extra


def run_random(cfg, functions, seeds):
    """BASE003: random action every step, sampler seeded from the episode seed."""
    rows = run_env_policy(
        cfg, functions, seeds,
        lambda env, s: random_policy(env.action, seed=s))
    extra = {"strategy": None, "F": None, "CR": None,
             "budget_frac": None, "sampling_box": None}
    return rows, extra


def write_results(out_dir, rows):
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(RESULT_COLUMNS)
        for seed, fid, best_error, evals_used in rows:
            w.writerow([seed, fid, f"{best_error:.6e}", evals_used])


def write_baseline_config(out_dir, name, cfg, functions, seeds, extra):
    payload = {
        "baseline": name,
        "experiment": cfg.exp_id,
        "smoke": cfg.is_smoke,
        # Parity fields evaluate.py checks (must match the agent evaluation):
        "functions": [int(f) for f in functions],
        "dim": int(cfg.get("benchmark.dim")),
        "budget": int(cfg.get("benchmark.budget")),
        "seeds": [int(s) for s in seeds],
        "seed_offset": int(cfg.get("evaluation.seed_offset")),
        "runs_per_function": int(cfg.get("evaluation.runs_per_function")),
        # Shared DE settings:
        "pop_size": int(cfg.get("episode.pop_size")),
        "n_checkpoints": int(cfg.get("environment.n_checkpoints")),
        # Baseline-specific constants:
        **extra,
    }
    with open(os.path.join(out_dir, "baseline_config.yaml"), "w") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, choices=KNOWN)
    parser.add_argument("--config", required=True,
                        help="Experiment config.yaml the baseline mirrors")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-scale baseline (matches a --smoke run)")
    # BASE001 single strategy (fixed F/CR); BASE002 constant action. Defaults
    # match the spec; overridable, and whatever is used is recorded.
    parser.add_argument("--strategy", default="rand/1/bin")
    parser.add_argument("--f", dest="F", type=float, default=0.5)
    parser.add_argument("--cr", dest="CR", type=float, default=0.9)
    parser.add_argument("--budget-frac", type=float, default=0.10)
    parser.add_argument("--sampling-box", type=int, default=2)
    parser.add_argument("--function", type=int, default=None,
                        help="Produce the baseline for a single function id, "
                             "into baselines/<exp>/f<id>/<name>/. Omit to produce "
                             "it for every function in the config.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a (function, baseline) whose results.csv "
                             "already exists. Lets a job generate its baselines "
                             "at start-up idempotently, so a resumed/restarted "
                             "run does not recompute them.")
    args = parser.parse_args()

    cfg = Config.from_file(args.config, smoke=args.smoke)
    env = DEEnv.from_config(cfg)               # resolves "all"/list functions
    seeds = seed_list(cfg)

    functions = ([args.function] if args.function is not None
                 else list(env.functions))

    # One frozen baseline per function, each in its own baselines/f<N>/<name>/
    # folder so a per-function experiment job finds exactly its reference.
    for fid in functions:
        # Same path helper evaluate.py reads from, so producer and consumer
        # never disagree; baselines are namespaced by experiment
        # (baselines/<exp_id>[/smoke]/f<N>/<name>/) because a baseline is only
        # valid for the config that produced it.
        out_dir = baseline_dir(args.baseline, fid, cfg.exp_id, cfg.is_smoke)
        if args.skip_existing and os.path.exists(
                os.path.join(out_dir, "results.csv")):
            print(f"Skipping {args.baseline} for f{fid}: results.csv already "
                  f"present in {out_dir}")
            continue
        os.makedirs(out_dir, exist_ok=True)

        print(f"Producing {args.baseline} for f{fid}: dim={env.dim} "
              f"budget={env.budget} seeds={seeds[0]}..{seeds[-1]} "
              f"({len(seeds)} each)" + (" [SMOKE]" if cfg.is_smoke else ""))

        run_metadata.write_start(out_dir, cfg, kind="baseline")
        try:
            if args.baseline == DE_PLAIN:
                rows, extra = run_de_plain(cfg, [fid], seeds,
                                           args.strategy, args.F, args.CR)
            elif args.baseline == FIXED_SCHEDULE:
                rows, extra = run_fixed_schedule(cfg, [fid], seeds,
                                                 args.strategy, args.budget_frac,
                                                 args.sampling_box)
            else:  # RANDOM
                rows, extra = run_random(cfg, [fid], seeds)

            write_results(out_dir, rows)
            write_baseline_config(out_dir, args.baseline, cfg, [fid], seeds,
                                  extra)
        except Exception:
            run_metadata.write_finish(out_dir, "failed")
            raise
        run_metadata.write_finish(out_dir, "completed")

        print(f"Wrote {len(rows)} rows to {out_dir}/results.csv")


if __name__ == "__main__":
    main()
