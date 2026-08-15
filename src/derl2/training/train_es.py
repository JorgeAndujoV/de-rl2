"""Sep-CMA-ES training loop (EXP026) -- additive; does NOT use train.py.

Per-function job (like EXP016): trains the policy weights on ONE benchmark
function with separable CMA-ES. Each generation:
  ask  -> a population of perturbed weight vectors,
  eval -> one full greedy episode per member (population-parallel workers),
  tell -> update the Sep-CMA-ES state from the shaped fitnesses.
Fitness is quality = -log10(final_error + ERR_EPS): higher is better, evaluated
at the full FE budget, no reward bookkeeping. Reuses DEEnv, the Hybrid-PPO
policy net + decode, and evaluate.run_episode -- none of them edited.

Resume: the Sep-CMA-ES state (mean, sigma, covariance, RNG key, generation) is
checkpointed every `es.checkpoint_every` generations. Re-running the SAME command
resumes from the last checkpoint and continues to `es.num_generations`, so a
Slurm timeout mid-run loses at most one checkpoint interval (mirrors train.py's
`while completed < episodes`).

Monitor: every `es.eval_every` generations the current distribution MEAN (the
deployable policy) is scored on a small fixed validation seed set -> eval_log.csv
and best_mean.npy. This is a learning-curve monitor only; the paired baseline
comparison (SS8's files) is produced afterwards by evaluate_es.py, which loads
mean.npy (or best_mean.npy) and runs the standard evaluate_policy.

Run:
  python -m derl2.training.train_es --exp EXP026_sepcmaes_boxnp --function 11
  python -m derl2.training.train_es --exp EXP026_sepcmaes_boxnp --function 11 \
      --smoke --set es.population_size=4 --set es.num_generations=2 \
      --set es.eval_runs=2 --workers 4
"""

import argparse
import csv
import os
import pickle
import time

import numpy as np

from derl2 import run_metadata
from derl2.config import Config
from derl2.environments.env import DEEnv
from derl2.training.es_pool import ESWorkerPool
from derl2.agents.sep_cmaes import (SepCMAES, build_policy_net, flat_size,
                                    ERR_EPS)

# Seed band for ES generation episodes: 7e5, clear of training (seed*1e6+ep) and
# of the 8e5 validation-monitor and 9e5 final-evaluation bands used below/
# elsewhere.
_ES_SEED_BASE = 700_000
# Validation-monitor seed band (mean policy learning curve); distinct from the
# per-generation training seeds (7e5) and the 9e5 final-eval seeds.
_VAL_SEED_BASE = 800_000


def resolve_function(cfg, arg_function):
    if arg_function is not None:
        return int(arg_function)
    funcs = cfg.get("benchmark.functions")
    if isinstance(funcs, (list, tuple)) and len(funcs) == 1:
        return int(funcs[0])
    raise SystemExit(
        "Sep-CMA-ES trains ONE function per job. Pass --function N "
        f"(config.benchmark.functions = {funcs}).")


def _save_checkpoint(ckpt_dir, es, best_error, next_gen):
    """Atomically persist the resume state (tmp + os.replace) so a kill mid-write
    never corrupts the checkpoint. Also drops mean.npy (float64 flat weights) for
    evaluate_es.py -- loading the deployable policy then needs no JAX."""
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {"es": es.state_dict(), "best_error": float(best_error),
               "next_gen": int(next_gen)}
    tmp = os.path.join(ckpt_dir, ".es_state.pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, os.path.join(ckpt_dir, "es_state.pkl"))
    np.save(os.path.join(ckpt_dir, "mean.npy"), es.mean())


def _load_checkpoint(ckpt_dir, es):
    """Restore es state; return (next_gen, best_error) or (0, inf) if none."""
    path = os.path.join(ckpt_dir, "es_state.pkl")
    if not os.path.exists(path):
        return 0, np.inf
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    es.load_state_dict(payload["es"])
    return int(payload["next_gen"]), float(payload["best_error"])


def _open_log(path, header):
    new = not os.path.exists(path) or os.stat(path).st_size == 0
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(header)
    return fh, w


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="path to an experiment config.yaml")
    g.add_argument("--exp", help="experiment id (experiments/<id>/config.yaml)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny scaled-down run; checks for crashes only")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set es.workers=64")
    p.add_argument("--function", type=int, default=None,
                   help="benchmark function id to train on (one per job)")
    p.add_argument("--workers", type=int, default=None,
                   help="override es.workers (population-parallel worker count)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="checkpoint and exit(42) after this many wall hours "
                        "(Slurm walltime safety); re-run to resume.")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    path = args.config or os.path.join("experiments", args.exp, "config.yaml")
    cfg = Config.from_file(path, args.set, smoke=args.smoke)
    fid = resolve_function(cfg, args.function)

    kind = "smoke" if cfg.is_smoke else "train"
    out_dir = args.out_dir or os.path.join("experiments", cfg.exp_id, kind,
                                           f"f{fid}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    # --- ES + net config ---
    pop = int(cfg.get("es.population_size"))
    gens = int(cfg.get("es.num_generations"))
    workers = int(args.workers if args.workers is not None
                  else cfg.get("es.workers"))
    fitness_seeds = int(cfg.get("es.fitness_seeds", 1))
    elite_ratio = float(cfg.get("es.elite_ratio", 0.3))
    sigma_init = float(cfg.get("es.sigma_init", 0.05))
    antithetic = bool(cfg.get("es.antithetic", True))
    checkpoint_every = int(cfg.get("es.checkpoint_every", 10))
    eval_every = int(cfg.get("es.eval_every", 20))
    eval_runs = int(cfg.get("es.eval_runs", 8))
    param_dim = int(cfg.get("agent.param_dim"))
    policy_hidden = cfg.get("agent.policy_hidden")
    seed = int(cfg.get("seed"))

    # SMOKE_OVERRIDES (config.py) scales training.*/benchmark.* but NOT es.*, so a
    # bare --smoke would otherwise run the full es.num_generations. Shrink es.* to
    # a crash-check size here -- but only for keys the user did NOT pass via --set,
    # so an explicit `--set es.num_generations=5` still wins.
    if cfg.is_smoke:
        set_keys = {s.split("=", 1)[0].strip() for s in args.set}
        if "es.population_size" not in set_keys:
            pop = 4
        if "es.num_generations" not in set_keys:
            gens = 2
        if "es.eval_runs" not in set_keys:
            eval_runs = 2
        if "es.fitness_seeds" not in set_keys:
            fitness_seeds = 2          # keep >1 so the multi-seed path is smoked
    fitness_seeds = max(1, fitness_seeds)

    # Workers beyond the population can never help (a generation has only `pop`
    # episodes); cap so a big --workers/cpus never spawns idle workers.
    workers = min(workers, pop)

    # obs_dim / n_actions come from the env; a template net gives the flat dim.
    env = DEEnv.from_config(cfg)
    obs_dim, n_actions = env.obs_dim, env.n_actions
    template = build_policy_net(obs_dim, n_actions, param_dim, policy_hidden,
                               seed=seed)
    num_dims = flat_size(template.policy_net.trainable_variables)

    print(f"{cfg.exp_id} | Sep-CMA-ES f{fid} dim={cfg.get('benchmark.dim')} "
          f"budget={cfg.get('benchmark.budget')} | obs_dim={obs_dim} "
          f"n_actions={n_actions} num_dims={num_dims}", flush=True)
    print(f"pop={pop} gens={gens} fitness_seeds={fitness_seeds} "
          f"workers={workers} elite_ratio={elite_ratio} "
          f"sigma_init={sigma_init} antithetic={antithetic} "
          f"ckpt_every={checkpoint_every} eval_every={eval_every} "
          f"eval_runs={eval_runs}"
          f"{' [SMOKE]' if cfg.is_smoke else ''} | output {out_dir}", flush=True)

    # run_metadata: written once (resumes keep the original start record) so
    # evaluate_es.py can read the config this job ACTUALLY ran, exactly as
    # evaluate.py does for PPO jobs.
    run_metadata.ensure_start(out_dir, cfg, kind)

    es = SepCMAES(num_dims, pop, elite_ratio=elite_ratio, sigma_init=sigma_init,
                  antithetic=antithetic, seed=seed)
    start_gen, best_error = _load_checkpoint(ckpt_dir, es)
    if start_gen > 0:
        print(f"resume: checkpoint found -> continuing from generation "
              f"{start_gen}/{gens} (best_so_far {best_error:.4e})", flush=True)
    if start_gen >= gens:
        print(f"Nothing to do: checkpoint already at generation {start_gen} >= "
              f"num_generations {gens}.", flush=True)
        return

    pool = ESWorkerPool(cfg, workers, obs_dim, n_actions, param_dim,
                        policy_hidden, seed=seed)

    log, writer = _open_log(
        os.path.join(out_dir, "es_log.csv"),
        ["generation", "gen_seconds", "episodes_per_sec", "best_error_gen",
         "mean_error_gen", "best_error_so_far"])
    eval_log, eval_writer = _open_log(
        os.path.join(out_dir, "eval_log.csv"),
        ["generation", "val_runs", "median_error", "mean_error", "best_error"])

    val_seeds = [_VAL_SEED_BASE + i for i in range(eval_runs)]
    best_val_median = np.inf
    max_hours = args.max_hours
    wall0 = time.perf_counter()
    status = "failed"
    try:
        for gen in range(start_gen, gens):
            # Walltime safety: checkpoint everything completed so far and exit 42
            # (the launcher's "timeout" code) rather than being SIGKILLed mid
            # generation. next_gen=gen so a resume re-runs this not-yet-done gen.
            if (max_hours is not None
                    and (time.perf_counter() - wall0) / 3600.0 >= max_hours):
                _save_checkpoint(ckpt_dir, es, best_error, next_gen=gen)
                status = "timeout"
                print(f"walltime stop at generation {gen}/{gens} "
                      f"(>= {max_hours}h); re-run to resume.", flush=True)
                raise SystemExit(42)
            t0 = time.perf_counter()
            x = es.ask()                                   # (pop, num_dims)
            # K distinct seeds this generation, shared by all members (CRN); the
            # band advances by K each generation so no instance repeats. With
            # fitness_seeds=1 this is the old single-seed path (a (pop, 1) column).
            gen_seeds = [_ES_SEED_BASE + gen * fitness_seeds + i
                         for i in range(fitness_seeds)]
            err = pool.evaluate_multiseed(x, gen_seeds, fid)   # (pop, K)
            # Fitness = MEAN log-quality across the K seeds: rewards members that
            # reach low error CONSISTENTLY across instances (a bad seed drags the
            # mean down), which is exactly what the multi-seed final eval rewards.
            quality = np.mean(-np.log10(err + ERR_EPS), axis=1)   # (pop,)
            es.tell(x, quality)

            dt = time.perf_counter() - t0
            member_err = err.mean(axis=1)                  # per-member mean error
            g_best = float(np.min(member_err))
            g_mean = float(np.mean(member_err))
            best_error = min(best_error, g_best)
            eps = (pop * fitness_seeds) / dt if dt > 0 else float("nan")
            writer.writerow([gen, f"{dt:.3f}", f"{eps:.2f}",
                             f"{g_best:.6e}", f"{g_mean:.6e}",
                             f"{best_error:.6e}"])
            log.flush()
            print(f"gen {gen:4d} | best {g_best:.4e} | mean {g_mean:.4e} "
                  f"| best_so_far {best_error:.4e} | {eps:.1f} ep/s | {dt:.1f}s",
                  flush=True)

            last = (gen == gens - 1)

            # --- periodic mean-policy monitor ---
            if eval_every > 0 and ((gen + 1) % eval_every == 0 or last):
                mean_flat = es.mean()
                verr = pool.evaluate_seeds(mean_flat, val_seeds, fid)
                v_med = float(np.median(verr))
                eval_writer.writerow([gen, len(val_seeds), f"{v_med:.6e}",
                                      f"{np.mean(verr):.6e}",
                                      f"{np.min(verr):.6e}"])
                eval_log.flush()
                print(f"  [eval] gen {gen}: mean-policy median {v_med:.4e} "
                      f"mean {np.mean(verr):.4e} best {np.min(verr):.4e} "
                      f"over {len(val_seeds)} val seeds", flush=True)
                if v_med < best_val_median:
                    best_val_median = v_med
                    os.makedirs(ckpt_dir, exist_ok=True)
                    np.save(os.path.join(ckpt_dir, "best_mean.npy"), mean_flat)

            # --- resume checkpoint ---
            if (gen + 1) % checkpoint_every == 0 or last:
                _save_checkpoint(ckpt_dir, es, best_error, next_gen=gen + 1)
        status = "completed"
    finally:
        pool.close()
        log.close()
        eval_log.close()
        run_metadata.write_finish(out_dir, status)

    print(f"Done. Sep-CMA-ES f{fid}: best final-error {best_error:.6e} over "
          f"{gens} generations. Checkpoints/logs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
