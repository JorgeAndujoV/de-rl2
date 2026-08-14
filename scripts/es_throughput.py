"""Sep-CMA-ES throughput benchmark (EXP026 Stage 3) -- standalone, additive.

Sizes the full run BEFORE committing a day-long job: measures episodes/sec at a
range of worker counts using the real EXP026 building blocks (SepCMAES +
ESWorkerPool + full-scale DEEnv), then projects the wall-clock of the full run
(population_size x num_generations episodes). Writes nothing into experiment
dirs; prints a table and, with --out, a CSV.

It runs the REAL config scale by default (dim 30, budget 300k) so a projection
is meaningful -- that is why the validation Slurm job runs it on the cluster, not
laptop. --smoke shrinks it (dim 5, budget 3000) only to check the script itself.

    # local code check (tiny):
    python -m scripts.es_throughput --exp EXP026_sepcmaes_boxnp --function 11 \
        --smoke --workers 2,4 --gens 2

    # cluster sizing (a validation job runs this on 64 cores):
    python -m scripts.es_throughput --exp EXP026_sepcmaes_boxnp --function 11 \
        --workers 8,16,32,64 --gens 3 --out throughput.csv
"""

import argparse
import csv
import os
import time

import numpy as np

from derl2.config import Config
from derl2.environments.env import DEEnv
from derl2.training.es_pool import ESWorkerPool
from derl2.agents.sep_cmaes import SepCMAES, build_policy_net, flat_size

_BENCH_SEED_BASE = 600_000          # distinct from train (7e5) / val (8e5) bands


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config")
    g.add_argument("--exp")
    p.add_argument("--function", type=int, required=True)
    p.add_argument("--workers", default="8,16,32,64",
                   help="comma-separated worker counts to benchmark")
    p.add_argument("--gens", type=int, default=3,
                   help="generations to time per worker count (>=2 to warm up)")
    p.add_argument("--pop", type=int, default=None,
                   help="population size (default: config es.population_size)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", default=None, help="optional CSV path for the table")
    args = p.parse_args()

    path = args.config or os.path.join("experiments", args.exp, "config.yaml")
    cfg = Config.from_file(path, args.set, smoke=args.smoke)
    fid = int(args.function)

    pop = int(args.pop if args.pop is not None
              else cfg.get("es.population_size"))
    gens_full = int(cfg.get("es.num_generations"))
    param_dim = int(cfg.get("agent.param_dim"))
    policy_hidden = cfg.get("agent.policy_hidden")
    seed = int(cfg.get("seed"))
    worker_counts = [int(w) for w in str(args.workers).split(",") if w.strip()]

    env = DEEnv.from_config(cfg)
    obs_dim, n_actions = env.obs_dim, env.n_actions
    template = build_policy_net(obs_dim, n_actions, param_dim, policy_hidden,
                               seed=seed)
    num_dims = flat_size(template.policy_net.trainable_variables)

    print(f"{cfg.exp_id} throughput | f{fid} dim={cfg.get('benchmark.dim')} "
          f"budget={cfg.get('benchmark.budget')} | pop={pop} num_dims={num_dims} "
          f"| timing {args.gens} gens/point"
          f"{' [SMOKE]' if cfg.is_smoke else ''}", flush=True)
    print(f"full run = pop {pop} x gens {gens_full} = {pop * gens_full} episodes")
    print(f"{'workers':>8} {'gen_s(mean)':>12} {'ep/s':>8} "
          f"{'full_run_h':>11}", flush=True)

    es = SepCMAES(num_dims, pop, elite_ratio=float(cfg.get("es.elite_ratio", 0.3)),
                  sigma_init=float(cfg.get("es.sigma_init", 0.05)),
                  antithetic=bool(cfg.get("es.antithetic", True)), seed=seed)

    rows = []
    for nw in worker_counts:
        pool = ESWorkerPool(cfg, nw, obs_dim, n_actions, param_dim,
                            policy_hidden, seed=seed)
        try:
            gen_times = []
            for gi in range(args.gens):
                x = es.ask()
                t0 = time.perf_counter()
                _ = pool.evaluate(x, _BENCH_SEED_BASE + gi, fid)
                gen_times.append(time.perf_counter() - t0)
                # feed a neutral fitness so ask() keeps producing valid pops;
                # values are irrelevant to timing.
                es.tell(x, np.zeros(pop))
        finally:
            pool.close()
        # Drop the first generation (worker warm-up / lazy TF graph build).
        timed = gen_times[1:] if len(gen_times) > 1 else gen_times
        gen_s = float(np.mean(timed))
        ep_s = pop / gen_s if gen_s > 0 else float("nan")
        full_h = (pop * gens_full / ep_s) / 3600.0 if ep_s > 0 else float("nan")
        rows.append({"workers": nw, "gen_seconds_mean": gen_s,
                     "episodes_per_sec": ep_s, "full_run_hours": full_h})
        print(f"{nw:>8} {gen_s:>12.2f} {ep_s:>8.2f} {full_h:>11.2f}", flush=True)

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["workers", "gen_seconds_mean",
                                               "episodes_per_sec",
                                               "full_run_hours"])
            w.writeheader()
            for r in rows:
                w.writerow({"workers": r["workers"],
                            "gen_seconds_mean": f"{r['gen_seconds_mean']:.4f}",
                            "episodes_per_sec": f"{r['episodes_per_sec']:.4f}",
                            "full_run_hours": f"{r['full_run_hours']:.4f}"})
        print(f"wrote {args.out}", flush=True)

    best = min(rows, key=lambda r: r["full_run_hours"])
    print(f"\nfastest: {best['workers']} workers -> "
          f"{best['episodes_per_sec']:.2f} ep/s, "
          f"~{best['full_run_hours']:.2f} h for the full run.", flush=True)
    print("Scaling note: workers beyond population_size cannot help (a "
          "generation has only `pop` episodes); near-linear speedup holds only "
          "up to workers ~= pop. Size cpus accordingly.", flush=True)


if __name__ == "__main__":
    main()
