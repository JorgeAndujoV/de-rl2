"""Plain vectorized DE on CEC'13 at 30D — the exact DE engine training uses.

Runs canonical DE/rand/1/bin (NP=50, F=0.5, CR=0.9) as a SINGLE full-budget run
(no restarts, no agent) on every function, `--runs` times (51 by design), and
reports the error statistics per function: median, mean, std, best, worst.

"Vectorized DE (the one training uses)" = derl2.optimizers.run_segment driven by
spec.evaluate_tf (float32), the same objective the environment feeds the DE
during training (environments/env.py: self._objective = spec.evaluate_tf).

No literature references, no comparison columns — just the raw statistics.

Usage:
    ./.venv/bin/python scripts/run_cec13_de_comparison.py                 # all 28 fns, 51 runs
    ./.venv/bin/python scripts/run_cec13_de_comparison.py --functions f1 f11
    ./.venv/bin/python scripts/run_cec13_de_comparison.py --runs 10       # cheaper spot-check

Note: this is ~6000 DE generations x runs x functions of TF work; the full
28x51 run takes a while. Subset with --functions / --runs if needed.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from derl2.benchmarks.registry import build_benchmark
from derl2.optimizers import run_segment

# Canonical DE/rand/1/bin protocol.
NP, CR, F = 50, 0.9, 0.5
STRATEGY = "rand/1/bin"
ALL_FUNCTIONS = [f"f{i}" for i in range(1, 29)]   # f1 .. f28


def run_plain_de(spec, dim, seed, max_fes):
    """One full-budget DE run over the whole domain, using the SAME vectorized
    (float32) engine the agent's environment drives. Returns the final error
    (best fitness - optimum), floored at 0 to absorb float round-off."""
    lo = np.full(dim, spec.lower, dtype=np.float64)
    hi = np.full(dim, spec.upper, dtype=np.float64)
    seg = run_segment(
        spec.evaluate_tf, dim, NP,
        box_lo=lo, box_hi=hi,            # initial population over the full domain
        domain_lo=lo, domain_hi=hi,
        strategy=STRATEGY, F=F, CR=CR,
        fe_budget=max_fes, n_checkpoints=20, seed=seed,
    )
    return max(0.0, float(seg.best_fitness) - float(spec.optimum))


def summarize(errs):
    e = np.asarray(errs, dtype=np.float64)
    return {
        "median": float(np.median(e)),
        "mean": float(e.mean()),
        "std": float(e.std(ddof=1)) if e.size > 1 else 0.0,
        "best": float(e.min()),
        "worst": float(e.max()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dim", type=int, default=30)
    p.add_argument("--runs", type=int, default=51)
    p.add_argument("--functions", nargs="+", default=ALL_FUNCTIONS)
    p.add_argument("--name", default="cec13_de_plain_d30")
    args = p.parse_args()

    max_fes = 10000 * args.dim            # CEC'13 budget = 1e4 * D
    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)

    per_run_rows, summary_rows = [], []
    print(f"Plain vectorized DE/rand/1/bin  NP={NP} F={F} CR={CR}  "
          f"D={args.dim}  runs={args.runs}  max_fes={max_fes}\n")
    print(f"{'fn':>4} {'name':30s} {'median':>11} {'mean':>11} "
          f"{'std':>11} {'best':>11} {'worst':>11}")

    for fname in args.functions:
        spec = build_benchmark(f"cec13:{fname}", args.dim)
        errs = []
        t0 = time.time()
        for r in range(args.runs):
            seed = 700000 + 1000 * int(fname[1:]) + r
            err = run_plain_de(spec, args.dim, seed, max_fes)
            errs.append(err)
            per_run_rows.append({"benchmark": spec.id, "function_name": spec.name,
                                 "dim": args.dim, "seed": seed, "error": err})
        s = summarize(errs)
        summary_rows.append({
            "benchmark": spec.id, "function": fname, "function_name": spec.name,
            "dim": args.dim, "pop_size": NP, "F": F, "CR": CR,
            "strategy": STRATEGY, "runs": args.runs, "max_fes": max_fes,
            **s, "seconds": round(time.time() - t0, 1),
        })
        print(f"{fname:>4} {spec.name:30.30s} {s['median']:11.4e} "
              f"{s['mean']:11.4e} {s['std']:11.4e} {s['best']:11.4e} "
              f"{s['worst']:11.4e}", flush=True)

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary.csv"),
                                      index=False)
    pd.DataFrame(per_run_rows).to_csv(os.path.join(out_dir, "per_run.csv"),
                                      index=False)
    print(f"\nWrote {out_dir}/summary.csv and per_run.csv")


if __name__ == "__main__":
    main()
