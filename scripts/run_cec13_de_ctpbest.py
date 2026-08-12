"""Plain vectorized DE/current-to-pbest/1/bin on CEC'13 at 30D.

Sibling of run_cec13_de_comparison.py, with a different DE protocol:

    strategy = current-to-pbest/1/bin   (JADE mutation, archive-free)
    NP = 100   F = 0.5   CR = 0.5   p_best = 0.11   D = 30

Runs it as a SINGLE full-budget run (no restarts, no agent) on every function,
`--runs` times (51 by design), using the exact vectorized (float32) engine the
environment feeds the DE during training (derl2.optimizers.run_segment driven by
spec.evaluate_tf). Reports the error statistics per function: median, mean, std,
best, worst — the same two files as the plain-DE comparison.

p_best note: run_segment takes no `p` argument; DE reads it from the strategy
class attribute (CurrentToPbest1.p_best) at tf.function trace time, where it
becomes pbest_size = round(p * NP). Setting the class attribute ONCE before the
first run (as main() does) bakes p=0.11 -> top 11 of NP=100 into every trace.

Usage:
    python scripts/run_cec13_de_ctpbest.py                    # all 28 fns, 51 runs
    python scripts/run_cec13_de_ctpbest.py --functions f2     # one function
    python scripts/run_cec13_de_ctpbest.py --runs 5           # cheaper spot-check
    python scripts/run_cec13_de_ctpbest.py --merge            # assemble _parts/*

On Nibi this is run one-function-per-array-task into runs/<name>/_parts/f<N>/,
then `--merge` concatenates the parts into runs/<name>/{summary,per_run}.csv.
See scripts/submit_cec13_ctpbest.sh.
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

# DE/current-to-pbest/1/bin protocol (overridable on the CLI; whatever is used
# is recorded in the output CSVs).
NP, F, CR, P_BEST = 100, 0.5, 0.5, 0.11
STRATEGY = "current-to-pbest/1/bin"
ALL_FUNCTIONS = [f"f{i}" for i in range(1, 29)]   # f1 .. f28


def run_de(spec, dim, seed, max_fes, np_, f, cr, strategy):
    """One full-budget DE run over the whole domain, using the SAME vectorized
    (float32) engine the agent's environment drives. Returns the final error
    (best fitness - optimum), floored at 0 to absorb float round-off."""
    from derl2.optimizers import run_segment

    lo = np.full(dim, spec.lower, dtype=np.float64)
    hi = np.full(dim, spec.upper, dtype=np.float64)
    seg = run_segment(
        spec.evaluate_tf, dim, np_,
        box_lo=lo, box_hi=hi,            # initial population over the full domain
        domain_lo=lo, domain_hi=hi,
        strategy=strategy, F=f, CR=cr,
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


def merge_parts(out_dir):
    """Concatenate runs/<name>/_parts/f<N>/{summary,per_run}.csv (one per array
    task) into runs/<name>/{summary,per_run}.csv, ordered by function number."""
    parts = sorted(glob.glob(os.path.join(out_dir, "_parts", "f*")),
                   key=lambda d: int(os.path.basename(d)[1:]))
    if not parts:
        raise SystemExit(f"no _parts/f* under {out_dir}; nothing to merge.")
    summ = [pd.read_csv(os.path.join(p, "summary.csv")) for p in parts
            if os.path.exists(os.path.join(p, "summary.csv"))]
    runs = [pd.read_csv(os.path.join(p, "per_run.csv")) for p in parts
            if os.path.exists(os.path.join(p, "per_run.csv"))]
    summary = pd.concat(summ, ignore_index=True)
    per_run = pd.concat(runs, ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    per_run.to_csv(os.path.join(out_dir, "per_run.csv"), index=False)
    print(f"Merged {len(parts)} parts -> {out_dir}/summary.csv "
          f"({len(summary)} fns) and per_run.csv ({len(per_run)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dim", type=int, default=30)
    p.add_argument("--runs", type=int, default=51)
    p.add_argument("--functions", nargs="+", default=ALL_FUNCTIONS)
    p.add_argument("--np", dest="np_", type=int, default=NP)
    p.add_argument("--f", dest="F", type=float, default=F)
    p.add_argument("--cr", dest="CR", type=float, default=CR)
    p.add_argument("--p", dest="p_best", type=float, default=P_BEST,
                   help="pbest fraction; DE uses top round(p*NP) as donors")
    p.add_argument("--strategy", default=STRATEGY)
    p.add_argument("--name", default="cec13_de_ctpbest_d30")
    p.add_argument("--out-subdir", default="",
                   help="write into runs/<name>/<out-subdir>/ instead of "
                        "runs/<name>/ (used for one-function-per-task parts)")
    p.add_argument("--merge", action="store_true",
                   help="assemble runs/<name>/_parts/* into the final files "
                        "and exit (no DE runs)")
    args = p.parse_args()

    out_dir = os.path.join("runs", args.name)

    if args.merge:
        merge_parts(out_dir)
        return

    if args.out_subdir:
        out_dir = os.path.join(out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Set p_best BEFORE any run_segment call: DE reads it at tf.function trace
    # time (pbest_size = round(p_best * NP)), so it must be in place before the
    # first trace. Only meaningful for a *pbest strategy; harmless otherwise.
    from derl2.benchmarks.registry import build_benchmark
    from derl2.optimizers.strategies import CurrentToPbest1
    CurrentToPbest1.p_best = args.p_best

    max_fes = 10000 * args.dim            # CEC'13 budget = 1e4 * D
    per_run_rows, summary_rows = [], []
    print(f"Plain vectorized DE/{args.strategy}  NP={args.np_} F={args.F} "
          f"CR={args.CR} p_best={args.p_best}  D={args.dim}  runs={args.runs}  "
          f"max_fes={max_fes}\n")
    print(f"{'fn':>4} {'name':30s} {'median':>11} {'mean':>11} "
          f"{'std':>11} {'best':>11} {'worst':>11}")

    for fname in args.functions:
        spec = build_benchmark(f"cec13:{fname}", args.dim)
        errs = []
        t0 = time.time()
        for r in range(args.runs):
            seed = 700000 + 1000 * int(fname[1:]) + r
            err = run_de(spec, args.dim, seed, max_fes,
                         args.np_, args.F, args.CR, args.strategy)
            errs.append(err)
            per_run_rows.append({"benchmark": spec.id, "function_name": spec.name,
                                 "dim": args.dim, "seed": seed, "error": err})
        s = summarize(errs)
        summary_rows.append({
            "benchmark": spec.id, "function": fname, "function_name": spec.name,
            "dim": args.dim, "pop_size": args.np_, "F": args.F, "CR": args.CR,
            "p_best": args.p_best, "strategy": args.strategy,
            "runs": args.runs, "max_fes": max_fes,
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
