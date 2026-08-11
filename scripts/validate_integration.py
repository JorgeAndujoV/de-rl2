#!/usr/bin/env python3
"""Validate that the FRAMEWORK uses the (already-validated) CEC'13 benchmark
correctly -- a different question from whether the functions are mathematically
correct in isolation (that is scripts/validate_benchmarks.py).

Even with correct functions, the framework could still: subtract the wrong
optimum, feed the DE the wrong objective, mis-track the best-so-far, or minimise
the wrong sign. These checks exercise the real objects the training/eval pipeline
uses (run_segment, DEEnv) and assert the numbers line up.

Checks:
  1. OPTIMUM VALUE   for each function, evaluate_tf(x*) - spec.optimum ~= 0, i.e.
                     the value the environment subtracts really is the minimum.
  2. DE MINIMISES    plain DE (the training engine) drives a known-easy function
                     (Sphere) to ~0 -- confirms the DE optimises this objective.
  3. ENV ROUND-TRIP  after a full DEEnv episode, the reported best_error equals an
                     INDEPENDENT re-evaluation of the env's returned best solution,
                     max(0, evaluate_tf(x_best) - optimum). This isolates the
                     environment's optimum-subtraction and global-best tracking.
  4. ENV OBJECTIVE   the env feeds the DE spec.evaluate_tf (the float32 path this
                     project validated), not some other objective.

Run on WSL (needs the venv):  ./.venv/bin/python scripts/validate_integration.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
os.environ.setdefault("CEC13_DATA", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cec13", "input_data"))

import tensorflow as tf                                          # noqa: E402
from derl2.benchmarks.registry import build_benchmark            # noqa: E402
from derl2.benchmarks.cec13_data import shift_for                # noqa: E402
from derl2.optimizers import run_segment                         # noqa: E402
from derl2.environments.env import DEEnv                         # noqa: E402

DIM = 30
FUNCS = [1, 2, 6, 11, 17, 21]        # spread: unimodal, osz, multimodal, composition
RTOL = 1e-4                          # float32 round-off allowance
results = []


def record(name, passed, detail):
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def _tf_eval(spec, x):
    return float(spec.evaluate_tf(tf.constant(np.asarray(x, np.float32)[None, :]))
                 .numpy()[0])


# ---------------------------------------------------------- check 1: optimum
def check_optimum():
    print("\n1. OPTIMUM VALUE  (evaluate_tf(x*) - spec.optimum ~= 0)")
    ok = True
    for n in FUNCS:
        spec = build_benchmark(f"cec13:f{n}", DIM)
        xstar = shift_for(n, DIM)                    # block-0 shift = x*
        err = _tf_eval(spec, xstar) - spec.optimum
        good = abs(err) <= 1e-2                       # float32 near the bias ~1e3
        ok = ok and good
        record(f"f{n} optimum", good,
               f"error at x* = {err:+.3e} (optimum={spec.optimum:.1f})")
    return ok


# --------------------------------------------------- check 2: DE minimises
def check_de_minimises():
    print("\n2. DE MINIMISES  (plain DE drives Sphere to ~0)")
    spec = build_benchmark("cec13:f1", DIM)
    lo = np.full(DIM, spec.lower); hi = np.full(DIM, spec.upper)
    seg = run_segment(spec.evaluate_tf, DIM, 50, lo, hi, lo, hi,
                      "rand/1/bin", 0.5, 0.9, 100000, 20, seed=1)
    err = max(0.0, float(seg.best_fitness) - spec.optimum)
    good = err < 1.0                                  # Sphere always converges well
    record("f1 plain DE converges", good, f"final error = {err:.3e} (expect << 1)")
    return good


# ------------------------------------------------ check 3+4: env round-trip
def _make_env():
    """A minimal DEEnv built directly (no config file) with a small budget for
    speed. Uses the discrete profiles_budget_area space so a fixed action is a
    plain integer index."""
    return DEEnv(
        suite="cec13", functions=FUNCS, dim=DIM, budget=30000, pop_size=50,
        warmup_frac=0.05,
        warmup={"strategy": "rand/1/bin", "F": 0.5, "CR": 0.9},
        box_center="centroid", box_scales=[0.25, 0.5, 1.0, 2.0, 3.0],
        box_min_frac=0.0417, elitism=False, truncate_last_segment=True,
        observation="traj20", n_checkpoints=20,
        action_space="profiles_budget_area", budget_fracs=[0.1, 0.25, 0.5],
        strategy_profiles=[{"strategy": "rand/1/bin", "F": 0.5, "CR": 0.9},
                           {"strategy": "best/1/bin", "F": 0.5, "CR": 0.9}],
        reward={"name": "log_stagnation", "lambda": 0.1, "tau_stag": 3},
    )


def check_env_roundtrip():
    print("\n3. ENV ROUND-TRIP  (reported best_error == re-eval of env's best x)")
    print("4. ENV OBJECTIVE   (env feeds the DE spec.evaluate_tf)")
    env = _make_env()
    action = 0                                        # fixed policy: action index 0
    ok_rt = ok_obj = True
    for n in FUNCS:
        obs, info = env.reset(seed=123, function_id=n)
        # check 4: the objective the env handed the DE is evaluate_tf of this spec
        spec = env._spec(n)
        if env._objective is not spec.evaluate_tf:
            ok_obj = False
            record(f"f{n} objective", False, "env._objective is NOT spec.evaluate_tf")
        done = False
        while not done:
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
        reported = float(info["best_error"])
        x_best = env._global_best_solution
        recomputed = max(0.0, _tf_eval(spec, x_best) - spec.optimum)
        # A real integration bug (wrong optimum / mis-tracked best) offsets by >=1;
        # float32 re-eval noise is ~1e-4. atol 1e-2 separates them cleanly.
        good = abs(reported - recomputed) <= 1e-2 + RTOL * abs(recomputed)
        ok_rt = ok_rt and good
        record(f"f{n} round-trip", good,
               f"reported={reported:.6e}  re-eval={recomputed:.6e}")
    if ok_obj:
        record("env objective", True, "env feeds spec.evaluate_tf on all functions")
    return ok_rt and ok_obj


def main():
    print("=" * 64)
    print("Framework <-> benchmark integration validation")
    print("=" * 64)
    all_ok = True
    for fn in (check_optimum, check_de_minimises, check_env_roundtrip):
        try:
            all_ok = bool(fn()) and all_ok
        except Exception as e:
            all_ok = False
            record(fn.__name__, False, f"raised {type(e).__name__}: {e}")
    n_pass = sum(1 for _, p, _ in results if p)
    print("\n" + "=" * 64)
    print(f"{n_pass}/{len(results)} checks passed.")
    print("INTEGRATION OK: the framework uses the validated benchmark correctly."
          if all_ok else "INTEGRATION FAILURES above -- investigate.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
