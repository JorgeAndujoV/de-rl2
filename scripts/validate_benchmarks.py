#!/usr/bin/env python3
"""Rigorous 3-way validation of the CEC'13 benchmark implementations.

Reference (ground truth):     the official C code  (cec13-c-code/, via test_func).
Candidate 1 (published):      the Python port      (cec13 python benchmark/).
Candidate 2 (ours):           src/derl2/benchmarks/cec13.py, in TWO precisions
                                - TF  : evaluate_tf  (float32, the vectorized path
                                        the DE actually runs);
                                - NP  : evaluate_np  (float64, the reference-quality
                                        math), reported alongside so a float32
                                        rounding gap is never confused with a bug.

The C code is authoritative. Agreement between the port and ours must NOT be read
as correctness if both disagree with C.

Every implementation is evaluated on the SAME deterministic test points (float64)
for every function x dimension. See generate_points() for the point categories.

Reuses the C build/driver and port loader from scripts/validate_vs_c.py; does NOT
modify any benchmark math.

Run (WSL, needs g++ and the venv):
    ./.venv/bin/python scripts/validate_benchmarks.py --smoke      # quick check
    ./.venv/bin/python scripts/validate_benchmarks.py              # full run
See --help for all options.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import shutil

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("CEC13_DATA",
                      os.path.join(REPO, "data", "cec13", "input_data"))

import validate_vs_c as vc                    # C build/driver + paths (reused)
from derl2.benchmarks.cec13 import _build     # raw spec: evaluate_np / evaluate_tf
import tensorflow as tf                        # noqa: E402

DIMS_ALL = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
FUNCS_ALL = list(range(1, 29))
LOWER, UPPER = -100.0, 100.0                   # domain, all functions
_NEAR = (1e-6, 1e-4, 1e-2, 1e-1)               # perturbation / boundary fractions


# --------------------------------------------------------------- test points
def generate_points(dim, o, n_random, seed):
    """Deterministic test set for one dimension. Returns (X (M,D) float64,
    categories, point_ids). `o` is the official optimum vector x* (block-0
    shift). All points are clipped into the legal domain."""
    lo, hi, w = LOWER, UPPER, UPPER - LOWER
    rng = np.random.default_rng(seed)
    idx = np.arange(dim)
    X, cats = [], []

    def add(x, cat):
        X.append(np.clip(np.asarray(x, np.float64), lo, hi))
        cats.append(cat)

    add(o, "optimum")                                          # 1
    add(np.zeros(dim), "zero")                                 # 2
    add(np.full(dim, 0.5 * (lo + hi)), "center")
    add(np.full(dim, lo), "lower_bound")                       # 3
    add(np.full(dim, hi), "upper_bound")
    add(np.where(idx % 2 == 0, lo, hi), "mixed_bounds")
    add(np.where(idx % 2 == 0, hi, lo), "mixed_bounds")
    for f in _NEAR[:3]:                                        # 4
        add(np.full(dim, lo + f * w), "near_lower_bound")
        add(np.full(dim, hi - f * w), "near_upper_bound")
    unit, alt = np.ones(dim), np.where(idx % 2 == 0, 1.0, -1.0)
    for f in _NEAR:                                            # 5 near-optimum
        d = f * w
        add(o + d * unit, "near_optimum")
        add(o - d * unit, "near_optimum")
        add(o + d * alt, "near_optimum")
        for _ in range(2):
            add(o + d * rng.standard_normal(dim), "near_optimum")
    add(np.full(dim, lo + 0.25 * w), "structured")            # 6
    add(np.full(dim, lo + 0.75 * w), "structured")
    add(np.where(idx % 2 == 0, lo + 0.25 * w, lo + 0.75 * w), "structured")
    add(np.linspace(lo, hi, dim), "structured")
    add(np.linspace(hi, lo, dim), "structured")
    for c in (-50.0, 0.0, 50.0):
        add(np.full(dim, c), "structured")
    for _ in range(n_random):                                 # 7
        add(rng.uniform(lo, hi, dim), "random")

    ids = [f"d{dim}_{i:06d}_{cats[i]}" for i in range(len(X))]
    return np.array(X, dtype=np.float64), cats, ids


# ---------------------------------------------------------------- evaluators
def eval_c(binary, X, dim):
    """Official C values, shape (28, M). NaN where the C emitted nan/inf."""
    pts = os.path.join(vc.CDIR, "_vb_points.txt")
    with open(pts, "w") as fh:
        fh.write(f"{len(X)} {dim}\n")
        for row in X:
            fh.write(" ".join(f"{v:.17g}" for v in row) + "\n")
    try:
        out = subprocess.run([binary, "_vb_points.txt"], cwd=vc.CDIR,
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.remove(pts)
    vals = np.full((28, len(X)), np.nan)
    for line in out.strip().splitlines():
        fn, pi, v = line.split()
        try:
            vals[int(fn) - 1, int(pi)] = float(v)
        except ValueError:
            pass                                   # "nan"/"inf" -> stays nan
    return vals


def load_port(dim):
    """Fresh port instance + a snapshot of its shift/rotation state (the
    composition functions mutate it, so we restore before each eval)."""
    import importlib.util
    cwd = os.getcwd()
    os.chdir(vc.PORT_DIR)
    try:
        spec = importlib.util.spec_from_file_location(
            "cecport", os.path.join(vc.PORT_DIR, "functions.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        inst = m.CEC_functions(dim)
    finally:
        os.chdir(cwd)
    return inst, (inst.O.copy(), inst.M1.copy(), inst.M2.copy())


def eval_port(dim, X):
    inst, (O0, M10, M20) = load_port(dim)
    vals = np.full((28, len(X)), np.nan)
    for pi, x in enumerate(X):
        xf = np.asarray(x, np.float64)
        for fn in range(1, 29):
            # The port's composition functions mutate self.O/M1/M2 in place and
            # leave them corrupted, so we MUST restore the pristine block-0 state
            # before EVERY evaluation (not only fn>=21) -- otherwise f1-f20 on the
            # next point would silently use the wrong shift/rotation.
            inst.O, inst.M1, inst.M2 = O0.copy(), M10.copy(), M20.copy()
            try:
                with np.errstate(all="ignore"):
                    vals[fn - 1, pi] = float(inst.Y(xf, fn))
            except Exception:
                vals[fn - 1, pi] = np.nan
    return vals


def eval_ours(dim, X, errors):
    """Our values in both precisions: (tf float32, np float64), each (28, M).
    `errors` collects (fn, message) for any function that raised."""
    tf_vals = np.full((28, len(X)), np.nan)
    np_vals = np.full((28, len(X)), np.nan)
    X32 = tf.constant(X, tf.float32)
    for fn in range(1, 29):
        try:
            spec = _build(f"f{fn}", dim)
        except Exception as e:                      # unsupported dim, etc.
            errors.append((fn, dim, f"build: {e}"))
            continue
        try:
            tf_vals[fn - 1] = np.asarray(spec.evaluate_tf(X32).numpy(),
                                         np.float64)
        except Exception as e:
            errors.append((fn, dim, f"evaluate_tf: {e}"))
        try:
            for pi, x in enumerate(X):
                np_vals[fn - 1, pi] = float(spec.evaluate_np(x))
        except Exception as e:
            errors.append((fn, dim, f"evaluate_np: {e}"))
    return tf_vals, np_vals


# ----------------------------------------------------------------- metrics
def compare(cand, ref, atol, rtol, eps):
    """Per-point abs/rel error and pass mask vs the reference `ref`.
    A point where either value is non-finite fails and is excluded from the
    error percentiles (tracked separately as non-finite)."""
    cand, ref = np.asarray(cand, float), np.asarray(ref, float)
    both = np.isfinite(cand) & np.isfinite(ref)
    abs_err = np.where(both, np.abs(cand - ref), np.inf)
    rel_err = np.where(both, np.abs(cand - ref) / np.maximum(np.abs(ref), eps),
                       np.inf)
    passed = both & (np.abs(cand - ref) <= atol + rtol * np.abs(ref))
    return abs_err, rel_err, passed, both


def _pct(a, q):
    return float(np.percentile(a, q)) if a.size else float("nan")


def summarize(abs_err, rel_err, passed, both):
    fin = both
    ae, re = abs_err[fin], rel_err[fin]
    n = len(passed)
    return {
        "mean_abs_error": float(ae.mean()) if ae.size else float("nan"),
        "median_abs_error": float(np.median(ae)) if ae.size else float("nan"),
        "std_abs_error": float(ae.std()) if ae.size else float("nan"),
        "max_abs_error": float(ae.max()) if ae.size else float("nan"),
        "p95_abs_error": _pct(ae, 95), "p99_abs_error": _pct(ae, 99),
        "mean_rel_error": float(re.mean()) if re.size else float("nan"),
        "median_rel_error": float(np.median(re)) if re.size else float("nan"),
        "max_rel_error": float(re.max()) if re.size else float("nan"),
        "p95_rel_error": _pct(re, 95), "p99_rel_error": _pct(re, 99),
        "num_failed": int((~passed).sum()),
        "failure_rate": float((~passed).mean()) if n else float("nan"),
        "all_pass": bool(passed.all()),
        "num_nonfinite": int((~both).sum()),
    }


def classify(py, tf, tf_py_maxrel, num_tol):
    """C is the authority. `py`/`tf` are the summarize() dicts vs C."""
    py_ok, tf_ok = py["all_pass"], tf["all_pass"]
    if py_ok and tf_ok:
        return "ALL_AGREE"
    if py_ok and not tf_ok:
        return ("MIXED_OR_NUMERICAL" if tf["max_rel_error"] < num_tol
                else "TENSORFLOW_ONLY_DISAGREES")
    if tf_ok and not py_ok:
        return ("MIXED_OR_NUMERICAL" if py["max_rel_error"] < num_tol
                else "PYTHON_ONLY_DISAGREES")
    # both fail C
    if max(py["max_rel_error"], tf["max_rel_error"]) < num_tol:
        return "MIXED_OR_NUMERICAL"
    if tf_py_maxrel < num_tol:                       # candidates agree together
        return "BOTH_CANDIDATES_DISAGREE_WITH_C"
    return "MIXED_OR_NUMERICAL"


# ------------------------------------------------------------------- driver
def run(args):
    funcs = FUNCS_ALL if args.functions == "all" else \
        [int(x) for x in args.functions.split(",")]
    dims = DIMS_ALL if args.dimensions == "all" else \
        [int(x) for x in args.dimensions.split(",")]
    if args.smoke:
        funcs, dims, args.random_points = [1, 2, 11, 17, 21], [10], 50

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "test_points"), exist_ok=True)
    A, R, EPS, NT = args.atol, args.rtol, args.rel_eps, args.numerical_tol
    shifts = np.loadtxt(os.path.join(os.environ["CEC13_DATA"],
                                     "shift_data.txt")).ravel()

    print(f"Building the official C reference (once)...")
    build_dir = tempfile.mkdtemp(prefix="cec13c_")
    binary = vc.build_c_binary(build_dir)

    overall, category, failures, optima, allrows = [], [], [], [], []
    errors = []
    try:
        for dim in dims:
            o = shifts[:dim].copy()
            X, cats, ids = generate_points(dim, o, args.random_points, args.seed)
            cats = np.array(cats)
            np.savez_compressed(
                os.path.join(outdir, "test_points", f"d{dim}.npz"),
                X=X, categories=cats, ids=np.array(ids))
            n_rand = int((cats == "random").sum())
            print(f"  D={dim}: {len(X)} points ({n_rand} random)  "
                  f"evaluating C / port / ours ...", flush=True)

            cvals = eval_c(binary, X, dim)
            pvals = eval_port(dim, X)
            tvals, nvals = eval_ours(dim, X, errors)

            for fn in funcs:
                spec_bias = _build(f"f{fn}", dim).optimum
                c, p = cvals[fn - 1], pvals[fn - 1]
                t, nn = tvals[fn - 1], nvals[fn - 1]
                cmp = {
                    "py": compare(p, c, A, R, EPS),
                    "tf": compare(t, c, A, R, EPS),
                    "np": compare(nn, c, A, R, EPS),
                    "tfpy": compare(t, p, A, R, EPS),
                }
                s = {k: summarize(*cmp[k]) for k in ("py", "tf", "np", "tfpy")}
                cls = classify(s["py"], s["tf"], s["tfpy"]["max_rel_error"], NT)

                # ---- overall_summary row ----
                row = {"function": fn, "dimension": dim,
                       "num_total_points": len(X),
                       "num_random_points": n_rand,
                       "num_deterministic_points": len(X) - n_rand,
                       "classification": cls, "atol": A, "rtol": R,
                       "seed": args.seed}
                for who in ("py", "tf", "np"):
                    for m, v in s[who].items():
                        row[f"{who}_c_{m}"] = v
                row["tf_py_mean_abs_error"] = s["tfpy"]["mean_abs_error"]
                row["tf_py_max_abs_error"] = s["tfpy"]["max_abs_error"]
                row["tf_py_mean_rel_error"] = s["tfpy"]["mean_rel_error"]
                row["tf_py_max_rel_error"] = s["tfpy"]["max_rel_error"]
                overall.append(row)

                # ---- category_summary rows ----
                for cat in sorted(set(cats)):
                    mask = cats == cat
                    crow = {"function": fn, "dimension": dim,
                            "point_category": cat, "num_points": int(mask.sum())}
                    for who in ("py", "tf"):
                        ae, re, pa, bo = cmp[who]
                        cs = summarize(ae[mask], re[mask], pa[mask], bo[mask])
                        crow[f"{who}_c_mean_abs_error"] = cs["mean_abs_error"]
                        crow[f"{who}_c_max_abs_error"] = cs["max_abs_error"]
                        crow[f"{who}_c_mean_rel_error"] = cs["mean_rel_error"]
                        crow[f"{who}_c_max_rel_error"] = cs["max_rel_error"]
                        crow[f"{who}_c_num_failed"] = cs["num_failed"]
                        crow[f"{who}_c_failure_rate"] = cs["failure_rate"]
                    category.append(crow)

                # ---- optimum_validation row (point 0 is the optimum) ----
                exp = float(spec_bias)
                cv, pv, tv, nv = c[0], p[0], t[0], nn[0]
                optima.append({
                    "function": fn, "dimension": dim,
                    "expected_optimum_value": exp,
                    "c_value_at_optimum": cv,
                    "python_value_at_optimum": pv,
                    "tensorflow_value_at_optimum": tv,
                    "np_value_at_optimum": nv,
                    "c_error_vs_expected": abs(cv - exp),
                    "python_error_vs_c": abs(pv - cv),
                    "tensorflow_error_vs_c": abs(tv - cv),
                    "np_error_vs_c": abs(nv - cv),
                    "pass_c_expected": bool(abs(cv - exp) <= A + R * abs(exp)),
                    "pass_python_c": bool(abs(pv - cv) <= A + R * abs(cv)),
                    "pass_tensorflow_c": bool(abs(tv - cv) <= A + R * abs(cv)),
                    "pass_np_c": bool(abs(nv - cv) <= A + R * abs(cv)),
                })

                # ---- failures.csv (any point failing py-vs-C or tf-vs-C) ----
                fail_mask = (~cmp["py"][2]) | (~cmp["tf"][2])
                for pi in np.nonzero(fail_mask)[0]:
                    rec = {"function": fn, "dimension": dim,
                           "point_category": cats[pi], "point_index": int(pi),
                           "test_point_id": ids[pi], "seed": args.seed,
                           "c_value": c[pi], "python_value": p[pi],
                           "tensorflow_value": t[pi], "np_value": nn[pi],
                           "abs_error_py_c": cmp["py"][0][pi],
                           "rel_error_py_c": cmp["py"][1][pi],
                           "pass_py_c": bool(cmp["py"][2][pi]),
                           "abs_error_tf_c": cmp["tf"][0][pi],
                           "rel_error_tf_c": cmp["tf"][1][pi],
                           "pass_tf_c": bool(cmp["tf"][2][pi]),
                           "abs_error_tf_py": cmp["tfpy"][0][pi],
                           "rel_error_tf_py": cmp["tfpy"][1][pi]}
                    if dim <= 10:
                        rec["x"] = ";".join(f"{v:.17g}" for v in X[pi])
                    failures.append(rec)

                if args.save_all_evaluations:
                    for pi in range(len(X)):
                        allrows.append({
                            "function": fn, "dimension": dim,
                            "point_category": cats[pi], "point_index": int(pi),
                            "c_value": c[pi], "python_value": p[pi],
                            "tensorflow_value": t[pi], "np_value": nn[pi],
                            "abs_error_py_c": cmp["py"][0][pi],
                            "abs_error_tf_c": cmp["tf"][0][pi],
                            "abs_error_np_c": cmp["np"][0][pi]})
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    _write(outdir, overall, category, failures, optima, allrows, errors)
    _console(overall, funcs, dims, args)


def _write(outdir, overall, category, failures, optima, allrows, errors):
    def dump(name, rows, sort_key=None):
        if sort_key:
            rows = sorted(rows, key=sort_key)
        path = os.path.join(outdir, name)
        cols = list({k for r in rows for k in r}) if rows else []
        # stable, readable column order: identity fields first
        head = [c for c in ("function", "dimension", "point_category",
                            "point_index", "test_point_id", "classification")
                if c in cols]
        cols = head + [c for c in sorted(cols) if c not in head]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  wrote {path}  ({len(rows)} rows)")

    dump("overall_summary.csv", overall)
    dump("category_summary.csv", category)
    dump("failures.csv", failures,
         sort_key=lambda r: -max(_finite(r["abs_error_py_c"]),
                                 _finite(r["abs_error_tf_c"])))
    dump("optimum_validation.csv", optima)
    if allrows:
        dump("all_evaluations.csv", allrows)
    if errors:
        with open(os.path.join(outdir, "errors.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["function", "dimension", "message"])
            w.writerows(errors)
        print(f"  wrote {os.path.join(outdir, 'errors.csv')} "
              f"({len(errors)} rows)")


def _finite(v):
    v = float(v)
    return v if np.isfinite(v) else 1e308


def _console(overall, funcs, dims, args):
    def block(who, label):
        full_pass = sum(1 for r in overall if r[f"{who}_c_all_pass"])
        failed = sum(r[f"{who}_c_num_failed"] for r in overall)
        worst = max(overall, key=lambda r: _finite(r[f"{who}_c_max_abs_error"]))
        maxrel = max(_finite(r[f"{who}_c_max_rel_error"]) for r in overall)
        print(f"\n{label} vs C:")
        print(f"  function/dimension combos fully passing: "
              f"{full_pass}/{len(overall)}")
        print(f"  total failed evaluations: {failed}")
        print(f"  worst function: f{worst['function']} (D={worst['dimension']}), "
              f"max abs error {worst[f'{who}_c_max_abs_error']:.3e}")
        print(f"  maximum relative error: {maxrel:.3e}")

    print("\n" + "=" * 60 + "\nBenchmark validation completed.")
    print(f"Functions tested: {funcs}")
    print(f"Dimensions tested: {dims}")
    print(f"atol={args.atol}  rtol={args.rtol}  seed={args.seed}")
    block("py", "Python (port)")
    block("tf", "TensorFlow (ours, float32)")
    block("np", "NumPy (ours, float64)")

    from collections import Counter
    counts = Counter(r["classification"] for r in overall)
    print("\nClassification (C is the authority):")
    for k, v in counts.most_common():
        print(f"  {k:32s} {v}")
    for label, cls in (("Python disagrees, our TF agrees", "PYTHON_ONLY_DISAGREES"),
                       ("Our TF disagrees, Python agrees", "TENSORFLOW_ONLY_DISAGREES"),
                       ("Both candidates disagree with C (but agree together)",
                        "BOTH_CANDIDATES_DISAGREE_WITH_C")):
        fns = sorted({(r["function"], r["dimension"]) for r in overall
                      if r["classification"] == cls})
        if fns:
            print(f"\n{label}:\n  {fns}")


def _selftest():
    """Sanity-check the metric utilities on hand values."""
    ae, re, pa, bo = compare([1.0, 2.0, np.inf], [1.0, 2.0000001, 3.0],
                             atol=1e-9, rtol=1e-6, eps=1e-12)
    assert bo.tolist() == [True, True, False]
    assert pa.tolist() == [True, True, False]          # #1 within rtol, #2 near
    assert abs(ae[1] - 1e-7) < 1e-12
    assert not np.isfinite(ae[2])
    s = summarize(ae, re, pa, bo)
    assert s["num_failed"] == 1 and s["num_nonfinite"] == 1
    assert classify({"all_pass": True, "max_rel_error": 0},
                    {"all_pass": True, "max_rel_error": 0}, 0.0, 1e-4) \
        == "ALL_AGREE"
    assert classify({"all_pass": False, "max_rel_error": 0.5},
                    {"all_pass": True, "max_rel_error": 0}, 0.0, 1e-4) \
        == "PYTHON_ONLY_DISAGREES"
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--functions", default="all",
                    help="'all' or comma list, e.g. 2,11,17")
    ap.add_argument("--dimensions", default="all",
                    help="'all' or comma list, e.g. 10,30,50")
    ap.add_argument("--random-points", type=int, default=2000,
                    dest="random_points")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--rel-eps", type=float, default=1e-12, dest="rel_eps")
    ap.add_argument("--numerical-tol", type=float, default=1e-4,
                    dest="numerical_tol",
                    help="disagreement below this relative size is labelled "
                         "MIXED_OR_NUMERICAL rather than a formula bug")
    ap.add_argument("--output-dir", default=os.path.join(
        REPO, "results", "benchmark_validation"), dest="output_dir")
    ap.add_argument("--save-all-evaluations", action="store_true",
                    dest="save_all_evaluations")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: funcs [1,2,11,17,21], D=10, 50 random pts")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    args = ap.parse_args()
    if args.self_test:
        _selftest()
        return
    run(args)


if __name__ == "__main__":
    main()
