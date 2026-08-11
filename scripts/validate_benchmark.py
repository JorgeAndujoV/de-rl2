"""Validate our CEC'13 implementation against the reference Python port.

Both read the SAME official data (shift_data.txt, M_D30.txt), so any nonzero
difference is a CODE discrepancy in one of the two. Where they differ, judge
against the official CEC'13 report (Liang et al. 2013) to decide which is right.

Run from the repo root on a machine with the venv (TensorFlow is needed to import
our benchmark module):

    ./.venv/bin/python scripts/validate_benchmark.py            # the 11 used functions
    ./.venv/bin/python scripts/validate_benchmark.py --all      # all 28

The reference (`cec13 python benchmark/`) mutates its own shift/rotation state
inside the composition functions, so we build a FRESH reference object for every
single evaluation — correct, if a little slow.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(REPO, "cec13 python benchmark", "cec13 python benchmark")
DIM = 30
USED = [2, 6, 11, 12, 17, 3, 7, 15, 10, 16, 22]

# Our benchmark reads data from CEC13_DATA (absolute), so it is independent of
# the working directory; the reference reads a RELATIVE 'extdata/', so we run
# with cwd = REF_DIR and give ours an absolute path.
os.environ.setdefault(
    "CEC13_DATA", os.path.join(REPO, "data", "cec13", "input_data"))
sys.path.insert(0, os.path.join(REPO, "src"))
from derl2.benchmarks import build_benchmark          # noqa: E402


def _load(path):
    return np.loadtxt(path).ravel()


def data_parity():
    ours = os.path.join(REPO, "data", "cec13", "input_data")
    ref = os.path.join(REF_DIR, "extdata")
    ok = True
    for f in ("shift_data.txt", "M_D30.txt"):
        a, b = _load(os.path.join(ours, f)), _load(os.path.join(ref, f))
        n = min(a.size, b.size)
        same = np.allclose(a[:n], b[:n])
        print(f"  data parity {f:16s}: identical={same}  "
              f"max|diff|={np.max(np.abs(a[:n] - b[:n])):.3e}  "
              f"(sizes {a.size} vs {b.size})")
        ok = ok and same
    return ok


def _fresh_reference():
    """A brand-new reference object (re-reads its data), so the composition
    functions' in-place mutation of self.O/M never leaks between evaluations."""
    spec = importlib.util.spec_from_file_location(
        "cecref", os.path.join(REF_DIR, "functions.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.CEC_functions(DIM)


def compare(functions):
    o = _load(os.path.join(REPO, "data", "cec13", "input_data",
                           "shift_data.txt"))[:DIM]
    rng = np.random.default_rng(0)
    pts = [("optimum x=o", o)] + [
        (f"random {i}", rng.uniform(-100.0, 100.0, DIM)) for i in range(8)]

    print(f"\n{'fn':>4}  {'name':30s}  {'max|our-ref|':>14}  verdict")
    rows = []
    cwd = os.getcwd()
    os.chdir(REF_DIR)                       # for the reference's relative extdata
    try:
        for n in functions:
            spec = build_benchmark(f"cec13:f{n}", DIM)
            worst = 0.0
            for _lbl, x in pts:
                ours = float(spec.evaluate_np(x))
                ref = float(_fresh_reference().Y(np.asarray(x, np.float64), n))
                worst = max(worst, abs(ours - ref))
            rel = worst / (abs(spec.optimum) + 1.0)
            verdict = "MATCH" if rel < 1e-4 else "*** DIFFERS ***"
            rows.append((n, spec.name, worst, verdict))
            print(f"{n:>4}  {spec.name:30s}  {worst:14.4e}  {verdict}")
    finally:
        os.chdir(cwd)

    bad = [r for r in rows if "DIFFERS" in r[3]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} match; "
          f"{len(bad)} differ: {[r[0] for r in bad]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true",
                    help="check all 28 functions (default: the 11 used).")
    args = ap.parse_args()
    print("=== data parity (must be identical for a fair comparison) ===")
    if not data_parity():
        print("WARNING: data differs; comparison below is not apples-to-apples.")
    compare(list(range(1, 29)) if args.all else USED)
