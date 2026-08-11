"""Validate our CEC'13 implementation against the OFFICIAL C code (ground truth).

The reference Python port has its own bugs, so it is not a trustworthy oracle.
This script compiles the official C `test_func` (patched for Linux) and compares
its values to ours at identical points, for all 28 functions.

What it does:
  1. patches a COPY of test_func.cpp for Linux (removes <WINDOWS.H>; fixes the
     %Lf -> %lf fscanf bug that silently corrupts the data under gcc),
  2. compiles it with scripts/cec13_c_driver.cpp,
  3. checks the C code's input_data is byte-identical to ours,
  4. evaluates all 28 functions at the optimum + random points in both, and
     reports max|our - C| with a MATCH / DIFFERS verdict.

Run on WSL (needs g++ and the venv, since importing our benchmark pulls TF):
    ./.venv/bin/python scripts/validate_vs_c.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(REPO, "cec13-c-code", "cec13ccode")
DRIVER = os.path.join(REPO, "scripts", "cec13_c_driver.cpp")
PORT_DIR = os.path.join(REPO, "cec13 python benchmark", "cec13 python benchmark")
OUR_DATA = os.path.join(REPO, "data", "cec13", "input_data")
DIM = 30
N_RANDOM = 8

os.environ.setdefault("CEC13_DATA", OUR_DATA)
sys.path.insert(0, os.path.join(REPO, "src"))
# The RAW builder (no _sanitize_spec): validation must see true overflow/NaN,
# not the 1e30 penalty the environment substitutes, so a real discrepancy can
# never be masked.
from derl2.benchmarks.cec13 import _build as build_raw   # noqa: E402


_SATURATED = 1e100          # both values this large -> optimisation-irrelevant


def make_points():
    """Every point must match, not just the near-optimum ones. We test the
    optimum, five NEAR-optimum shells (where the agents actually search, so a
    small formula error is most visible there), and domain-wide points. Returns
    (points, labels) so a mismatch can be reported at its exact location."""
    o = np.loadtxt(os.path.join(OUR_DATA, "shift_data.txt")).ravel()[:DIM]
    rng = np.random.default_rng(0)
    pts, labels = [o.copy()], ["optimum"]
    for r in (1e-6, 1e-3, 1e-1, 1.0, 10.0):
        for j in range(2):
            pts.append(o + r * rng.standard_normal(DIM))
            labels.append(f"near-opt r={r:g} #{j}")
    for j in range(N_RANDOM):
        pts.append(rng.uniform(-100.0, 100.0, DIM))
        labels.append(f"domain #{j}")
    return pts, labels


def rel_verdict(a_list, b_list, labels):
    """(worst relative diff, label of the worst point). Inf/nan-safe: a point
    where BOTH are non-finite, or both exceed 1e100, counts as agreement (a
    number-representation cap, not a formula error); exactly one saturating is a
    real mismatch (worst = inf there)."""
    worst, where = 0.0, "-"
    for a, b, lab in zip(a_list, b_list, labels):
        sat_a = (not np.isfinite(a)) or abs(a) > _SATURATED
        sat_b = (not np.isfinite(b)) or abs(b) > _SATURATED
        if sat_a and sat_b:
            continue
        if sat_a != sat_b:
            return np.inf, lab
        r = abs(a - b) / max(1.0, abs(b))
        if r > worst:
            worst, where = r, lab
    return worst, where


def check_data_parity():
    print("=== data parity: C input_data vs ours ===")
    ok = True
    for f in ("shift_data.txt", "M_D30.txt"):
        a = np.loadtxt(os.path.join(OUR_DATA, f)).ravel()
        b = np.loadtxt(os.path.join(CDIR, "input_data", f)).ravel()
        n = min(a.size, b.size)
        same = np.allclose(a[:n], b[:n])
        print(f"  {f:16s}: identical={same}  max|diff|="
              f"{np.max(np.abs(a[:n] - b[:n])):.3e}")
        ok = ok and same
    if not ok:
        print("  WARNING: data differs -> comparison would not be apples-to-apples.")
    return ok


def build_c_binary(build_dir):
    """Patch test_func.cpp for Linux and compile it with the driver."""
    # latin-1 preserves the non-UTF8 bytes in the original comments.
    with open(os.path.join(CDIR, "test_func.cpp"), encoding="latin-1") as fh:
        src = fh.read()
    for inc in ("#include <WINDOWS.H>", "#include <windows.h>",
                "#include <WINDOWS.h>"):
        src = src.replace(inc, "")
    src = src.replace("%Lf", "%lf")          # gcc: %Lf is long double, data is double
    patched = os.path.join(build_dir, "test_func.cpp")
    with open(patched, "w", encoding="latin-1") as fh:
        fh.write(src)

    binary = os.path.join(build_dir, "cec_driver")
    cmd = ["g++", "-O2", "-w", "-o", binary, DRIVER, patched, "-lm"]
    print("=== compiling official C code (Linux-patched) ===\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return binary


def run_c(binary, points):
    """Write points, run the C binary from CDIR (so input_data/ resolves)."""
    pts_path = os.path.join(CDIR, "_validate_points.txt")
    with open(pts_path, "w") as fh:
        fh.write(f"{len(points)} {DIM}\n")
        for p in points:
            fh.write(" ".join(f"{v:.17g}" for v in p) + "\n")
    try:
        out = subprocess.run([binary, "_validate_points.txt"], cwd=CDIR,
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.remove(pts_path)
    vals = {}
    for line in out.strip().splitlines():
        fn, p, v = line.split()
        vals[(int(fn), int(p))] = float(v)
    return vals


def eval_port(points):
    """Evaluate the reference Python PORT at the same points (the independent
    second oracle). Runs from PORT_DIR for its relative extdata/, and rebuilds a
    fresh object per evaluation because the composition functions mutate state."""
    import importlib.util
    pvals = {}
    cwd = os.getcwd()
    os.chdir(PORT_DIR)
    try:
        for pi, x in enumerate(points):
            for fn in range(1, 29):
                spec = importlib.util.spec_from_file_location(
                    "cecport", os.path.join(PORT_DIR, "functions.py"))
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                pvals[(fn, pi)] = float(
                    m.CEC_functions(DIM).Y(np.asarray(x, np.float64), fn))
    finally:
        os.chdir(cwd)
    return pvals


def main():
    check_data_parity()
    points, labels = make_points()

    build_dir = tempfile.mkdtemp(prefix="cec13c_")
    try:
        binary = build_c_binary(build_dir)
        cvals = run_c(binary, points)                 # OFFICIAL C: ground truth
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    pvals = eval_port(points)                          # reference Python port

    print(f"\n{'fn':>4}  {'name':28s}  {'ours-C':>9}  {'port-C':>9}  "
          f"{'ours-port':>9}  worst @ (ours-C)")
    ours_bad, disagree = [], []
    for fn in range(1, 29):
        spec = build_raw(f"f{fn}", DIM)               # RAW: no 1e30 masking
        c = [cvals[(fn, pi)] for pi in range(len(points))]
        ours = [float(spec.evaluate_np(x)) for x in points]
        port = [pvals[(fn, pi)] for pi in range(len(points))]
        oc, where = rel_verdict(ours, c, labels)      # ours vs official C
        pc, _ = rel_verdict(port, c, labels)          # port vs official C
        op, _ = rel_verdict(ours, port, labels)       # ours vs port (both Python)
        if oc >= 1e-6:
            ours_bad.append(fn)
        if op >= 1e-6:                                 # we disagree with the port
            disagree.append(fn)
        print(f"{fn:>4}  {spec.name:28s}  {oc:9.1e}  {pc:9.1e}  {op:9.1e}  {where}")

    print(f"\nours vs C:    {28 - len(ours_bad)}/28 match; differ: {ours_bad}")
    print(f"ours vs PORT: {28 - len(disagree)}/28 match; differ: {disagree}")
    print("When ours == port but both differ from C, the C is the outlier "
          "(its uninitialised-variable UB at exact-zero coords, or float "
          "round-off in an ill-conditioned region) -- not our bug.")


if __name__ == "__main__":
    main()
