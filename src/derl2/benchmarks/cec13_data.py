"""Shift vectors and rotation matrices for CEC'13.

Fixed competition data, not generated. Expected layout:

    data/cec13/input_data/{shift_data.txt, M_D2.txt, M_D10.txt, ...}

Override the location with the CEC13_DATA environment variable (useful on the
cluster, where the repo may not sit where you expect).
"""

import os
import functools
import numpy as np

# Dimensions for which M_D<dim>.txt exists in the distribution.
SUPPORTED_ROTATION_DIMS = (2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def data_root():
    return os.environ.get(
        "CEC13_DATA",
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "data", "cec13", "input_data"),
    )


def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CEC'13 data file not found: {path}\n"
            f"Place input_data under {data_root()} or set CEC13_DATA."
        )
    return path


@functools.lru_cache(maxsize=None)
def load_shifts():
    """All shift values as one flat array.

    The file is a flat stream of sub-optima, not a per-function table. For
    f1-f20 the shift is the first `dim` values; composition functions f21-f28
    consume successive blocks of `dim` values for their sub-optima.
    """
    raw = np.loadtxt(_require(os.path.join(data_root(), "shift_data.txt")))
    return raw.ravel().astype(np.float64)



@functools.lru_cache(maxsize=None)
def load_rotations(dim):
    """Ten D*D orthogonal matrices, shape (10, D, D)."""
    if dim not in SUPPORTED_ROTATION_DIMS:
        raise ValueError(
            f"No rotation matrices for D={dim}. "
            f"Available: {SUPPORTED_ROTATION_DIMS}."
        )
    raw = np.loadtxt(
        _require(os.path.join(data_root(), f"M_D{dim}.txt"))).astype(np.float64)
    expected = 10 * dim * dim
    if raw.size != expected:
        raise ValueError(
            f"M_D{dim}.txt has {raw.size} values, expected {expected} "
            f"(10 matrices of {dim}x{dim}). File layout may differ from "
            f"the standard distribution."
        )
    return raw.reshape(10, dim, dim)


def shift_for(func_no, dim, block=0):
    """Shift vector o. func_no is unused for f1-f20, which all share block 0."""
    return load_shifts()[block * dim:(block + 1) * dim].copy()

def rotation_for(dim, index=0):
    """Rotation matrix M_{index+1} for dimension dim."""
    return load_rotations(dim)[index].copy()