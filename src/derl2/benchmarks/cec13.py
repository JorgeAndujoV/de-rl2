"""CEC'13 real-parameter optimization suite, functions 1-20.

Reference: Liang, Qu, Suganthan, Hernandez-Diaz, Technical Report 201212,
Zhengzhou University / NTU, 2013.

Minimization on [-100, 100]^D. Each f_i has a bias f_i(x*) given in Table I;
the quantity to report is the error f(x) - f_i(x*).

Composition functions f21-f28 are deliberately not implemented.

The NumPy path is float64 and is the reference. The TensorFlow path is float32
(matching VectorizedDE) and will lose precision on the ill-conditioned
functions f2, f3, f4 — expected, not a bug.
"""

import numpy as np
import tensorflow as tf

from derl2.benchmarks.registry import BenchmarkSpec, register_suite
from derl2.benchmarks.cec13_data import shift_for, rotation_for

LOWER, UPPER = -100.0, 100.0

_OPTIMA = {
    1: -1400.0, 2: -1300.0, 3: -1200.0, 4: -1100.0, 5: -1000.0,
    6: -900.0, 7: -800.0, 8: -700.0, 9: -600.0, 10: -500.0,
    11: -400.0, 12: -300.0, 13: -200.0, 14: -100.0, 15: 100.0,
    16: 200.0, 17: 300.0, 18: 400.0, 19: 500.0, 20: 600.0,
}

_NAMES = {
    1: "Sphere", 2: "Rotated High Conditioned Elliptic",
    3: "Rotated Bent Cigar", 4: "Rotated Discus", 5: "Different Powers",
    6: "Rotated Rosenbrock", 7: "Rotated Schaffers F7", 8: "Rotated Ackley",
    9: "Rotated Weierstrass", 10: "Rotated Griewank", 11: "Rastrigin",
    12: "Rotated Rastrigin", 13: "Non-Continuous Rotated Rastrigin",
    14: "Schwefel", 15: "Rotated Schwefel", 16: "Rotated Katsuura",
    17: "Lunacek bi-Rastrigin", 18: "Rotated Lunacek bi-Rastrigin",
    19: "Rotated Expanded Griewank plus Rosenbrock",
    20: "Rotated Expanded Schaffer F6",
}


# ----------------------------------------------------------- transformations

def _lambda_diag(dim, alpha):
    """Conditioning matrix Lambda^alpha as a (D,) vector."""
    i = np.arange(dim, dtype=np.float64)
    return alpha ** (i / (2.0 * (dim - 1)))


def _t_osz_np(x):
    """T_osz: smooth local irregularities. Report section 1.1."""
    sign = np.sign(x)
    nz = x != 0
    x_hat = np.zeros_like(x)
    x_hat[nz] = np.log(np.abs(x[nz]))
    c1 = np.where(x > 0, 10.0, 5.5)
    c2 = np.where(x > 0, 7.9, 3.1)
    return sign * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat)
                                          + np.sin(c2 * x_hat)))


def _t_asy_np(x, beta):
    """T_asy^beta: symmetry breaking, applied only where x > 0."""
    dim = x.shape[-1]
    i = np.arange(dim, dtype=np.float64)
    expo = 1.0 + beta * (i / (dim - 1)) * np.sqrt(np.abs(x))
    return np.where(x > 0, np.abs(x) ** expo, x)


def _t_osz_tf(x):
    sign = tf.sign(x)
    nz = tf.not_equal(x, 0.0)
    safe = tf.where(nz, tf.abs(x), tf.ones_like(x))
    x_hat = tf.where(nz, tf.math.log(safe), tf.zeros_like(x))
    c1 = tf.where(x > 0.0, 10.0, 5.5)
    c2 = tf.where(x > 0.0, 7.9, 3.1)
    return sign * tf.exp(x_hat + 0.049 * (tf.sin(c1 * x_hat)
                                          + tf.sin(c2 * x_hat)))


def _t_asy_tf(x, beta):
    dim = x.shape[-1]
    i = tf.range(dim, dtype=x.dtype)
    expo = 1.0 + beta * (i / (dim - 1.0)) * tf.sqrt(tf.abs(x))
    return tf.where(x > 0.0, tf.pow(tf.abs(x), expo), x)


# ------------------------------------------------------------ base functions

def _sphere_np(z):
    return np.sum(z ** 2, axis=-1)


def _elliptic_np(z):
    d = z.shape[-1]
    w = 1e6 ** (np.arange(d, dtype=np.float64) / (d - 1))
    return np.sum(w * z ** 2, axis=-1)


def _bent_cigar_np(z):
    return z[..., 0] ** 2 + 1e6 * np.sum(z[..., 1:] ** 2, axis=-1)


def _discus_np(z):
    return 1e6 * z[..., 0] ** 2 + np.sum(z[..., 1:] ** 2, axis=-1)


def _different_powers_np(z):
    d = z.shape[-1]
    e = 2.0 + 4.0 * np.arange(d, dtype=np.float64) / (d - 1)
    return np.sqrt(np.sum(np.abs(z) ** e, axis=-1))


def _rosenbrock_np(z):
    a, b = z[..., :-1], z[..., 1:]
    return np.sum(100.0 * (a ** 2 - b) ** 2 + (a - 1.0) ** 2, axis=-1)


def _schaffer_f7_np(y):
    d = y.shape[-1]
    z = np.sqrt(y[..., :-1] ** 2 + y[..., 1:] ** 2)
    t = np.sqrt(z) + np.sqrt(z) * np.sin(50.0 * z ** 0.2) ** 2
    return (np.sum(t, axis=-1) / (d - 1)) ** 2


def _ackley_np(z):
    d = z.shape[-1]
    s1 = np.sum(z ** 2, axis=-1) / d
    s2 = np.sum(np.cos(2.0 * np.pi * z), axis=-1) / d
    return -20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2) + 20.0 + np.e


def _weierstrass_np(z):
    a, b, kmax = 0.5, 3.0, 20
    k = np.arange(kmax + 1, dtype=np.float64)
    ak, bk = a ** k, b ** k
    term = np.sum(ak * np.cos(2.0 * np.pi * bk * (z[..., None] + 0.5)),
                  axis=-1)
    d = z.shape[-1]
    const = d * np.sum(ak * np.cos(2.0 * np.pi * bk * 0.5))
    return np.sum(term, axis=-1) - const


def _griewank_np(z):
    d = z.shape[-1]
    i = np.sqrt(np.arange(1, d + 1, dtype=np.float64))
    return (np.sum(z ** 2, axis=-1) / 4000.0
            - np.prod(np.cos(z / i), axis=-1) + 1.0)


def _rastrigin_np(z):
    return np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=-1)


def _schwefel_np(z):
    d = z.shape[-1]
    mid = z * np.sin(np.sqrt(np.abs(z)))
    mh = np.mod(z, 500.0)
    hi = ((500.0 - mh) * np.sin(np.sqrt(np.abs(500.0 - mh)))
          - (z - 500.0) ** 2 / (10000.0 * d))
    ml = np.mod(np.abs(z), 500.0)
    lo = ((ml - 500.0) * np.sin(np.sqrt(np.abs(ml - 500.0)))
          - (z + 500.0) ** 2 / (10000.0 * d))
    g = np.where(np.abs(z) <= 500.0, mid, np.where(z > 500.0, hi, lo))
    return 418.9828872724338 * d - np.sum(g, axis=-1)


def _katsuura_np(z):
    d = z.shape[-1]
    j = np.arange(1, 33, dtype=np.float64)
    twoj = 2.0 ** j
    tz = twoj * z[..., None]
    inner = np.sum(np.abs(tz - np.round(tz)) / twoj, axis=-1)
    i = np.arange(1, d + 1, dtype=np.float64)
    prod = np.prod((1.0 + i * inner) ** (10.0 / d ** 1.2), axis=-1)
    return 10.0 / d ** 2 * prod - 10.0 / d ** 2


def _griewank_rosenbrock_np(z):
    a = z
    b = np.roll(z, -1, axis=-1)
    inner = 100.0 * (a ** 2 - b) ** 2 + (a - 1.0) ** 2
    return np.sum(inner ** 2 / 4000.0 - np.cos(inner) + 1.0, axis=-1)


def _schaffer_f6_np(z):
    a = z
    b = np.roll(z, -1, axis=-1)
    s = a ** 2 + b ** 2
    return np.sum(0.5 + (np.sin(np.sqrt(s)) ** 2 - 0.5)
                  / (1.0 + 0.001 * s) ** 2, axis=-1)


def _sphere_tf(z):
    return tf.reduce_sum(z ** 2, axis=-1)


def _elliptic_tf(z):
    d = z.shape[-1]
    w = tf.constant(1e6 ** (np.arange(d) / (d - 1)), z.dtype)
    return tf.reduce_sum(w * z ** 2, axis=-1)


def _bent_cigar_tf(z):
    return z[..., 0] ** 2 + 1e6 * tf.reduce_sum(z[..., 1:] ** 2, axis=-1)


def _discus_tf(z):
    return 1e6 * z[..., 0] ** 2 + tf.reduce_sum(z[..., 1:] ** 2, axis=-1)


def _different_powers_tf(z):
    d = z.shape[-1]
    e = tf.constant(2.0 + 4.0 * np.arange(d) / (d - 1), z.dtype)
    return tf.sqrt(tf.reduce_sum(tf.abs(z) ** e, axis=-1))


def _rosenbrock_tf(z):
    a, b = z[..., :-1], z[..., 1:]
    return tf.reduce_sum(100.0 * (a ** 2 - b) ** 2 + (a - 1.0) ** 2, axis=-1)


def _schaffer_f7_tf(y):
    d = y.shape[-1]
    z = tf.sqrt(y[..., :-1] ** 2 + y[..., 1:] ** 2)
    t = tf.sqrt(z) + tf.sqrt(z) * tf.sin(50.0 * z ** 0.2) ** 2
    return (tf.reduce_sum(t, axis=-1) / (d - 1.0)) ** 2


def _ackley_tf(z):
    d = tf.cast(tf.shape(z)[-1], z.dtype)
    s1 = tf.reduce_sum(z ** 2, axis=-1) / d
    s2 = tf.reduce_sum(tf.cos(2.0 * np.pi * z), axis=-1) / d
    return (-20.0 * tf.exp(-0.2 * tf.sqrt(s1)) - tf.exp(s2)
            + 20.0 + float(np.e))


def _weierstrass_tf(z):
    a, b, kmax = 0.5, 3.0, 20
    k = np.arange(kmax + 1, dtype=np.float64)
    ak = tf.constant(a ** k, z.dtype)
    bk = tf.constant(b ** k, z.dtype)
    term = tf.reduce_sum(ak * tf.cos(2.0 * np.pi * bk * (z[..., None] + 0.5)),
                         axis=-1)
    d = tf.cast(tf.shape(z)[-1], z.dtype)
    const = d * tf.reduce_sum(ak * tf.cos(2.0 * np.pi * bk * 0.5))
    return tf.reduce_sum(term, axis=-1) - const


def _griewank_tf(z):
    d = z.shape[-1]
    i = tf.constant(np.sqrt(np.arange(1, d + 1)), z.dtype)
    return (tf.reduce_sum(z ** 2, axis=-1) / 4000.0
            - tf.reduce_prod(tf.cos(z / i), axis=-1) + 1.0)


def _rastrigin_tf(z):
    return tf.reduce_sum(z ** 2 - 10.0 * tf.cos(2.0 * np.pi * z) + 10.0,
                         axis=-1)


def _schwefel_tf(z):
    d = tf.cast(tf.shape(z)[-1], z.dtype)
    mid = z * tf.sin(tf.sqrt(tf.abs(z)))
    mh = tf.math.floormod(z, 500.0)
    hi = ((500.0 - mh) * tf.sin(tf.sqrt(tf.abs(500.0 - mh)))
          - (z - 500.0) ** 2 / (10000.0 * d))
    ml = tf.math.floormod(tf.abs(z), 500.0)
    lo = ((ml - 500.0) * tf.sin(tf.sqrt(tf.abs(ml - 500.0)))
          - (z + 500.0) ** 2 / (10000.0 * d))
    g = tf.where(tf.abs(z) <= 500.0, mid, tf.where(z > 500.0, hi, lo))
    return 418.9828872724338 * d - tf.reduce_sum(g, axis=-1)


def _katsuura_tf(z):
    d = z.shape[-1]
    j = np.arange(1, 33, dtype=np.float64)
    twoj = tf.constant(2.0 ** j, z.dtype)
    tz = twoj * z[..., None]
    inner = tf.reduce_sum(tf.abs(tz - tf.round(tz)) / twoj, axis=-1)
    i = tf.constant(np.arange(1, d + 1, dtype=np.float64), z.dtype)
    prod = tf.reduce_prod((1.0 + i * inner) ** (10.0 / d ** 1.2), axis=-1)
    return 10.0 / d ** 2 * prod - 10.0 / d ** 2


def _griewank_rosenbrock_tf(z):
    a = z
    b = tf.roll(z, shift=-1, axis=-1)
    inner = 100.0 * (a ** 2 - b) ** 2 + (a - 1.0) ** 2
    return tf.reduce_sum(inner ** 2 / 4000.0 - tf.cos(inner) + 1.0, axis=-1)


def _schaffer_f6_tf(z):
    a = z
    b = tf.roll(z, shift=-1, axis=-1)
    s = a ** 2 + b ** 2
    return tf.reduce_sum(0.5 + (tf.sin(tf.sqrt(s)) ** 2 - 0.5)
                         / (1.0 + 0.001 * s) ** 2, axis=-1)


_BASE_NP = {
    1: _sphere_np, 2: _elliptic_np, 3: _bent_cigar_np, 4: _discus_np,
    5: _different_powers_np, 6: _rosenbrock_np, 7: _schaffer_f7_np,
    8: _ackley_np, 9: _weierstrass_np, 10: _griewank_np, 11: _rastrigin_np,
    12: _rastrigin_np, 13: _rastrigin_np, 14: _schwefel_np, 15: _schwefel_np,
    16: _katsuura_np, 19: _griewank_rosenbrock_np, 20: _schaffer_f6_np,
}

_BASE_TF = {
    1: _sphere_tf, 2: _elliptic_tf, 3: _bent_cigar_tf, 4: _discus_tf,
    5: _different_powers_tf, 6: _rosenbrock_tf, 7: _schaffer_f7_tf,
    8: _ackley_tf, 9: _weierstrass_tf, 10: _griewank_tf, 11: _rastrigin_tf,
    12: _rastrigin_tf, 13: _rastrigin_tf, 14: _schwefel_tf, 15: _schwefel_tf,
    16: _katsuura_tf, 19: _griewank_rosenbrock_tf, 20: _schaffer_f6_tf,
}


# ------------------------------------------------------------ z construction

def _z_builders(func_no, dim):
    """Return (z_np, z_tf): x -> the transformed z the base function consumes.

    Encodes the per-function shift / rotate / scale chain from section 1.3.
    Rotation is applied as z @ M.T, i.e. M x for a column vector x.
    """
    o = shift_for(func_no, dim)
    o_tf = tf.constant(o, tf.float32)
    lam10 = _lambda_diag(dim, 10.0)
    lam100 = _lambda_diag(dim, 100.0)
    lam10_tf = tf.constant(lam10, tf.float32)
    lam100_tf = tf.constant(lam100, tf.float32)

    def rot(i):
        M = rotation_for(dim, i)
        return M, tf.constant(M, tf.float32)

    n = func_no

    if n in (1, 5):                                     # shift only
        return (lambda x: x - o,
                lambda x: x - o_tf)

    if n in (2, 4):                                     # Elliptic / Discus
        M1, M1t = rot(0)
        return (lambda x: _t_osz_np((x - o) @ M1.T),
                lambda x: _t_osz_tf(tf.matmul(x - o_tf, M1t,
                                              transpose_b=True)))

    if n == 3:                                          # Bent Cigar
        M1, M1t = rot(0)
        M2, M2t = rot(1)
        return (lambda x: _t_asy_np((x - o) @ M1.T, 0.5) @ M2.T,
                lambda x: tf.matmul(
                    _t_asy_tf(tf.matmul(x - o_tf, M1t, transpose_b=True), 0.5),
                    M2t, transpose_b=True))

    if n == 6:                                          # Rosenbrock
        M1, M1t = rot(0)
        return (lambda x: (2.048 * (x - o) / 100.0) @ M1.T + 1.0,
                lambda x: tf.matmul(2.048 * (x - o_tf) / 100.0, M1t,
                                    transpose_b=True) + 1.0)

    if n in (7, 8):                                     # Schaffer F7 / Ackley
        M1, M1t = rot(0)
        M2, M2t = rot(1)
        return (lambda x: lam10 * (_t_asy_np((x - o) @ M1.T, 0.5) @ M2.T),
                lambda x: lam10_tf * tf.matmul(
                    _t_asy_tf(tf.matmul(x - o_tf, M1t, transpose_b=True), 0.5),
                    M2t, transpose_b=True))

    if n == 9:                                          # Weierstrass
        M1, M1t = rot(0)
        M2, M2t = rot(1)
        return (lambda x: lam10 * (_t_asy_np(
                    (0.5 * (x - o) / 100.0) @ M1.T, 0.5) @ M2.T),
                lambda x: lam10_tf * tf.matmul(
                    _t_asy_tf(tf.matmul(0.5 * (x - o_tf) / 100.0, M1t,
                                        transpose_b=True), 0.5),
                    M2t, transpose_b=True))

    if n == 10:                                         # Griewank
        M1, M1t = rot(0)
        return (lambda x: lam100 * ((600.0 * (x - o) / 100.0) @ M1.T),
                lambda x: lam100_tf * tf.matmul(600.0 * (x - o_tf) / 100.0,
                                                M1t, transpose_b=True))

    if n == 11:                                         # Rastrigin, separable
        return (lambda x: lam10 * _t_asy_np(
                    _t_osz_np(5.12 * (x - o) / 100.0), 0.2),
                lambda x: lam10_tf * _t_asy_tf(
                    _t_osz_tf(5.12 * (x - o_tf) / 100.0), 0.2))

    if n == 12:                                         # Rotated Rastrigin
        M1, M1t = rot(0)
        M2, M2t = rot(1)

        def z_np(x):
            y = _t_osz_np((5.12 * (x - o) / 100.0) @ M1.T)
            return (lam10 * (_t_asy_np(y, 0.2) @ M2.T)) @ M1.T

        def z_tf(x):
            y = _t_osz_tf(tf.matmul(5.12 * (x - o_tf) / 100.0, M1t,
                                    transpose_b=True))
            w = lam10_tf * tf.matmul(_t_asy_tf(y, 0.2), M2t, transpose_b=True)
            return tf.matmul(w, M1t, transpose_b=True)

        return z_np, z_tf

    if n == 13:                                         # Non-continuous
        M1, M1t = rot(0)
        M2, M2t = rot(1)

        def z_np(x):
            xb = (5.12 * (x - o) / 100.0) @ M1.T
            y = np.where(np.abs(xb) <= 0.5, xb, np.round(2.0 * xb) / 2.0)
            return (lam10 * (_t_asy_np(_t_osz_np(y), 0.2) @ M2.T)) @ M1.T

        def z_tf(x):
            xb = tf.matmul(5.12 * (x - o_tf) / 100.0, M1t, transpose_b=True)
            y = tf.where(tf.abs(xb) <= 0.5, xb, tf.round(2.0 * xb) / 2.0)
            w = lam10_tf * tf.matmul(_t_asy_tf(_t_osz_tf(y), 0.2), M2t,
                                     transpose_b=True)
            return tf.matmul(w, M1t, transpose_b=True)

        return z_np, z_tf

    if n == 14:                                         # Schwefel, separable
        c = 4.209687462275036e2
        return (lambda x: lam10 * (1000.0 * (x - o) / 100.0) + c,
                lambda x: lam10_tf * (1000.0 * (x - o_tf) / 100.0) + c)

    if n == 15:                                         # Rotated Schwefel
        c = 4.209687462275036e2
        M1, M1t = rot(0)
        return (lambda x: lam10 * ((1000.0 * (x - o) / 100.0) @ M1.T) + c,
                lambda x: lam10_tf * tf.matmul(1000.0 * (x - o_tf) / 100.0,
                                               M1t, transpose_b=True) + c)

    if n == 16:                                         # Katsuura
        M1, M1t = rot(0)
        M2, M2t = rot(1)
        return (lambda x: lam100 * (((5.0 * (x - o) / 100.0) @ M1.T) @ M2.T),
                lambda x: lam100_tf * tf.matmul(
                    tf.matmul(5.0 * (x - o_tf) / 100.0, M1t, transpose_b=True),
                    M2t, transpose_b=True))

    if n in (17, 18):                                   # Lunacek: raw y
        return (lambda x: 10.0 * (x - o) / 100.0,
                lambda x: 10.0 * (x - o_tf) / 100.0)

    if n == 19:                                         # Griewank+Rosenbrock
        M1, M1t = rot(0)
        return (lambda x: (5.0 * (x - o) / 100.0) @ M1.T + 1.0,
                lambda x: tf.matmul(5.0 * (x - o_tf) / 100.0, M1t,
                                    transpose_b=True) + 1.0)

    if n == 20:                                         # Expanded Schaffer F6
        M1, M1t = rot(0)
        M2, M2t = rot(1)
        return (lambda x: _t_asy_np((x - o) @ M1.T, 0.5) @ M2.T,
                lambda x: tf.matmul(
                    _t_asy_tf(tf.matmul(x - o_tf, M1t, transpose_b=True), 0.5),
                    M2t, transpose_b=True))

    raise KeyError(f"CEC'13 function {n} not implemented.")


# ------------------------------------------------------ Lunacek bi-Rastrigin

def _lunacek_params(dim):
    mu0, d_par = 2.5, 1.0
    s = 1.0 - 1.0 / (2.0 * np.sqrt(dim + 20.0) - 8.2)
    mu1 = -np.sqrt((mu0 ** 2 - d_par) / s)
    return mu0, mu1, s, d_par


def _lunacek_np(y, dim, rotate):
    mu0, mu1, s, d_par = _lunacek_params(dim)
    x_bar = 2.0 * np.sign(y) * y + mu0
    t1 = np.sum((x_bar - mu0) ** 2, axis=-1)
    t2 = d_par * dim + s * np.sum((x_bar - mu1) ** 2, axis=-1)
    lam = _lambda_diag(dim, 100.0)
    if rotate:
        M1, M2 = rotation_for(dim, 0), rotation_for(dim, 1)
        z = (lam * ((x_bar - mu0) @ M1.T)) @ M2.T
    else:
        z = lam * (x_bar - mu0)
    return (np.minimum(t1, t2)
            + 10.0 * (dim - np.sum(np.cos(2.0 * np.pi * z), axis=-1)))


def _lunacek_tf(y, dim, rotate):
    mu0, mu1, s, d_par = _lunacek_params(dim)
    x_bar = 2.0 * tf.sign(y) * y + mu0
    t1 = tf.reduce_sum((x_bar - mu0) ** 2, axis=-1)
    t2 = d_par * dim + s * tf.reduce_sum((x_bar - mu1) ** 2, axis=-1)
    lam = tf.constant(_lambda_diag(dim, 100.0), tf.float32)
    if rotate:
        M1 = tf.constant(rotation_for(dim, 0), tf.float32)
        M2 = tf.constant(rotation_for(dim, 1), tf.float32)
        z = tf.matmul(lam * tf.matmul(x_bar - mu0, M1, transpose_b=True),
                      M2, transpose_b=True)
    else:
        z = lam * (x_bar - mu0)
    return (tf.minimum(t1, t2)
            + 10.0 * (dim - tf.reduce_sum(tf.cos(2.0 * np.pi * z), axis=-1)))


# -------------------------------------------------------------------- assembly

def _build(name, dim):
    if not (name.startswith("f") and name[1:].isdigit()):
        raise KeyError(f"CEC'13 names look like 'f11', got {name!r}.")
    n = int(name[1:])
    if n not in _OPTIMA:
        raise KeyError(
            f"CEC'13 f{n} is not available. Implemented: f1-f20 "
            f"(composition functions f21-f28 are out of scope)."
        )

    z_np, z_tf = _z_builders(n, dim)
    bias = _OPTIMA[n]

    if n in (17, 18):
        rotate = (n == 18)

        def eval_np(x):
            y = z_np(np.atleast_2d(np.asarray(x, np.float64)))
            return float(_lunacek_np(y, dim, rotate)[0] + bias)

        def eval_tf(x):
            return _lunacek_tf(z_tf(x), dim, rotate) + bias
    else:
        g_np, g_tf = _BASE_NP[n], _BASE_TF[n]

        def eval_np(x):
            z = z_np(np.atleast_2d(np.asarray(x, np.float64)))
            return float(g_np(z)[0] + bias)

        def eval_tf(x):
            return g_tf(z_tf(x)) + bias

    return BenchmarkSpec(
        id=f"cec13:{name}", suite="cec13", name=_NAMES[n], dim=dim,
        lower=LOWER, upper=UPPER, optimum=bias,
        evaluate_np=eval_np, evaluate_tf=eval_tf,
    )


register_suite("cec13", _build)