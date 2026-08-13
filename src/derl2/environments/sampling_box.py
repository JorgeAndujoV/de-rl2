"""Sampling-box transform (spec §6.1).

The sampling box is the one piece of state that connects consecutive segments.
Given the previous segment's final population and a chosen `sampling_box` action
index, this produces the box `[box_lo, box_hi]` inside which the next segment's
fresh population is sampled uniformly. A converged population yields a small
box; the action's scale factor decides how much to trust that region.

Convention: everything here is in HALF-WIDTHS. `half_width_j` is half the
per-dimension extent of the population; `scaled_j` is the (scaled, floored)
half-width actually used; the box is `[center - scaled, center + scaled]`. The
full-width fractions the observation needs (box_contraction, current box width)
are computed by the caller from `box_lo`/`box_hi`, never mixed in here.

`box_at_floor` and `incumbent_in_box` are returned for logging only — the
environment records them; it does not act on them.
"""

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# freedom++ geometry (EXP021/EXP022): a UNIFIED sampling region for both the
# axis-aligned and covariance shapes, plus low-discrepancy / opposition fills.
#
# The whole idea (see the EXP021 design): a region is a linear map of the
# reference cube z in [-1, 1]^D,
#
#       x = center + L @ z
#
# where L is a lower-triangular Cholesky-style factor. `axis` is simply the
# DIAGONAL-L special case (L = diag(half_width)); `covariance` is the full
# Cholesky of the regularized population covariance. So one sampler handles both
# shapes, one whitening L^-1 handles every shape-aware observation feature, and
# the axis case is a genuine special case, not a separate code branch.
# --------------------------------------------------------------------------- #

# First 64 primes: Halton base for dimension d is _PRIMES[d]. CEC'13 runs at
# D=30 (base up to 113); the list is long enough for any D we use.
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
           67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
           139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
           211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
           281, 283, 293, 307, 311]


def _halton_vec(indices, base):
    """Radical inverse of each index in `base` -> values in [0, 1). Vectorized
    over `indices` (a 1-D int array). index 0 maps to 0.0, so callers start at 1."""
    idx = np.asarray(indices, dtype=np.int64).copy()
    result = np.zeros(idx.shape, dtype=np.float64)
    f = 1.0
    while np.any(idx > 0):
        f /= base
        result += f * (idx % base)
        idx //= base
    return result


@dataclass
class FreedomBox:
    """A freedom++ sampling region, parameterized by the linear map x=center+L z.

    Reduces to an axis-aligned box exactly when L is diagonal. `L_inv` is the
    whitening map the shape-aware observation features share (A1, B4, B5, C4);
    `eff_radius` = det(L)^(1/D) is the volume-based "radius" B5/C4 use;
    box_lo/box_hi are the domain-clipped bounding box of the cube image (for
    logging and incumbent containment)."""

    center: np.ndarray          # (D,)
    L: np.ndarray               # (D,D) lower-triangular factor (diagonal if axis)
    L_inv: np.ndarray           # (D,D) whitening map = inv(L)
    box_lo: np.ndarray          # (D,) domain-clipped bounding box lower
    box_hi: np.ndarray          # (D,) domain-clipped bounding box upper
    eff_radius: float           # det(L)^(1/D), the box's effective radius
    box_at_floor: bool
    incumbent_in_box: bool


def _resolve_center(pop, domain_lo, domain_hi, box_center, incumbent, rng):
    if box_center == "centroid":
        return pop.mean(axis=0)
    if box_center == "incumbent":
        return incumbent.copy()
    if box_center == "random":
        if rng is None:
            raise ValueError("box_center='random' requires an rng.")
        return domain_lo + rng.random(domain_lo.shape) * (domain_hi - domain_lo)
    raise ValueError(
        f"box_center must be 'centroid', 'incumbent' or 'random', "
        f"got {box_center!r}.")


def build_frame(box_shape, final_population, scale, box_min_frac,
                domain_lo, domain_hi, box_center, incumbent, rng=None):
    """Build the freedom++ region factor L (and its inverse) for the next segment.

    box_shape : 'axis' -> L = diag(half_width); 'covariance' -> L = chol(Sigma_reg).
    scale     : the agent's box_scale; multiplies the region's half-width / std.
    Everything is symmetric about `center` (the cube z in [-1,1]^D is symmetric),
    so the axis case is exactly the current per-dimension rescaling with a
    diagonal L, and opposition reflection x'=2c-x holds for any L.
    """
    pop = np.asarray(final_population, dtype=np.float64)
    domain_lo = np.asarray(domain_lo, dtype=np.float64)
    domain_hi = np.asarray(domain_hi, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    d = pop.shape[1]
    domain_half = (domain_hi - domain_lo) / 2.0
    floor = box_min_frac * domain_half                       # per-dim half-width floor

    center = _resolve_center(pop, domain_lo, domain_hi, box_center, incumbent, rng)

    if box_shape == "axis":
        half_width = (pop.max(axis=0) - pop.min(axis=0)) / 2.0
        scaled = scale * half_width
        box_at_floor = bool(np.any(scaled < floor))
        scaled = np.maximum(scaled, floor)
        L = np.diag(scaled)
    elif box_shape == "covariance":
        sigma = (np.cov(pop, rowvar=False) if pop.shape[0] > 1
                 else np.zeros((d, d)))
        sigma = np.atleast_2d(sigma)
        floor_var = floor ** 2
        box_at_floor = bool(np.any(np.diag(sigma) < floor_var))
        # Additive per-dim variance floor keeps Sigma_reg positive-definite
        # (PSD + strictly-positive diagonal), so the Cholesky always succeeds.
        sigma_reg = sigma + np.diag(floor_var)
        L = scale * np.linalg.cholesky(sigma_reg)
    else:
        raise ValueError(
            f"box_shape must be 'axis' or 'covariance', got {box_shape!r}.")

    # Bounding box of the cube image {center + L z : z in [-1,1]^D}: per dim the
    # reach is the L1 norm of that row of L (= |L_dd| = half_width for a diagonal
    # L). Domain-clipped, for logging and incumbent containment only.
    extent = np.sum(np.abs(L), axis=1)
    box_lo = np.clip(center - extent, domain_lo, domain_hi)
    box_hi = np.clip(center + extent, domain_lo, domain_hi)
    incumbent_in_box = bool(np.all(incumbent >= box_lo)
                            and np.all(incumbent <= box_hi))

    # det(L)^(1/D): the box's effective radius (volume^(1/D)); reduces to the
    # geometric-mean half-width when L is diagonal. |det| guards sign/round-off.
    sign, logabsdet = np.linalg.slogdet(L)
    eff_radius = float(np.exp(logabsdet / d)) if sign != 0 else 0.0

    return FreedomBox(
        center=center.astype(np.float64),
        L=L.astype(np.float64),
        L_inv=np.linalg.inv(L).astype(np.float64),
        box_lo=box_lo.astype(np.float32),
        box_hi=box_hi.astype(np.float32),
        eff_radius=eff_radius,
        box_at_floor=box_at_floor,
        incumbent_in_box=incumbent_in_box,
    )


def sample_from_box(rule, L, center, n, domain_lo, domain_hi, rng,
                    index_offset=0):
    """Sample `n` points in the region x = center + L z, z filling [-1,1]^D by
    `rule`, then clip to the domain. One implementation for both box shapes.

    uniform     : z_d ~ U(-1, 1) i.i.d.
    halton      : z_d = 2*halton(i, prime_d) - 1  (low-discrepancy; index_offset
                  advances the sequence per restart so restarts differ).
    opposition  : sample ceil(n/2) points and add their mirrors z -> -z, i.e.
                  x' = center - L z = 2*center - x  (same FE count as uniform).
    """
    center = np.asarray(center, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    d = center.shape[0]
    if rule == "uniform":
        z = rng.uniform(-1.0, 1.0, size=(n, d))
    elif rule == "halton":
        idx = index_offset + np.arange(1, n + 1)             # skip index 0 (->0.0)
        z = np.empty((n, d), dtype=np.float64)
        for j in range(d):
            z[:, j] = 2.0 * _halton_vec(idx, _PRIMES[j]) - 1.0
    elif rule == "opposition":
        m = (n + 1) // 2
        zb = rng.uniform(-1.0, 1.0, size=(m, d))
        z = np.concatenate([zb, -zb], axis=0)[:n]            # mirror pairs
    else:
        raise ValueError(
            f"sampling_rule must be 'uniform', 'halton' or 'opposition', "
            f"got {rule!r}.")
    x = center[None, :] + z @ L.T
    x = np.clip(x, np.asarray(domain_lo, dtype=np.float64),
                np.asarray(domain_hi, dtype=np.float64))
    return x.astype(np.float32)


@dataclass
class SamplingBox:
    """The transformed box plus the two logged diagnostics.

    center/half_width describe the intended (pre-domain-clip) box geometry and
    are what the observation's B4 (dist_center_to_best) uses. box_lo/box_hi are
    the actual sampling bounds after clipping to the domain, and can be
    asymmetric about center when the box runs off a domain edge.
    """

    box_lo: np.ndarray          # (D,) actual lower sampling bound (domain-clipped)
    box_hi: np.ndarray          # (D,) actual upper sampling bound (domain-clipped)
    center: np.ndarray          # (D,) intended box center
    half_width: np.ndarray      # (D,) intended box half-width (scaled + floored)
    box_at_floor: bool          # did the collapse floor bind in any dimension?
    incumbent_in_box: bool      # does best-so-far lie inside [box_lo, box_hi]?
    chol: object = None         # (D,D) scaled Cholesky for a COVARIANCE box, else
                                # None (axis-aligned): population = center + z @ chol.T


def transform_box(final_population, scale, box_min_frac,
                  domain_lo, domain_hi, box_center, incumbent, rng=None):
    """Build the next segment's sampling box.

    final_population : (NP, D) previous segment's final population
    scale            : half-width multiplier for this segment. The caller
                       resolves it: a discrete action space looks the index up
                       in episode.box_scales; a continuous one passes the
                       agent's scale float directly. Decoupling the lookup from
                       the geometry lets both action shapes share this transform.
    box_min_frac     : collapse floor as a fraction of the domain half-width
                       (episode.box_min_frac, 1/24 ≈ 0.0417)
    domain_lo, domain_hi : (D,) true search domain, for the floor and the clip
    box_center       : 'centroid' | 'incumbent' | 'random'. The first two are
                       deterministic; 'random' places the center at a uniform
                       point in the domain (a learned diversification / restart
                       move) and therefore requires `rng`.
    incumbent        : (D,) best-so-far solution; the center when box_center is
                       'incumbent', and always the point tested for containment
    rng              : a numpy Generator, required only for box_center='random'
    """
    pop = np.asarray(final_population, dtype=np.float64)
    domain_lo = np.asarray(domain_lo, dtype=np.float64)
    domain_hi = np.asarray(domain_hi, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)

    # 1. per-dimension half-width of the population
    half_width = (pop.max(axis=0) - pop.min(axis=0)) / 2.0

    # 2. per-dimension center
    if box_center == "centroid":
        center = pop.mean(axis=0)
    elif box_center == "incumbent":
        center = incumbent.copy()
    elif box_center == "random":
        if rng is None:
            raise ValueError("box_center='random' requires an rng.")
        center = domain_lo + rng.random(domain_lo.shape) * (domain_hi - domain_lo)
    else:
        raise ValueError(
            f"box_center must be 'centroid', 'incumbent' or 'random', "
            f"got {box_center!r}."
        )

    # 3. scale the half-width by the action's factor
    scaled = scale * half_width

    # 4. floor: never collapse below box_min_frac of the domain half-width
    domain_half_width = (domain_hi - domain_lo) / 2.0
    floor = box_min_frac * domain_half_width
    box_at_floor = bool(np.any(scaled < floor))
    scaled = np.maximum(scaled, floor)

    # 5. clip the resulting box to the domain
    box_lo = np.clip(center - scaled, domain_lo, domain_hi)
    box_hi = np.clip(center + scaled, domain_lo, domain_hi)

    incumbent_in_box = bool(np.all(incumbent >= box_lo)
                            and np.all(incumbent <= box_hi))

    return SamplingBox(
        box_lo=box_lo.astype(np.float32),
        box_hi=box_hi.astype(np.float32),
        center=center.astype(np.float32),
        half_width=scaled.astype(np.float32),
        box_at_floor=box_at_floor,
        incumbent_in_box=incumbent_in_box,
    )


def transform_covariance(final_population, scale, box_min_frac,
                         domain_lo, domain_hi, box_center, incumbent, rng=None):
    """Covariance sampling box (spec §6.1 variant).

    Same concept as transform_box -- the agent expands/contracts the region the
    next population is drawn from -- but the region is now an ORIENTED ELLIPSOID
    carrying the covariance of the previous segment's final population, not an
    axis-aligned box. The next population is sampled as

        x = center + z @ chol.T ,   z ~ N(0, I)

    where `chol = scale * L`, `L = cholesky(Sigma_reg)`, so `scale` multiplies the
    STANDARD DEVIATION (a 2x scale doubles the ellipsoid's radius) -- matching
    transform_box's linear box-scale semantics. box_lo/box_hi are the +/-1-sigma
    bounding box, exposed only for the observation's box features.

    Sigma_reg is the population covariance with an additive per-dimension variance
    floor of (box_min_frac * domain_half_width)^2, which keeps it positive-definite
    (Cholesky-able) and stops the ellipsoid collapsing below the same floor the
    axis box uses. `center` follows the same centroid/incumbent/random choice.
    """
    pop = np.asarray(final_population, dtype=np.float64)
    domain_lo = np.asarray(domain_lo, dtype=np.float64)
    domain_hi = np.asarray(domain_hi, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    d = pop.shape[1]
    half = (domain_hi - domain_lo) / 2.0

    if box_center == "centroid":
        center = pop.mean(axis=0)
    elif box_center == "incumbent":
        center = incumbent.copy()
    elif box_center == "random":
        if rng is None:
            raise ValueError("box_center='random' requires an rng.")
        center = domain_lo + rng.random(d) * (domain_hi - domain_lo)
    else:
        raise ValueError(
            f"box_center must be 'centroid', 'incumbent' or 'random', "
            f"got {box_center!r}.")

    sigma_mat = (np.cov(pop, rowvar=False) if pop.shape[0] > 1
                 else np.zeros((d, d)))
    sigma_mat = np.atleast_2d(sigma_mat)
    floor_var = (box_min_frac * half) ** 2                 # (D,) min variance
    box_at_floor = bool(np.any(np.diag(sigma_mat) < floor_var))
    sigma_mat = sigma_mat + np.diag(floor_var)             # additive variance floor
    L = np.linalg.cholesky(sigma_mat)                      # (D,D) lower-triangular
    chol = scale * L                                       # scale the std (spread)

    sd = scale * np.sqrt(np.diag(sigma_mat))               # +/-1-sigma per dim
    box_lo = np.clip(center - sd, domain_lo, domain_hi)
    box_hi = np.clip(center + sd, domain_lo, domain_hi)
    incumbent_in_box = bool(np.all(incumbent >= box_lo)
                            and np.all(incumbent <= box_hi))

    return SamplingBox(
        box_lo=box_lo.astype(np.float32),
        box_hi=box_hi.astype(np.float32),
        center=center.astype(np.float32),
        half_width=sd.astype(np.float32),
        box_at_floor=box_at_floor,
        incumbent_in_box=incumbent_in_box,
        chol=chol.astype(np.float32),
    )
