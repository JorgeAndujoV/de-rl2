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
