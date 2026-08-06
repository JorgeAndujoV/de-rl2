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


def transform_box(final_population, scale, box_min_frac,
                  domain_lo, domain_hi, box_center, incumbent):
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
    box_center       : 'centroid' | 'incumbent' (episode.box_center)
    incumbent        : (D,) best-so-far solution; the center when box_center is
                       'incumbent', and always the point tested for containment
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
    else:
        raise ValueError(
            f"box_center must be 'centroid' or 'incumbent', got {box_center!r}."
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
