"""Observation builders (spec §6.2).

Each builder turns the state after a finished segment into a fixed-length
feature vector and declares its own `dim`, so the agent's input size follows
from the config rather than being asserted anywhere. Which features to include
is a primary axis of experimentation, hence the registry: a new observation is
a new class plus one dictionary line, and never touches optimizer or env code
(principle 9 — the optimizer records a raw superset; observations select from
it).

All fitness quantities are log-scaled; all spatial quantities are expressed as
fractions of the domain width. Fitness differences (A3, A4) are taken between
raw checkpoint fitnesses, where the benchmark's optimum offset cancels, so they
equal the corresponding error differences.

The context dict a builder receives from the environment (assembled there so
this module stays decoupled from episode bookkeeping):

    trajectory            (K, 5) raw per-checkpoint statistics from the finished
                          segment (SegmentResult.trajectory); channels are
                          0 box_width_frac, 1 success_rate, 2 best_fitness,
                          3 mean_fitness, 4 fes_used
    initial_best_fitness  best fitness of the segment's freshly sampled
                          population, before generation 1 — the k=1 baseline
                          for the improvement channel A3

    incumbent_error_start global best error BEFORE this segment
    segment_best_error    best error found within this segment
    improved_flag         1.0 if the global best improved this segment, else 0.0
    n_stag                consecutive segments with no global-best improvement
    segment_best_solution (D,) best solution found within this segment
    box_lo, box_hi        (D,) the domain-clipped sampling box the segment's
                          population was actually drawn from (B4 normalizes by
                          this actual region, not the pre-clip requested one)
    box_width_full_final  mean over dims of the final population's full width
    box_width_full_initial mean over dims of the sampling box's full width
    global_best_error     best-so-far error across the episode

    budget_remaining_frac fraction of the FE budget still unspent [0, 1]
    n_steps_taken         number of agent decisions taken so far
    prev_strategy_index   index (0..N_STRATEGIES-1) of the strategy just used
    prev_F, prev_CR       the F/CR just used
    prev_budget_frac      the budget fraction just used
    prev_sampling_box_index index (0..N_BOXES-1) of the box action just used,
                          or None for step 1 (warmup used full-domain init, no
                          box action) -> an all-zeros one-hot
    current_box_width_frac full width of the current sampling box / domain width
"""

import numpy as np

EPS = 1e-12          # tiny guard against division by zero in ratios
ERR_EPS = 1e-8       # error floor for log-ratio features, matching the CEC'13
                     # threshold below which errors are reported as zero


def _one_hot(index, size):
    """One-hot of length `size`; an index of None yields all zeros (used when
    there is no prior box action, i.e. the warmup that precedes step 1)."""
    vec = np.zeros(size, dtype=np.float32)
    if index is not None:
        vec[index] = 1.0
    return vec


class Traj20:
    """101 features in three blocks (spec §6.2).

    Block A (80): the K=20-checkpoint trajectory, four channels, laid out
    channel-major — all 20 values of A1, then A2, A3, A4.
    Block B (6):  segment-outcome summary.
    Block C (15): context (budget, step count, previous action, current box).

    traj20 is a fixed contract tied to a 4-profile / 5-box action space with 20
    checkpoints. A different checkpoint count or action-space shape is a
    different observation variant (a new class and registry line), not a
    reconfiguration of this one.
    """

    name = "traj20"
    K = 20
    N_STRATEGIES = 4
    N_BOXES = 5
    dim = 4 * K + 6 + (6 + N_STRATEGIES + N_BOXES)   # 80 + 6 + 15 = 101

    def build(self, ctx):
        traj = np.asarray(ctx["trajectory"], dtype=np.float64)
        if traj.shape != (self.K, 5):
            raise ValueError(
                f"traj20 expects a ({self.K}, 5) trajectory (set "
                f"environment.n_checkpoints={self.K}); got {traj.shape}."
            )

        box_width = traj[:, 0]
        success = traj[:, 1]
        best = traj[:, 2]
        mean = traj[:, 3]

        # ---- Block A: trajectory, channel-major (A1[0:K], A2, A3, A4) ----
        # A1 box_width[k]      : raw channel 0
        # A2 success_rate[k]   : raw channel 1
        # A3 log_improvement[k]: log10(1 + best[k-1] - best[k]); best is
        #    non-increasing so the argument is >= 1. The k=0 baseline is the
        #    pre-generation initial_best_fitness.
        # A4 mean_best_gap[k]  : log10(1 + mean[k] - best[k]); mean >= best.
        prev_best = np.concatenate(([ctx["initial_best_fitness"]], best[:-1]))
        a3 = np.log10(1.0 + np.maximum(prev_best - best, 0.0))
        a4 = np.log10(1.0 + np.maximum(mean - best, 0.0))
        block_a = np.concatenate([box_width, success, a3, a4]).astype(np.float32)

        # ---- Block B: segment outcome (6) ----
        inc_start = ctx["incumbent_error_start"]
        seg_best = ctx["segment_best_error"]
        # B1: orders-of-magnitude change of the global best this segment, as the
        # log-ratio of errors: positive on improvement, ~0 when flat, negative
        # on regression (a normal outcome here — no elitism, so a restart can
        # land worse than the incumbent). ε = ERR_EPS matches the CEC'13 floor;
        # clipped to [-8, 8] so this feature can't dwarf the [0, 1]-scale ones.
        b1 = float(np.clip(
            np.log10((inc_start + ERR_EPS) / (seg_best + ERR_EPS)), -8.0, 8.0))
        b2 = float(ctx["improved_flag"])
        b3 = min(ctx["n_stag"] / 5.0, 1.0)
        # B4: displacement of the segment best from the center of the box it was
        # actually sampled in (the domain-clipped box, not the requested
        # geometry), normalized per-dimension by that box's half-width and
        # combined as an RMS. 1.0 ~ the best sits at the boundary; clipped to
        # [0, 10] so a floored (tiny) box can't blow the ratio up.
        box_lo = np.asarray(ctx["box_lo"], dtype=np.float64)
        box_hi = np.asarray(ctx["box_hi"], dtype=np.float64)
        center = (box_lo + box_hi) / 2.0
        half = (box_hi - box_lo) / 2.0
        delta = np.asarray(ctx["segment_best_solution"], dtype=np.float64) - center
        b4 = float(np.clip(
            np.sqrt(np.mean((delta / (half + EPS)) ** 2)), 0.0, 10.0))
        # B5: contraction of the box, full width final / full width initial.
        b5 = float(ctx["box_width_full_final"]
                   / (ctx["box_width_full_initial"] + EPS))
        b6 = float(np.log10(ctx["global_best_error"] + 1e-12))
        block_b = np.array([b1, b2, b3, b4, b5, b6], dtype=np.float32)

        # ---- Block C: context (15) ----
        block_c = np.concatenate([
            np.array([ctx["budget_remaining_frac"]], dtype=np.float32),   # C1
            np.array([ctx["n_steps_taken"] / 19.0], dtype=np.float32),    # C2
            _one_hot(ctx["prev_strategy_index"], self.N_STRATEGIES),      # C3a
            np.array([ctx["prev_F"]], dtype=np.float32),                  # C3b
            np.array([ctx["prev_CR"]], dtype=np.float32),                 # C3c
            np.array([ctx["prev_budget_frac"]], dtype=np.float32),        # C3d
            _one_hot(ctx["prev_sampling_box_index"], self.N_BOXES),       # C3e
            np.array([ctx["current_box_width_frac"]], dtype=np.float32),  # C4
        ]).astype(np.float32)

        return np.concatenate([block_a, block_b, block_c]).astype(np.float32)


# --------------------------------------------------------------- registry

OBSERVATIONS = {
    "traj20": Traj20,
}


def build_observation(name):
    if name not in OBSERVATIONS:
        raise KeyError(
            f"Unknown observation {name!r}. Available: {sorted(OBSERVATIONS)}. "
            f"Add it to OBSERVATIONS in src/derl2/environments/observations.py."
        )
    return OBSERVATIONS[name]()
