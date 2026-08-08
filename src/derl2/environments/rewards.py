"""Reward functions (spec §6.4).

Only the stagnation-aware reward is implemented; the normalized and standard
variants of de-rl are deliberately not ported. Reward design is a primary axis
of experimentation, so rewards are selected by name from this registry.

A reward is built with its config parameters and then called once per step with
a context dict:

    t                step index, 1-based (t == 1 is the first agent decision,
                     the segment immediately after warmup)
    error_best       running global best error across the episode, BEFORE the
                     segment just completed is folded in (= e_{t-1}; at t == 1
                     this is the post-warmup error e_0)
    error_new        running global best error AFTER the segment is folded in
                     (= min(error_best, this segment's best error) = e_t)
    n_stag           consecutive segments with no global-best improvement
                     ENTERING this step (the streak the current segment's
                     outcome is judged against, i.e. n_stag_before). It must be
                     the pre-update streak: on an improving step the streak
                     resets to 0, so passing the post-update value would make
                     the λ penalty — which only applies on improving steps —
                     impossible to ever trigger. The environment logs both the
                     before and after streaks; the observation's B3 uses after.

Errors below 1e-8 are taken as zero (CEC'13 protocol).
"""

import csv
import math
import os
import statistics

ERROR_FLOOR = 1e-8


def _clamp(error):
    """CEC'13: errors below 1e-8 are treated as solved (zero)."""
    return 0.0 if error < ERROR_FLOOR else error


def _base001_median(function_id, exp_id, is_smoke):
    """m_i for the median-normalized reward: the median final error of
    BASE001_de_plain (plain DE) on this function, read from the frozen baseline.

    The baseline is generated BEFORE training (run_experiment.sh) and BASE001 is
    produced first and without the environment (run_segment, no reward), so by
    the time any env-driven reward is evaluated the file exists. Loaded lazily
    and cached per function by the reward, so building the env never touches the
    file (which would race baseline generation). The import is function-local to
    avoid an environments -> evaluation import cycle."""
    from derl2.evaluation.evaluate import baseline_dir
    path = os.path.join(
        baseline_dir("BASE001_de_plain", function_id, exp_id, is_smoke),
        "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"median_stagnation needs BASE001_de_plain for f{function_id} "
            f"(experiment {exp_id!r}) but {path} is missing. Baselines must be "
            f"generated before training — run_experiment.sh does this at job "
            f"start; BASE001 must precede the median reward.")
    errors = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            errors.append(float(row["best_error"]))
    if not errors:
        raise ValueError(f"BASE001 results for f{function_id} are empty ({path}).")
    median = statistics.median(errors)
    if median <= 0:
        raise ValueError(
            f"BASE001 median error for f{function_id} is {median} (<= 0); "
            f"cannot use it to normalize the reward.")
    return float(median)


class Stagnation:
    """Stagnation-aware reward (spec §6.4, corrected).

        R_t = (error_best − error_new) − λ·I[n_stag ≥ τ]·improved   (every t)
            +  (−error_best)                                        (only at t = 1)

    where improved = error_new < error_best. Because the global best is
    monotone (error_new = min(prev best, this segment's best)), the improvement
    term is ≥ 0 and the per-step differences telescope: Σ R = −error_final
    (before penalties), so maximizing return is minimizing final error.

    The literal §6.4 form replaced the first step's reward with −e₀ instead of
    adding it, which dropped the first segment's improvement (e₀ − e₁) from the
    return entirely — the first agent decision got no credit for what it
    achieved. Treating −e₀ as an additive one-off restores that credit and
    makes the telescoping exact. The additive constant −e₀ is fixed by the
    warmup, so it is an uncontrollable term on the step-1 transition, present
    only for return interpretability; step-1 Q-values are expected to be noisy.

    (With γ^τ SMDP discounting the telescoping is only approximate — the
    ordering of returns is preserved, which is what matters, but logged episode
    return will not equal −error_final numerically.)

    The "0 otherwise" branch of §6.4 is redundant and dropped: no improvement
    means error_new == error_best and the difference is already 0; the branch
    only ever gated the penalty, which now gates on `improved` directly.
    """

    name = "stagnation"

    def __init__(self, lam=0.1, tau_stag=3, **_ignored):
        self.lam = float(lam)
        self.tau_stag = int(tau_stag)

    def __call__(self, ctx):
        error_best = _clamp(ctx["error_best"])
        error_new = _clamp(ctx["error_new"])

        improved = error_new < error_best
        penalty = self.lam if (improved and ctx["n_stag"] >= self.tau_stag) else 0.0
        reward = (error_best - error_new) - penalty
        if ctx["t"] == 1:
            reward += -error_best
        return float(reward)


class LogStagnation:
    """Log-improvement variant of Stagnation.

        R_t = (log10(e_{t-1}+c) − log10(e_t+c)) − λ·I[n_stag ≥ τ]·improved   (every t)
            +  (−log10(e_0+c))                                              (only t = 1)

    Same shape as Stagnation, but the per-segment reward is the reduction in
    log10(error) instead of raw error. This makes a fixed FACTOR reduction worth
    the same reward regardless of the absolute scale: an order of magnitude early
    (1e7 → 1e6) earns the same +1 as one late (1e2 → 1e1). Two consequences:

      * late-episode fine-tuning finally gets credit equal to early gains, so the
        agent learns to value the whole trajectory rather than only segment 1
        (under the raw reward the first segment dominates the return by ~5 orders
        of magnitude on ill-conditioned functions);
      * TD targets are bounded to a small range (~0–16) instead of ~0–1e7, which
        is far kinder to the DQN's stability.

    It preserves the telescoping objective: with the t=1 term, Σ R = −log10(e_final+c)
    (before penalties), so maximizing return still minimizes final error — the
    ordering over policies is identical to Stagnation's, only the scale changes.
    c = ERROR_FLOOR keeps log finite when a segment solves the problem (e → 0).

    NOTE: because rewards are now O(1) rather than O(1e6), the λ penalty (default
    0.1) is on a comparable scale to a per-step reward for the first time — it is
    an active term here, not the effectively-inert one it was under raw error.
    """

    name = "log_stagnation"

    def __init__(self, lam=0.1, tau_stag=3, floor=ERROR_FLOOR, **_ignored):
        self.lam = float(lam)
        self.tau_stag = int(tau_stag)
        self.floor = float(floor)

    def __call__(self, ctx):
        error_best = _clamp(ctx["error_best"])
        error_new = _clamp(ctx["error_new"])
        log_best = math.log10(error_best + self.floor)
        log_new = math.log10(error_new + self.floor)

        improved = error_new < error_best
        penalty = self.lam if (improved and ctx["n_stag"] >= self.tau_stag) else 0.0
        reward = (log_best - log_new) - penalty
        if ctx["t"] == 1:
            reward += -log_best
        return float(reward)


class LogImprovement:
    """Log-improvement reward WITHOUT the stagnation penalty.

        R_t = log10(e_{t-1}+c) − log10(e_t+c)          (every t)
            + (−log10(e_0+c))                          (only t = 1)

    Identical to LogStagnation with λ = 0. Dropping the penalty is not just a
    simplification: it makes the reward a pure potential-based shaping of
    Φ = log10(error) (Ng, Harada & Russell 1999). Its telescoped return is
    exactly −log10(e_final+c), so it has the SAME optimal policy as the sparse
    terminal reward — the shaping adds no bias, it only redistributes credit
    across the episode. The λ penalty is the single term that breaks that
    invariance; removing it gives the cleanest well-posed reward of the set.
    (Wasteful restarts are still implicitly discouraged: a finite budget spent
    on a no-improvement segment is budget denied to a productive one.)
    """

    name = "log_improvement"

    def __init__(self, floor=ERROR_FLOOR, **_ignored):
        self.floor = float(floor)

    def __call__(self, ctx):
        error_best = _clamp(ctx["error_best"])
        error_new = _clamp(ctx["error_new"])
        reward = (math.log10(error_best + self.floor)
                  - math.log10(error_new + self.floor))
        if ctx["t"] == 1:
            reward += -math.log10(error_best + self.floor)
        return float(reward)


class NormInitial:
    """Linear improvement normalized by the episode's INITIAL error (EXP006).

        R_t = (e_{t-1} − e_t) / (e_0 + c)

    e_0 is the post-warmup error, captured at t = 1. The per-episode return is
    Σ R_t = (e_0 − e_final)/(e_0 + c) ∈ [0, 1] — the FRACTION of the initial
    error removed — so it is scale-free per episode and bounded regardless of
    the function's magnitude. There is no penalty and no additive −e_0 term
    (unlike Stagnation): the reward is purely the normalized improvement, and
    maximizing it still minimizes e_final (e_0 is fixed within an episode).

    Contrast with the log rewards: this keeps LINEAR improvement, so an early
    order-of-magnitude drop (when e ≈ e_0) still dominates the return and late
    fine-tuning (once e ≪ e_0) earns almost nothing. It fixes cross-function
    scale but NOT the within-episode first-segment dominance — the cheap
    comparison arm against the log reward.

    The reward object is stateful across a single episode (it remembers e_0 from
    t = 1); every episode begins at t = 1, which refreshes e_0, and the env is
    stepped sequentially, so a single reward instance is reused safely.
    """

    name = "norm_initial"

    def __init__(self, floor=ERROR_FLOOR, **_ignored):
        self.floor = float(floor)
        self._e0 = None

    def __call__(self, ctx):
        error_best = _clamp(ctx["error_best"])
        error_new = _clamp(ctx["error_new"])
        if ctx["t"] == 1:
            self._e0 = error_best              # e_0 = post-warmup error
        e0 = self._e0 if self._e0 is not None else error_best
        return float((error_best - error_new) / (e0 + self.floor))


class MedianStagnation:
    """Median-normalized reward (de-rl paper Eq. 4) (EXP008).

        R_t = (e_{t-1} − e_t) / m_i                    (every t)
            + (−e_{t-1} / m_i)                         (only t = 1)

    where m_i is a FIXED per-function constant: the median final error of
    BASE001_de_plain on function i, computed beforehand from the frozen baseline
    (loaded lazily, cached per function). Normalizing by a per-function constant
    makes rewards comparable in magnitude across functions of very different
    scale (f2 ~1e5, f11 ~1e1) while keeping LINEAR improvement.

    DEVIATION FROM THE LITERAL Eq. (4), matching this codebase's Stagnation
    convention (see its docstring): the paper REPLACES the t = 1 reward with
    −e_0/m_i, which drops — and actually negates — the first segment's
    improvement (e_0 − e_1)/m_i from the return, so the telescoped return is
    −e_final/m_i − (e_0 − e_1)/m_i and the agent gets NEGATIVE credit for a good
    first restart (a term it controls). Adding −e_0/m_i instead of replacing
    makes the return telescope exactly to −e_final/m_i, isolating "median
    normalization" as the only difference from Stagnation — which is the point
    of an ablation. Switch to the literal paper form (replace, not add) if a
    paper-faithful replication is wanted; here the clean-ablation form is used,
    consistent with Stagnation.

    Contrast with NormInitial: m_i is a per-FUNCTION constant (returns are
    comparable across episodes and functions, unbounded), whereas NormInitial's
    e_0 is per-EPISODE (returns bounded to [0,1] but not comparable across
    episodes with different warm-up draws).
    """

    name = "median_stagnation"

    def __init__(self, exp_id=None, is_smoke=False, floor=ERROR_FLOOR,
                 **_ignored):
        self.exp_id = exp_id
        self.is_smoke = bool(is_smoke)
        self.floor = float(floor)
        self._median = {}                      # function_id -> m_i (lazy)

    def _m(self, function_id):
        if function_id not in self._median:
            self._median[function_id] = _base001_median(
                function_id, self.exp_id, self.is_smoke)
        return self._median[function_id]

    def __call__(self, ctx):
        error_best = _clamp(ctx["error_best"])
        error_new = _clamp(ctx["error_new"])
        m = self._m(int(ctx["function_id"]))
        reward = (error_best - error_new) / m
        if ctx["t"] == 1:
            reward += -error_best / m
        return float(reward)


# --------------------------------------------------------------- registry

REWARDS = {
    "stagnation": Stagnation,
    "log_stagnation": LogStagnation,
    "log_improvement": LogImprovement,
    "norm_initial": NormInitial,
    "median_stagnation": MedianStagnation,
}


def build_reward(name, *, lam=0.1, tau_stag=3, exp_id=None, is_smoke=False,
                 floor=ERROR_FLOOR):
    """Build a reward by name. All known parameters are passed to every reward;
    each class keeps what it needs and swallows the rest (**_ignored), so the
    env has one call site regardless of which reward a config selects."""
    if name not in REWARDS:
        raise KeyError(
            f"Unknown reward {name!r}. Available: {sorted(REWARDS)}. "
            f"Add it to REWARDS in src/derl2/environments/rewards.py."
        )
    return REWARDS[name](lam=lam, tau_stag=tau_stag, exp_id=exp_id,
                         is_smoke=is_smoke, floor=floor)
