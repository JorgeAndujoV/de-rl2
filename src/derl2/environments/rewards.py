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

import math

ERROR_FLOOR = 1e-8


def _clamp(error):
    """CEC'13: errors below 1e-8 are treated as solved (zero)."""
    return 0.0 if error < ERROR_FLOOR else error


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

    def __init__(self, lam=0.1, tau_stag=3):
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
    """Log-improvement variant of Stagnation (EXP004).

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

    def __init__(self, lam=0.1, tau_stag=3, floor=ERROR_FLOOR):
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


# --------------------------------------------------------------- registry

REWARDS = {
    "stagnation": Stagnation,
    "log_stagnation": LogStagnation,
}


def build_reward(name, **params):
    if name not in REWARDS:
        raise KeyError(
            f"Unknown reward {name!r}. Available: {sorted(REWARDS)}. "
            f"Add it to REWARDS in src/derl2/environments/rewards.py."
        )
    return REWARDS[name](**params)
