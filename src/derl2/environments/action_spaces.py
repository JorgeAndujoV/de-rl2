"""Action spaces (spec §6.3).

An action space decodes an agent's own encoding into a config dict the
environment consumes; the environment never sees an action index (principle 7).
This is what lets a continuous-action agent be added later without touching the
environment: it would emit a different encoding and its own ActionSpace would
decode it to the same dict shape.

Adding a variant is a new class plus one registry line.

Every action space exposes a small uniform surface the environment and the
baselines rely on:
  * ``decode(action)``    -> the {strategy, F, CR, budget_frac, sampling_box}
                            config dict;
  * ``n``                 -> number of DISCRETE actions (a full menu for the
                            discrete space; the strategy count for a
                            parameterized one);
  * ``profiles``          -> list of (strategy, F, CR) triples; the environment
                            reads only position 0 (the strategy name) for the
                            observation's strategy one-hot;
  * ``budget_min``        -> the smallest budget fraction the space can request,
                            which the environment uses as its SMDP tau unit;
  * ``continuous_box``    -> whether ``decoded["sampling_box"]`` is a continuous
                            scale (True) or an index into episode.box_scales
                            (False);
  * ``sample_random(rng)``-> a uniform random action, for the random baseline
                            (which must match each experiment's own space).
"""

import numpy as np


class ProfilesBudgetArea:
    """Discrete space of n = |profiles| × |budget_fracs| × 5 actions.

    Three independent choices are packed into one index:
      * strategy profile — a (strategy, F, CR) triple, declared in the config
        (environment.strategy_profiles), not hard-coded here;
      * budget_frac — from environment.budget_fracs;
      * sampling_box — index 0..4 (the five box scales of §6.1).

    An index `raw` in [0, n) decomposes as

        strategy_idx = raw // (n_budget * 5)
        budget_idx   = (raw // 5) % n_budget
        box_idx      = raw % 5

    so the 5 box indices vary fastest, then budget, then strategy.
    """

    name = "profiles_budget_area"
    N_BOXES = 5
    continuous_box = False               # sampling_box is an index into box_scales

    def __init__(self, strategy_profiles, budget_fracs, **_ignored):
        self.profiles = [
            (p["strategy"], float(p["F"]), float(p["CR"]))
            for p in strategy_profiles
        ]
        self.budget_fracs = [float(b) for b in budget_fracs]
        self.n = len(self.profiles) * len(self.budget_fracs) * self.N_BOXES
        self.budget_min = min(self.budget_fracs)   # SMDP tau unit reference

    def decode(self, raw):
        """Map an action index to the environment's config dict."""
        raw = int(raw)
        if not 0 <= raw < self.n:
            raise IndexError(
                f"action {raw} out of range for {self.name} with n={self.n}."
            )
        n_budget = len(self.budget_fracs)
        strategy_idx = raw // (n_budget * self.N_BOXES)
        budget_idx = (raw // self.N_BOXES) % n_budget
        box_idx = raw % self.N_BOXES

        strategy, F, CR = self.profiles[strategy_idx]
        return {
            "strategy": strategy,
            "F": F,
            "CR": CR,
            "budget_frac": self.budget_fracs[budget_idx],
            "sampling_box": box_idx,
        }

    def sample_random(self, rng):
        """A uniform random action index."""
        return int(rng.integers(self.n))


class ParamStrategyContinuous:
    """Parameterized action space for MP-DQN / continuous-parameter agents.

    Discrete backbone: the strategy (K profiles, declared in
    environment.strategy_profiles — only the strategy NAME is read; F/CR there
    are ignored, the agent sets them). Continuous parameters attached to the
    chosen strategy: F, CR, budget_frac, box_scale — each emitted by the agent
    as a raw value in [0, 1] and affine-mapped to its configured range here.

    An action is the pair ``(k, raw_params)``: ``k`` the strategy index in
    [0, K), and ``raw_params`` a length-4 array [F, CR, budget, box] in [0,1]^4.
    The environment consumes the same {strategy, F, CR, budget_frac,
    sampling_box} dict as any other space; sampling_box is a continuous box-scale
    multiplier (continuous_box=True), not an index, so the environment passes it
    straight to transform_box.
    """

    name = "param_strategy_continuous"
    PARAM_NAMES = ("F", "CR", "budget_frac", "box_scale")
    continuous_box = True

    def __init__(self, strategy_profiles, budget_fracs=None, *,
                 f_range=(0.0, 1.0), cr_range=(0.0, 1.0),
                 budget_range=(0.05, 0.95), box_scale_range=(0.25, 3.0),
                 **_ignored):
        self.strategies = [p["strategy"] for p in strategy_profiles]
        # profiles kept for the env's strategy one-hot (only position 0, the
        # strategy name, is read); the F/CR placeholders are never used.
        self.profiles = [(s, 0.0, 0.0) for s in self.strategies]
        # per-parameter (lo, hi), in PARAM_NAMES order.
        self.ranges = [tuple(map(float, r)) for r in
                       (f_range, cr_range, budget_range, box_scale_range)]
        self.n = len(self.strategies)          # discrete actions = strategies
        self.param_dim = len(self.PARAM_NAMES)  # continuous params per action
        self.budget_min = float(budget_range[0])   # SMDP tau unit reference

    def _scale(self, raw):
        """Affine-map raw params in [0,1] to their configured ranges (clamped)."""
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw.shape[0] != self.param_dim:
            raise ValueError(
                f"{self.name} expects {self.param_dim} params, got {raw.shape[0]}."
            )
        out = []
        for v, (lo, hi) in zip(raw, self.ranges):
            v = min(max(float(v), 0.0), 1.0)
            out.append(lo + v * (hi - lo))
        return out

    def decode(self, action):
        """action = (strategy_idx, raw_params[4]) -> env config dict."""
        k, raw = action
        k = int(k)
        if not 0 <= k < self.n:
            raise IndexError(
                f"strategy index {k} out of range [0,{self.n}) for {self.name}."
            )
        F, CR, budget_frac, box_scale = self._scale(raw)
        return {
            "strategy": self.strategies[k],
            "F": F,
            "CR": CR,
            "budget_frac": budget_frac,
            "sampling_box": box_scale,       # continuous scale (continuous_box)
        }

    def sample_random(self, rng):
        """A uniform random action in this space (strategy + [0,1]^4 params),
        for the per-experiment random baseline."""
        return (int(rng.integers(self.n)), rng.random(self.param_dim))


# --------------------------------------------------------------- registry

ACTION_SPACES = {
    "profiles_budget_area": ProfilesBudgetArea,
    "param_strategy_continuous": ParamStrategyContinuous,
}


def build_action_space(name, strategy_profiles, budget_fracs, **kwargs):
    if name not in ACTION_SPACES:
        raise KeyError(
            f"Unknown action space {name!r}. Available: {sorted(ACTION_SPACES)}. "
            f"Add it to ACTION_SPACES in src/derl2/environments/action_spaces.py."
        )
    return ACTION_SPACES[name](strategy_profiles, budget_fracs, **kwargs)
