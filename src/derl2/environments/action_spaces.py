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


class ParamStrategyBoxNP(ParamStrategyContinuous):
    """param_strategy_continuous extended with two new freedoms (EXP013).

    On top of the {strategy, F, CR, budget_frac, box_scale} of the parent it adds:

      * box_center — a discrete choice of WHERE to place the next sampling box:
        'centroid' (previous population mean), 'incumbent' (best-so-far), or
        'random' (a uniform domain point — a learned restart/diversification).
        It is folded into the discrete head rather than made a separate policy
        factor, so a plain Categorical agent needs no change: the discrete index
        is the JOINT (strategy, box_center), n = |strategies| * |box_centers|,
        decoded as ``strategy_idx = k // n_bc``, ``box_center_idx = k % n_bc``.

      * pop_size (NP) — a 5th continuous parameter, LOG-scaled over np_range so
        the agent controls it multiplicatively (equal resolution at NP=20 and
        NP=200). This is per-restart adaptive population sizing; the warm-up NP
        stays fixed (episode.pop_size), only the agent's segments vary it.

    An action is still ``(k, raw_params)`` with ``k`` the joint index and
    ``raw_params`` now length 5 ([F, CR, budget, box, NP] in [0,1]). The env
    reads the extra ``box_center`` and ``pop_size`` keys from the decoded dict
    (older spaces omit them, so the env falls back to its fixed values).
    """

    name = "param_strategy_boxnp"
    PARAM_NAMES = ("F", "CR", "budget_frac", "box_scale", "pop_size")
    continuous_box = True
    DEFAULT_BOX_CENTERS = ("centroid", "incumbent", "random")

    def __init__(self, strategy_profiles, budget_fracs=None, *,
                 f_range=(0.0, 1.0), cr_range=(0.0, 1.0),
                 budget_range=(0.05, 0.95), box_scale_range=(0.25, 3.0),
                 np_range=(16, 400), box_centers=None, **_ignored):
        # Parent sets strategies/profiles/ranges(4)/budget_min for F,CR,budget,box.
        super().__init__(strategy_profiles, budget_fracs,
                         f_range=f_range, cr_range=cr_range,
                         budget_range=budget_range,
                         box_scale_range=box_scale_range)
        self.box_centers = list(box_centers) if box_centers is not None \
            else list(self.DEFAULT_BOX_CENTERS)
        self.np_min, self.np_max = int(np_range[0]), int(np_range[1])
        if self.np_min < 2 or self.np_max <= self.np_min:
            raise ValueError(
                f"np_range must be increasing with np_min >= 2, got {np_range}.")
        self._log_np_min = float(np.log(self.np_min))
        self._log_np_max = float(np.log(self.np_max))
        # NP is scaled separately (log, not affine), so the parent's affine
        # `ranges` list covers only the first four params; param_dim is 5.
        self.param_dim = len(self.PARAM_NAMES)
        # Discrete head is the JOINT (strategy, box_center).
        self.n = len(self.strategies) * len(self.box_centers)

    def np_from_raw(self, v):
        """Log-map a raw [0,1] value to an integer population size."""
        v = min(max(float(v), 0.0), 1.0)
        return int(round(np.exp(self._log_np_min
                                + v * (self._log_np_max - self._log_np_min))))

    def np_norm(self, n):
        """Inverse of np_from_raw (up to rounding): population size -> [0,1], for
        the observation's prev_NP feature. Uses the same log scale."""
        n = float(min(max(n, self.np_min), self.np_max))
        return (np.log(n) - self._log_np_min) \
            / (self._log_np_max - self._log_np_min + 1e-12)

    def decode(self, action):
        """action = (joint_idx, raw_params[5]) -> env config dict."""
        k, raw = action
        k = int(k)
        if not 0 <= k < self.n:
            raise IndexError(
                f"joint index {k} out of range [0,{self.n}) for {self.name}.")
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw.shape[0] != self.param_dim:
            raise ValueError(
                f"{self.name} expects {self.param_dim} params, got {raw.shape[0]}.")
        n_bc = len(self.box_centers)
        strategy_idx = k // n_bc
        box_center = self.box_centers[k % n_bc]
        # First four params affine-scaled by the parent's ranges; NP log-scaled.
        F, CR, budget_frac, box_scale = [
            lo + min(max(float(v), 0.0), 1.0) * (hi - lo)
            for v, (lo, hi) in zip(raw[:4], self.ranges)
        ]
        return {
            "strategy": self.strategies[strategy_idx],
            "box_center": box_center,
            "F": F,
            "CR": CR,
            "budget_frac": budget_frac,
            "sampling_box": box_scale,        # continuous scale (continuous_box)
            "pop_size": self.np_from_raw(raw[4]),
        }

    def sample_random(self, rng):
        """Uniform random action: a joint index + 5 params in [0,1]."""
        return (int(rng.integers(self.n)), rng.random(self.param_dim))


# --------------------------------------------------------------- registry

ACTION_SPACES = {
    "profiles_budget_area": ProfilesBudgetArea,
    "param_strategy_continuous": ParamStrategyContinuous,
    "param_strategy_boxnp": ParamStrategyBoxNP,
}


def build_action_space(name, strategy_profiles, budget_fracs, **kwargs):
    if name not in ACTION_SPACES:
        raise KeyError(
            f"Unknown action space {name!r}. Available: {sorted(ACTION_SPACES)}. "
            f"Add it to ACTION_SPACES in src/derl2/environments/action_spaces.py."
        )
    return ACTION_SPACES[name](strategy_profiles, budget_fracs, **kwargs)
