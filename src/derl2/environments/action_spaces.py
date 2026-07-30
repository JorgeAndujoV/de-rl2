"""Action spaces (spec §6.3).

An action space decodes an agent's own encoding into a config dict the
environment consumes; the environment never sees an action index (principle 7).
This is what lets a continuous-action agent be added later without touching the
environment: it would emit a different encoding and its own ActionSpace would
decode it to the same dict shape.

Adding a variant is a new class plus one registry line.
"""


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

    def __init__(self, strategy_profiles, budget_fracs):
        self.profiles = [
            (p["strategy"], float(p["F"]), float(p["CR"]))
            for p in strategy_profiles
        ]
        self.budget_fracs = [float(b) for b in budget_fracs]
        self.n = len(self.profiles) * len(self.budget_fracs) * self.N_BOXES

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


# --------------------------------------------------------------- registry

ACTION_SPACES = {
    "profiles_budget_area": ProfilesBudgetArea,
}


def build_action_space(name, strategy_profiles, budget_fracs):
    if name not in ACTION_SPACES:
        raise KeyError(
            f"Unknown action space {name!r}. Available: {sorted(ACTION_SPACES)}. "
            f"Add it to ACTION_SPACES in src/derl2/environments/action_spaces.py."
        )
    return ACTION_SPACES[name](strategy_profiles, budget_fracs)
