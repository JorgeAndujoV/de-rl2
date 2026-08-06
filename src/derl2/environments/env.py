"""The episode environment (spec §1, §6.5).

One episode is one complete optimization of one CEC'13 function under a fixed FE
budget, run as a *sequence of independent DE segments*:

  warm-up segment (fixed config, full-domain init, 5% of budget)
    -> observation
    -> agent action -> {strategy, F, CR, budget_frac, sampling_box}
    -> transform the previous segment's final-population box by sampling_box
    -> sample a fresh NP-population inside that box, run DE for budget_frac
    -> observation, reward, tau
  ... repeat until the budget is exhausted.

There is no elitism: the incumbent is never injected into a new population. The
environment nonetheless tracks the best-ever solution across the episode, for
the reward and for the reported result.

Dependency direction (principle 4): this module knows the optimizer
(`run_segment`) and the observation / action-space / reward registries; it does
NOT know about agents. It never runs a generation loop — the optimizer's public
surface is a whole segment (principle 8) — and it never sees a raw action's
meaning: the ActionSpace decodes the agent's encoding into a config dict, which
is the only action interface (principle 7).

There is deliberately no environment registry: variants are expressed through
the observation / action-space / reward registries and config flags, so this is
constructed directly (via `DEEnv.from_config`) rather than looked up by name.
"""

import numpy as np

from derl2.benchmarks import build_benchmark
from derl2.environments.action_spaces import build_action_space
from derl2.environments.observations import build_observation
from derl2.environments.rewards import build_reward
from derl2.environments.sampling_box import transform_box

# "all" resolves to the full implemented CEC'13 set (f1–f28, including the
# composition functions f21–f28; see benchmarks/cec13.py).
_ALL_FUNCTIONS = list(range(1, 29))


class DEEnv:
    def __init__(self, *, suite, functions, dim, budget, pop_size, warmup_frac,
                 warmup, box_center, box_scales, box_min_frac, elitism,
                 truncate_last_segment, observation, n_checkpoints,
                 action_space, budget_fracs, strategy_profiles, reward,
                 budget_range=None, box_scale_range=None,
                 f_range=None, cr_range=None,
                 exp_id=None, is_smoke=False):
        if elitism:
            raise ValueError(
                "elitism=true is not implemented: this environment never "
                "injects the incumbent into a new population (spec §1)."
            )
        self.suite = suite
        self.functions = (_ALL_FUNCTIONS if functions == "all"
                          else [int(f) for f in functions])
        self.dim = dim
        self.budget = int(budget)
        self.pop_size = pop_size
        self.warmup_frac = warmup_frac
        self.warmup = warmup                       # {strategy, F, CR}
        self.box_center = box_center
        self.box_scales = list(box_scales)
        self.box_min_frac = box_min_frac
        self.truncate_last_segment = truncate_last_segment
        self.n_checkpoints = n_checkpoints
        self.observation = build_observation(observation)
        # Range kwargs are only meaningful to a continuous action space; a
        # discrete one ignores them. Pass through only the ones the config set.
        _as_kwargs = {}
        for _name, _val in (("f_range", f_range), ("cr_range", cr_range),
                            ("budget_range", budget_range),
                            ("box_scale_range", box_scale_range)):
            if _val is not None:
                _as_kwargs[_name] = _val
        self.action = build_action_space(action_space, strategy_profiles,
                                         budget_fracs, **_as_kwargs)

        # SMDP tau unit: the SMALLEST budget fraction THIS experiment's action
        # space can request, so tau is self-consistent within the experiment
        # (one smallest segment = 1 tau). Read from the action space so a
        # discrete menu (min of the list) and a continuous budget range (its
        # lower bound) are handled uniformly; each experiment therefore sets its
        # own discount unit. Used by the reward/tau block in step().
        self.tau_frac = self.action.budget_min
        # exp_id/is_smoke are threaded through only for rewards that need to read
        # a frozen per-function baseline (median_stagnation -> m_i); every other
        # reward ignores them. lambda/tau_stag default so a reward config that
        # omits them (the penalty-free variants) still builds.
        self.reward_fn = build_reward(
            reward["name"], lam=reward.get("lambda", 0.0),
            tau_stag=reward.get("tau_stag", 3),
            exp_id=exp_id, is_smoke=is_smoke)

        # Strategy-profile name -> one-hot index, for the observation's C3a.
        # Profiles in profiles_budget_area are distinct strategies, so the map
        # is unambiguous; the warmup strategy resolves through it too (None if
        # it is not one of the profiles -> an all-zeros one-hot).
        self._strategy_index = {
            strat: i for i, (strat, _F, _CR) in enumerate(self.action.profiles)
        }

        self.obs_dim = self.observation.dim
        self.n_actions = self.action.n
        self._specs = {}

    @classmethod
    def from_config(cls, cfg):
        """Build the environment from a Config, reading only what it needs."""
        return cls(
            suite=cfg.get("benchmark.name"),
            functions=cfg.get("benchmark.functions"),
            dim=cfg.get("benchmark.dim"),
            budget=cfg.get("benchmark.budget"),
            pop_size=cfg.get("episode.pop_size"),
            warmup_frac=cfg.get("episode.warmup_frac"),
            warmup=cfg.get("episode.warmup"),
            box_center=cfg.get("episode.box_center"),
            box_scales=cfg.get("episode.box_scales"),
            box_min_frac=cfg.get("episode.box_min_frac"),
            elitism=cfg.get("episode.elitism"),
            truncate_last_segment=cfg.get("episode.truncate_last_segment"),
            observation=cfg.get("environment.observation"),
            n_checkpoints=cfg.get("environment.n_checkpoints"),
            action_space=cfg.get("environment.action_space"),
            budget_fracs=cfg.get("environment.budget_fracs", None),
            strategy_profiles=cfg.get("environment.strategy_profiles"),
            reward=cfg.get("environment.reward"),
            budget_range=cfg.get("environment.budget_range", None),
            box_scale_range=cfg.get("environment.box_scale_range", None),
            f_range=cfg.get("environment.f_range", None),
            cr_range=cfg.get("environment.cr_range", None),
            exp_id=cfg.exp_id,
            is_smoke=cfg.is_smoke,
        )

    def describe(self):
        return (f"DEEnv suite={self.suite} functions={self.functions} "
                f"dim={self.dim} budget={self.budget} | obs_dim={self.obs_dim} "
                f"n_actions={self.n_actions}")

    # ------------------------------------------------------------- helpers
    def _spec(self, function_id):
        if function_id not in self._specs:
            self._specs[function_id] = build_benchmark(
                f"{self.suite}:f{function_id}", self.dim)
        return self._specs[function_id]

    def _error(self, fitness):
        """Error f(x) - f(x*), floored at 0 (float noise can dip below)."""
        return max(0.0, float(fitness) - self._optimum)

    def _width_frac(self, lo, hi):
        """Mean over dimensions of (hi - lo) / domain width."""
        return float(np.mean((hi - lo) / self._domain_width))

    # --------------------------------------------------------------- reset
    def reset(self, seed=0, function_id=None):
        """Start an episode: pick a function, run the warm-up segment, and
        return the first observation. Segment seeds derive from `seed`, so a
        given (seed, function) episode is fully reproducible."""
        self._seed_rng = np.random.default_rng(seed)
        if function_id is None:
            function_id = int(self._seed_rng.choice(self.functions))
        self.function_id = function_id

        spec = self._spec(function_id)
        self._objective = spec.evaluate_tf
        self._optimum = spec.optimum
        self._domain_lo = np.full(self.dim, spec.lower, dtype=np.float32)
        self._domain_hi = np.full(self.dim, spec.upper, dtype=np.float32)
        self._domain_width = self._domain_hi - self._domain_lo

        # Warm-up: full-domain init, fixed config, warmup_frac of the budget.
        from derl2.optimizers import run_segment
        warm = run_segment(
            self._objective, self.dim, self.pop_size,
            self._domain_lo, self._domain_hi, self._domain_lo, self._domain_hi,
            self.warmup["strategy"], self.warmup["F"], self.warmup["CR"],
            int(self.warmup_frac * self.budget), self.n_checkpoints,
            seed=int(self._seed_rng.integers(0, 2**31 - 1)),
        )

        self._fes_used = warm.fes_used
        self._global_best_error = self._error(warm.best_fitness)
        self._global_best_solution = warm.best_solution
        self._n_stag = 0
        self._t = 0
        self._prev_segment = warm
        # The warm-up's "sampling box" is the whole domain.
        self._box_lo = self._domain_lo
        self._box_hi = self._domain_hi

        # First observation: the warm-up is the "segment that just finished".
        # Its incumbent-at-start is its own initial random-population best, so
        # B1 reflects how much the warm-up improved on random sampling.
        ctx = {
            "trajectory": warm.trajectory,
            "initial_best_fitness": warm.initial_best_fitness,
            "incumbent_error_start": self._error(warm.initial_best_fitness),
            "segment_best_error": self._global_best_error,
            "improved_flag": 1.0 if self._global_best_error
                             < self._error(warm.initial_best_fitness) else 0.0,
            "n_stag": 0,
            "segment_best_solution": warm.best_solution,
            "box_lo": self._domain_lo,
            "box_hi": self._domain_hi,
            "box_width_full_final": float(np.mean(
                warm.final_box_hi - warm.final_box_lo)),
            "box_width_full_initial": float(np.mean(self._domain_width)),
            "global_best_error": self._global_best_error,
            "budget_remaining_frac": (self.budget - self._fes_used) / self.budget,
            "n_steps_taken": 0,
            "prev_strategy_index": self._strategy_index.get(
                self.warmup["strategy"]),
            "prev_F": self.warmup["F"],
            "prev_CR": self.warmup["CR"],
            "prev_budget_frac": self.warmup_frac,
            "prev_sampling_box_index": None,      # warm-up used no box action
            "prev_box_scale_norm": 0.0,           # ... and no continuous scale
            "current_box_width_frac": 1.0,        # it used the whole domain
        }
        obs = self.observation.build(ctx)
        info = {
            "function_id": function_id,
            "best_error": self._global_best_error,
            "fes_used": self._fes_used,
        }
        return obs, info

    # ---------------------------------------------------------------- step
    def step(self, action):
        from derl2.optimizers import run_segment
        self._t += 1
        decoded = self.action.decode(action)

        # Resolve the box half-width multiplier: a continuous action space
        # emits the scale directly; a discrete one emits an index into
        # box_scales. prev_box_scale_norm is the scale normalized to [0,1] over
        # the continuous range (0.0 for the discrete space, whose observation
        # uses a box one-hot instead and ignores this feature).
        if self.action.continuous_box:
            box_scale = float(decoded["sampling_box"])
            _blo, _bhi = self.action.ranges[3]
            prev_box_scale_norm = (box_scale - _blo) / (_bhi - _blo + 1e-12)
        else:
            box_scale = self.box_scales[int(decoded["sampling_box"])]
            prev_box_scale_norm = 0.0

        # Transform the previous segment's population into this segment's box.
        box = transform_box(
            self._prev_segment.final_population, box_scale, self.box_min_frac,
            self._domain_lo, self._domain_hi,
            self.box_center, self._global_best_solution,
        )

        fes_before = self._fes_used
        remaining = self.budget - fes_before
        requested = int(round(decoded["budget_frac"] * self.budget))
        if requested <= remaining:
            fe_budget = requested
        elif self.truncate_last_segment:
            fe_budget = remaining                 # run the partial last segment
        else:
            fe_budget = requested                 # run full, exhausting budget

        seg = run_segment(
            self._objective, self.dim, self.pop_size,
            box.box_lo, box.box_hi, self._domain_lo, self._domain_hi,
            decoded["strategy"], decoded["F"], decoded["CR"],
            fe_budget, self.n_checkpoints,
            seed=int(self._seed_rng.integers(0, 2**31 - 1)),
        )
        self._fes_used += seg.fes_used

        segment_best_error = self._error(seg.best_fitness)
        error_best_before = self._global_best_error
        n_stag_before = self._n_stag
        improved = segment_best_error < error_best_before
        if improved:
            self._global_best_error = segment_best_error
            self._global_best_solution = seg.best_solution
            self._n_stag = 0
        else:
            self._n_stag = n_stag_before + 1
        error_new = self._global_best_error

        # Reward uses the pre-update streak (see rewards.py); tau is elapsed
        # budget in units of the smallest budget fraction of THIS experiment
        # (self.tau_frac, spec §6.5) — a per-experiment-consistent unit, so the
        # smallest agent segment is exactly 1 tau. Changing the budget menu (e.g.
        # EXP004's 0.025) shifts this unit with it; the γ^τ horizon is therefore
        # a property of each experiment's own configuration.
        reward = self.reward_fn({
            "t": self._t,
            "error_best": error_best_before,
            "error_new": error_new,
            "n_stag": n_stag_before,
            "function_id": self.function_id,     # median_stagnation keys m_i on it
        })
        tau = seg.fes_used / (self.tau_frac * self.budget)

        # Budget exhaustion is a genuine terminal state (spec §6.5): done when
        # the remainder cannot even seed another population.
        terminated = (self.budget - self._fes_used) < self.pop_size
        budget_remaining_frac = max(0.0,
                                    (self.budget - self._fes_used) / self.budget)

        box_width_frac_initial = self._width_frac(box.box_lo, box.box_hi)
        box_width_frac_final = self._width_frac(seg.final_box_lo, seg.final_box_hi)

        ctx = {
            "trajectory": seg.trajectory,
            "initial_best_fitness": seg.initial_best_fitness,
            "incumbent_error_start": error_best_before,
            "segment_best_error": segment_best_error,
            "improved_flag": 1.0 if improved else 0.0,
            "n_stag": self._n_stag,                       # after -> B3
            "segment_best_solution": seg.best_solution,
            "box_lo": box.box_lo,
            "box_hi": box.box_hi,
            "box_width_full_final": float(np.mean(
                seg.final_box_hi - seg.final_box_lo)),
            "box_width_full_initial": float(np.mean(box.box_hi - box.box_lo)),
            "global_best_error": self._global_best_error,
            "budget_remaining_frac": budget_remaining_frac,
            "n_steps_taken": self._t,
            "prev_strategy_index": self._strategy_index.get(decoded["strategy"]),
            "prev_F": decoded["F"],
            "prev_CR": decoded["CR"],
            "prev_budget_frac": decoded["budget_frac"],
            "prev_sampling_box_index": decoded["sampling_box"],
            "prev_box_scale_norm": prev_box_scale_norm,
            "current_box_width_frac": box_width_frac_initial,
        }
        next_obs = self.observation.build(ctx)

        self._prev_segment = seg
        self._box_lo, self._box_hi = box.box_lo, box.box_hi

        # Everything the reporting layer needs is logged here (principle 11);
        # step_trace.csv reads these fields directly.
        info = {
            "function_id": self.function_id,
            "step_idx": self._t,
            "tau": tau,
            "budget_used_before": fes_before,
            "budget_remaining_frac_before": (self.budget - fes_before) / self.budget,
            "fes_used_segment": seg.fes_used,
            "fes_used_total": self._fes_used,
            "strategy": decoded["strategy"],
            "F": decoded["F"],
            "CR": decoded["CR"],
            "budget_frac_action": decoded["budget_frac"],
            "sampling_box_action": decoded["sampling_box"],
            "box_width_frac_initial": box_width_frac_initial,
            "box_width_frac_final": box_width_frac_final,
            "box_at_floor": box.box_at_floor,
            "incumbent_in_box": box.incumbent_in_box,
            "n_stag_before": n_stag_before,
            "n_stag_after": self._n_stag,
            "segment_best_error": segment_best_error,
            "global_best_error_before": error_best_before,
            "global_best_error_after": self._global_best_error,
            "reward": reward,
            "done": terminated,
            "best_error": self._global_best_error,
            # Full (K, 5) raw trajectory, carried for evaluate.py's
            # save_trajectory_seeds figure export; not used by step_trace.
            "trajectory": seg.trajectory,
        }
        # Budget exhaustion is terminal, not a time-limit truncation, so
        # truncated is always False (the "truncate" in the config names the
        # segment-budget clamp, a different thing).
        return next_obs, reward, terminated, False, info
