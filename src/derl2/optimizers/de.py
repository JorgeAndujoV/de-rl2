"""The optimizer's entire public surface: one function, `run_segment`.

A *segment* is one complete DE run over a fixed FE budget, started from a fresh
population sampled inside a given box and evolved with a single fixed
configuration. The environment strings segments together into an episode; the
optimizer itself has no RL knowledge and never runs a generation loop on the
environment's behalf.

The generation-level mechanics — distinct random index selection, mutation,
binomial crossover with one forced dimension, boundary clipping, and
synchronous selection — are carried over VERBATIM from de-rl's validated
`VectorizedDE._generation`, whose behaviour was statistically validated against
a NumPy reference. They are authoritative and are not re-derived here. The only
two departures from that class, both required by the segment semantics, are:

  1. The initial population is sampled uniformly inside `[box_lo, box_hi]`
     (the sampling box) rather than across the whole domain. The box
     constrains only the *start*; the reachable space is still the domain.
  2. `self.lower` / `self.upper` — the bounds `_generation` clips mutants to —
     are the true search domain `[domain_lo, domain_hi]`, not the sampling box.

Around that core, `run_segment` runs generations until the FE budget is spent
and records a fixed-length raw trajectory (K checkpoints) that observation
modules later select and transform from. The optimizer records; it does not
interpret.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from derl2.optimizers.strategies import build_strategy


@dataclass
class SegmentResult:
    """Everything one segment reports back to the environment."""

    final_population: np.ndarray     # (NP, D) population at segment end
    final_fitness: np.ndarray        # (NP,) its fitnesses
    best_solution: np.ndarray        # (D,) best found within this segment
    best_fitness: float
    initial_best_fitness: float      # best of the freshly sampled population,
                                     # before generation 1; the k=1 baseline
                                     # for the trajectory's improvement channel
    fes_used: int                    # actual FEs consumed
    generations: int
    trajectory: np.ndarray           # (K, 5) raw per-checkpoint statistics
    final_box_lo: np.ndarray         # (D,) bounding box of final_population
    final_box_hi: np.ndarray         # (D,)


class _SegmentDE:
    """One population evolved with the validated DE generation.

    `box_lo`/`box_hi` bound the initial sampling; `domain_lo`/`domain_hi` are
    the clip bounds used during evolution. Everything else matches de-rl's
    VectorizedDE.
    """

    def __init__(self, objective, dim, pop_size, box_lo, box_hi,
                 domain_lo, domain_hi, strategy, seed):
        self.objective = objective
        self.dim = dim
        self.pop_size = pop_size
        # Clip bounds during evolution are the true domain.
        self.lower = tf.constant(domain_lo, dtype=tf.float32)
        self.upper = tf.constant(domain_hi, dtype=tf.float32)
        # Initial sampling is confined to the box.
        self.box_lo = tf.constant(box_lo, dtype=tf.float32)
        self.box_hi = tf.constant(box_hi, dtype=tf.float32)
        self.strategy = build_strategy(strategy)

        self._default_strategy_name = strategy
        self._strategy_cache = {strategy: self.strategy}

        self._require_pop_size(self.strategy)

        self.gen = tf.random.Generator.from_seed(seed)
        self.population = tf.Variable(
            tf.zeros((pop_size, dim), dtype=tf.float32), trainable=False
        )
        self.fitness = tf.Variable(
            tf.zeros((pop_size,), dtype=tf.float32), trainable=False
        )
        self.reset(seed)

    def _require_pop_size(self, strategy):
        """Fail loudly if the population is too small for `strategy`."""
        if self.pop_size < strategy.n_random + 1:
            raise ValueError(
                f"pop_size {self.pop_size} too small for strategy "
                f"{strategy.name}, which needs "
                f"{strategy.n_random} distinct indices plus the target."
            )

    def _get_strategy(self, name):
        """Resolve a strategy name to a cached Strategy, building lazily."""
        strat = self._strategy_cache.get(name)
        if strat is None:
            strat = build_strategy(name)
            self._strategy_cache[name] = strat
        return strat

    def reset(self, seed=None):
        """Sample a fresh population uniformly inside the sampling box.

        This is VectorizedDE.reset with the initial draw confined to
        [box_lo, box_hi] instead of the full domain — the one intended change
        to the validated behaviour (see module docstring)."""
        if seed is not None:
            self.gen.reset_from_seed(seed)
        init = self.gen.uniform(
            (self.pop_size, self.dim),
            minval=self.box_lo, maxval=self.box_hi, dtype=tf.float32,
        )
        self.population.assign(init)
        self.fitness.assign(self.objective(init))
        self.evals = self.pop_size
        self.best_fitness = float(tf.reduce_min(self.fitness).numpy())
        best = int(tf.argmin(self.fitness).numpy())
        self.best_solution = self.population.numpy()[best].copy()

    @tf.function
    def _generation(self, F, CR, strategy_name):
        # Carried over VERBATIM from VectorizedDE._generation. Clipping uses
        # self.lower/self.upper, which here are the domain bounds.
        strategy = self._strategy_cache[strategy_name]
        NP, D = self.pop_size, self.dim
        n_rand = strategy.n_random
        x = self.population
        best = tf.gather(x, tf.argmin(self.fitness))

        # Distinct random indices per row, none equal to the row's own index.
        # Take n_rand + 1 from a random permutation: a permutation holds each
        # index exactly once, so at most one of the first n_rand can collide
        # with i, and the spare entry replaces it.
        perm = tf.argsort(self.gen.uniform((NP, NP)), axis=1)
        cand = perm[:, : n_rand + 1]
        sel = cand[:, :n_rand]
        spare = cand[:, n_rand:]
        row = tf.range(NP, dtype=sel.dtype)[:, None]
        sel = tf.where(tf.equal(sel, row), tf.tile(spare, [1, n_rand]), sel)
        idx = [sel[:, k] for k in range(n_rand)]

        mutant = strategy.mutate_tf(x, idx, best, x, F)
        mutant = tf.clip_by_value(mutant, self.lower, self.upper)

        # binomial crossover, one dimension forced from the mutant
        cross = self.gen.uniform((NP, D)) < CR
        j_rand = self.gen.uniform((NP,), maxval=D, dtype=tf.int32)
        cross = tf.logical_or(cross, tf.cast(tf.one_hot(j_rand, D), tf.bool))
        trial = tf.where(cross, mutant, x)

        trial_fitness = self.objective(trial)
        improved = trial_fitness <= self.fitness
        self.population.assign(tf.where(improved[:, None], trial, x))
        self.fitness.assign(tf.where(improved, trial_fitness, self.fitness))

        return tf.reduce_mean(tf.cast(improved, tf.float32))

    def step_generation(self, F, CR, strategy=None):
        """Run one generation. `strategy` overrides the instance default for
        this call only; the single population persists unchanged across a
        switch — only the mutation formula applied to it differs."""
        name = self._default_strategy_name if strategy is None else strategy
        strat = self._get_strategy(name)
        self._require_pop_size(strat)

        success_rate = self._generation(
            tf.constant(F, dtype=tf.float32),
            tf.constant(CR, dtype=tf.float32),
            name,
        )
        self.evals += self.pop_size

        current_best = float(tf.reduce_min(self.fitness).numpy())
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            best = int(tf.argmin(self.fitness).numpy())
            self.best_solution = self.population.numpy()[best].copy()

        return self.stats(success_rate=float(success_rate.numpy()))

    def stats(self, success_rate):
        return {
            "best_fitness": self.best_fitness,
            "mean_fitness": float(tf.reduce_mean(self.fitness).numpy()),
            "success_rate": success_rate,
            "evals": self.evals,
        }


# ----------------------------------------------------------- trajectory
# Raw per-checkpoint channels recorded by run_segment (spec §5). These are
# recorded, not interpreted — observation modules select and transform them.
#   0  box_width_frac  mean over dims of (max - min) / domain width
#   1  success_rate    fraction of trials that replaced their target since
#                      the previous checkpoint
#   2  best_fitness    best fitness within the segment at this checkpoint
#   3  mean_fitness    population mean fitness at this checkpoint
#   4  fes_used        FEs consumed at this checkpoint


def run_segment(objective, dim, pop_size, box_lo, box_hi, domain_lo, domain_hi,
                strategy, F, CR, fe_budget, n_checkpoints, seed):
    """Run one DE segment and return a SegmentResult.

    The initial population evaluation costs `pop_size` FEs and each generation
    costs another `pop_size`, so the number of affordable generations is
    `fe_budget // pop_size - 1` and the segment consumes the largest multiple
    of `pop_size` that fits within `fe_budget`.
    """
    box_lo = np.asarray(box_lo, dtype=np.float32)
    box_hi = np.asarray(box_hi, dtype=np.float32)
    domain_lo = np.asarray(domain_lo, dtype=np.float32)
    domain_hi = np.asarray(domain_hi, dtype=np.float32)
    domain_width = domain_hi - domain_lo

    de = _SegmentDE(objective, dim, pop_size, box_lo, box_hi,
                    domain_lo, domain_hi, strategy, seed)

    def box_width_frac():
        pop = de.population.numpy()
        widths = pop.max(axis=0) - pop.min(axis=0)
        return float(np.mean(widths / domain_width))

    K = n_checkpoints
    generations = max(0, fe_budget // pop_size - 1)

    # Per-generation history, index 0 = the initial population (before any
    # generation). Checkpoints index into these arrays.
    best_hist = [de.best_fitness]
    mean_hist = [float(tf.reduce_mean(de.fitness).numpy())]
    box_hist = [box_width_frac()]
    succ_hist = [0.0]                       # index 0 has no preceding trials
    fes_hist = [de.evals]

    for _ in range(generations):
        stats = de.step_generation(F, CR)
        best_hist.append(de.best_fitness)
        mean_hist.append(stats["mean_fitness"])
        box_hist.append(box_width_frac())
        succ_hist.append(stats["success_rate"])
        fes_hist.append(de.evals)

    # Checkpoints at evenly spaced generation fractions (spec §5), so the
    # trajectory has fixed length K regardless of fe_budget.
    trajectory = np.zeros((K, 5), dtype=np.float32)
    prev = 0
    for i in range(K):
        c = round((i + 1) * generations / K)
        interval = succ_hist[prev + 1:c + 1]   # generations (prev, c]
        succ = float(np.mean(interval)) if interval else 0.0
        trajectory[i] = (box_hist[c], succ, best_hist[c], mean_hist[c],
                         fes_hist[c])
        prev = c

    final_population = de.population.numpy()
    final_fitness = de.fitness.numpy()
    return SegmentResult(
        final_population=final_population,
        final_fitness=final_fitness,
        best_solution=de.best_solution,
        best_fitness=de.best_fitness,
        initial_best_fitness=best_hist[0],
        fes_used=de.evals,
        generations=generations,
        trajectory=trajectory,
        final_box_lo=final_population.min(axis=0),
        final_box_hi=final_population.max(axis=0),
    )
