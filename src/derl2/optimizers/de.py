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
                 domain_lo, domain_hi, strategy, seed, cov=None,
                 init_population=None):
        self.objective = objective
        self.dim = dim
        self.pop_size = pop_size
        # Clip bounds during evolution are the true domain.
        self.lower = tf.constant(domain_lo, dtype=tf.float32)
        self.upper = tf.constant(domain_hi, dtype=tf.float32)
        # Initial sampling is confined to the box.
        self.box_lo = tf.constant(box_lo, dtype=tf.float32)
        self.box_hi = tf.constant(box_hi, dtype=tf.float32)
        # A covariance box (center, scaled Cholesky) draws the initial population
        # from a Gaussian ellipsoid instead of uniformly in [box_lo, box_hi].
        self.cov = None
        if cov is not None:
            center, chol = cov
            self.cov = (tf.constant(center, dtype=tf.float32),
                        tf.constant(chol, dtype=tf.float32))
        # freedom++ (EXP021+): the caller may supply the whole initial population
        # verbatim (elite carryover + a rule-filled sample from the region). When
        # given it is used as-is and box_lo/box_hi/cov are ignored for the draw.
        self.init_population = (np.asarray(init_population, dtype=np.float32)
                                if init_population is not None else None)
        self.strategy = build_strategy(strategy)

        self._default_strategy_name = strategy
        self._strategy_cache = {strategy: self.strategy}

        self._require_pop_size(self.strategy)

        self.gen = tf.random.Generator.from_seed(seed)
        self.population = tf.Variable(
            tf.zeros((pop_size, dim), dtype=tf.float32), trainable=False
        )
        # float64 so extreme-conditioning functions (f3: bent_cigar reaches
        # ~1e42, past float32's 3.4e38) store finite fitness instead of inf.
        # For every other function the objective still returns float32; the
        # upcast on assign is lossless, so selection stays byte-identical.
        self.fitness = tf.Variable(
            tf.zeros((pop_size,), dtype=tf.float64), trainable=False
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
        if self.init_population is not None:
            # freedom++: caller-provided population (elites + rule-filled sample),
            # already domain-clipped by the sampler. Used verbatim.
            init = tf.constant(self.init_population, dtype=tf.float32)
        elif self.cov is None:
            init = self.gen.uniform(
                (self.pop_size, self.dim),
                minval=self.box_lo, maxval=self.box_hi, dtype=tf.float32,
            )
        else:
            # x = center + z @ chol.T,  z ~ N(0, I); clip to the search domain.
            center, chol = self.cov
            z = self.gen.normal((self.pop_size, self.dim), dtype=tf.float32)
            init = center[None, :] + tf.matmul(z, chol, transpose_b=True)
            init = tf.clip_by_value(init, self.lower, self.upper)
        self.population.assign(init)
        self.fitness.assign(tf.cast(self.objective(init), tf.float64))
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

        # A pbest strategy needs a per-row donor drawn from the top p_best of the
        # population. `needs_pbest` is a static Python bool (the trace is keyed on
        # strategy_name), so this branch — and its extra RNG draw — exists ONLY
        # for that strategy; every other strategy's draws (and thus its results)
        # are untouched. The draw is placed here, after `perm`, so the pbest
        # stream is self-contained.
        if strategy.needs_pbest:
            pbest_size = max(2, int(round(strategy.p_best * NP)))
            top = tf.argsort(self.fitness)[:pbest_size]     # best-first indices
            pick = self.gen.uniform((NP,), maxval=pbest_size, dtype=tf.int32)
            pbest = tf.gather(x, tf.gather(top, pick))       # (NP, D)
            mutant = strategy.mutate_tf(x, idx, best, x, F, pbest)
        else:
            mutant = strategy.mutate_tf(x, idx, best, x, F)
        mutant = tf.clip_by_value(mutant, self.lower, self.upper)

        # binomial crossover, one dimension forced from the mutant
        cross = self.gen.uniform((NP, D)) < CR
        j_rand = self.gen.uniform((NP,), maxval=D, dtype=tf.int32)
        cross = tf.logical_or(cross, tf.cast(tf.one_hot(j_rand, D), tf.bool))
        trial = tf.where(cross, mutant, x)

        trial_fitness = tf.cast(self.objective(trial), tf.float64)
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
                strategy, F, CR, fe_budget, n_checkpoints, seed, cov=None,
                init_population=None, whiten=None):
    """Run one DE segment and return a SegmentResult.

    The initial population evaluation costs `pop_size` FEs and each generation
    costs another `pop_size`, so the number of affordable generations is
    `fe_budget // pop_size - 1` and the segment consumes the largest multiple
    of `pop_size` that fits within `fe_budget`.

    init_population : (pop_size, dim) freedom++ initial population, used verbatim
                     when given (elites + rule-filled sample); box_lo/box_hi/cov
                     then only describe the region for logging.
    whiten         : (L_inv, center) freedom++ whitening frame. When given,
                     trajectory channel 0 becomes the population's RMS whitened
                     spread `sqrt(mean ||L_inv (x - center)||^2 / D)` (measured in
                     the chosen region's coordinates) instead of the axis-aligned
                     mean box-width fraction. Diagonal L_inv reproduces the same
                     per-dimension normalized spread the axis path reports.
    """
    box_lo = np.asarray(box_lo, dtype=np.float32)
    box_hi = np.asarray(box_hi, dtype=np.float32)
    domain_lo = np.asarray(domain_lo, dtype=np.float32)
    domain_hi = np.asarray(domain_hi, dtype=np.float32)
    domain_width = domain_hi - domain_lo

    de = _SegmentDE(objective, dim, pop_size, box_lo, box_hi,
                    domain_lo, domain_hi, strategy, seed, cov=cov,
                    init_population=init_population)

    if whiten is not None:
        _L_inv = np.asarray(whiten[0], dtype=np.float64)
        _center = np.asarray(whiten[1], dtype=np.float64)

    def box_width_frac():
        pop = de.population.numpy()
        if whiten is None:
            widths = pop.max(axis=0) - pop.min(axis=0)
            return float(np.mean(widths / domain_width))
        w = (pop.astype(np.float64) - _center) @ _L_inv.T     # whitened coords
        return float(np.sqrt(np.mean(w ** 2)))

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
    # float64: channels 2 (best) and 3 (mean) hold raw fitness, which for f3
    # reaches ~1e42 -- float32 here would re-truncate to inf and NaN the obs.
    # The observation compresses these via log10 before its own float32 cast.
    trajectory = np.zeros((K, 5), dtype=np.float64)
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
