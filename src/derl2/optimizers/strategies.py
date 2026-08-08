"""DE mutation strategies.

Each strategy provides both a NumPy version (for the loop-based reference
optimizer) and a TensorFlow version (for the vectorized optimizer), kept in
one class so a new strategy cannot be added to one backend and silently
forgotten in the other.

Interface contract:
  n_random          how many distinct random indices the strategy needs
  mutate_np(...)    one mutant vector,  shapes: pop (NP, D) -> (D,)
  mutate_tf(...)    all mutant vectors, shapes: pop (NP, D) -> (NP, D)

Both mutate methods receive:
  pop     the current population
  idx     the random indices (NumPy: a tuple of ints; TF: a list of (NP,)
          index tensors), already guaranteed distinct and != the target
  best    the current best individual, (D,)
  target  the individual being mutated ((D,) for NumPy, (NP, D) for TF)
  F       the scale factor

Strategies that don't need `best` or `target` simply ignore them, so all
strategies share one call signature and the optimizers stay generic.
"""

import tensorflow as tf


class Strategy:
    """Base class. Subclasses set `name`, `n_random`, and both mutate methods.

    `needs_pbest` marks strategies (current-to-pbest/1) that need a per-row
    `pbest` donor — a random member of the top `p_best` fraction of the
    population by fitness. The optimizer computes it (only such a strategy has
    the fitness ranking) and passes it as an extra positional argument; the
    default False keeps the 5-argument mutate signature for every other
    strategy, so their calls — and their RNG draws — are unchanged."""

    name = None
    n_random = 0
    needs_pbest = False
    p_best = 0.1

    def mutate_np(self, pop, idx, best, target, F):
        raise NotImplementedError

    def mutate_tf(self, pop, idx, best, target, F):
        raise NotImplementedError

    def __repr__(self):
        return f"<Strategy {self.name}>"


class Rand1(Strategy):
    """DE/rand/1:  v = x_r1 + F * (x_r2 - x_r3)

    The canonical strategy. Purely explorative — no information from the
    best individual enters the mutant."""

    name = "rand/1"
    n_random = 3

    def mutate_np(self, pop, idx, best, target, F):
        r1, r2, r3 = idx
        return pop[r1] + F * (pop[r2] - pop[r3])

    def mutate_tf(self, pop, idx, best, target, F):
        r1, r2, r3 = idx
        return tf.gather(pop, r1) + F * (tf.gather(pop, r2) - tf.gather(pop, r3))


class Best1(Strategy):
    """DE/best/1:  v = x_best + F * (x_r1 - x_r2)

    Strongly exploitative: every mutant is built around the current best,
    so convergence is fast but diversity collapses quickly on multimodal
    landscapes."""

    name = "best/1"
    n_random = 2

    def mutate_np(self, pop, idx, best, target, F):
        r1, r2 = idx
        return best + F * (pop[r1] - pop[r2])

    def mutate_tf(self, pop, idx, best, target, F):
        r1, r2 = idx
        return best[None, :] + F * (tf.gather(pop, r1) - tf.gather(pop, r2))


class CurrentToBest1(Strategy):
    """DE/current-to-best/1:  v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)

    A middle ground: each individual moves partway toward the best while
    also taking a random differential step."""

    name = "current-to-best/1"
    n_random = 2

    def mutate_np(self, pop, idx, best, target, F):
        r1, r2 = idx
        return target + F * (best - target) + F * (pop[r1] - pop[r2])

    def mutate_tf(self, pop, idx, best, target, F):
        r1, r2 = idx
        return (target
                + F * (best[None, :] - target)
                + F * (tf.gather(pop, r1) - tf.gather(pop, r2)))


class Rand2(Strategy):
    """DE/rand/2:  v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)

    Two differentials give a more diverse mutant distribution than rand/1,
    at the cost of needing a larger population."""

    name = "rand/2"
    n_random = 5

    def mutate_np(self, pop, idx, best, target, F):
        r1, r2, r3, r4, r5 = idx
        return pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])

    def mutate_tf(self, pop, idx, best, target, F):
        r1, r2, r3, r4, r5 = idx
        return (tf.gather(pop, r1)
                + F * (tf.gather(pop, r2) - tf.gather(pop, r3))
                + F * (tf.gather(pop, r4) - tf.gather(pop, r5)))


class CurrentToPbest1(Strategy):
    """DE/current-to-pbest/1 (JADE, archive-free):

        v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)

    Like current-to-best/1 but the greedy attractor is a *random* member of the
    top `p_best` fraction rather than the single global best, so the pull is
    strong yet not all toward one point — the key JADE idea for keeping
    diversity while exploiting. `x_pbest` is chosen per row by the optimizer
    (needs_pbest=True) and supplied as the extra `pbest` argument. The external
    archive of JADE is deliberately omitted for this first pbest experiment;
    x_r1, x_r2 come from the current population only."""

    name = "current-to-pbest/1"
    n_random = 2
    needs_pbest = True

    def mutate_np(self, pop, idx, best, target, F, pbest):
        r1, r2 = idx
        return target + F * (pbest - target) + F * (pop[r1] - pop[r2])

    def mutate_tf(self, pop, idx, best, target, F, pbest):
        r1, r2 = idx
        return (target
                + F * (pbest - target)
                + F * (tf.gather(pop, r1) - tf.gather(pop, r2)))


# --------------------------------------------------------------- registry

# This dictionary is the list of available mutation strategies. To add one:
# write the class above, then add a line here. The names are what appear in
# a config's environment.params.strategy.
STRATEGIES = {
    "rand/1": Rand1,
    "best/1": Best1,
    "current-to-best/1": CurrentToBest1,
    "rand/2": Rand2,
    "current-to-pbest/1": CurrentToPbest1,
}


def build_strategy(name):
    """Instantiate a mutation strategy by name.

    Accepts either 'rand/1' or the full 'rand/1/bin' — the crossover half is
    stripped, since binomial crossover is currently the only option. When
    exponential crossover is added this should become an explicit
    environment.params.crossover key rather than more parsing here.
    """
    key = name
    if key.endswith("/bin"):
        key = key[: -len("/bin")]
    if key not in STRATEGIES:
        raise KeyError(
            f"Unknown DE strategy {name!r}. Available: {sorted(STRATEGIES)}. "
            f"Add it to STRATEGIES in src/derl2/optimizers/strategies.py."
        )
    return STRATEGIES[key]()