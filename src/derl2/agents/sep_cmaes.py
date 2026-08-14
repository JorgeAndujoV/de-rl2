"""Sep-CMA-ES agent (evosax) for de-rl2 -- additive, EXP026.

Optimises the WEIGHTS of the existing Hybrid-PPO policy network (same MLP shape,
same deterministic decode) with separable CMA-ES instead of PPO. No value
network is searched -- ES needs no critic -- so we flatten ONLY the policy net's
trainable variables. Evaluating a weight vector is one full greedy episode in
the existing DEEnv (reusing evaluate.run_episode), so the hybrid discrete+
continuous action needs no special handling: it is just argmax / Beta-mean off a
forward pass, no logprobs, no distributions.

Isolation: this module only IMPORTS shared code (HybridPPOAgent for the exact
net + deterministic decode; nothing is edited). JAX/evosax are imported ONLY by
the main-process strategy wrapper below; the worker pool builds policy nets in
TensorFlow and never imports JAX.

evosax convention (pinned to evosax 0.1.6 / jax 0.10.2, Nibi wheelhouse):
strategies MINIMISE the (shaped) fitness passed to tell(). Our raw per-member
fitness is quality = -log10(final_error + ERR_EPS) (HIGHER is better), so it is
shaped with FitnessShaper(maximize=True, centered_rank=True), which negates +
rank-transforms so evosax's minimiser drives quality UP. The training loop, not
this class, applies the -log10; this class only shapes and tells.
"""

import numpy as np

from derl2.agents.hybrid_ppo import HybridPPOAgent

# Error floor used when log-scaling the final error into a fitness -- matches the
# observation vector's ERR_EPS so the ES optimises exactly the quantity the rest
# of the pipeline reports.
ERR_EPS = 1e-8


# --------------------------------------------------------- policy net + weights

def build_policy_net(obs_dim, n_actions, param_dim, policy_hidden, seed=0):
    """A HybridPPOAgent reused purely as (policy MLP + deterministic act()).

    Reusing the real agent guarantees byte-identical architecture and decode to
    EXP016 without editing shared files. The value net is built tiny and never
    touched -- ES searches only the policy weights (`value network dropped`).
    Used identically in the main process (to read the weight shapes) and in
    every worker (for the episode forward pass).
    """
    return HybridPPOAgent(
        obs_dim, n_actions,
        param_dim=param_dim,
        policy_hidden=tuple(policy_hidden),
        value_hidden=(8,),          # unused by ES; kept tiny to save worker RAM
        seed=seed,
    )


def flat_size(variables):
    return int(sum(int(np.prod(v.shape)) for v in variables))


def get_flat(variables):
    """policy_net.trainable_variables -> one float64 vector (Keras order)."""
    return np.concatenate(
        [v.numpy().ravel() for v in variables]).astype(np.float64)


def set_flat(variables, flat):
    """Write a flat vector back into policy_net.trainable_variables, in order."""
    flat = np.asarray(flat, dtype=np.float32)
    i = 0
    for v in variables:
        n = int(np.prod(v.shape))
        v.assign(flat[i:i + n].reshape(v.shape))
        i += n


# ------------------------------------------------------ Sep-CMA-ES (main proc)

class SepCMAES:
    """Main-process Sep-CMA-ES driver (evosax). Holds the strategy, its params,
    state, fitness shaper and RNG key.

    ask()  -> (population_size, num_dims) float64 population.
    tell(x, quality) -> update from raw quality fitness (higher = better).
    mean() -> current distribution mean as a flat float64 vector.

    Antithetic (mirrored) sampling is applied around state.mean, since evosax's
    Sep_CMA_ES exposes no native antithetic flag (that lives on its NES-style
    strategies). We take evosax's own ask() samples and, for the second half of
    an even population, replace the perturbation delta = x - mean with -delta.
    This uses evosax for the sampling scale AND for every CMA adaptation step,
    touching only the public state.mean -- no internal covariance fields, no
    hand-rolled Gaussian.
    """

    def __init__(self, num_dims, population_size, *, elite_ratio=0.3,
                 sigma_init=0.05, antithetic=True, seed=0):
        import jax
        from evosax import Sep_CMA_ES, FitnessShaper

        if antithetic and population_size % 2 != 0:
            raise ValueError(
                f"antithetic sampling needs an even population_size, got "
                f"{population_size}.")

        self._jax = jax
        self.num_dims = int(num_dims)
        self.population_size = int(population_size)
        self.antithetic = bool(antithetic)

        # evosax 0.1.6 constructor kwargs (confirm on first smoke): popsize,
        # num_dims, elite_ratio, sigma_init.
        self.strategy = Sep_CMA_ES(
            popsize=self.population_size, num_dims=self.num_dims,
            elite_ratio=float(elite_ratio), sigma_init=float(sigma_init))
        self.es_params = self.strategy.default_params
        # maximize=True: our raw fitness is quality (higher better); the shaper
        # negates + centered-rank-transforms it for evosax's internal minimiser.
        self.shaper = FitnessShaper(centered_rank=True, z_score=False,
                                    w_decay=0.0, maximize=True)

        self.key = jax.random.PRNGKey(int(seed))
        self.key, sub = jax.random.split(self.key)
        self.state = self.strategy.initialize(sub, self.es_params)
        self.generation = 0

    def ask(self):
        jax = self._jax
        self.key, sub = jax.random.split(self.key)
        x, self.state = self.strategy.ask(sub, self.state, self.es_params)
        x = np.asarray(x, dtype=np.float64)
        if self.antithetic:
            mean = np.asarray(self.state.mean, dtype=np.float64)
            half = self.population_size // 2
            delta = x[:half] - mean
            x[half:2 * half] = mean - delta
        return x

    def tell(self, x, quality):
        """x: (P, D) population just evaluated. quality: (P,) raw fitness,
        higher = better (-log10(error))."""
        import jax.numpy as jnp
        xj = jnp.asarray(np.asarray(x))
        fit = self.shaper.apply(xj, jnp.asarray(np.asarray(quality)))
        self.state = self.strategy.tell(xj, fit, self.state, self.es_params)
        self.generation += 1

    def mean(self):
        return np.asarray(self.state.mean, dtype=np.float64)

    # --------------------------------------------------------- resume support
    def state_dict(self):
        """Everything needed to resume EXACTLY where a killed job stopped: the
        evosax EvoState (mean, sigma, per-dim covariance, path terms...), the RNG
        key (so the next ask() draws the same samples), and the generation count.
        The hyperparameters (num_dims, population, elite_ratio, sigma_init,
        antithetic) are NOT stored -- they are rebuilt from config, and a resume
        that changed them would be a different run. Holds JAX arrays; the caller
        pickles it (JAX arrays pickle via NumPy), so loading needs JAX present
        (main process only)."""
        return {"state": self.state, "key": self.key,
                "generation": int(self.generation)}

    def load_state_dict(self, d):
        self.state = d["state"]
        self.key = d["key"]
        self.generation = int(d["generation"])
