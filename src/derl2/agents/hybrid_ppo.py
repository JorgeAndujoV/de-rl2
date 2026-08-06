"""Hybrid PPO agent (spec §6.5, EXP010).

On-policy PPO over the SAME parameterized action space as MP-DQN
(param_strategy_continuous): the action is (strategy k, [F, CR, budget_frac,
box_scale]). The policy is FACTORIZED and conditioned on state only (independent
of the sampled strategy):

    pi(a|s) = Categorical(strategy) * prod_j Beta_j(param_j)

Each continuous parameter is a Beta on [0,1] with alpha,beta = softplus(.)+1 >= 1
(unimodal — never the U-shaped Beta whose density blows up at 0/1). Beta is used
instead of a squashed Gaussian because the params are genuinely bounded; log-prob
and entropy are computed in closed form from lgamma/digamma (no
tensorflow_probability dependency), and sampling uses NumPy's Beta.

A separate value network V(s) provides the PPO baseline. Advantages use
SMDP-GAE: the constant gamma of ordinary GAE is replaced per transition by
gamma**tau (tau the segment's elapsed budget in tau_frac units), matching the
gamma**tau Bellman discount the value-based agents use.

on_policy=True tells train.py to run the collect -> GAE -> clipped-surrogate
update loop instead of the off-policy replay loop; this agent has no replay
buffer. act(obs, epsilon=0) is the deterministic greedy policy evaluation uses.
"""

import numpy as np
import tensorflow as tf

_LOGPROB_EPS = 1e-6          # keep Beta log-prob finite at the [0,1] boundary


def compute_gae(rewards, values, dones, taus, last_value,
                gamma, lam, per_budget):
    """SMDP generalized advantage estimation over one contiguous rollout stream.

    values[t] = V(s_t); last_value = V(s_T) bootstraps the final step when the
    rollout was cut mid-episode (dones[-1]==0). The per-transition discount is
    gamma**tau[t] (per_budget) or gamma (per_step); a done resets the recursion
    so advantages never leak across an episode boundary. Returns (adv, returns).
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        d = gamma ** taus[t] if per_budget else gamma
        nonterminal = 1.0 - dones[t]
        next_v = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + d * nonterminal * next_v - values[t]
        gae = delta + d * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + np.asarray(values, dtype=np.float64)
    return adv, returns


def _mlp(input_dim, hidden, out_dim):
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in hidden:
        layers.append(tf.keras.layers.Dense(units, activation="tanh"))
    layers.append(tf.keras.layers.Dense(out_dim))
    return tf.keras.Sequential(layers)


class HybridPPOAgent:
    on_policy = True

    def __init__(self, obs_dim, n_actions, *,
                 param_dim,
                 discounting="per_budget",
                 gamma=0.99,
                 gae_lambda=0.95,
                 lr=3.0e-4,
                 policy_hidden=(128, 128),
                 value_hidden=(128, 128),
                 clip=0.2,
                 epochs=10,
                 minibatch_size=64,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 rollout_steps=512,
                 n_envs=8,
                 seed=0):
        if discounting not in ("per_budget", "per_step"):
            raise ValueError(
                f"agent.discounting must be 'per_budget' or 'per_step', "
                f"got {discounting!r}.")
        tf.random.set_seed(seed)
        self.obs_dim = obs_dim
        self.n_actions = int(n_actions)          # K strategies
        self.param_dim = int(param_dim)
        self.per_budget = (discounting == "per_budget")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.n_envs = n_envs

        K, pd = self.n_actions, self.param_dim
        self.policy_net = _mlp(obs_dim, list(policy_hidden), K + 2 * pd)
        self.value_net = _mlp(obs_dim, list(value_hidden), 1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.train_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------- policy distribution
    def _dist(self, obs):
        """obs (B, D) -> (logits (B,K), alpha (B,pd), beta (B,pd))."""
        K, pd = self.n_actions, self.param_dim
        out = self.policy_net(obs, training=False)
        logits = out[:, :K]
        ab = tf.nn.softplus(tf.reshape(out[:, K:], (-1, pd, 2))) + 1.0
        return logits, ab[:, :, 0], ab[:, :, 1]

    def _logprob(self, logits, alpha, beta, k, params):
        logp_cat = tf.reduce_sum(
            tf.nn.log_softmax(logits) * tf.one_hot(k, self.n_actions), axis=1)
        x = tf.clip_by_value(params, _LOGPROB_EPS, 1.0 - _LOGPROB_EPS)
        logB = (tf.math.lgamma(alpha) + tf.math.lgamma(beta)
                - tf.math.lgamma(alpha + beta))
        logp_beta = (alpha - 1.0) * tf.math.log(x) \
            + (beta - 1.0) * tf.math.log(1.0 - x) - logB
        return logp_cat + tf.reduce_sum(logp_beta, axis=1)

    def _entropy(self, logits, alpha, beta):
        logp = tf.nn.log_softmax(logits)
        cat_ent = -tf.reduce_sum(tf.exp(logp) * logp, axis=1)
        logB = (tf.math.lgamma(alpha) + tf.math.lgamma(beta)
                - tf.math.lgamma(alpha + beta))
        beta_ent = (logB - (alpha - 1.0) * tf.math.digamma(alpha)
                    - (beta - 1.0) * tf.math.digamma(beta)
                    + (alpha + beta - 2.0) * tf.math.digamma(alpha + beta))
        return cat_ent + tf.reduce_sum(beta_ent, axis=1)

    # ------------------------------------------------------ collection / eval
    def act_collect(self, obs):
        """Sample an action from the current policy. Returns
        ((k, params), logprob, value) — the on-policy loop stores all three."""
        o = tf.constant(np.asarray(obs)[None, :], dtype=tf.float32)
        logits, alpha, beta = self._dist(o)
        p = tf.nn.softmax(logits)[0].numpy().astype(np.float64)
        p = p / p.sum()                     # exact renorm (float softmax != 1.0)
        k = int(self.rng.choice(self.n_actions, p=p))
        a = alpha[0].numpy()
        b = beta[0].numpy()
        params = self.rng.beta(a, b).astype(np.float32)
        logp = float(self._logprob(logits, alpha, beta,
                                   tf.constant([k]),
                                   tf.constant(params[None, :]))[0].numpy())
        value = float(self.value_net(o, training=False)[0, 0].numpy())
        return (k, params), logp, value

    def act(self, obs, epsilon=0.0):
        """Deterministic greedy action (argmax strategy, Beta means) for eval.
        epsilon is accepted for a uniform agent interface and ignored."""
        o = tf.constant(np.asarray(obs)[None, :], dtype=tf.float32)
        logits, alpha, beta = self._dist(o)
        k = int(tf.argmax(logits[0]).numpy())
        params = (alpha / (alpha + beta))[0].numpy().astype(np.float32)
        return (k, params)

    def value(self, obs):
        o = tf.constant(np.asarray(obs)[None, :], dtype=tf.float32)
        return float(self.value_net(o, training=False)[0, 0].numpy())

    # ---------------------------------------------------------------- update
    @tf.function
    def _update_minibatch(self, obs, k, params, old_logp, adv, ret):
        with tf.GradientTape() as tape:
            logits, alpha, beta = self._dist(obs)
            new_logp = self._logprob(logits, alpha, beta, k, params)
            ratio = tf.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            value_pred = self.value_net(obs, training=True)[:, 0]
            value_loss = tf.reduce_mean((value_pred - ret) ** 2)
            entropy = tf.reduce_mean(self._entropy(logits, alpha, beta))
            loss = (policy_loss + self.value_coef * value_loss
                    - self.entropy_coef * entropy)
        variables = (self.policy_net.trainable_variables
                     + self.value_net.trainable_variables)
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, variables))
        return loss

    def update(self, batch):
        """K epochs of clipped-surrogate minibatch SGD on one rollout.

        batch: dict of arrays keyed obs (N,D), k (N,), params (N,pd),
        old_logp (N,), adv (N,), ret (N,). Advantages are normalized here.
        Returns the last minibatch loss for the learning-curve log."""
        obs = tf.constant(batch["obs"], dtype=tf.float32)
        k = tf.constant(batch["k"], dtype=tf.int32)
        params = tf.constant(batch["params"], dtype=tf.float32)
        old_logp = tf.constant(batch["old_logp"], dtype=tf.float32)
        adv = np.asarray(batch["adv"], dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = tf.constant(adv, dtype=tf.float32)
        ret = tf.constant(np.asarray(batch["ret"], dtype=np.float32))

        n = int(obs.shape[0])
        idx = np.arange(n)
        last = 0.0
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                loss = self._update_minibatch(
                    tf.gather(obs, mb), tf.gather(k, mb), tf.gather(params, mb),
                    tf.gather(old_logp, mb), tf.gather(adv, mb),
                    tf.gather(ret, mb))
                self.train_steps.assign_add(1)
                last = float(loss.numpy())
        return last

    # ------------------------------------------------------------ plumbing
    def checkpoint_items(self):
        return {
            "policy_net": self.policy_net,
            "value_net": self.value_net,
            "optimizer": self.optimizer,
            "train_steps": self.train_steps,
        }
