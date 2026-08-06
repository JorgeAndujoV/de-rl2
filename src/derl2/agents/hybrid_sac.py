"""Hybrid SAC agent (spec §6.5, EXP011).

Off-policy Soft Actor-Critic on the SAME parameterized action space as MP-DQN
and PPO (param_strategy_continuous): action = (strategy k, [F, CR, budget_frac,
box_scale]). Being off-policy, it reuses the ParamReplayBuffer and the ordinary
off-policy training loop unchanged — no vectorized envs, no train.py changes.

Policy (state-conditioned, params shared across strategies — the same factored
choice as PPO):
  * Categorical(strategy)                       — K logits;
  * a squashed Gaussian per continuous param     — u ~ N(mu, sigma), then
    a = 0.5*(tanh(u)+1) in [0,1]; reparameterized so the actor gradient flows
    through the sampled action (SAC needs this — hence Gaussian, not Beta).

Critics: twin Q(s, params) -> K (one value per strategy for those params), each
with a Polyak-averaged target. Because params are shared, the critic is a plain
(obs+params)->K net — no multi-pass. The discrete part uses the exact
discrete-SAC expectation over the K strategies (weighted by pi(k|s)); the
continuous part uses the reparameterized sample. Two entropy temperatures
(alpha_disc, alpha_cont) are auto-tuned toward target entropies. The soft
Bellman target discounts by gamma**tau (SMDP), as the other agents.

Entropy regularization + twin critics + soft targets make this the most
divergence-resistant of the three agents — a deliberate contrast to the
value-based instability seen in EXP004. act(obs, epsilon=0) is deterministic
(argmax strategy, Gaussian means) for evaluation; during training the loop
passes epsilon>0 and the policy is sampled (SAC explores via entropy, not
epsilon-greedy, so epsilon only toggles sample-vs-deterministic).
"""

import numpy as np
import tensorflow as tf

_LOG2 = float(np.log(2.0))
_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0


def _mlp(input_dim, hidden, out_dim):
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in hidden:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(out_dim))
    return tf.keras.Sequential(layers)


class HybridSACAgent:
    def __init__(self, obs_dim, n_actions, *,
                 param_dim,
                 discounting="per_budget",
                 gamma=0.99,
                 lr=3.0e-4,
                 actor_hidden=(256, 256),
                 critic_hidden=(256, 256),
                 buffer_size=100000,
                 batch_size=256,
                 warmup_transitions=1000,
                 tau_polyak=0.995,
                 initial_alpha=0.1,
                 target_entropy_disc_ratio=0.5,
                 target_entropy_cont=None,    # default -param_dim
                 max_grad_norm=10.0,
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
        self.tau_polyak = tau_polyak
        self.max_grad_norm = max_grad_norm

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.warmup_transitions = warmup_transitions

        K, pd = self.n_actions, self.param_dim
        self.target_H_disc = float(target_entropy_disc_ratio) * float(np.log(K))
        self.target_H_cont = float(-pd if target_entropy_cont is None
                                   else target_entropy_cont)

        self.actor = _mlp(obs_dim, list(actor_hidden), K + 2 * pd)
        self.critic1 = _mlp(obs_dim + pd, list(critic_hidden), K)
        self.critic2 = _mlp(obs_dim + pd, list(critic_hidden), K)
        self.critic1_t = _mlp(obs_dim + pd, list(critic_hidden), K)
        self.critic2_t = _mlp(obs_dim + pd, list(critic_hidden), K)
        self.critic1_t.set_weights(self.critic1.get_weights())
        self.critic2_t.set_weights(self.critic2.get_weights())

        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.alpha_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        log_a0 = float(np.log(initial_alpha))
        self.log_alpha_disc = tf.Variable(log_a0, dtype=tf.float32)
        self.log_alpha_cont = tf.Variable(log_a0, dtype=tf.float32)

        self.train_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------- policy
    def _actor_raw(self, obs):
        out = self.actor(obs, training=False)
        K, pd = self.n_actions, self.param_dim
        logits = out[:, :K]
        mu = out[:, K:K + pd]
        log_std = tf.clip_by_value(out[:, K + pd:], _LOG_STD_MIN, _LOG_STD_MAX)
        return logits, mu, log_std

    def _sample_continuous(self, mu, log_std):
        """Reparameterized squashed-Gaussian sample in [0,1] and its log-prob."""
        std = tf.exp(log_std)
        eps = tf.random.normal(tf.shape(mu))
        u = mu + std * eps
        a = tf.tanh(u)
        a01 = 0.5 * (a + 1.0)
        # logN(u) with u=mu+std*eps  ->  (u-mu)/std = eps
        log_n = -0.5 * eps ** 2 - log_std - 0.5 * np.log(2.0 * np.pi)
        # squash+affine correction: log|d a01/d u| = log(0.5) + log(1 - tanh^2 u)
        correction = -_LOG2 + tf.math.log(1.0 - a ** 2 + 1e-6)
        logp = tf.reduce_sum(log_n - correction, axis=1)
        return a01, logp

    # --------------------------------------------------- collection / eval
    def epsilon(self, step):
        """SAC explores via policy entropy, not epsilon-greedy; a positive
        constant just tells act() to sample during training."""
        return 1.0

    def act(self, obs, epsilon=0.0):
        o = tf.constant(np.asarray(obs)[None, :], dtype=tf.float32)
        logits, mu, log_std = self._actor_raw(o)
        if epsilon <= 0.0:                        # deterministic (eval)
            k = int(tf.argmax(logits[0]).numpy())
            a01 = (0.5 * (tf.tanh(mu[0]) + 1.0)).numpy().astype(np.float32)
        else:                                     # stochastic (training)
            p = tf.nn.softmax(logits)[0].numpy().astype(np.float64)
            p = p / p.sum()
            k = int(self.rng.choice(self.n_actions, p=p))
            std = tf.exp(log_std[0]).numpy()
            u = mu[0].numpy() + std * self.rng.standard_normal(self.param_dim)
            a01 = (0.5 * (np.tanh(u) + 1.0)).astype(np.float32)
        return (k, a01)

    # ---------------------------------------------------------- learning
    def _critic_in(self, obs, params):
        return tf.concat([obs, params], axis=1)

    @tf.function
    def _train_step(self, obs, k, params, rewards, next_obs, dones, tau):
        K = self.n_actions
        alpha_d = tf.exp(self.log_alpha_disc)
        alpha_c = tf.exp(self.log_alpha_cont)

        # ---- target soft value V(s') ----
        out2 = self.actor(next_obs, training=False)
        logits2 = out2[:, :K]
        mu2 = out2[:, K:K + self.param_dim]
        log_std2 = tf.clip_by_value(out2[:, K + self.param_dim:],
                                    _LOG_STD_MIN, _LOG_STD_MAX)
        a2, logp2_cont = self._sample_continuous(mu2, log_std2)
        logp2_disc = tf.nn.log_softmax(logits2)
        probs2 = tf.nn.softmax(logits2)
        q1t = self.critic1_t(self._critic_in(next_obs, a2), training=False)
        q2t = self.critic2_t(self._critic_in(next_obs, a2), training=False)
        qt_min = tf.minimum(q1t, q2t)                          # (B,K)
        v_next = tf.reduce_sum(
            probs2 * (qt_min - alpha_d * logp2_disc), axis=1) \
            - alpha_c * logp2_cont                             # (B,)
        discount = tf.pow(self.gamma, tau) if self.per_budget else self.gamma
        y = tf.stop_gradient(rewards + discount * (1.0 - dones) * v_next)

        onehot = tf.one_hot(k, K)
        # ---- critics ----
        with tf.GradientTape() as tape_c:
            q1 = self.critic1(self._critic_in(obs, params), training=True)
            q2 = self.critic2(self._critic_in(obs, params), training=True)
            q1_taken = tf.reduce_sum(q1 * onehot, axis=1)
            q2_taken = tf.reduce_sum(q2 * onehot, axis=1)
            critic_loss = tf.reduce_mean((q1_taken - y) ** 2) \
                + tf.reduce_mean((q2_taken - y) ** 2)
        cvars = self.critic1.trainable_variables + self.critic2.trainable_variables
        cg = tape_c.gradient(critic_loss, cvars)
        cg, _ = tf.clip_by_global_norm(cg, self.max_grad_norm)
        self.critic_opt.apply_gradients(zip(cg, cvars))

        # ---- actor ----
        with tf.GradientTape() as tape_a:
            out = self.actor(obs, training=True)
            logits = out[:, :K]
            mu = out[:, K:K + self.param_dim]
            log_std = tf.clip_by_value(out[:, K + self.param_dim:],
                                       _LOG_STD_MIN, _LOG_STD_MAX)
            a, logp_cont = self._sample_continuous(mu, log_std)
            logp_disc = tf.nn.log_softmax(logits)
            probs = tf.nn.softmax(logits)
            q1a = self.critic1(self._critic_in(obs, a), training=False)
            q2a = self.critic2(self._critic_in(obs, a), training=False)
            q_min = tf.minimum(q1a, q2a)                        # (B,K)
            actor_loss = tf.reduce_mean(
                tf.reduce_sum(probs * (alpha_d * logp_disc - q_min), axis=1)
                + alpha_c * logp_cont)
        ag = tape_a.gradient(actor_loss, self.actor.trainable_variables)
        ag, _ = tf.clip_by_global_norm(ag, self.max_grad_norm)
        self.actor_opt.apply_gradients(zip(ag, self.actor.trainable_variables))

        # ---- temperatures (drive entropy toward the targets) ----
        sg_probs = tf.stop_gradient(probs)
        sg_lpd = tf.stop_gradient(logp_disc)
        sg_lpc = tf.stop_gradient(logp_cont)
        with tf.GradientTape() as tape_al:
            ad_loss = tf.reduce_mean(tf.reduce_sum(
                sg_probs * (-self.log_alpha_disc
                            * (sg_lpd + self.target_H_disc)), axis=1))
            ac_loss = tf.reduce_mean(
                -self.log_alpha_cont * (sg_lpc + self.target_H_cont))
            alpha_loss = ad_loss + ac_loss
        alpha_vars = [self.log_alpha_disc, self.log_alpha_cont]
        alpha_g = tape_al.gradient(alpha_loss, alpha_vars)
        self.alpha_opt.apply_gradients(zip(alpha_g, alpha_vars))

        # ---- Polyak soft update of the target critics ----
        rho = self.tau_polyak
        for t, s in zip(self.critic1_t.variables, self.critic1.variables):
            t.assign(rho * t + (1.0 - rho) * s)
        for t, s in zip(self.critic2_t.variables, self.critic2.variables):
            t.assign(rho * t + (1.0 - rho) * s)

        return critic_loss

    def train(self, batch):
        loss = self._train_step(*batch)
        self.train_steps.assign_add(1)
        return float(loss.numpy())

    # ------------------------------------------------------------ plumbing
    def make_buffer(self, capacity, obs_dim, seed=0):
        from derl2.replay import ParamReplayBuffer
        return ParamReplayBuffer(capacity, obs_dim, self.param_dim, seed=seed)

    def checkpoint_items(self):
        return {
            "actor": self.actor,
            "critic1": self.critic1, "critic2": self.critic2,
            "critic1_t": self.critic1_t, "critic2_t": self.critic2_t,
            "actor_opt": self.actor_opt, "critic_opt": self.critic_opt,
            "alpha_opt": self.alpha_opt,
            "log_alpha_disc": self.log_alpha_disc,
            "log_alpha_cont": self.log_alpha_cont,
            "train_steps": self.train_steps,
        }
