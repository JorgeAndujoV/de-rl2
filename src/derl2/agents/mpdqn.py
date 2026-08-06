"""MP-DQN: Multi-Pass Parameterized DQN (spec §6.5, EXP009).

A parameterized-action agent for the continuous action space
(param_strategy_continuous): the DISCRETE action is the strategy k in [0, K);
each strategy carries a CONTINUOUS parameter vector x_k = [F, CR, budget_frac,
box_scale] in [0,1]^param_dim, produced by an actor network. A critic scores
Q(s, k, x_k).

Two networks:
  * actor  : obs -> K*param_dim raw params in [0,1] (sigmoid), reshaped to
             (K, param_dim) — the params for every strategy;
  * critic : (obs, K*param_dim params) -> K Q-values.

Multi-pass (the "MP" in MP-DQN): to score strategy k the critic is fed the
params with ONLY block k non-zero, and its k-th output is read. This decouples
∂Q_k/∂x_j = 0 for j != k — the bias P-DQN suffers when every Q shares all params.

SMDP: the Bellman target discounts by gamma**tau (per_budget) exactly as the DQN,
tau being the segment's elapsed budget in tau_frac units. Stability guardrails
(EXP004 saw Q divergence on long horizons): global-norm gradient clipping,
double-Q target (online net selects the strategy, target net evaluates it), and
TD3-style smoothing noise on the target actor's params. Exploration is
epsilon-greedy on the strategy plus decaying Gaussian noise on the params.

Parameter names match the `agent` block of an experiment's config.yaml (minus
`name`); param_dim must equal the action space's param_dim (the action space's
decode validates this at the first step).
"""

import numpy as np
import tensorflow as tf


def _mlp(input_dim, hidden, out_dim, out_activation=None):
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in hidden:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(out_dim, activation=out_activation))
    return tf.keras.Sequential(layers)


class MPDQNAgent:
    def __init__(self, obs_dim, n_actions, *,
                 param_dim,
                 discounting="per_budget",
                 gamma=0.99,
                 lr=1.0e-3,
                 actor_hidden=(128, 128),
                 critic_hidden=(128, 128),
                 buffer_size=100000,
                 batch_size=64,
                 target_update=500,
                 warmup_transitions=100,
                 grad_clip_norm=10.0,
                 double_q=True,
                 target_smooth_std=0.1,
                 epsilon=None,
                 param_noise=None,
                 seed=0):
        if discounting not in ("per_budget", "per_step"):
            raise ValueError(
                f"agent.discounting must be 'per_budget' or 'per_step', "
                f"got {discounting!r}.")
        tf.random.set_seed(seed)
        self.obs_dim = obs_dim
        self.n_actions = int(n_actions)          # K, the number of strategies
        self.param_dim = int(param_dim)
        self.discounting = discounting
        self.per_budget = (discounting == "per_budget")
        self.gamma = gamma
        self.target_update = target_update
        self.grad_clip_norm = grad_clip_norm
        self.double_q = bool(double_q)
        self.target_smooth_std = float(target_smooth_std)

        epsilon = epsilon or {"start": 1.0, "end": 0.05, "decay_steps": 40000}
        self.eps_start = epsilon["start"]
        self.eps_end = epsilon["end"]
        self.eps_decay_steps = epsilon["decay_steps"]
        param_noise = param_noise or {"start": 0.3, "end": 0.05,
                                      "decay_steps": 40000}
        self.pn_start = param_noise["start"]
        self.pn_end = param_noise["end"]
        self.pn_decay_steps = param_noise["decay_steps"]

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.warmup_transitions = warmup_transitions

        K, pd = self.n_actions, self.param_dim
        self.actor = _mlp(obs_dim, list(actor_hidden), K * pd, "sigmoid")
        self.actor_target = _mlp(obs_dim, list(actor_hidden), K * pd, "sigmoid")
        self.critic = _mlp(obs_dim + K * pd, list(critic_hidden), K)
        self.critic_target = _mlp(obs_dim + K * pd, list(critic_hidden), K)
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # Per-strategy masks: masks[k] keeps only block k of the flat param
        # vector, so critic(concat(obs, params*masks[k]))[:, k] = Q(s, k, x_k).
        masks = np.zeros((K, K * pd), dtype=np.float32)
        for k in range(K):
            masks[k, k * pd:(k + 1) * pd] = 1.0
        self.masks = tf.constant(masks)

        self.train_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------- schedules
    def epsilon(self, step):
        if self.eps_decay_steps <= 0:
            return self.eps_end
        frac = min(step / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _param_noise(self, step):
        if self.pn_decay_steps <= 0:
            return self.pn_end
        frac = min(step / self.pn_decay_steps, 1.0)
        return self.pn_start + frac * (self.pn_end - self.pn_start)

    # ------------------------------------------------------- multi-pass Q
    def _all_q(self, critic, obs, x_flat, training):
        """Q(s, k, x_k) for every k -> (batch, K), via K masked critic passes."""
        cols = []
        for k in range(self.n_actions):
            e_k = x_flat * self.masks[k]
            q_k = critic(tf.concat([obs, e_k], axis=1), training=training)[:, k]
            cols.append(q_k)
        return tf.stack(cols, axis=1)

    # ------------------------------------------------------- exploration
    def act(self, obs, epsilon=0.0):
        """Return (k, params). epsilon>0 explores (epsilon-greedy strategy +
        Gaussian param noise scheduled on train_steps); epsilon=0 is greedy —
        argmax strategy, deterministic actor params — which evaluation uses."""
        o = tf.constant(obs[None, :], dtype=tf.float32)
        x = self.actor(o, training=False)                         # (1, K*pd)
        if epsilon > 0.0:
            std = self._param_noise(int(self.train_steps.numpy()))
            if std > 0.0:
                x = tf.clip_by_value(
                    x + tf.random.normal(tf.shape(x), stddev=std), 0.0, 1.0)
        q = self._all_q(self.critic, o, x, False)[0].numpy()      # (K,)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            k = int(self.rng.integers(self.n_actions))
        else:
            k = int(np.argmax(q))
        params = x.numpy().reshape(self.n_actions, self.param_dim)[k]
        return (k, params)

    # ---------------------------------------------------------- learning
    @tf.function
    def _train_step(self, obs, actions, params, rewards, next_obs, dones, tau):
        K, pd = self.n_actions, self.param_dim

        # ---- target: smoothed target-actor params, double-Q evaluation ----
        xt = self.actor_target(next_obs, training=False)
        noise = tf.clip_by_value(
            tf.random.normal(tf.shape(xt), stddev=self.target_smooth_std),
            -0.5, 0.5)
        xt = tf.clip_by_value(xt + noise, 0.0, 1.0)
        qt_target = self._all_q(self.critic_target, next_obs, xt, False)  # (B,K)
        if self.double_q:
            qt_online = self._all_q(self.critic, next_obs, xt, False)
            a_star = tf.argmax(qt_online, axis=1)
            qt = tf.reduce_sum(qt_target * tf.one_hot(a_star, K), axis=1)
        else:
            qt = tf.reduce_max(qt_target, axis=1)
        discount = tf.pow(self.gamma, tau) if self.per_budget else self.gamma
        y = rewards + discount * (1.0 - dones) * qt

        # ---- critic update: Q of the TAKEN (k, params) toward y ----
        onehot = tf.one_hot(actions, K)                            # (B,K)
        e_taken = tf.reshape(onehot[:, :, None] * params[:, None, :],
                             (-1, K * pd))
        with tf.GradientTape() as ctape:
            q_all = self.critic(tf.concat([obs, e_taken], axis=1), training=True)
            q_taken = tf.reduce_sum(q_all * onehot, axis=1)
            critic_loss = tf.reduce_mean(tf.keras.losses.huber(y, q_taken))
        cg = ctape.gradient(critic_loss, self.critic.trainable_variables)
        cg, _ = tf.clip_by_global_norm(cg, self.grad_clip_norm)
        self.critic_opt.apply_gradients(zip(cg, self.critic.trainable_variables))

        # ---- actor update: raise Q of the actor's own params (critic fixed) ----
        with tf.GradientTape() as atape:
            x = self.actor(obs, training=True)
            q_actor = self._all_q(self.critic, obs, x, False)
            actor_loss = -tf.reduce_mean(tf.reduce_sum(q_actor, axis=1))
        ag = atape.gradient(actor_loss, self.actor.trainable_variables)
        ag, _ = tf.clip_by_global_norm(ag, self.grad_clip_norm)
        self.actor_opt.apply_gradients(zip(ag, self.actor.trainable_variables))

        return critic_loss

    def train(self, batch):
        loss = self._train_step(*batch)
        self.train_steps.assign_add(1)
        if int(self.train_steps.numpy()) % self.target_update == 0:
            self.critic_target.set_weights(self.critic.get_weights())
            self.actor_target.set_weights(self.actor.get_weights())
        return float(loss.numpy())

    # ------------------------------------------------------------ plumbing
    def make_buffer(self, capacity, obs_dim, seed=0):
        from derl2.replay import ParamReplayBuffer
        return ParamReplayBuffer(capacity, obs_dim, self.param_dim, seed=seed)

    def checkpoint_items(self):
        return {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "train_steps": self.train_steps,
        }
