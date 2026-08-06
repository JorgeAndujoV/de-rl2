"""DQN agent with a target network and SMDP (γ^τ) discounting.

Ported from de-rl's DQN with one change (spec §6.5): the Bellman target
discounts by `gamma ** tau` instead of a flat `gamma`, where tau is the
segment's elapsed budget in units of the smallest budget fraction (5%). Because
steps consume different amounts of budget, discounting by elapsed budget rather
than by step count is what makes the value function consistent. The config flag
`agent.discounting` (per_budget | per_step) selects between γ^τ and plain γ, so
the correction itself is an ablatable variable.

Parameter names here match the `agent` block in an experiment's config.yaml
(minus `name`), so the registry can pass them straight through. The exploration
schedule lives in the agent — different agents explore differently — and is a
function of the training-step count, matching the config's step-based
epsilon.decay_steps.
"""

import numpy as np
import tensorflow as tf


def build_q_network(obs_dim, n_actions, hidden):
    """MLP mapping an observation to one Q-value per action."""
    layers = [tf.keras.layers.Input(shape=(obs_dim,))]
    for units in hidden:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(n_actions))
    return tf.keras.Sequential(layers)


class DQNAgent:
    """Vanilla DQN: greedy target, periodic hard target-network sync, γ^τ."""

    def __init__(self, obs_dim, n_actions,
                 discounting="per_budget",
                 gamma=0.99,
                 lr=1.0e-3,
                 hidden=(100, 75, 50),
                 buffer_size=100000,
                 batch_size=64,
                 target_update=500,
                 warmup_transitions=100,
                 epsilon=None,
                 seed=0):
        """obs_dim and n_actions come from the environment; everything else
        from config.agent. buffer_size / batch_size / warmup_transitions are
        stored here (not used by the network) so train.py reads replay settings
        off the agent instead of guessing which config block owns them."""
        if discounting not in ("per_budget", "per_step"):
            raise ValueError(
                f"agent.discounting must be 'per_budget' or 'per_step', "
                f"got {discounting!r}."
            )
        tf.random.set_seed(seed)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.discounting = discounting
        self.per_budget = (discounting == "per_budget")
        self.gamma = gamma
        self.target_update = target_update

        epsilon = epsilon or {"start": 1.0, "end": 0.05, "decay_steps": 50000}
        self.eps_start = epsilon["start"]
        self.eps_end = epsilon["end"]
        self.eps_decay_steps = epsilon["decay_steps"]

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.warmup_transitions = warmup_transitions

        self.q_net = build_q_network(obs_dim, n_actions, list(hidden))
        self.target_net = build_q_network(obs_dim, n_actions, list(hidden))
        self.target_net.set_weights(self.q_net.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.train_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------- exploration
    def epsilon(self, step):
        """Linear decay from eps_start to eps_end over decay_steps TRAINING
        steps (gradient updates), then held at eps_end."""
        if self.eps_decay_steps <= 0:
            return self.eps_end
        frac = min(step / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, epsilon=0.0):
        """Epsilon-greedy action for a single observation. epsilon=0 is greedy,
        which is what evaluation uses."""
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        q_values = self.q_net(obs[None, :], training=False)
        return int(tf.argmax(q_values[0]).numpy())

    # ---------------------------------------------------------- learning
    @tf.function
    def _train_step(self, obs, actions, rewards, next_obs, dones, tau):
        next_q = tf.reduce_max(self.target_net(next_obs, training=False), axis=1)
        # SMDP discounting: γ^τ per transition (per_budget) or flat γ (per_step).
        # per_budget is the correction of §6.5; per_step is the ablation.
        if self.per_budget:
            discount = tf.pow(self.gamma, tau)
        else:
            discount = self.gamma
        targets = rewards + discount * (1.0 - dones) * next_q

        with tf.GradientTape() as tape:
            q_all = self.q_net(obs, training=True)
            mask = tf.one_hot(actions, self.n_actions)
            q = tf.reduce_sum(q_all * mask, axis=1)
            loss = tf.keras.losses.huber(targets, q)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def train(self, batch):
        """One gradient step on a sampled batch. Returns the loss."""
        loss = self._train_step(*batch)
        self.train_steps.assign_add(1)
        if int(self.train_steps.numpy()) % self.target_update == 0:
            self.target_net.set_weights(self.q_net.get_weights())
        return float(loss.numpy())

    # ------------------------------------------------------------ plumbing
    def make_buffer(self, capacity, obs_dim, seed=0):
        """The replay format this agent needs (a scalar-action buffer). train.py
        asks the agent for its buffer so it never names a concrete buffer class."""
        from derl2.replay import ReplayBuffer
        return ReplayBuffer(capacity, obs_dim, seed=seed)

    # ------------------------------------------------------ checkpointing
    def checkpoint_items(self):
        """Objects tf.train.Checkpoint should track, so train.py doesn't need
        to know this agent's internals."""
        return {
            "q_net": self.q_net,
            "target_net": self.target_net,
            "optimizer": self.optimizer,
            "train_steps": self.train_steps,
        }
