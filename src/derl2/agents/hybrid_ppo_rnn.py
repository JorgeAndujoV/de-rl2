"""Recurrent Hybrid PPO agent (EXP020).

A pure recurrence ablation of the Hybrid PPO agent (hybrid_ppo.py): SAME
parameterized action space (param_strategy_boxnp), SAME 99-dim observation, SAME
factorized policy

    pi(a | h_t) = Categorical(strategy | h_t) * prod_j Beta_j(param_j | h_t)

but the policy and value networks are now RECURRENT. Instead of conditioning on
the current observation s_t, each conditions on a hidden state h_t = GRU(s_t,
h_{t-1}) that carries information across the ~19 segment-decisions of one episode.
This generalizes the hand-engineered one-to-few-step memory in the observation
(n_stag, prev_*) into a learned state.

Design choices (see the EXP020 discussion):
  * GRU (not LSTM): the ~19-step horizon does not need LSTM's separate cell-state
    pathway; GRU is cheaper with the same conditioning behaviour.
  * SEPARATE recurrent trunks for policy and value (mirroring hybrid_ppo's two
    separate MLPs), so value-loss gradients never shape the policy's hidden state.
  * Hidden state resets to zero at every episode boundary (h_0 = 0); episodes are
    collected WHOLE (episode-aligned), so every training sequence starts at h_0=0
    and no state is carried across rollout boundaries.
  * Exact hidden-state recompute in the update: minibatches are whole episodes,
    replayed through the CURRENT policy each epoch (no cached/stale states). At
    this horizon the extra forward pass is trivial and removes a known source of
    recurrent-PPO instability.

Interface additions over the feedforward agent, consumed by train.py's recurrent
on-policy loop and evaluate.py's greedy policy wrapper:
  * ``recurrent = True``
  * ``initial_state()``            -> per-episode zero hidden state (h_pi, h_v)
  * ``act_collect(obs, state)``    -> ((k, params), logp, value, new_state)
  * ``act_eval(obs, state)``       -> ((k, params), new_state)   greedy, eval
  * ``update(episodes)``           -> last minibatch loss; ``episodes`` is a list
                                      of per-episode dicts with adv/ret already
                                      computed (SMDP-GAE, in train.py).
"""

import numpy as np
import tensorflow as tf

_LOGPROB_EPS = 1e-6          # keep Beta log-prob finite at the [0,1] boundary


def _head(in_dim, hidden, out_dim):
    """A small MLP head applied to the GRU output (works on (B, T, H) too, since
    Dense acts on the last axis)."""
    layers = [tf.keras.layers.Input(shape=(None, in_dim))]
    for units in hidden:
        layers.append(tf.keras.layers.Dense(units, activation="tanh"))
    layers.append(tf.keras.layers.Dense(out_dim))
    return tf.keras.Sequential(layers)


class HybridPPORNNAgent:
    on_policy = True
    recurrent = True

    def __init__(self, obs_dim, n_actions, *,
                 param_dim,
                 discounting="per_budget",
                 gamma=0.99,
                 gae_lambda=0.95,
                 lr=3.0e-4,
                 rnn_hidden=128,
                 head_hidden=(128,),
                 cell="gru",
                 clip=0.2,
                 epochs=6,
                 minibatch_episodes=32,
                 episodes_per_update=256,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 n_envs=8,
                 seed=0):
        if discounting not in ("per_budget", "per_step"):
            raise ValueError(
                f"agent.discounting must be 'per_budget' or 'per_step', "
                f"got {discounting!r}.")
        if cell.lower() != "gru":
            raise ValueError(
                f"agent.cell only supports 'gru' currently, got {cell!r}.")
        tf.random.set_seed(seed)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)          # K discrete actions
        self.param_dim = int(param_dim)
        self.per_budget = (discounting == "per_budget")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        self.minibatch_episodes = int(minibatch_episodes)
        self.episodes_per_update = int(episodes_per_update)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_envs = n_envs
        self.rnn_hidden = int(rnn_hidden)

        K, pd, H = self.n_actions, self.param_dim, self.rnn_hidden
        head_hidden = list(head_hidden)
        # Separate recurrent trunks + heads for policy and value.
        self.pi_gru = tf.keras.layers.GRU(
            H, return_sequences=True, return_state=True, name="pi_gru")
        self.pi_head = _head(H, head_hidden, K + 2 * pd)
        self.v_gru = tf.keras.layers.GRU(
            H, return_sequences=True, return_state=True, name="v_gru")
        self.v_head = _head(H, head_hidden, 1)
        # Build all weights eagerly (so checkpointing sees them before training).
        dummy = tf.zeros((1, 1, self.obs_dim), dtype=tf.float32)
        h0 = tf.zeros((1, H), dtype=tf.float32)
        for gru, head in ((self.pi_gru, self.pi_head), (self.v_gru, self.v_head)):
            seq, _ = gru(dummy, initial_state=h0)
            head(seq)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.train_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ hidden state
    def initial_state(self):
        """Per-episode zero hidden state (h_pi, h_v), each shape (H,)."""
        z = np.zeros(self.rnn_hidden, dtype=np.float32)
        return (z.copy(), z.copy())

    # ------------------------------------------------- single-step forward pass
    @tf.function(reduce_retracing=True)
    def _pi_step(self, obs1, h):
        """obs1 (1,1,D), h (1,H) -> (logits (1,K), alpha (1,pd), beta (1,pd),
        h_new (1,H))."""
        seq, h_new = self.pi_gru(obs1, initial_state=h)
        out = self.pi_head(seq)[:, 0, :]          # (1, K+2pd)
        K, pd = self.n_actions, self.param_dim
        logits = out[:, :K]
        ab = tf.nn.softplus(tf.reshape(out[:, K:], (-1, pd, 2))) + 1.0
        return logits, ab[:, :, 0], ab[:, :, 1], h_new

    @tf.function(reduce_retracing=True)
    def _v_step(self, obs1, h):
        """obs1 (1,1,D), h (1,H) -> (value (1,), h_new (1,H))."""
        seq, h_new = self.v_gru(obs1, initial_state=h)
        v = self.v_head(seq)[:, 0, 0]
        return v, h_new

    # --------------------------------------------------------- distribution math
    def _logprob(self, logits, alpha, beta, k, params):
        """All args carry a leading (...,) batch/time shape; returns that shape."""
        logp_cat = tf.reduce_sum(
            tf.nn.log_softmax(logits) * tf.one_hot(k, self.n_actions), axis=-1)
        x = tf.clip_by_value(params, _LOGPROB_EPS, 1.0 - _LOGPROB_EPS)
        logB = (tf.math.lgamma(alpha) + tf.math.lgamma(beta)
                - tf.math.lgamma(alpha + beta))
        logp_beta = (alpha - 1.0) * tf.math.log(x) \
            + (beta - 1.0) * tf.math.log(1.0 - x) - logB
        return logp_cat + tf.reduce_sum(logp_beta, axis=-1)

    def _entropy(self, logits, alpha, beta):
        logp = tf.nn.log_softmax(logits)
        cat_ent = -tf.reduce_sum(tf.exp(logp) * logp, axis=-1)
        logB = (tf.math.lgamma(alpha) + tf.math.lgamma(beta)
                - tf.math.lgamma(alpha + beta))
        beta_ent = (logB - (alpha - 1.0) * tf.math.digamma(alpha)
                    - (beta - 1.0) * tf.math.digamma(beta)
                    + (alpha + beta - 2.0) * tf.math.digamma(alpha + beta))
        return cat_ent + tf.reduce_sum(beta_ent, axis=-1)

    # ------------------------------------------------------ collection / eval
    def act_collect(self, obs, state):
        """Sample from the current policy, threading the hidden state.
        Returns ((k, params), logprob, value, new_state)."""
        h_pi, h_v = state
        o = tf.constant(np.asarray(obs, dtype=np.float32)[None, None, :])
        logits, alpha, beta, h_pi2 = self._pi_step(
            o, tf.constant(h_pi[None, :]))
        p = tf.nn.softmax(logits)[0].numpy().astype(np.float64)
        p = p / p.sum()
        k = int(self.rng.choice(self.n_actions, p=p))
        a = alpha[0].numpy(); b = beta[0].numpy()
        params = self.rng.beta(a, b).astype(np.float32)
        logp = float(self._logprob(
            logits, alpha, beta,
            tf.constant([k]), tf.constant(params[None, :]))[0].numpy())
        v, h_v2 = self._v_step(o, tf.constant(h_v[None, :]))
        new_state = (h_pi2[0].numpy(), h_v2[0].numpy())
        return (k, params), logp, float(v[0].numpy()), new_state

    def act_eval(self, obs, state):
        """Deterministic greedy action (argmax strategy, Beta means), threading
        the policy hidden state. Returns ((k, params), new_state)."""
        h_pi, h_v = state
        o = tf.constant(np.asarray(obs, dtype=np.float32)[None, None, :])
        logits, alpha, beta, h_pi2 = self._pi_step(
            o, tf.constant(h_pi[None, :]))
        k = int(tf.argmax(logits[0]).numpy())
        params = (alpha / (alpha + beta))[0].numpy().astype(np.float32)
        return (k, params), (h_pi2[0].numpy(), h_v)

    # ---------------------------------------------------------------- update
    @tf.function(reduce_retracing=True)
    def _update_minibatch(self, obs, k, params, old_logp, adv, ret, mask):
        """obs (B,T,D); k (B,T); params (B,T,pd); old_logp/adv/ret/mask (B,T).
        mask is 1.0 on real timesteps, 0.0 on padding."""
        mask_bool = mask > 0.5
        denom = tf.reduce_sum(mask) + 1e-8
        with tf.GradientTape() as tape:
            pi_seq, _ = self.pi_gru(obs, mask=mask_bool,
                                    initial_state=tf.zeros(
                                        (tf.shape(obs)[0], self.rnn_hidden)))
            out = self.pi_head(pi_seq)
            K, pd = self.n_actions, self.param_dim
            logits = out[:, :, :K]
            ab = tf.nn.softplus(tf.reshape(out[:, :, K:],
                                           (tf.shape(obs)[0], tf.shape(obs)[1],
                                            pd, 2))) + 1.0
            alpha, beta = ab[:, :, :, 0], ab[:, :, :, 1]
            new_logp = self._logprob(logits, alpha, beta, k, params)
            ratio = tf.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
            policy_loss = -tf.reduce_sum(
                tf.minimum(surr1, surr2) * mask) / denom

            v_seq, _ = self.v_gru(obs, mask=mask_bool,
                                  initial_state=tf.zeros(
                                      (tf.shape(obs)[0], self.rnn_hidden)))
            value_pred = self.v_head(v_seq)[:, :, 0]
            value_loss = tf.reduce_sum(
                ((value_pred - ret) ** 2) * mask) / denom

            entropy = tf.reduce_sum(
                self._entropy(logits, alpha, beta) * mask) / denom
            loss = (policy_loss + self.value_coef * value_loss
                    - self.entropy_coef * entropy)
        variables = (self.pi_gru.trainable_variables
                     + self.pi_head.trainable_variables
                     + self.v_gru.trainable_variables
                     + self.v_head.trainable_variables)
        grads = tape.gradient(loss, variables)
        # Divergence guard (as in hybrid_ppo): zero any non-finite gradient so an
        # ill-conditioned function can never NaN the policy. BPTT can raise how
        # often this fires; it is a no-op when gradients are finite.
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                 if g is not None else g for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, variables))
        return loss

    def update(self, episodes):
        """K epochs of clipped-surrogate SGD over whole-episode minibatches.

        `episodes`: list of per-episode dicts with arrays obs (T,D), k (T,),
        params (T,pd), old_logp (T,), adv (T,), ret (T,). Advantages are
        normalized here across all real timesteps. Episodes are padded to the
        rollout's global max length; a per-timestep mask zeros the padding in
        both the GRU (variable-length) and the loss. Returns the last minibatch
        loss for the learning-curve log."""
        if not episodes:
            return float("nan")
        T = max(len(e["k"]) for e in episodes)
        pd = self.param_dim

        # Normalize advantages over real (unpadded) timesteps.
        all_adv = np.concatenate([np.asarray(e["adv"], dtype=np.float64)
                                  for e in episodes])
        a_mean, a_std = all_adv.mean(), all_adv.std() + 1e-8

        def pad(ep):
            t = len(ep["k"])
            obs = np.zeros((T, self.obs_dim), np.float32)
            obs[:t] = ep["obs"]
            k = np.zeros(T, np.int32); k[:t] = ep["k"]
            par = np.zeros((T, pd), np.float32); par[:t] = ep["params"]
            olp = np.zeros(T, np.float32); olp[:t] = ep["old_logp"]
            adv = np.zeros(T, np.float32)
            adv[:t] = (np.asarray(ep["adv"]) - a_mean) / a_std
            ret = np.zeros(T, np.float32); ret[:t] = ep["ret"]
            mask = np.zeros(T, np.float32); mask[:t] = 1.0
            return obs, k, par, olp, adv, ret, mask

        packed = [pad(e) for e in episodes]
        n = len(packed)
        idx = np.arange(n)
        mb = max(1, self.minibatch_episodes)
        last = 0.0
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, mb):
                sel = idx[start:start + mb]
                batch = [packed[j] for j in sel]
                obs = tf.constant(np.stack([b[0] for b in batch]))
                k = tf.constant(np.stack([b[1] for b in batch]))
                par = tf.constant(np.stack([b[2] for b in batch]))
                olp = tf.constant(np.stack([b[3] for b in batch]))
                adv = tf.constant(np.stack([b[4] for b in batch]))
                ret = tf.constant(np.stack([b[5] for b in batch]))
                msk = tf.constant(np.stack([b[6] for b in batch]))
                loss = self._update_minibatch(obs, k, par, olp, adv, ret, msk)
                self.train_steps.assign_add(1)
                last = float(loss.numpy())
        return last

    # ------------------------------------------------------------ plumbing
    def checkpoint_items(self):
        return {
            "pi_gru": self.pi_gru,
            "pi_head": self.pi_head,
            "v_gru": self.v_gru,
            "v_head": self.v_head,
            "optimizer": self.optimizer,
            "train_steps": self.train_steps,
        }
