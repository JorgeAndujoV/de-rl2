"""Simple uniform circular replay buffer (NumPy storage).

Ported from de-rl with one change: each transition carries an additional `tau`
field — the segment's elapsed budget in units of the smallest budget fraction
(spec §6.5) — so the DQN can discount by γ^τ (SMDP) rather than a flat γ. tau
travels with the transition because the discount belongs to the transition, not
to a global step count that replay would otherwise scramble.
"""

import numpy as np
import tensorflow as tf


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, seed=0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.tau = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, tau):
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.tau[i] = tau
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            tf.constant(self.obs[idx]),
            tf.constant(self.actions[idx]),
            tf.constant(self.rewards[idx]),
            tf.constant(self.next_obs[idx]),
            tf.constant(self.dones[idx]),
            tf.constant(self.tau[idx]),
        )
