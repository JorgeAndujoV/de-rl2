"""Simple uniform circular replay buffer (NumPy storage).

Ported from de-rl with one change: each transition carries an additional `tau`
field — the segment's elapsed budget in units of the smallest budget fraction
(spec §6.5) — so the DQN can discount by γ^τ (SMDP) rather than a flat γ. tau
travels with the transition because the discount belongs to the transition, not
to a global step count that replay would otherwise scramble.
"""

import json
import os

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

    # --------------------------------------------------------- persistence
    # The buffer is NOT part of the tf.train.Checkpoint (it is plain NumPy), so
    # it is saved/restored separately. Without this a resumed run would start
    # with an empty buffer and retrain on a tiny, correlated set of fresh
    # transitions — a discontinuity that makes a segmented run differ from an
    # uninterrupted one. Persisting it keeps a checkpoint→resume exactly
    # continuous, and lets a finished run be extended later.
    #
    # Only the `size` filled rows are written (compressed); the ring index and
    # the sampler's RNG state travel with them so `add`/`sample` continue
    # identically after a load.
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            obs=self.obs[:self.size], actions=self.actions[:self.size],
            rewards=self.rewards[:self.size], next_obs=self.next_obs[:self.size],
            dones=self.dones[:self.size], tau=self.tau[:self.size],
            idx=np.int64(self.idx), size=np.int64(self.size),
            capacity=np.int64(self.capacity),
            rng_state=np.array(json.dumps(self.rng.bit_generator.state)),
        )

    def load(self, path):
        """Restore contents saved by save(). Fails loudly on an obs-dim or
        capacity mismatch rather than silently loading an incompatible buffer."""
        data = np.load(path, allow_pickle=False)
        size = int(data["size"])
        if data["obs"].shape[1:] != self.obs.shape[1:]:
            raise ValueError(
                f"Replay buffer obs shape {data['obs'].shape[1:]} in {path} "
                f"does not match this buffer's {self.obs.shape[1:]}."
            )
        if int(data["capacity"]) != self.capacity:
            raise ValueError(
                f"Replay buffer capacity {int(data['capacity'])} in {path} "
                f"does not match this buffer's {self.capacity}."
            )
        self.obs[:size] = data["obs"]
        self.actions[:size] = data["actions"]
        self.rewards[:size] = data["rewards"]
        self.next_obs[:size] = data["next_obs"]
        self.dones[:size] = data["dones"]
        self.tau[:size] = data["tau"]
        self.idx = int(data["idx"])
        self.size = size
        self.rng.bit_generator.state = json.loads(str(data["rng_state"]))
        return self
