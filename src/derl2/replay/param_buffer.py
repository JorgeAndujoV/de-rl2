"""Replay buffer for parameterized actions (MP-DQN and other hybrid agents).

Identical contract to ReplayBuffer — same tau/SMDP field, same persistence, same
save/load-a-resume-pair discipline — except the action is a (discrete index k,
continuous parameter vector) pair rather than a single integer. The discrete
index and the raw [0,1] parameters are stored in separate arrays; `sample`
returns them as separate tensors so the agent's train step can mask the params
into the taken action's block (the multi-pass parameterized-DQN update).
"""

import json
import os

import numpy as np
import tensorflow as tf


class ParamReplayBuffer:
    def __init__(self, capacity, obs_dim, param_dim, seed=0):
        self.capacity = capacity
        self.param_dim = param_dim
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)             # k
        self.params = np.zeros((capacity, param_dim), dtype=np.float32)  # raw[0,1]
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.tau = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, tau):
        """`action` is the (k, params) pair the parameterized action space emits."""
        k, params = action
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = int(k)
        self.params[i] = np.asarray(params, dtype=np.float32).reshape(self.param_dim)
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
            tf.constant(self.params[idx]),
            tf.constant(self.rewards[idx]),
            tf.constant(self.next_obs[idx]),
            tf.constant(self.dones[idx]),
            tf.constant(self.tau[idx]),
        )

    # --------------------------------------------------------- persistence
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            obs=self.obs[:self.size], actions=self.actions[:self.size],
            params=self.params[:self.size], rewards=self.rewards[:self.size],
            next_obs=self.next_obs[:self.size], dones=self.dones[:self.size],
            tau=self.tau[:self.size],
            idx=np.int64(self.idx), size=np.int64(self.size),
            capacity=np.int64(self.capacity), param_dim=np.int64(self.param_dim),
            rng_state=np.array(json.dumps(self.rng.bit_generator.state)),
        )

    def load(self, path):
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
        if int(data["param_dim"]) != self.param_dim:
            raise ValueError(
                f"Replay buffer param_dim {int(data['param_dim'])} in {path} "
                f"does not match this buffer's {self.param_dim}."
            )
        self.obs[:size] = data["obs"]
        self.actions[:size] = data["actions"]
        self.params[:size] = data["params"]
        self.rewards[:size] = data["rewards"]
        self.next_obs[:size] = data["next_obs"]
        self.dones[:size] = data["dones"]
        self.tau[:size] = data["tau"]
        self.idx = int(data["idx"])
        self.size = size
        self.rng.bit_generator.state = json.loads(str(data["rng_state"]))
        return self
