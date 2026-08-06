"""Replay storage. Transitions carry a `tau` field for SMDP (γ^τ) discounting
(spec §6.5). ReplayBuffer stores a scalar action (DQN); ParamReplayBuffer stores
a (discrete index, continuous parameter vector) action (parameterized agents)."""

from derl2.replay.buffer import ReplayBuffer
from derl2.replay.param_buffer import ParamReplayBuffer

__all__ = ["ReplayBuffer", "ParamReplayBuffer"]
