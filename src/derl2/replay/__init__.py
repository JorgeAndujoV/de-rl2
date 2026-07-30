"""Replay storage. One buffer; transitions carry a `tau` field for SMDP
(γ^τ) discounting (spec §6.5)."""

from derl2.replay.buffer import ReplayBuffer

__all__ = ["ReplayBuffer"]
