"""Environment package.

There is deliberately no ENVIRONMENTS registry (spec §4): there is a single
environment, `DEEnv`, and its variants are expressed through the observation,
action-space, and reward registries plus config flags — not by swapping the
environment class. It is constructed directly with `DEEnv.from_config(cfg)`
rather than looked up by name.

The observation / action-space / reward registries live in their own modules
(observations.py, action_spaces.py, rewards.py); the sampling-box transform in
sampling_box.py.
"""

from derl2.environments.env import DEEnv

__all__ = ["DEEnv"]
