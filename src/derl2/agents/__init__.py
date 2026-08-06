"""Agent registry.

This dictionary IS the documentation of what agents exist. To add one:

  1. Write the class in its own module in this package.
  2. Import it below and add one line to AGENTS.
  3. Reference the name in a new experiment's config.yaml.

Nothing else changes — not train.py, not evaluate.py, not config.py. An agent
knows only its observation and action dimensions (principle 4); the environment
is invisible to it.
"""

from derl2.agents.dqn import DQNAgent
from derl2.agents.mpdqn import MPDQNAgent
from derl2.agents.hybrid_ppo import HybridPPOAgent
from derl2.agents.hybrid_sac import HybridSACAgent

AGENTS = {
    "dqn": DQNAgent,
    "mpdqn": MPDQNAgent,
    "hybrid_ppo": HybridPPOAgent,
    "hybrid_sac": HybridSACAgent,
}


def build_agent(agent_name, obs_dim, n_actions, params, seed=0):
    """Instantiate the agent named by `agent_name`.

    `params` is the config's agent block with `name` removed, passed through as
    keyword arguments. An unknown name fails immediately with the valid options
    listed, rather than falling back to a default.
    """
    if agent_name not in AGENTS:
        raise KeyError(
            f"Unknown agent {agent_name!r}. Available: {sorted(AGENTS)}. "
            f"Add it to AGENTS in src/derl2/agents/__init__.py."
        )
    try:
        return AGENTS[agent_name](obs_dim, n_actions, seed=seed, **params)
    except TypeError as err:
        raise TypeError(
            f"Could not build agent {agent_name!r} from config agent block: "
            f"{err}. Check that every key in agent (except name) is a "
            f"constructor argument of {AGENTS[agent_name].__name__}."
        ) from err
