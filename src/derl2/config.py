"""Configuration loader for the DE+RL experiment workflow.

In short:

  * Each experiment's config.yaml declares every parameter explicitly (§7 of
    the build spec).
  * There is no inheritance between configs. What you read is what runs.
  * Once an experiment has run, its config.yaml is frozen. A different
    parameter value means a NEW experiment, never an edit to an old one.
  * There are NO hidden defaults. What the file declares is the whole story;
    every knob a run depended on — including SLURM resources — is written in
    config.yaml and recorded verbatim in each job's run_metadata.json (under
    resolved_config). A missing key is an error, not a silently-filled fallback:
    one config.yaml plus the modules it names must fully explain what ran.

This module deliberately knows nothing about which agents, observations,
rewards, action spaces, or strategies exist. It reads and writes nested keys
and checks a few structural invariants; anything type-specific is validated by
the component that owns it.

The loader mechanics (dotted get/set, deep-merge overrides, --smoke scaling, the
frozen resolved config) are carried over from de-rl unchanged; the schema it
validates is the de-rl2 schema of §7, and the fallback shim de-rl used is
removed — de-rl2 has no hidden defaults.

Typical use:

    from derl2.config import Config

    cfg = Config.from_args()
    obs_name = cfg.get("environment.observation")
    # what a job actually ran is recorded in its run_metadata.json
    # (resolved_config); see derl2.run_metadata.
"""

import argparse
import copy
import os

import yaml


# --------------------------------------------------------------- locations

def repo_root():
    """Repo root, inferred from this file's path (src/derl2/config.py)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def experiments_dir():
    """Where experiment folders live. Override with $DERL2_EXPERIMENTS_DIR."""
    return os.environ.get("DERL2_EXPERIMENTS_DIR",
                          os.path.join(repo_root(), "experiments"))


# ----------------------------------------------------------------- overrides

# Scaled-down overrides applied by --smoke. Workflow Stage 1 (WSL) and Stage 3
# (cluster) share this, so "smoke" means the same thing in both and a cluster
# smoke failure is a real environment problem rather than a difference in flags
# typed by hand. n_checkpoints is deliberately NOT scaled: the observation
# length (traj20 -> 20 checkpoints) depends on it, so a smoke run must keep the
# same feature dimension as the real run.
SMOKE_OVERRIDES = {
    "benchmark": {"dim": 5, "budget": 3000},
    "episode": {"pop_size": 20},
    "agent": {
        "buffer_size": 2000,
        "batch_size": 32,
        "warmup_transitions": 32,
        "epsilon": {"decay_steps": 500},
    },
    "training": {
        "episodes": 3,
        "eval_every": 50,
        "eval_episodes": 2,
        "checkpoint_every": 50,
    },
    "evaluation": {"runs_per_function": 2},
}

# Structural keys every config must contain. Deliberately short: this module
# checks the SHAPE of a config, not the meaning of any component's parameters.
REQUIRED_KEYS = [
    "experiment",
    "seed",
    "benchmark.name",
    "benchmark.functions",
    "benchmark.dim",
    "benchmark.budget",
    "episode.pop_size",
    "environment.observation",
    "environment.action_space",
    "environment.reward.name",
    "agent.name",
    "training.episodes",
    "slurm.time",
    "slurm.mem",
    "slurm.cpus",
    "slurm.gpus",
    "slurm.account",
]


class ConfigError(Exception):
    """Raised when a configuration is missing required keys or malformed."""


# ----------------------------------------------------------------- helpers

def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base`."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_scalar(text):
    """Parse a CLI override value with YAML rules (int, float, bool, list)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


_MISSING = object()   # distinguishes "no default given" from "default is None"


# ------------------------------------------------------------------ Config

class Config:
    def __init__(self, data, source=None):
        self._data = data
        self.source = source                    # path it was loaded from
        self.validate()

    # -------------------------------------------------------- construction
    @classmethod
    def from_file(cls, path, overrides=None, smoke=False):
        """Load an experiment config.

        Order (later wins): file contents -> --smoke scaling -> --set overrides.
        No keys are filled in: a config missing a required key fails to load.
        """
        if not os.path.exists(path):
            raise ConfigError(f"Config file not found: {path}")
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config file must contain a mapping: {path}")

        if smoke:
            # Scale everything except the agent block wholesale. For the agent
            # block, scale ONLY keys the config already declares — otherwise a
            # DQN-shaped smoke override (buffer_size, epsilon, ...) is injected
            # into an agent whose constructor does not accept it (PPO/SAC have no
            # buffer_size/epsilon; DQN/MP-DQN do), which build_agent rejects.
            smoke_ovr = copy.deepcopy(SMOKE_OVERRIDES)
            agent_ovr = smoke_ovr.pop("agent", {})
            data = _deep_merge(data, smoke_ovr)
            agent_cfg = data.get("agent")
            if isinstance(agent_cfg, dict):
                for k, v in agent_ovr.items():
                    if k not in agent_cfg:
                        continue                 # agent doesn't declare it -> skip
                    if isinstance(v, dict) and isinstance(agent_cfg.get(k), dict):
                        agent_cfg[k] = _deep_merge(agent_cfg[k], v)
                    else:
                        agent_cfg[k] = v
            data["smoke"] = True
            # If the experiment does periodic in-process evaluation, scale its
            # cadence down too — otherwise a 3-episode smoke never reaches
            # periodic_eval.start (e.g. 1000) and the eval path goes untested,
            # defeating the point of a smoke. Only applied when the config
            # already declares periodic_eval (EXP001/EXP002 are unaffected).
            if isinstance(data.get("training"), dict) \
                    and "periodic_eval" in data["training"]:
                data["training"]["periodic_eval"] = {"start": 1, "every": 1}

        cfg = cls(data, source=path)

        for item in overrides or []:
            if "=" not in item:
                raise ConfigError(f"Override must be KEY=VALUE, got: {item!r}")
            key, value = item.split("=", 1)
            cfg.set(key.strip(), _parse_scalar(value.strip()))
        if overrides:
            cfg.validate()

        return cfg

    @classmethod
    def from_experiment(cls, exp_id, **kwargs):
        """Load experiments/<exp_id>/config.yaml."""
        return cls.from_file(
            os.path.join(experiments_dir(), exp_id, "config.yaml"), **kwargs
        )

    @classmethod
    def from_args(cls, argv=None, extra_args=None):
        """Parse --config / --exp / --smoke / --set from the command line.

        If `extra_args` (a callable receiving the parser) is given, returns
        (config, namespace); otherwise returns the config alone.
        """
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--config", help="Path to an experiment config.yaml")
        group.add_argument("--exp", help="Experiment id "
                                         "(loads experiments/<id>/config.yaml)")
        parser.add_argument("--smoke", action="store_true",
                            help="Tiny scaled-down run; checks for crashes only")
        parser.add_argument("--set", action="append", default=[],
                            metavar="KEY=VALUE",
                            help="Override a value, e.g. "
                                 "--set agent.lr=3e-4")
        if extra_args is not None:
            extra_args(parser)

        args = parser.parse_args(argv)
        path = args.config or os.path.join(experiments_dir(), args.exp,
                                           "config.yaml")
        cfg = cls.from_file(path, args.set, smoke=args.smoke)
        return (cfg, args) if extra_args is not None else cfg

    # -------------------------------------------------------------- access
    def get(self, dotted_key, default=_MISSING):
        """Read a value by dotted path, e.g. get("benchmark.dim").

        Raises ConfigError if absent and no default is given: a typo'd key
        silently becoming None is an expensive bug to chase.
        """
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(
                        f"Missing config key '{dotted_key}' in {self.source}"
                    )
                return default
            node = node[part]
        return node

    def set(self, dotted_key, value):
        """Write a value by dotted path, creating intermediate dicts as needed."""
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def block(self, dotted_key):
        """A whole config block as a plain deep-copied dict, ready to pass on."""
        return copy.deepcopy(self.get(dotted_key))

    def as_dict(self):
        return copy.deepcopy(self._data)

    @property
    def exp_id(self):
        return self.get("experiment")

    @property
    def is_smoke(self):
        return bool(self.get("smoke", default=False))

    # ---------------------------------------------------------- validation
    def validate(self):
        """Check structural invariants only — not component-specific values."""
        for key in REQUIRED_KEYS:
            self.get(key)   # raises ConfigError if absent

        if not isinstance(self.exp_id, str) or not self.exp_id.strip():
            raise ConfigError("experiment must be a non-empty string")

        seed = self.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigError(f"seed must be an integer, got {seed!r}")

        for block in ("benchmark", "episode", "environment", "agent",
                      "training", "evaluation"):
            if not isinstance(self.get(block, default={}), dict):
                raise ConfigError(f"{block} must be a mapping")

        for key in ("agent.name", "environment.observation",
                    "environment.action_space", "environment.reward.name"):
            if not isinstance(self.get(key), str):
                raise ConfigError(f"{key} must be a string")

        episodes = self.get("training.episodes")
        if isinstance(episodes, bool) or not isinstance(episodes, int) \
                or episodes <= 0:
            raise ConfigError(
                f"training.episodes must be a positive integer, got {episodes!r}"
            )

    # -------------------------------------------------------------- output
    def save(self, path):
        """Write the fully resolved config to `path` as YAML.

        A utility for inspecting a resolved config; the authoritative per-job
        record is run_metadata.json (resolved_config), written by
        derl2.run_metadata — not this file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.as_dict()
        payload["_source"] = os.path.relpath(self.source, repo_root()) \
            if self.source else None
        with open(path, "w") as fh:
            yaml.safe_dump(payload, fh, default_flow_style=False,
                           sort_keys=False)
        return path

    def summary(self):
        """One-line description for job logs and console output."""
        return (f"{self.exp_id} | agent={self.get('agent.name')} "
                f"obs={self.get('environment.observation')} "
                f"reward={self.get('environment.reward.name')} "
                f"functions={self.get('benchmark.functions')} "
                f"dim={self.get('benchmark.dim')} "
                f"episodes={self.get('training.episodes')} "
                f"seed={self.get('seed')}"
                + (" [SMOKE]" if self.is_smoke else ""))

    def diff(self, other):
        """Dotted keys where this config differs from `other`.

        Answers "what was actually different between these two experiments?"
        Returns {key: (other_value, this_value)}.
        """
        def flatten(node, prefix=""):
            flat = {}
            for key, value in node.items():
                dotted = f"{prefix}{key}"
                if isinstance(value, dict):
                    flat.update(flatten(value, dotted + "."))
                else:
                    flat[dotted] = value
            return flat

        mine, theirs = flatten(self.as_dict()), flatten(other.as_dict())
        return {k: (theirs.get(k), mine.get(k))
                for k in set(mine) | set(theirs)
                if mine.get(k) != theirs.get(k)}

    def __repr__(self):
        return f"<Config {self.exp_id} from {self.source}>"


# ------------------------------------------------------------ shell access

def _main():
    """Inspect configs from the shell. Used by run_experiment.sh.

        python -m derl2.config --exp EXP001_slug
        python -m derl2.config --exp EXP001_slug slurm.mem
        python -m derl2.config --exp EXP001_slug --resolved
        python -m derl2.config --exp EXP007_foo --diff EXP001_slug
    """
    parser = argparse.ArgumentParser(description="Inspect a derl2 config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--exp")
    parser.add_argument("key", nargs="?",
                        help="Dotted key to print, e.g. slurm.mem")
    parser.add_argument("--resolved", action="store_true",
                        help="Print the full config as YAML")
    parser.add_argument("--diff", metavar="EXP_ID",
                        help="Show differences against another experiment")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    path = args.config or os.path.join(experiments_dir(), args.exp,
                                       "config.yaml")
    cfg = Config.from_file(path, smoke=args.smoke)

    if args.diff:
        changes = cfg.diff(Config.from_experiment(args.diff))
        if not changes:
            print(f"No differences vs {args.diff}")
        for key in sorted(changes):
            theirs, mine = changes[key]
            print(f"{key}: {theirs!r} -> {mine!r}")
    elif args.resolved:
        print(yaml.safe_dump(cfg.as_dict(), default_flow_style=False,
                             sort_keys=False))
    elif args.key:
        print(cfg.get(args.key))
    else:
        print(cfg.summary())


if __name__ == "__main__":
    _main()
