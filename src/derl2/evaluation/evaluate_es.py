"""Independent evaluation of a Sep-CMA-ES policy (EXP026) -- additive.

The standard evaluator (derl2.evaluation.evaluate) restores a trained policy via
build_agent(agent_name, ...), i.e. through the agent REGISTRY. Sep-CMA-ES is
deliberately NOT registered (isolation: no shared-code edits), so this sibling
entry point restores the policy a different way -- load the Sep-CMA-ES weight
vector (checkpoints/mean.npy, or best_mean.npy) into the Hybrid-PPO policy net --
and then calls the SAME agent-agnostic evaluate_policy. That means EXP026 gets
byte-identical SS8 outputs (episode_results, step_trace, summary, eval_results,
comparison, trajectories) and the identical paired baseline comparison every
PPO experiment gets, with zero change to evaluate.py or the registry.

    python -m derl2.evaluation.evaluate_es --train-dir experiments/EXP026_.../train/f11
    python -m derl2.evaluation.evaluate_es --train-dir ... --weights mean

By default (`--weights auto`) this evaluates `best_mean.npy` (the mean whose
validation median was best during training -- an early-stopping-style choice)
when a validation monitor ran, else falls back to `mean.npy` (the final
Sep-CMA-ES distribution mean). Pass `--weights mean` to force the final mean.
Reads the config the training job actually ran from run_metadata.json, exactly
as evaluate.py does, so a smoke run evaluates the smoke-scaled config.
"""

import argparse
import os

import numpy as np

from derl2 import run_metadata
from derl2.environments import DEEnv
from derl2.evaluation.evaluate import GreedyPolicy, evaluate_policy
from derl2.agents.sep_cmaes import build_policy_net, set_flat


def _resolve_weights(train_dir, weights):
    ckpt = os.path.join(train_dir, "checkpoints")
    if weights == "auto":
        # Prefer the early-stopping pick (best validation median); fall back to
        # the final distribution mean if no monitor ran (e.g. eval_every=0). On
        # the oscillating hard functions the final mean can overshoot its best
        # validation point, so best_mean is the better default deployable.
        best = os.path.join(ckpt, "best_mean.npy")
        path = best if os.path.exists(best) else os.path.join(ckpt, "mean.npy")
    elif weights in ("mean", "best_mean"):
        path = os.path.join(ckpt, f"{weights}.npy")
    else:
        path = weights                       # explicit path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sep-CMA-ES weights not found at {path}. Train first (train_es.py "
            f"writes checkpoints/mean.npy and, once a validation eval runs, "
            f"best_mean.npy), or pass --weights <path>.")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True,
                        help="Sep-CMA-ES job dir (has run_metadata.json, "
                             "checkpoints/mean.npy)")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write result files; defaults to the "
                             "train dir itself (matches evaluate.py).")
    parser.add_argument("--function", type=int, default=None,
                        help="Restrict evaluation to one function id.")
    parser.add_argument("--weights", default="auto",
                        help="auto (default: best_mean if a validation monitor "
                             "ran, else mean), mean, best_mean, or a path to a "
                             ".npy flat weight vector.")
    args = parser.parse_args()

    # Config the training job actually ran (smoke-scaled if it was a smoke job).
    cfg = run_metadata.load_resolved_config(args.train_dir)
    if args.function is not None:
        cfg.set("benchmark.functions", [args.function])
    env = DEEnv.from_config(cfg)
    print(cfg.summary())
    print(env.describe())

    wpath = _resolve_weights(args.train_dir, args.weights)
    flat = np.load(wpath)
    param_dim = int(cfg.get("agent.param_dim"))
    policy_hidden = cfg.get("agent.policy_hidden")
    agent = build_policy_net(env.obs_dim, env.n_actions, param_dim,
                             policy_hidden, seed=int(cfg.get("seed")))
    expected = int(sum(int(np.prod(v.shape))
                       for v in agent.policy_net.trainable_variables))
    if flat.shape[0] != expected:
        raise ValueError(
            f"weight vector {wpath} has {flat.shape[0]} params but the policy "
            f"net has {expected}; config and checkpoint disagree.")
    set_flat(agent.policy_net.trainable_variables, flat)
    print(f"loaded Sep-CMA-ES weights: {wpath} ({flat.shape[0]} params)")

    # GreedyPolicy(agent) == agent.act(obs, epsilon=0) for this feedforward net,
    # i.e. the exact deterministic decode used in training and in every eval.
    policy = GreedyPolicy(agent)
    evaluate_policy(cfg, env, policy, args.out_dir or args.train_dir)


if __name__ == "__main__":
    main()
