"""Training entry point.

Builds the agent and environment from an experiment's config.yaml, so this file
never needs editing to accommodate a new agent, benchmark, reward, observation,
or DE strategy — those are registry choices named in the config.

    python -m derl2.training.train --exp EXP001_slug
    python -m derl2.training.train --exp EXP001_slug --smoke
    python -m derl2.training.train --config path/to/config.yaml --out-dir DIR

Re-issuing the identical command resumes from the latest checkpoint.

Three-stage discipline (spec §8): smoke / train / train_eval write the SAME
files with the SAME schema at different scale, so a smoke run is a structural
rehearsal of a real one. The evaluation-checkpoint cadence is in TRAINING STEPS
(gradient updates), matching the config's step-based eval_every / decay_steps.

training_log.csv rows are written at each evaluation checkpoint:
    training_step, elapsed_seconds, mean_return, std_return,
    mean_loss, mean_eval_fitness, std_eval_fitness, mean_episode_length
elapsed_seconds is cumulative wall-clock — the source for reported training time
— and mean_episode_length tracks whether the agent learns to take more or fewer
restarts.
"""

import csv
import os
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

from derl2.agents import build_agent
from derl2.config import Config, experiments_dir
from derl2.environments import DEEnv
from derl2.replay import ReplayBuffer

# Seed offset for the greedy evaluation rollouts run during training. Kept well
# clear of training seeds (seed·1e6 + episode) and of final-eval seeds
# (evaluation.seed_offset, 900000), so the learning-curve signal never reuses a
# seed the agent trained on.
_TRAIN_EVAL_SEED_OFFSET = 800_000


def _extra_args(parser):
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Defaults to "
                             "experiments/<exp_id>/<kind>/job_<id>/")
    parser.add_argument("--kind", default="train",
                        help="Run kind: train | smoke | dev")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop cleanly after this many hours, "
                             "checkpointing first (cluster walltime safety)")


def git_commit():
    """Current commit, so every run can be traced back to exact code."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def resolve_out_dir(cfg, args):
    """experiments/<exp_id>/<kind>/job_<slurm_id or 'local'>/"""
    if args.out_dir:
        return args.out_dir
    kind = "smoke" if cfg.is_smoke else args.kind
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return os.path.join(experiments_dir(), cfg.exp_id, kind, f"job_{job_id}")


def eval_rollouts(env, agent, n_episodes):
    """Greedy rollouts for the learning curve; returns final errors.

    Uses a dedicated env so it never disturbs the training episode, and a fixed
    seed set so successive checkpoints are comparable."""
    fits = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=_TRAIN_EVAL_SEED_OFFSET + i)
        done = False
        info = {}
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs, epsilon=0.0))
            done = term or trunc
        fits.append(info["best_error"])
    return np.array(fits, dtype=np.float64)


def main():
    cfg, args = Config.from_args(extra_args=_extra_args)
    out_dir = resolve_out_dir(cfg, args)
    os.makedirs(out_dir, exist_ok=True)

    # --- build from config; nothing below names a concrete class ---
    env = DEEnv.from_config(cfg)
    eval_env = DEEnv.from_config(cfg)          # separate, for learning-curve eval
    seed = cfg.get("seed")

    agent_block = cfg.block("agent")
    agent_name = agent_block.pop("name")
    agent = build_agent(agent_name, env.obs_dim, env.n_actions, agent_block,
                        seed=seed)
    buffer = ReplayBuffer(agent.buffer_size, env.obs_dim, seed=seed)

    commit = git_commit()
    print(cfg.summary())
    print(env.describe())
    print(f"commit {commit} | output {out_dir}")

    # The authoritative record of what this job ran.
    cfg.save(os.path.join(out_dir, "config_used.yaml"))
    with open(os.path.join(out_dir, "git_commit.txt"), "w") as fh:
        fh.write(commit + "\n")

    # --- checkpointing ---
    episode_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt = tf.train.Checkpoint(episode=episode_var, **agent.checkpoint_items())
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(out_dir, "checkpoints"), max_to_keep=3
    )
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Resumed from {manager.latest_checkpoint} "
              f"(episode {int(episode_var.numpy())})")

    log_path = os.path.join(out_dir, "training_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    logger = csv.writer(log_file)
    if write_header:
        logger.writerow(["training_step", "elapsed_seconds", "mean_return",
                         "std_return", "mean_loss", "mean_eval_fitness",
                         "std_eval_fitness", "mean_episode_length"])

    episodes = cfg.get("training.episodes")
    eval_every = cfg.get("training.eval_every")
    eval_episodes = cfg.get("training.eval_episodes")
    checkpoint_every = cfg.get("training.checkpoint_every")
    batch_size = agent.batch_size
    warmup = agent.warmup_transitions

    start = int(episode_var.numpy())
    t_start = time.perf_counter()      # monotonic; time.time() can step back

    # Cadence is in training steps; recompute the next thresholds on resume.
    ts0 = int(agent.train_steps.numpy())
    next_eval_at = (ts0 // eval_every + 1) * eval_every
    next_ckpt_at = (ts0 // checkpoint_every + 1) * checkpoint_every

    # Accumulators over the training episodes since the last logged checkpoint.
    returns, lengths, losses = [], [], []

    def log_checkpoint():
        ts = int(agent.train_steps.numpy())
        fits = eval_rollouts(eval_env, agent, eval_episodes)
        logger.writerow([
            ts, f"{time.perf_counter() - t_start:.2f}",
            f"{np.mean(returns):.6f}" if returns else "nan",
            f"{np.std(returns):.6f}" if returns else "nan",
            f"{np.mean(losses):.6f}" if losses else "nan",
            f"{fits.mean():.6e}", f"{fits.std():.6e}",
            f"{np.mean(lengths):.4f}" if lengths else "nan",
        ])
        log_file.flush()
        print(f"step {ts:6d} | eval_fit {fits.mean():.4e} | "
              f"return {(np.mean(returns) if returns else float('nan')):.3e} | "
              f"ep_len {(np.mean(lengths) if lengths else float('nan')):.1f} | "
              f"{time.perf_counter() - t_start:.0f}s", flush=True)
        returns.clear(); lengths.clear(); losses.clear()

    for episode in range(start, episodes):
        obs, _ = env.reset(seed=seed * 1_000_000 + episode)
        done, ep_return, steps = False, 0.0, 0

        while not done:
            epsilon = agent.epsilon(int(agent.train_steps.numpy()))
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, terminated, info["tau"])
            obs = next_obs
            ep_return += reward
            steps += 1

            if buffer.size >= warmup:
                losses.append(agent.train(buffer.sample(batch_size)))
                ts = int(agent.train_steps.numpy())
                if ts >= next_ckpt_at:
                    manager.save()
                    next_ckpt_at += checkpoint_every
                if ts >= next_eval_at:
                    # Aggregates the episodes completed since the last
                    # checkpoint; the in-progress episode is not yet counted.
                    log_checkpoint()
                    next_eval_at += eval_every

        episode_var.assign(episode + 1)
        returns.append(ep_return)
        lengths.append(steps)

        # Cluster walltime safety: checkpoint and exit cleanly rather than
        # being killed mid-episode.
        if args.max_hours and (time.perf_counter() - t_start) / 3600.0 > args.max_hours:
            manager.save()
            log_file.close()
            print(f"Walltime limit reached at episode {episode + 1}; "
                  f"checkpointed. Re-run the same command to resume.")
            sys.exit(0)

    # A final checkpoint row so the end state is always recorded, even if the
    # last eval threshold fell between here and the previous checkpoint.
    if returns or losses:
        log_checkpoint()
    manager.save()
    log_file.close()
    print(f"Training finished. Output in {out_dir}")


if __name__ == "__main__":
    main()
