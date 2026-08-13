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
import shutil
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

from derl2 import run_metadata
from derl2.agents import build_agent
from derl2.config import Config, experiments_dir
from derl2.environments import DEEnv
from derl2.evaluation.evaluate import (GreedyPolicy, check_baselines_available,
                                       evaluate_policy)

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
    """experiments/<exp_id>/<kind>/  (one job per experiment mode: a fixed path,
    no job_<id> layer — a failed run is deleted and re-run, a changed parameter
    becomes a new EXP00N)."""
    if args.out_dir:
        return args.out_dir
    kind = "smoke" if cfg.is_smoke else args.kind
    return os.path.join(experiments_dir(), cfg.exp_id, kind)


def eval_rollouts(env, agent, n_episodes):
    """Greedy rollouts for the learning curve; returns final errors.

    Uses a dedicated env so it never disturbs the training episode, and a fixed
    seed set so successive checkpoints are comparable. GreedyPolicy is
    recurrent-aware: for a recurrent agent it threads a per-episode hidden state
    (reset here at each episode start), for a feedforward one it is agent.act."""
    policy = GreedyPolicy(agent)
    fits = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=_TRAIN_EVAL_SEED_OFFSET + i)
        policy.reset()
        done = False
        info = {}
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
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
    # The agent owns its replay format (a scalar-action buffer for DQN, a
    # (k, params) buffer for parameterized agents), so this stays agent-agnostic.
    # On-policy agents (PPO) have no replay buffer — they collect a fresh rollout
    # each update — so none is built for them.
    on_policy = getattr(agent, "on_policy", False)
    buffer = None if on_policy else agent.make_buffer(agent.buffer_size,
                                                      env.obs_dim, seed=seed)

    commit = git_commit()
    print(cfg.summary())
    print(env.describe())
    print(f"commit {commit} | output {out_dir}")

    # The authoritative record of what this job ran lives in run_metadata.json.
    # run_experiment.sh writes it before training starts; when train.py is run by
    # hand (outside run_experiment.sh) this fallback writes one if absent, so
    # evaluate.py can always recover the resolved config from it.
    run_metadata.ensure_start(out_dir, cfg, args.kind, commit=commit)

    # --- checkpointing ---
    # checkpoints/ holds periodic snapshots for walltime/crash resume. At the end
    # of a SUCCESSFUL run they are pruned to just checkpoints/final/ (end state)
    # and checkpoints/best/ (lowest mean_eval_fitness); that pruning lives in
    # run_experiment.sh's persist trap, which knows the exit status — a timeout
    # or failure must keep the intermediates so the job can resume. best_eval is
    # tracked inside the checkpoint so "best so far" survives a resume.
    ckpts_dir = os.path.join(out_dir, "checkpoints")
    episode_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    best_eval_var = tf.Variable(np.inf, dtype=tf.float64, trainable=False)
    ckpt = tf.train.Checkpoint(episode=episode_var, best_eval=best_eval_var,
                               **agent.checkpoint_items())
    manager = tf.train.CheckpointManager(ckpt, ckpts_dir, max_to_keep=3)
    best_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**agent.checkpoint_items()),
        os.path.join(ckpts_dir, "best"), max_to_keep=1)
    final_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**agent.checkpoint_items()),
        os.path.join(ckpts_dir, "final"), max_to_keep=1)
    # The replay buffer is plain NumPy, so it is saved/restored alongside the
    # resume checkpoint (not inside it). Restoring it keeps a resumed run
    # continuous instead of retraining on a fresh, tiny, correlated buffer.
    buffer_path = os.path.join(ckpts_dir, "replay_buffer.npz")
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        if buffer is not None and os.path.exists(buffer_path):
            buffer.load(buffer_path)
        print(f"Resumed from {manager.latest_checkpoint} "
              f"(episode {int(episode_var.numpy())}, "
              f"best_eval {float(best_eval_var.numpy()):.4e}, "
              f"buffer {buffer.size if buffer is not None else 'n/a (on-policy)'})")

    log_path = os.path.join(out_dir, "training_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    logger = csv.writer(log_file)
    if write_header:
        logger.writerow(["training_step", "elapsed_seconds", "mean_return",
                         "std_return", "mean_loss", "mean_eval_fitness",
                         "std_eval_fitness", "mean_episode_length"])

    # progress_log.csv: a lightweight DENSE record of the free per-update rollout
    # statistics (return, loss, episode length). It runs NO evaluation and touches
    # NO checkpoint -- it only writes numbers training already computed -- so it
    # yields smooth training/reward curves at zero cost while leaving evaluation
    # and the checkpoint cadence byte-identical. Written once per PPO update (the
    # on-policy loop); the off-policy loop leaves it header-only for now.
    progress_path = os.path.join(out_dir, "progress_log.csv")
    progress_new = not os.path.exists(progress_path)
    progress_file = open(progress_path, "a", newline="")
    progress_logger = csv.writer(progress_file)
    if progress_new:
        progress_logger.writerow(["training_step", "episode", "mean_return",
                                  "mean_loss", "mean_episode_length"])
        progress_file.flush()

    episodes = cfg.get("training.episodes")
    eval_every = cfg.get("training.eval_every")
    eval_episodes = cfg.get("training.eval_episodes")
    checkpoint_every = cfg.get("training.checkpoint_every")
    # Replay settings — used only by the off-policy loop; absent on PPO.
    batch_size = getattr(agent, "batch_size", None)
    warmup = getattr(agent, "warmup_transitions", None)

    # Episode-based periodic checkpoint+evaluation (EXP003): every
    # `periodic_eval.every` episodes from `periodic_eval.start`, save a named
    # checkpoint and run the FULL §8 evaluation on the current greedy policy,
    # writing eval/chkp<k>/ — so a long run yields results as it goes rather
    # than only at the end. Absent from a config (EXP001/EXP002) -> inactive.
    periodic_every = cfg.get("training.periodic_eval.every", default=None)
    periodic_start = cfg.get("training.periodic_eval.start",
                             default=periodic_every)
    eval_root = os.path.join(out_dir, "eval")
    if periodic_every:
        # Fail before training, not at the first checkpoint: a missing baseline
        # must not surface hours into a multi-day job.
        check_baselines_available(cfg, env.functions)

    def periodic_checkpoint(k, n_ep):
        """Save checkpoints/chkp<k>/, snapshot the learning curve, and run the
        full evaluation on the live greedy policy into eval/chkp<k>/. Greedy and
        on a separate env, so it does not perturb the training trajectory."""
        tf.train.Checkpoint(**agent.checkpoint_items()).save(
            os.path.join(ckpts_dir, f"chkp{k}", "ckpt"))
        eval_dir = os.path.join(eval_root, f"chkp{k}")
        os.makedirs(eval_dir, exist_ok=True)
        log_file.flush()
        if os.path.exists(log_path):
            shutil.copyfile(log_path,
                            os.path.join(eval_dir, "training_log.csv"))
        try:
            evaluate_policy(cfg, eval_env, GreedyPolicy(agent), eval_dir)
            # Copy this checkpoint's outputs to the persisted location NOW, so
            # $HOME fills in progressively rather than only when the persist trap
            # runs at job end — the whole point of evaluating during training.
            # run_experiment.sh exports DERL2_FINAL_DIR; a hand-run leaves it
            # unset and just keeps everything in out_dir.
            final_dir = os.environ.get("DERL2_FINAL_DIR")
            if final_dir:
                for rel in (os.path.join("eval", f"chkp{k}"),
                            os.path.join("checkpoints", f"chkp{k}")):
                    src = os.path.join(out_dir, rel)
                    if os.path.isdir(src):
                        shutil.copytree(src, os.path.join(final_dir, rel),
                                        dirs_exist_ok=True)
                os.makedirs(final_dir, exist_ok=True)
                shutil.copyfile(log_path,
                                os.path.join(final_dir, "training_log.csv"))
                if os.path.exists(progress_path):
                    shutil.copyfile(progress_path,
                                    os.path.join(final_dir, "progress_log.csv"))
                print(f"  chkp{k}: results persisted to {final_dir}", flush=True)
        except Exception as err:      # never lose training progress to an eval
            print(f"WARNING: periodic evaluation at episode {n_ep} (chkp{k}) "
                  f"failed: {err}", flush=True)

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
        # Retain the checkpoint with the lowest mean_eval_fitness seen so far
        # (error, so lower is better) as checkpoints/best/.
        mean_eval_fitness = float(fits.mean())
        if mean_eval_fitness < float(best_eval_var.numpy()):
            best_eval_var.assign(mean_eval_fitness)
            best_manager.save()
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

    # --------------------------------------------------- on-policy loop (PPO)
    def run_on_policy_training():
        """Collect rollouts from n_envs parallel DEEnv copies, compute SMDP-GAE
        per env stream, and do a PPO update — reusing the SAME checkpoint / eval
        / logging / walltime cadence as the off-policy loop below. No replay
        buffer; evaluation stays single-env greedy (log_checkpoint and
        periodic_checkpoint), so every output file is identical in shape."""
        nonlocal next_eval_at, next_ckpt_at
        from derl2.environments.vec_env import make_vec_env
        from derl2.agents.hybrid_ppo import compute_gae

        n_envs = agent.n_envs
        per_env = max(1, agent.rollout_steps // n_envs)
        venv = make_vec_env(cfg, n_envs, base_seed=seed,
                            backend=cfg.get("environment.vec_backend", None))
        obs = venv.reset()
        completed = start
        k_done = (start // periodic_every) if periodic_every else 0
        ep_ret = [0.0] * n_envs
        ep_len = [0] * n_envs
        try:
            while completed < episodes:
                # ---- collect: per_env steps from each of n_envs streams ----
                streams = [[] for _ in range(n_envs)]
                roll_returns, roll_lengths = [], []   # this rollout only (dense log)
                for _t in range(per_env):
                    acts, logps, vals = [], [], []
                    for i in range(n_envs):
                        a, lp, v = agent.act_collect(obs[i])
                        acts.append(a); logps.append(lp); vals.append(v)
                    next_obs, rewards, dones, infos = venv.step(acts)
                    for i in range(n_envs):
                        streams[i].append(
                            (obs[i], acts[i][0], acts[i][1], logps[i], vals[i],
                             float(rewards[i]), float(dones[i]),
                             float(infos[i]["tau"])))
                        ep_ret[i] += float(rewards[i]); ep_len[i] += 1
                        if dones[i] > 0.5:
                            returns.append(ep_ret[i]); lengths.append(ep_len[i])
                            roll_returns.append(ep_ret[i])
                            roll_lengths.append(ep_len[i])
                            ep_ret[i] = 0.0; ep_len[i] = 0
                            completed += 1
                    obs = next_obs

                # ---- per-stream SMDP-GAE, then one PPO update ----
                cols = {k: [] for k in ("obs", "k", "params", "logp", "adv", "ret")}
                for i in range(n_envs):
                    s = streams[i]
                    last_v = agent.value(obs[i])
                    adv, ret = compute_gae(
                        [r[5] for r in s], [r[4] for r in s], [r[6] for r in s],
                        [r[7] for r in s], last_v, agent.gamma,
                        agent.gae_lambda, agent.per_budget)
                    cols["obs"] += [r[0] for r in s]
                    cols["k"] += [r[1] for r in s]
                    cols["params"] += [r[2] for r in s]
                    cols["logp"] += [r[3] for r in s]
                    cols["adv"] += list(adv); cols["ret"] += list(ret)
                loss = agent.update({
                    "obs": np.asarray(cols["obs"], dtype=np.float32),
                    "k": np.asarray(cols["k"], dtype=np.int32),
                    "params": np.asarray(cols["params"], dtype=np.float32),
                    "old_logp": np.asarray(cols["logp"], dtype=np.float32),
                    "adv": np.asarray(cols["adv"], dtype=np.float32),
                    "ret": np.asarray(cols["ret"], dtype=np.float32)})
                losses.append(loss)
                episode_var.assign(completed)

                # Dense per-update progress row (free stats; no eval, no ckpt).
                progress_logger.writerow([
                    int(agent.train_steps.numpy()), completed,
                    f"{np.mean(roll_returns):.6f}" if roll_returns else "nan",
                    f"{float(loss):.6f}",
                    f"{np.mean(roll_lengths):.4f}" if roll_lengths else "nan"])
                progress_file.flush()

                # ---- cadence: eval/ckpt on train steps, periodic on episodes ----
                ts = int(agent.train_steps.numpy())
                if ts >= next_ckpt_at:
                    manager.save(); next_ckpt_at += checkpoint_every
                if ts >= next_eval_at:
                    log_checkpoint(); next_eval_at += eval_every
                if periodic_every:
                    # Fire at most ONE periodic eval per update. A PPO rollout
                    # completes many episodes at once, so `completed` can jump
                    # past several `every`-multiples in a single update; the
                    # policy changed only once, so re-evaluating it for every
                    # crossed multiple is pure waste (it exploded smoke runs into
                    # ~190 checkpoints). Evaluate once, at the highest multiple
                    # reached. In the real run a rollout never crosses two
                    # multiples (rollout << every), so behaviour is unchanged.
                    k_target = completed // periodic_every
                    if k_target > k_done and completed >= periodic_start:
                        k_done = k_target
                        periodic_checkpoint(k_done, completed)

                # ---- cluster walltime safety ----
                if args.max_hours and (time.perf_counter() - t_start) / 3600.0 \
                        > args.max_hours:
                    manager.save(); venv.close(); log_file.close(); progress_file.close()
                    print(f"Walltime limit reached at episode {completed}; "
                          f"checkpointed. Re-run the same command to resume.")
                    sys.exit(42)
        finally:
            venv.close()

        if returns or losses:
            log_checkpoint()
        manager.save()
        final_manager.save()
        log_file.close(); progress_file.close()
        print(f"Training finished. Output in {out_dir}")

    # --------------------------------------------- recurrent on-policy loop (RNN)
    def run_on_policy_recurrent():
        """Episode-aligned recurrent PPO loop (EXP020). Collects WHOLE episodes
        from n_envs parallel envs, threading a per-env hidden state that resets
        at each episode boundary; computes SMDP-GAE per complete episode; and
        updates the recurrent agent over whole-episode minibatches. Reuses the
        SAME checkpoint / eval / logging / walltime cadence as the other loops.

        Episodes may span collection rounds, so per-env transitions accumulate in
        `acc[i]` until that env signals done; only then is the whole episode
        flushed (as one training sequence starting at h_0=0). `state[i]` is the
        recurrent hidden state, reset to zero on each done."""
        nonlocal next_eval_at, next_ckpt_at
        from derl2.environments.vec_env import make_vec_env
        from derl2.agents.hybrid_ppo import compute_gae

        n_envs = agent.n_envs
        target = agent.episodes_per_update
        venv = make_vec_env(cfg, n_envs, base_seed=seed,
                            backend=cfg.get("environment.vec_backend", None))
        obs = venv.reset()
        acc = [[] for _ in range(n_envs)]      # per-env in-progress transitions
        state = [agent.initial_state() for _ in range(n_envs)]
        completed = start
        k_done = (start // periodic_every) if periodic_every else 0
        try:
            while completed < episodes:
                # ---- collect whole episodes until >= target complete ----
                batch_eps = []
                roll_returns, roll_lengths = [], []
                while len(batch_eps) < target:
                    triples = [agent.act_collect(obs[i], state[i])
                               for i in range(n_envs)]
                    for i in range(n_envs):
                        state[i] = triples[i][3]
                    next_obs, rewards, dones, infos = venv.step(
                        [t[0] for t in triples])
                    for i in range(n_envs):
                        (a, lp, v, _) = triples[i]
                        acc[i].append(
                            (obs[i], a[0], a[1], lp, v, float(rewards[i]),
                             float(dones[i]), float(infos[i]["tau"])))
                        if dones[i] > 0.5:
                            ep = acc[i]
                            # Complete episode: dones[-1]=1 so the GAE bootstrap
                            # term vanishes (last_value=0 is unused).
                            adv, ret = compute_gae(
                                [r[5] for r in ep], [r[4] for r in ep],
                                [r[6] for r in ep], [r[7] for r in ep], 0.0,
                                agent.gamma, agent.gae_lambda, agent.per_budget)
                            batch_eps.append({
                                "obs": np.asarray([r[0] for r in ep], np.float32),
                                "k": np.asarray([r[1] for r in ep], np.int32),
                                "params": np.asarray([r[2] for r in ep],
                                                     np.float32),
                                "old_logp": np.asarray([r[3] for r in ep],
                                                       np.float32),
                                "adv": np.asarray(adv, np.float32),
                                "ret": np.asarray(ret, np.float32)})
                            ep_ret = float(sum(r[5] for r in ep))
                            returns.append(ep_ret); lengths.append(len(ep))
                            roll_returns.append(ep_ret)
                            roll_lengths.append(len(ep))
                            completed += 1
                            acc[i] = []
                            state[i] = agent.initial_state()
                    obs = next_obs

                # ---- one recurrent PPO update over the whole episodes ----
                loss = agent.update(batch_eps)
                losses.append(loss)
                episode_var.assign(completed)

                progress_logger.writerow([
                    int(agent.train_steps.numpy()), completed,
                    f"{np.mean(roll_returns):.6f}" if roll_returns else "nan",
                    f"{float(loss):.6f}",
                    f"{np.mean(roll_lengths):.4f}" if roll_lengths else "nan"])
                progress_file.flush()

                # ---- cadence: eval/ckpt on train steps, periodic on episodes ---
                ts = int(agent.train_steps.numpy())
                if ts >= next_ckpt_at:
                    manager.save(); next_ckpt_at += checkpoint_every
                if ts >= next_eval_at:
                    log_checkpoint(); next_eval_at += eval_every
                if periodic_every:
                    k_target = completed // periodic_every
                    if k_target > k_done and completed >= periodic_start:
                        k_done = k_target
                        periodic_checkpoint(k_done, completed)

                # ---- cluster walltime safety ----
                if args.max_hours and (time.perf_counter() - t_start) / 3600.0 \
                        > args.max_hours:
                    manager.save(); venv.close()
                    log_file.close(); progress_file.close()
                    print(f"Walltime limit reached at episode {completed}; "
                          f"checkpointed. Re-run the same command to resume.")
                    sys.exit(42)
        finally:
            venv.close()

        if returns or losses:
            log_checkpoint()
        manager.save()
        final_manager.save()
        log_file.close(); progress_file.close()
        print(f"Training finished. Output in {out_dir}")

    if on_policy:
        if getattr(agent, "recurrent", False):
            run_on_policy_recurrent()
        else:
            run_on_policy_training()
        return

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
                    buffer.save(buffer_path)      # keep the resume pair in sync
                    next_ckpt_at += checkpoint_every
                if ts >= next_eval_at:
                    # Aggregates the episodes completed since the last
                    # checkpoint; the in-progress episode is not yet counted.
                    log_checkpoint()
                    next_eval_at += eval_every

        episode_var.assign(episode + 1)
        returns.append(ep_return)
        lengths.append(steps)

        # Episode-based periodic checkpoint + full evaluation (EXP003).
        completed = episode + 1
        if periodic_every and completed >= periodic_start \
                and completed % periodic_every == 0:
            periodic_checkpoint(completed // periodic_every, completed)

        # Cluster walltime safety: checkpoint and exit cleanly rather than
        # being killed mid-episode.
        if args.max_hours and (time.perf_counter() - t_start) / 3600.0 > args.max_hours:
            manager.save()
            buffer.save(buffer_path)
            log_file.close(); progress_file.close()
            print(f"Walltime limit reached at episode {episode + 1}; "
                  f"checkpointed. Re-run the same command to resume.")
            # Exit 42 signals a clean walltime stop (not completion) to
            # run_experiment.sh, which labels the run "timeout", keeps the
            # intermediate checkpoints, and does not run evaluation.
            sys.exit(42)

    # A final checkpoint row so the end state is always recorded, even if the
    # last eval threshold fell between here and the previous checkpoint.
    if returns or losses:
        log_checkpoint()
    manager.save()
    buffer.save(buffer_path)       # final buffer: retained for extending later
    final_manager.save()           # the retained end-state (checkpoints/final/)
    log_file.close(); progress_file.close()
    print(f"Training finished. Output in {out_dir}")


if __name__ == "__main__":
    main()
