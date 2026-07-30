"""Independent evaluation (spec §8).

Loads a checkpoint produced by train.py and runs the greedy policy on seeds
never used during training, writing the five result files:

    episode_results.csv   one row per (function, seed)
    step_trace.csv        one row per (function, seed, step)
    summary.csv           one row per function
    comparison.csv        one row per (function, baseline) in compare_against
    key_result.txt        one line, for pasting into REGISTRY.csv

    python -m derl2.evaluation.evaluate --train-dir experiments/EXP001/train/job_123

A decreasing training loss is not evidence of better optimization; the files
written here are. Training and evaluation are strictly separate: this reads the
config the training job actually ran (config_used.yaml) and never trains.

Baselines are frozen artifacts (principle 10): comparison.csv reads baseline
results.csv files (same schema as episode_results.csv) from
baselines/<name>/ for the names in evaluation.compare_against, and never
generates them as a side effect of evaluating the agent. The fixed / random
comparator policies below exist so a baseline can be produced DELIBERATELY —
`--policy fixed --out-dir baselines/BASE00N_slug` runs a fixed-action DE
reference and writes results.csv in the episode_results schema — an explicit
invocation, not something the agent evaluation triggers.
"""

import argparse
import csv
import os
import time

import numpy as np
import tensorflow as tf

try:
    from scipy.stats import mannwhitneyu
except Exception:                       # scipy is a declared dependency; this
    mannwhitneyu = None                 # guard only keeps import errors legible

from derl2.agents import build_agent
from derl2.config import Config, repo_root
from derl2.environments import DEEnv


# --------------------------------------------------- comparator policies
# Two small functions (spec §8): not classes, not a registry, not a module.

def fixed_policy(action):
    """Always take the same action (a raw action index)."""
    return lambda obs: action


def random_policy(action_space, seed=12345):
    """Uniform random action every step."""
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(action_space.n))


# ----------------------------------------------------------- schemas

EPISODE_COLUMNS = ["function_id", "seed", "final_error", "n_steps",
                   "total_FEs", "episode_return", "wall_clock_seconds"]

STEP_COLUMNS = ["episode_id", "function_id", "seed", "step_idx", "tau",
                "budget_used_before", "budget_remaining_frac_before",
                "strategy", "F", "CR", "budget_frac_action",
                "sampling_box_action", "box_width_frac_initial",
                "box_width_frac_final", "box_at_floor", "incumbent_in_box",
                "n_stag_before", "n_stag_after", "segment_best_error",
                "global_best_error_before", "global_best_error_after",
                "reward", "done"]


def run_episode(env, seed, function_id, policy):
    """One greedy episode; returns (episode_row, step_infos, trajectories)."""
    t0 = time.perf_counter()          # monotonic; time.time() can step back
    obs, info = env.reset(seed=seed, function_id=function_id)
    step_infos, trajectories, ep_return = [], [], 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        ep_return += reward
        step_infos.append(info)
        trajectories.append(info["trajectory"])
        done = terminated or truncated
    episode_row = {
        "function_id": function_id,
        "seed": seed,
        "final_error": info["best_error"],
        "n_steps": info["step_idx"],
        "total_FEs": info["fes_used_total"],
        "episode_return": ep_return,
        "wall_clock_seconds": time.perf_counter() - t0,
    }
    return episode_row, step_infos, trajectories


def build_policy(args, cfg, env):
    """Resolve the requested policy (agent by default)."""
    if args.policy == "fixed":
        return fixed_policy(args.fixed_action)
    if args.policy == "random":
        return random_policy(env.action, seed=cfg.get("seed"))
    # agent: restore the trained network from the training job's checkpoint
    agent_block = cfg.block("agent")
    agent_name = agent_block.pop("name")
    agent = build_agent(agent_name, env.obs_dim, env.n_actions, agent_block,
                        seed=cfg.get("seed"))
    ckpt_dir = os.path.join(args.train_dir, "checkpoints")
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    tf.train.Checkpoint(**agent.checkpoint_items()).restore(latest) \
        .expect_partial()
    print(f"checkpoint {latest}")
    return lambda obs: agent.act(obs, epsilon=0.0)


def write_summary(path, episode_rows):
    """One row per function: mean, median, std, best, worst, n_runs of error."""
    by_function = {}
    for row in episode_rows:
        by_function.setdefault(row["function_id"], []).append(row["final_error"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function_id", "mean", "median", "std", "best", "worst",
                    "n_runs"])
        for fid in sorted(by_function):
            e = np.array(by_function[fid], dtype=np.float64)
            w.writerow([fid, f"{e.mean():.6e}", f"{np.median(e):.6e}",
                        f"{e.std():.6e}", f"{e.min():.6e}", f"{e.max():.6e}",
                        len(e)])


def load_baseline(name):
    """Read baselines/<name>/results.csv (episode_results schema) as
    {function_id: [final_error, ...]}."""
    path = os.path.join(repo_root(), "baselines", name, "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline {name!r} not found at {path}. compare_against names "
            f"folders under baselines/ that already exist (principle 10)."
        )
    by_function = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            by_function.setdefault(int(row["function_id"]), []).append(
                float(row["final_error"]))
    return by_function


def write_comparison(path, episode_rows, compare_against):
    """One row per (function, baseline); empty (header only) when none."""
    agent = {}
    for row in episode_rows:
        agent.setdefault(row["function_id"], []).append(row["final_error"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function_id", "baseline_name", "agent_mean", "agent_median",
                    "baseline_mean", "baseline_median", "pct_diff",
                    "test_statistic", "p_value"])
        for name in compare_against:
            base = load_baseline(name)
            for fid in sorted(agent):
                a = np.array(agent[fid], dtype=np.float64)
                b = np.array(base.get(fid, []), dtype=np.float64)
                if b.size and mannwhitneyu is not None:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                else:
                    stat, p = float("nan"), float("nan")
                base_mean = b.mean() if b.size else float("nan")
                pct = ((a.mean() - base_mean) / base_mean * 100.0
                       if b.size and base_mean != 0 else float("nan"))
                w.writerow([fid, name, f"{a.mean():.6e}",
                            f"{np.median(a):.6e}", f"{base_mean:.6e}",
                            f"{np.median(b):.6e}" if b.size else "nan",
                            f"{pct:.4f}", f"{stat:.6g}", f"{p:.6g}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True,
                        help="Training job dir (has config_used.yaml, "
                             "checkpoints/)")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write the five result files. Defaults to "
                             "the training job dir itself, so a run_experiment.sh "
                             "job keeps all of §8's outputs in one folder that "
                             "its persist step copies back. Pass this to separate "
                             "them (e.g. a hand-run train/ then train_eval/).")
    parser.add_argument("--policy", default="agent",
                        choices=["agent", "fixed", "random"],
                        help="agent (default); fixed/random deliberately "
                             "produce a baseline results.csv")
    parser.add_argument("--fixed-action", type=int, default=0,
                        help="raw action index for --policy fixed")
    args = parser.parse_args()

    cfg_path = os.path.join(args.train_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"No config_used.yaml in {args.train_dir}. Evaluation must use the "
            f"config the training job actually ran."
        )
    cfg = Config.from_file(cfg_path)
    env = DEEnv.from_config(cfg)
    print(cfg.summary())
    print(env.describe())

    policy = build_policy(args, cfg, env)

    runs = cfg.get("evaluation.runs_per_function")
    offset = cfg.get("evaluation.seed_offset")
    save_traj = cfg.get("evaluation.save_trajectory_seeds")
    compare_against = cfg.get("evaluation.compare_against")

    out_dir = os.path.abspath(args.out_dir or args.train_dir)
    os.makedirs(out_dir, exist_ok=True)
    traj_dir = os.path.join(out_dir, "trajectories")

    episode_rows, step_rows = [], []
    episode_id = 0
    for fid in env.functions:
        for i in range(runs):
            seed = offset + i
            ep, infos, trajs = run_episode(env, seed, fid, policy)
            episode_rows.append(ep)
            for info in infos:
                row = {"episode_id": episode_id, "function_id": fid,
                       "seed": seed}
                row.update({k: info[k] for k in STEP_COLUMNS[3:]})
                step_rows.append(row)
            # Full within-segment trajectories only for the first N seeds of
            # each function, for figure-making (spec §8).
            if i < save_traj:
                os.makedirs(traj_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(traj_dir, f"f{fid}_seed{seed}.npz"),
                    trajectory=np.stack(trajs))
            episode_id += 1

    # episode_results.csv
    with open(os.path.join(out_dir, "episode_results.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EPISODE_COLUMNS)
        w.writeheader()
        for row in episode_rows:
            w.writerow({k: (f"{row[k]:.6e}" if isinstance(row[k], float)
                            else row[k]) for k in EPISODE_COLUMNS})

    # step_trace.csv
    with open(os.path.join(out_dir, "step_trace.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=STEP_COLUMNS)
        w.writeheader()
        for row in step_rows:
            w.writerow(row)

    write_summary(os.path.join(out_dir, "summary.csv"), episode_rows)
    write_comparison(os.path.join(out_dir, "comparison.csv"), episode_rows,
                     compare_against)

    finals = np.array([r["final_error"] for r in episode_rows], dtype=np.float64)
    key_result = (f"policy={args.policy} functions={env.functions} "
                  f"dim={env.dim} runs/function={runs}: "
                  f"median error {np.median(finals):.3e}, "
                  f"mean {finals.mean():.3e} over {len(finals)} episodes")
    with open(os.path.join(out_dir, "key_result.txt"), "w") as fh:
        fh.write(key_result + "\n")

    print(f"\n{key_result}")
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
