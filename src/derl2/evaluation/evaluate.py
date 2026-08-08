"""Independent evaluation (spec §8).

Loads a checkpoint produced by train.py and runs the greedy policy on seeds
never used during training, writing the result files:

    episode_results.csv   one row per (function, seed)
    step_trace.csv        one row per (function, seed, step)
    summary.csv           one row per function
    eval_results.csv      one row per (function, seed): agent + every baseline
                          side by side, with the budget each consumed
    comparison.csv        one row per (function, baseline) in compare_against

    python -m derl2.evaluation.evaluate --train-dir experiments/EXP002/train_eval

A decreasing training loss is not evidence of better optimization; the files
written here are. Training and evaluation are strictly separate: this reads the
resolved config the training job actually ran (run_metadata.json ->
resolved_config) and never trains.

Baselines are frozen artifacts (principle 10): the comparison reads
baselines/<name>/results.csv produced DELIBERATELY by scripts/run_baseline.py,
never generated as a side effect of evaluating the agent. A name in
evaluation.compare_against with no baselines/<name>/ folder is a hard error
naming the exact command to produce it, not a silent skip. Because every method
runs the identical seed list, the comparison is paired (Wilcoxon signed-rank),
and a baseline produced on a different function / dim / budget / seed list is
refused outright rather than compared as if it were comparable.
"""

import argparse
import csv
import os
import time

import numpy as np
import tensorflow as tf
import yaml

try:
    from scipy.stats import wilcoxon
except Exception:                       # scipy is a declared dependency; this
    wilcoxon = None                     # guard only keeps import errors legible

from derl2 import run_metadata
from derl2.agents import build_agent
from derl2.config import repo_root
from derl2.environments import DEEnv


# --------------------------------------------------- comparator policies
# Small helpers reused by scripts/run_baseline.py to drive the fixed-schedule
# (BASE002) and random (BASE003) baselines through the real environment. They
# live here, next to run_episode, so a baseline runs the identical episode loop
# the agent does.

def fixed_policy(action):
    """Always take the same action (a raw action index)."""
    return lambda obs: int(action)


def random_policy(action_space, seed=12345):
    """Uniform random action every step, seeded for reproducibility. Each action
    space defines its own uniform sample (a discrete index for the discrete
    space, a (strategy, [0,1]^param_dim) pair for the parameterized one), so the
    random baseline matches whatever action space the experiment uses."""
    rng = np.random.default_rng(seed)
    return lambda obs: action_space.sample_random(rng)


# ----------------------------------------------------------- schemas

EPISODE_COLUMNS = ["function_id", "seed", "final_error", "n_steps",
                   "total_FEs", "episode_return", "wall_clock_seconds"]

STEP_COLUMNS = ["episode_id", "function_id", "seed", "step_idx", "tau",
                "budget_used_before", "budget_remaining_frac_before",
                "strategy", "box_center", "pop_size_action", "F", "CR",
                "budget_frac_action",
                "sampling_box_action", "box_width_frac_initial",
                "box_width_frac_final", "box_at_floor", "incumbent_in_box",
                "n_stag_before", "n_stag_after", "segment_best_error",
                "global_best_error_before", "global_best_error_after",
                "reward", "done"]
# box_center / pop_size_action are only emitted by the freedom action space
# (param_strategy_boxnp); for every other experiment the env's info lacks them
# and the column is written blank (see the info.get in evaluate_policy).


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


def build_agent_policy(train_dir, cfg, env):
    """Restore the trained network and return a greedy policy.

    Checkpoints are looked up in checkpoints/final/ first (the retained end
    state of a completed run), then the top-level checkpoints/ (present during a
    train_eval job, before the persist trap prunes intermediates), then
    checkpoints/best/ — so evaluation works both mid-job and on a pruned,
    persisted run."""
    agent_block = cfg.block("agent")
    agent_name = agent_block.pop("name")
    agent = build_agent(agent_name, env.obs_dim, env.n_actions, agent_block,
                        seed=cfg.get("seed"))
    ckpt_dir = os.path.join(train_dir, "checkpoints")
    latest = (tf.train.latest_checkpoint(os.path.join(ckpt_dir, "final"))
              or tf.train.latest_checkpoint(ckpt_dir)
              or tf.train.latest_checkpoint(os.path.join(ckpt_dir, "best")))
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint under {ckpt_dir} (looked in final/, ./, best/)."
        )
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


# ------------------------------------------------------------- baselines

# The properties a baseline MUST share with the agent evaluation for the numbers
# to be comparable at all. A mismatch on any of these is refused, not compared.
BASELINE_PARITY_KEYS = ("functions", "dim", "budget", "seeds")


def baseline_dir(name, function_id, exp_id, smoke=False):
    """Per-experiment, per-function baseline folder:
    baselines/<exp_id>[/smoke]/f<fid>/<name>/.

    Baselines are namespaced by EXPERIMENT because a baseline is only valid for
    the exact episode/environment config it was produced under — warmup fraction,
    box scales, budget-fraction menu, dim, budget, seeds. Two experiments that
    change any of those (e.g. EXP003 vs EXP004, which alters warmup/box/budget)
    have genuinely different BASE002/BASE003, so they must not share or clobber
    one folder. A consequence we rely on: an experiment whose baselines have not
    been generated fails fast in check_baselines_available rather than silently
    reusing another experiment's numbers. Smoke-scale baselines live under a
    /smoke subtree so a rehearsal never overwrites the real reference."""
    parts = [repo_root(), "baselines", exp_id]
    if smoke:
        parts.append("smoke")
    parts += [f"f{function_id}", name]
    return os.path.join(*parts)


def baseline_namespace(cfg):
    """The baselines folder an experiment reads/writes. Defaults to the
    experiment id (the per-experiment namespacing baseline_dir documents), but an
    experiment may set evaluation.baseline_namespace to SHARE a baseline set with
    other experiments that are byte-identical in everything a baseline depends on
    — dim, budget, pop, warmup, seeds, and (for BASE003) the action space. The
    reward and the agent never affect a baseline, so EXP009/010/011 (same env,
    same continuous action space) can and do share one namespace. check_parity
    still guards function/dim/budget/seeds, so a mismatched share fails loudly."""
    return cfg.get("evaluation.baseline_namespace", default=cfg.exp_id)


def _missing_baseline_error(name, path, function_id, smoke=False):
    smoke_flag = " --smoke" if smoke else ""
    return FileNotFoundError(
        f"Baseline {name!r} for f{function_id} not found at {path}. Baselines "
        f"are produced deliberately and never auto-generated. Create it with:\n"
        f"    python -m scripts.run_baseline --baseline {name} "
        f"--config <this experiment's config.yaml> --function {function_id}"
        f"{smoke_flag}"
    )


def load_baseline(name, function_id, exp_id, smoke=False):
    """Read baselines/<exp_id>[/smoke]/f<fid>/<name>/{results.csv,
    baseline_config.yaml}.

    Returns (by_seed, baseline_cfg) where by_seed maps (function_id, seed) ->
    (best_error, evals_used)."""
    d = baseline_dir(name, function_id, exp_id, smoke)
    results = os.path.join(d, "results.csv")
    cfg_path = os.path.join(d, "baseline_config.yaml")
    if not os.path.exists(results):
        raise _missing_baseline_error(name, d, function_id, smoke)
    by_seed = {}
    with open(results, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["function_id"]), int(row["seed"]))
            by_seed[key] = (float(row["best_error"]),
                            int(float(row["evals_used"])))
    baseline_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            baseline_cfg = yaml.safe_load(fh) or {}
    return by_seed, baseline_cfg


def check_baselines_available(cfg, functions):
    """Fail fast (before a long training run) if any baseline this experiment
    compares against is missing for a function it will evaluate. A periodic
    evaluation mid-training must never be the thing that discovers a missing
    baseline and kills a multi-day job. A smoke run checks the smoke tree."""
    compare_against = cfg.get("evaluation.compare_against")
    ns = baseline_namespace(cfg)
    for fid in functions:
        for name in compare_against:
            d = baseline_dir(name, fid, ns, cfg.is_smoke)
            if not os.path.exists(os.path.join(d, "results.csv")):
                raise _missing_baseline_error(name, d, fid, cfg.is_smoke)


def check_parity(name, baseline_cfg, expected):
    """Fail loudly if a baseline was produced on different function / dim /
    budget / seeds than this evaluation — the single most important correctness
    property here: comparing incomparable numbers is worse than not comparing."""
    if not baseline_cfg:
        raise ValueError(
            f"Baseline {name!r} has no baseline_config.yaml, so its function / "
            f"dim / budget / seed list cannot be verified against this "
            f"evaluation. Regenerate it with scripts/run_baseline.py."
        )
    for key in BASELINE_PARITY_KEYS:
        got = baseline_cfg.get(key)
        exp = expected[key]
        got_cmp = sorted(got) if isinstance(got, list) else got
        exp_cmp = sorted(exp) if isinstance(exp, list) else exp
        if got_cmp != exp_cmp:
            raise ValueError(
                f"Baseline {name!r} was produced with {key}={got!r} but this "
                f"evaluation uses {key}={exp!r}. Comparing them would compare "
                f"incomparable numbers — refusing. Regenerate the baseline "
                f"against this experiment's config with scripts/run_baseline.py."
            )


def _require_full_coverage(name, keys, by_seed):
    """Strict join: a baseline must cover every (function, seed) the agent ran.
    Missing rows are an error naming them, never a dropped row or a NaN."""
    missing = [k for k in keys if k not in by_seed]
    if missing:
        listed = ", ".join(f"f{f}/seed{s}" for f, s in missing)
        raise ValueError(
            f"Baseline {name!r} is missing results for: {listed}. Every method "
            f"must cover the same seeds; refusing to drop rows or fill NaN. "
            f"Regenerate {name} with scripts/run_baseline.py."
        )


def write_eval_results(path, keys, agent_by_seed, baselines, compare_against):
    """Per-seed raw results, one row per (function, seed), all methods side by
    side. The *_evals columns are mandatory: budget consumed at termination is
    part of the result (reaching a target in fewer evaluations is a real finding
    that final-error-only reporting hides). Column pairs are generated from
    compare_against, so a new baseline needs no code change here."""
    for name in compare_against:
        _require_full_coverage(name, keys, baselines[name])
    header = ["seed", "function_id", "agent_best", "agent_evals"]
    for name in compare_against:
        header += [f"{name}_best", f"{name}_evals"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for (fid, seed) in keys:
            a_best, a_evals = agent_by_seed[(fid, seed)]
            row = [seed, fid, f"{a_best:.6e}", a_evals]
            for name in compare_against:
                b_best, b_evals = baselines[name][(fid, seed)]
                row += [f"{b_best:.6e}", b_evals]
            w.writerow(row)


def write_comparison(path, keys, agent_by_seed, baselines, compare_against):
    """Aggregate statistics, one row per (function, baseline).

    pct_diff = (baseline_mean - agent_mean) / |baseline_mean| * 100, so a
    POSITIVE value means the agent reached a LOWER error (is better). Because
    every method runs the identical seed list the samples are PAIRED, so the
    test is Wilcoxon signed-rank (not Mann-Whitney). Degenerate cases are handled
    explicitly: all-zero paired differences (realistic — the agent hits 0.0 on
    some seeds) would make wilcoxon raise, so they map to (statistic 0, p 1.0),
    and a zero baseline_mean guards pct_diff against division by zero."""
    functions = sorted({fid for (fid, _seed) in keys})
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function_id", "baseline_name", "agent_mean", "agent_median",
                    "baseline_mean", "baseline_median", "pct_diff",
                    "test_statistic", "p_value"])
        for name in compare_against:
            for fid in functions:
                fkeys = [k for k in keys if k[0] == fid]
                a = np.array([agent_by_seed[k][0] for k in fkeys],
                             dtype=np.float64)
                b = np.array([baselines[name][k][0] for k in fkeys],
                             dtype=np.float64)
                if wilcoxon is None or np.allclose(a - b, 0.0):
                    stat, p = 0.0, 1.0
                else:
                    try:
                        stat, p = wilcoxon(a, b)      # two-sided by default
                    except ValueError:                # all differences zero
                        stat, p = 0.0, 1.0
                if not np.isfinite(p):                 # degenerate tiny sample
                    stat, p = 0.0, 1.0
                a_mean, b_mean = float(a.mean()), float(b.mean())
                pct = ((b_mean - a_mean) / abs(b_mean) * 100.0
                       if b_mean != 0 else float("nan"))
                w.writerow([fid, name, f"{a_mean:.6e}", f"{np.median(a):.6e}",
                            f"{b_mean:.6e}", f"{np.median(b):.6e}",
                            f"{pct:.4f}", f"{stat:.6g}", f"{p:.6g}"])


def evaluate_policy(cfg, env, policy, out_dir):
    """Run the greedy `policy` on the evaluation seeds and write §8's files into
    `out_dir` (episode_results, step_trace, summary, eval_results, comparison,
    trajectories). Used by both the CLI (policy restored from a checkpoint) and
    train.py's in-process periodic evaluation (policy = the live agent), so a
    mid-training checkpoint produces the identical set of files a final run does.

    Baselines are read per function from baselines/f<fid>/<name>/, checked for
    function/dim/budget/seed parity, and refused on mismatch."""
    runs = cfg.get("evaluation.runs_per_function")
    offset = cfg.get("evaluation.seed_offset")
    save_traj = cfg.get("evaluation.save_trajectory_seeds")
    compare_against = cfg.get("evaluation.compare_against")

    out_dir = os.path.abspath(out_dir)
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
                row.update({k: info.get(k, "") for k in STEP_COLUMNS[3:]})
                step_rows.append(row)
            # Full within-segment trajectories only for the first N seeds of
            # each function, for figure-making (spec §8). Kept, not pruned.
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

    # Per-seed agent results, keyed by (function_id, seed), for the paired
    # baseline comparison and the side-by-side eval_results.csv.
    agent_by_seed = {
        (r["function_id"], r["seed"]): (r["final_error"], r["total_FEs"])
        for r in episode_rows
    }
    keys = sorted(agent_by_seed)

    seed_list = [offset + i for i in range(runs)]
    ns = baseline_namespace(cfg)
    baselines = {name: {} for name in compare_against}
    for fid in env.functions:
        expected = {"functions": [fid], "dim": env.dim,
                    "budget": env.budget, "seeds": sorted(seed_list)}
        for name in compare_against:
            by_seed, baseline_cfg = load_baseline(name, fid, ns,
                                                  cfg.is_smoke)
            check_parity(name, baseline_cfg, expected)
            baselines[name].update(by_seed)

    write_eval_results(os.path.join(out_dir, "eval_results.csv"), keys,
                       agent_by_seed, baselines, compare_against)
    write_comparison(os.path.join(out_dir, "comparison.csv"), keys,
                     agent_by_seed, baselines, compare_against)

    finals = np.array([r["final_error"] for r in episode_rows], dtype=np.float64)
    key_result = (f"policy=agent functions={env.functions} "
                  f"dim={env.dim} runs/function={runs}: "
                  f"median error {np.median(finals):.3e}, "
                  f"mean {finals.mean():.3e} over {len(finals)} episodes")
    # No key_result.txt: this line duplicates summary.csv; print for the log.
    print(f"\n{key_result}")
    print(f"Results in {out_dir}")
    return key_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True,
                        help="Training job dir (has run_metadata.json, "
                             "checkpoints/)")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write the result files. Defaults to the "
                             "training job dir itself, so a run_experiment.sh "
                             "job keeps all of §8's outputs in one folder that "
                             "its persist step copies back.")
    parser.add_argument("--function", type=int, default=None,
                        help="Restrict evaluation to one function id (the "
                             "per-function EXP003 layout); defaults to whatever "
                             "the training job's config recorded.")
    args = parser.parse_args()

    # Evaluation uses the config the training job ACTUALLY ran (smoke-scaled if
    # it was a smoke job), recovered from run_metadata.json — never the possibly
    # unscaled on-disk config.yaml.
    cfg = run_metadata.load_resolved_config(args.train_dir)
    if args.function is not None:
        cfg.set("benchmark.functions", [args.function])
    env = DEEnv.from_config(cfg)
    print(cfg.summary())
    print(env.describe())

    policy = build_agent_policy(args.train_dir, cfg, env)
    evaluate_policy(cfg, env, policy, args.out_dir or args.train_dir)


if __name__ == "__main__":
    main()
