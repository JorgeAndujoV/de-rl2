"""Generate the behavioural-analysis figures for an EXP003-structured experiment.

Given an experiment folder laid out per function with a periodic-eval history
(the EXP003_dqn_restart layout) this writes, under tests/<experiment>/:

    tests/<exp>/
        f<N>/                       one folder per function the experiment ran
            learning_curve.png      how much / when the agent learned
            actions_evolution.png   how the action mix emerges across checkpoints
            actions_by_phase.png    what the mature policy does at each episode phase
            episode_restructure.png number of restarts and how it evolved
            convergence.png         best-so-far error vs budget, chkp1 vs mature
        cross_function.png          one scale-free view: agent vs baselines, all fns

It reads only the files the evaluation pipeline already produced
(training_log.csv, eval/chkp<k>/{step_trace,episode_results,summary,comparison}.csv,
eval/chkp<k>/trajectories/*.npz); it never re-runs anything. Functions and
checkpoints are discovered from disk, so the same script serves EXP004 (the same
layout with a different reward) or any experiment shaped like EXP003.

    python -m scripts.analyze_experiment --exp EXP003_dqn_restart

Design: every categorical encoding uses the Okabe-Ito colourblind-safe palette in
a fixed order (identity follows the entity, never its rank); error axes that span
orders of magnitude are logarithmic; grids and spines are recessive so the marks
carry the ink.
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")                       # headless; write files, never a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CEC'13 function names, for readable titles. A function id absent here (a
# different suite) just falls back to "f<N>".
_CEC13_NAMES = {
    1: "Sphere", 2: "Rotated High Conditioned Elliptic", 3: "Rotated Bent Cigar",
    4: "Rotated Discus", 5: "Different Powers", 6: "Rotated Rosenbrock",
    7: "Rotated Schaffers F7", 8: "Rotated Ackley", 9: "Rotated Weierstrass",
    10: "Rotated Griewank", 11: "Rastrigin", 12: "Rotated Rastrigin",
    13: "Non-Continuous Rotated Rastrigin", 14: "Schwefel", 15: "Rotated Schwefel",
    16: "Rotated Katsuura", 17: "Lunacek Bi-Rastrigin",
    18: "Rotated Lunacek Bi-Rastrigin", 19: "Expanded Griewank+Rosenbrock",
    20: "Expanded Scaffer F6", 21: "Composition Function 1",
    22: "Composition Function 2", 23: "Composition Function 3",
    24: "Composition Function 4", 25: "Composition Function 5",
    26: "Composition Function 6", 27: "Composition Function 7",
    28: "Composition Function 8",
}

# Okabe-Ito: colourblind-safe categorical order, strong blue first so a single
# highlighted series reads well. Identity is assigned in this fixed order and
# never cycled or reassigned by rank.
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
         "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# Preferred display order for the DE strategies (any strategy not listed is
# appended in sorted order, so an experiment with different profiles still plots).
STRATEGY_ORDER = ["rand/1/bin", "rand/2/bin", "current-to-best/1/bin", "best/1/bin"]

INK, MUTED, GRID, SURFACE = "#222222", "#666666", "#E6E6E6", "#FFFFFF"

# CEC'13 reports an error below 1e-8 as exactly 0 (the optimum was effectively
# reached). A log axis cannot place 0, which is what broke f10/f11/f17's curves
# (they solve to the floor). Clip to this floor for DISPLAY only, and draw a
# reference line so "the curve dropped to the floor" reads as "solved".
_ERR_FLOOR = 1e-8


def _floorclip(a):
    return np.clip(np.asarray(a, dtype=float), _ERR_FLOOR, None)


# --------------------------------------------------------------- styling
def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "legend.fontsize": 8.5, "font.size": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"    wrote {os.path.relpath(path, REPO_ROOT)}")


def fname(fid):
    nm = _CEC13_NAMES.get(fid)
    return f"f{fid} — {nm}" if nm else f"f{fid}"


# --------------------------------------------------------------- discovery
def discover_functions(exp_dir):
    """Function ids that have an eval history, in numeric order."""
    out = []
    for d in sorted(glob.glob(os.path.join(exp_dir, "f*"))):
        m = re.match(r"^f(\d+)$", os.path.basename(d))
        if m and os.path.isdir(os.path.join(d, "eval")):
            out.append(int(m.group(1)))
    return sorted(out)


def discover_checkpoints(func_dir):
    """chkp folders under eval/, ordered by their integer index."""
    ks = []
    for d in glob.glob(os.path.join(func_dir, "eval", "chkp*")):
        m = re.match(r"^chkp(\d+)$", os.path.basename(d))
        if m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def _chkp_dir(func_dir, k):
    return os.path.join(func_dir, "eval", f"chkp{k}")


def _ordered_categories(values, axis):
    """Stable category order: strategies by preference then extras; numeric
    axes (budget_frac, sampling_box) by value; anything else lexically."""
    uniq = list(dict.fromkeys(values))
    if axis == "strategy":
        known = [s for s in STRATEGY_ORDER if s in uniq]
        return known + sorted(x for x in uniq if x not in known)
    try:
        return sorted(uniq, key=lambda v: float(v))
    except ValueError:
        return sorted(uniq)


def _color_map(categories):
    return {c: OKABE[i % len(OKABE)] for i, c in enumerate(categories)}


def _stacked(ax, x, frac_by_cat, categories, cmap, width=0.8):
    """Stacked bars: frac_by_cat[cat] is an array aligned with x. A 0.6px
    surface-coloured edge gives the 2px inter-segment gap the marks want."""
    bottom = np.zeros(len(x), dtype=float)
    for c in categories:
        vals = np.asarray(frac_by_cat.get(c, np.zeros(len(x))), dtype=float)
        ax.bar(x, vals, bottom=bottom, width=width, label=str(c),
               color=cmap[c], edgecolor=SURFACE, linewidth=0.6)
        bottom += vals


# --------------------------------------------------------------- axis labels
_AXIS_TITLE = {"strategy": "DE strategy",
               "box_center": "sampling-box centre",
               "budget_frac_action": "budget fraction / segment",
               "sampling_box_action": "sampling box"}


# --------------------------------------------------------------- 1. learning curve
def plot_learning_curve(func_dir, fid, checkpoints, out_path):
    log_path = os.path.join(func_dir, "training_log.csv")
    if not os.path.exists(log_path):
        print(f"    [skip] learning_curve: no training_log.csv")
        return
    df = pd.read_csv(log_path)
    series = _checkpoint_series(func_dir, checkpoints)
    # Return/loss/length source. An on-policy PPO run writes progress_log.csv --
    # one row PER UPDATE, dense and free (no evaluation) -- so prefer it; the
    # sparse training_log (a few eval rows) would degenerate to ~2 points. DQN
    # has no progress_log but a dense training_log (one row per gradient-step
    # eval), so it keeps the step-axis view. The error panel is driven by the
    # EPISODE-indexed 51-seed periodic checkpoints whenever the step log is sparse.
    prog_path = os.path.join(func_dir, "progress_log.csv")
    prog = pd.read_csv(prog_path) if os.path.exists(prog_path) else None
    use_prog = prog is not None and len(prog) >= 2
    dense = (not use_prog) and len(df) >= 8

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    tag = ("" if dense else
           "  (return/loss/length: dense per-update log; error: 51-seed evals)"
           if use_prog else
           "  (episode axis — batched-update agent, sparse step log)")
    fig.suptitle(f"{fname(fid)} — learning curve{tag}", fontweight="bold", y=0.99)

    # (a) optimisation error. Dense: 10-seed step eval + 51-seed checkpoint dots.
    # Sparse: the 51-seed checkpoint median (+IQR) on the episode axis.
    ax = axes[0, 0]
    if dense:
        x = df["training_step"] / 1000.0
        m, sd = _floorclip(df["mean_eval_fitness"]), df["std_eval_fitness"]
        ax.fill_between(x, _floorclip(df["mean_eval_fitness"] - sd), m + sd,
                        color=OKABE[0], alpha=0.15, linewidth=0)
        ax.plot(x, m, color=OKABE[0], lw=2, label="10-seed eval (±std)")
        if series["ep"].size and np.all(np.isfinite(series["step"])):
            ax.plot(series["step"] / 1000.0, _floorclip(series["med"]), "o",
                    color=OKABE[3], ms=6, zorder=5, label="51-seed median")
        xlabel = "training step (×1000)"
    else:
        if series["ep"].size:
            ax.fill_between(series["ep"], _floorclip(series["q1"]),
                            _floorclip(series["q3"]), color=OKABE[0],
                            alpha=0.15, linewidth=0)
            ax.plot(series["ep"], _floorclip(series["med"]), color=OKABE[0],
                    lw=2, marker="o", ms=5, label="51-seed median (IQR)")
        xlabel = "episode"
    ax.set_yscale("log")
    ax.set_ylabel("best error (log)")
    ax.set_title("optimisation error vs training", loc="left")
    if series["med"].size and np.nanmin(series["med"]) <= _ERR_FLOOR:
        ax.axhline(_ERR_FLOOR, color=OKABE[2], lw=1, ls="--",
                   label="optimum floor (error ≤ 1e-8 = solved)")
        ax.set_ylim(bottom=_ERR_FLOOR / 3)
    ax.legend(frameon=False, loc="upper right")

    # (b,c,d) return / loss / episode length. PPO: the DENSE per-update
    # progress_log on the EPISODE axis (smooth). DQN: the dense training_log on
    # its gradient-step axis. Sparse fallback: training_log points as markers.
    if use_prog:
        src, xr = prog, prog["episode"].to_numpy(float)
        xlab, mk = "episode", ""
    else:
        src, xr = df, df["training_step"].to_numpy(float) / 1000.0
        xlab, mk = "training step (×1000)", ("" if dense else "o")
    for ax, col, title, color in (
        (axes[0, 1], "mean_return", "mean episode return", OKABE[2]),
        (axes[1, 0], "mean_loss", "mean loss", OKABE[1]),
        (axes[1, 1], "mean_episode_length", "mean episode length (restarts)",
         OKABE[4]),
    ):
        ax.plot(xr, src[col].to_numpy(float), color=color, lw=2,
                marker=mk, ms=(0 if mk == "" else 5))
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlab)

    axes[0, 0].set_xlabel(xlabel)     # error panel: episode (sparse) or step
    for ax in axes.flat:
        _despine(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_path)


def _periodic_every(func_dir):
    """Episodes between periodic checkpoints, from the job's recorded resolved
    config (so chkp<k> is episode k*every). Falls back to 1000 (the EXP003
    cadence) if the metadata is absent or shaped differently."""
    try:
        with open(os.path.join(func_dir, "run_metadata.json")) as fh:
            d = json.load(fh)
        return int(d["resolved_config"]["training"]["periodic_eval"]["every"])
    except Exception:
        return 1000


def _checkpoint_series(func_dir, checkpoints):
    """Per-checkpoint learning trajectory with the TRUE episode<->step mapping.

    Each eval/chkp<k>/training_log.csv is the cumulative training log at the
    instant checkpoint k was taken, so its last training_step is exactly the
    step count reached at episode k*every — the honest x-position (no fudging).
    Returns arrays: ep (episode), step (training step), med/q1/q3 (51-seed final
    error median and IQR)."""
    every = _periodic_every(func_dir)
    ep, step, med, q1, q3 = [], [], [], [], []
    for k in checkpoints:
        cdir = _chkp_dir(func_dir, k)
        sp, ep_p = (os.path.join(cdir, "summary.csv"),
                    os.path.join(cdir, "episode_results.csv"))
        if not os.path.exists(sp):
            continue
        m = float(pd.read_csv(sp).iloc[0]["median"])
        ep.append(k * every); med.append(m)
        if os.path.exists(ep_p):
            fe = pd.read_csv(ep_p)["final_error"].to_numpy(float)
            q1.append(np.percentile(fe, 25)); q3.append(np.percentile(fe, 75))
        else:
            q1.append(m); q3.append(m)
        tl = os.path.join(cdir, "training_log.csv")
        # The cumulative log can be header-only at an early periodic checkpoint
        # (e.g. PPO, whose gradient-step count trails eval_every early on), so
        # training_step.max() is NaN — carry it as NaN rather than crashing on
        # int(); downstream plots already guard on np.isfinite(step).
        ts_max = np.nan
        if os.path.exists(tl):
            col = pd.read_csv(tl)["training_step"]
            if len(col) and np.isfinite(col.max()):
                ts_max = int(col.max())
        step.append(ts_max)
    return {k: np.array(v, dtype=float) for k, v in
            dict(ep=ep, step=step, med=med, q1=q1, q3=q3).items()}


def plot_learning_progress(func_dir, fid, checkpoints, out_path):
    """THE 'when did the agent stop learning?' figure. One continuous training
    run on an episode axis: the 51-seed checkpoint median (+IQR) at its true
    episode, the dense 10-seed eval mapped from steps onto episodes via the
    checkpoint anchors, and the best checkpoint marked."""
    s = _checkpoint_series(func_dir, checkpoints)
    if s["ep"].size == 0:
        print("    [skip] learning_progress: no checkpoint summaries")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    log_path = os.path.join(func_dir, "training_log.csv")
    if os.path.exists(log_path) and np.all(np.isfinite(s["step"])):
        df = pd.read_csv(log_path)
        if len(df):
            anchor_step = np.concatenate([[0.0], s["step"]])
            anchor_ep = np.concatenate([[0.0], s["ep"]])
            ep_of_step = np.interp(df["training_step"], anchor_step, anchor_ep)
            ax.plot(ep_of_step, _floorclip(df["mean_eval_fitness"]), color=MUTED,
                    lw=1, alpha=0.7,
                    label="dense 10-seed eval (per training step)")

    med, q1, q3 = _floorclip(s["med"]), _floorclip(s["q1"]), _floorclip(s["q3"])
    ax.fill_between(s["ep"], q1, q3, color=OKABE[0], alpha=0.15, linewidth=0)
    ax.plot(s["ep"], med, color=OKABE[0], lw=2.2, marker="o", ms=7,
            label="checkpoint 51-seed median (IQR band)")
    bi = int(np.argmin(s["med"]))
    ax.plot(s["ep"][bi], med[bi], marker="*", ms=18, color=OKABE[3],
            zorder=6, label=f"best: episode {int(s['ep'][bi])}")

    ax.set_yscale("log")
    # If the agent reached the optimum floor, show the floor line so the drop to
    # it reads as "solved" rather than falling off the axis.
    if np.nanmin(s["med"]) <= _ERR_FLOOR:
        ax.axhline(_ERR_FLOOR, color=OKABE[2], lw=1, ls="--",
                   label="optimum floor (error ≤ 1e-8 = solved)")
        ax.set_ylim(bottom=_ERR_FLOOR / 3)
    ax.set_xlabel("episode  (ONE continuous training run — checkpoints are "
                  "non-destructive snapshots)")
    ax.set_ylabel("best error (log)")
    ax.set_title(f"{fname(fid)} — when did the agent stop learning?", loc="left")
    ax.legend(frameon=False)
    _despine(ax)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------ continuous action-space (EXP009+) support
# The parameterized action space (param_strategy_continuous, EXP009/010/011) makes
# budget_frac / sampling_box continuous scales and F/CR agent-chosen, so they can
# no longer be a categorical mix (every row is a unique float). Strategy stays a
# discrete choice; the four continuous parameters are shown as median + IQR.
_CONT_PARAMS = [("F", "mutation F"),
                ("CR", "crossover CR"),
                ("budget_frac_action", "budget fraction / segment"),
                ("sampling_box_action", "sampling-box scale")]


def _action_space(func_dir):
    """environment.action_space from the job's resolved config, or None."""
    try:
        with open(os.path.join(func_dir, "run_metadata.json")) as fh:
            return json.load(fh)["resolved_config"]["environment"]["action_space"]
    except Exception:
        return None


def _is_continuous_actions(func_dir):
    """True if this experiment used a continuous parameterized action space
    (param_strategy_continuous or param_strategy_boxnp). Falls back to sniffing a
    step trace (a non-integer sampling_box means a continuous box scale)."""
    sp = _action_space(func_dir)
    if sp is not None:
        return sp in ("param_strategy_continuous", "param_strategy_boxnp")
    ks = discover_checkpoints(func_dir)
    if ks:
        p = os.path.join(_chkp_dir(func_dir, ks[-1]), "step_trace.csv")
        if os.path.exists(p):
            box = pd.read_csv(p, usecols=["sampling_box_action"])["sampling_box_action"]
            return bool((box.to_numpy(float) % 1 != 0).any())
    return False


def _cont_param_series(func_dir, checkpoints):
    """Per-checkpoint median/IQR of each continuous parameter, and the mix of each
    DISCRETE choice, pooled over all seeds/steps of a checkpoint's step_trace. The
    boxnp action space adds the NP parameter and the box_center choice.

    Returns (ks, params, stats, disc, mix, all_disc): params is the [(col,title)]
    list actually present; disc is the discrete axes ('strategy' [, 'box_center'])."""
    boxnp = _action_space(func_dir) == "param_strategy_boxnp"
    params = list(_CONT_PARAMS)
    if boxnp:
        params = params + [("pop_size_action", "population size NP")]
    disc = ["strategy"] + (["box_center"] if boxnp else [])
    cols = [c for c, _ in params]
    ks = []
    stats = {c: {"med": [], "q1": [], "q3": []} for c in cols}
    mix = {d: {} for d in disc}
    all_disc = {d: [] for d in disc}
    for k in checkpoints:
        p = os.path.join(_chkp_dir(func_dir, k), "step_trace.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, usecols=cols + disc)
        ks.append(k)
        for d in disc:
            mix[d][k] = df[d].value_counts(normalize=True)
            all_disc[d].extend(df[d].astype(str).tolist())
        for c in cols:
            v = df[c].to_numpy(float)
            stats[c]["med"].append(float(np.median(v)))
            stats[c]["q1"].append(float(np.percentile(v, 25)))
            stats[c]["q3"].append(float(np.percentile(v, 75)))
    return ks, params, stats, disc, mix, all_disc


def _plot_actions_evolution_continuous(func_dir, fid, checkpoints, out_path):
    ks, params, stats, disc, mix, all_disc = _cont_param_series(func_dir,
                                                                checkpoints)
    if not ks:
        print("    [skip] actions_evolution: no step traces")
        return
    n = len(disc) + len(params)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{fname(fid)} — discrete choices & chosen parameters across "
                 f"training", fontweight="bold", y=0.995)
    ai = 0
    for d in disc:                                 # strategy [, box_center]
        ax = axes[ai]; ai += 1
        cats = _ordered_categories(all_disc[d], d)
        cmap = _color_map(cats)
        frac = {c: [mix[d][k].get(c, 0.0) * 100 for k in ks] for c in cats}
        _stacked(ax, ks, frac, cats, cmap)
        ax.set_ylim(0, 100); ax.set_ylabel("usage (%)")
        ax.set_title(f"{_AXIS_TITLE.get(d, d)} (discrete choice)", loc="left")
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5),
                  title=d)
        _despine(ax)
    for j, (col, title) in enumerate(params):      # F, CR, budget, box [, NP]
        ax = axes[ai]; ai += 1
        color = OKABE[1 + j % (len(OKABE) - 1)]
        med = np.array(stats[col]["med"])
        q1, q3 = np.array(stats[col]["q1"]), np.array(stats[col]["q3"])
        ax.fill_between(ks, q1, q3, color=color, alpha=0.18, linewidth=0)
        ax.plot(ks, med, color=color, lw=2, marker="o", ms=4)
        ax.set_ylabel(title)
        ax.set_title(f"{title} — median chosen (IQR band)", loc="left")
        _despine(ax)
    for ax in axes:
        ax.set_xticks(ks)
    axes[-1].set_xlabel("checkpoint (episode ×1000)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out_path)


def _plot_actions_by_phase_continuous(func_dir, fid, mature_k, out_path):
    p = os.path.join(_chkp_dir(func_dir, mature_k), "step_trace.csv")
    if not os.path.exists(p):
        print("    [skip] actions_by_phase: no mature step_trace.csv")
        return
    cols = [c for c, _ in _CONT_PARAMS]
    df = pd.read_csv(p, usecols=cols + ["strategy", "step_idx"])
    steps = sorted(df["step_idx"].unique(), key=int)
    counts = df.groupby("step_idx").size()
    xs = list(range(len(steps)))

    fig, axes = plt.subplots(5, 1, figsize=(10, 15))
    fig.suptitle(f"{fname(fid)} — mature policy (chkp{mature_k}) by episode phase",
                 fontweight="bold", y=0.995)

    ax = axes[0]                                   # strategy mix per phase
    cats = _ordered_categories(df["strategy"].tolist(), "strategy")
    cmap = _color_map(cats)
    frac = {c: [] for c in cats}
    for st in steps:
        share = df[df["step_idx"] == st]["strategy"].value_counts(normalize=True)
        for c in cats:
            frac[c].append(share.get(c, 0.0) * 100)
    _stacked(ax, xs, frac, cats, cmap)
    ax.set_ylim(0, 100); ax.set_ylabel("usage (%)")
    ax.set_title("DE strategy (discrete choice)", loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5),
              title="strategy")
    _despine(ax)

    for ax, (col, title), color in zip(axes[1:], _CONT_PARAMS, OKABE[1:5]):
        med, q1, q3 = [], [], []
        for st in steps:
            v = df[df["step_idx"] == st][col].to_numpy(float)
            med.append(np.median(v)); q1.append(np.percentile(v, 25))
            q3.append(np.percentile(v, 75))
        ax.fill_between(xs, q1, q3, color=color, alpha=0.18, linewidth=0)
        ax.plot(xs, med, color=color, lw=2, marker="o", ms=4)
        ax.set_ylabel(title)
        ax.set_title(f"{title} — median chosen (IQR band)", loc="left")
        _despine(ax)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s}\n(n={counts[s]})" for s in steps])
    axes[-1].set_xlabel("segment index within episode (episode phase →)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out_path)


# --------------------------------------------------- 2. action-mix evolution
def plot_actions_evolution(func_dir, fid, checkpoints, out_path):
    if _is_continuous_actions(func_dir):
        return _plot_actions_evolution_continuous(func_dir, fid, checkpoints,
                                                  out_path)
    axes_cols = ["strategy", "budget_frac_action", "sampling_box_action"]
    per_axis_counts = {a: {} for a in axes_cols}      # axis -> {chkp -> Series}
    all_vals = {a: [] for a in axes_cols}
    for k in checkpoints:
        p = os.path.join(_chkp_dir(func_dir, k), "step_trace.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, dtype={c: str for c in axes_cols})
        for a in axes_cols:
            vc = df[a].value_counts(normalize=True)
            per_axis_counts[a][k] = vc
            all_vals[a].extend(df[a].tolist())

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.suptitle(f"{fname(fid)} — action mix across training",
                 fontweight="bold", y=0.995)
    for ax, a in zip(axes, axes_cols):
        cats = _ordered_categories(all_vals[a], a)
        cmap = _color_map(cats)
        ks = sorted(per_axis_counts[a])
        frac = {c: [per_axis_counts[a][k].get(c, 0.0) * 100 for k in ks]
                for c in cats}
        _stacked(ax, ks, frac, cats, cmap)
        ax.set_ylim(0, 100)
        ax.set_xticks(ks)
        ax.set_ylabel("usage (%)")
        ax.set_title(_AXIS_TITLE[a], loc="left")
        ax.legend(frameon=False, ncol=min(len(cats), 5),
                  loc="upper center", bbox_to_anchor=(0.5, -0.12))
        _despine(ax)
    axes[-1].set_xlabel("checkpoint (episode ×1000)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out_path)


# ------------------------------------------- 3. mature policy, action by phase
def plot_actions_by_phase(func_dir, fid, mature_k, out_path):
    if _is_continuous_actions(func_dir):
        return _plot_actions_by_phase_continuous(func_dir, fid, mature_k, out_path)
    p = os.path.join(_chkp_dir(func_dir, mature_k), "step_trace.csv")
    if not os.path.exists(p):
        print("    [skip] actions_by_phase: no mature step_trace.csv")
        return
    axes_cols = ["strategy", "budget_frac_action", "sampling_box_action"]
    df = pd.read_csv(p, dtype={c: str for c in axes_cols})
    steps = sorted(df["step_idx"].unique(), key=int)
    counts = df.groupby("step_idx").size()

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.suptitle(f"{fname(fid)} — mature policy (chkp{mature_k}) by episode phase",
                 fontweight="bold", y=0.995)
    for ax, a in zip(axes, axes_cols):
        cats = _ordered_categories(df[a].tolist(), a)
        cmap = _color_map(cats)
        frac = {c: [] for c in cats}
        for st in steps:
            sub = df[df["step_idx"] == st][a]
            share = sub.value_counts(normalize=True)
            for c in cats:
                frac[c].append(share.get(c, 0.0) * 100)
        _stacked(ax, list(range(len(steps))), frac, cats, cmap)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f"{s}\n(n={counts[s]})" for s in steps])
        ax.set_ylabel("usage (%)")
        ax.set_title(_AXIS_TITLE[a], loc="left")
        ax.legend(frameon=False, ncol=min(len(cats), 5),
                  loc="upper center", bbox_to_anchor=(0.5, -0.12))
        _despine(ax)
    axes[-1].set_xlabel("segment index within episode (episode phase →)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out_path)


# ------------------------------------------------- 4. episode restructure
def plot_episode_restructure(func_dir, fid, checkpoints, mature_k, out_path):
    mp = os.path.join(_chkp_dir(func_dir, mature_k), "episode_results.csv")
    if not os.path.exists(mp):
        print("    [skip] episode_restructure: no mature episode_results.csv")
        return
    mature = pd.read_csv(mp)

    means, stds, ks = [], [], []
    for k in checkpoints:
        ep = os.path.join(_chkp_dir(func_dir, k), "episode_results.csv")
        if os.path.exists(ep):
            n = pd.read_csv(ep)["n_steps"]
            ks.append(k); means.append(n.mean()); stds.append(n.std())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle(f"{fname(fid)} — episode restructure (restarts per episode)",
                 fontweight="bold", y=1.0)

    ax = axes[0]
    vals = mature["n_steps"].astype(int)
    lo, hi = vals.min(), vals.max()
    bins = np.arange(lo - 0.5, hi + 1.5, 1)
    ax.hist(vals, bins=bins, color=OKABE[0], edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(range(lo, hi + 1))
    ax.set_xlabel("restarts in episode (n_steps)")
    ax.set_ylabel("episodes (of 51)")
    ax.set_title(f"distribution at chkp{mature_k}", loc="left")
    _despine(ax)

    ax = axes[1]
    ax.errorbar(ks, means, yerr=stds, color=OKABE[3], lw=2, marker="o", ms=6,
                capsize=3, ecolor=MUTED, elinewidth=1)
    ax.set_xticks(ks)
    ax.set_xlabel("checkpoint (episode ×1000)")
    ax.set_ylabel("mean restarts / episode")
    ax.set_title("how it evolved (greedy policy, ±std)", loc="left")
    _despine(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_path)


# ------------------------------------------------- 5. convergence traces
def _stepwise(xs, ys, grid):
    """Right-continuous step interpolation: value at grid[g] is the best-so-far
    error known by that budget (last knot with xs <= grid[g])."""
    idx = np.clip(np.searchsorted(xs, grid, side="right") - 1, 0, len(ys) - 1)
    return ys[idx]


def _convergence_band(func_dir, k):
    """Median and IQR of best-so-far error vs cumulative budget, over all eval
    seeds at checkpoint k. Built from step_trace's global_best_error_before per
    segment boundary plus the final after-value — the honest best-so-far curve."""
    st = os.path.join(_chkp_dir(func_dir, k), "step_trace.csv")
    er = os.path.join(_chkp_dir(func_dir, k), "episode_results.csv")
    if not (os.path.exists(st) and os.path.exists(er)):
        return None
    df = pd.read_csv(st)
    total = float(pd.read_csv(er)["total_FEs"].max())
    starts = []
    curves = []
    for _seed, g in df.groupby("seed"):
        g = g.sort_values("step_idx")
        bub = g["budget_used_before"].to_numpy(float)
        xs = np.append(bub, total)
        ys = np.append(g["global_best_error_before"].to_numpy(float),
                       g["global_best_error_after"].to_numpy(float)[-1])
        starts.append(bub[0])
        curves.append((xs, ys))
    grid = np.linspace(min(starts), total, 200)
    stacked = np.vstack([_stepwise(xs, ys, grid) for xs, ys in curves])
    med = np.median(stacked, axis=0)
    q1 = np.percentile(stacked, 25, axis=0)
    q3 = np.percentile(stacked, 75, axis=0)
    return grid, med, q1, q3


def plot_convergence(func_dir, fid, first_k, mature_k, out_path):
    series = [(first_k, "untrained", MUTED),
              (mature_k, "mature", OKABE[0])]
    bands = [(lbl, col, _convergence_band(func_dir, k))
             for k, lbl, col in series]
    bands = [(lbl, col, b) for lbl, col, b in bands if b is not None]
    if not bands:
        print("    [skip] convergence: no step_trace data")
        return

    # Log floor: half the smallest positive median across the plotted series
    # (an agent that solves a function reaches error 0, which log-y can't show).
    pos = [b[1][b[1] > 0] for _l, _c, b in bands if np.any(b[1] > 0)]
    floor = max(np.concatenate(pos).min() * 0.5, 1e-12) if pos else 1e-8

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lbl, col, (grid, med, q1, q3) in bands:
        gk = grid / 1000.0
        ax.fill_between(gk, np.clip(q1, floor, None), np.clip(q3, floor, None),
                        color=col, alpha=0.15, linewidth=0)
        ax.plot(gk, np.clip(med, floor, None), color=col, lw=2,
                label=f"chkp{first_k if lbl == 'untrained' else mature_k} "
                      f"({lbl})")
    ax.set_yscale("log")
    ax.set_xlabel("budget used (FEs ×1000)")
    ax.set_ylabel("best-so-far error (log, median + IQR)")
    ax.set_title(f"{fname(fid)} — convergence: what training bought",
                 loc="left")
    ax.legend(frameon=False, loc="upper right")
    _despine(ax)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------- 6. cross-function comparison
def plot_cross_function(exp_dir, functions, out_path):
    """One scale-free view: agent vs each baseline, per function, as % error
    improvement (positive = agent reaches lower error). Raw errors span orders
    of magnitude across functions, so only a normalised measure is comparable;
    pct_diff is already in comparison.csv, paired and Wilcoxon-tested."""
    records = []      # (fid, baseline, pct_diff, p_value)
    baselines = []
    for fid in functions:
        func_dir = os.path.join(exp_dir, f"f{fid}")
        ks = discover_checkpoints(func_dir)
        if not ks:
            continue
        cp = os.path.join(_chkp_dir(func_dir, ks[-1]), "comparison.csv")
        if not os.path.exists(cp):
            continue
        for _i, r in pd.read_csv(cp).iterrows():
            b = r["baseline_name"]
            if b not in baselines:
                baselines.append(b)
            records.append((fid, b, float(r["pct_diff"]), float(r["p_value"])))
    if not records:
        print("  [skip] cross_function: no comparison.csv found")
        return
    df = pd.DataFrame(records, columns=["fid", "baseline", "pct", "p"])

    cmap = _color_map(baselines)
    fids = [f for f in functions if f in set(df["fid"])]
    x = np.arange(len(fids))
    n = len(baselines)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(fids)), 6))
    for j, b in enumerate(baselines):
        heights, ps = [], []
        for f in fids:
            row = df[(df["fid"] == f) & (df["baseline"] == b)]
            heights.append(float(row["pct"].iloc[0]) if len(row) else 0.0)
            ps.append(float(row["p"].iloc[0]) if len(row) else 1.0)
        xb = x + (j - (n - 1) / 2) * w
        ax.bar(xb, heights, width=w, color=cmap[b], edgecolor=SURFACE,
               linewidth=0.6, label=b.replace("BASE00", "B"))
        for xi, h, pv in zip(xb, heights, ps):
            star = ("***" if pv < 1e-3 else "**" if pv < 1e-2
                    else "*" if pv < 5e-2 else "")
            if star:
                ax.text(xi, h + (2 if h >= 0 else -2), star, ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=9,
                        color=MUTED)

    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([fname(f).split(" — ")[0] for f in fids])
    ax.set_ylabel("agent vs baseline: % lower error  (↑ agent better)")
    ax.set_title("How differently is the agent doing across functions?\n"
                 "mature policy (final checkpoint) vs each baseline; "
                 "* p<0.05  ** p<0.01  *** p<0.001 (Wilcoxon, paired)",
                 loc="left")
    ax.legend(frameon=False, ncol=n, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    _despine(ax)
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", required=True,
                    help="Experiment folder name under experiments/ "
                         "(e.g. EXP003_dqn_restart)")
    ap.add_argument("--experiments-dir",
                    default=os.path.join(REPO_ROOT, "experiments"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "tests"),
                    help="Root for the generated figures (default: tests/)")
    args = ap.parse_args()

    _style()
    exp_dir = os.path.join(args.experiments_dir, args.exp)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"No experiment at {exp_dir}")
    functions = discover_functions(exp_dir)
    if not functions:
        raise SystemExit(f"No per-function eval folders under {exp_dir}")

    out_root = os.path.join(args.out_dir, args.exp)
    os.makedirs(out_root, exist_ok=True)
    print(f"{args.exp}: functions {functions} -> "
          f"{os.path.relpath(out_root, REPO_ROOT)}")

    for fid in functions:
        func_dir = os.path.join(exp_dir, f"f{fid}")
        checkpoints = discover_checkpoints(func_dir)
        if not checkpoints:
            print(f"  f{fid}: no checkpoints, skipping")
            continue
        first_k, mature_k = checkpoints[0], checkpoints[-1]
        fdir = os.path.join(out_root, f"f{fid}")
        os.makedirs(fdir, exist_ok=True)
        print(f"  f{fid}: chkp{first_k}..chkp{mature_k}")

        # The retained figure set (per user): learning_progress, learning_curve,
        # actions_evolution. convergence / episode_restructure / actions_by_phase
        # were dropped. plot_* for the dropped ones remain in the module (still
        # importable) but are no longer generated.
        plot_learning_progress(func_dir, fid, checkpoints,
                               os.path.join(fdir, "learning_progress.png"))
        plot_learning_curve(func_dir, fid, checkpoints,
                            os.path.join(fdir, "learning_curve.png"))
        plot_actions_evolution(func_dir, fid, checkpoints,
                               os.path.join(fdir, "actions_evolution.png"))

    # Experiment-level summary (agent vs baselines across functions); kept.
    plot_cross_function(exp_dir, functions,
                        os.path.join(out_root, "cross_function.png"))
    print("done.")


if __name__ == "__main__":
    main()
