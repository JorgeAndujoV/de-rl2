# de-rl2 — Machinery Manual

A practical, detailed description of how the project works **as currently
configured** (experiment `EXP002_dqn_restart`). It is written for someone who
understands optimization and machine learning broadly but is not a specialist in
Differential Evolution or reinforcement learning.

> **Nothing here is permanent.** The agent, its hyperparameters, the action
> space, the observation vector, the reward, and the DE strategies are all
> *choices*, selected by name in the config and implemented behind small
> registries. A different choice is a **new experiment**, not an edit to an old
> one. This manual describes what is wired up *today*; every "current" value can
> change in a future `EXP00N`.

Contents:
1. [The framework in general](#1-the-framework-in-general)
2. [The agent](#2-the-agent)
3. [The environment (and the observation vector in depth)](#3-the-environment)
4. [The optimizer](#4-the-optimizer)
5. [How the files connect](#5-how-the-files-connect)
6. [The config file, parameter by parameter](#6-the-config-file)

---

## 1. The framework in general

### The problem

We are minimizing a hard benchmark function — currently **CEC'13 function 11
(Rastrigin, 30 dimensions)** — under a fixed budget of **300,000 function
evaluations (FEs)**. "One function evaluation" = computing `f(x)` for one
candidate point `x`. The score of a run is the **error**: `f(best point found) −
f(global optimum)`, floored at 0. Lower is better; 0 means solved.

### The core idea: optimization as a sequence of restarts

A normal Differential Evolution (DE) run evolves a single population
continuously until the budget runs out. This project does something different:
it runs DE as a **sequence of independent short bursts** ("segments"), and a
reinforcement-learning **agent** decides, between bursts, how the next burst
should be configured. Each burst starts a *fresh* population — there is **no
elitism**, meaning the best-so-far point is never copied into the new
population. (The framework still *remembers* the best-ever point across the whole
episode for scoring and reward; it just never re-injects it.)

The intuition: a single long DE run can get stuck in a local basin. Restarting
with a fresh population — sometimes in a wide region to explore, sometimes in a
tight region around where the last burst converged — can escape those traps. The
question the project asks is: **can an agent learn *when* and *how* to restart
better than fixed rules?**

### One episode, step by step

An **episode** is one complete optimization of one function under the full FE
budget. It unfolds as:

1. **Warm-up segment (no agent decision).** DE runs with a fixed configuration
   (strategy `rand/1/bin`, F=0.5, CR=0.9), its initial population sampled
   uniformly over the *entire* search domain, consuming **5%** of the budget
   (15,000 FEs). This establishes a first population and a first best point.

2. **Observe.** The environment summarizes the segment that just finished into a
   fixed-length **observation vector** (101 numbers — see §3).

3. **Act.** The agent reads the observation and emits a single **action** — one
   integer in `0..59`. That integer decodes into three simultaneous choices:
   - which **DE strategy** to use (5 options),
   - what fraction of the total budget to spend on this next burst [0.05, 0.95],
   - how to build the **sampling box** — the region the fresh population is drawn
     from [0.25, 3.0].
      f_range: [0.0, 1.0]
      cr_range: [0.0, 1.0]
      we are trying to see if we add the sampling box as a regular box or as the covariance matrix of final population.
      also we have as actions the NP and where the sampling box is initialized (random, center or incumbent), te complete action space is the following: 
        action_space: param_strategy_boxnp
        f_range: [0.0, 1.0]
        cr_range: [0.0, 1.0]
        budget_range: [0.05, 0.95]
        box_scale_range: [0.25, 3.0]
        np_range: [16, 400]
        box_centers: [centroid, incumbent, random]
        strategy_profiles:
          - {strategy: rand/1/bin}
          - {strategy: best/1/bin}
          - {strategy: current-to-best/1/bin}
          - {strategy: rand/2/bin}
          - {strategy: current-to-pbest/1/bin}
        reward: {name: log_stagnation, lambda: 0.1, tau_stag: 3}

4. **Transform the box.** The previous segment's final population has a bounding
   box. The chosen sampling-box action scales that box (wider or tighter),
   floors it so it can't collapse to a point, and clips it to the domain.

5. **Run the segment.** A fresh population is sampled uniformly inside that box,
   and DE evolves it for `budget_fraction × 300,000` FEs.

6. **Reward and repeat.** The environment computes a reward from how much the
   global best improved, produces the next observation, and returns control to
   the agent. Steps 2–6 repeat until the remaining budget is too small to seed
   another population.

Because different actions spend different amounts of budget, **episode length is
variable**: roughly **7 decisions** (if the agent always picks 15%) up to **19
decisions** (if it always picks 5%).

### What the agent controls vs. what is fixed

| Controlled by the agent (per step)      | Fixed (config constants)                  |
|-----------------------------------------|-------------------------------------------|
| DE strategy (4 options)                 | F and CR (attached to each strategy)      |
| Budget fraction (5% / 10% / 15%)        | Population size (50)                       |
| Sampling-box scale (5 options)          | Warm-up config and length (5%)            |
|                                         | Domain, budget, benchmark function        |

---

## 2. The agent

The current agent is a **Deep Q-Network (DQN)** — a value-based RL method.

### What a DQN does

For a given observation, the network outputs one number per possible action:
the estimated **Q-value**, i.e. the expected total future reward of taking that
action and behaving well afterward. To act **greedily** (as in evaluation), the
agent picks the action with the highest Q-value. During training it uses
**ε-greedy** exploration: with probability ε it picks a random action instead, so
it keeps discovering alternatives.

### Network

A plain multilayer perceptron (`src/derl2/agents/dqn.py`):

```
input: 101 observation features
  → Dense(100, ReLU) → Dense(75, ReLU) → Dense(50, ReLU)
  → Dense(60)   # one Q-value per action
```

### How it learns

- **Replay buffer.** Every step produces a transition `(obs, action, reward,
  next_obs, done, τ)`; these are stored in a circular buffer (capacity 100,000).
  Training samples random mini-batches from it, which decorrelates consecutive
  steps.
- **Target network.** A second, frozen copy of the network provides stable
  targets for the Bellman update; it is hard-synced to the live network every
  500 training steps.
- **Bellman target with SMDP discounting (the one non-standard piece).** In
  ordinary DQN the target is `r + γ · max_a' Q_target(next_obs, a')`. Here each
  step consumes a *different amount of budget*, so discounting by a flat γ per
  step would be inconsistent. Instead we discount by **γ^τ**, where **τ is the
  segment's elapsed budget measured in units of the smallest budget fraction
  (5%)**. A 15%-budget step therefore discounts three times as hard as a 5% step.
  This "semi-Markov" (SMDP) correction is why τ travels with every transition.
  The config flag `agent.discounting` can switch this off (`per_step`) as an
  ablation; currently it is `per_budget`.
- **Loss.** Huber loss between predicted Q and the Bellman target; Adam
  optimizer, learning rate 1e-3.
- **Exploration schedule.** ε decays **linearly** from 1.0 to 0.05 over 8,000
  *training steps* (gradient updates, not episodes), then stays at 0.05.

The agent knows only two things about the outside world: the observation size
(101) and the number of actions (60). It knows nothing about DE, restarts, or the
benchmark — that separation is deliberate (see §5).

---

## 3. The environment

The environment (`src/derl2/environments/env.py`, class `DEEnv`) is the glue
between the optimizer and the agent. It follows the familiar RL `reset()` /
`step(action)` interface. It **never runs a DE generation loop itself** — it asks
the optimizer to run a whole segment — and it **never sees a raw action's
meaning** — an "action space" object decodes the integer for it.

Its responsibilities:
- pick the function and run the warm-up (`reset`);
- decode the agent's action into a DE configuration (`action_space`);
- transform the sampling box (`sampling_box`);
- call the optimizer for one segment (`run_segment`);
- track the global best across the episode (no elitism);
- compute the reward (`reward`);
- build the next observation (`observation`);
- log everything the reporting layer needs into the `info` dict.

### 3.1 The action space (how the integer becomes a configuration)

Current action space: `profiles_budget_area`
(`src/derl2/environments/action_spaces.py`). It packs three independent choices
into one index in `0..59`:

```
n = |strategies| × |budget_fracs| × |boxes| = 4 × 3 × 5 = 60

strategy_idx = raw // (3 × 5)     # 0..3   (boxes vary fastest, then budget, then strategy)
budget_idx   = (raw // 5) % 3     # 0..2
box_idx      =  raw % 5           # 0..4
```

The decoded configuration handed to the optimizer is
`{strategy, F, CR, budget_frac, sampling_box}`, where **F and CR come from the
chosen strategy's profile**, not from the action index. The four strategy
profiles in the current experiment are:

| idx | strategy               | F   | CR  | character                         |
|-----|------------------------|-----|-----|-----------------------------------|
| 0   | rand/1/bin             | 0.5 | 0.9 | pure exploration                  |
| 1   | best/1/bin             | 0.8 | 0.5 | strong exploitation of the best   |
| 2   | current-to-best/1/bin  | 0.5 | 0.3 | middle ground                     |
| 3   | rand/2/bin             | 0.5 | 0.3 | more diverse mutants than rand/1  |

The three budget fractions are `{0.05, 0.10, 0.15}`; the five sampling-box scales
are `{2.0, 1.5, 1.0, 0.667, 0.5}` (see §3.3).

### 3.2 The sampling-box transform (what connects two segments)

The sampling box is the *only* state carried from one segment to the next
(`src/derl2/environments/sampling_box.py`, `transform_box`). Given the previous
segment's final population and the chosen box action:

1. Compute the population's per-dimension **half-width** (how spread out it is).
2. Choose a **center**: currently the population **centroid** (`box_center:
   centroid`; the alternative is the incumbent best point).
3. **Scale** the half-width by the action's factor (`box_scales[box_idx]`): 2.0
   doubles the region (explore wider than the population), 0.5 halves it (exploit
   tighter). 1.0 reuses the population's own spread.
4. **Floor** the half-width so it never collapses below `box_min_frac` (1/24) of
   the domain half-width — a fully converged population would otherwise give a
   zero-size box.
5. **Clip** the box to the true search domain.

The fresh population for the next segment is then drawn uniformly inside
`[box_lo, box_hi]`. Note the box constrains only *where the population starts*;
DE mutation can still move points anywhere in the domain.

### 3.3 The observation vector — in depth

This is the agent's entire view of the world after each segment. It is a
**101-dimensional float vector**, built by the `traj20` observation
(`src/derl2/environments/observations.py`, class `Traj20`).

**Where the raw material comes from.** During a segment the optimizer records a
**raw trajectory**: a `(20, 5)` array — 20 evenly spaced checkpoints across the
segment's generations, each with 5 raw channels:

| raw channel | meaning                                                        |
|-------------|----------------------------------------------------------------|
| 0 `box_width_frac` | population spread ÷ domain width (a diversity measure)  |
| 1 `success_rate`   | fraction of trial points that replaced their target since the previous checkpoint |
| 2 `best_fitness`   | best raw fitness in the segment at that checkpoint      |
| 3 `mean_fitness`   | population mean fitness at that checkpoint              |
| 4 `fes_used`       | cumulative FEs spent at that checkpoint                 |

The optimizer only **records** this raw superset; the observation module
**selects and transforms** it into features. (This is why the raw array is
`20×5` but the observation is not simply "flatten those 100 numbers + 1".) All
fitness quantities are **log-scaled**; all spatial quantities are expressed as
**fractions of the domain width**, so no single feature dominates by raw
magnitude. The 101 features come in three blocks.

#### Block A — the trajectory (80 features)

Four channels, each a length-20 vector over the checkpoints, laid out
**channel-major** (all 20 of A1, then all 20 of A2, A3, A4):

- **A1 `box_width[k]`** (20) — the raw diversity channel: how spread out the
  population is at each checkpoint. A shrinking A1 means the population is
  converging.
- **A2 `success_rate[k]`** (20) — the raw success channel: how often trials are
  improving on their parents. High = still making progress; near zero = stalled.
- **A3 `log_improvement[k]`** (20) — `log10(1 + (best[k−1] − best[k]))`: how much
  the best fitness improved between checkpoints, log-compressed. The baseline for
  the first checkpoint is the freshly sampled population's best (before any
  generation). Non-negative because best fitness is non-increasing.
- **A4 `mean_best_gap[k]`** (20) — `log10(1 + (mean[k] − best[k]))`: the gap
  between the average individual and the best, log-compressed. A large gap means
  a diverse population with one strong leader; a gap near zero means the whole
  population has converged onto the best.

Together, Block A is a compact "shape of the burst": did it converge fast or
slow, keep improving or stall, stay diverse or collapse.

#### Block B — the segment outcome (6 features)

A summary of what the just-finished segment achieved:

- **B1** — `log10((incumbent_error_before + ε) / (segment_best_error + ε))`,
  clipped to `[−8, 8]`. The orders-of-magnitude change in the global best from
  this segment. **Positive = the segment improved things**; ≈0 = flat; **negative
  = the segment landed *worse* than the incumbent**, which is a normal outcome
  here because there is no elitism.
- **B2** — `improved_flag`: 1.0 if the global best improved this segment, else 0.
- **B3** — `min(n_stag / 5, 1)`: the **stagnation streak** (number of consecutive
  segments with no global-best improvement) *after* this segment, normalized to
  `[0, 1]`.
- **B4** — how far the segment's best point sits from the **center of the box it
  was sampled in**, as an RMS over dimensions of (displacement ÷ box half-width),
  clipped to `[0, 10]`. ≈1 means the best sits near the box boundary (the good
  region may lie *outside* the box we chose); ≈0 means it's near the center.
- **B5** — **box contraction**: final population width ÷ initial box width. <1
  means the population contracted during the segment (converging).
- **B6** — `log10(global_best_error + 1e-12)`: the absolute scale of the
  best-so-far error. Tells the agent *how well it is doing overall*, not just the
  last delta.

#### Block C — context (15 features)

Where we are in the episode and what was just done:

- **C1** — `budget_remaining_frac`: fraction of the 300k budget still unspent
  `[0, 1]`.
- **C2** — `n_steps_taken / 19`: how many decisions have been made, normalized by
  the maximum possible (~19).
- **C3a** — one-hot of the **previous strategy** (length 4).
- **C3b** — previous **F**.
- **C3c** — previous **CR**. (C3b/C3c are informative even though F/CR are tied to
  the strategy — the observation contract is generic and doesn't assume that.)
- **C3d** — previous **budget fraction**.
- **C3e** — one-hot of the **previous sampling-box** action (length 5). All zeros
  on the very first step, because the warm-up used full-domain initialization,
  not a box action.
- **C4** — `current_box_width_frac`: the width of the current sampling box ÷
  domain width.

Total: **80 + 6 + 15 = 101**.

The very first observation (right after the warm-up) describes the warm-up
segment: its previous-strategy fields hold the warm-up config, its box one-hot is
all zeros, and its `current_box_width_frac` is 1.0 (the whole domain).

### 3.4 The reward

Current reward: `stagnation` (`src/derl2/environments/rewards.py`), with λ=0.1,
τ_stag=3.

```
R_t = (error_best − error_new)  −  λ · 1[n_stag ≥ τ_stag] · improved      (every step)
      + (−error_best)                                                     (only at step t = 1)
```

- `error_best` / `error_new` = global best error **before** / **after** folding in
  the segment just run. Since the global best only ever improves, `error_best −
  error_new ≥ 0`: **the step reward is the amount of error removed this segment.**
- The penalty term subtracts λ only on a step that *improves* while the agent had
  already been stagnating for ≥ τ_stag segments — discouraging a strategy of
  sitting idle and then getting lucky.
- The one-off `−error_best` on the first step makes the per-step rewards
  **telescope** to `−(final error)`, so **maximizing episode return is equivalent
  to minimizing final error** (exactly under flat γ; approximately under the γ^τ
  discounting, which preserves the *ordering* of returns — what matters for
  learning).
- Errors below `1e-8` are treated as exactly 0 (the CEC'13 convention).

---

## 4. The optimizer

The optimizer's entire public surface is a **single function**,
`run_segment(...)` (`src/derl2/optimizers/de.py`). It runs **one DE segment** —
one fresh population evolved with one fixed configuration over a fixed FE budget
— and knows nothing about RL, episodes, or restarts.

### What one segment does

Given a sampling box, a strategy, F, CR, and an FE budget:

1. **Sample** `pop_size` (50) points uniformly inside the box; evaluate them
   (`pop_size` FEs). This is generation 0.
2. **Evolve.** The number of affordable generations is `fe_budget // pop_size −
   1`, each costing another `pop_size` FEs. Every generation does the classic DE
   step (see below).
3. **Record** the 20-checkpoint raw trajectory (§3.3) and return a
   `SegmentResult`: final population, best point and fitness, FEs used, and the
   trajectory.

### One DE generation (carried over verbatim from the validated de-rl core)

For each individual `x_i` in the population:

1. **Pick distinct random indices** (how many depends on the strategy), none
   equal to `i`.
2. **Mutate** to form a donor vector `v`, per the strategy
   (`src/derl2/optimizers/strategies.py`):
   - `rand/1`: `x_r1 + F·(x_r2 − x_r3)` — pure exploration, ignores the best.
   - `best/1`: `x_best + F·(x_r1 − x_r2)` — pulls every mutant toward the best.
   - `current-to-best/1`: `x_i + F·(x_best − x_i) + F·(x_r1 − x_r2)` — partway to
     the best plus a random step.
   - `rand/2`: `x_r1 + F·(x_r2 − x_r3) + F·(x_r4 − x_r5)` — two differentials,
     more diverse.
   Here **F** is the scale factor (how big a step the differences make).
3. **Clip** the donor to the true domain (not the sampling box — the box only
   constrains the *initial* sampling; evolution can roam the whole domain).
4. **Binomial crossover** with rate **CR**: each dimension of the trial takes the
   donor's value with probability CR, else keeps `x_i`'s value; one dimension is
   always forced from the donor so the trial differs from the parent.
5. **Greedy selection**: evaluate the trial; if it is no worse than `x_i`, it
   replaces `x_i`. (Selection is synchronous — the whole population updates at
   once.)

The implementation is **vectorized in TensorFlow** (the whole population's
generation runs as tensor ops), which is why TensorFlow is a dependency even for
the non-learning baselines. The generation-level behavior is treated as
authoritative and is not modified by this project — only how segments are
*strung together* is new.

---

## 5. How the files connect

The guiding rule is a **one-way dependency chain**: the optimizer knows nothing
about the environment; the environment knows the optimizer and its registries but
nothing about the agent; the agent knows only its input/output sizes. Swapping
any one layer never forces edits in the others.

```
optimizer  ─────▶  environment  ─────▶  agent
(run_segment)      (DEEnv)              (DQN)
```

### The modules

| File | Role | Called by | Calls / uses |
|------|------|-----------|--------------|
| `config.py` | Loads, validates, and (with `--smoke`/`--set`) resolves the YAML config. No hidden defaults. | everything | — |
| `run_metadata.py` | Writes/reads `run_metadata.json` (the authoritative per-job record: git commit, versions, resolved config, status). | `run_experiment.sh`, `train.py`, `evaluate.py`, `run_baseline.py` | `config.py` |
| `training/train.py` | Training entry point: builds env + agent + buffer, runs the episode loop, checkpoints. | `run_experiment.sh` | `DEEnv`, `build_agent`, `ReplayBuffer`, `run_metadata` |
| `environments/env.py` (`DEEnv`) | The episode loop (`reset`/`step`); orchestrates the pieces below. | `train.py`, `evaluate.py`, `run_baseline.py` | `run_segment`, `build_observation`, `build_action_space`, `build_reward`, `build_benchmark`, `transform_box` |
| `environments/action_spaces.py` | Decodes an action index → `{strategy, F, CR, budget_frac, sampling_box}`. | `DEEnv` | — |
| `environments/observations.py` | Builds the 101-feature observation from the environment's context dict. | `DEEnv` | — |
| `environments/rewards.py` | Computes the scalar reward per step. | `DEEnv` | — |
| `environments/sampling_box.py` | Transforms the previous population into the next segment's sampling box. | `DEEnv` | — |
| `optimizers/de.py` (`run_segment`) | Runs one DE segment; records the raw trajectory. | `DEEnv`, `run_baseline.py` (BASE001) | `build_strategy` |
| `optimizers/strategies.py` | The DE mutation strategies (NumPy + TF). | `de.py` | — |
| `benchmarks/…` | `build_benchmark('cec13:f11', dim)` → a `BenchmarkSpec` (objective, optimum, bounds). | `DEEnv`, `run_baseline.py` | CEC'13 data files |
| `agents/dqn.py` (`DQNAgent`) | The learner: `act`, `train`, `epsilon`, checkpoint items. | `train.py`, `evaluate.py` | — |
| `replay/buffer.py` (`ReplayBuffer`) | Stores transitions (with τ); samples mini-batches. | `train.py` | — |
| `evaluation/evaluate.py` | Independent evaluation: restores a checkpoint, runs greedy episodes, writes the result CSVs and the baseline comparison. | `run_experiment.sh` | `run_metadata`, `DEEnv`, `build_agent`, baselines |
| `scripts/run_baseline.py` | Produces the frozen baselines (hand-run, once). | you (by hand) | `DEEnv`, `run_segment`, policies |
| `run_experiment.sh` | Cluster driver: reads Slurm resources, writes metadata, submits the job, runs train→eval, persists results. | you (by hand) | `config.py`, `run_metadata`, `train.py`, `evaluate.py` |

### Data flow of one training step

```
train.py loop
  obs ──▶ agent.act(obs, ε) ──▶ action (0..59)
  action ──▶ env.step(action)
                 │  action_space.decode(action)      → {strategy, F, CR, budget_frac, sampling_box}
                 │  sampling_box.transform_box(...)   → next box
                 │  run_segment(...)                  → SegmentResult (+ raw trajectory)
                 │  reward_fn(ctx)                    → reward
                 │  observation.build(ctx)            → next_obs (101)
                 ▼
        returns (next_obs, reward, done, info)
  buffer.add(obs, action, reward, next_obs, done, τ)
  if buffer warm enough: agent.train(buffer.sample(batch)) → one gradient step
```

Evaluation (`evaluate.py`) is the same loop with ε=0 (greedy) and no learning; it
reads the resolved config from `run_metadata.json`, restores the trained network
from `checkpoints/final/`, runs the held-out seeds (starting at 900000), and
writes the result files — including `comparison.csv` and `eval_results.csv`,
which pit the agent against the frozen baselines on the identical seeds (paired
Wilcoxon test).

---

## 6. The config file

Every knob a run depends on is declared explicitly in
`experiments/EXP002_dqn_restart/config.yaml` — there are no hidden defaults, and a
missing key is an error, not a silent fallback. Below is each block. (`--smoke`
scales a few of these down for a fast crash-test: dim→5, budget→3000, tiny
buffers, 3 episodes, 2 eval seeds; the observation length is deliberately *not*
scaled.)

### Top level

| key | current | meaning |
|-----|---------|---------|
| `experiment` | `EXP002_dqn_restart` | Experiment id; also the output folder name. |
| `seed` | `0` | Master seed. Training episode seeds derive as `seed·1e6 + episode`; evaluation seeds are separate (see `evaluation.seed_offset`). |

### `benchmark` — the problem

| key | current | meaning |
|-----|---------|---------|
| `name` | `cec13` | Benchmark suite. |
| `functions` | `[11]` | Which function id(s) to optimize (a list, or `"all"` for f1–f20). f11 is Rastrigin. |
| `dim` | `30` | Dimensionality `D` of the search space. |
| `budget` | `300000` | Total function evaluations per episode. |

### `episode` — the restart mechanics

| key | current | meaning |
|-----|---------|---------|
| `pop_size` | `50` | Population size `NP` per segment. |
| `warmup_frac` | `0.05` | Fraction of the budget spent on the fixed warm-up segment (5% = 15,000 FEs). |
| `warmup.strategy/F/CR` | `rand/1/bin`, 0.5, 0.9 | The warm-up's fixed DE configuration. |
| `box_center` | `centroid` | Center of the next sampling box: `centroid` (population mean) or `incumbent` (best point). |
| `box_scales` | `[2.0, 1.5, 1.0, 0.667, 0.5]` | The five sampling-box half-width multipliers the agent chooses among (>1 explore wider, <1 exploit tighter). |
| `box_min_frac` | `0.0417` | Floor on box half-width as a fraction of the domain half-width (1/24), so a converged population can't produce a zero-size box. |
| `elitism` | `false` | Never inject the incumbent into a new population. (Only `false` is implemented — the restart idea depends on it.) |
| `truncate_last_segment` | `true` | If the last segment's requested budget exceeds what's left, clamp it to the remainder instead of overshooting. |

### `environment` — observation, actions, reward

| key | current | meaning |
|-----|---------|---------|
| `observation` | `traj20` | Which observation builder (→ 101 features; see §3.3). |
| `n_checkpoints` | `20` | Checkpoints `K` in the raw trajectory. Must match the observation (`traj20` expects 20). |
| `action_space` | `profiles_budget_area` | Which action encoding (→ 60 actions; see §3.1). |
| `budget_fracs` | `[0.05, 0.10, 0.15]` | The three per-segment budget fractions the agent chooses among. |
| `strategy_profiles` | 4 × `{strategy, F, CR}` | The DE strategies and **their fixed F/CR** (see §3.1 table). **This is where EXP002 differs from EXP001**: per-strategy F/CR instead of uniform 0.5/0.9. |
| `reward.name` | `stagnation` | Which reward (see §3.4). |
| `reward.lambda` | `0.1` | Penalty weight λ for improving only after a long stagnation streak. |
| `reward.tau_stag` | `3` | Stagnation threshold τ (segments) before that penalty can apply. |

The action count is `|strategy_profiles| × |budget_fracs| × 5 = 4 × 3 × 5 = 60`,
and the observation size is fixed at 101 by `traj20`.

### `agent` — the DQN

| key | current | meaning |
|-----|---------|---------|
| `name` | `dqn` | Which agent. |
| `discounting` | `per_budget` | `per_budget` = SMDP γ^τ (budget-aware); `per_step` = flat γ (ablation). |
| `gamma` | `0.99` | Discount base γ. |
| `lr` | `1.0e-3` | Adam learning rate. |
| `hidden` | `[100, 75, 50]` | Hidden layer sizes of the Q-network MLP. |
| `buffer_size` | `100000` | Replay buffer capacity (transitions). |
| `batch_size` | `64` | Mini-batch size per gradient step. |
| `target_update` | `500` | Training steps between hard target-network syncs. |
| `warmup_transitions` | `100` | Transitions collected before learning starts. |
| `epsilon.start/end/decay_steps` | 1.0 / 0.05 / 8000 | Linear ε schedule over *training steps* (gradient updates). |

### `training`

| key | current | meaning |
|-----|---------|---------|
| `episodes` | `1200` | Number of training episodes (~14k gradient steps at this scale). |
| `eval_every` | `500` | Training steps between learning-curve evaluation checkpoints. |
| `eval_episodes` | `10` | Greedy rollouts averaged per learning-curve checkpoint. |
| `checkpoint_every` | `500` | Training steps between saved checkpoints. |

### `evaluation`

| key | current | meaning |
|-----|---------|---------|
| `runs_per_function` | `51` | Independent evaluation seeds per function. |
| `seed_offset` | `900000` | First evaluation seed (seeds 900000–900050); kept clear of training seeds so evaluation never reuses a trained-on seed. |
| `save_trajectory_seeds` | `3` | Persist full within-segment trajectories for the first N seeds of each function (for figures). |
| `compare_against` | `[BASE001_de_plain, BASE002_fixed_schedule, BASE003_random]` | The frozen baselines the agent is compared against. Each must be produced (once, by hand) with `scripts/run_baseline.py` against this same config; the comparison is refused if function/dim/budget/seeds don't match. |

### `slurm` — cluster resources (read by `run_experiment.sh`)

| key | current | meaning |
|-----|---------|---------|
| `time` | `"12:00:00"` | Walltime request. The job checkpoints and exits cleanly at ~90% of this (a "timeout" status), so it is never killed mid-episode. |
| `mem` | `"16G"` | Memory request. |
| `cpus` | `4` | CPU cores. |
| `gpus` | `0` | GPUs. CPU-only here: the DE core synchronizes every generation, so a GPU is slower for this workload. |
| `account` | `def-bolufe` | The Nibi allocation to charge. |

---

### The three baselines (context for the comparison)

The agent is judged against three frozen references, produced once by
`scripts/run_baseline.py` on the *same* seeds/function/dim/budget:

- **BASE001 — plain DE:** one DE run over the whole 300k budget, no restarts. Asks
  *does the multi-restart idea help at all?*
- **BASE002 — fixed schedule:** the full restart machinery, but a **constant**
  action every step (rand/1/bin, budget 10%, box index 2). Asks *does restarting
  help, independent of whether the agent learned anything?*
- **BASE003 — random:** the full machinery with a **random** action every step —
  the floor the agent must clear.

`evaluate.py` writes `eval_results.csv` (agent and every baseline side by side,
per seed, with the budget each consumed) and `comparison.csv` (paired Wilcoxon
signed-rank test, where a positive `pct_diff` means the agent has the lower
error).
