# de-rl2

An RL agent controls a **multi-restart** Differential Evolution optimizer on the
IEEE CEC'13 benchmark suite. One episode is one complete optimization of one
benchmark function under a fixed budget of function evaluations (FEs), carried
out as a *sequence of independent DE runs* rather than one continuous run.

- **Warm-up segment** (no agent decision): DE/rand/1/bin, NP = 50, F = 0.5,
  CR = 0.9, uniform initialization over the full domain, consuming 5% of the
  budget.
- **Each step:** the environment builds an observation from the segment that
  just finished; the agent emits an action decoding to
  `{strategy, F, CR, budget_frac, sampling_box}`; the `sampling_box` transforms
  the bounding box of the previous segment's final population; a fresh
  population of NP = 50 is sampled uniformly inside that box; DE runs for
  `budget_frac × total_budget` FEs. This repeats until the budget is exhausted.

Episode length is variable: 7 decisions (all 15%) to 19 decisions (all 5%).
There is no elitism — the incumbent is never injected into a new population —
but the environment tracks the best-ever solution throughout for reward and for
the reported episode result.

## Layout

```
src/derl2/
  config.py            explicit, self-contained, frozen-after-running config loader
  benchmarks/          CEC'13 suite (ported unchanged from de-rl)
  optimizers/          validated DE core, exposed as one function: run_segment
  environments/        episode loop + observation / action-space / reward registries
  agents/              DQN with SMDP (gamma ** tau) discounting
  replay/              replay buffer (transitions carry tau)
  training/            training entry point
  evaluation/          independent evaluation against frozen baselines
experiments/           one folder per experiment; REGISTRY.csv indexes them
baselines/             frozen baseline artifacts, produced deliberately
data/cec13/input_data/ CEC'13 shift vectors and rotation matrices
```

## Design principles

Readability over cleverness; string-selected registries (a new variant is one
module plus one dictionary line); configs are explicit and frozen after running;
one-way dependencies (optimizer → environment → agent); the validated DE core is
authoritative and its generation-level behaviour does not change; training and
evaluation are strictly separate.

## Running

```
python -m derl2.training.train    --config experiments/EXP001_slug/config.yaml
python -m derl2.evaluation.evaluate --train-dir experiments/EXP001_slug/train/job_<id>
```

On the cluster, `run_experiment.sh <exp_id> <smoke|train|eval|train_eval>` submits
the corresponding SLURM job.

The CEC'13 data files are read from `data/cec13/input_data/` by default; override
the location with the `CEC13_DATA` environment variable.
