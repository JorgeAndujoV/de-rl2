#!/bin/bash
# Generic experiment runner for Nibi. One script, every experiment.
#
#   ./run_experiment.sh <exp_id> <smoke|train|eval|train_eval> \
#       [--function N] [--force|--resume] [extra args...]
#
# Reads SLURM resources from experiments/<exp_id>/config.yaml, submits the job,
# and writes output into experiments/<exp_id>/<kind>/ (one job per experiment
# mode: a fixed path, no job_<id> layer). A failed run is deleted and re-run; a
# changed parameter becomes a new EXP00N — outputs are never accumulated side by
# side.
#
# Per-function experiments (EXP003+): a config that declares training.periodic_eval
# is trained ONE FUNCTION PER JOB, so functions run in parallel on Nibi. Output
# goes to experiments/<exp_id>/f<N>/ (smoke -> experiments/<exp_id>/smokes/f<N>/).
#   --function N   submit one job for function N
#   (no --function) submit one job per function in config.benchmark.functions
# Such a config also does its evaluation IN-PROCESS at each checkpoint, so the
# separate post-training eval step is skipped. Each per-function job first
# generates its own function's baselines (evaluation.compare_against) into
# baselines/<exp_id>/f<N>/ before training, so baselines are never shared across
# experiments and never regenerated per checkpoint.
#
# Because the output path is fixed, a careless resubmit could destroy a completed
# run, so a non-empty target is refused unless you say what to do:
#   (default)   refuse and exit non-zero
#   --force     move the existing dir to <name>.bak.<timestamp>/ and start fresh
#   --resume    continue from the existing dir's checkpoints
#
# Examples:
#   ./run_experiment.sh EXP002_dqn_restart smoke
#   ./run_experiment.sh EXP002_dqn_restart train_eval --force
#   ./run_experiment.sh EXP003_dqn_restart train_eval --function 11
#   ./run_experiment.sh EXP003_dqn_restart train_eval          # all functions
#   ./run_experiment.sh EXP002_dqn_restart eval --train-dir <path>

set -euo pipefail

# ------------------------------------------------------------ cluster setup
# Resources (account, time, mem, cpus, gpus) are NOT hard-coded here: they are
# read from the experiment's config.yaml slurm block below, so run_metadata.json
# (resolved_config) is the complete record of what a job requested.
PYTHON_MODULE="python/3.11"
VENV="$HOME/de-rl2-env"
REPO="$HOME/de-rl2"

# ---------------------------------------------------------------- arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <exp_id> <smoke|train|eval|train_eval> [--function N] [--force|--resume] [extra args...]"
    echo ""
    echo "Available experiments:"
    ls -1 "$REPO/experiments" | grep -v '^_' | grep -v '\.csv$' | sed 's/^/  /'
    exit 1
fi

EXP_ID="$1"
KIND="$2"
shift 2

# Pull run_experiment.sh's own flags out of the remaining args; the rest is
# forwarded verbatim to the python entry points.
FORCE=0
RESUME=0
FUNCTION=""
PASS_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --force)    FORCE=1 ;;
        --resume)   RESUME=1 ;;
        --function) FUNCTION="$2"; shift ;;
        --function=*) FUNCTION="${1#*=}" ;;
        *)          PASS_ARGS+=("$1") ;;
    esac
    shift
done
EXTRA_ARGS="${PASS_ARGS[*]:-}"

if [ "$FORCE" = "1" ] && [ "$RESUME" = "1" ]; then
    echo "ERROR: --force and --resume are mutually exclusive."
    exit 1
fi

CONFIG="$REPO/experiments/$EXP_ID/config.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: no config at $CONFIG"
    exit 1
fi

case "$KIND" in
    smoke|train|eval|train_eval) ;;
    *) echo "ERROR: kind must be smoke, train, eval, or train_eval"; exit 1 ;;
esac

# ------------------------------------------- read resources from the config
# Uses the config loader rather than grep/awk, so the YAML has one parser.
module load "$PYTHON_MODULE" > /dev/null 2>&1 || true
source "$VENV/bin/activate"
cd "$REPO"

ACCOUNT=$(python -m derl2.config --config "$CONFIG" slurm.account)
CPUS=$(python -m derl2.config --config "$CONFIG" slurm.cpus)
MEM=$(python -m derl2.config --config "$CONFIG" slurm.mem)
GPUS=$(python -m derl2.config --config "$CONFIG" slurm.gpus)
TIME=$(python -m derl2.config --config "$CONFIG" slurm.time)

# Per-function experiments declare training.periodic_eval; that flag also means
# evaluation happens in-process (train.py), so the separate eval step is skipped.
HAS_PERIODIC=$(python -c "from derl2.config import Config; print(Config.from_file('$CONFIG').get('training.periodic_eval.every', default=''))" 2>/dev/null || echo "")

# Baseline method names (evaluation.compare_against), space-joined by python so
# the shell never has to parse a YAML/py list. Each per-function job generates
# these for its function before training (see the baseline block in the heredoc).
BASELINES=$(python -c "from derl2.config import Config; print(' '.join(Config.from_file('$CONFIG').get('evaluation.compare_against')))" 2>/dev/null || echo "")

# Walltime safety stop: checkpoint and exit at 90% of the requested time, rather
# than being killed mid-episode by SLURM. Supports D-HH:MM:SS and HH:MM:SS.
HOURS=$(echo "$TIME" | awk -F'[-:]' '{ if (NF==4) print $1*24 + $2 + $3/60 + $4/3600; else print $1 + $2/60 + $3/3600 }')
MAX_HOURS=$(echo "$HOURS" | awk '{printf "%.2f", $1 * 0.9}')

# Full commit + dirty flag, both folded into run_metadata.json at job start.
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo "unknown")
DIRTY_FLAG=""
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
    DIRTY_FLAG="--git-dirty"
fi

SMOKE_FLAG=""
if [ "$KIND" = "smoke" ]; then
    SMOKE_FLAG="--smoke"
fi

# Only request GPUs when the config asks for a nonzero count; a CPU-only job
# must not emit "--gpus-per-node=0" (some SLURM setups reject it).
GPU_DIRECTIVE=""
if [ -n "${GPUS}" ] && [ "${GPUS}" != "0" ]; then
    GPU_DIRECTIVE="#SBATCH --gpus-per-node=${GPUS}"
fi

mkdir -p "$REPO/slurm_logs"

# --------------------------------------------------------- submit one job
# submit_one <function-id-or-empty>: build the output path for this function,
# run the overwrite guard, and submit the sbatch job. Called once for a single
# run (legacy or --function N) or once per function for a fan-out.
submit_one() {
    local FUNCTION="$1"
    local SUBPATH FUNC_SET FUNC_EVAL_ARG FINAL_DIR JOB_SUFFIX

    if [ -n "$FUNCTION" ]; then
        if [ "$KIND" = "smoke" ]; then
            SUBPATH="smokes/f${FUNCTION}"
        else
            SUBPATH="f${FUNCTION}"
        fi
        # Single-quoted so the runtime shell can't glob the [N] before argparse
        # sees it.
        FUNC_SET="--set 'benchmark.functions=[${FUNCTION}]'"
        FUNC_EVAL_ARG="--function ${FUNCTION}"
        JOB_SUFFIX="_f${FUNCTION}"
    else
        SUBPATH="$KIND"
        FUNC_SET=""
        FUNC_EVAL_ARG=""
        JOB_SUFFIX=""
    fi
    FINAL_DIR="$REPO/experiments/$EXP_ID/$SUBPATH"

    # --------------------------------------------------------- overwrite guard
    # Refuse to clobber a completed run unless the user chose --force or --resume.
    if [ "$KIND" != "eval" ] && [ -d "$FINAL_DIR" ] && \
       [ -n "$(ls -A "$FINAL_DIR" 2>/dev/null)" ]; then
        if [ "$FORCE" = "1" ]; then
            local BAK="${FINAL_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
            mv "$FINAL_DIR" "$BAK"
            echo "note: moved existing results to $BAK"
        elif [ "$RESUME" = "1" ]; then
            echo "note: resuming into existing $FINAL_DIR"
        else
            echo "ERROR: $FINAL_DIR already exists and is non-empty."
            echo "Refusing to overwrite a completed run. Choose one:"
            echo "  --force    move it aside to <name>.bak.<timestamp>/ and start fresh"
            echo "  --resume   continue from its checkpoints"
            exit 1
        fi
    fi

    echo "=============================================="
    echo " experiment : $EXP_ID"
    echo " kind       : $KIND${FUNCTION:+  (function f${FUNCTION})}"
    echo " commit     : ${COMMIT:0:12}${DIRTY_FLAG:+ (dirty)}"
    echo " output     : $FINAL_DIR"
    echo " resources  : ${CPUS} cpus, ${GPUS} gpus, ${MEM}, ${TIME} (stop at ${MAX_HOURS}h)"
    echo " account    : $ACCOUNT"
    echo "=============================================="

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_ID}_${KIND}${JOB_SUFFIX}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS}
${GPU_DIRECTIVE}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${REPO}/slurm_logs/%x_%j.out
#SBATCH --error=${REPO}/slurm_logs/%x_%j.err

set -euo pipefail

module purge
module load ${PYTHON_MODULE}
source ${VENV}/bin/activate
cd ${REPO}

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

echo "==== JOB CONTEXT ===="
date; hostname
echo "job      \${SLURM_JOB_ID}"
echo "commit   ${COMMIT}"
echo "kind     ${KIND}${FUNCTION:+  f${FUNCTION}}"
python --version
echo "====================="

# ------------------------------------------------------------------ eval-only
# Re-evaluate an existing run the user points at with --train-dir. No fresh
# output dir, no guard, no metadata (the target already has its own).
if [ "${KIND}" = "eval" ]; then
    echo "==== EVALUATION ===="
    python -m derl2.evaluation.evaluate ${EXTRA_ARGS}
    echo "Job finished."
    exit 0
fi

# ---------------------------------------------------- smoke / train / train_eval
# Jobs run on \$SCRATCH (fast); results are copied back to the repo at the end.
# The scratch dir keeps the job id so concurrent scratch runs never collide; the
# persisted repo dir (FINAL_DIR) does not.
WORK_DIR="\${SCRATCH}/derl2/${EXP_ID}/${SUBPATH}/job_\${SLURM_JOB_ID}"
FINAL_DIR="${FINAL_DIR}"
mkdir -p "\${WORK_DIR}"
echo "work     \${WORK_DIR}"
echo "final    \${FINAL_DIR}"

# So train.py's periodic evaluation can copy each checkpoint's results to the
# persisted repo dir as it is produced (progressive results, not only at exit).
export DERL2_FINAL_DIR="\${FINAL_DIR}"

# STATUS is read by the persist trap. It stays "failed" unless the run reaches a
# clean completion or a clean walltime stop, so a crash is labelled correctly.
STATUS="failed"

# Copy results back even if the job fails or times out; rewrite run_metadata.json
# with the final status; on a SUCCESSFUL run prune periodic checkpoints down to
# final/ + best/ + chkp*/ (a timeout/failure keeps the intermediates so it can
# resume). The replay buffer (replay_buffer.npz) is kept in all cases.
persist() {
    echo "==== PERSIST (status \${STATUS}) ===="
    python -m derl2.run_metadata finish --out-dir "\${WORK_DIR}" \\
        --status "\${STATUS}" || true
    if [ "\${STATUS}" = "completed" ] && [ -d "\${WORK_DIR}/checkpoints" ]; then
        echo "pruning intermediate resume checkpoints (keeping final/, best/, chkp*/, replay_buffer.npz)"
        find "\${WORK_DIR}/checkpoints" -maxdepth 1 -type f -name 'ckpt-*' -delete
        rm -f "\${WORK_DIR}/checkpoints/checkpoint"
    fi
    mkdir -p "\${FINAL_DIR}"
    cp -r "\${WORK_DIR}"/. "\${FINAL_DIR}"/ 2>/dev/null || true
    echo "copied to \${FINAL_DIR}"
    ls -la "\${FINAL_DIR}"
    date
}
trap persist EXIT

# --resume: seed the fresh scratch dir from the persisted partial run so
# training restores from its checkpoints (and replay buffer).
if [ "${RESUME}" = "1" ] && [ -d "\${FINAL_DIR}" ]; then
    echo "==== RESUME (seeding work dir from \${FINAL_DIR}) ===="
    cp -r "\${FINAL_DIR}"/. "\${WORK_DIR}"/ 2>/dev/null || true
fi

# run_metadata.json written BEFORE any training, so even an import-time crash
# leaves a record the persist trap flips from "running" to "failed". The
# per-function --set makes resolved_config record functions=[N] for this job.
python -m derl2.run_metadata start \\
    --out-dir "\${WORK_DIR}" --config "${CONFIG}" --kind "${KIND}" ${SMOKE_FLAG} \\
    --slurm-job-id "\${SLURM_JOB_ID}" --commit "${COMMIT}" ${DIRTY_FLAG} \\
    ${FUNC_SET} ${EXTRA_ARGS}

# ---------------------------------------------------- baselines (job-generated)
# Each per-function job produces THIS experiment's baselines for its function
# ONCE, before training, into baselines/<exp>/f<N>/<method>/. repo_root is
# \$HOME/de-rl2 (editable install), so they persist there and every one of the 10
# checkpoint evaluations reuses the same frozen numbers — baselines are never
# regenerated per checkpoint. --skip-existing makes a resumed job skip baselines
# it already produced. A baseline is valid only for the exact config that made
# it, so each experiment generates its own instead of sharing another's; a
# failure here (before training) correctly fails the job via the persist trap.
if [ -n "${FUNCTION}" ]; then
    echo "==== BASELINES (f${FUNCTION}, generate-once, reused by all checkpoints) ===="
    for BL in ${BASELINES}; do
        python -m scripts.run_baseline --baseline \$BL \\
            --config "${CONFIG}" --function ${FUNCTION} ${SMOKE_FLAG} --skip-existing
    done
fi

echo "==== TRAINING ===="
set +e
python -m derl2.training.train \\
    --config "${CONFIG}" ${SMOKE_FLAG} --kind ${KIND} \\
    --out-dir "\${WORK_DIR}" --max-hours ${MAX_HOURS} ${FUNC_SET} ${EXTRA_ARGS}
RC=\$?
set -e
if [ \$RC -eq 42 ]; then
    STATUS="timeout"
    echo "walltime stop; re-run with --resume to continue."
    exit 0
elif [ \$RC -ne 0 ]; then
    echo "training failed (rc \$RC)."
    exit \$RC
fi

if [ "${KIND}" = "train_eval" ] || [ "${KIND}" = "smoke" ]; then
    if [ -n "${HAS_PERIODIC}" ]; then
        echo "==== EVALUATION (in-process, per checkpoint) ===="
        echo "handled by train.py's periodic_eval; no separate eval step."
    else
        echo "==== EVALUATION ===="
        set +e
        python -m derl2.evaluation.evaluate --train-dir "\${WORK_DIR}" ${FUNC_EVAL_ARG} ${EXTRA_ARGS}
        RC=\$?
        set -e
        if [ \$RC -ne 0 ]; then
            echo "evaluation failed (rc \$RC)."
            exit \$RC
        fi
    fi
fi

STATUS="completed"
echo "Job finished."
EOF
}

# ------------------------------------------------------------ dispatch
if [ "$KIND" = "eval" ] || [ -n "$FUNCTION" ]; then
    # eval-only, or an explicit single function.
    submit_one "$FUNCTION"
elif [ -n "$HAS_PERIODIC" ]; then
    # Per-function experiment, no --function: fan out one job per function.
    FUNCS=$(python -m derl2.config --config "$CONFIG" benchmark.functions | tr -d '[],')
    echo "per-function experiment: submitting one job per function: $FUNCS"
    for f in $FUNCS; do
        submit_one "$f"
    done
else
    # Legacy single-job experiment (EXP001/EXP002 layout).
    submit_one ""
fi

echo ""
echo "Submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -u \$USER --format=JobID,JobName%30,State,Elapsed,MaxRSS -X"
echo "  tail -f $REPO/slurm_logs/${EXP_ID}_${KIND}_<jobid>.out"
