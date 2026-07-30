#!/bin/bash
# Generic experiment runner for Nibi. One script, every experiment.
#
#   ./run_experiment.sh <exp_id> <smoke|train|eval|train_eval> [--force|--resume] [extra args...]
#
# Reads SLURM resources from experiments/<exp_id>/config.yaml, submits the job,
# and writes output into experiments/<exp_id>/<kind>/ (one job per experiment
# mode: a fixed path, no job_<id> layer). A failed run is deleted and re-run; a
# changed parameter becomes a new EXP00N — outputs are never accumulated side by
# side.
#
# Because the output path is now fixed, a careless resubmit could destroy a
# completed run, so a non-empty target is refused unless you say what to do:
#   (default)   refuse and exit non-zero
#   --force     move the existing dir to <kind>.bak.<timestamp>/ and start fresh
#   --resume    continue from the existing dir's checkpoints
#
# Examples:
#   ./run_experiment.sh EXP002_dqn_restart smoke
#   ./run_experiment.sh EXP002_dqn_restart train_eval
#   ./run_experiment.sh EXP002_dqn_restart train_eval --force
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
    echo "Usage: $0 <exp_id> <smoke|train|eval|train_eval> [--force|--resume] [extra args...]"
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
PASS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --force)  FORCE=1 ;;
        --resume) RESUME=1 ;;
        *)        PASS_ARGS+=("$arg") ;;
    esac
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

# Walltime safety stop: checkpoint and exit at 90% of the requested time,
# rather than being killed mid-episode by SLURM.
HOURS=$(echo "$TIME" | awk -F: '{print $1 + $2/60 + $3/3600}')
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

# Fixed output path (no job_<id> layer). eval re-evaluates an existing run the
# user points at with --train-dir, so it has no fresh output dir of its own.
FINAL_DIR="$REPO/experiments/$EXP_ID/$KIND"

# --------------------------------------------------------- overwrite guard
# Refuse to clobber a completed run unless the user chose --force or --resume.
if [ "$KIND" != "eval" ] && [ -d "$FINAL_DIR" ] && \
   [ -n "$(ls -A "$FINAL_DIR" 2>/dev/null)" ]; then
    if [ "$FORCE" = "1" ]; then
        BAK="${FINAL_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
        mv "$FINAL_DIR" "$BAK"
        echo "note: moved existing results to $BAK"
    elif [ "$RESUME" = "1" ]; then
        echo "note: resuming into existing $FINAL_DIR"
    else
        echo "ERROR: $FINAL_DIR already exists and is non-empty."
        echo "Refusing to overwrite a completed run. Choose one:"
        echo "  --force    move it aside to ${KIND}.bak.<timestamp>/ and start fresh"
        echo "  --resume   continue from its checkpoints"
        exit 1
    fi
fi

# Only request GPUs when the config asks for a nonzero count; a CPU-only job
# must not emit "--gpus-per-node=0" (some SLURM setups reject it).
GPU_DIRECTIVE=""
if [ -n "${GPUS}" ] && [ "${GPUS}" != "0" ]; then
    GPU_DIRECTIVE="#SBATCH --gpus-per-node=${GPUS}"
fi

echo "=============================================="
echo " experiment : $EXP_ID"
echo " kind       : $KIND"
echo " commit     : ${COMMIT:0:12}${DIRTY_FLAG:+ (dirty)}"
echo " output     : $FINAL_DIR"
echo " resources  : ${CPUS} cpus, ${GPUS} gpus, ${MEM}, ${TIME} (stop at ${MAX_HOURS}h)"
echo " account    : $ACCOUNT"
echo "=============================================="

mkdir -p "$REPO/slurm_logs"

# ------------------------------------------------------------ submit to SLURM
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_ID}_${KIND}
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
echo "kind     ${KIND}"
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
WORK_DIR="\${SCRATCH}/derl2/${EXP_ID}/${KIND}/job_\${SLURM_JOB_ID}"
FINAL_DIR="${FINAL_DIR}"
mkdir -p "\${WORK_DIR}"
echo "work     \${WORK_DIR}"
echo "final    \${FINAL_DIR}"

# STATUS is read by the persist trap. It stays "failed" unless the run reaches a
# clean completion or a clean walltime stop, so a crash is labelled correctly.
STATUS="failed"

# Copy results back even if the job fails or times out; rewrite run_metadata.json
# with the final status; on a SUCCESSFUL run prune periodic checkpoints down to
# final/ + best/ (a timeout/failure keeps the intermediates so it can resume).
persist() {
    echo "==== PERSIST (status \${STATUS}) ===="
    python -m derl2.run_metadata finish --out-dir "\${WORK_DIR}" \\
        --status "\${STATUS}" || true
    if [ "\${STATUS}" = "completed" ] && [ -d "\${WORK_DIR}/checkpoints" ]; then
        echo "pruning intermediate checkpoints (keeping final/ + best/)"
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
# training restores from its checkpoints.
if [ "${RESUME}" = "1" ] && [ -d "\${FINAL_DIR}" ]; then
    echo "==== RESUME (seeding work dir from \${FINAL_DIR}) ===="
    cp -r "\${FINAL_DIR}"/. "\${WORK_DIR}"/ 2>/dev/null || true
fi

# run_metadata.json written BEFORE any training, so even an import-time crash
# leaves a record the persist trap flips from "running" to "failed".
python -m derl2.run_metadata start \\
    --out-dir "\${WORK_DIR}" --config "${CONFIG}" --kind "${KIND}" ${SMOKE_FLAG} \\
    --slurm-job-id "\${SLURM_JOB_ID}" --commit "${COMMIT}" ${DIRTY_FLAG} \\
    ${EXTRA_ARGS}

echo "==== TRAINING ===="
set +e
python -m derl2.training.train \\
    --config "${CONFIG}" ${SMOKE_FLAG} --kind ${KIND} \\
    --out-dir "\${WORK_DIR}" --max-hours ${MAX_HOURS} ${EXTRA_ARGS}
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
    echo "==== EVALUATION ===="
    set +e
    python -m derl2.evaluation.evaluate --train-dir "\${WORK_DIR}" ${EXTRA_ARGS}
    RC=\$?
    set -e
    if [ \$RC -ne 0 ]; then
        echo "evaluation failed (rc \$RC)."
        exit \$RC
    fi
fi

STATUS="completed"
echo "Job finished."
EOF

echo ""
echo "Submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -u \$USER --format=JobID,JobName%30,State,Elapsed,MaxRSS -X"
echo "  tail -f $REPO/slurm_logs/${EXP_ID}_${KIND}_<jobid>.out"
