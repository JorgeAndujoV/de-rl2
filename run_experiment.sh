#!/bin/bash
# Generic experiment runner for Nibi. One script, every experiment.
#
#   ./run_experiment.sh <exp_id> <smoke|train|eval|train_eval> [extra args...]
#
# Reads SLURM resources from experiments/<exp_id>/config.yaml, submits the
# job, and writes output into experiments/<exp_id>/<kind>/job_<slurm_id>/.
#
# Examples:
#   ./run_experiment.sh EXP001_dqn-baseline smoke
#   ./run_experiment.sh EXP001_dqn-baseline train_eval
#   ./run_experiment.sh EXP001_dqn-baseline eval --train-dir <path>

set -euo pipefail

# ------------------------------------------------------------ cluster setup
# Resources (account, time, mem, cpus, gpus) are NOT hard-coded here: they are
# read from the experiment's config.yaml slurm block below, so config_used.yaml
# is the complete record of what a job requested.
PYTHON_MODULE="python/3.11"
VENV="$HOME/de-rl2-env"
REPO="$HOME/de-rl2"

# ---------------------------------------------------------------- arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <exp_id> <smoke|train|eval|train_eval> [extra args...]"
    echo ""
    echo "Available experiments:"
    ls -1 "$REPO/experiments" | grep -v '^_' | grep -v '\.csv$' | sed 's/^/  /'
    exit 1
fi

EXP_ID="$1"
KIND="$2"
shift 2
EXTRA_ARGS="$@"

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

COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Only request GPUs when the config asks for a nonzero count; a CPU-only job
# must not emit "--gpus-per-node=0" (some SLURM setups reject it).
GPU_DIRECTIVE=""
if [ -n "${GPUS}" ] && [ "${GPUS}" != "0" ]; then
    GPU_DIRECTIVE="#SBATCH --gpus-per-node=${GPUS}"
fi

echo "=============================================="
echo " experiment : $EXP_ID"
echo " kind       : $KIND"
echo " commit     : $COMMIT"
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

# Jobs run on \$SCRATCH (fast); results are copied back to the repo at the end.
WORK_DIR="\${SCRATCH}/derl2/${EXP_ID}/${KIND}/job_\${SLURM_JOB_ID}"
FINAL_DIR="${REPO}/experiments/${EXP_ID}/${KIND}/job_\${SLURM_JOB_ID}"
mkdir -p "\${WORK_DIR}"

echo "==== JOB CONTEXT ===="
date; hostname
echo "job      \${SLURM_JOB_ID}"
echo "commit   ${COMMIT}"
echo "work     \${WORK_DIR}"
echo "final    \${FINAL_DIR}"
python --version
echo "====================="

# Copy results back even if the job fails or times out.
persist() {
    echo "==== PERSIST ===="
    mkdir -p "\${FINAL_DIR}"
    cp -r "\${WORK_DIR}"/. "\${FINAL_DIR}"/ 2>/dev/null || true
    echo "copied to \${FINAL_DIR}"
    ls -la "\${FINAL_DIR}"
    date
}
trap persist EXIT

SMOKE_FLAG=""
if [ "${KIND}" = "smoke" ]; then
    SMOKE_FLAG="--smoke"
fi

if [ "${KIND}" != "eval" ]; then
    echo "==== TRAINING ===="
    python -m derl2.training.train \\
        --config "${CONFIG}" \\
        \${SMOKE_FLAG} \\
        --kind ${KIND} \\
        --out-dir "\${WORK_DIR}" \\
        --max-hours ${MAX_HOURS} \\
        ${EXTRA_ARGS}
fi

if [ "${KIND}" = "eval" ] || [ "${KIND}" = "train_eval" ] || [ "${KIND}" = "smoke" ]; then
    echo "==== EVALUATION ===="
    python -m derl2.evaluation.evaluate \\
        --train-dir "\${WORK_DIR}" \\
        ${EXTRA_ARGS}
fi

echo "Job finished."
EOF

echo ""
echo "Submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -u \$USER --format=JobID,JobName%30,State,Elapsed,MaxRSS -X"
echo "  tail -f $REPO/slurm_logs/${EXP_ID}_${KIND}_<jobid>.out"
