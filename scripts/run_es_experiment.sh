#!/bin/bash
# Sep-CMA-ES experiment runner for Nibi (EXP026) -- additive sibling of
# run_experiment.sh. That launcher is hard-wired to derl2.training.train +
# derl2.evaluation.evaluate (PPO/DQN); this one drives the ES stack
# (derl2.training.train_es + derl2.evaluation.evaluate_es) and is kept SEPARATE
# so EXP016/020-025 keep using run_experiment.sh unchanged.
#
#   ./scripts/run_es_experiment.sh <exp_id> <smoke|validation|train_eval|eval> \
#       [--function N] [--force|--resume] [--cpus N] [--time D-HH:MM:SS] [extra...]
#
# Kinds:
#   validation  short throughput-sizing job (scripts.es_throughput) on ONE
#               function; measures ep/s at a worker sweep and projects the full
#               run wall-clock. Runs on --cpus cores (default: config slurm.cpus,
#               = 64). Use it BEFORE a full run to confirm the core count.
#   train_eval  full Sep-CMA-ES training (es.num_generations) then evaluate_es,
#               ONE function per job (fan-out over benchmark.functions, or
#               --function N). train_es gets --workers = cpus.
#   smoke       tiny scaled-down train_eval (crash check only).
#   eval        re-evaluate an existing run: pass --train-dir DIR [extra...].
#
# Cores: default is config slurm.cpus (64). The documented path to 180 is
# `--cpus 180` (needs a whole Nibi node; do NOT default to it). Workers beyond
# population_size (64) give no speedup -- a generation has only `pop` episodes --
# so 64 cores already saturate the default population; 180 helps only if you also
# raise es.population_size.

set -euo pipefail

# ------------------------------------------------------------ cluster setup
PYTHON_MODULE="python/3.11"
VENV="$HOME/de-rl2-env"
REPO="$HOME/de-rl2"

# ---------------------------------------------------------------- arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <exp_id> <smoke|validation|train_eval|eval> [--function N] [--force|--resume] [--cpus N] [--time T] [extra args...]"
    echo ""
    echo "Available experiments:"
    ls -1 "$REPO/experiments" | grep -v '^_' | grep -v '\.csv$' | sed 's/^/  /'
    exit 1
fi

EXP_ID="$1"
KIND="$2"
shift 2

FORCE=0
RESUME=0
FUNCTION=""
CPUS_OVERRIDE=""
TIME_OVERRIDE=""
PASS_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --force)      FORCE=1 ;;
        --resume)     RESUME=1 ;;
        --function)   FUNCTION="$2"; shift ;;
        --function=*) FUNCTION="${1#*=}" ;;
        --cpus)       CPUS_OVERRIDE="$2"; shift ;;
        --cpus=*)     CPUS_OVERRIDE="${1#*=}" ;;
        --time)       TIME_OVERRIDE="$2"; shift ;;
        --time=*)     TIME_OVERRIDE="${1#*=}" ;;
        *)            PASS_ARGS+=("$1") ;;
    esac
    shift
done
EXTRA_ARGS="${PASS_ARGS[*]:-}"

if [ "$FORCE" = "1" ] && [ "$RESUME" = "1" ]; then
    echo "ERROR: --force and --resume are mutually exclusive."; exit 1
fi

CONFIG="$REPO/experiments/$EXP_ID/config.yaml"
if [ ! -f "$CONFIG" ]; then echo "ERROR: no config at $CONFIG"; exit 1; fi

case "$KIND" in
    smoke|validation|train_eval|eval) ;;
    *) echo "ERROR: kind must be smoke, validation, train_eval, or eval"; exit 1 ;;
esac

# ------------------------------------------- read resources from the config
module load "$PYTHON_MODULE" > /dev/null 2>&1 || true
source "$VENV/bin/activate"
cd "$REPO"

ACCOUNT=$(python -m derl2.config --config "$CONFIG" slurm.account)
CPUS=${CPUS_OVERRIDE:-$(python -m derl2.config --config "$CONFIG" slurm.cpus)}
MEM=$(python -m derl2.config --config "$CONFIG" slurm.mem)
GPUS=$(python -m derl2.config --config "$CONFIG" slurm.gpus)
TIME=${TIME_OVERRIDE:-$(python -m derl2.config --config "$CONFIG" slurm.time)}
# Validation is short by construction; unless overridden, cap it to 2h.
if [ "$KIND" = "validation" ] && [ -z "$TIME_OVERRIDE" ]; then
    TIME="0-02:00:00"
fi

BASELINES=$(python -c "from derl2.config import Config; print(' '.join(Config.from_file('$CONFIG').get('evaluation.compare_against')))" 2>/dev/null || echo "")

HOURS=$(echo "$TIME" | awk -F'[-:]' '{ if (NF==4) print $1*24 + $2 + $3/60 + $4/3600; else print $1 + $2/60 + $3/3600 }')
MAX_HOURS=$(echo "$HOURS" | awk '{printf "%.2f", $1 * 0.9}')

COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo "unknown")
DIRTY_FLAG=""
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
    DIRTY_FLAG="--git-dirty"
fi

SMOKE_FLAG=""
if [ "$KIND" = "smoke" ]; then SMOKE_FLAG="--smoke"; fi

GPU_DIRECTIVE=""
if [ -n "${GPUS}" ] && [ "${GPUS}" != "0" ]; then
    GPU_DIRECTIVE="#SBATCH --gpus-per-node=${GPUS}"
fi

mkdir -p "$REPO/slurm_logs"

# --------------------------------------------------------- submit one job
submit_one() {
    local FUNCTION="$1"
    local SUBPATH FUNC_SET FINAL_DIR JOB_SUFFIX

    if [ -n "$FUNCTION" ]; then
        if [ "$KIND" = "smoke" ]; then SUBPATH="smokes/f${FUNCTION}"
        else SUBPATH="${KIND}/f${FUNCTION}"; fi
        FUNC_SET="--set 'benchmark.functions=[${FUNCTION}]'"
        JOB_SUFFIX="_f${FUNCTION}"
    else
        SUBPATH="$KIND"; FUNC_SET=""; JOB_SUFFIX=""
    fi
    FINAL_DIR="$REPO/experiments/$EXP_ID/$SUBPATH"

    # overwrite guard (same contract as run_experiment.sh)
    if [ "$KIND" != "eval" ] && [ -d "$FINAL_DIR" ] && \
       [ -n "$(ls -A "$FINAL_DIR" 2>/dev/null)" ]; then
        if [ "$FORCE" = "1" ]; then
            local BAK="${FINAL_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
            mv "$FINAL_DIR" "$BAK"; echo "note: moved existing results to $BAK"
        elif [ "$RESUME" = "1" ]; then
            echo "note: resuming into existing $FINAL_DIR"
        else
            echo "ERROR: $FINAL_DIR already exists and is non-empty."
            echo "  --force    move it aside to <name>.bak.<timestamp>/ and start fresh"
            echo "  --resume   continue from its checkpoints"
            exit 1
        fi
    fi

    echo "=============================================="
    echo " experiment : $EXP_ID (Sep-CMA-ES)"
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
echo "cpus     ${CPUS}"
python --version
echo "====================="

# ------------------------------------------------------------------ eval-only
if [ "${KIND}" = "eval" ]; then
    echo "==== EVALUATION (evaluate_es) ===="
    python -m derl2.evaluation.evaluate_es ${EXTRA_ARGS}
    echo "Job finished."
    exit 0
fi

# ------------------------------------------------------------------ validation
# Throughput sizing only: no training, no persisted checkpoints. Writes the
# ep/s table to the experiment's validation/ dir for the record.
if [ "${KIND}" = "validation" ]; then
    mkdir -p "${FINAL_DIR}"
    echo "==== THROUGHPUT (es_throughput) ===="
    python -m scripts.es_throughput --config "${CONFIG}" --function "${FUNCTION}" \\
        --workers "8,16,32,${CPUS}" --gens 3 \\
        --out "${FINAL_DIR}/throughput.csv" ${EXTRA_ARGS}
    echo "Job finished."
    exit 0
fi

# ---------------------------------------------------- smoke / train_eval
WORK_DIR="\${SCRATCH}/derl2/${EXP_ID}/${SUBPATH}/job_\${SLURM_JOB_ID}"
FINAL_DIR="${FINAL_DIR}"
mkdir -p "\${WORK_DIR}"
echo "work     \${WORK_DIR}"
echo "final    \${FINAL_DIR}"

STATUS="failed"
persist() {
    echo "==== PERSIST (status \${STATUS}) ===="
    mkdir -p "\${FINAL_DIR}"
    cp -r "\${WORK_DIR}"/. "\${FINAL_DIR}"/ 2>/dev/null || true
    echo "copied to \${FINAL_DIR}"
    ls -la "\${FINAL_DIR}"
    date
}
trap persist EXIT

# --resume: seed the fresh scratch dir from the persisted partial run so
# train_es restores from checkpoints/es_state.pkl.
if [ "${RESUME}" = "1" ] && [ -d "\${FINAL_DIR}" ]; then
    echo "==== RESUME (seeding work dir from \${FINAL_DIR}) ===="
    cp -r "\${FINAL_DIR}"/. "\${WORK_DIR}"/ 2>/dev/null || true
fi

# ---------------------------------------------------- baselines (job-generated)
if [ -n "${FUNCTION}" ]; then
    echo "==== BASELINES (f${FUNCTION}, generate-once) ===="
    for BL in ${BASELINES}; do
        python -m scripts.run_baseline --baseline \$BL \\
            --config "${CONFIG}" --function ${FUNCTION} ${SMOKE_FLAG} --skip-existing
    done
fi

echo "==== TRAINING (train_es) ===="
set +e
python -m derl2.training.train_es \\
    --config "${CONFIG}" ${SMOKE_FLAG} --function ${FUNCTION} \\
    --out-dir "\${WORK_DIR}" --workers ${CPUS} --max-hours ${MAX_HOURS} ${EXTRA_ARGS}
RC=\$?
set -e
if [ \$RC -eq 42 ]; then
    STATUS="timeout"
    echo "walltime stop; re-run with --resume to continue."
    exit 0
elif [ \$RC -ne 0 ]; then
    echo "training failed (rc \$RC)."; exit \$RC
fi

echo "==== EVALUATION (evaluate_es) ===="
set +e
python -m derl2.evaluation.evaluate_es \\
    --train-dir "\${WORK_DIR}" --function ${FUNCTION} ${EXTRA_ARGS}
RC=\$?
set -e
if [ \$RC -ne 0 ]; then echo "evaluation failed (rc \$RC)."; exit \$RC; fi

STATUS="completed"
echo "Job finished."
EOF
}

# ------------------------------------------------------------ dispatch
if [ "$KIND" = "eval" ]; then
    submit_one ""
elif [ "$KIND" = "validation" ]; then
    if [ -z "$FUNCTION" ]; then
        echo "ERROR: validation needs --function N (it sizes one function)."; exit 1
    fi
    submit_one "$FUNCTION"
elif [ -n "$FUNCTION" ]; then
    submit_one "$FUNCTION"
else
    # Per-function fan-out (Sep-CMA-ES trains one function per job).
    FUNCS=$(python -m derl2.config --config "$CONFIG" benchmark.functions | tr -d '[],')
    echo "Sep-CMA-ES per-function experiment: one job per function: $FUNCS"
    for f in $FUNCS; do submit_one "$f"; done
fi

echo ""
echo "Submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $REPO/slurm_logs/${EXP_ID}_${KIND}_<jobid>.out"
