#!/bin/bash
# Submit the DE/current-to-pbest/1/bin CEC'13 sweep to Nibi as a job ARRAY:
# one function per array task (28 tasks), then a dependent merge job that
# assembles runs/<NAME>/{summary,per_run}.csv from the per-task parts.
#
#   bash scripts/submit_cec13_ctpbest.sh
#
# Each task runs `--runs` full-budget DE runs of ONE function into
# runs/<NAME>/_parts/f<N>/, so the 28 functions run in parallel (~minutes of
# wall-clock instead of ~3h serial). The merge job runs only afterok of the
# whole array and writes the two final files at runs/<NAME>/.
#
# Re-running: the array overwrites runs/<NAME>/_parts/f<N>/ per task; delete
# runs/<NAME>/ first if you want a guaranteed-clean sweep.
set -euo pipefail

# ------------------------------------------------------------ protocol
DIM=30
RUNS=51
NPOP=100
FVAL=0.5
CRVAL=0.5
PBEST=0.11
STRATEGY="current-to-pbest/1/bin"
NAME="cec13_de_ctpbest_d30"
NFUNCS=28                     # CEC'13 f1..f28

# ------------------------------------------------------------ cluster
PYTHON_MODULE="python/3.11"
VENV="$HOME/de-rl2-env"
REPO="$HOME/de-rl2"
ACCOUNT="def-bolufe"
ARRAY_TIME="01:00:00"        # per function; ~6-10 min in practice, generous
ARRAY_MEM="8G"
ARRAY_CPUS=4
MERGE_TIME="00:15:00"

cd "$REPO"
mkdir -p "$REPO/slurm_logs"

COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo "unknown")

echo "=============================================="
echo " sweep      : DE/${STRATEGY}"
echo " protocol   : NP=${NPOP} F=${FVAL} CR=${CRVAL} p=${PBEST} D=${DIM} runs=${RUNS}"
echo " output     : ${REPO}/runs/${NAME}/{summary,per_run}.csv"
echo " commit     : ${COMMIT:0:12}"
echo " array      : 1-${NFUNCS} (one function per task), ${ARRAY_CPUS} cpus ${ARRAY_MEM} ${ARRAY_TIME}"
echo " account    : ${ACCOUNT}"
echo "=============================================="

# ---------------------------------------------------------- array job
ARRAY_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}_run
#SBATCH --account=${ACCOUNT}
#SBATCH --array=1-${NFUNCS}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${ARRAY_CPUS}
#SBATCH --mem=${ARRAY_MEM}
#SBATCH --time=${ARRAY_TIME}
#SBATCH --output=${REPO}/slurm_logs/%x_%A_%a.out
#SBATCH --error=${REPO}/slurm_logs/%x_%A_%a.err

set -euo pipefail
module purge
module load ${PYTHON_MODULE}
source ${VENV}/bin/activate
cd ${REPO}
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

FUNC="f\${SLURM_ARRAY_TASK_ID}"
echo "==== task \${SLURM_ARRAY_TASK_ID}  function \${FUNC} ===="
date; hostname; python --version

python scripts/run_cec13_de_ctpbest.py \\
    --dim ${DIM} --runs ${RUNS} --functions "\${FUNC}" \\
    --np ${NPOP} --f ${FVAL} --cr ${CRVAL} --p ${PBEST} \\
    --strategy "${STRATEGY}" --name "${NAME}" \\
    --out-subdir "_parts/\${FUNC}"

echo "task \${FUNC} done."; date
EOF
)
echo "submitted array job ${ARRAY_ID} (${NFUNCS} tasks)"

# --------------------------------------------------- dependent merge job
MERGE_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}_merge
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=${MERGE_TIME}
#SBATCH --output=${REPO}/slurm_logs/%x_%j.out
#SBATCH --error=${REPO}/slurm_logs/%x_%j.err

set -euo pipefail
module purge
module load ${PYTHON_MODULE}
source ${VENV}/bin/activate
cd ${REPO}
export PYTHONUNBUFFERED=1

echo "==== merge ${NAME} ===="; date
python scripts/run_cec13_de_ctpbest.py --name "${NAME}" --merge
echo "merge done."; date
EOF
)
echo "submitted merge job ${MERGE_ID} (runs afterok:${ARRAY_ID})"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Result:   ${REPO}/runs/${NAME}/summary.csv and per_run.csv (after merge)"
