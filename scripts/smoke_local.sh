#!/bin/bash
# Local smoke test (no Slurm): validate one experiment end-to-end before
# uploading to Nibi. Reproduces what run_experiment.sh does for a `smoke` job --
# generate the function's baselines, then a tiny --smoke train + in-process eval
# -- but runs directly with the local venv. If this passes locally, the code path
# is sound; only Slurm/walltime can still differ on Nibi.
#
# Usage:  ./scripts/smoke_local.sh <EXP_ID> [function]
#   e.g.  ./scripts/smoke_local.sh EXP017_hybridppo_cov_15k 2
set -euo pipefail

EXP="${1:?usage: smoke_local.sh <EXP_ID> [function]}"
FN="${2:-2}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
CFG="$REPO/experiments/$EXP/config.yaml"
[ -x "$PY" ]   || { echo "no venv python at $PY"; exit 1; }
[ -f "$CFG" ]  || { echo "no config: $CFG"; exit 1; }

OUT="$REPO/experiments/$EXP/smokes/f$FN"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$REPO"
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1

echo "==== BASELINES  ($EXP f$FN, smoke) ===="
# Read the baselines THIS experiment declares (BASE001/003 for the continuous and
# boxnp spaces; DQN experiments also declare BASE002_fixed_schedule), exactly as
# run_experiment.sh does -- so the smoke covers whatever the real job will need.
BASELINES=$("$PY" -c "import sys; from derl2.config import Config; \
print(' '.join(Config.from_file(sys.argv[1]).get('evaluation.compare_against')))" "$CFG")
for BL in $BASELINES; do
    "$PY" -m scripts.run_baseline --baseline "$BL" --config "$CFG" \
        --function "$FN" --smoke --skip-existing
done

echo "==== TRAINING + IN-PROCESS EVAL  ($EXP f$FN, smoke) ===="
"$PY" -m derl2.training.train --config "$CFG" --smoke --kind smoke \
    --out-dir "$OUT" --max-hours 1 --set "benchmark.functions=[$FN]"

echo "==== SMOKE OK: $EXP f$FN ===="
ls "$OUT"
echo "(delete $OUT when done; it is not committed)"
