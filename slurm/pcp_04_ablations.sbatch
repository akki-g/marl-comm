#!/bin/bash
#SBATCH --job-name=commstudy_pcp_ablations
#SBATCH --partition=normal
#SBATCH --array=0-200%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/pcp_ablations/%A_%a.out
#SBATCH --error=slurm/logs/pcp_ablations/%A_%a.err
# Use: bash slurm/pcp_04_ablations.sbatch
# The login-node branch checks approval BEFORE invoking sbatch. Direct sbatch
# bypasses that outer check; the worker still rejects before srun/training.
set -euo pipefail

TASK_REPO="${COMMSTUDY_REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
export COMMSTUDY_REPO="$TASK_REPO"
cd "$TASK_REPO"
MANIFEST="runs/_manifests/pcp_ablations_corrected_v1.csv"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  TASK_PYTHON="${COMMSTUDY_PYTHON:-$TASK_REPO/.venv-newton/bin/python}"
  if [[ ! -x "$TASK_PYTHON" ]]; then
    TASK_PYTHON="$TASK_REPO/.venv/bin/python"
  fi
  "$TASK_PYTHON" scripts/protocol.py check --manifest "$MANIFEST" --stage ablation
  mkdir -p slurm/logs/pcp_ablations
  exec sbatch "slurm/pcp_04_ablations.sbatch" "$@"
fi

# shellcheck disable=SC1091
source slurm/newton_env.sh
python scripts/protocol.py check --manifest "$MANIFEST" --stage ablation \
  --index "$SLURM_ARRAY_TASK_ID" --runtime
srun python scripts/sweep.py --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" --reclaim-stale 3600 --run
