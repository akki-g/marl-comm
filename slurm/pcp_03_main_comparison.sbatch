#!/bin/bash
#SBATCH --job-name=commstudy_pcp_main
#SBATCH --partition=normal
#SBATCH --array=0-29%15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/pcp_main/%A_%a.out
#SBATCH --error=slurm/logs/pcp_main/%A_%a.err
# Use: bash slurm/pcp_03_main_comparison.sbatch
# The login-node branch checks approval BEFORE invoking sbatch. Direct sbatch
# bypasses that outer check; the worker still rejects before srun/training.
set -euo pipefail

TASK_REPO="${COMMSTUDY_REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
export COMMSTUDY_REPO="$TASK_REPO"
cd "$TASK_REPO"
MANIFEST="runs/pcp_comm_main_corrected_v1/manifest.csv"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  TASK_PYTHON="${COMMSTUDY_PYTHON:-$TASK_REPO/.venv-newton/bin/python}"
  if [[ ! -x "$TASK_PYTHON" ]]; then
    TASK_PYTHON="$TASK_REPO/.venv/bin/python"
  fi
  "$TASK_PYTHON" scripts/protocol.py check --manifest "$MANIFEST" --stage comparison
  mkdir -p slurm/logs/pcp_main
  exec sbatch "slurm/pcp_03_main_comparison.sbatch" "$@"
fi

# shellcheck disable=SC1091
source slurm/newton_env.sh
python scripts/protocol.py check --manifest "$MANIFEST" --stage comparison \
  --index "$SLURM_ARRAY_TASK_ID" --runtime
srun python scripts/sweep.py --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" --reclaim-stale 3600 --run
