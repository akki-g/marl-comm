#!/bin/bash
# Shared environment bootstrap for UCF ARCC Newton.
#
# Sourced by every job script so the module/venv setup lives in exactly one
# place. Override any value by exporting it before submission, e.g.
#
#     COMMSTUDY_VENV=$HOME/envs/other sbatch slurm/run_suite.sbatch ...
#
# Newton specifics this encodes (verify against `module avail` and `sinfo`,
# which are the only authoritative sources on your account):
#   * Slurm queues are `normal` (default) and `preemptable`.
#   * GPUs are requested with --gres=gpu:N.
#   * Modules are named <family>/<name-version>, e.g. cuda/cuda-12.4.0.
#   * $HOME lives on /lustre/fs1 with a 1 TB / 1,000,000-file quota and is
#     NOT backed up.

set -euo pipefail

# --- Repository and environment locations ------------------------------------
COMMSTUDY_REPO="${COMMSTUDY_REPO:-$HOME/marl-comm}"
COMMSTUDY_VENV="${COMMSTUDY_VENV:-$COMMSTUDY_REPO/.venv-newton}"

# --- Modules ------------------------------------------------------------------
# Newton lists python-3.11.4-gcc-12.2.0; pyproject requires >=3.11.
COMMSTUDY_PYTHON_MODULE="${COMMSTUDY_PYTHON_MODULE:-python/python-3.11.4-gcc-12.2.0}"
COMMSTUDY_CUDA_MODULE="${COMMSTUDY_CUDA_MODULE:-cuda/cuda-12.4.0}"

module purge
module load "$COMMSTUDY_PYTHON_MODULE"
module load "$COMMSTUDY_CUDA_MODULE"

if [[ ! -d "$COMMSTUDY_VENV" ]]; then
  echo "ERROR: virtualenv not found at $COMMSTUDY_VENV" >&2
  echo "Create it once on a login node with slurm/bootstrap_newton.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$COMMSTUDY_VENV/bin/activate"

cd "$COMMSTUDY_REPO"
export PYTHONPATH="$COMMSTUDY_REPO/src:${PYTHONPATH:-}"

# --- Thread pinning -----------------------------------------------------------
# Every row in the study must share one numerical environment, so the intra-op
# thread count is fixed rather than inherited from whatever the node allocates.
# It is derived from the CPUs Slurm actually gave this task.
COMMSTUDY_THREADS="${COMMSTUDY_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="$COMMSTUDY_THREADS"
export MKL_NUM_THREADS="$COMMSTUDY_THREADS"
export VECLIB_MAXIMUM_THREADS="$COMMSTUDY_THREADS"
export NUMEXPR_NUM_THREADS="$COMMSTUDY_THREADS"

# --- Device -------------------------------------------------------------------
# The three BenchMARL device settings are passed as overrides by the caller.
COMMSTUDY_DEVICE="${COMMSTUDY_DEVICE:-cuda}"
export COMMSTUDY_DEVICE

echo "[env] repo=$COMMSTUDY_REPO"
echo "[env] venv=$COMMSTUDY_VENV"
echo "[env] python=$(python -V 2>&1)"
echo "[env] device=$COMMSTUDY_DEVICE threads=$COMMSTUDY_THREADS"
echo "[env] node=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
python - <<'PY'
import torch
print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[env] gpu={torch.cuda.get_device_name(0)}")
PY
