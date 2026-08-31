#!/bin/bash
# Shared environment bootstrap for UCF ARCC Newton.
#
# Sourced (never executed) by every job script, so the module/venv setup lives
# in exactly one place. Sourcing needs no execute permission.
#
# Module names are read from slurm/.resolved_env when it exists, which
# 01_setup.sbatch writes after verifying an interpreter actually works. That
# guarantees the training jobs use the same Python the setup validated instead
# of re-guessing. Override anything by exporting it before submission:
#
#     sbatch --export=ALL,COMMSTUDY_CUDA_MODULE=cuda/cuda-12.6.0 \
#       slurm/02_main_comparison.sbatch
#
# Newton specifics this encodes (verify with `sbatch slurm/00_probe.sbatch`,
# which prints the ground truth for your account):
#   * Slurm queues are `normal` (default) and `preemptable`.
#   * GPUs are requested with --gres=gpu:N.
#   * Modules are named <family>/<name-version>, e.g. cuda/cuda-12.4.0.
#   * $HOME lives on /lustre/fs1 with a 1 TB / 1,000,000-file quota and is
#     NOT backed up.

set -euo pipefail

# --- Repository and environment locations ------------------------------------
# SLURM_SUBMIT_DIR is where sbatch was invoked, which is the repo root in the
# documented workflow. It keeps the scripts working from any clone path.
COMMSTUDY_REPO="${COMMSTUDY_REPO:-${SLURM_SUBMIT_DIR:-$HOME/marl-comm}}"

# Values verified and recorded by 01_setup.sbatch. Anything already exported
# wins, so a manual override is still honoured.
if [[ -f "$COMMSTUDY_REPO/slurm/.resolved_env" ]]; then
  _resolved_python="${COMMSTUDY_PYTHON_MODULE:-}"
  _resolved_cuda="${COMMSTUDY_CUDA_MODULE:-}"
  _resolved_venv="${COMMSTUDY_VENV:-}"
  # shellcheck disable=SC1091
  source "$COMMSTUDY_REPO/slurm/.resolved_env"
  [[ -n "$_resolved_python" ]] && COMMSTUDY_PYTHON_MODULE="$_resolved_python"
  [[ -n "$_resolved_cuda" ]] && COMMSTUDY_CUDA_MODULE="$_resolved_cuda"
  [[ -n "$_resolved_venv" ]] && COMMSTUDY_VENV="$_resolved_venv"
  unset _resolved_python _resolved_cuda _resolved_venv
fi

COMMSTUDY_VENV="${COMMSTUDY_VENV:-$COMMSTUDY_REPO/.venv-newton}"
COMMSTUDY_PYTHON_MODULE="${COMMSTUDY_PYTHON_MODULE:-python/python-3.11.4-gcc-12.2.0}"
COMMSTUDY_CUDA_MODULE="${COMMSTUDY_CUDA_MODULE:-cuda/cuda-12.4.0}"

# --- Modules ------------------------------------------------------------------
module purge >/dev/null 2>&1 || true
module load "$COMMSTUDY_PYTHON_MODULE" >/dev/null 2>&1 \
  || echo "[env] WARNING: could not load $COMMSTUDY_PYTHON_MODULE"
module load "$COMMSTUDY_CUDA_MODULE" >/dev/null 2>&1 \
  || echo "[env] WARNING: could not load $COMMSTUDY_CUDA_MODULE"

if [[ ! -x "$COMMSTUDY_VENV/bin/python" ]]; then
  echo "ERROR: virtualenv not found at $COMMSTUDY_VENV" >&2
  echo "Create it once with:  sbatch slurm/01_setup.sbatch" >&2
  echo "If that fails, diagnose with:  sbatch slurm/00_probe.sbatch" >&2
  exit 1
fi
# Activating the venv puts a guaranteed `python` on PATH, so the job scripts
# never depend on whether a module happened to provide one.
# shellcheck disable=SC1091
source "$COMMSTUDY_VENV/bin/activate"

# A venv DIRECTORY existing is not the same as a venv that WORKS. If setup died
# partway through -- most often because a compute node has no outbound network
# and pip failed -- the directory is still there and every array task would sail
# past a mere existence check and then die deep inside an import traceback.
# Fail here instead, once, with the actual remedy.
if ! python -c 'import torch, benchmarl, torchrl, tensordict, vmas' 2>/dev/null; then
  {
    echo "ERROR: the virtualenv at $COMMSTUDY_VENV is incomplete."
    echo
    echo "It exists but its packages are missing, so 01_setup.sbatch did not"
    echo "finish. The usual cause is that compute nodes have no outbound"
    echo "network, so pip could not reach PyPI."
    echo
    echo "Missing import:"
    # `|| true` is essential: this import is EXPECTED to fail, and under
    # `set -e -o pipefail` its failure would abort the block and swallow the
    # remediation text below -- the only part of this message that helps.
    python -c 'import torch, benchmarl, torchrl, tensordict, vmas' 2>&1 | tail -3 || true
    echo
    echo "Fix it from a LOGIN node (which does have network):"
    echo "  cd $COMMSTUDY_REPO"
    echo "  module load $COMMSTUDY_PYTHON_MODULE $COMMSTUDY_CUDA_MODULE"
    echo "  source $COMMSTUDY_VENV/bin/activate"
    echo "  pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.7'"
    echo "  pip install -e '.[dev,analysis]'"
    echo
    echo "Then resubmit:  sbatch slurm/01_setup.sbatch"
  } >&2
  exit 1
fi

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
echo "[env] python=$(command -v python) ($(python -V 2>&1))"
echo "[env] device=$COMMSTUDY_DEVICE threads=$COMMSTUDY_THREADS"
echo "[env] node=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
python - <<'PY'
import torch
print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[env] gpu={torch.cuda.get_device_name(0)}")
PY
