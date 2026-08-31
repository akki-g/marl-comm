#!/bin/bash
# One-time environment build for UCF ARCC Newton. Run on a LOGIN node.
#
#     bash slurm/bootstrap_newton.sh
#
# Compute nodes may not have outbound network access, so all package
# installation happens here rather than inside a job.

set -euo pipefail

COMMSTUDY_REPO="${COMMSTUDY_REPO:-$HOME/marl-comm}"
COMMSTUDY_VENV="${COMMSTUDY_VENV:-$COMMSTUDY_REPO/.venv-newton}"
COMMSTUDY_PYTHON_MODULE="${COMMSTUDY_PYTHON_MODULE:-python/python-3.11.4-gcc-12.2.0}"
COMMSTUDY_CUDA_MODULE="${COMMSTUDY_CUDA_MODULE:-cuda/cuda-12.4.0}"
# Match the CUDA module. Check https://pytorch.org for the current index URL.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

module purge
module load "$COMMSTUDY_PYTHON_MODULE"
module load "$COMMSTUDY_CUDA_MODULE"

cd "$COMMSTUDY_REPO"
python -m venv "$COMMSTUDY_VENV"
# shellcheck disable=SC1091
source "$COMMSTUDY_VENV/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Install the CUDA build of torch first so the CPU wheel is not pulled in as a
# transitive dependency of benchmarl/torchrl.
python -m pip install --index-url "$TORCH_INDEX" "torch>=2.7"
python -m pip install -e ".[dev,analysis]"

echo
echo "=== verification ==="
python - <<'PY'
import torch
print("torch          ", torch.__version__)
print("cuda available ", torch.cuda.is_available())
print("cuda version   ", torch.version.cuda)
if torch.cuda.is_available():
    print("device         ", torch.cuda.get_device_name(0))
import benchmarl, torchrl, tensordict, vmas
print("benchmarl      ", benchmarl.__version__)
print("torchrl        ", torchrl.__version__)
print("tensordict     ", tensordict.__version__)
print("vmas           ", vmas.__version__)
PY

echo
echo "Environment ready at $COMMSTUDY_VENV"
echo "Note: torch.cuda.is_available() is False on login nodes without a GPU;"
echo "confirm on a compute node with: srun --gres=gpu:1 --pty nvidia-smi"
