#!/bin/bash
#SBATCH --job-name=commstudy_setup
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/setup_%j.out
#SBATCH --error=slurm/logs/setup_%j.err
#
# STEP 1 of 4. Run once, before anything else.
#
#     cd ~/marl-comm
#     sbatch slurm/01_setup.sbatch
#
# Builds the virtualenv, verifies CUDA is visible from a compute node, and
# writes every run manifest so the training arrays never race to create them.
#
# IF THIS FAILS WITH NETWORK/DNS ERRORS: Newton compute nodes may have no
# outbound internet. In that case run the pip commands on the LOGIN node
# instead -- they are printed in the log and repeated in slurm/README.md --
# then submit this script again to do the manifest half.

set -euo pipefail

REPO_DIR="${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO_DIR"
mkdir -p slurm/logs

VENV="${COMMSTUDY_VENV:-$REPO_DIR/.venv-newton}"
PYTHON_MODULE="${COMMSTUDY_PYTHON_MODULE:-python/python-3.11.4-gcc-12.2.0}"
CUDA_MODULE="${COMMSTUDY_CUDA_MODULE:-cuda/cuda-12.4.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "=============================================================="
echo "repo   $REPO_DIR"
echo "venv   $VENV"
echo "python $PYTHON_MODULE"
echo "cuda   $CUDA_MODULE"
echo "=============================================================="

module purge
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"

# ---------------------------------------------------------------- environment
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[setup] creating virtualenv"
  python -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[setup] installing packages"
echo "[setup] if this step fails, run these four commands on a LOGIN node:"
echo "        module load $PYTHON_MODULE $CUDA_MODULE"
echo "        source $VENV/bin/activate"
echo "        pip install --index-url $TORCH_INDEX 'torch>=2.7'"
echo "        pip install -e '.[dev,analysis]'"

python -m pip install --upgrade pip wheel setuptools
# Install the CUDA torch build first so benchmarl/torchrl cannot pull the CPU
# wheel in as a transitive dependency.
python -m pip install --index-url "$TORCH_INDEX" "torch>=2.7"
python -m pip install -e ".[dev,analysis]"

# ------------------------------------------------------------- verify device
echo "[setup] === verification ==="
python - <<'PY'
import torch, benchmarl, torchrl, tensordict, vmas
print(f"torch          {torch.__version__}")
print(f"cuda build     {torch.version.cuda}")
print(f"cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu            {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit(
        "CUDA is not visible on this compute node. Training would silently "
        "run on CPU. Check the cuda module and --gres before continuing."
    )
print(f"benchmarl      {benchmarl.__version__}")
print(f"torchrl        {torchrl.__version__}")
print(f"tensordict     {tensordict.__version__}")
print(f"vmas           {vmas.__version__}")
PY

# ----------------------------------------------------------------- manifests
# Written here, once, so the training arrays only ever read them.
echo "[setup] === building manifests ==="

echo "[setup] main comparison"
python scripts/sweep.py configs/sweeps/simple_spread_comm_v2.yaml | tail -2

echo "[setup] ablation study (combined)"
python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_stage2_message_dim.yaml \
  configs/sweeps/simple_spread_comm_v2_stage2_dropout.yaml \
  configs/sweeps/simple_spread_comm_v2_stage3_rounds.yaml \
  configs/sweeps/simple_spread_comm_v2_stage3_heads.yaml \
  configs/sweeps/simple_spread_comm_v2_stage4_graph_topology.yaml \
  configs/sweeps/simple_spread_comm_v2_stage4_sender_budget.yaml \
  configs/sweeps/simple_spread_comm_v2_stage4_self_communication.yaml \
  --combine-out runs/_manifests/ablations.csv | tail -2

echo
echo "=============================================================="
echo "Setup complete. Next:"
echo "  sbatch slurm/02_main_comparison.sbatch     # 30 rows"
echo "  sbatch slurm/03_ablations.sbatch           # 201 rows"
echo "=============================================================="
