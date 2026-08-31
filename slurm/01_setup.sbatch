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
# Interpreter selection is deliberately defensive. Spack-built modules on this
# cluster provide `python3` but not always `python`, so calling `python` can
# silently fall through to /usr/bin/python -- which has no `venv` module. Each
# candidate module is therefore loaded and then VERIFIED to yield a non-system
# interpreter >= 3.11 that can actually run `venv`.
#
# If every candidate fails, run `sbatch slurm/00_probe.sbatch` and read
# slurm/logs/probe_<jobid>.out for the ground truth on your account.

set -euo pipefail

REPO_DIR="${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO_DIR"
mkdir -p slurm/logs

VENV="${COMMSTUDY_VENV:-$REPO_DIR/.venv-newton}"
CUDA_MODULE="${COMMSTUDY_CUDA_MODULE:-cuda/cuda-12.4.0}"
# cu126, not cu124: PyTorch 2.7 DROPPED the cu124 index, so `torch>=2.7` has no
# matching distribution there. 2.7+ ships cu118, cu126, and cu128 only.
#
# cu126 is the default because its wheels cover both architectures on this
# cluster -- Volta sm_70 (V100) and Hopper sm_90 (H100) -- so a job is safe
# wherever it lands. cu128 targets newer architectures; if you use it, pin the
# job to H100 nodes or a V100 allocation may fail at runtime with
# "no kernel image is available for execution on the device".
#
# The pip wheel bundles its own CUDA runtime, so this need not match the loaded
# cuda module; only the node's driver has to be new enough.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

# Ordered best-first. Override with COMMSTUDY_PYTHON_MODULE to force one.
if [[ -n "${COMMSTUDY_PYTHON_MODULE:-}" ]]; then
  PYTHON_MODULE_CANDIDATES=("$COMMSTUDY_PYTHON_MODULE")
else
  PYTHON_MODULE_CANDIDATES=(
    "python/python-3.11.4-gcc-12.2.0"
    "python/python-3.11.4-oneapi-2023.1.0"
    "python-3.11.4-gcc-12.2.0"
    "anaconda/anaconda-2024.10"
    "anaconda/anaconda-2023.09"
  )
fi

echo "=============================================================="
echo "repo   $REPO_DIR"
echo "venv   $VENV"
echo "cuda   $CUDA_MODULE"
echo "=============================================================="

# Return a usable interpreter path on stdout, or fail.
# "Usable" = not the system python, >= 3.11, and `-m venv` actually works.
resolve_python() {
  local candidate path
  for candidate in python3.11 python3 python; do
    path="$(command -v "$candidate" 2>/dev/null)" || continue
    case "$path" in /usr/bin/*|/bin/*) continue ;; esac
    if "$path" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,11) else 1)' 2>/dev/null \
       && "$path" -m venv --help >/dev/null 2>&1; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN=""
PYTHON_MODULE=""
for candidate_module in "${PYTHON_MODULE_CANDIDATES[@]}"; do
  module purge >/dev/null 2>&1 || true
  if ! module load "$candidate_module" >/dev/null 2>&1; then
    echo "[setup] module not found: $candidate_module"
    continue
  fi
  module load "$CUDA_MODULE" >/dev/null 2>&1 || true
  if PYTHON_BIN="$(resolve_python)"; then
    PYTHON_MODULE="$candidate_module"
    echo "[setup] using module  $PYTHON_MODULE"
    echo "[setup] interpreter   $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"
    break
  fi
  echo "[setup] loaded $candidate_module but found no usable python3 >= 3.11 with venv"
  PYTHON_BIN=""
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo >&2
  echo "ERROR: no usable Python module found." >&2
  echo "Tried: ${PYTHON_MODULE_CANDIDATES[*]}" >&2
  echo >&2
  echo "What is actually available:" >&2
  # `|| true`: these are best-effort diagnostics on a failure path. Under
  # `set -e -o pipefail` a failing `module avail` would abort before the
  # instructions below ever print.
  { module avail python 2>&1 | head -40 >&2; } || true
  { module avail anaconda 2>&1 | head -20 >&2; } || true
  echo >&2
  echo "Then rerun with the correct name:" >&2
  echo "  sbatch --export=ALL,COMMSTUDY_PYTHON_MODULE=<name> slurm/01_setup.sbatch" >&2
  echo "Or run 'sbatch slurm/00_probe.sbatch' for a full report." >&2
  exit 1
fi

# -------------------------------------------------------------- network check
# Compute nodes on many clusters have no outbound internet. Detect that BEFORE
# building anything, so the failure is one clear message instead of a half-built
# venv that later makes all 231 array tasks die inside an import traceback.
NEED_INSTALL=1
if [[ -x "$VENV/bin/python" ]] \
   && "$VENV/bin/python" -c 'import torch, benchmarl, torchrl, tensordict, vmas' 2>/dev/null; then
  echo "[setup] existing virtualenv is already complete; skipping install"
  NEED_INSTALL=0
fi

if [[ "$NEED_INSTALL" -eq 1 ]] \
   && ! "$PYTHON_BIN" -c '
import socket, sys
socket.setdefaulttimeout(10)
try:
    socket.create_connection(("pypi.org", 443)).close()
except OSError as error:
    sys.exit(str(error) or "unreachable")
' 2>/dev/null; then
  cat >&2 <<EOF

==============================================================
ERROR: this compute node cannot reach pypi.org.

Package installation must happen on a LOGIN node. Run:

  cd $REPO_DIR
  module load $PYTHON_MODULE $CUDA_MODULE
  $PYTHON_BIN -m venv $VENV
  source $VENV/bin/activate
  pip install --upgrade pip wheel setuptools
  pip install --index-url $TORCH_INDEX 'torch>=2.7'
  pip install -e '.[dev,analysis]'

Then resubmit this script. It will detect the completed
environment, skip the install, and just build the manifests.
==============================================================
EOF
  exit 1
fi

# ---------------------------------------------------------------- environment
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[setup] creating virtualenv at $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Inside an activated venv, `python` always exists, so plain `python` is safe
# from here on -- unlike before activation.
echo "[setup] venv python    $(command -v python) ($(python -V 2>&1))"

if [[ "$NEED_INSTALL" -eq 1 ]]; then
  echo "[setup] installing packages"
  python -m pip install --upgrade pip wheel setuptools
  # Install the CUDA torch build first so benchmarl/torchrl cannot pull the CPU
  # wheel in as a transitive dependency.
  python -m pip install --index-url "$TORCH_INDEX" "torch>=2.7"
  python -m pip install -e ".[dev,analysis]"
fi

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

# Record the resolved module so the training jobs use the identical one.
cat > "$REPO_DIR/slurm/.resolved_env" <<EOF
# Written by 01_setup.sbatch. Sourced by newton_env.sh so every training job
# uses exactly the interpreter this setup verified.
COMMSTUDY_PYTHON_MODULE="$PYTHON_MODULE"
COMMSTUDY_CUDA_MODULE="$CUDA_MODULE"
COMMSTUDY_VENV="$VENV"
EOF
echo "[setup] recorded resolved environment in slurm/.resolved_env"

echo
echo "=============================================================="
echo "Setup complete. Next:"
echo "  sbatch slurm/02_main_comparison.sbatch     # 30 rows"
echo "  sbatch slurm/03_ablations.sbatch           # 201 rows"
echo "=============================================================="
