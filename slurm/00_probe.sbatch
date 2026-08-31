#!/bin/bash
#SBATCH --job-name=commstudy_probe
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/probe_%j.out
#SBATCH --error=slurm/logs/probe_%j.err
#
# STEP 0 (optional but recommended). Report what this cluster actually provides.
#
#     sbatch slurm/00_probe.sbatch
#     cat slurm/logs/probe_<jobid>.out
#
# Read-only: loads no environment, creates nothing, changes nothing. Run it
# whenever a job fails with a module or interpreter error -- the published ARCC
# docs are thin, and this prints the ground truth for your account.

set -uo pipefail   # deliberately no -e: every probe must run even if one fails

banner() { printf '\n===== %s =====\n' "$1"; }

banner "node"
hostname
echo "job ${SLURM_JOB_ID:-none}  submit_dir ${SLURM_SUBMIT_DIR:-unset}"

banner "gpu"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || echo "nvidia-smi unavailable"

banner "partitions, GRES, and node features"
# The GRES and AVAIL_FEATURES columns are what you need to target a specific
# GPU model. Whatever appears there is the exact string to pass to sbatch.
sinfo -o "%20P %10l %24G %20f %6D %t" 2>&1 | head -40

banner "how to request a specific GPU model"
# Newton mixes V100 (sm_70) and H100 (sm_90) nodes. Slurm sites expose the
# choice either as a GRES type or as a node feature; report both so the
# correct one is obvious rather than guessed.
echo "GRES types configured on this cluster:"
sinfo -h -o "%G" 2>/dev/null | tr ',' '\n' | sed 's/(.*)//' | sort -u | sed 's/^/  /'
echo
echo "Node features (usable with --constraint):"
sinfo -h -o "%f" 2>/dev/null | tr ',' '\n' | sort -u | grep -v '^(null)$' | sed 's/^/  /'
echo
echo "GPU models actually present, by node:"
scontrol show nodes 2>/dev/null \
  | awk '/NodeName=/{n=$1} /Gres=/{print "  " n "  " $0}' \
  | grep -i gpu | head -20
echo
echo "If a type appears above (e.g. gpu:h100), request it with:"
echo "  sbatch --gres=gpu:h100:1 slurm/02_main_comparison.sbatch"
echo "If instead a feature appears (e.g. h100), use:"
echo "  sbatch --constraint=h100 slurm/02_main_comparison.sbatch"

banner "system interpreters (these usually LACK venv)"
for candidate in python python3 python3.11 python3.12; do
  path="$(command -v "$candidate" 2>/dev/null)" || { echo "$candidate -> not found"; continue; }
  printf '%-10s -> %-45s %s\n' "$candidate" "$path" "$("$path" -V 2>&1)"
done

banner "module avail python"
module avail python 2>&1 | head -40

banner "module avail anaconda"
module avail anaconda 2>&1 | head -20

banner "module avail cuda"
module avail cuda 2>&1 | head -30

banner "candidate python modules: does each give a usable interpreter?"
# This is the exact check 01_setup.sbatch performs. A module is usable only if
# it yields an interpreter that is NOT the system one, is >= 3.11, and can
# actually run the venv module.
CANDIDATES=(
  "python/python-3.11.4-gcc-12.2.0"
  "python/python-3.11.4-oneapi-2023.1.0"
  "python-3.11.4-gcc-12.2.0"
  "anaconda/anaconda-2024.10"
  "anaconda/anaconda-2023.09"
)
for mod in "${CANDIDATES[@]}"; do
  module purge >/dev/null 2>&1
  if ! module load "$mod" >/dev/null 2>&1; then
    printf '%-42s MODULE NOT FOUND\n' "$mod"
    continue
  fi
  found=""
  for candidate in python3.11 python3 python; do
    path="$(command -v "$candidate" 2>/dev/null)" || continue
    case "$path" in /usr/bin/*|/bin/*) continue ;; esac
    if "$path" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,11) else 1)' 2>/dev/null \
       && "$path" -m venv --help >/dev/null 2>&1; then
      found="$path"
      break
    fi
  done
  if [[ -n "$found" ]]; then
    printf '%-42s USABLE  %s  (%s)\n' "$mod" "$found" "$("$found" -V 2>&1)"
  else
    printf '%-42s loaded, but no usable python3 >= 3.11 with venv\n' "$mod"
  fi
done
module purge >/dev/null 2>&1

banner "outbound network (needed to install packages)"
# If this fails, packages must be installed from a login node.
python3 -c '
import socket, sys
socket.setdefaulttimeout(10)
for host in ("pypi.org", "download.pytorch.org"):
    try:
        socket.create_connection((host, 443)).close()
        print(f"{host:28} reachable")
    except OSError as error:
        print(f"{host:28} UNREACHABLE ({error})")
' 2>&1 || echo "could not run network check"

banner "existing virtualenv"
VENV="${COMMSTUDY_VENV:-${SLURM_SUBMIT_DIR:-$PWD}/.venv-newton}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "$VENV -> does not exist (run 01_setup.sbatch)"
else
  echo "$VENV -> exists ($("$VENV/bin/python" -V 2>&1))"
  # A venv that exists but cannot import the stack is the failure mode that
  # otherwise kills every array task individually with an import traceback.
  for pkg in torch benchmarl torchrl tensordict vmas; do
    if version=$("$VENV/bin/python" -c "import $pkg; print(getattr($pkg,'__version__','?'))" 2>/dev/null); then
      printf '  %-12s %s\n' "$pkg" "$version"
    else
      printf '  %-12s MISSING -- venv is incomplete\n' "$pkg"
    fi
  done
  "$VENV/bin/python" -c 'import torch; print(f"  cuda available {torch.cuda.is_available()}")' 2>/dev/null || true
fi

banner "done"
echo "Use a module marked USABLE above. Override the default with:"
echo "  sbatch --export=ALL,COMMSTUDY_PYTHON_MODULE=<name> slurm/01_setup.sbatch"
