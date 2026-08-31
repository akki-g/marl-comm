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

banner "partitions"
sinfo -o "%20P %10l %15G %6D %t" 2>&1 | head -30

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

banner "done"
echo "Use a module marked USABLE above. Override the default with:"
echo "  sbatch --export=ALL,COMMSTUDY_PYTHON_MODULE=<name> slurm/01_setup.sbatch"
