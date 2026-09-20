#!/usr/bin/env bash
set -euo pipefail
# sbatch --chdir sets the immutable source root, even from Slurm spool copies.
source "$PWD/slurm/communication_comparison_v1/common_env.sh"
suite_profile "$1" "$2"
srun --kill-on-bad-exit=1 "$PCP_PYTHON" "$SOURCE_ROOT/scripts/communication_suite.py" analyze --run-root "$2"
