#!/usr/bin/env bash
set -euo pipefail
# sbatch --chdir sets the immutable source root, even from Slurm spool copies.
source "$PWD/slurm/communication_comparison_v1/common_env.sh"
profile="$1" root="$2" task="$3"
shift 3
suite_worker "$profile" "$root" "$task" smoke --index "${SLURM_ARRAY_TASK_ID:?}" "$@"
