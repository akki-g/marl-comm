#!/bin/bash
#SBATCH --job-name=commstudy
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

if (( $# != 0 )); then
    echo "Usage: sbatch [--array=0|1] slurm/run_experiment.sbatch" >&2
    exit 2
fi

case "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a Slurm array}" in
    0) config=configs/pcp.yaml ;;
    1) config=configs/mapdn.yaml ;;
    *) echo "Expected array index 0 (PCP) or 1 (MAPDN)" >&2; exit 2 ;;
esac

exec uv run --locked python scripts/run_experiment.py "$config"
