#!/bin/bash
#SBATCH --job-name=commstudy
#SBATCH --partition=normal
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
# Module/Conda initialization may refer to unset shell variables.
set -eo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch on Newton}"
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

# UCF: https://arcc.ist.ucf.edu/docs/software/anaconda/
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
module purge
module load anaconda/anaconda-2024.10
conda activate base
set -u
module list
python_bin=$(python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected Python 3.12 from the Anaconda module"; print(sys.executable)')

# Install only inside this compute allocation. Both array tasks share one venv;
# serialize setup so they cannot install into it at the same time.
uv_dir="$PWD/.tools/uv-0.12.5"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-newton"
mkdir -p .tools
(
    flock --exclusive 9
    if [[ ! -x "$uv_dir/uv" ]]; then
        curl --fail --location --silent --show-error https://astral.sh/uv/0.12.5/install.sh \
            | env UV_UNMANAGED_INSTALL="$uv_dir" sh
    fi
    "$uv_dir/uv" --version
    "$uv_dir/uv" sync --locked --python "$python_bin" --no-python-downloads
) 9>.tools/newton-setup.lock

source "$UV_PROJECT_ENVIRONMENT/bin/activate"
exec python -u scripts/run_experiment.py "$config"
