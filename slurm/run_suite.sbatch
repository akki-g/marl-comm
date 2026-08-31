#!/bin/bash
#SBATCH --job-name=commstudy
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x/%A_%a.out
#SBATCH --error=slurm/logs/%x/%A_%a.err
#
# Slurm array worker for one commstudy suite.
#
# Each array task owns a contiguous CHUNK of manifest rows and runs them
# sequentially. Rows are tiny (a 19k-35k parameter actor on 10 VMAS
# environments), so one row per GPU wastes the allocation badly; chunking keeps
# a GPU busy and keeps the DPH bill proportional to real work.
#
# Submit with slurm/submit.sh, which computes the array range for you:
#
#     bash slurm/submit.sh configs/sweeps/simple_spread_comm_v2.yaml
#
# Or submit directly once a manifest exists:
#
#     sbatch --array=0-29 --job-name=simple_spread_comm_v2 \
#       slurm/run_suite.sbatch runs/simple_spread_comm_v2/manifest.csv 1
#
# Array tasks treat the manifest as immutable input and write only their own
# per-run status.json, so concurrent tasks never race on a shared file.

set -euo pipefail

MANIFEST="${1:?usage: run_suite.sbatch <manifest.csv> [chunk]}"
CHUNK="${2:-1}"

source "$(dirname "${BASH_SOURCE[0]}")/newton_env.sh"

START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

echo "[task] manifest=$MANIFEST chunk=$CHUNK start_index=$START"
echo "[task] begin $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --reclaim-stale lets a requeued task pick up rows abandoned by a preempted or
# walltime-killed predecessor instead of leaving them stuck in "running".
srun python scripts/sweep.py \
  --manifest "$MANIFEST" \
  --index "$START" \
  --count "$CHUNK" \
  --reclaim-stale 3600 \
  --run \
  experiment.sampling_device="$COMMSTUDY_DEVICE" \
  experiment.train_device="$COMMSTUDY_DEVICE" \
  experiment.buffer_device="$COMMSTUDY_DEVICE"

echo "[task] end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
