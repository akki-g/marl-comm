#!/bin/bash
#SBATCH --job-name=commstudy_ablations
#SBATCH --partition=normal
#SBATCH --array=0-200%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/ablations/%A_%a.out
#SBATCH --error=slurm/logs/ablations/%A_%a.err
#
# STEP 3 of 4. The full Simple Spread ablation study.
#
#     sbatch slurm/03_ablations.sbatch
#
# 201 rows across seven ablations, 600,000 frames each, seeds 0-2, using the
# same frozen protocol as the main comparison so every ablation is directly
# comparable to it:
#
#     message_dim          48   8 / 16 / 32 / 64, all four learned modules
#     communication_dropout 48  p = 0.00 / 0.25 / 0.50 / 0.75
#     communication_rounds  18  1 / 2 / 3, attention + graph
#     attention_heads       24  1 / 2 / 4 / 8 at fixed total width
#     graph_topology         9  full / directed ring / Erdos-Renyi
#     sender_budget         36  learned-K vs seeded random-K control
#     self_communication    18  exclude_self true / false
#
# All seven share ONE combined manifest, so this is a single contiguous array
# instead of seven ranges to track by hand. Each row still writes into its own
# suite directory, so per-suite analysis is unchanged.
#
# Every ablation deliberately re-runs its default point (message_dim=32,
# rounds=1, topology=full, heads=4, exclude_self=true). Those are matched
# in-suite controls and should reproduce the main-suite rows on seeds 0-2,
# which doubles as a reproducibility check.
#
# Ordering note: rows are ordered by suite, so if you cut the study short the
# earliest-finishing ablations are message_dim and dropout -- the two highest
# priority ones.

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs/ablations
# shellcheck disable=SC1091
source slurm/newton_env.sh

MANIFEST="runs/_manifests/ablations.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: $MANIFEST missing. Run: sbatch slurm/01_setup.sbatch" >&2
  exit 1
fi

echo "[ablations] row index $SLURM_ARRAY_TASK_ID of manifest $MANIFEST"

srun python scripts/sweep.py \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --reclaim-stale 3600 \
  --run \
  experiment.sampling_device="$COMMSTUDY_DEVICE" \
  experiment.train_device="$COMMSTUDY_DEVICE" \
  experiment.buffer_device="$COMMSTUDY_DEVICE"
