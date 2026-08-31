#!/bin/bash
#SBATCH --job-name=commstudy_main
#SBATCH --partition=normal
#SBATCH --array=0-29%15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/main/%A_%a.out
#SBATCH --error=slurm/logs/main/%A_%a.err
#
# STEP 2 of 4. The main Simple Spread communication comparison.
#
#     sbatch slurm/02_main_comparison.sbatch
#
# 30 rows = 6 models x 5 seeds, 600,000 frames each:
#
#     benchmarl_mlp    framework reference MLP
#     comm_identity    no-communication control (same actor shell)
#     comm_broadcast   CommNet-inspired mean broadcast
#     comm_gated       IC3Net-inspired sender gating
#     comm_attention   TarMAC-inspired targeted attention
#     comm_graph       DGN/GAT-inspired graph message passing
#
# One array task = one row. The array size matches the 30 manifest rows written
# by 01_setup.sbatch; %15 caps concurrency so the DPH burn stays predictable.
#
# Re-submitting is safe and is the normal way to recover: completed rows are
# skipped, and rows abandoned by a preempted or walltime-killed task are
# retried under an explicit __retryNN id.

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs/main
# shellcheck disable=SC1091
source slurm/newton_env.sh

MANIFEST="runs/simple_spread_comm_v2/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: $MANIFEST missing. Run: sbatch slurm/01_setup.sbatch" >&2
  exit 1
fi

echo "[main] row index $SLURM_ARRAY_TASK_ID of manifest $MANIFEST"

srun python scripts/sweep.py \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --reclaim-stale 3600 \
  --run \
  experiment.sampling_device="$COMMSTUDY_DEVICE" \
  experiment.train_device="$COMMSTUDY_DEVICE" \
  experiment.buffer_device="$COMMSTUDY_DEVICE"
