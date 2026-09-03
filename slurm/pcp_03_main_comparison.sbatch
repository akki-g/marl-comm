#!/bin/bash
#SBATCH --job-name=commstudy_pcp_main
#SBATCH --partition=normal
#SBATCH --array=0-29%15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/pcp_main/%A_%a.out
#SBATCH --error=slurm/logs/pcp_main/%A_%a.err
#
# STEP 3 of 5 for Predator-Capture-Prey. The main PCP communication comparison.
#
#     sbatch slurm/pcp_03_main_comparison.sbatch
#
# 30 rows = 6 models x 5 seeds, on three trained predators pursuing one scripted
# prey:
#
#     pcp_actor           framework reference MLP
#     pcp_comm_identity   no-communication control (same actor shell)
#     pcp_comm_broadcast  CommNet-inspired mean broadcast
#     pcp_comm_gated      IC3Net-inspired sender gating
#     pcp_comm_attention  TarMAC-inspired targeted attention
#     pcp_comm_graph      DGN/GAT-inspired graph message passing
#
# Communication lives on the `adversary` group only. The `agent` (prey) group
# still builds and optimizes a policy, which the scenario discards every step in
# favour of its scripted pursuit-evasion force; that wasted compute is known and
# accepted, not a bug in the logs.
#
# The zero-return defect is fixed (2026-09-03): the study return is now the
# predators' own, declared per task as `return_groups`. Two things it did NOT
# fix, both of which still gate this suite -- deterministic evaluation had not
# caught the prey once at this budget, and the horizon has never been
# calibrated. See slurm/README.md, "Predator-Capture-Prey".
#
# DO NOT SUBMIT THIS BEFORE READING THE PILOT. The suite runs at 60,000 frames,
# a first-pass number chosen because PCP has no stability gate, not because
# anything showed 60,000 is enough. At that budget there are ten PPO update
# rounds and five evaluation points, so the final-10% statistic is one point. If
# the pilot shows returns still flat, raise the budget in the suite YAMLs and
# rebuild the manifests before spending 30 rows here -- exactly the step V1 on
# Simple Spread skipped.
#
# One array task = one row. %15 caps concurrency so the DPH burn stays
# predictable. Re-submitting is safe and is the normal way to recover: completed
# rows are skipped, and rows abandoned by a preempted or walltime-killed task
# are retried under an explicit __retryNN id.

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs/pcp_main
# shellcheck disable=SC1091
source slurm/newton_env.sh

MANIFEST="runs/pcp_comm_main/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: $MANIFEST missing. Run: sbatch slurm/pcp_01_setup.sbatch" >&2
  echo "(the virtualenv itself comes from: sbatch slurm/01_setup.sbatch)" >&2
  exit 1
fi

echo "[pcp-main] row index $SLURM_ARRAY_TASK_ID of manifest $MANIFEST"

srun python scripts/sweep.py \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --reclaim-stale 3600 \
  --run \
  experiment.sampling_device="$COMMSTUDY_DEVICE" \
  experiment.train_device="$COMMSTUDY_DEVICE" \
  experiment.buffer_device="$COMMSTUDY_DEVICE"
