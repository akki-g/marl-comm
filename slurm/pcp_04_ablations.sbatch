#!/bin/bash
#SBATCH --job-name=commstudy_pcp_ablations
#SBATCH --partition=normal
#SBATCH --array=0-200%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/pcp_ablations/%A_%a.out
#SBATCH --error=slurm/logs/pcp_ablations/%A_%a.err
#
# STEP 4 of 5 for Predator-Capture-Prey. The full PCP ablation study.
#
#     sbatch slurm/pcp_04_ablations.sbatch
#
# 201 rows across seven ablations, seeds 0-2, under the same protocol as the PCP
# main comparison so every ablation is directly comparable to it:
#
#     message_dim           48  8 / 16 / 32 / 64, all four learned modules
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
# Every communication override in these suites is group-scoped
# (model_config.groups.adversary.params...), because PCP's actor is a group
# ensemble. The flat Simple Spread path does not fail on a grouped config -- it
# merges in as a key nothing reads -- so an ablation written that way would run
# 201 rows at default settings and report them as a swept study.
# `test_pcp_integration.py` asserts the override lands on the adversary group.
#
# Gated by the main comparison, and by the same two open questions it is: at the
# measured budget the deterministic evaluation return is still zero because the
# predators never catch the prey, and the horizon is uncalibrated. See
# slurm/README.md, "Predator-Capture-Prey".
#
# DO NOT SUBMIT THIS BEFORE THE MAIN COMPARISON IS READ. An ablation grid is
# only interpretable once the main comparison shows the mechanisms separate at
# all on this task; 201 rows around a null result costs the same as 201 rows
# around a real one.
#
# Ordering note: rows are ordered by suite, so if you cut the study short the
# earliest-finishing ablations are message_dim and dropout -- the two highest
# priority ones.

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs/pcp_ablations
# shellcheck disable=SC1091
source slurm/newton_env.sh

MANIFEST="runs/_manifests/pcp_ablations.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: $MANIFEST missing. Run: sbatch slurm/pcp_01_setup.sbatch" >&2
  echo "(the virtualenv itself comes from: sbatch slurm/01_setup.sbatch)" >&2
  exit 1
fi

echo "[pcp-ablations] row index $SLURM_ARRAY_TASK_ID of manifest $MANIFEST"

srun python scripts/sweep.py \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --reclaim-stale 3600 \
  --run \
  experiment.sampling_device="$COMMSTUDY_DEVICE" \
  experiment.train_device="$COMMSTUDY_DEVICE" \
  experiment.buffer_device="$COMMSTUDY_DEVICE"
