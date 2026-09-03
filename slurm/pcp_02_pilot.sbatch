#!/bin/bash
#SBATCH --job-name=commstudy_pcp_pilot
#SBATCH --partition=normal
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/pcp_pilot/%A_%a.out
#SBATCH --error=slurm/logs/pcp_pilot/%A_%a.err
#
# STEP 2 of 5 for Predator-Capture-Prey. Two rows. Read them before step 3.
#
#     sbatch slurm/pcp_02_pilot.sbatch
#
# The no-communication control at two training budgets, 60,000 and 120,000
# frames, seed 0. This is the cheapest thing that answers the question the rest
# of the study depends on: do the predators learn anything on PCP at all, under
# the optimizer protocol frozen for Simple Spread?
#
# Why this step exists at all. Simple Spread's protocol was frozen only after a
# five-seed MLP gate passed at full horizon, and V1 is preserved in this repo as
# evidence of what happens without one: 30 rows launched under an unvalidated
# protocol, stopped at six completed and one failed, none of them usable. PCP
# has no such gate. It reuses Simple Spread's optimizer protocol on a different
# reward structure -- sparse capture events rather than dense coverage cost --
# and nothing has yet shown that transfers.
#
# The reported return is the predators' own (`return_groups: [adversary]` in
# configs/tasks/vmas_predator_capture_prey.yaml). Before 2026-09-03 it was an
# average over both groups, which is identically zero on this zero-sum scenario;
# metrics.csv now also carries a per-group breakdown, so the scripted prey's
# mirrored return stays visible as its own row.
#
# What to look for in results/pcp_identity_pilot/REPORT.md:
#   * the adversary return rises between the 60k and 120k rows, rather than
#     sitting flat. A uniformly random policy scores about 1.5 per episode on
#     this scenario, so anything near that is not learning -- and 231 rows at
#     that budget would measure noise with confidence intervals around it.
#   * deterministic evaluation catches the prey at all. In the 60k rows
#     measured so far it never did, which makes the evaluation return a
#     five-sample rare-event count rather than a usable comparison metric.
#   * all logged values finite, policy entropy stable, no action saturation.
#     These are the same failure modes that killed V1 on Simple Spread.
#
# Rows are cheap: a 60,000-frame PCP row is roughly one to two minutes of
# compute. The cost of this step is queue time, not DPH, and it is the only
# thing standing between an uncalibrated horizon and 231 rows spent on it.

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs/pcp_pilot
# shellcheck disable=SC1091
source slurm/newton_env.sh

MANIFEST="runs/pcp_identity_pilot/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: $MANIFEST missing. Run: sbatch slurm/pcp_01_setup.sbatch" >&2
  echo "(the virtualenv itself comes from: sbatch slurm/01_setup.sbatch)" >&2
  exit 1
fi

echo "[pcp-pilot] row index $SLURM_ARRAY_TASK_ID of manifest $MANIFEST"

srun python scripts/sweep.py \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --reclaim-stale 3600 \
  --run \
  experiment.sampling_device="$COMMSTUDY_DEVICE" \
  experiment.train_device="$COMMSTUDY_DEVICE" \
  experiment.buffer_device="$COMMSTUDY_DEVICE"
