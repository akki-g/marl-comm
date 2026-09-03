#!/bin/bash
#SBATCH --job-name=commstudy_pcp_protocol
#SBATCH --partition=normal
#SBATCH --array=0-26
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/pcp_protocol/%A_%a.out
#SBATCH --error=slurm/logs/pcp_protocol/%A_%a.err
#
# STEP 2b of 5 for Predator-Capture-Prey, and the new gate before step 3.
#
#     sbatch slurm/pcp_01_setup.sbatch        # rebuild manifests first
#     sbatch slurm/pcp_02b_protocol.sbatch
#
# 27 rows: the no-communication control at gamma x entropy_coef, 3 x 3, three
# seeds, 600,000 frames. Roughly 40 min per row on a V100, so ~18 GPU-hours.
#
# This supersedes pcp_02_pilot. That step asked whether 60k or 120k frames was
# enough and answered "neither, by about an order of magnitude", so budget is
# settled as a floor and two inherited-but-untested knobs are what remain. The
# full reasoning is in docs/RESULTS_pcp_pilot.md; the short version:
#
#   * gamma. pcp_comm_main still carries Simple Spread's frozen 0.9. The prey's
#     max_speed is 1.3 against the predators' 1.0, so a one-predator chase never
#     converges and a capture needs a pincer that takes far longer than gamma
#     0.9's ~10-step horizon to arrange. The pilot learned *because* an
#     uncommitted edit meant it accidentally ran 0.99.
#   * entropy_coef. 94% of random episodes score exactly zero, so exploration
#     matters -- but the pilot also showed deterministic evaluation
#     underperforming stochastic collection, which a large entropy bonus widens.
#
# Note the environment changed under this step. configs/tasks/vmas_predator_
# capture_prey.yaml now sets predator_sensing_radius: 1.0, which hides the prey
# from a predator past that distance and makes the predator observation 17-dim
# rather than 16. Nothing measured before 2026-09-03 is comparable to these
# rows: the pilot ran a fully observable task, in which a message can only
# re-transmit what its receiver already has.
#
# What to look for in results/pcp_protocol_gate/REPORT.md:
#   * which (gamma, entropy) cell maximises mean_final_return, and whether the
#     gamma effect is as one-sided as the geometry predicts. If gamma 0.9 wins,
#     something in the above reasoning is wrong -- find out what before step 3.
#   * where the winner's return curve flattens. That frame count, not 600k, is
#     what pcp_comm_main should carry. The pilot's +1.75-per-100k slope had not
#     flattened at 120k, so treat 600k as a floor.
#   * evaluation return that is no longer a five-sample rare-event count. These
#     rows use evaluation_episodes: 128, taking the standard error from 1.69 to
#     0.34 for +24% evaluation wall-clock.
#   * all logged values finite, policy entropy stable, no action saturation.
#
# Then freeze the winning gamma and entropy_coef into every configs/sweeps/
# pcp_comm_*.yaml, rerun pcp_01_setup, and only then submit step 3.

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs/pcp_protocol
# shellcheck disable=SC1091
source slurm/newton_env.sh

MANIFEST="runs/pcp_protocol_gate/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: $MANIFEST missing. Run: sbatch slurm/pcp_01_setup.sbatch" >&2
  echo "(the virtualenv itself comes from: sbatch slurm/01_setup.sbatch)" >&2
  exit 1
fi

echo "[pcp-protocol] row index $SLURM_ARRAY_TASK_ID of manifest $MANIFEST"

srun python scripts/sweep.py \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --reclaim-stale 3600 \
  --run \
  experiment.sampling_device="$COMMSTUDY_DEVICE" \
  experiment.train_device="$COMMSTUDY_DEVICE" \
  experiment.buffer_device="$COMMSTUDY_DEVICE"
