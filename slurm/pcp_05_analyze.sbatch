#!/bin/bash
#SBATCH --job-name=commstudy_pcp_analyze
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/pcp_analyze_%j.out
#SBATCH --error=slurm/logs/pcp_analyze_%j.err
#
# STEP 5 of 5 for Predator-Capture-Prey. Turn finished PCP runs into results.
#
#     sbatch slurm/pcp_05_analyze.sbatch
#
# Or chain it so it starts the moment training ends:
#
#     sbatch --dependency=afterany:<main_jobid>:<ablation_jobid> \
#       slurm/pcp_05_analyze.sbatch
#
# Run it after the pilot too -- that is how you read the pilot before deciding
# whether the main comparison's budget is defensible. Suites with no runs yet
# are skipped, so this is safe at any point in the study.
#
# For every PCP suite that has runs, this does four passes:
#   1. status sync   -- reclaim rows abandoned by preempted/killed workers so
#                       they are reported as failed rather than sitting at
#                       "running" forever
#   2. protocol gate -- non-finite scan + TanhNormal saturation audit of each
#                       frozen actor
#   3. saliency      -- paired evaluation with and without communication
#   4. aggregate     -- CSVs, plots, and REPORT.md
#
# Saliency is the metric that matters most on this task. PCP was added on the
# argument that a predator's view of the prey is worth transmitting to a
# predator that has none; saliency is what tests that claim, rather than message
# counts, which rise whether or not the channel carries anything useful. The
# MLP and Identity rows must score exactly 0.0 -- that is the control value and
# a live wiring check.
#
# Safe to re-run at any time, including while training is still going: it only
# reads run directories and rewrites results/.

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
# shellcheck disable=SC1091
source slurm/newton_env.sh

EPISODES="${COMMSTUDY_SALIENCY_EPISODES:-32}"

SUITES=(
  pcp_identity_pilot
  pcp_comm_main
  pcp_comm_stage2_message_dim
  pcp_comm_stage2_dropout
  pcp_comm_stage3_rounds
  pcp_comm_stage3_heads
  pcp_comm_stage4_graph_topology
  pcp_comm_stage4_sender_budget
  pcp_comm_stage4_self_communication
)

for SUITE in "${SUITES[@]}"; do
  RUNS="runs/$SUITE"
  RESULTS="results/$SUITE"
  if [[ ! -d "$RUNS" ]]; then
    echo "[pcp-analyze] skip $SUITE (no runs yet)"
    continue
  fi
  echo
  echo "=============================================================="
  echo "[pcp-analyze] $SUITE"
  echo "=============================================================="
  mkdir -p "$RESULTS"

  echo "[pcp-analyze] --- status ---"
  python scripts/sweep.py --manifest "$RUNS/manifest.csv" --reclaim-stale 3600 --status

  # || true: a suite with zero finished checkpoints must not abort the loop and
  # lose analysis for every later suite.
  echo "[pcp-analyze] --- protocol gate audit ---"
  python scripts/audit_runs.py "$RUNS" \
    --out "$RESULTS/${SUITE}_run_audit.csv" || true

  echo "[pcp-analyze] --- communication saliency ---"
  python scripts/saliency.py "$RUNS" \
    --episodes "$EPISODES" --seed 0 \
    --out "$RESULTS/${SUITE}_saliency.csv" || true

  echo "[pcp-analyze] --- aggregate and report ---"
  python scripts/analyze.py "$RUNS" --results-dir "$RESULTS"
  python scripts/report.py "$RUNS" --results-dir "$RESULTS"
done

echo
echo "=============================================================="
echo "Done. Read the reports:"
for SUITE in "${SUITES[@]}"; do
  [[ -f "results/$SUITE/REPORT.md" ]] && echo "  results/$SUITE/REPORT.md"
done
echo "=============================================================="
