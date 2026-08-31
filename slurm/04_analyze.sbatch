#!/bin/bash
#SBATCH --job-name=commstudy_analyze
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/analyze_%j.out
#SBATCH --error=slurm/logs/analyze_%j.err
#
# STEP 4 of 4. Turn finished runs into results.
#
#     sbatch slurm/04_analyze.sbatch
#
# Or chain it so it starts the moment training ends:
#
#     sbatch --dependency=afterany:<main_jobid>:<ablation_jobid> slurm/04_analyze.sbatch
#
# For every suite that has runs, this does four passes:
#   1. status sync   -- reclaim rows abandoned by preempted/killed workers so
#                       they are reported as failed rather than sitting at
#                       "running" forever
#   2. protocol gate -- non-finite scan + TanhNormal saturation audit of each
#                       frozen actor
#   3. saliency      -- paired evaluation with and without communication
#   4. aggregate     -- CSVs, plots, and REPORT.md
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
  simple_spread_comm_v2
  simple_spread_comm_v2_stage2_message_dim
  simple_spread_comm_v2_stage2_dropout
  simple_spread_comm_v2_stage3_rounds
  simple_spread_comm_v2_stage3_heads
  simple_spread_comm_v2_stage4_graph_topology
  simple_spread_comm_v2_stage4_sender_budget
  simple_spread_comm_v2_stage4_self_communication
)

for SUITE in "${SUITES[@]}"; do
  RUNS="runs/$SUITE"
  RESULTS="results/$SUITE"
  if [[ ! -d "$RUNS" ]]; then
    echo "[analyze] skip $SUITE (no runs yet)"
    continue
  fi
  echo
  echo "=============================================================="
  echo "[analyze] $SUITE"
  echo "=============================================================="
  mkdir -p "$RESULTS"

  echo "[analyze] --- status ---"
  python scripts/sweep.py --manifest "$RUNS/manifest.csv" --reclaim-stale 3600 --status

  # || true: a suite with zero finished checkpoints must not abort the loop and
  # lose analysis for every later suite.
  echo "[analyze] --- protocol gate audit ---"
  python scripts/audit_runs.py "$RUNS" \
    --out "$RESULTS/${SUITE}_run_audit.csv" || true

  echo "[analyze] --- communication saliency ---"
  python scripts/saliency.py "$RUNS" \
    --episodes "$EPISODES" --seed 0 \
    --out "$RESULTS/${SUITE}_saliency.csv" || true

  echo "[analyze] --- aggregate and report ---"
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
