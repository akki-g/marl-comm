#!/bin/bash
#SBATCH --job-name=commstudy-analyze
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/analyze/%x_%j.out
#SBATCH --error=slurm/logs/analyze/%x_%j.err
#
# Post-process one finished suite into publication-ready artifacts.
#
#     sbatch --dependency=afterany:<training_job_id> \
#       slurm/analyze_suite.sbatch simple_spread_comm_v2
#
# Runs three passes in order:
#   1. status sync  -- reclaim rows abandoned by preempted/killed workers
#   2. saliency     -- reload each frozen actor and measure channel benefit
#   3. analyze      -- aggregate CSVs, plots, and REPORT.md
#
# Saliency needs the GPU because it replays real policies; it is cheap
# (2 x 32 short episodes per run) but should match the training device so the
# measured policy behaviour is the one that was trained.

set -euo pipefail

SUITE_ID="${1:?usage: analyze_suite.sbatch <suite_id> [saliency_episodes]}"
EPISODES="${2:-32}"

source "$(dirname "${BASH_SOURCE[0]}")/newton_env.sh"

RUNS="runs/$SUITE_ID"
RESULTS="results/$SUITE_ID"
mkdir -p "$RESULTS" slurm/logs/analyze

echo "[analyze] === status sync ==="
python scripts/sweep.py --manifest "$RUNS/manifest.csv" --reclaim-stale 3600 --status

echo "[analyze] === communication saliency ==="
python scripts/saliency.py "$RUNS" \
  --episodes "$EPISODES" \
  --seed 0 \
  --out "$RESULTS/${SUITE_ID}_saliency.csv"

echo "[analyze] === aggregate, plots, report ==="
python scripts/analyze.py "$RUNS" --results-dir "$RESULTS"
python scripts/report.py "$RUNS" --results-dir "$RESULTS"

echo "[analyze] artifacts in $RESULTS"
ls -1 "$RESULTS"
