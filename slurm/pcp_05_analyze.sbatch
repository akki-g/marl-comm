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
# The default suite family is corrected_v1 confirmation/main/ablations.
# Retired calibration/comparison suites remain available explicitly:
#     sbatch slurm/pcp_05_analyze.sbatch --historical
# Historical mode reads saved metrics, without replaying policies under the
# corrected task semantics. See docs/PCP_PROTOCOL.md before training or review.
#
# For every PCP suite that has runs, this does four passes:
#   1. status sync   -- reclaim rows abandoned by preempted/killed workers so
#                       they are reported as failed rather than sitting at
#                       "running" forever
#   2. health audit -- non-finite scan + TanhNormal saturation audit of each
#                       frozen actor
#   3. saliency      -- paired evaluation with and without communication
#   4. aggregate     -- CSVs, plots, and REPORT.md
#
# Reliance uses paired complete episodes; influence uses identical recorded
# inputs. An Identity null checks the instrument, while a reliance effect does
# not by itself establish task-level communication necessity.
#
# Status sync and saliency write managed run artifacts; aggregate/report write
# derived results. This script is not read-only and does not approve a protocol.

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
# shellcheck disable=SC1091
source slurm/newton_env.sh

EPISODES="${COMMSTUDY_SALIENCY_EPISODES:-32}"
HISTORICAL=false
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--historical" ) ]]; then
  echo "Usage: sbatch slurm/pcp_05_analyze.sbatch [--historical]" >&2
  exit 2
fi
if [[ "${1:-}" == "--historical" ]]; then
  HISTORICAL=true
  SUITES=(
    pcp_protocol_gate
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
else
  SUITES=(
    pcp_candidate_confirmation_corrected_v1
    pcp_comm_main_corrected_v1
    pcp_comm_stage2_message_dim_corrected_v1
    pcp_comm_stage2_dropout_corrected_v1
    pcp_comm_stage3_rounds_corrected_v1
    pcp_comm_stage3_heads_corrected_v1
    pcp_comm_stage4_graph_topology_corrected_v1
    pcp_comm_stage4_sender_budget_corrected_v1
    pcp_comm_stage4_self_communication_corrected_v1
  )
fi
AUDIT_ARGS=(--group adversary --episodes "$EPISODES")
if [[ "$HISTORICAL" == true ]]; then AUDIT_ARGS+=(--no-policy); fi
FAILURES=0
FAILED_STAGES=""
shopt -s nullglob

record_failure() {
  FAILURES=$((FAILURES + 1))
  FAILED_STAGES+="  ${SUITE}: $1"$'\n'
  echo "[pcp-analyze] FAILED ${SUITE}: $1; continuing with independent suites" >&2
  echo "[pcp-analyze] existing reports for ${SUITE} may belong to an earlier invocation" >&2
}

for SUITE in "${SUITES[@]}"; do
  RUNS="runs/$SUITE"
  RESULTS="results/$SUITE"
  if [[ ! -d "$RUNS" ]]; then
    echo "[pcp-analyze] skip $SUITE (no runs yet)"
    continue
  fi
  MANAGED_METADATA=("$RUNS"/*/metadata.json)
  if [[ ${#MANAGED_METADATA[@]} -eq 0 ]]; then
    echo "[pcp-analyze] skip $SUITE (planned but no managed runs have started)"
    continue
  fi
  echo
  echo "=============================================================="
  echo "[pcp-analyze] $SUITE"
  echo "=============================================================="
  if ! mkdir -p "$RESULTS"; then
    record_failure "create results directory"
    continue
  fi

  echo "[pcp-analyze] --- status ---"
  if ! python scripts/sweep.py --manifest "$RUNS/manifest.csv" --reclaim-stale 3600 --status; then
    record_failure "status synchronization"
    continue
  fi

  # Discover completed IDs explicitly. An unfinished row cannot turn a valid
  # completed-row audit into an empty-checkpoint failure. Malformed status is
  # still an error, not evidence that no run has completed.
  if ! COMPLETED_ROWS="$(python - "$RUNS" <<'PY'
import json
import sys
from pathlib import Path

for metadata in sorted(Path(sys.argv[1]).glob("*/metadata.json")):
    status = json.loads((metadata.parent / "status.json").read_text())
    if status.get("status") == "completed":
        print(metadata.parent.name)
PY
)"; then
    record_failure "completed-run discovery"
    continue
  fi

  if [[ -z "$COMPLETED_ROWS" ]]; then
    echo "[pcp-analyze] skip health/saliency for $SUITE (no completed runs); report current statuses"
  else
    RUN_ARGS=()
    MISSING_CHECKPOINT=false
    while IFS= read -r RUN_ID; do
      if [[ -n "$RUN_ID" ]]; then
        RUN_ARGS+=(--run-id "$RUN_ID")
        if [[ "$HISTORICAL" == false && ! -f "$RUNS/$RUN_ID/checkpoints/policy_state.pt" ]]; then
          echo "[pcp-analyze] missing completed checkpoint: $SUITE/$RUN_ID" >&2
          MISSING_CHECKPOINT=true
        fi
      fi
    done <<< "$COMPLETED_ROWS"
    if [[ "$MISSING_CHECKPOINT" == true ]]; then
      record_failure "completed checkpoint coverage"
      continue
    fi

    echo "[pcp-analyze] --- health audit (does not approve a launch) ---"
    if ! python scripts/audit_runs.py "$RUNS" \
      "${AUDIT_ARGS[@]}" "${RUN_ARGS[@]}" \
      --out "$RESULTS/${SUITE}_run_audit.csv"; then
      record_failure "completed-policy health audit"
      continue
    fi

    if [[ "$HISTORICAL" == true ]]; then
      echo "[pcp-analyze] historical mode: no checkpoint replay or new saliency measurement"
    else
      echo "[pcp-analyze] --- communication reliance and same-input influence ---"
      if ! python scripts/saliency.py "$RUNS" \
        "${RUN_ARGS[@]}" --episodes "$EPISODES" --seed 0 \
        --out "$RESULTS/${SUITE}_saliency.csv"; then
        record_failure "communication reliance/influence"
        continue
      fi
    fi
  fi

  echo "[pcp-analyze] --- aggregate and report ---"
  if ! python scripts/analyze.py "$RUNS" --results-dir "$RESULTS"; then
    record_failure "aggregation"
    continue
  fi
  if ! python scripts/report.py "$RUNS" --results-dir "$RESULTS"; then
    record_failure "report generation"
    continue
  fi
  echo "[pcp-analyze] completed $SUITE: $RESULTS/REPORT.md"
done

echo
echo "=============================================================="
if [[ "$FAILURES" -gt 0 ]]; then
  printf '[pcp-analyze] %s suite stage(s) failed:\n%s' "$FAILURES" "$FAILED_STAGES" >&2
  exit 1
fi
echo "[pcp-analyze] all available suites processed without analysis errors"
echo "=============================================================="
