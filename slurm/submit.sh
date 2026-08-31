#!/bin/bash
# Build a suite manifest and submit it as a correctly sized Slurm array.
#
#     bash slurm/submit.sh configs/sweeps/simple_spread_comm_v2.yaml
#     bash slurm/submit.sh configs/sweeps/simple_spread_comm_v2.yaml --chunk 4
#     bash slurm/submit.sh --all                 # every V2 study suite
#     bash slurm/submit.sh <suite.yaml> --dry-run
#
# The array range is derived from the manifest, so it can never drift from the
# number of planned rows. Job name = suite id, which is what makes the log tree
# and `squeue` output readable.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CHUNK=4              # manifest rows per array task
MAX_CONCURRENT=25    # array throttle; keep well under your DPH burn rate
PARTITION="normal"
TIME_LIMIT="08:00:00"
GPUS=1
CPUS=4
MEM="16G"
DRY_RUN=0
DEPENDENCY=""
SUITES=()

ALL_SUITES=(
  configs/sweeps/simple_spread_comm_v2.yaml
  configs/sweeps/simple_spread_comm_v2_stage2_message_dim.yaml
  configs/sweeps/simple_spread_comm_v2_stage2_dropout.yaml
  configs/sweeps/simple_spread_comm_v2_stage3_rounds.yaml
  configs/sweeps/simple_spread_comm_v2_stage3_heads.yaml
  configs/sweeps/simple_spread_comm_v2_stage4_graph_topology.yaml
  configs/sweeps/simple_spread_comm_v2_stage4_sender_budget.yaml
  configs/sweeps/simple_spread_comm_v2_stage4_self_communication.yaml
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)            SUITES=("${ALL_SUITES[@]}"); shift ;;
    --chunk)          CHUNK="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --partition)      PARTITION="$2"; shift 2 ;;
    --time)           TIME_LIMIT="$2"; shift 2 ;;
    --gpus)           GPUS="$2"; shift 2 ;;
    --cpus)           CPUS="$2"; shift 2 ;;
    --mem)            MEM="$2"; shift 2 ;;
    --dependency)     DEPENDENCY="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -h|--help)        sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)               echo "unknown option: $1" >&2; exit 2 ;;
    *)                SUITES+=("$1"); shift ;;
  esac
done

if [[ ${#SUITES[@]} -eq 0 ]]; then
  echo "usage: submit.sh <suite.yaml> [...] | --all   [options]" >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
if [[ -d "${COMMSTUDY_VENV:-$REPO/.venv-newton}" ]]; then
  PYTHON="${COMMSTUDY_VENV:-$REPO/.venv-newton}/bin/python"
fi

for SUITE in "${SUITES[@]}"; do
  [[ -f "$SUITE" ]] || { echo "no such suite: $SUITE" >&2; exit 2; }
  SUITE_ID="$(basename "$SUITE" .yaml)"
  MANIFEST="runs/$SUITE_ID/manifest.csv"

  # Creating the manifest is idempotent for planning purposes: it rewrites the
  # planned rows, and per-run status.json files remain authoritative, so
  # already-completed rows are still skipped at execution time.
  "$PYTHON" scripts/sweep.py "$SUITE" > /dev/null
  ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
  TASKS=$(( (ROWS + CHUNK - 1) / CHUNK ))
  LAST=$(( TASKS - 1 ))

  mkdir -p "slurm/logs/$SUITE_ID"

  ARGS=(
    --job-name="$SUITE_ID"
    --partition="$PARTITION"
    --gres="gpu:$GPUS"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME_LIMIT"
    --array="0-${LAST}%${MAX_CONCURRENT}"
    --output="slurm/logs/$SUITE_ID/%A_%a.out"
    --error="slurm/logs/$SUITE_ID/%A_%a.err"
  )
  [[ -n "$DEPENDENCY" ]] && ARGS+=(--dependency="$DEPENDENCY")

  printf '%-46s rows=%-4s chunk=%-2s tasks=%-4s array=0-%s%%%s\n' \
    "$SUITE_ID" "$ROWS" "$CHUNK" "$TASKS" "$LAST" "$MAX_CONCURRENT"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY RUN: sbatch ${ARGS[*]} slurm/run_suite.sbatch $MANIFEST $CHUNK"
    continue
  fi

  JOB_ID=$(sbatch --parsable "${ARGS[@]}" slurm/run_suite.sbatch "$MANIFEST" "$CHUNK")
  echo "  submitted job $JOB_ID"
  echo "  logs:    slurm/logs/$SUITE_ID/${JOB_ID}_<task>.out"
  echo "  status:  $PYTHON scripts/sweep.py --manifest $MANIFEST --status"
done
