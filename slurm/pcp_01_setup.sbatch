#!/bin/bash
# Planning only: use bash slurm/pcp_01_setup.sbatch on a login node.
# This builds inspectable manifests without approving or submitting training.
set -euo pipefail
TASK_REPO="${COMMSTUDY_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$TASK_REPO"
TASK_PYTHON="${COMMSTUDY_PYTHON:-$TASK_REPO/.venv-newton/bin/python}"
if [[ ! -x "$TASK_PYTHON" ]]; then TASK_PYTHON="$TASK_REPO/.venv/bin/python"; fi
"$TASK_PYTHON" scripts/sweep.py configs/sweeps/pcp_candidate_confirmation.yaml
"$TASK_PYTHON" scripts/sweep.py configs/sweeps/pcp_comm_main.yaml
"$TASK_PYTHON" scripts/sweep.py \
  configs/sweeps/pcp_comm_stage2_message_dim.yaml \
  configs/sweeps/pcp_comm_stage2_dropout.yaml \
  configs/sweeps/pcp_comm_stage3_rounds.yaml \
  configs/sweeps/pcp_comm_stage3_heads.yaml \
  configs/sweeps/pcp_comm_stage4_graph_topology.yaml \
  configs/sweeps/pcp_comm_stage4_sender_budget.yaml \
  configs/sweeps/pcp_comm_stage4_self_communication.yaml \
  --combine-out runs/_manifests/pcp_ablations_corrected_v1.csv
printf '%s\n' 'Manifests prepared. No training was submitted or approved.'
