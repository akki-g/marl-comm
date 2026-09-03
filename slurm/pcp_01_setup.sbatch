#!/bin/bash
#SBATCH --job-name=commstudy_pcp_setup
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/pcp_setup_%j.out
#SBATCH --error=slurm/logs/pcp_setup_%j.err
#
# STEP 1 of 5 for Predator-Capture-Prey. Run once, before any PCP training.
#
#     cd ~/marl-comm
#     sbatch slurm/pcp_01_setup.sbatch
#
# This does NOT build the virtualenv. `slurm/01_setup.sbatch` owns that, and
# building it in two places is how the two copies drift apart. Run that first
# if you have not already; newton_env.sh fails here with instructions if the
# environment is missing or incomplete.
#
# What this does own: writing every PCP manifest, once, so the training arrays
# only ever read them and never race to create them.
#
# Manifests written:
#   runs/pcp_identity_pilot/manifest.csv         2 rows   (pilot)
#   runs/pcp_comm_main/manifest.csv             30 rows   (main comparison)
#   runs/_manifests/pcp_ablations.csv          201 rows   (seven ablations)

set -euo pipefail

cd "${COMMSTUDY_REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
# shellcheck disable=SC1091
source slurm/newton_env.sh

echo "[pcp-setup] === verification ==="
python - <<'PY'
import torch

from commstudy.tasks import resolve_task

task = resolve_task("vmas_predator_capture_prey")
print(f"task           {task.name} ({task.env_name()})")
print(f"landmarks      {task.config['num_landmarks']}")
print(f"predators      {task.config['num_adversaries']}")
print(f"prey           {task.config['num_good_agents']}")
print(f"cuda available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not visible on this compute node. Training would silently "
        "run on CPU. Check the cuda module and --gres before continuing."
    )
PY

echo "[pcp-setup] === building manifests ==="

echo "[pcp-setup] pilot"
python scripts/sweep.py configs/sweeps/pcp_identity_pilot.yaml | tail -2

echo "[pcp-setup] main comparison"
python scripts/sweep.py configs/sweeps/pcp_comm_main.yaml | tail -2

echo "[pcp-setup] ablation study (combined)"
python scripts/sweep.py \
  configs/sweeps/pcp_comm_stage2_message_dim.yaml \
  configs/sweeps/pcp_comm_stage2_dropout.yaml \
  configs/sweeps/pcp_comm_stage3_rounds.yaml \
  configs/sweeps/pcp_comm_stage3_heads.yaml \
  configs/sweeps/pcp_comm_stage4_graph_topology.yaml \
  configs/sweeps/pcp_comm_stage4_sender_budget.yaml \
  configs/sweeps/pcp_comm_stage4_self_communication.yaml \
  --combine-out runs/_manifests/pcp_ablations.csv | tail -2

echo
echo "=============================================================="
echo "PCP setup complete. Next:"
echo "  sbatch slurm/pcp_02_pilot.sbatch            # 2 rows  -- READ THIS FIRST"
echo "  sbatch slurm/pcp_03_main_comparison.sbatch  # 30 rows"
echo "  sbatch slurm/pcp_04_ablations.sbatch        # 201 rows"
echo
echo "PCP has no protocol gate yet. The pilot exists to decide the training"
echo "budget and confirm the predators learn at all before 231 rows commit to"
echo "an uncalibrated horizon. See slurm/README.md, 'Predator-Capture-Prey'."
echo "=============================================================="
