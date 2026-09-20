# Implementation audit — September 19, 2026

Baseline HEAD is `9eb028ec5ad66bf4dd105623206ffd1c8da7561b`, exactly the specification's revision. At intake, the only untracked files were the user-supplied `docs/PCP_MAPDN_FULL_COMPARISON_SPEC.md`, `docs/RESEARCH_REFERENCES.md`, and `docs/references.bib`. Those files, existing environments, all historical run/result directories, old protocols, and old Slurm launchers are preserved. No commit, push, full scientific training, qualification training, or scheduler submission is authorized or performed in this handoff.

The latest September 14 entry in `agents.md` records a completed historical 15-policy MAPDN diagnostic, mixed/inconclusive reliance, and the serialized string-path failure recovered with an external evaluation adapter. It does not establish a six-method, 131,072-frame comparison. Earlier log snapshots remain historical. Current task/model/runner/protocol/replay/retention/evaluation/data interfaces were inspected directly, including the two historical launchers called out in the specification.

Observed workspace: macOS arm64, local `.venv/bin/python` is Python 3.12; Torch reports 2.13.0. BenchMARL imports locally. About 2.6 GiB free was observed at intake. Neither `sbatch` nor `shellcheck` is on PATH. No Newton session, account, partition entitlement, module inventory, Linux wheel lock, target quota, or real case33 `model.p` was available in the repository or sibling MAPDN tree. Historical Simple Spread runs/results are available; complete recent PCP/MAPDN historical banks are not assumed mounted. No held-out scientific trajectories were opened. Preflight audits every accessible bank declaration and training metadata file in the configured historical roots and explicitly reports unavailable history.

The preserved local environment initially lacked importable PettingZoo and Gymnasium. Local synthetic-grid test dependencies are installed only into `/tmp/communication-suite-v1-test-dependencies`, used via explicit PYTHONPATH. This does not certify a Linux runtime or supply real case33 data.

Implementation mapping:

| Requirement | Implementation |
|---|---|
| Explicit six-method schema, 60+30 rows, optional cohorts | `experiments/mechanism_suite.py`; new experiment/protocol/sweep configs |
| New PCP launch authorization | Narrow `kind=communication_mechanisms_v1` branch in `protocols.validate_protocol_launch`; all other bindings retain the historical validator |
| Managed training and replay/input guards | Existing `runner.run_managed_experiment`, `TrainingHealthCallback`, `RunContext`, existing metrics callback |
| Real data, training-only normalization, topology mapping | `experiments/mechanism_data.py`, reusing `mapdn_data`, `next_phase.select_nonoverlapping_rows`, and `_roll_bank` |
| Qualification, native checkpoints, strict reconstruction, archives | `experiments/mechanism_execution.py`, reusing diagnostics and native-state/archive helpers |
| General PCP paired live/severed instrument | `analysis/mechanism_suite.py`, reusing domain metrics, fresh environment, RNG isolation, paired validation, action metrics |
| MAPDN paired instrument | Existing `analysis/mapdn_paired.py`, including full replay, real physical denominators, and unchanged critic contract |
| JSON string-path failure | `bookkeeping.read_json` normalizes path-like inputs; suite's serialized training-summary-to-reconstruction route is regression-tested |
| Seed-block analysis and figures | `analysis/mechanism_suite.py`, `analysis/mechanism_report.py` |
| CLI and scheduler | `scripts/communication_suite.py`, `experiments/mechanism_slurm.py`, all named scripts in `slurm/communication_comparison_v1/` |

Two integration findings required care. PCP's tiny optimized-but-overridden prey MLP stores TorchRL `torch.Size`/`None` metadata as well as tensors, so the MAPDN tensor-only actor hash cannot be applied unchanged. The new suite hash and native parity checker retain every group and validate this metadata. Slurm executes spool copies of sbatch scripts, so workers locate `common_env.sh` through the submitted absolute `--chdir`, rather than assuming the spool directory contains repository helpers.

The optional Graph `fixed_mask` is a configuration-bound restriction intersected with runtime masks; it adds no trainable parameters and does not change existing Graph behavior when absent. It prevents an all-ones adapter mask from overriding the electrical supplement. Physical feeder topology and actor communication adjacency remain separate artifacts.

See `VALIDATION.md` for measured checks and the final source inventory identity. No result is labeled scientific merely because its class exists, its tests pass, or Slurm exits successfully.
