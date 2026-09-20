# Communication comparison v1 — cluster runbook

This directory implements the fresh **60 PCP + 30 MAPDN** core. It does not submit the historical 30-row PCP manifest. Default submission behavior is dry-run; only `--execute` calls `sbatch`. No full suite was submitted during implementation. CPU is the only supported scientific profile in v1; CUDA remains unqualified and is rejected rather than silently falling back.

Read the [experiment contract](../../docs/experiments/pcp_mapdn_mechanisms_v1/README.md), [audit](../../docs/experiments/pcp_mapdn_mechanisms_v1/IMPLEMENTATION_AUDIT.md), and [validation record](../../docs/experiments/pcp_mapdn_mechanisms_v1/VALIDATION.md). Implementation readiness, local tests, real-task smokes, cluster qualification, and scientific execution are separate statuses.

## Files and stage behavior

| File | Actual entry point / responsibility |
|---|---|
| `common_env.sh` | Strict literal profile parser, exact imports/interpreter, one numerical thread |
| `cluster.env.example` | Required explicit source/run/interpreter/staging/account/resource values |
| `00_probe.sbatch` | Allocation, Slurm version/array limits, hardware/partition/storage discovery |
| `01_bootstrap.sbatch` | Create fresh isolated environments from staged hash-locked Linux wheels; real task preflight |
| `02_smoke.sbatch` | `communication_suite.py smoke`: 12 PCP and 6 real case33 rows |
| `03_qualify.sbatch` | `qualify`: four PCP and two MAPDN full-budget Identity rows |
| `04_train_pcp.sbatch` | `train-row --task pcp`: manifest-derived 0–59 core |
| `05_train_mapdn.sbatch` | `train-row --task mapdn`: manifest-derived 0–29 core |
| `06_close_training.sbatch` | `close-training`: seal valid checkpoints or explicit scientific failures |
| `07_evaluate_pcp.sbatch` | `evaluate-row --task pcp`: deterministic paired 256-episode bank, full replay |
| `08_evaluate_mapdn.sbatch` | `evaluate-row --task mapdn`: paired 16-window bank, full replay |
| `09_analyze.sbatch` | `analyze`: seed-block effects, raw tables, headless plots, incomplete matrix |
| `submit_full.sh` | Dry-run/execute submission DAG, absolute logs, ledger, duplicate lock |
| `status.sh` | Artifact status plus separate `sacct` observations |
| `retry_failed.sh` | Submit only explicit infrastructure-failure retry indices; dry-run default |

`prepare` submits probe → bootstrap/preflight → task smoke arrays → task qualification arrays. Freeze is deliberately a separate inspected step. `main` submits both training arrays; each closure uses `afterany` so failures are inspected. Both evaluation arrays depend on both successful integrity closures (`afterok`). Analysis uses `afterany` on evaluation arrays and can report partial results. `--kill-on-invalid-dep=yes` prevents impossible dependencies from remaining invisible indefinitely. `--no-requeue` is explicit. A scheduler-completed job is never sufficient scientific evidence. [Slurm arrays](https://slurm.schedmd.com/job_array.html), [sbatch dependencies/options/parsable IDs](https://slurm.schedmd.com/sbatch.html), [srun](https://slurm.schedmd.com/srun.html).

## Preparation prerequisites

Supply real authorized account/partition/module values; this repository cannot infer them from an old `normal`/GPU script. Newton has heterogeneous hardware; its public inventory does not establish this account's entitlements or available modules. [Newton resource page](https://arcc.ist.ucf.edu/index.php/resources/newton/about-newton).

Prepare an immutable **source snapshot including all new untracked implementation files**. Do not copy only the baseline commit. A simple source-only export, run from this development checkout with its existing interpreter, is:

```bash
export SNAPSHOT=/absolute/new/source-snapshot
.venv/bin/python - "$SNAPSHOT" <<'PY'
import shutil, sys
from pathlib import Path
from commstudy.experiments.mechanism_suite import ROOT, inventory
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
for name in inventory():
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
shutil.copytree(ROOT / 'docs', out / 'docs')
shutil.copy2(ROOT / 'agents.md', out / 'agents.md')
PY
```

Copy that directory to shared cluster storage; use the same absolute snapshot path at planning and execution. Inventory excludes runs, logs, caches, virtual environments and historical result trees. Keep queued workers' source unchanged. A changed source/config/runtime requires a new plan and qualification, not an old approval edit. Operational artifact moves require an explicit preserved relocation mapping; v1 intentionally fails closed on unrecorded path changes.

Stage a **Linux** wheelhouse and a complete pip requirements lock with hashes for the selected Python/architecture. It must include compatible pinned BenchMARL 1.5.2, TorchRL 0.11.x, TensorDict 0.11.x, Torch, VMAS, NumPy 2.0–<2.3, pandas 2.2–<3, pandapower 3.3.0, Gymnasium >=1.1,<2 (1.0 is incompatible with this TorchRL backend), PettingZoo, all transitive dependencies, and build tooling (`pip`, `setuptools`, `wheel`). Add pytest and matplotlib for validation/analysis. Resolve/download on a permitted networked Linux staging host; **do not use the macOS uv.lock as a Linux wheel lock**. Bootstrap uses `--no-index --require-hashes`; there is no silent network or version fallback. Each installed runtime records package versions, distribution RECORD identities, Torch build configuration and import origins. Missing/incompatible wheels are explicit setup failures.

Place real case33 `model.p`, `pv_active.csv`, `load_active.csv`, `load_reactive.csv` and its attribution/download receipt in a shared data directory. The existing downloader can be used on a permitted staging host (inspect `python scripts/fetch_mapdn_data.py --help`); workers never download data. No synthetic grid may be substituted for real case33. Add all accessible historical evidence roots in the suite YAML **before source snapshot/plan generation**. The audit always also scans source `runs/` and `results/`, but those directories are normally excluded from snapshots, so explicit shared historical roots matter.

You need a management interpreter able to import this project's pinned configuration dependencies to generate a plan and freeze it. This may read an existing environment; installation only occurs into the two new paths in the profile. `COMM_SUITE_CONTROL_PYTHON` selects that interpreter for local CLI/status wrappers. Dry-run/submission and profile parsing themselves are stdlib-only and can use the system Python. No installation modifies preserved `.venv`, `.venv-newton`, `.resolved_env` or a historical MAPDN environment.

## Copy-paste command sequence

Replace the five absolute paths below, then edit the copied profile's account/partition/modules/staging/new-venv paths. The source root and run root in the profile must equal these values. `MIN_FREE_GIB=20`, 8 GiB RAM, two allocated CPUs, two concurrent rows per task and two-day task walltime in the example are **unmeasured requests**, not a runtime guarantee. Check actual quotas and MaxArraySize in the probe and adjust requests based on smoke/qualification measurements. Numerical threads remain one regardless of allocated CPUs.

```bash
export SOURCE=/absolute/read-only/source-snapshot
export STUDY=/absolute/new/study-root
export PROFILE=/absolute/communication-v1-cluster.env
export DATA=/absolute/data/case33_3min_final
export PLAN_PYTHON=/absolute/management-environment/bin/python
export COMM_SUITE_CONTROL_PYTHON="$PLAN_PYTHON"
export PYTHONPATH="$SOURCE/src:$SOURCE/scripts:$SOURCE/vendor/mapdn-source"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$SOURCE"
cp slurm/communication_comparison_v1/cluster.env.example "$PROFILE"
# Edit PROFILE with actual absolute paths and authorized cluster resource values.

"$PLAN_PYTHON" scripts/communication_suite.py plan \
  --suite configs/experiments/pcp_mapdn_mechanisms_v1.yaml \
  --out "$STUDY" --data-path "$DATA"

# Inspection only: zero scheduler submissions.
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase prepare --run-root "$STUDY" --profile "$PROFILE"

# Explicitly authorized preparation: real smokes and six qualification policies.
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase prepare --run-root "$STUDY" --profile "$PROFILE" --execute

# After all preparation jobs finish, inspect row receipts, validation curves,
# resource measurements and the preflight data/history/contracts. Then freeze.
"$PLAN_PYTHON" scripts/communication_suite.py freeze --run-root "$STUDY"

# Inspect all 90 training rows and the closure/evaluation/analysis DAG.
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase main --run-root "$STUDY" --profile "$PROFILE"

# This is the later, explicit authorization to submit the full scientific suite.
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase main --run-root "$STUDY" --profile "$PROFILE" --execute

bash slurm/communication_comparison_v1/status.sh --run-root "$STUDY"
```

All command flags above are implemented and covered by CLI/shell tests. The handoff does **not** execute either submission command. `freeze` rejects local-only preflight, missing/failed smokes, incomplete qualification, source/runtime/bank changes, and modified retained evidence. It never requires communication to outperform a control. A still-improving curve means budget adequacy is unresolved. Amend infeasible budgets for the whole comparison before collection; never secretly truncate a method or tune against test outcomes.

For single-task submissions, use `--task pcp` or `--task mapdn` with `--phase main`. This runs that task's training/closure and a partial report. Test arrays deliberately wait for the separate `--phase evaluate` after **both** core closures exist:

```bash
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase main --task pcp --run-root "$STUDY" --profile "$PROFILE"
# Add --execute after inspection; submit mapdn separately in the same way.
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase evaluate --run-root "$STUDY" --profile "$PROFILE"
```

Optional supplements must be declared at plan generation, e.g. `plan ... --include capacity,mapdn_topology`. Pass the same `--include capacity,mapdn_topology` to `submit_full.sh` for an explicit consistency check. It never adds rows to an already frozen plan. The four `configs/sweeps/*mechanisms*`, `mapdn_topology_v1.yaml` and `local_capacity_sensitivity_v1.yaml` files are strict suite allocation descriptors accepted with `plan --sweep PATH`, not legacy `sweep.py` inputs. They retain the complete core and select only the named optional addition; task selection is a submission concern.

## Failures, monitoring and retries

Receipts distinguish `completed`, `scientific_failure` (numerical divergence), `integrity_failure`, `infrastructure_failure`, `interrupted_or_running`, and missing. Failed attempts/logs/checkpoints remain. Scientific failure stays in the planned denominator and may permit an explicitly incomplete cohort closure; corruption/replay mismatch and unresolved ownership block closure. Failed rows emit `not_evaluable` instead of invented test scores.

```bash
"$PLAN_PYTHON" scripts/communication_suite.py status --run-root "$STUDY" --scheduler
"$PLAN_PYTHON" scripts/communication_suite.py retry-plan \
  --run-root "$STUDY" --task pcp --stage main --kind runs \
  --reason 'Node failure confirmed in Slurm accounting; restart same settings from initialization'
# Set RETRY to the exact path printed by retry-plan; review its eligible indices.
export RETRY=/absolute/study-root/retry_plans/printed-id.json
bash slurm/communication_comparison_v1/retry_failed.sh \
  --run-root "$STUDY" --profile "$PROFILE" --task pcp --retry-plan "$RETRY"
# Add --execute only after inspection. Earlier attempts stay immutable.
```

`retry-plan` is nonexecuting and permits only recorded infrastructure failures. It binds the prior receipt's hash and cannot be reused to select the best attempt. For a node death/SIGKILL with no final receipt, reconcile the exact recorded scheduler owner first:

```bash
"$PLAN_PYTHON" scripts/communication_suite.py reconcile \
  --run-root "$STUDY" --task pcp --stage main --kind runs --index 7 \
  --reason 'sacct confirms the recorded array element timed out'
```

Reconciliation refuses running/unknown jobs and plain `FAILED` states without a known infrastructure diagnosis. It accepts terminal timeout/node failure/OOM/preemption/cancellation evidence, retains the stale ownership record, and never reclaims by timestamp. Local abandoned processes require manual audit; this scheduler recovery path does not guess their state.

After retries finish, explicitly resubmit the still-missing closure and then evaluation stages. Review dry-runs first:

```bash
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase close --task pcp --run-root "$STUDY" --profile "$PROFILE"
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase evaluate --run-root "$STUDY" --profile "$PROFILE"
# Evaluation infrastructure retries use retry-plan --kind evaluation.
# After those retries, request a new preserved analysis directory:
bash slurm/communication_comparison_v1/submit_full.sh \
  --phase analyze --run-root "$STUDY" --profile "$PROFILE"
```

A nonzero sbatch submission aborts further dependent submissions and preserves prior job IDs in `submissions.jsonl`. Inspect those IDs and use the printed remaining commands or separate recovery phases; do not resubmit an overlapping full DAG. The ledger also records unparseable scheduler responses for manual reconciliation. `sbatch --parsable` cluster suffixes are parsed separately. Log directories are absolute and precreated; arrays use `%A_%a`, singleton jobs `%j`. There is no shell `eval` or executable profile sourcing. Workers propagate Python exit status through `srun --kill-on-bad-exit=1`. Signals write best-effort interruption receipts; SIGKILL and node loss still require accounting reconciliation.

Failed preflight partial preparations remain preserved. Correct missing staging prerequisites and create a new study root; do not edit a partly materialized bank or approve it manually. A bootstrap-created environment with a successful matching ownership receipt can be reused read-only. Incomplete or unowned existing environments are refused and require a new environment path. This preserves both original environments and failure evidence.

## Artifacts and resources

`preparation/` contains exact specs/manifests, source inventory, runtime/data/history contracts, training-only normalizer and prospective banks. `gates/` contains per-task preflight, main approval and training closures. `runs/<task>/<run_id>/attempt_NNN/` contains the existing managed `RunContext` tree, validation points, strict native/managed parity, paired development result, and a read-back-verified archive. `evaluation/` preserves both complete final paired runs and strict reconstruction. `analysis/<unique-id>/` contains outcomes, seed contrasts/interactions, capacity/payload/physical tables, learning curves, figures and the failure/missing matrix. No empty scientific cell is filled with synthetic data.

Per-row receipts measure total wall time, development-evaluation time, archive time, peak RSS with platform units, retained bytes and file count. Native runtime metrics retain collection/optimization and ordinary evaluation timing. Final evaluation receipts separately measure full paired replay time. Review representative rows and apply an explicit operational margin, e.g. 1.5× measured resource use. The example requests are not forecasts; unmeasured Linux training/evaluation/serialization/storage/quota fields remain unmeasured until target execution. Do not extrapolate old Simple Spread GPU throughput to MAPDN's CPU power-flow solver.
