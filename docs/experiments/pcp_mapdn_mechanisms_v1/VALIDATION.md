# Implementation validation — September 19–20, 2026

Implementation is complete for the requested CPU suite. Local checks and real PCP engineering smokes passed. Target Linux/Slurm verification, real case33 smokes, full-budget qualification, and the scientific cohort have **not** run. There is no main approval and no scientific result from this suite. Final scientific test trajectories remained unopened.

## Source and generated artifacts

Baseline Git HEAD remains `9eb028ec5ad66bf4dd105623206ffd1c8da7561b`. No commit or push was made. The final executable source inventory SHA256 is `8b0024963a6a612184fc6bf0cc809cc3199c5b8b208626fe47ba52a70b83feec`. This inventory includes code, tests, configs, the vendored adapter, package metadata, and this suite's Slurm directory. It excludes documentation outside that directory, environments, caches, and results. The exact inventory is in each plan's `preparation/source_inventory.json`.

The final core plan is at `/Users/akshatguduru/Desktop/Thesis/communication_comparison_v1_final_plan_20260920_v3`. Its `preparation/plan.json` receipt has identity `ddbe257dbdca706a5edaf5e3248be1f48694c84ddeae445c80fc2c13c3ebbbcf`; JSON and CSV manifests contain 60 PCP and 30 MAPDN main rows, 4+2 qualification rows, and 12+6 smoke rows. This is an inspection plan with unavailable real-data paths, not a cluster approval. Regenerate a plan on the immutable cluster snapshot with actual absolute paths.

Separate optional previews were generated; their allocation descriptors also passed the actual `--sweep` interface:

| Preview directory under `/Users/akshatguduru/Desktop/Thesis/` | Core | Explicit addition | Total main rows |
|---|---:|---:|---:|
| `communication_comparison_v1_capacity_preview_20260920_v3` | 90 | 15 LocalCapacity64 | 105 |
| `communication_comparison_v1_topology_preview_20260920_v3` | 90 | 10 MAPDN Graph topology | 100 |

Both share the final source identity. They do not amend the 90-row core plan. Earlier development plans remain preserved and are not current approvals. `HANDOFF_FILES.json` enumerates actual created/modified files and content hashes; `READINESS.json` records the execution limits.

## Test evidence

Final new-suite tests cover strict schema/counts/portable configuration identity, source and receipt tampering, string/Path closure reconstruction, deadline boundaries, training-seed statistics, active local capacity, graph adjacency, smoke/qualification/main barriers, failure classification and retries, all six PCP paired instruments, all six synthetic-grid MAPDN paired instruments and optimizer/native-reload paths. Synthetic fixtures exercise code; they provide no case33 scientific evidence.

The retained regression results are:

| Check | Result |
|---|---|
| New suite contract, Slurm, and task tests | 48 passed in the final consolidated run; see `evidence/new_suite_tests.log` |
| Existing tests, first segment | 1,001 passed, 2 skipped, 106 subtests passed; gracefully interrupted at local disk floor |
| Existing tests, remaining 329 items | 328 passed; one new-descriptor routing assertion failed and was corrected |
| Corrected historical routing assertion plus six Slurm tests | 7 passed |
| Distinct existing tests after correction | 1,330 passed, 2 skipped, plus 106 subtests |
| Ruff on every changed/new Python file | Passed |
| Python compilation, shell syntax, `git diff --check` | Passed |
| Real master CLI preparation/main dry-runs | Passed using system Python; zero scheduler calls and no submission ledger |
| Empty main analysis | Reports all 90 policies missing and `scientific_complete=false` |
| Headless plotting | Four PNGs generated from an explicitly synthetic fixture outside the scientific study |

This is a segmented regression record, not a claim that one uninterrupted invocation passed the entire repository. The original failure log is retained. A final direct CLI check also found a missing optional-statistics field in report generation. The corrected producer now always provides the field and computes optional seed-paired contrasts; 24 contract tests passed, including new empty-study and optional-contrast regressions, before the final consolidated rerun. The failed report directory and original traceback are preserved. Its correction teaches the historical sweep-coverage test to recognize and validate the new explicit descriptor schema and new analyzer; it does not weaken the historical suite checks. Test-only temporary output from the ended first invocation was removed after its summary was recorded to recover disk space; historical results and scientific attempts were preserved.

The local environment was not modified. PettingZoo 1.24.3, Gymnasium 1.2.3, and Farama Notifications 0.0.4 were installed with `uv --target` only into `/tmp/communication-suite-v1-test-dependencies`. Gymnasium 1.0 proved incompatible with the installed TorchRL backend; that temporary target alone was updated. These local dependencies and the preserved macOS environment do not certify a Linux runtime.

Reproduce the new suite tests from the checkout, with those isolated optional dependencies available:

```bash
export PYTHONPATH="/tmp/communication-suite-v1-test-dependencies:$PWD/src:$PWD/scripts:$PWD/vendor/mapdn-source"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/pytest -q src/tests/test_mechanism_suite_contracts.py \
  src/tests/test_mechanism_suite_slurm.py src/tests/test_mechanism_suite_tasks.py
.venv/bin/ruff check src/commstudy/experiments/mechanism*.py \
  src/commstudy/analysis/mechanism*.py scripts/communication_suite.py \
  src/commstudy/communication/graph.py src/commstudy/experiments/bookkeeping.py \
  src/commstudy/experiments/protocols.py src/tests/test_mechanism_suite*.py \
  src/tests/test_slurm_scripts.py
```

Mocked Slurm checks cover dry-run nonexecution, `jobid;cluster` parsing, manifest-sized arrays, dependency wiring, submission failures, duplicate rejection, paths with spaces, and all eight Python workers executed from copied spool locations with a mocked `srun`. Real Slurm, module loading, Linux bootstrap, quota discovery, and ShellCheck remain untested locally.

## Real local PCP measurements

All 12 configured PCP engineering rows completed at 6,000 frames each (72,000 total), covering every method under radius 1.0 and global sensing. Each changed its actor, retained finite parameters/logged values, collected the exact frames/batches, passed paired development replay and strict native actor/critic reload, and read back its archive. Controls passed exact intervention null checks. These are development-bank checks, not held-out performance estimates or evidence of convergence.

Receipts and preserved logs are under `/Users/akshatguduru/Desktop/Thesis/communication_comparison_v1_local_20260919_b`. This complete smoke set binds development source inventory `e42660a5fd4825e7de32748f4482e7572eb9bbb032a7f75eda618a4323367725`. Later changes hardened provenance, reporting, scheduler probing and tests. A subsequent Identity 6,000-frame smoke also passed under `/Users/akshatguduru/Desktop/Thesis/communication_comparison_v1_handoff_20260920` in 46.58 seconds. Neither earlier source receipt is relabeled as a final-source cluster qualification. The final-source test run checks all six task paths; fresh target smoke/qualification receipts are still mandatory.

| Measured local quantity for the 12-row set | Value |
|---|---:|
| End-to-end elapsed time per row | 32.83–42.03 seconds |
| Sum of row elapsed times | 436.88 seconds |
| Peak RSS per row | 278.31–355.77 MiB |
| Sum of retained bytes | 178,635,110 bytes (170.36 MiB) |
| Sum of retained files | 2,308 |

`evidence/pcp_smoke_resources.json` binds these measurements to all 12 receipt hashes and the development source. Regression tests ran concurrently with parts of this measurement. The figures are observed macOS smoke costs, not full-budget or cluster forecasts. Full 600,000/131,072-frame training, validation, native serialization, full final paired replay, Linux peak memory, storage growth and quota headroom remain unmeasured. The example 8 GiB/two-day/20 GiB-free settings are operational placeholders to revise after target qualification. Numerical computation remains CPU/one thread in this version; CUDA is explicitly rejected as unqualified.

## Remaining target checks and scientific status

Real case33 data are missing: attempted MAPDN preflight failed at `data/case33_3min_final/model.p`. No synthetic dataset was substituted into a scientific plan. Preflight audited 55 accessible historical training metadata files without a new training-seed collision; older bank declarations were unavailable. Their overlap status remains unavailable, and target preflight must audit the mounted historical declarations.

Before main execution the target must establish authorized account/partition/modules, Linux interpreter and complete hashed wheel lock, supported dependencies, compute-allocation imports, hardware/Slurm limits, free space and quota, real case33 data/mapping/physical diagnostics, training-only normalization and bank footprints, all 18 real task smokes, and six full-budget Identity qualifications. Freeze then binds the actual source/runtime/data/banks and retained evidence. A positive communication effect is never a gate.

| Readiness dimension | Status |
|---|---|
| Requested infrastructure implemented | Complete |
| Local contracts/regressions/CLI/mock scheduler | Passed |
| Real local PCP smoke coverage | 12/12 passed on recorded development source; later Identity recheck passed |
| Real case33 smoke coverage | 0/6; blocked by missing data |
| Target cluster/runtime/profile | Not verified |
| Full-budget qualification | 0/6 executed |
| Main approval / ready to submit scientific training | No |
| Scientific main training | 0/90 executed |
| Scientific final test rollouts | 0 |

Use the [cluster runbook](../../../slurm/communication_comparison_v1/README.md) for exact source-copy, staging, preparation, freeze, main, status, retry and recovery commands. No scientific launch, installation into an existing preserved environment, historical result deletion, or repository push occurred during this handoff.
