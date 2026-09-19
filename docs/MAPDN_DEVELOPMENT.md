# MAPDN development and engineering smoke

MAPDN development lives in the isolated source snapshot at
`.codex-worktrees/mapdn-development`. Its vendor copy preserves the original
MAPDN implementation and adds the versioned research adapter. These changes do
not establish a confirmed power-grid learning baseline or close D5's scientific
acceptance gate.

The task is `mapdn_voltage_control`. It uses continuous, distributed inverter
actions, the existing BenchMARL MAPPO optimizer, and the existing communication
actor interface. Actor and critic model inputs contain the declared local
`observation` tensors. Network diagnostics travel separately for measurement.

## Data and temporal contract

The real case33 directory contains `model.p`, `pv_active.csv`, `load_active.csv`,
and `load_reactive.csv`. The split manifest validates finite profile values,
synchronized strictly increasing regular timestamps, load profile dimensions
and column order, and each file's SHA256. The model is hashed by the data module;
the simulator validates its feeder structure. A manifest's data path may move,
but changed bytes fail verification.

The default split is chronological 60% training, 20% validation, and 20% test.
Explicit half-open row blocks may be supplied through `build_split_manifest`.
For reset row `r`, the reserved footprint is
`[r - history + 1, r + max_steps + 1)`. Every start in every split includes its
complete history reservation and terminal bootstrap observation inside that
split. History is zero padded at reset; the reservation remains conservative.

The dynamics version is `mapdn_aligned_transition_v1`: the current observation
is solved at the current exogenous row; the action is solved and rewarded on
that row; then exogenous inputs advance and the next observation is solved.
Action-solve and next-row-solve failures are distinct recorded diagnostics.
An unsolvable fixed reset fails once and does not silently select another row.

Raw measurement-noise standard deviations and inverter ratings use the training
block only. Ratings are `1.2 * training maximum PV`, after declared PV scaling;
excess held-out active generation is recorded as curtailment. Observation
normalization is fitted separately on training observations, saved once, and
bound to the manifest SHA. Constant features receive scale one. Held-out
evaluation never updates these statistics. Measurement noise is a declared
task setting independent of whether policy actions are random or deterministic.

## Bounded real-data smoke

Run from the isolated snapshot after coordinating the shared CPU budget:

```bash
PYTHONPATH=src:vendor/mapdn-source:.deps \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/Users/akki/Desktop/Thesis/marl-comm/.venv/bin/python scripts/run_mapdn_smoke.py \
  --data-path /Users/akki/Desktop/Thesis/data/mapdn/case33 \
  --out-dir results/mapdn_case33_engineering_smoke_001 \
  --max-steps 16 --frames 64 \
  --normalization-episodes 2 --evaluation-episodes 2 --seed 0
```

The output directory must be new. `--frames` must be a positive multiple of 64.
The recorded defaults use one CPU collector environment, one 64-frame MAPPO
batch, one minibatch pass, `comm_identity`, deterministic held-out actions, and
measurement noise disabled. Validation uses its separate temporal split. Native
checkpoints are retained at the requested frame count and at the end, with no
retention cap. The managed final actor checkpoint is also retained.

Before fitting, the script declares equally spaced reset rows and fixed seeds
for a small training normalization bank and a separate validation bank. A zero-action
policy collects normalization observations from the training bank only. The
initial actor and final actor are evaluated on the identical validation bank. Those
measurements preserve training RNG streams. The saved final actor is rebuilt
using strict source/runtime/task-contract checks and evaluated on the same bank
again. The test split is reserved for subsequent prespecified scientific
confirmation and is not evaluated by this smoke. Exact parameter-state equality
and actual action/reward/diagnostic
trajectory equality are required for the engineering replay check.

The artifact directory contains:

- `split_manifest.json`, `normalizer.json`, and `declared_banks.json` for data,
  training statistics, normalization and fixed episode provenance.
- `normalization_collection.json`, `heldout_before.json`, `heldout_after.json`,
  and `heldout_reloaded.json`, with every transition and terminal diagnostic.
- `training_transitions.jsonl` and `validation_evaluations.json`, preserving
  collector diagnostics and validation episode-completion counts.
- `runs/mapdn_engineering_smoke/...`, containing managed status, resolved
  configuration, source/runtime metadata, metrics, and retained checkpoints.
- `report.json`, with parameter-change checks, source/data/runtime provenance,
  throughput, strict reload results, and observed before/after domain changes.
  An interrupted or failed attempt additionally preserves `failure.json`.

Domain summaries count each environment transition once; repeated per-agent
copies are transport fields. Solver-failure terminal transitions remain in the
failure, reward, and feasibility denominators. Physical voltage/line-loss/reactive-power
means include only transitions with `voltage_metrics_valid=1`; their separate
denominators are saved, and missing means are null. The report records
improvements, unchanged metrics, and regressions
as observed. `scientific_learning_confirmed` is always false for this command.

## Remaining scientific work

A passing smoke proves bounded execution, optimization, measurement transport,
checkpoint retention, and replay on the declared bank. D5 still requires a
separate prespecified real-data learning confirmation with a justified horizon,
training-only calibration, held-out domain metrics, adequate episode/seed banks,
and demonstrated improvement. Keep previous failed smoke attempts and their
artifacts; change the output path when an implementation fix requires another
attempt.

The original PCP comparison's frozen inputs, pending allocation, and historical
evidence remain governed by the main `agents.md` and its D4 execution records.
Copying these isolated implementation files into that frozen package requires
the active source binding to have ended first.
