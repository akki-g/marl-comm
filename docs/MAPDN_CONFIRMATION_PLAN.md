# MAPDN case33 bounded learning confirmation

**Status: prospective settings and criteria approved by the coordinating agent;
implementation and frozen preparation require independent review before launch.**
This records agent review, not an additional human approval. The implementation
agent must bind the reviewed plan, exact resolved
configurations, source/runtime fingerprints, data manifest, normalization
artifact, episode banks, and allocation before launching either row. Revisions
before that binding must remain explicit; changes after launch require a new
protocol and preservation of the original attempt.

This is a two-seed, bounded domain-learning confirmation. It does not establish
statistical significance, broad seed generalization, communication superiority,
or equivalence to the upstream MAPDN paper. An engineering smoke is separate
evidence and cannot satisfy this plan's learning gate.

## Fixed allocation and optimization

| Setting | Fixed value |
|---|---|
| Task and feeder | `mapdn_voltage_control`, original real case33 data |
| Dynamics | `mapdn_aligned_transition_v1` |
| Training seeds | `30`, `31`; no replacements |
| Training frames | Exactly `8192` per seed |
| Episode horizon / observation history | `239` / `1` |
| Algorithm | Existing BenchMARL MAPPO implementation |
| Actor | `comm_identity`; shared local encoder, hidden width 128, two encoder layers, Tanh |
| Communication | Parameterless `IdentityComm` |
| Critic | Existing centralized BenchMARL MLP, widths `[128,128]`, Tanh |
| Devices and concurrency | CPU; one environment per collector; one active confirmation worker |
| Threads | One Torch/BLAS thread per worker |
| Collection batch | `256` environment frames |
| Minibatch / passes | `64` frames / `4` passes per collection batch |
| Training iterations | `32` collection/optimization iterations per seed |
| Learning rate / Adam epsilon | `5e-5` / `1e-6` |
| Discount / GAE lambda | `0.99` / `0.9` |
| PPO clipping / entropy coefficient | `0.2` / `0.0` |
| Critic coefficient / loss | `1.0` / `l2` |
| Gradient clipping | Global norm `5.0` |
| Action distribution | Existing tanh-normal; `biased_softplus_1.0` scale mapping |
| Critic sharing / actor sharing | Both enabled |
| Checkpoint policy | Native checkpoints at 4096 and 8192 frames, end checkpoint, no retention cap |
| Model-selection rule | Final 8192-frame actor only; no best-checkpoint selection |

The batch/minibatch/pass values were selected a priori to provide 32 collection
batches and four passes over each batch. They differ from the 64-frame,
one-pass engineering smoke. They received prospective coordinating-agent review
as a new protocol, without selection using confirmation test outcomes. All remaining
algorithm and model fields must appear in the bound resolved configuration;
implicit mutable library defaults are insufficient provenance.

The fixed task configuration uses distributed inverter control; `l1` voltage
barrier; voltage weight `1.0`; reactive-power weight `0.1`; no line-loss reward
term; voltage bounds `[0.95,1.05]`; PV and demand scales `1.0`; action scale
`0.8`; action bias `0.0`; measurement noise disabled; and zero reactive action
at reset. Local observation fields are `pv`, `demand`, `reactive`, `vm_pu`, and
`va_degree`. Diagnostics are excluded from both model inputs. Newton-Raphson
uses maximum 20 iterations, tolerance `1e-8` MVA, `numba=False`, `init=auto`,
and voltage-angle calculation enabled.

## Data, normalization, and fixed banks

Use all four original case33 files with their download receipt and SHA256
hashes. Validate synchronized regular finite profiles and feeder dimensions.
Create a new horizon-239, history-1 manifest using chronological fractions
`(0.6,0.2,0.2)`, stride one, and the existing explicit full-window contract.
Every reset row reserves its terminal observation inside its split. The
manifest records raw training statistics, declared task/solver settings, and
the imported MAPDN source fingerprint.

PV capacity is `1.2 * maximum training-block PV`, after declared scaling.
Out-of-capability active generation is curtailed and recorded. No test or
validation statistic may determine ratings, noise scales, observation
normalization, reward settings, horizons, or hyperparameters.

Create and hash all three banks before normalization or either training run.
For each split, let its ordered manifest starts be `R`, with length `L`.
Select exactly 16 rows `R[floor(i*(L-1)/15)]`, for `i=0,...,15`. Require distinct
rows and nonoverlapping episode footprints within each bank; insufficient data
is a prelaunch failure. Persist the selected absolute rows, episode IDs, seeds,
split identity, and manifest hash. This is a fixed subset, not the entire
held-out partition.

| Bank | Split | Episodes | Environment seeds | Use |
|---|---|---:|---|---|
| Normalization | Training | 16 | `100000 + i` | Zero-action observations only |
| Development evaluation | Validation | 16 | `200000 + i` | Fixed descriptive initial/final evaluation |
| Confirmation | Test | 16 | `300000 + i` | Fixed initial/final measurement after both trainings |

Environment seeds and row banks are identical for the two training seeds and
for their initial/final actors. Evaluation actions are deterministic. No failed
reset or episode is replaced, skipped, or supplemented.

Fit one shared immutable observation normalizer from the normalization bank's
reset and every next observation, including the terminal observation: exactly
`16 * 240` observation frames, pooled over agents for each feature. Collection
uses the fixed zero-action policy with measurement noise disabled. Require
complete horizon-239 episodes and finite observations. Save population means
and scales, sample count, fit split, manifest SHA, and normalization-collection
provenance. Constant-feature scales are one. Neither training nor evaluation
updates the fitted statistics afterward.

## Execution and test isolation

1. Verify the immutable reviewed packet, source/runtime/data bindings, available
   disk and memory, and the existing shared-workspace CPU allocation. Create new
   output directories. Record the launch time and ordered rows `[30,31]`.
2. Build each seeded experiment and save its initial full actor state before
   the first training collection or optimizer step. Archive the initial state's
   tensor fingerprint, raw checkpoint hash, resolved configuration, and model
   input/output contract. Verify read-back immediately. Preserve this initial
   checkpoint independently of native checkpoint cleanup.
3. Train both original seeds to exactly 8192 frames under the fixed settings.
   Log every transition's row alignment, validity flags, reward and domain
   diagnostics, plus ordinary PPO health and parameter-change measurements.
   Validation is descriptive; it cannot select checkpoints, change settings,
   stop a successful run early, extend its horizon, or replace a seed. Run the
   fixed validation bank at the initial and final actor states; preserve those
   reports separately from the scientific test result.
4. After **both** full training rows, their initial/final checkpoint archives,
   and training-integrity audits are complete, close training and strictly
   reload each initial and final actor. Only then run the previously serialized
   test bank. No test simulator rollout or test metric inspection occurs before
   this point. Test metadata and file-integrity validation are allowed earlier.
   The implementation also defers initial/final validation-bank rollouts until
   this measurement stage; unbound online evaluation is disabled.
5. Use the same exact test rows, environment seeds, deterministic action mode,
   and normalization artifact for each paired initial/final evaluation. Save
   every action, reward, transition diagnostic, reset identifier, terminal flag,
   and complete-episode count. Verify strict source/runtime/task compatibility
   and exact reload of actor state. Repeat the final actor's first two declared
   validation episodes under the same runtime to establish exact
   action/reward/diagnostic replay, keeping both records. Their selection is
   fixed before training; test-bank episodes are each evaluated once per actor.
6. Apply the frozen per-seed decision below mechanically. Publish both seeds,
   their paired episode differences, all denominators, every failure, and the
   combined verdict. Preserve archives and their read-back receipts before
   calling the packet complete.

Any implementation/runtime/source mismatch stops expansion and preserves the
affected row as an incomplete or failed attempt. Restoring an operationally
interrupted process with unchanged checkpoints and allocation is distinct from
changing the protocol or retrying a failed scientific row. No automatic retry,
replacement seed, replacement episode, or test-conditioned rescue is permitted.

## Exact scientific decision

For each training seed, the initial and final test banks must each contain
exactly **16 complete episodes of 239 transitions**, or **3824 transitions**.
All must have `feasible_step=1`, `destroy=0`, `action_solver_failure=0`,
`advance_solver_failure=0`, and `voltage_metrics_valid=1`. Fixed resets must all
succeed. This prevents early solver failure from reducing the averaging window
and making voltage violations or reactive effort appear better. Failure of
any condition yields **NOT CONFIRMED** for that seed.

For a valid bank, define:

- `V`: the mean of `voltage_violation_magnitude` across its 3824 transitions.
  Each transition is the mean across buses of
  `max(0,0.95-v) + max(0,v-1.05)` in per-unit voltage.
- `F`: the fraction of bus-transition observations outside `[0.95,1.05]`.
  Record the exact violated-bus count and bus-transition denominator, and use
  counts to compare frequency without floating-point accumulation ambiguity.
- `Q`: the mean of `q_loss` across its 3824 transitions, where each transition
  is the mean absolute inverter reactive output in MVAr.

Denote the initial and final values by subscripts `0` and `1`. A seed passes
exactly one of these mutually exclusive rules:

| Prespecified initial condition | Required final outcome |
|---|---|
| `V0 > 0` | `V1 <= 0.95 * V0` and `F1 <= F0`, with all feasibility conditions above |
| `V0 == 0` and `F0 == 0` and `Q0 > 0` | `Q1 <= 0.95 * Q0`, `V1 == 0`, and `F1 == 0`, with all feasibility conditions above |

The first branch requires at least 5% relative reduction in voltage-violation
magnitude without increased violation frequency. The second permits at least
5% reactive-effort reduction only when the initial policy is already entirely
voltage-safe and the final policy remains so. Initial `V0=0`, `F0=0`, `Q0=0`
has no available improvement target under this plan and is **NOT CONFIRMED**.
No tolerance, alternative metric, relative-return threshold, or post hoc third
branch is added after the test is observed. Report absolute effects alongside
relative changes, including small numerical effects.

The overall learning verdict is **CONFIRMED** only if **both seed 30 and seed
31 pass separately** and all integrity, full-training, checkpoint, replay and
artifact-retention gates pass. Improvement pooled across seeds cannot rescue a
failed seed. Otherwise the verdict is **NOT CONFIRMED**, with the exact unmet
conditions and complete observed outcomes retained.

## Required descriptive outputs and interpretation

Report initial/final per-episode and pooled voltage magnitude, bus violation
frequency, reactive effort, average/reference voltage deviations, worst
lower/upper deviations, total line loss, curtailed PV, solver failures,
feasibility, reward, and episode return. Line losses and all additional metrics
are descriptive; they cannot become replacement success criteria. Report the
physical-bound action saturation rule `abs(action)/0.8 > 0.999`, the bounds
`[-0.8,0.8]`, and its explicit action-scalar denominator (`3824 * 6` for case33)
for each initial/final test bank. This is descriptive, with no added learning
gate. The inherited absolute-action threshold `0.999` is not this measure.
Report the
physical-observation denominator separately when describing any failed bank;
missing physical means are null. Failure/reward/feasibility summaries retain
all attempted transitions and all terminal failures.

Save the immutable plan hash, scientific configuration hash, initial/final
actor and native checkpoint hashes, training parameter changes, every data and
source hash, dependency versions, hardware/thread settings, per-stage elapsed
time/throughput, training health, strict replay evidence, and archive receipts.
Explicitly distinguish a completed experiment with no improvement from a
runtime failure or unfinished allocation.

Even a passing result supports only bounded learning for these two fixed seeds,
this feeder, these temporal banks, and this protocol. Broader robustness,
communication comparisons, and a full power-grid study remain separate work.

## Frozen preparation and launch commands

From the isolated source snapshot, run preparation on one CPU thread after
coordinating the shared worker allocation:

```bash
PYTHONPATH=src:vendor/mapdn-source:.deps \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/Users/akki/Desktop/Thesis/marl-comm/.venv/bin/python scripts/run_mapdn_confirmation.py \
  --prepare-only --data-path /Users/akki/Desktop/Thesis/data/mapdn/case33 \
  --out-dir results/mapdn_case33_confirmation_v1
```

Preparation collects only the fixed training normalization bank. Independently
review `preparation.json` and its exact resolved configurations, bank rows,
normalizer count, source/runtime/data receipt and artifact hashes. Launch the
prepared allocation using that reviewed content digest:

```bash
PYTHONPATH=src:vendor/mapdn-source:.deps \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/Users/akki/Desktop/Thesis/marl-comm/.venv/bin/python scripts/run_mapdn_confirmation.py \
  --run results/mapdn_case33_confirmation_v1 \
  --expected-preparation-sha256 REVIEWED_64_CHARACTER_DIGEST
```

The launch marker is exclusive: rerunning an already launched allocation fails.
Frozen preparation verification repeats before each seed and around every
held-out bank. Initial actor read-back, readable finite 4096/8192 native
actor/critic states, exact final-native/model equality, and initial plus final
checkpoint archives must finish before the first test rollout. Archive checks
reject symlinks/special entries, verify every member, and compare the complete
source inventory before and after archiving.
