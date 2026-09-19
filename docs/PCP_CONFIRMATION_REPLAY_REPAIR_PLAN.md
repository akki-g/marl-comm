# PCP replay-instrument repair and CPU v2 confirmation plan

Prepared September 6, 2026, after both CPU v1 attempts failed and before any
CPU v2 training outcome. This is a prespecified engineering and scientific
review plan. It does not approve a launch, establish the exact cause of the
original failures, or presume successful confirmation. Final source tests,
runtime evidence, protocol bindings and a separately recorded review are
required before either new full run.

The original [confirmation plan](PCP_CONFIRMATION_PLAN.md), SHA256
`5d0928d07ba01b9abb77f64dc8c9cd88fdca4459de61d96be178002d890aff24`, remains
unchanged. Its scientific settings, learning criteria, domain definitions,
action audit and interpretation limits are retained. This new document adds
the replay-repair scope, additional compute, failure preservation and explicit
behavior-invariance checks. Once included in a launch review, this document
must remain unchanged through execution and adjudication; later findings
belong in separate evidence files.

## Original outcome and evidence for a bounded repair

Both rows in `runs/pcp_candidate_confirmation_cpu_v1/` failed the predator
behavior-log-probability replay guard after collecting 564,000 frames, before
optimizing that collection. Their last complete logged iteration covers
558,000 frames: 93 completed collection/training iterations. Seed 10's reported
maximum absolute replay error was `1.9669533e-5`; seed 11's was `1.4424324e-5`.
The logged values were finite. Under the original plan, the cohort is **not
confirmed**. Finite logs do not waive the failed replay/completion requirements.

The original research source hash was
`73bc2cb332c0d4ccd5c6afb893b1e929f385b4cabfc6cb743c99eb6d0ec9d90c`, with
CPU v1 protocol hash
`d61d39f0e47a234e47a7fd6c3c3bea2c3ece91f586917e284608fa938e840bd0`.
Preserve the original failed statuses, metadata, manifests, metrics, logs,
360k/480k checkpoints, protocol, approval and all diagnostic evidence. Do not
rename a failed row into a successful row, discard an attempt, or change its
source binding. The exact failing 564k batch and policy state were not saved.

The read-only [support-point probe](../results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/support_point_probe.py)
and its [results](../results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/support_point_probe.json)
identify a reproducible false-rejection mode on both frozen 480k actors. The
probe uses their saved observations, recomputes a fixed policy at collection
width ten, and searches a declared 33-by-33 grid of constructed Normal support
points spanning -4 to 4 per action coordinate. It constructs behavior actions
and log probabilities at that width, then compares whole-batch and original
collection-shape replay without any parameter update.

| Frozen actor | Whole-batch maximum absolute error | Scalars outside original tolerance | Maximum error / allowed error | Collection-shape error |
|---|---:|---:|---:|---:|
| Seed 10, 480k | 1.4781952e-5 | 2 | 1.42356 | Exactly zero |
| Seed 11, 480k | 1.9073486e-5 | 2 | 1.71835 | Exactly zero |

The recorded parameter hashes, RNG state and input-artifact hashes remain
unchanged. The saved minimal-tensor artifact hashes were independently checked.
The [natural-draw probe](../results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/natural_draw_probe.py)
and [results](../results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/natural_draw_probe.json)
use five action-draw seeds per frozen actor on those observations. All ten
whole-batch checks pass the original tolerance; all collection-shape replays
are bitwise exact. The separate
[distribution-only probe](../results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/distribution_only.md)
finds cache size zero and no tolerance violations when location and scale are
held fixed across batching layouts.

Together these results justify investigating a narrowly scoped guard repair:
actor evaluation at different batch shapes can introduce floating-point output
differences that amplify in action log probabilities despite unchanged policy
parameters. The selected support grid establishes existence, not failure
frequency under natural sampling. Neither probe reproduces the unavailable
564k historical batch or proves that this mechanism caused either original
failure. That attribution remains a required investigation in the new cohort.

## Allowed instrument change and prerequisite validation

The proposed repair retains the existing whole-batch comparison. Only if it
fails may the guard recompute the same stored actions using each slice of an
unambiguous named `time` dimension, retaining the original collection batch
shape. It must recompute the actor from its genuine stored inputs and replay
masks, rather than merely reusing saved distribution parameters or averaging
errors. Every accepted log-probability scalar must satisfy the unchanged rule:

```text
abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)
```

Reject non-finite data, unsupported or ambiguous layouts, and mismatches that
remain at the collection shape. Preserve CPU/Python/NumPy and channel RNG,
communication statistics, model state and the original batch on success and
exceptions. The repair must not change the actor, critic, collector, actions,
optimizer, PPO loss, minibatch scheduling, reward, horizon or task semantics.
The production PPO minibatch computation remains unchanged; a successful
collection-shape check does not claim bitwise equality of every batched
floating-point implementation of that policy.

Before training, complete and retain the final full source test gate, lint,
compilation and diff checks. Include the two frozen-actor support regressions,
stored-action/log-probability and parameter/input corruption tests at the
existing tolerance, non-finite rejection, named-time validation, real
optimization, channel-mask replay and RNG/statistics preservation. The saved
fallback snapshot must reload and reproduce both comparison results. Run a
new exact-runtime 6,000-frame smoke retaining the full candidate's 128-episode
evaluation setting. Review the complete resolved v1/v2 scientific/runtime diff:
only the declared diagnostic contract, source/protocol identifiers and artifact
placement may differ. Passing these engineering checks does not establish
learning feasibility.

## Separate cohort, unchanged design and additional compute

Use separately versioned protocol `pcp_corrected_cpu_v2`, suite
`pcp_candidate_confirmation_cpu_v2`, new source bindings and a new reviewed
decision. The old v1 decision does not authorize changed source. Start **both
original seeds 10 and 11 from scratch**, with one new run per seed under the
same reviewed v2 source. These are reused confirmation seeds after instrument
repair, not two previously unseen training seeds. Do not combine a v1 attempt
with a v2 attempt. Comparison seeds 20–24 and ablation seeds 20–22 remain unused
for this repair.

Each v2 trajectory remains exactly 600,000 training frames and 100 iterations.
This adds **600,000 frames per seed, 1,200,000 training frames total**, beyond
the two preserved v1 attempts, which already collected 1,128,000 frames total
and completed optimization through 558,000 frames per seed. Evaluation,
diagnostic and smoke compute must be recorded separately. This is a fresh
bounded rerun, not a continuation to a longer training horizon. Native
checkpoints lack exact optimizer/RNG resume state and must not be presented
as scientific continuation points.

Retain gamma 0.95, entropy 0.1, ten minibatch passes, learning rate 5e-5, GAE
lambda 0.9, 6,000-frame collections, 400-frame minibatches and ten environments.
Retain the default 17-feature predator actor, 22-feature physical-state critic,
parameterless Identity, both optimized groups, predator-only measured return,
shared unshaped repeated-contact reward, no respawn, and corrected 100-step
time-limit bootstrap. Preserve all task/RNG/input/evaluation/channel contracts.

Runtime remains CPU for sampling, training and buffering, Python 3.12.13,
PyTorch 2.13.0, BenchMARL 1.5.2, TorchRL 0.11.1, TensorDict 0.11.0 and VMAS
1.5.2. Set torch and OMP/MKL/VECLIB/NUMEXPR threads to one. There is no device,
precision or runtime substitution, hyperparameter tuning, seed replacement,
additional method, horizon extension or CUDA confirmation in this scope.

## Mandatory comparison with the original trajectory

For each seed, compare the entire new and original `metrics.csv` prefixes
through **558,000 frames inclusive**, beginning at the first 6,000-frame
collection. Key each scalar by `(frames, iteration, phase, group, metric,
sample)`, reject duplicate/missing keys and require exact numeric equality.
Include collection, training, evaluation, per-episode domain, aggregate domain
and communication measurements for both groups where recorded. This comparison
tests the observable numerical trajectory, not only the learning-curve means.

The only exclusions are the `timestamp` column, metric `wall_time_seconds`,
and these explicit replay-diagnostic metrics:

```text
replay_log_prob_max_abs_error
replay_log_prob_scalar_count
replay_transition_count
replay_batched_log_prob_max_abs_error
replay_batched_log_prob_max_tolerance_ratio
replay_accepted_log_prob_max_tolerance_ratio
replay_collection_shape_fallback_count
replay_collection_shape_fallback_slices
```

Do not round numeric values, widen comparison tolerances, remove inconvenient
frames, or add exclusions after seeing the result. Export input-file hashes,
retained-key counts and hashes, exclusions, equality result, and the first
differences if any. Independently require **no collection-shape fallback before
564,000 frames** in either group: v1 passed the unchanged whole-batch guard
through the 558k prefix. Any earlier fallback or other prefix difference
prevents a behavior-invariance claim and requires investigation.

Use the [prefix comparison script](../scripts/compare_pcp_confirmation_prefix.py)
with `--through-frames 558000` and preserve its report. In addition to the
metric comparison, require **bitwise equality of every 480k native loss-state
tensor for both `adversary` and `agent`**, matching tensor key sets, dtypes and
shapes. This checks the saved actor/critic state at an earlier common checkpoint
as well as the logged numerical trajectory. Hash both original and v2 native
checkpoint files and report the compared tensor count and any unequal keys;
whole checkpoint files need not have identical serialization bytes. The report
must also verify unchanged resolved specifications except `save_folder`,
matching software/runtime declarations, distinct implementation source hashes,
original failed statuses and new completed statuses. A checkpoint-state
difference fails the invariance requirement even if aggregate returns match.

At **564,000 frames**, capture the first predator fallback for each seed in
inspectable files before the batch is optimized. Retain the named batch shape,
frame/iteration/group, fixed tolerances, stored observation/context/mask/action/
location/scale/log-probability tensors, whole-batch and collection-shape replay
outputs, policy weights, source/protocol identifiers and artifact hashes.
Record both maximum absolute and tolerance-normalized errors, failing-scalar
counts and offending expected/actual pairs. The snapshot is diagnostic-only,
not a resume checkpoint. First-fallback capture is sufficient only if the
required absence of earlier fallback is established.

Review whether the original whole-batch failure recurs at this frame with the
reported seed-specific error, while collection-shape replay passes the same
tolerance. Re-run the saved snapshot offline under unchanged weights/inputs
and compare both paths. The original rounded traceback cannot supply all
missing per-scalar historical values; state that limit. If the original
failure does not recur, preserve the evidence and report attribution unresolved
rather than claiming that the exact old failure was reproduced. A missing
snapshot or unresolved trajectory discrepancy cannot be hidden by a completed
training process.

## Full-horizon review and stopping rules

The original plan's complete 600k/100-iteration, finite-health and task/domain
accounting requirements remain mandatory for both new rows. Preserve the
51-point evaluation schedule, 128 complete deterministic episodes per point,
last-six evaluation mean, normalized AUC and positive first-six to last-six
return improvement in **each** seed. No historical return, entropy or action
saturation observation becomes a newly selected acceptance threshold.

Audit each final policy in RANDOM and DETERMINISTIC modes with exactly 32
complete 100-step episodes per mode, sharing held-out seeds 100000–100031
across modes and training seeds. Require strict checkpoint/source/runtime
bindings, complete action/scale/location and domain records, positive scales,
finite values and the original repeated-contact accounting identity. Report
the validator's artifact-integrity result separately from the learning,
behavior-invariance and repair-attribution reviews.

Retain every failure and all additional compute. A persistent collection-shape
mismatch, incomplete run or failed learning condition leaves confirmation
unsuccessful; a completed run with unexplained repair or domain behavior cannot
be promoted automatically. Do not issue repeated fresh runs until one happens
to pass. Any further repair, budget or scientific change requires separately
documented evidence and scope, while these attempts remain visible.

Only a reviewed complete v2 packet can support freezing its declared CPU
protocol. The original v1 cohort remains failed/not confirmed regardless of
the v2 outcome. Neither this plan, an engineering review, nor an integrity
validator pass establishes convergence, communication necessity, a CUDA result
or permission for the main comparison.
