# PCP checkpoint-retention repair and CPU v3 confirmation plan

Prepared September 6, 2026, after CPU v2 completed but before any CPU v3
training outcome. This document declares a bounded retention-only rerun. It
does not approve a launch or promote a protocol. Once included in its launch
review, preserve this document unchanged; record later findings separately.

## Evidence-preservation failure and unchanged earlier decisions

Both CPU v1 Identity seeds 10 and 11 failed the behavior replay guard at 564k.
Both CPU v2 runs completed 600k, passed run integrity, positive first-six to
last-six learning improvement, final action/domain audits and replay snapshot
attribution. Their scientific metrics matched the original trajectories exactly
through 558k. However, the mandatory archived CPU v2 480k native checkpoints
were deleted by configured retention. The earlier live report of equal 480k
tensors is historical observation; it is not independently replayable evidence.
CPU v2 therefore remains **NOT_CONFIRMED** under its immutable repair plan.

The assistant's orchestration failed to preserve required evidence. This is
not an observed learning or finite-optimization failure, and it does not waive
the original checkpoint requirement. Preserve the existing cohorts, approvals,
plans, comparators, reports and all surviving checkpoints unchanged:

- [Original confirmation plan](PCP_CONFIRMATION_PLAN.md), SHA256
  `5d0928d07ba01b9abb77f64dc8c9cd88fdca4459de61d96be178002d890aff24`.
- [Replay repair plan](PCP_CONFIRMATION_REPLAY_REPAIR_PLAN.md), SHA256
  `19e9e860bff67069bdacbbd9e158f96f5f3dbefb6f0abc2b4ea58b476abdf6f4`.
- [CPU v2 independent review](../results/pcp_candidate_confirmation_cpu_v2/independent_review.json),
  SHA256 `6de7b0f93f7ea8176153fd61e006008b6e65708d8a978ef083696d6c24476dc6`.
- Original [prefix comparator](../scripts/compare_pcp_confirmation_prefix.py),
  SHA256 `44d2ac34c675ca8dfdc0f8dc33bd14a7676b536d6416bf44cf128c4ca91e1e9f`.
- Reviewed [snapshot auditor](../scripts/audit_pcp_replay_snapshot.py), SHA256
  `17bcdde41492d1a1efaabd45daab0252b99fb2bb28b98cec4dcff79d95184403`.

## The only execution change

Use new candidate protocol `pcp_corrected_cpu_v3` and suite
`pcp_candidate_confirmation_cpu_v3`. Its complete resolved specification differs
from CPU v2 only in output placement and this existing BenchMARL setting:

```yaml
experiment:
  keep_checkpoints_num: null  # Python None: retain all native checkpoints
```

The old value is integer 2. No package source, actor, critic, optimizer, loss,
collector, RNG, domain definition, replay tolerance, scheduling, runtime,
evaluation setting, reward, seed or horizon changes. Protocol identity,
description and the new evidence reference explicitly identify this cohort;
they do not alter its scientific design. Scientific configuration hashes
already exclude retention, but the new comparator must still check the full
resolved diff and reject every undeclared change.

Installed BenchMARL 1.5.2 validates `keep_checkpoints_num=None` as supported.
Its actual `_save_experiment` prunes the saved-path deque before each write.
At 600k, the periodic save and `checkpoint_at_end` both save the same filename;
with keep=2, the second save removes 480k and appends a duplicate 600k path.
With keep=None, neither call prunes. The bounded regression uses a real built
PCP Experiment and its actual checkpoint writer at counters 120k, 240k, 360k,
480k, 600k and 600k, without collecting or training. It must reproduce loss of
480k for keep=2, retain its exact bytes for keep=None, retain all five unique
checkpoint filenames, round-trip both real loss states, and preserve experiment
and Python/NumPy/Torch/environment RNG state. Do not patch BenchMARL or its deque.

## Frozen execution while deliverable 4 proceeds

CPU v3 executes only the frozen research package fingerprint:

```text
7124976af366d28d40a3ccf0ebce62aa2b4019783d13974b9f98dd1c316af262
```

The execution copy is
`/Users/akshatguduru/Desktop/Thesis/.codex-worktrees/pcp-retention-confirmation`,
created from the archived D3 source on a detached checkout. Main-workspace D4
edits must not enter this copy. Preserve its source archive, per-file hashes,
archive hash and reviewed new script/config/plan hashes. The shared `.venv`
symlink supplies the pinned dependencies; **run with `PYTHONPATH=src` from the
execution copy** so the editable installation cannot import the changing main
package. Verify the loaded `commstudy` path, source fingerprint and runtime
before the smoke, review, each worker and post-run frozen audit. Do not install
or update packages in the shared environment during this cohort.

Use CPU sampling/training/buffering, Python 3.12.13, Torch 2.13.0, BenchMARL
1.5.2, TorchRL 0.11.1, TensorDict 0.11.0 and VMAS 1.5.2. Set Torch and all four
OMP/MKL/VECLIB/NUMEXPR thread settings to one. Preserve the exact candidate 6k
smoke with 128 deterministic episodes, final frozen-source test evidence and
the new real retention/comparator tests before issuing a separate candidate
review. Any earlier approval binds its earlier cohort and is not reused.

## Budget, seeds and artifact locations

Run exactly two fresh trajectories, seeds **10 and 11**, from initialization,
600,000 frames and 100 optimization iterations each: **1,200,000 additional
training frames total**. These are reused seeds after an engineering retention
repair, not previously unseen confirmation seeds. The historical v1 collection
budget was 1,128,000 frames and v2 used 1,200,000; neither disappears from the
compute record. Smoke, evaluations and offline diagnostics are separate costs.
No horizon extension, resume, seed replacement or repeated attempts until a
pass is permitted. Comparison seeds 20–24 and ablation seeds 20–22 stay unused
by this remediation. A failed gate leaves the new cohort unconfirmed and
requires a separately documented next decision.

The new sweep writes `runs/pcp_candidate_confirmation_cpu_v3/` under the frozen
execution root. Preserve it there until the complete packet is archived and
hash verified. Derived reports belong under
`results/pcp_candidate_confirmation_cpu_v3/`, outside every audited run tree.
The old suites remain at their original main-workspace locations and are read
only. If durable copies are made, preserve original bytes, record source and
destination paths/hashes and never merge run identities or overwrite attempts.

Immediately after each completed run, require and hash a retained manifest
containing native `checkpoint_120000.pt`, `checkpoint_240000.pt`,
`checkpoint_360000.pt`, `checkpoint_480000.pt`, `checkpoint_600000.pt`, the final
managed `checkpoints/policy_state.pt` and its 564k first-predator-fallback
snapshot. Validate each native frame/update header. Archive all these bytes,
metadata, resolved configuration, metrics, status, manifests and logs before
reviewing promotion. A missing required file fails the evidence gate; do not
substitute an earlier assistant statement, a final policy or a replay snapshot.

## Mandatory invariance comparisons

Use the new [retention comparator](../scripts/compare_pcp_retention_confirmation.py),
which hash-binds the unchanged original prefix helper. For each seed require:

1. CPU v1 versus v3 exact numeric metric equality through 558,000 frames,
   keyed by `(frames, iteration, phase, group, metric, sample)`. Retain the
   original exclusions: timestamp column, `wall_time_seconds`, and exactly
   the eight replay diagnostic names listed in the immutable replay repair
   plan. Require no earlier fallback in either group. Compare every 480k
   native actor/critic loss-state tensor for both groups with identical keys,
   dtype, shape and bytes; compare TensorDict metadata with exact types/values.
2. CPU v2 versus v3 exact numeric metric equality for the **complete 600k
   trajectory**, including every replay diagnostic, training value, collection
   value, evaluation episode, domain measurement and communication scalar.
   The only exclusions are timestamps and `wall_time_seconds`. Compare every
   600k native actor/critic loss-state tensor and its metadata for both groups
   with the same exact rules. This directly tests that retaining files did not
   alter the complete observed numerical trajectory.
3. Source/runtime and saved protocol bindings must identify original v1,
   unchanged-source v2 and v3 correctly. V1 remains failed; v2 and v3 must be
   completed. Resolved differences are limited to `save_folder` and integer
   keep=2 versus None. Reject missing/extra/duplicate metric keys, missing
   checkpoints, non-finite values, dtype coercion, signed-zero changes,
   unsupported exclusions or unexpected configuration differences.

Preserve full input hashes, retained key/value hashes and counts, compared
tensor counts, first differences, all retained-artifact hashes and before/after
immutability checks. File serialization bytes need not match across native
checkpoints; the required equality concerns every saved loss-state tensor.
The original v2 480k comparison remains unavailable and is not retroactively
declared passed. V3 supplies its own auditable original-v3 480k evidence.

## Replay attribution and complete scientific review

Use the separate [CPU v3 snapshot wrapper](../scripts/audit_pcp_retention_snapshot.py)
with the original v1 suite. It hash-binds the reviewed v2 auditor and invokes
its same strict implementation under the new protocol. Require first predator
fallback exactly at 564k, zero earlier fallback count/slices for both groups,
actual CSV collection iteration 93, all-group strict loaded state, preserved
named batch `[10, 600]`, stored actions/parameters and captured tensors. Recompute
the whole batch and all 600 collection-width slices at original rtol=atol=1e-5.
Collection replay must pass, whole-batch rejection and captured errors must
reproduce, and model/RNG/input artifact hashes must remain unchanged.

Matching original rounded errors supports attribution but cannot reconstruct
the lost exact v1 564k tensors. The snapshot is diagnostic-only, not a resume
checkpoint or final-policy evaluation. Preserve the report and its metadata,
config, protocol, runtime, source, CSV iteration and snapshot hash sidecar.

All original scientific criteria remain mandatory: both 600k rows complete,
finite required health metrics for both optimized groups, exactly 51 scheduled
evaluations with 128 complete 100-step deterministic episodes, exact domain
reward accounting, and positive first-six to last-six return improvement in
each seed. Report last-six return, normalized AUC and prespecified descriptive
windows without introducing a return, entropy, saturation or convergence
threshold. Audit each final policy in RANDOM and DETERMINISTIC modes with
32 complete 100-step episodes, the unchanged held-out seed bank 100000–100031,
positive finite scales and complete finite action/location/domain evidence.

The new retention, replay, integrity and scientific reviews are separate
requirements. Only a complete reviewed CPU v3 packet can support promotion of
its CPU protocol. CPU v1 remains failed and CPU v2 remains unconfirmed under
their own plans. Promotion does not establish convergence, a CUDA result or
communication necessity. D4 uses its separately audited source and capacity
evidence; a source change requires an explicit reviewed bridge to this frozen
candidate rather than silently relabeling the confirmation source.

## Execution commands after review

From the frozen copy, prefix each command with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src`.
Use its `.venv/bin/python`. Resolve old suite paths to the original workspace.
Before a worker launch, review its exact seed/run ID and the two-row manifest;
no command below itself issues approval.

```text
scripts/smoke_pcp_candidate.py --protocol configs/protocols/pcp_corrected_cpu_v3.yaml
scripts/sweep.py configs/sweeps/pcp_candidate_confirmation_cpu_v3.yaml
scripts/sweep.py --manifest <reviewed-v3-manifest> --run-id <one-declared-seed-run> --run
scripts/compare_pcp_retention_confirmation.py --original-suite <v1-suite> --repaired-suite <v2-suite> --retained-suite runs/pcp_candidate_confirmation_cpu_v3 --protocol configs/protocols/pcp_corrected_cpu_v3.yaml --out results/pcp_candidate_confirmation_cpu_v3/retention_invariance.json
scripts/audit_pcp_retention_snapshot.py runs/pcp_candidate_confirmation_cpu_v3 --previous-suite <v1-suite> --protocol configs/protocols/pcp_corrected_cpu_v3.yaml --out results/pcp_candidate_confirmation_cpu_v3/replay_snapshot_audit.json
```

Use the existing generic action/domain and strict confirmation CLIs with the
v3 protocol and cohort, preserving their JSON output, final-policy checkpoint
hashes and separate reviewer decision. Verify actual CLI help before preparing
the executable preflight packet; placeholders here identify required inputs.
