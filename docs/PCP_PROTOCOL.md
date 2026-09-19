# Corrected PCP protocol and launch contract

September 6, 2026. Deliverables 1–3 now provide strict specifications, corrected
PCP input/RNG/time-limit contracts, transition measurements, health guards and
full-horizon audit tooling. The two repaired CPU v2 runs completed and passed
learning/action/domain checks. Their **full prespecified confirmation is not
confirmed**: required 480k checkpoint bytes were pruned before archival, leaving
the mandatory invariance evidence incomplete. The [results and incident
review](PCP_CONFIRMATION_CPU_V2_RESULTS.md) distinguish the passing checks from
this preservation failure. No frozen-protocol promotion or main launch follows.

The original CPU v1 attempts remain failed. The CUDA
[`pcp_corrected_v1.yaml`](../configs/protocols/pcp_corrected_v1.yaml) is separately
unconfirmed. Historical Slurm array 795630 supports candidate selection only.
The [original plan](PCP_CONFIRMATION_PLAN.md) and
[replay-repair plan](PCP_CONFIRMATION_REPLAY_REPAIR_PLAN.md) remain unchanged.

## Scientific specification

The protocol contains the complete resolved task, actor, critic, MAPPO, and
experiment settings, including defaults that were previously implicit. Gamma
0.95, entropy coefficient 0.1, ten minibatch passes, 6,000-frame collections,
400-frame minibatches, learning rate 5e-5, and GAE lambda 0.9 are retained from
the leading historical candidate. The declared budget is 600,000 frames, with
128 deterministic evaluation episodes at the declared 12,000-frame interval.
This is a fixed-compute budget; it does not claim convergence.

The task uses corrected PCP RNG and visibility semantics, a 17-feature predator
observation, and the same 22-feature physical-state critic input across visibility
conditions in the default three-predator/one-prey layout. Reward remains repeated
contact reward inherited from Simple Tag, without first-contact termination.
The `pcp_time_limit_v1` environment disables raw VMAS time limits and applies a
100-step TorchRL StepCounter. At that limit, `done=true`, `truncated=true`, and
`terminated=false`, preserving time-limit value bootstrapping in the unchanged
MAPPO path. Raw VMAS previously marked the horizon as termination, suppressing
that bootstrap. Historical rows therefore cannot confirm the corrected task.
`adversary` is the measured group. Both `adversary` and `agent` are still optimized;
the scripted prey discards its learned action. Removing that unused optimization
would change RNG/scheduling and requires a separately validated protocol.

The original CUDA candidate pins the historical HPC package versions (Python 3.11.4, PyTorch
2.13.0+cu126, BenchMARL 1.5.2, TorchRL 0.11.1, TensorDict 0.11.0, VMAS 1.5.2),
CUDA, and four compute threads. A successful CPU source test does not meet this
runtime contract. Device or thread changes require a reviewed protocol revision.

The selected CPU variant declares Python 3.12.13, PyTorch 2.13.0 and the same
BenchMARL/TorchRL/TensorDict/VMAS versions, with CPU for all three devices and
one compute thread. It was chosen before fresh outcomes because present HPC
access was not established. It retains every scientific hyperparameter and
seed allocation, while using its own protocol and suite IDs. A resulting CPU
verdict cannot confirm the CUDA candidate or be pooled with CUDA results; MPS
is not selected. Both variants remain fixed-budget candidates, not convergence
claims.

| Stage | Planned rows | Training seeds | Evidence needed |
|---|---:|---|---|
| Candidate confirmation | 2 Identity | 10, 11 | Review of candidate and corrected instrument |
| Main comparison | 30, six named methods | 20–24 | Fresh corrected-protocol confirmation and frozen specification |
| Declared ablations | 201 across seven factors | 20–22 | Main-comparison review; separate evidence for guarded factors |

The completed repaired execution is the two-row
[`pcp_candidate_confirmation_cpu_v2.yaml`](../configs/sweeps/pcp_candidate_confirmation_cpu_v2.yaml)
suite, with suite ID `pcp_candidate_confirmation_cpu_v2`. The existing CUDA main
and ablation catalogue is separate; CPU confirmation does not authorize it.

The row counts preserve an inspectable catalogue, not a commitment to execute
every method or ablation. Identity and the framework MLP are equivalent
implementation controls, not distinct scientific baselines. Width 64 and rounds
2/3 remain guarded even when an ablation stage is otherwise approved. Their
factor IDs must be explicitly validated with content-hashed supporting evidence.

## Planning and review

From the repository root, with the installed environment:

```bash
python scripts/protocol.py inspect configs/protocols/pcp_corrected_cpu_v2.yaml
python scripts/sweep.py configs/sweeps/pcp_candidate_confirmation_cpu_v2.yaml
```

These commands inspect and generate the CPU manifest without training.
`bash slurm/pcp_01_setup.sbatch` separately generates the CUDA catalogue without
submitting jobs. Each row includes its complete resolved specification, scientific
SHA256, protocol SHA256, source SHA256, stage, seed, factor, and a contract for
the actual built actor/critic classes, parameter shapes/counts, and task inputs.
Architecture inspection builds temporary CPU experiments without rollouts or
optimization and preserves RNG state. All paths and run IDs are disjoint from
the historical PCP suites.

Unknown task/model/suite keys, flat keys mixed into grouped actors, unsupported
channel options, invalid finite/range values, conflicting labels/overrides, and
undeclared factors fail. A protocol can define its critic independently of the
named critic YAML; export/reload retains that complete definition.

To prepare a reviewable decision, generate a **pending** template:

```bash
python scripts/protocol.py approval-template configs/protocols/pcp_corrected_cpu_v2.yaml \
  --stage confirmation \
  --out /tmp/pcp_confirmation_review_template.json
```

The template does not authorize anything. A completed review records its
reviewer, rationale, exact source/protocol hashes, stage, `decision: approved`,
and evidence records with `path`, file `sha256`, and `kind`. Confirmation requires
`candidate_review`; comparison requires `corrected_protocol_confirmation`;
ablation requires `main_comparison_review`. Guarded factors additionally require
an ID such as `message_dim/64` in `validated_factors` and an evidence record of
kind `factor:message_dim/64`. The gate verifies bindings and evidence integrity;
the reviewer is responsible for whether the evidence supports the decision.
It is an experiment-control mechanism, not a cryptographic identity/signature
system. Tests use approvals confined to temporary fixture directories. Review
identity and rationale must describe the actual reviewer; generating the
template or this guide supplies neither a review nor a scientific verdict.

The two launch decisions record `Codex (assistant engineering review)` as the
actual reviewer of the user-authorized bounded task. The preserved
[CPU v1 decision](../configs/protocols/approvals/pcp_corrected_cpu_v1_confirmation.json)
binds the initial source and failed attempts. The separate
[CPU v2 decision](../configs/protocols/approvals/pcp_corrected_cpu_v2_confirmation.json)
binds the unchanged scientific/runtime design, repaired guard, both plans and
[CPU v2 preflight](../results/pcp_candidate_confirmation_cpu_v2/preflight_review.json).
Neither decision is human adjudication or successful-learning evidence, and
neither authorizes comparison or ablation rows. Both protocols remain candidate
artifacts so their executed hashes remain reproducible. The final scientific
review does not issue a main-stage approval.

Promoting a candidate to frozen changes the protocol hash. Keep the confirmation
artifacts as evidence of the candidate hash, review the promotion explicitly,
and regenerate manifests and decisions for the frozen hash. A source change
under `src/commstudy` likewise invalidates the old binding; documentation/test
edits do not change the source fingerprint. Evidence file edits require a new
review of their hashes. Historical manifests remain readable for status and
analysis, but cannot launch scientific PCP work without these bindings.

## Submission and worker checks

For the selected CPU runtime, explicitly pin all four thread variables before
the [one-batch runtime check](../scripts/smoke_pcp_candidate.py):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/smoke_pcp_candidate.py \
  --protocol configs/protocols/pcp_corrected_cpu_v2.yaml
```

The script changes only `max_n_frames` to 6,000 and preserves the full candidate
settings, including 128 evaluation episodes. It checks the declared runtime,
records its validation purpose and checks the built model contract. Its result
is instrument evidence, not a confirmation row. The repaired-source smoke
completed with 4,116 finite logged values, 128 complete truncated
evaluation episodes, no terminal flags and zero maximum reward-accounting
error. Its artifact hashes are retained in the preflight evidence. After source validation,
completed smoke evidence and the source-bound candidate review, the managed
CPU entry point below is retained for reproducibility. Both rows are completed;
this record does not request another launch:

```bash
python scripts/protocol.py check \
  --manifest runs/pcp_candidate_confirmation_cpu_v2/manifest.csv \
  --stage confirmation
python scripts/sweep.py \
  --manifest runs/pcp_candidate_confirmation_cpu_v2/manifest.csv \
  --run --stop-on-failure
```

The distinct CUDA submission workflow remains:

```bash
python scripts/protocol.py check \
  --manifest runs/pcp_candidate_confirmation_corrected_v1/manifest.csv \
  --stage confirmation
bash slurm/pcp_02b_protocol.sbatch
```

Use the `bash` entry point: it validates every selected row before calling
`sbatch`, so a missing or stale approval cannot allocate the training array.
Calling `sbatch` directly bypasses that outer branch; its worker still repeats
the gate before `srun` and training. Workers also check Python/packages, CUDA
availability, torch thread count and the four thread environment variables.
The managed runner independently checks the binding and actual built model
contract before calling `experiment.run()`.

Full protocol comparison permits output-directory relocation. Logging,
evaluation, checkpoint, restore, runtime and scientific settings are frozen.
The separate scientific hash excludes explicitly classified operational fields;
it alone is not launch approval. A manifest's saved specification and its
explanatory override string must agree. Retries preserve original artifacts;
native checkpoints are still not exact optimizer/RNG resume points.

Ad hoc managed PCP checks use `scripts/train.py --validation-run` and are capped
at 6,000 frames, with their purpose recorded as validation. This does not allow
a full-horizon study without review. Low-level construction remains available
for tests and controlled analysis.

## Diagnostic contracts

Training health is group-specific. Collection logs action/location/scale
quantiles, finite rates and saturation. Optimization logs value-target and
advantage moments alongside the framework's PPO diagnostics and pre-clipping
gradient norms. Communication health records encoder and contribution norms,
their ratio, gate saturation and normalization gain where applicable. Counts
identify transition/scalar denominators. Collection, optimization replay and
evaluation have separate phase contexts, so replay compute is not counted as
deployed communication cost. Deterministic and stochastic evaluation both obey
the declared channel intervention.

Before each collection batch is optimized, stored behavior log probabilities
are recomputed under the stored channel realization and compared for every
trained MAPPO group. Loss/gradient/parameter/optimizer-state hooks reject
non-finite values before the affected optimizer step, preserving failed-run
status and traceback. Negative differential entropy or low explained variance
alone does not reject a run. No MAPPO loss or optimization loop is forked.

Checkpoint diagnostics reconstruct from `resolved_config.yaml`, select the
measured group explicitly, and execute the requested number of episodes in
fresh single-world environments. Source, task and package compatibility are
checked before loading. `--analysis-device` and `--allow-runtime-mismatch` are
explicit analysis-only controls with exported provenance; they cannot bypass
weight-shape or group checks and never authorize training. Legacy incompatible
actors/critics require a separately scoped historical probe, not silent loading.

## PCP domain evidence and confirmation report

`pcp_transition_metrics_v1` exposes read-only simulator measurements from the
transition that generated reward, before reward-triggered respawn can alter
positions. It records predator–prey contact-pair transitions, first contact and
contact duration, simultaneous contacts, nearest-prey distances, boundary
occupancy, obstacle/teammate collisions, visibility counts and ordered
sender-visible/receiver-invisible opportunities. Measurements do not enter
actor/critic inputs or consume RNG; duplicate copies of team measurements are
validated and counted once.

For the candidate's unshaped, shared predator reward and no-respawn setting,
mean predator episode return equals `10 * contact_pair_transitions`. This
measures repeated-contact reward, not distinct captures or demonstrated
cooperation. Shaped or unshared task variants require a different accounting
reference. The [confirmation plan](PCP_CONFIRMATION_PLAN.md) fixes the exact
measurement definitions and interpretation before new results.

After both full rows complete, generate final-policy evidence and validate the
complete suite using the new [audit](../scripts/audit_pcp_confirmation.py) and
[confirmation](../scripts/confirm_pcp.py) entry points, under the same CPU
runtime/thread declaration:

```bash
python scripts/audit_pcp_confirmation.py \
  runs/pcp_candidate_confirmation_cpu_v2 \
  --out-dir results/pcp_candidate_confirmation_cpu_v2 \
  --episodes 32 --seed 100000
python scripts/confirm_pcp.py runs/pcp_candidate_confirmation_cpu_v2 \
  --protocol configs/protocols/pcp_corrected_cpu_v2.yaml \
  --action-audits results/pcp_candidate_confirmation_cpu_v2/action_audits.json \
  --domain-audits results/pcp_candidate_confirmation_cpu_v2/domain_audits.json \
  --out results/pcp_candidate_confirmation_cpu_v2/confirmation_report.json
```

The audits select `adversary` and require a real final checkpoint. Each policy
runs 32 complete 100-step episodes in both DETERMINISTIC and RANDOM modes, using
held-out episode seeds 100000–100031. That is 3,200 transitions and 19,200
predator action scalars per mode. The audit script preserves completed evidence
and writes `audit_failures.json` if another run fails. Missing checkpoints or
cross-runtime diagnostics cannot satisfy confirmation.

The validator checks exactly seeds 10/11, completed 600k/100-iteration runs,
source/protocol/specification/runtime bindings, health and domain streams,
checkpoint identity, and the complete action/domain audit records. The expected
evaluation schedule contains 51 points: the first is after the first update at
6,000 frames, followed by the 12,000-frame schedule through 600k, with exactly
128 complete episodes at each point. Final performance averages the last six
means (540k–600k); normalized AUC uses the recorded frame span. The report
exports `early_window_mean`, `late_window_mean`, and `early_late_delta` for
return and domain measures plus the full curves. Additional 408k–492k versus
504k–600k comparisons remain review calculations from those curves.

A `checks_passed=true` result is artifact-integrity evidence and still reports
`review_status=required` and `approval_decision=not_issued`. The prespecified
learning review requires positive first-six to last-six improvement in each
seed and an explanation of action/domain behavior. No numerical performance,
entropy or saturation threshold is selected after seeing these rows. Failures
are retained; replacing a seed, extending the budget or tuning a method is not
part of this confirmation. Protocol promotion requires the separate reviewed
decision and regenerated hash bindings described above.

The September 5 engineering gate comprises 810 passing full-run tests plus one
stale Slurm expectation, corrected before a 55-test targeted pass; the current
confirmation-validator suite separately passes 32 tests. This is the observed
composite verification, not a claim that a later full rerun was completed.
The two CPU v2 rows and their audits are complete. The run-artifact validator
passes, but the full repair plan remains **not confirmed** because both required
480k checkpoint files were removed by native retention before archival. The
strict prefix comparator intentionally remains failed. Its separately saved
metric-only comparison passes all 201,665 retained scalars per seed through 558k;
it cannot substitute for the missing checkpoint requirement.

The [retention incident](../results/pcp_candidate_confirmation_cpu_v2/checkpoint_retention_incident.json)
reproduces the installed save behavior in a temporary diagnostic: the periodic
600k save followed by the end save prunes the older 480k file under `keep=2`.
The assistant should have archived the required evidence before cleanup. Before
any new confirmation, separately review either retaining all checkpoints or
archiving required files and hashes outside the pruning directory, and verify
survival through both final saves. A future run cannot recover the deleted
artifacts. No new cohort, horizon extension, protocol freeze, main comparison or
ablation is part of this completed execution record.

Additional read-only checks used for this packet:

```bash
python scripts/compare_pcp_confirmation_prefix.py \
  --original-suite runs/pcp_candidate_confirmation_cpu_v1 \
  --repaired-suite runs/pcp_candidate_confirmation_cpu_v2 \
  --through-frames 558000 \
  --out results/pcp_candidate_confirmation_cpu_v2/prefix_invariance.json
# Expected rejection: the required repaired 480k checkpoints are missing.
python scripts/audit_pcp_replay_snapshot.py \
  runs/pcp_candidate_confirmation_cpu_v2 \
  --previous-suite runs/pcp_candidate_confirmation_cpu_v1 \
  --protocol configs/protocols/pcp_corrected_cpu_v2.yaml \
  --out results/pcp_candidate_confirmation_cpu_v2/replay_snapshot_audit.json
```

The second audit reconstructs the saved 564k diagnostic policy, not the final
600k policy. Both original rounded errors recur; collection-shape replay is
bitwise exact under unchanged tolerance. That is strong mechanism evidence,
with the unavailable original 564k tensors and missing repaired 480k artifacts
kept explicit in the [final review](PCP_CONFIRMATION_CPU_V2_RESULTS.md).
