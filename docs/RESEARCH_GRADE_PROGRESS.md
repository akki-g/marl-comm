# Research code implementation record

**Continuation — September 9, 2026:** repaired the relocated workspace's exact
runtime and preserved original evidence through a verified path-only relocation
packet. Both original controls now have complete, strictly reviewed held-out
banks; exact local null interventions and pre-migration action/domain audit
equivalence passed. The corrected two-worker coordinator started the remaining
allocation at 21:49 EDT, after preserving both completed originals; D4 remains
incomplete. The isolated MAPDN implementation
now supports real case33 data, managed MAPPO, chronological data splits,
training-only normalization, per-transition domain diagnostics and strict
checkpoint replay. Its real engineering smoke and two-seed bounded
voltage-improvement gate passed; seed 31's costs and return worsened, and
residual voltage violations remain. The first D4 continuation stopped at an
archive-provenance mismatch before new training; the failed launch is preserved
and the narrow archive-reuse fix passed 150 focused tests and independent review.
See [MAPDN scope and results](MAPDN_IMPLEMENTATION_STATUS.md),
the latest [main agent log](../agents.md) entries
and `../../.codex-worktrees/mapdn-development/` for the isolated implementation.

**Historical operational status, September 8, 22:16 UTC (superseded):** **2 of 40 D4 training
rows completed and were archived; 38 remain unstarted.** Both completed rows
passed the full training-integrity, final action/domain and retained-checkpoint
audits. Their held-out evaluators are now suspended and incomplete; neither
`heldout_evaluation.json` exists. The queue coordinator is also suspended.
The hold recorded 783,843,328 free bytes (about 0.73 GiB). Restore at least
8 GiB and verify frozen inputs before resuming: evaluators first, then the queue
only after audits close, with no more than two active CPU workers. D4 remains
incomplete. See the [evaluation resource hold](../results/pcp_visibility_budget_cpu_v1/evaluation_resource_hold_01.json).

**Historical operational update, September 8, 21:43 UTC (superseded):** new D4 submissions are on a
reversible storage hold after available disk space fell below the initial
allocation allowance. Both initial seed-20 runs completed 600k and were archived;
their full final evaluations are running. The other 38 rows remain pending.
Only the queue coordinator is suspended. Run status files remain
authoritative; restore headroom and verify frozen inputs before resuming the
existing coordinator. See the [resource hold record](../results/pcp_visibility_budget_cpu_v1/resource_hold_01.json).

Started September 5, 2026, from source `40a97a6`. The controlling plan is the
[thesis-wide roadmap](../../RESEARCH_CODE_ROADMAP.md). PCP and MAPDN are the
scientific tasks; Simple Spread remains an infrastructure regression task.

**Current status, September 8, 2026:** deliverables 1–3 are implemented and the
complete CPU v3 confirmation gate passed. The CPU baseline is promoted through
a separate frozen copy with only its status changed; CPU v1 remains failed and
CPU v2 remains NOT CONFIRMED. See
[v3 evidence and interpretation](PCP_CONFIRMATION_CPU_V3_RESULTS.md).
Deliverable 4's implementation, controls, intervention/cost evaluator, typed
promotion gate, preservation queue and statistical analysis are validated.
All eight real 6k condition smokes and the corrected revision-2 independent
launch review passed. Preparation evidence was archived with full read-back
verification. The 40-row queue started September 8 at 21:33:45 UTC with two
pinned CPU workers; its initial storage check passed at 9.6 GiB free.
Two rows are trained and archived, with full training-integrity, action/domain
and checkpoint audits passed. Their held-out evaluations are suspended incomplete;
38 rows remain unstarted. Remaining training, held-out banks and the paired
contrast remain pending; restore at least 8 GiB before staged resumption.
The [execution record](PCP_VISIBILITY_BUDGET_STATUS.md) gives the exact scope and
commands. Earlier failed preflight, withdrawn eligibility and storage-blocked
records are preserved. MAPDN remains deliverable 5.

**Historical status, September 6, 2026:** deliverables 1 and 2 are implemented.
Deliverable 3's two repaired CPU runs and final audits are complete, but its
full confirmation gate is **not confirmed** because mandatory 480k checkpoint
artifacts were not preserved. The final dated record below and
[results review](PCP_CONFIRMATION_CPU_V2_RESULTS.md) supersede earlier launch
snapshots. There is no frozen-protocol promotion or deliverable-4 launch.

## First deliverable: trustworthy intervention evaluation and PCP inputs

**Completed and validated: 577 tests passed.** The first implementation covers
the roadmap's first completion gate. This is
an engineering validation, not a new learning-feasibility or convergence result.

| Contract | Implementation and evidence |
|---|---|
| Same-input influence | A separate evaluator clones each declared group input, recomputes both channel conditions, and never steps the simulator. Optional replay/context masks are retained; stale outputs and other groups are excluded. Inputs, keys, distribution, count, and SHA256 are exportable. |
| Paired episode reliance | Fresh one-world environments from an explicit episode seed bank; exact requested episode count; recorded IDs, seeds, initial-input hashes, lengths, termination/completion, and returns. Missing/reordered/duplicate IDs and mismatched initial conditions fail. |
| Dominant severing | A temporary sender intervention intersects ordinary availability. All-true replay masks cannot defeat it. Fully severed channels consume no RNG; original channel objects and intervention state survive exceptions. |
| Reset history | Full and partial PCP resets clear wander state and noise cursors. Partial reset leaves other worlds and their future disturbance schedules intact. |
| Exogenous disturbances | PCP noise has private streams and explicit per-world episode/prey/timestep indexing, including rollover beyond the initial 256-step block. Matched actions are tested after unrelated channel draws and inserted evaluation. |
| Evaluation isolation | A thin BenchMARL Experiment subclass wraps evaluation without copying its optimization loop or losses. It preserves CPU/Python/NumPy, VMAS's class-shared state, initialized CUDA state, and private channel RNG on success and exceptions. Evaluation sampling uses its own reproducible namespace. |
| Actor information | `actor_observation_keys` is an explicit allowlist. Undeclared spec leaves, privileged keys, and overlapping context fail at construction. Runtime extra diagnostics/state do not enter the actor. |
| Controlled critic | An explicit root `state` leaf feeds the unchanged centralized BenchMARL MLP. It contains agents' absolute positions/velocities, landmark positions, and scripted-prey wander directions in fixed world order. It excludes rewards, visibility, future noise, and diagnostic labels. |
| Controlled observation width | Global and radius-limited prey visibility keep the same fields and flags: 17 predator features in the default 3-predator/1-prey layout. Changing visibility does not change the physical critic state. |
| Group provenance | Actor, communication, and critic parameters are counted separately for each group. The default PCP critic state has 22 features; predator/prey critics have 19,585/193 parameters. Identity remains parameterless. |

Key regression tests live in
[`test_paired_pcp_evaluation.py`](../src/tests/test_paired_pcp_evaluation.py),
[`test_pcp_randomness.py`](../src/tests/test_pcp_randomness.py),
[`test_information_contracts.py`](../src/tests/test_information_contracts.py),
[`test_saliency.py`](../src/tests/test_saliency.py), and
[`test_channel.py`](../src/tests/test_channel.py). Existing real MAPPO/QMIX
optimization and stored-log-probability replay checks remain part of the gate.

## Interpretation and compatibility

Corrected task identities are `pcp_rng_v2` and `pcp_visibility_v2`; the
physical-state critic and shared evaluation/channel RNG changes also belong to
the new protocol. No old suite should receive rows from this source as though
they had the same scientific semantics. This includes Simple Spread when its
old evaluation calls previously advanced VMAS's shared RNG. Identity's tensor
operation and actor architecture/parameter equality remain unchanged; an entire
historical training trajectory is a different claim.

The 795630 runs remain historical calibration. Gamma 0.95, entropy 0.1, and ten
passes remain a candidate requiring fresh confirmation with the whole corrected
protocol. A full critic checkpoint from the old concatenated-observation critic
is incompatible with the new physical-state critic. An old globally visible
16-feature actor is incompatible with the corrected 17-feature layout. No
historical results or imported checkpoints were rewritten.

Measurement schema 2 distinguishes same-input influence from paired full-episode
reliance. Schema-1 intervention files remain on disk but new aggregation warns
and omits them instead of mixing incompatible measurements. Prefix rollouts
require an explicit API opt-out and cannot be aggregated as complete episodes.
The default influence dataset comes from live-policy trajectories; it is not
automatically a held-out or representative state bank. The standalone evaluator
accepts another explicitly named dataset or mixture.

Channel reliance is a property of a frozen trained policy. It does not establish
that a separately trained local policy cannot succeed. The retained
`return_delta_fraction` is a legacy descriptive ratio whose interpretation
changes with reward origin; it is not a cross-domain necessity threshold.

## Validation and remaining work

Validation runtime: macOS arm64 CPU, Python 3.12.13, PyTorch 2.13.0,
BenchMARL 1.5.2, TorchRL 0.11.1, TensorDict 0.11.0, VMAS 1.5.2.
CUDA guards are implemented but CUDA execution is unavailable on this host.
RNG contexts assume serialized environment calls within each worker process.
They do not turn native checkpoints into exact optimizer/RNG resume points.
Recurrent policies are rejected by the new evaluators pending explicit
history/hidden-state replay contracts.

Final gate: **577 passed, 98 warnings in 127.17 seconds**; Ruff, compileall, and
git diff checks pass. Warnings include upstream deprecations and the existing
missing PCP typed-task-schema warning, which belongs to deliverable 2.
A separate smoke check strictly reloaded a real untrained PCP actor, ran three
six-step stochastic paired episodes through the CLI, wrote schema-2 JSON and a
content-addressed
`saliency_inputs_<hash>.pt` dataset, and reloaded the dataset with a matching
SHA256. Temporary smoke artifacts were removed. No full-horizon scientific
training was launched.

This was the deliverable-1 stopping point. Deliverable 2 below replaces the
stale comparison YAMLs and implements finite configuration validation, strict
schemas, launch gates, and group-aware diagnostics. PCP domain contact metrics
remain part of the next confirmation work.

MAPDN's adapter was inspected read-only. Its local observation protections,
per-instance NumPy RNG, history handling, and solver-failure/time-limit routing
remain intact. Its BenchMARL binding, real-data split manifests, transition
metrics bridge, throughput measurement, and real-data learning gate remain
roadmap deliverable 5. Existing MAPDN worktree changes belong to the user.


## Second deliverable: strict specifications, launch gates, and health metrics

**Completed and validated September 5, 2026: 754 tests passed.**
The [protocol guide](PCP_PROTOCOL.md) describes the
concrete candidate and its review/submission workflow.

| Contract | Implementation |
|---|---|
| Strict schemas | Task, experiment, algorithm, actor, critic, grouped-model, channel and suite fields are validated before construction; unused flat/grouped keys and invalid finite/range values fail. A typed PCP TaskConfig replaces the missing-schema warning. |
| Saved reconstruction | Effective defaults are materialized without constructing random models. Manifests and managed resolved YAML carry full specifications and scientific hashes. Analysis strictly reloads saved specs instead of reapplying old overrides to current YAML. |
| Whole protocol | `configs/protocols/pcp_corrected_v1.yaml` contains task/RNG/input/reward versions, full six-method definitions, critic, optimizer/evaluation/checkpoints, trained/measured groups, seven declared factors, seeds, runtime constraints, criterion and evidence references. |
| Launch enforcement | Manifest source/protocol/configuration hashes, stage/seed/factor, decision and evidence-file hashes are checked before managed output/training. Actual actor/critic classes, shapes/counts and task contracts must match the manifest. Only output placement may change within a protocol. |
| Submission enforcement | Bash wrappers preflight before calling sbatch; spooled workers repeat approval and runtime checks before srun. Historical pilot execution is retired. Direct sbatch bypasses the login branch but cannot bypass worker training checks. |
| Phase accounting | Collection, optimization replay and evaluation use explicit channel phases. Both stochastic and deterministic evaluation respect the intervention. Communication sums are weighted by transitions and replay cost remains separate from deployed cost. |
| Group health | Actor loc/scale/action quantiles, saturation, value/target/advantage moments, PPO diagnostics, raw gradients, encoder/contribution norms, ratio, gate saturation and optional normalization gain carry group and denominator information. Predator audits cannot absorb prey diagnostics. |
| Update guard | Each collected MAPPO group replays its stored behavior log probabilities before any update. Supported PyTorch loss/gradient/optimizer hooks reject non-finite inputs before the affected optimizer step and preserve managed failure evidence. |
| Checkpoint audit | Explicit group selection, exact requested episode count through fresh one-world environments, strict saved weights/task compatibility, recorded runtime provenance and explicit analysis-only mismatch acknowledgement. |

Independent real model inspection checked PCP MLP, Identity, Attention and Graph.
Critic parameter shapes agree with the framework's functional parameter storage:
19,585 predator critic parameters and 193 prey critic parameters in the default
layout. Identity retains zero communication parameters and exact framework-MLP
numerical equivalence. No MAPPO loss or optimization loop was copied.

The candidate uses gamma 0.95, entropy 0.1, ten passes and a fixed 600k-frame
budget. Confirmation uses new seeds 10/11; planned comparison seeds are 20–24,
with ablations on 20–22. Two confirmation, 30 main, and 201 ablation rows form
an inspectable catalogue in disjoint `*_corrected_v1` namespaces. None is
approved by generating a manifest. Width 64 and rounds 2/3 require additional
factor-specific evidence. No real approval decision was created, and no
full-horizon experiment or Slurm job was launched.

The protocol records that the scripted prey's unused actor/critic are still
optimized. Removing them changes training scheduling/RNG and is deferred to a
separately validated version. PCP contact/visibility counters, fresh learning
confirmation and its action/domain audit are next (deliverable 3). MAPDN's
binding, splits, real-data metrics and learning gate remain deliverable 5.
The runtime gate pins the historical CUDA package/thread environment; CUDA
execution is still untested locally. Review evidence integrity is machine-checked;
its scientific sufficiency remains the reviewer's judgment.

Integration findings included unknown grouped keys silently ignored by old
configuration merging, missing critic defaults in manifest explanatory exports,
ambiguous schema booleans/guarded-factor lists, and the Slurm spool-directory
lookup defect. All received regression coverage. Protocol promotion or any
source edit invalidates old bindings by design; regenerate/review after the
scientific source is settled. Native checkpoint resume remains inexact.

Reproducibility fingerprints for this implementation (not approvals):

```text
source_commit       40a97a6 (dirty, uncommitted implementation)
source_sha256       347e6dcaad7d990cb38d9ac63db0befcf02b232028fafa091d45ff37a94f73e1
protocol_id         pcp_corrected_v1
protocol_sha256     a8da84c0d388c4ab60e73d035ad5c0b2b4a5aa8e043384a1c827e84695a75c69
protocol_status     candidate
launch_decisions    none
```

The local two-row confirmation, 30-row main, and 201-row combined ablation
manifests were generated and all 233 full specifications survived export/reload,
matching these fingerprints and their declared factors. All rows are pending.
The actual confirmation preflight rejected the missing approval with exit 2.
Manifests are ignored local artifacts; use the planning script to recreate them.


Final deliverable-2 gate: **754 passed, 167 warnings in 402.87 seconds**, with
OMP/MKL/VECLIB/NUMEXPR threads pinned to one on the local CPU environment.
Ruff, compileall, all 12 Slurm/shared-shell syntax checks, Markdown status links,
and git diff checks pass. Warnings are upstream deprecations/future warnings;
the missing typed-PCP-schema warning is eliminated. The full gate includes real
6,000-frame MAPPO/QMIX and PCP optimization, exact Identity equivalence, behavior
log-probability replay, group/phase diagnostics, strict configuration/protocol
rejection, final validation-budget cases, and simulated Slurm spool workers.
An additional 11 isolated mocked PCP analysis-shell cases verified failure
propagation, independent-suite continuation, missing/unfinished skips, and
historical metrics-only mode without running real analysis or Slurm.

No source changed during the final full suite. The initial full run's 17
failures were stale PCP launcher assertions; those were replaced with the
current executable protocol checks before the passing full rerun. All changes
remain uncommitted alongside preserved earlier work.

## Third deliverable: corrected PCP confirmation instrument and CPU execution

**Engineering status, September 5, 2026: implemented and validated; fresh
learning confirmation pending.** Two full 600,000-frame Identity rows, seeds
10 and 11, are being prepared under the separately versioned
[`pcp_corrected_cpu_v1` candidate](../configs/protocols/pcp_corrected_cpu_v1.yaml)
and [CPU confirmation suite](../configs/sweeps/pcp_candidate_confirmation_cpu.yaml).
No full-horizon result or confirmed learning protocol is established by this
status entry. The deliverable-2 fingerprints and test counts above describe that
earlier implementation; they are preserved as historical records, not current
launch bindings.

The prespecified [confirmation plan](PCP_CONFIRMATION_PLAN.md) fixes both seeds,
the 600k budget, the selection criterion, and the action/domain audit before new
outcomes are examined. Gamma 0.95, entropy 0.1, ten minibatch passes and the
128-episode deterministic evaluation setting are unchanged. The plan is held
unchanged during review and execution, with SHA256
`5d0928d07ba01b9abb77f64dc8c9cd88fdca4459de61d96be178002d890aff24`.
It is evidence for review, not itself a launch decision.

| Contract | Deliverable-3 implementation |
|---|---|
| Time limit and bootstrap | `pcp_time_limit_v1` disables the raw VMAS horizon and applies a TorchRL `StepCounter(max_steps=100)`. The horizon now sets `done=true`, `truncated=true`, `terminated=false`; the unchanged MAPPO path therefore retains its time-limit bootstrap. The former VMAS limit marked termination and suppressed this bootstrap. |
| Transition measurements | `pcp_transition_metrics_v1` records read-only contact, distance, boundary, collision and visibility information from the transition that generated reward. A pre-reward snapshot preserves accounting even when a task configuration respawns prey. Counters do not consume RNG or enter actor/critic inputs. |
| Reward accounting | For this unshaped, shared-reward, no-respawn candidate, mean predator return equals ten times predator–prey contact-pair transitions. Repeated contacts are not distinct captures; identical team reward copies are counted once. Shaped or unshared configurations need their own accounting reference. |
| Frozen-policy evidence | The final actor is strictly reconstructed from its saved specification and checkpoint. Action and domain audits cover 32 complete 100-step episodes in each of RANDOM and DETERMINISTIC modes, using the same held-out seed bank 100000–100031 for both training seeds. |
| Confirmation validator | Exact two-seed coverage, saved source/protocol/specification/runtime bindings, completed 600k/100-iteration budgets, evaluation schedule and episode counts, finite health metrics, checkpoint identity and matching audit evidence are checked. Missing or malformed evidence fails. A passing integrity report still requires scientific review. |
| Learning summaries | The report exports the full curve, final six-point return, normalized AUC, and `early_window_mean`, `late_window_mean`, and `early_late_delta` for return and domain measures. Each seed must improve from its first six evaluations to its last six under the prespecified review; no automatic convergence or approval verdict is issued. |

The horizon correction is a scientific change requiring fresh confirmation.
The historical 795630 calibration cannot confirm the new bootstrap targets or
be pooled with these rows. Both predator and scripted-prey groups remain
optimized, the measured group remains `adversary`, and Identity remains
parameterless. No MAPPO loss or optimization loop was copied or retuned.

CPU execution was selected before observing fresh results because current HPC
access was not established. The explicit variant pins Python 3.12.13, PyTorch
2.13.0, BenchMARL 1.5.2, TorchRL 0.11.1, TensorDict 0.11.0 and VMAS 1.5.2,
with CPU sampling/training/buffering and one torch/OMP/MKL/VECLIB/NUMEXPR
thread. The separate CUDA candidate retains its Python 3.11.4,
PyTorch 2.13.0+cu126 and four-thread contract and remains unconfirmed. A CPU
confirmation verdict would apply only to its own declared runtime, not to CUDA
or MPS, and would not justify pooling runtimes.

The [candidate runtime check](../scripts/smoke_pcp_candidate.py) completed its
current-source `retry01` run with one 6,000-frame batch, retaining the exact
candidate architecture, optimizer, 128-episode evaluation and runtime settings.
All 4,106 logged values were finite; all 128 evaluation episodes completed as
time-limit truncations with `terminated=false`; the maximum reward-accounting
error was zero. The [preflight evidence](../results/pcp_candidate_confirmation_cpu_v1/preflight_review.json)
preserves the test logs and hashes the smoke status, metadata, resolved
specification, checkpoint and metrics. The
[candidate decision](../configs/protocols/approvals/pcp_corrected_cpu_v1_confirmation.json)
records the actual reviewer as `Codex (assistant engineering review)` and binds
that evidence plus the unchanged plan. Its scope is the user-requested two-row
deliverable-3 confirmation; it records neither human approval nor successful
learning, and does not authorize comparison or ablation rows. The
[audit script](../scripts/audit_pcp_confirmation.py) writes separate action and
domain evidence plus an explicit failure record. The
[confirmation script](../scripts/confirm_pcp.py) consumes both audit files and
exports a review-required report; it never issues an approval or launches the
main comparison. See [PCP_PROTOCOL.md](PCP_PROTOCOL.md) for usage.

The observed source verification is a composite gate: the full run had **810
passing tests and one failure from a stale Slurm expectation**. That expectation
was corrected, and the subsequent targeted regression set passed **55 tests**.
The current confirmation-validator tests separately passed **32 tests**. This
records the actual checks performed rather than describing them as a subsequent
all-green full-suite rerun. Coverage includes real PCP optimization, corrected
time-limit flags, domain accounting and reset behavior, strict audit bindings,
and rejection of incomplete or manipulated confirmation evidence. The sole
stale assertion was a test expectation, not a failed training run.

Current deliverable-3 launch bindings, distinct from the historical
deliverable-2 fingerprints above:

```text
source_sha256       73bc2cb332c0d4ccd5c6afb893b1e929f385b4cabfc6cb743c99eb6d0ec9d90c
protocol_id         pcp_corrected_cpu_v1
protocol_sha256     d61d39f0e47a234e47a7fd6c3c3bea2c3ece91f586917e284608fa938e840bd0
protocol_status     candidate
review_scope        two Identity CPU confirmation rows only, seeds 10 and 11
full_run_status     pending launch in this engineering snapshot
```

The remaining deliverable-3 work is to execute both bound CPU rows, preserve
all attempts, audit both final policies, and review learning and domain behavior
against the unchanged plan. A failed seed cannot be replaced, an unexplained
behavior remains review-pending, and two fixed-budget rows do not establish
convergence or a communication-necessity result. No main or ablation launch is
authorized by this engineering record. MAPDN work remains a separate roadmap
deliverable.


## Third deliverable: final CPU execution and review, September 6, 2026

**Execution and analysis finished; full confirmation NOT CONFIRMED.** Both CPU
v2 Identity rows completed 600,000 frames and 100 updates under the reviewed
source/runtime. Each recorded 51 evaluations with 128 complete 100-step episodes
and 220,579 finite logged scalars. Seed 10's first-six return mean rose from
0.208333 to 489.674479; seed 11 rose from 0.572917 to 550.638021. Normalized AUCs
are 148.538905 and 179.148516 over the actual 6k–600k interval. Both late diagnostic
windows also improve. The budget remains fixed at 600k; there is no convergence
claim or longer-horizon extension.

The [run-artifact validator](../results/pcp_candidate_confirmation_cpu_v2/confirmation_report.json)
passes completion, source/specification/runtime, finite optimizer/replay, action
and domain requirements. Final action and domain audits each use 32 complete
episodes per mode and held-out seeds 100000–100031. All 76,800 measured action
scalars and all locations/scales are finite, scales are positive and saturation
above 0.999 is zero. Domain reward accounting is exact. Sustained and simultaneous
contacts explain return growth; boundary occupancy is substantial. These are
repeated-contact outcomes, not unique captures or proof of communication necessity.

The initial CPU v1 runs remain failed at the 564k collection, after optimization
through 558k. The narrowly scoped guard repair retains the whole-batch fast check
and original tolerance, then recomputes disputed batches at collection shape.
Only `experiments/health.py` differs between the two package snapshots. The
[saved 564k replay audit](../results/pcp_candidate_confirmation_cpu_v2/replay_snapshot_audit.json)
reproduces both whole-batch errors (1.9669533e-5 and 1.4424324e-5) with exactly zero
collection-shape error. The [metric-prefix comparison](../results/pcp_candidate_confirmation_cpu_v2/metric_prefix_invariance.json)
verifies all 201,665 retained scientific scalars per seed exactly through 558k,
with no earlier fallback. No tolerance was widened and no MAPPO update changed.

The mandatory complete invariance check nevertheless fails. The assistant
performed a live 480k comparison and observed equality of all 30 tensors and
eight TensorDict metadata values for both seeds, but did not archive the new
checkpoint files or their hashes. BenchMARL's periodic final save and subsequent
end save pruned 480k under `keep_checkpoints_num=2`. The
[retention incident](../results/pcp_candidate_confirmation_cpu_v2/checkpoint_retention_incident.json)
reproduces this behavior using the installed method on temporary diagnostic
state. The earlier observed result cannot supply independently repeatable
checkpoint evidence. The strict comparator stays failed, the immutable plans
are unchanged, and no files are fabricated or substituted.

Engineering verification on the unchanged repaired source: **821 tests passed**
in the full pre-launch gate. The subsequent standalone snapshot auditor first
passed 20 tests, then a script-only compatibility repair for valid TensorDict
metadata passed **24 tests**. These overlapping runs are reported separately,
not added into a fictional combined full-suite count. The prefix comparator's
36 synthetic/read-only cases also pass. Its final rejection of missing real
checkpoints is correct. The initial rejected snapshot-audit invocation remains
preserved alongside the fixed reader's passing audit. Ruff/compilation/diff
checks pass; final plots explicitly carry the non-confirmation verdict.

The [compute record](../results/pcp_candidate_confirmation_cpu_v2/compute_accounting.json)
separates 2,328,000 collected confirmation frames (2,316,000 optimized), including
the preserved failed attempts, from 18,000 managed smoke training frames,
2,508,800 online confirmation evaluation transitions, 38,400 smoke evaluation
transitions and 25,600 final action/domain audit transitions. Tests and frozen
probes are separate diagnostic compute, not extra independent training seeds.
Two reused training seeds remain the experimental units.

The [full results](PCP_CONFIRMATION_CPU_V2_RESULTS.md) and
[scientific review](../results/pcp_candidate_confirmation_cpu_v2/scientific_review.json)
record separate learning, run-integrity, replay and missing-evidence verdicts.
Both CPU protocol files stay candidate artifacts with their original hashes;
there is no promotion or main-stage approval. The next prerequisite is a
reviewed retention/archive mechanism with a bounded test covering both final
saves, before any separately declared confirmation attempt. No further training
was launched. CUDA confirmation, deliverable 4 and MAPDN remain pending.
