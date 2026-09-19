# PCP CPU v2 confirmation: completed runs, incomplete confirmation evidence

**September 6, 2026: NOT CONFIRMED under the unchanged replay-repair plan.**
Both CPU v2 runs completed 600,000 frames, improved on the declared learning
measurements, and passed the final action/domain integrity audit. The required
repaired 480k checkpoint files were deleted by the existing framework retention
cleanup before they were archived. The final mandatory state-byte comparison
therefore cannot be repeated, and the frozen prefix comparator correctly fails.
This evidence-preservation failure blocks confirmation despite the observed
learning. The original CPU v1 cohort remains failed and not confirmed.
Deliverable 3 execution and analysis are recorded here; its confirmation gate
has not passed. No deliverable 4, main comparison, ablation or replacement
confirmation cohort is approved or launched by this report.

The governing documents are the unchanged
[original confirmation plan](PCP_CONFIRMATION_PLAN.md) and the separately
prespecified [replay-repair plan](PCP_CONFIRMATION_REPLAY_REPAIR_PLAN.md).
The [CPU v2 protocol](../configs/protocols/pcp_corrected_cpu_v2.yaml),
[preflight review](../results/pcp_candidate_confirmation_cpu_v2/preflight_review.json)
and [confirmation launch decision](../configs/protocols/approvals/pcp_corrected_cpu_v2_confirmation.json)
bind the repaired source and bounded rerun. The recorded decision is an
assistant engineering review for this confirmation stage; it is not a human
research adjudication or a presumption of successful learning.

The scientific question remains whether the corrected PCP Identity baseline
can complete the declared budget with valid learning, action and domain
evidence. Each training seed is an experimental unit. The 128 evaluation
episodes at a checkpoint estimate that seed's behavior; they are not 128
independent training replications. The comparison is fixed at 600,000 training
frames per run and makes no convergence claim.

| Declared setting | CPU v2 design, unchanged scientifically from CPU v1 |
|---|---|
| Training seeds | 10 and 11, both restarted from initialization after instrument repair |
| Method and task | Parameterless Identity communication; three predators and one scripted prey |
| Actor and critic inputs | Predator observation width 17 at sensing radius 1.0; centralized physical state width 22 |
| Optimized and measured groups | Both `adversary` and `agent` optimized; only predator return measured; the scripted prey discards its policy action |
| Optimization | MAPPO; gamma 0.95; entropy coefficient 0.1; GAE lambda 0.9; learning rate 5e-5 |
| Data reuse | 6,000-frame collections; ten environments; 400-frame minibatches; ten minibatch passes |
| Per-run training budget | 600,000 frames and 100 iterations |
| Online evaluation | 128 complete deterministic 100-step episodes at 6k, then every 12k through 600k: 51 evaluation points |
| Final-policy audit | 32 complete 100-step episodes in each of RANDOM and DETERMINISTIC modes, using held-out seeds 100000–100031 |
| Runtime | CPU for sampling, training and buffering; one thread; Python 3.12.13, PyTorch 2.13.0, BenchMARL 1.5.2, TorchRL 0.11.1, TensorDict 0.11.0, VMAS 1.5.2 |

These are the same reused training seeds as CPU v1, not previously unseen
seeds selected after the failures. CPU v2 allocates **an additional 1,200,000
training frames**, 600,000 per seed. The original attempts already collected
1,128,000 frames and completed optimization through 558,000 frames per seed.
CPU v2 actually consumed the full additional 1,200,000 training frames.
Evaluation, smoke, probe and audit compute are accounted for separately. The
runs start from initialization;
the retained native checkpoints do not contain the exact optimizer/RNG state
needed to claim scientific continuation. Concurrent worker wall times are not
a clean compute-cost benchmark.

Both CPU cohorts use the time-limit correction implemented before CPU v1.
The raw VMAS horizon formerly combined the time limit with termination, which
suppressed value bootstrapping. The corrected `pcp_time_limit_v1` factory uses
raw `max_steps=None` and a TorchRL `StepCounter(max_steps=100)`. At the horizon,
`done=true`, `truncated=true` and `terminated=false`; the unchanged MAPPO path
retains its time-limit bootstrap. This semantic change requires fresh evidence
relative to the earlier PCP calibration. It is not a change introduced by the
CPU v2 replay repair. The implementation is in the
[PCP task factory](../src/commstudy/tasks/vmas/tasks.py).

The task retains shared, unshaped, repeated-contact reward and no respawn.
A predator–prey pair is in contact when center distance is strictly less than
the sum of radii, 0.125 in the declared world. Every contacting pair contributes
10 on each transition it remains in contact. The measured team return is the
mean of the predators' identical shared rewards, so
`predator_return = 10 × contact_pair_steps`. Summing those three shared copies
would incorrectly triple the return. Contact does not end an episode, and
contact-pair steps are not counts of distinct captures.

The [domain measurement implementation](../src/commstudy/tasks/vmas/domain_metrics.py)
publishes read-only team values repeated in predator `info`; aggregation reads
one copy after checking agreement. It records any contact, contact duration,
first contact with no-contact episodes censored at the horizon, simultaneous
contact by at least two predators on the same prey, contacting predator counts,
nearest-prey distances, obstacle and teammate overlaps, and boundary occupancy.
The boundary region is explicitly `max(abs(position)) >= 0.9 * bound`, with
predator/prey occupancy reported as per-transition fractions and then episode
means. Visibility bins count prey seen by zero through three predators at
radius 1.0. Directed sender-visible/receiver-blind opportunities describe
available private information independently of communication topology.

Measurements snapshot the post-physics state before reward-side effects.
They consume no RNG and enter neither actor observations nor critic state.
An independently computed `expected_predator_reward` also supports the accepted
shaped or unshared task variants, but this confirmation uses only the declared
shared, unshaped identity above. Sustained contact, simultaneous contact and
boundary behavior must be interpreted together; a high return alone does not
establish communication necessity or a distinct-capture success rate.

The preserved [CPU v1 failure summary](../results/pcp_candidate_confirmation_cpu_v1/failed_attempt_summary.json)
and [blocked confirmation report](../results/pcp_candidate_confirmation_cpu_v1/confirmation_report.json)
record two rejected runs. The guard failed after collecting the 564k batch and
before optimizing it. Each run has 93 completed logged updates through 558k,
47 evaluations ending at 552k, and 202,549 finite logged scalars. Finite logs
and improving observed returns do not satisfy the missing completion and
replay requirements.

| Original CPU v1 seed | Managed outcome | Reported maximum replay error at rejection | Last evaluation frame | Last observed predator return |
|---|---|---:|---:|---:|
| 10 | Failed; not confirmed | 1.9669533e-5 | 552,000 | 437.890625 |
| 11 | Failed; not confirmed | 1.4424324e-5 | 552,000 | 506.25 |

These returns describe failed prefixes, not final 600k outcomes. The
[prefix outcome figures](../results/pcp_candidate_confirmation_cpu_v1/failed_evidence_figures/pcp_outcomes.pdf)
and [prefix health figures](../results/pcp_candidate_confirmation_cpu_v1/failed_evidence_figures/pcp_health.pdf)
retain both failed labels and stop at the actual observed frames. Original
statuses, metadata, metrics, protocol, approval and native 360k/480k checkpoints
remain preserved. The exact historical 564k failing batch and policy state
were not saved, which limits retrospective attribution.

The [constructed-support probe](../results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/support_point_probe.json)
demonstrated a guard false-rejection mode on each unchanged 480k actor and
its saved observations. A declared 33-by-33 Normal support grid spanning -4 to
4 per action coordinate constructs actions and behavior log probabilities
at collection width ten. Replaying the actor at the whole-batch shape rejects
some scalars under the original tolerance, whereas replay at collection shape
is bitwise exact. Parameter, RNG and input-artifact checks remain unchanged.

| Frozen 480k actor | Whole-batch maximum absolute error | Violating scalars | Maximum error / allowed error | Collection-shape error |
|---|---:|---:|---:|---:|
| Seed 10 | 1.4781952e-5 | 2 | 1.42356 | 0 |
| Seed 11 | 1.9073486e-5 | 2 | 1.71835 | 0 |

The [natural-draw probe](../results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/natural_draw_probe.json)
tested five draw seeds per frozen actor: all ten whole-batch checks passed
the original tolerance and all collection-shape checks were bitwise exact.
The [distribution-only diagnostic](../results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/distribution_only.md)
found cache size zero and no violations with fixed location/scale tensors.
These probes establish the existence of shape-dependent actor roundoff that
can amplify in log probabilities. The selected support grid does not estimate
its natural frequency or reproduce the unavailable historical 564k batch.
The completed CPU v2 snapshot audits below support this mechanism on the saved
564k batches. The full prespecified attribution gate remains incomplete because
the repaired 480k checkpoint files were not preserved.

The [source-scope reconstruction](../results/pcp_candidate_confirmation_cpu_v2/source_repair_scope.json)
shows that only `src/commstudy/experiments/health.py` changed in the production
package between the cohorts. The
[repair evidence record](../results/pcp_candidate_confirmation_cpu_v2/replay_repair_evidence.json)
describes a retained whole-batch fast path followed, only on failure, by actor
recomputation over slices of an unambiguous named `time` axis at collection
shape. Stored actions and masks remain fixed. Both paths retain the original
elementwise condition:

```text
abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)
```

The repair does not widen tolerance or substitute stored distribution
parameters for actor recomputation. Non-finite data, ambiguous layouts and
persistent collection-shape mismatches still reject before optimization.
The actor, critic, actions, collector, optimizer, PPO loss and update schedule
are unchanged. A first accepted fallback records an inspectable diagnostic
snapshot; it is not a resume checkpoint. The preserved model/RNG/statistics
contract is part of the engineering validation.

| Engineering evidence | Established result | Scope and remaining limit |
|---|---|---|
| [Full source suite](../results/pcp_candidate_confirmation_cpu_v2/full_suite.log) | 821 passed; 174 warnings; 616.36 seconds | Pre-launch repaired-package gate; includes the focused health tests |
| [Focused health checks](../results/pcp_candidate_confirmation_cpu_v2/focused_health_tests.log) | 23 passed; 10 warnings; 22.46 seconds | Corruption rejection, real fixed-actor regression, preservation and real 6k optimization; not added to the full-suite count |
| [Exact-runtime smoke](../results/pcp_candidate_confirmation_cpu_v2/preflight_review.json) | 6,000 frames; 4,116 finite logged values; 128 complete truncations; zero terminations and reward-accounting error | Only `max_n_frames` reduced from the candidate; runtime evidence, not learning confirmation |
| [Prefix comparator validation](../results/pcp_candidate_confirmation_cpu_v1/prefix_comparator_validation.json) | 36 synthetic/read-only cases passed | Each original retained 201,665 prefix metrics and passed a 30-tensor/eight-metadata native self-check; the final actual comparison fails on missing repaired 480k files |
| [Initial standalone snapshot-audit validation](../results/pcp_candidate_confirmation_cpu_v2/replay_snapshot_audit_validation.before_metadata_fix.json) | 20 passed; 18 warnings; 34.80 seconds | Separate post-gate run; the first actual CLI invocation subsequently rejected valid prey TensorDict metadata |
| [Final standalone snapshot-audit validation](../results/pcp_candidate_confirmation_cpu_v2/replay_snapshot_audit_validation.json) | 24 passed; 26 warnings; 16.88 seconds | Script-only metadata compatibility repair; both actual 564k snapshots subsequently passed; package source unchanged |

The recorded 821-test full-suite result is kept distinct from the later
standalone runs and the 36-case comparator verification. There is no combined
full-suite claim. Ruff, compilation and whitespace checks passed in preflight.
The final standalone auditor accepts only the expected, exactly typed
TensorDict batch/device metadata and still requires actual tensor byte, dtype
and shape equality. Its script SHA256 is
`17bcdde41492d1a1efaabd45daab0252b99fb2bb28b98cec4dcff79d95184403`;
the rejected earlier invocation and earlier validation remain archived. This
analysis-script repair changed no training package source, plan or protocol.

The binding hashes below distinguish canonical protocol-content hashes from
file-byte hashes and the package-source fingerprint. They identify the reviewed
design and implementation; they are not performance results.

| Binding | SHA256 |
|---|---|
| CPU v1 package source | `73bc2cb332c0d4ccd5c6afb893b1e929f385b4cabfc6cb743c99eb6d0ec9d90c` |
| CPU v2 package source | `7124976af366d28d40a3ccf0ebce62aa2b4019783d13974b9f98dd1c316af262` |
| CPU v1 canonical protocol content | `d61d39f0e47a234e47a7fd6c3c3bea2c3ece91f586917e284608fa938e840bd0` |
| CPU v2 canonical protocol content | `6f99321ab3b89222dccf2140183f8c387d3268d174d66bca78a6bbcc88a9c87a` |
| Original confirmation plan file | `5d0928d07ba01b9abb77f64dc8c9cd88fdca4459de61d96be178002d890aff24` |
| Replay-repair plan file | `19e9e860bff67069bdacbbd9e158f96f5f3dbefb6f0abc2b4ea58b476abdf6f4` |
| Prefix comparator file bound at launch | `44d2ac34c675ca8dfdc0f8dc33bd14a7676b536d6416bf44cf128c4ca91e1e9f` |

The required [prefix comparator](../scripts/compare_pcp_confirmation_prefix.py)
must establish exact numeric equality through **558,000 frames inclusive**
for every recorded scientific scalar, keyed by frame, iteration, phase, group,
metric and sample. It includes both groups, per-episode domain records and
setup counts. It rejects duplicate keys, changed or missing values, and
unsupported exclusions. The only exclusions are the timestamp column,
`wall_time_seconds`, and the eight replay-diagnostic fields named below:

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

There must be no collection-shape fallback in either group before **564,000
frames**; count and slice fields must agree. The native **480k/80-update**
checkpoints must match every actor/critic loss-state tensor byte, dtype, shape
and key for both groups, including TensorDict batch/device metadata. Signed
zero differences therefore matter. Whole checkpoint serialization hashes are
recorded but need not be equal. Full resolved specifications must match apart
from `save_folder`, software/runtime declarations must agree, implementation
sources must differ as declared, and metadata/status files must preserve
original failures and new completed runs.

At **564,000 frames**, the first predator fallback for each seed must be saved
before that batch is optimized and rechecked with the
[offline snapshot auditor](../scripts/audit_pcp_replay_snapshot.py). The packet
must bind source, protocol, configuration, CSV frame/iteration and snapshot
hashes; preserve stored observations, actions, masks, distribution parameters,
log probabilities and policy weights; and reproduce both whole-batch and
collection-shape results under unchanged tolerances. The auditor also checks
stored distribution consistency and actual tensor bytes, dtypes and shapes.
A recurrence matching the rounded historical error is evidence for the
mechanism only in combination with the full invariance checks. It cannot
recover missing historical per-scalar data. An absent snapshot, earlier
fallback or unexplained difference leaves attribution unresolved.

The completed [confirmation integrity report](../results/pcp_candidate_confirmation_cpu_v2/confirmation_report.json)
has `checks_passed=true`, no integrity issues, `review_status=required`, and
`approval_decision=not_issued`. It validates completion, saved source/runtime
contracts, learning measurements and final action/domain audit artifacts.
It does not implement or waive the separate replay-repair plan's mandatory
480k preservation requirement. Its passing scope and the failed full prefix
comparison must both remain visible.

The [strict prefix report](../results/pcp_candidate_confirmation_cpu_v2/prefix_invariance.json)
has `checks_passed=false`. The separate
[partial metric-prefix report](../results/pcp_candidate_confirmation_cpu_v2/metric_prefix_invariance.json)
uses the immutable comparator's existing functions without changing its
criteria. For **each** seed it compares **201,665** retained scientific metrics
through 558k with **zero missing, extra or unequal values**. The full saved
specifications match apart from output placement, versions and declared
runtime agree, metadata/status agree, and the source hashes differ exactly as
declared. Count/slice consistency holds over the complete repaired streams;
neither group used fallback before 564k. That report explicitly records
`full_plan_invariance_confirmed=false` and binds all input hashes. Its SHA256 is
`203406a9e912d96b03629a45b454fbc69a14a31dfce5a8d85d2c50c97878fd1f`.

The [checkpoint-retention incident](../results/pcp_candidate_confirmation_cpu_v2/checkpoint_retention_incident.json)
records an earlier live root-assistant comparison: 30 loss-state tensors and
eight TensorDict metadata values matched for each seed at 480k. The command
returned equality in the conversation, but did not archive the repaired files
or their hashes. That observation is recorded retrospectively and cannot now
be independently repeated. With `keep_checkpoints_num=2`, the periodic 600k
save followed by `checkpoint_at_end` appends the same 600k path twice and
removes both the 360k and 480k paths. An isolated reproduction using the
installed BenchMARL save method confirmed this cleanup sequence without
training. Searches found no repaired 480k backup. The assistant failed to
preserve the required evidence before cleanup; the original files are not
substituted, the plans are not amended after seeing the outcome, and the
earlier observation is not promoted into a passing final artifact check.

Both [offline 564k snapshot audits](../results/pcp_candidate_confirmation_cpu_v2/replay_snapshot_audit.json)
passed with the [provenance sidecar](../results/pcp_candidate_confirmation_cpu_v2/replay_snapshot_audit.provenance.json).
At the first predator fallback, after collecting frame 564,000 and before its
optimization, the saved whole-batch maximum errors were 1.9669532775878906e-5
and 1.4424324035644531e-5 for seeds 10 and 11. These match the rounded original
failure messages. Collection-shape recomputation over all 600 time slices
was bitwise exact with zero log-probability error in each seed, under unchanged
1e-5 absolute/relative tolerance. Captured tensors and stored distribution
consistency passed, and bound input artifacts remained unchanged. Subsequent
predator fallbacks occurred at 588k/594k for seed 10 and 570k/594k/600k for
seed 11; the prey had none. This supports the replay roundoff mechanism on
preserved CPU v2 batches. The missing repaired 480k evidence leaves the plan's
full attribution gate incomplete, and the unavailable original 564k tensors
prevent claiming exact historical batch reproduction.

Return results use each training seed separately. The first six evaluations
are at 6k, 12k, 24k, 36k, 48k and 60k; the final six are at 540k, 552k, 564k,
576k, 588k and 600k. Normalized AUC is trapezoidal return over the observed
6k–600k interval divided by its 594,000-frame span. The first observation is
after one optimizer update; no time-zero performance is invented. The
[return/domain curves](../results/pcp_candidate_confirmation_cpu_v2/figures/pcp_outcomes.pdf),
[health curves](../results/pcp_candidate_confirmation_cpu_v2/figures/pcp_health.pdf),
[per-seed CSV](../results/pcp_candidate_confirmation_cpu_v2/figures/per_seed.csv)
and [figure provenance](../results/pcp_candidate_confirmation_cpu_v2/figures/provenance.json)
retain actual frames and source/report bindings. A completed-run label on a
curve describes managed execution, not a confirmation approval.

| CPU v2 run or return measurement | Seed 10 | Seed 11 |
|---|---:|---:|
| Managed status | completed | completed |
| Training frames / iterations | 600,000 / 100 | 600,000 / 100 |
| Finite logged scalars / total, both groups | 220,579 / 220,579 | 220,579 / 220,579 |
| Online evaluations | 51 | 51 |
| First-six mean return | 0.208333 | 0.572917 |
| Final-six mean return | 489.674479 | 550.638021 |
| Final-six minus first-six | +489.466146 | +550.065104 |
| Normalized AUC, 6k–600k | 148.538905 | 179.148516 |
| Final 600k evaluation return | 511.718750 | 578.671875 |
| Mean return at 408k–492k, eight points | 316.728516 | 356.718750 |
| Mean return at 504k–600k, nine points | 459.479167 | 526.050347 |

Every online evaluation contained exactly 128 complete 100-step episodes and
12,800 transitions, with all episodes truncated and none terminated. All
reward-accounting errors were zero. The declared early-to-final learning
condition is positive in both seeds. The descriptive late-window comparison
uses the eight and nine actual evaluations shown above; it is not a matched
sample-size hypothesis test. Returns continue to improve late, so these data
do not establish a plateau or convergence. The two-seed mean final-window
return is 520.156250 and mean normalized AUC is 163.843711; these summaries
do not increase the number of independent training units beyond two.

The following domain values average each seed's six final online evaluation
means; episode lengths remain 100 steps. Contact duration counts transitions
with any contact, while contact-pair steps sum contacting predator–prey pairs.
Simultaneous contact means at least two predators contact the same prey.

| Final-six domain measurement | Seed 10 | Seed 11 |
|---|---:|---:|
| Episodes with any contact | 92.448% | 92.057% |
| First-contact step, no contact censored at 100 | 18.880 | 19.096 |
| Contact duration, steps | 38.125 | 39.958 |
| Contact-pair steps | 48.967 | 55.064 |
| Simultaneous-contact duration, steps | 10.279 | 13.536 |
| Episodes with any simultaneous contact | 66.406% | 74.870% |
| Predator boundary occupancy | 37.871% | 34.198% |
| Prey boundary occupancy | 38.081% | 40.647% |
| Predator–obstacle collision-pair steps | 11.294 | 12.508 |
| Predator–teammate collision-pair steps | 23.605 | 30.836 |
| Mean nearest predator–prey distance | 0.2714 | 0.2775 |
| Mean per-predator nearest-prey distance | 0.6603 | 0.5477 |
| Prey visibility to 0 / 1 / 2 / 3 predators | 8.521% / 22.365% / 27.951% / 41.164% | 9.017% / 14.935% / 21.342% / 54.706% |
| Directed sender-visible / receiver-blind opportunities per transition | 1.0063 | 0.7255 |

By the 408k–492k window, contact probability was already 91.70% and 90.82%.
Its increase to 92.10% and 91.93% in the 504k–600k window was small compared
with the return increase. Contact duration rose from 26.407 to 36.234 steps
and from 27.718 to 38.514, while simultaneous-contact duration rose from
5.074 to 9.268 and from 7.319 to 12.668. Under repeated-contact reward, longer
and more simultaneous contact therefore explain the measured return growth
more directly than a large increase in episodes achieving any contact.

Substantial boundary occupancy and teammate/obstacle overlap accompany these
returns. Prey boundary occupancy increased across those late windows from
29.20% to 37.09% and from 33.43% to 40.53%. These observations warrant a
boundary-behavior review; they do not establish a causal strategy or successful
coordination. Likewise, sender-visible/receiver-blind opportunities show
available private information, but the trained method is parameterless
Identity. Neither these opportunities nor simultaneous contact demonstrate
that learned communication is necessary or beneficial.

The [final action audit](../results/pcp_candidate_confirmation_cpu_v2/action_audits.json)
and [final domain audit](../results/pcp_candidate_confirmation_cpu_v2/domain_audits.json)
strictly reconstructed each saved 600k policy with the saved source, resolved
specification, runtime and task contract. Each mode used the declared 32
held-out seeds 100000–100031, producing 3,200 transitions and 19,200 predator
action scalars. All actions, locations and scales were finite, all episodes
were complete truncations, all accounting errors were zero, and saturation
at `abs(action) > 0.999` was zero in every row.

| Seed / final audit mode | Mean return | Any-contact episodes | Contact / simultaneous steps | Predator / prey boundary | Maximum absolute action | Mean scale |
|---|---:|---:|---:|---:|---:|---:|
| 10 DETERMINISTIC | 588.4375 | 30 / 32 | 43.250 / 14.188 | 34.71% / 37.06% | 0.981987 | 0.251160 |
| 10 RANDOM | 571.5625 | 30 / 32 | 43.500 / 12.375 | 40.93% / 48.25% | 0.996807 | 0.248372 |
| 11 DETERMINISTIC | 675.6250 | 30 / 32 | 43.406 / 20.156 | 32.58% / 41.25% | 0.992469 | 0.245704 |
| 11 RANDOM | 512.5000 | 31 / 32 | 39.250 / 11.656 | 32.61% / 36.63% | 0.997900 | 0.242076 |

These audit returns use a different held-out episode set from the online
evaluations and should not be substituted for the declared final-six return.
The difference between modes is descriptive; the audit has two training
seeds and does not establish a population-level advantage of either mode.
Zero observed saturation is useful evidence at the declared budget, not a
guarantee of longer-horizon stability.

| Selected predator health at 600k | Seed 10 | Seed 11 |
|---|---:|---:|
| Logged transformed-policy entropy | -0.948692 | -1.204802 |
| Critic loss | 975.899658 | 1568.301147 |
| Critic explained variance | 0.199772 | 0.040813 |
| Pre-clipping actor gradient norm | 61.816837 | 60.026264 |
| Pre-clipping critic gradient norm | 287.971558 | 399.736603 |
| Approximate KL | 0.003694 | 0.002731 |

All logged values were finite, and online deterministic saturation stayed zero.
However, entropy contracted, critic losses and gradient norms rose, and
explained variance remained variable and weak at the final seed-11 point.
Continuous transformed-policy entropy may be negative; its value alone is
not a non-finite failure. These trends are cautions against extrapolating
finite 600k execution into a convergence or long-horizon stability claim.

The resulting decisions remain separate: fixed-budget execution and the
declared early-to-final learning condition passed; final action/domain
integrity passed; the retained numeric prefix matched exactly; the mandatory
archived 480k invariance evidence is missing; the saved 564k batches support
the replay roundoff mechanism, while the full prespecified attribution gate
is incomplete. **Overall confirmation remains NOT CONFIRMED.** The completed
execution/analysis record does not make deliverable 3 scientifically complete
or authorize the next experimental stage. No additional cohort is launched
to compensate for the preservation incident in this report.

The [root scientific/engineering review](../results/pcp_candidate_confirmation_cpu_v2/scientific_review.json)
and [independent artifact review](../results/pcp_candidate_confirmation_cpu_v2/independent_review.json)
reach that same separated verdict and bind the inspected evidence hashes.
Both identify assistant reviewers; neither claims human approval. The
independent reviewer separately recomputed the scalar prefix, learning
windows and audit summaries. The scientific review issues no protocol freeze
or launch approval. Before any separately declared future confirmation,
checkpoint archive survival through both final saves must be reviewed and
verified. A new cohort would have its own budget and could not recover the
deleted CPU v2 files.

The [compute accounting record](../results/pcp_candidate_confirmation_cpu_v2/compute_accounting.json)
keeps the failed and repaired attempts, runtime smokes, and analysis executions
separate. Environment transitions count simulator steps, not the three shared
predator copies.

| Recorded computation | Training frames collected | Training frames optimized | Online evaluation transitions |
|---|---:|---:|---:|
| Original failed CPU v1, two seeds | 1,128,000 | 1,116,000 | 1,203,200 |
| Repaired CPU v2, same two seeds | 1,200,000 | 1,200,000 | 1,305,600 |
| Three managed runtime smokes | 18,000 | 18,000 | 38,400 |

Final action audits executed another 12,800 environment transitions and
measured 76,800 action scalars; the separately executed domain audits used
another 12,800 transitions over the same held-out seeds. Those are two audit
executions, not additional training replications. Test-generated optimization
frames and offline probes are listed separately in the accounting record;
their total FLOPs and elapsed compute were not instrumented and are not
inferred. The full 821-test gate, historical 20-test standalone run, final
24-test standalone run and 36-case comparator validation are not summed into
a new full-suite count. Concurrent CPU wall times remain contended rather
than a clean method-overhead measurement.

A standalone [source and surviving-evidence archive](../results/pcp_deliverable3_evidence_20260906.tar.gz)
includes both cohorts, runtime smokes, audit reports, reviewed plans, source,
configuration, scripts and thesis status documents. Its embedded manifest hashes
every included file. The deleted CPU v2 480k files are explicitly missing; this
archive does not recover or replace them. The
[archive verification](../results/pcp_deliverable3_evidence_20260906.verification.json)
records the bundle hash and independent readback of every member.
