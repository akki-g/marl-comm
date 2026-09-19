# Corrected PCP candidate confirmation plan

Prepared September 5, 2026, before examining new confirmation results. This is
a proposed, prespecified review plan for roadmap deliverable 3. It is **not a
launch approval, a frozen protocol, or a report of successful confirmation**.
The source, measurement implementation, this plan, and the selected runtime
must be settled and included in the reviewed evidence before training starts.

The selected local execution candidate is
[`pcp_corrected_cpu_v1.yaml`](../configs/protocols/pcp_corrected_cpu_v1.yaml).
It preserves the scientific design of the separately versioned CUDA candidate
[`pcp_corrected_v1.yaml`](../configs/protocols/pcp_corrected_v1.yaml), which remains
unconfirmed. The execution and approval mechanism is documented in
[PCP_PROTOCOL.md](PCP_PROTOCOL.md). This review did not edit any protocol or
approval file.

## Candidate rationale and limits of the historical evidence

The read-only [795630 audit](../../results/pcp_795630_audit/per_cell.csv) covers
27 completed runs: nine gamma/entropy cells, three training seeds each, all at
600,000 frames and ten minibatch passes. Its
[per-run table](../../results/pcp_795630_audit/per_run.csv) contains 385,020 logged
values and zero non-finite values. Among the entropy-0.1 cells:

| Gamma | Mean final-window return | Across-seed standard deviation | Mean normalized AUC | Episodes with positive final-window return |
|---|---:|---:|---:|---:|
| 0.90 | 615.317 | 55.236 | 186.737 | 91.493% |
| 0.95 | 668.381 | 34.447 | 303.249 | 91.016% |
| 0.99 | 586.823 | 27.866 | 329.832 | 91.884% |

Gamma 0.95 leads on the recorded primary criterion: the seed mean of the final
`ceil(10%)` evaluation means. Gamma 0.99 leads on AUC. This supports testing
gamma 0.95; it does not establish that it is uniformly best or that its small
three-seed advantage is statistically settled. Do not switch the primary
criterion after seeing the corrected runs.

The historical gamma-0.95/entropy-0.1 per-seed final returns were 686.979,
628.633, and 689.531. These values and the approximately 91% positive-return
fraction are reference observations, **not acceptance thresholds** for corrected
semantics. Old artifacts used the previous environment/evaluation RNG path and
the previous critic information set. Their raw VMAS horizon was also reported
as a termination, causing a different value-bootstrap target. The corrected
`pcp_time_limit_v1` path removes the raw VMAS limit and applies a 100-step
StepCounter: at the time limit, `done=true`, `truncated=true`, and
`terminated=false`. This is a substantive additional protocol change, not an
administrative relabeling. Historical rows cannot confirm it or be pooled with
new runs.

The historical [CPU actor probe](../../results/pcp_795630_audit/checkpoint_cpu_probe.csv)
also argues against automatically importing the Simple Spread saturation fix.
For the selected cell, all measured actions, locations and scales were finite;
deterministic saturation above `|action| > 0.999` was zero in all three seeds.
The last logged predator entropies were approximately -1.38, -1.77, and -1.93.
Negative transformed-policy entropy alone is therefore not a failure criterion.
The probe is explicitly a CPU diagnostic of historical actors, not a CUDA
return reproduction or evidence about unobserved checkpoints.

## Fixed design and primary summaries

Run exactly two fresh Identity rows, training seeds **10 and 11**, from
[`pcp_candidate_confirmation_cpu.yaml`](../configs/sweeps/pcp_candidate_confirmation_cpu.yaml),
under suite ID `pcp_candidate_confirmation_cpu_v1` and protocol
`pcp_corrected_cpu_v1`. The original
[`pcp_candidate_confirmation.yaml`](../configs/sweeps/pcp_candidate_confirmation.yaml)
describes the separate, unconfirmed CUDA execution plan.
Seeds 20–24 remain reserved for the main comparison, with 20–22 used by the
declared ablations. Do not spend held-out comparison seeds on candidate selection.

Both rows retain gamma 0.95, entropy coefficient 0.1, ten minibatch passes,
learning rate 5e-5, GAE lambda 0.9, 6,000-frame collections, 400-frame minibatches,
ten training environments, and a **fixed 600,000-frame budget**. That is 100
collection/training iterations, not 100 individual optimizer steps. There is
no planned horizon extension, method-specific tuning, or repeat of the full
nine-cell calibration grid.

Evaluation uses 128 deterministic episodes, a 100-step task horizon, the
declared 12,000-frame interval, and `evaluation_static=false`. The existing
BenchMARL schedule records an additional first evaluation at 6,000 frames;
the historical path therefore has 51 evaluation points, not an evaluation at
frame zero. Verify the exact schedule in the bounded runtime check and reject
missing or duplicate points rather than silently interpolating them.

For each training seed, report:

- The mean of its last `ceil(10% of evaluation points)` evaluation means. With
  51 complete points, this is six means, covering 540k through 600k frames and
  768 evaluation episodes. Keep this definition even if one isolated point is poor.
- Trapezoidal return-versus-recorded-frame AUC divided by the observed frame
  span. Do not invent a frame-zero value or extrapolate outside recorded frames.
- The first-six versus last-six evaluation-mean difference as the prespecified
  within-seed learning check. A positive difference is required in **each**
  fresh seed for a successful learning confirmation. This zero-improvement
  boundary is a basic learning check, not a claim of historical performance
  equivalence or an application-level task target.
- The complete curve and the already-used late windows, 408k–492k and
  504k–600k, to expose late deterioration. These are diagnostic summaries;
  no new decline percentage will be selected after seeing the curves.

The current confirmation report checks artifact integrity and exports the full
learning curve. It directly exports the first-six and last-six means and their
difference as `early_window_mean`, `late_window_mean`, and `early_late_delta`
for return and domain measurements. It also reports final-window minus
first-logged-evaluation return and the final 20% mean. The additional
408k–492k versus 504k–600k comparisons remain explicit review calculations
from the exported curve. The report does not issue an automatic learning
verdict: a positive `checks_passed` flag does not mean that the learning review
passed or that the protocol is approved.

Report both seed values and their arithmetic mean. Two training seeds cannot
support a strong population-level uncertainty or superiority claim. Episode
intervals describe evaluation variability conditional on those policies; the
768 episodes in a final window are not 768 independent training seeds.

## Required evidence before a confirmation verdict

All integrity conditions below are mandatory. The separate behavior review
explains what the returns mean; a successful process exit is insufficient.

1. **Exact execution and complete artifacts.** Both rows must have completed
   600k frames and 100 iterations under the reviewed protocol/source/runtime
   hashes. Preserve status, metadata, resolved specification, manifest,
   checkpoints, metric streams, scheduler logs and analysis provenance. Each
   scheduled evaluation must contain exactly 128 complete episodes. Report
   failed and missing attempts. An infrastructure retry must preserve the
   original attempt and use the same seed/specification; a numerical failure
   is evidence against the candidate, not an excuse to replace a seed.
2. **Finite optimization and replay.** Require zero non-finite critical
   quantities and zero non-finite logged measurements, with required training
   metrics present. Check both optimized groups for execution integrity while
   reporting predator learning and action health separately. Behavior-policy
   log probabilities must replay before updates using the already implemented
   tolerances. Include losses, gradients, parameters, optimizer state, KL,
   clipping, ESS, advantages, value targets, entropy and explained variance.
   Empty logs cannot pass because their finite count happens to be zero.
3. **Predator action audit.** Audit the final saved actor in both DETERMINISTIC
   and RANDOM modes, explicitly selecting `adversary`, with exactly **32
   complete 100-step episodes per mode**. Use the prespecified analysis seed
   bank **100000–100031**, identical across modes and training seeds, independent
   of the training seed allocation. The existing action auditor implements
   `base_seed + episode_index`. Each mode must contain 3,200 transitions and
   19,200 predator action scalars for this three-predator, two-action layout.
   This post-training audit size follows the historical CPU diagnostic and
   leaves the training protocol's 128-episode evaluations unchanged.
   Require all action/location/scale finite fractions to equal one, positive
   distribution scales, successful strict checkpoint loading, and complete
   episode counts. Save quantiles and extrema, plus the fraction exceeding
   `|action| > 0.999`. The 0.999 marker is an established diagnostic convention;
   no arbitrary maximum allowable saturation fraction, entropy floor, or
   explained-variance floor is introduced here. Finite but suspicious trends
   require a documented behavior review before promotion.
4. **Task and information semantics.** Confirm `pcp_rng_v2`,
   `pcp_visibility_v2`, `pcp_physical_state_v1`, and `pcp_time_limit_v1`;
   independently test the 100-step truncation and bootstrap semantics. The
   default predator actor has 17 inputs, the separate privileged critic state
   has 22 inputs, and
   Identity communication remains parameterless. Preserve all input-leakage,
   partial-reset, history-independent reset, evaluation-isolation and channel
   replay checks. Both `adversary` and `agent` remain optimized; the prey's
   learned action is discarded by its script. Removing its optimization is a
   new protocol change, not part of this confirmation.
5. **Domain measurements and accounting.** Produce per-episode contact and
   visibility records for the same final audit episodes, using simulator
   state at the transition that generated the reward. Retain episode IDs,
   seeds, step counts, checkpoint hashes and summary denominators. Audit the
   recording on hand-placed states and ensure its counters neither enter
   actor inputs nor consume training RNG. Required measures are specified below.

The final policy audit must explicitly say `policy_audited=true`. The existing
audit API can return metrics-only evidence when a checkpoint is missing; that
result does not satisfy this plan. An acknowledged runtime mismatch is useful
diagnostic provenance, but does not satisfy a same-runtime confirmation audit.

## Domain definitions and interpretation

The frozen reward is inherited repeated-contact reward: each predator–prey
contact contributes +10 per transition, shared across predators. There is no
first-contact termination and no respawn at contact. The corrected time limit
truncates the episode without claiming task termination. Since the study averages
the predators' identical shared rewards, the episode accounting identity is
`predator_return = 10 × total predator–prey contact-pair transitions` for this
unshaped configuration. Summing the three identical shared reward streams would
incorrectly multiply this quantity by three.

Use the simulator's strict geometric predicate: distance smaller than the sum
of the entities' radii. In the default world a predator–prey contact threshold
is 0.075 + 0.05 = 0.125. Check the accounting identity at every transition and
episode; a mismatch or omitted terminal transition is an instrument failure.
Record:

- Any-contact probability; first-contact step with no-contact episodes
  explicitly censored at the horizon; number of steps with any contact; total
  contact-pair transitions; simultaneous-contact steps involving at least two
  predators on the same prey; and mean/maximum numbers of contacting predators.
  The underlying transition count is distinct from the episode mean or maximum.
  None of these is a count of distinct captures.
- Minimum and mean nearest-prey distance; predator/prey occupancy in the fixed
  outer-10% region where any absolute coordinate is at least `0.9 * bound`;
  obstacle and teammate collision counts using the same simulator geometry.
  State the region predicate and per-agent/per-transition denominator.
- Fractions of transitions with zero, one, two or three predators seeing the
  prey, and sender-visible/receiver-invisible opportunities over ordered,
  off-diagonal predator pairs. Record the 1.0 sensing radius and verify that
  visibility fractions sum to one. Hidden prey fields cannot affect an
  invisible receiver; globally visible prey would still leave private teammate
  velocities and implicit information in teammate motion.

Contact probability and sustained contact explain why a high reward was earned.
They are not interchangeable targets: the historical entropy-0.1 gamma cells
had similar any-contact rates despite markedly different returns. Boundary
trapping, reset overlaps, prolonged contact and simultaneous team contact must
be distinguishable in the evidence. Legitimate boundary behavior should be
reported rather than silently excluded; an environment/accounting defect is a
hard failure. No capture-rate or cooperation-success target is claimed by this
confirmation. Such a target belongs to the later necessity experiment.

The current implementation publishes these domain counters in predator info
and summarizes their collection/evaluation episodes. They are pending the final
source validation gate and review; their existence does not establish successful
full-horizon learning. The read-only review found the counter calculations,
pre-reward transition snapshot, duplicate-team-value handling and the corrected
StepCounter placement consistent with the declared candidate. Measurements are
excluded from actor/critic inputs and contain no random draws.

The `10 * contact_pairs` accounting identity is specific to the candidate's
unshaped, shared predator reward. Other accepted task configurations need a
different reference: shaping subtracts `0.1 * sum(predator nearest-prey distances)`
from the shared total, and non-shared reward divides the group mean by the number
of predators. A residual against the candidate formula must not be called an
accounting failure for those alternate configurations. This plan confirms only
the exact candidate settings; alternate reward configurations need explicitly
scoped accounting rather than extrapolation from this check.

## Runtime choice and remaining execution constraints

The original CUDA candidate pins Python 3.11.4, PyTorch 2.13.0+cu126, BenchMARL 1.5.2,
TorchRL 0.11.1, TensorDict 0.11.0, VMAS 1.5.2, and four compute threads including
all four thread environment variables. Historical
[runtime provenance](../../results/pcp_795630_audit/provenance.json) includes
both V100 and H100 devices. The current gate checks CUDA availability and the
declared software/threads; it does not pin a particular GPU model or driver.
Record actual hardware/driver details and use a consistent available device
class for the two confirmation rows where feasible. Do not claim bitwise
cross-device trajectory reproduction.

Local inspection on September 5 found macOS 26.2/arm64, Python 3.12.13, PyTorch
2.13.0 without a CUDA build, matching versions of the other four MARL packages,
four torch threads, and unset thread environment variables. CUDA is unavailable;
MPS is available but is not the validated or declared experiment backend.
Therefore that ambient local environment does not meet the CUDA launch contract.
Present cluster access has not been established; historical paths are not proof
of current access. The historical candidate rows took approximately 35–47
minutes each on contended CUDA workers; that is context, not a CPU runtime
estimate or a method-overhead benchmark.

Before observing fresh confirmation outcomes, the selected execution runtime is
the separately declared **`pcp_corrected_cpu_v1`** variant: CPU for sampling,
training and buffering; Python 3.12.13; PyTorch 2.13.0; BenchMARL 1.5.2;
TorchRL 0.11.1; TensorDict 0.11.0; VMAS 1.5.2; and **one compute thread**.
Workers must explicitly set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` to `1`, and verify the
effective torch thread count. The initial ambient four-thread inspection does
not satisfy this declaration. MPS is not selected.

Existing real CPU MAPPO, QMIX and PCP optimization checks make this a feasible
bounded execution choice. The two full runs remain conditional on the final
source validation gate, a fresh 6,000-frame instrument check retaining the
candidate's exact 128-episode evaluation setting, and a source-bound review of
the resulting evidence. No full-horizon results or successful confirmation are
claimed here, and this document creates no launch approval.

The CPU variant retains the scientific hyperparameters, 600k budget and seed
allocations above, with disjoint protocol and suite IDs. Any resulting verdict
applies to that declared CPU runtime only. It cannot confirm the CUDA candidate,
be pooled with CUDA rows, or substitute for a future CUDA runtime check and fresh
confirmation. Hardware and software provenance must accompany both executions;
neither a device override nor an analysis mismatch flag changes their scope.

The two-run confirmation is a fixed-compute learning and instrument check.
If either seed fails a mandatory condition or fails to improve from its first
to final window, report **not confirmed** and preserve the evidence. If
integrity and learning checks pass but behavior remains unexplained, report
**review pending**, with no main launch. Only a reviewed confirmation packet
can support promotion to a frozen protocol; promotion changes the protocol
hash and requires regenerated manifests/decisions. The review must explain
any material departure from historical learning rather than selecting new
thresholds, dropping a seed, extending the horizon, or tuning a communication
method after seeing these results.
