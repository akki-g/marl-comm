# PCP controlled visibility and communication budget plan

Prepared September 6, 2026 for roadmap deliverable 4, before examining any new
comparison outcomes. The user requested protocol promotion and deliverable 4.
This plan becomes immutable when the first comparison worker starts. It does
not retroactively change the failed CPU v1 or unconfirmed CPU v2 findings.

## Promotion dependency and scientific scope

First complete the separately planned CPU v3 retention confirmation: unchanged
CPU v2 training implementation and scientific settings, both seeds 10 and 11,
all required checkpoints retained. Review full-horizon learning, domain/action
health, original 558k prefix/480k state invariance and complete CPU v2 trajectory
equivalence. A passing process or integrity flag alone does not promote a
protocol. Preserve the original missing-checkpoint incident permanently.

The comparison inherits gamma 0.95, entropy coefficient 0.1, ten minibatch
passes, learning rate 5e-5, lambda 0.9, 6,000-frame batches, ten environments,
400-frame minibatches and exactly 600,000 frames per policy. All five native
checkpoint frames, the final managed actor and any replay diagnostics must be
retained and hashed outside the pruning mechanism. CPU runtime remains pinned
to the confirmed package versions and one thread. The horizon is fixed compute,
with no claim of convergence or optimality. Both framework groups remain
optimized; the learned prey action is discarded by its script. MAPPO is unchanged.

Additive comparison code requires a source review and an exact first-batch
Identity metric comparison against the confirmed CPU v3 run. This bridge does
not establish that every communicating model learns; each architecture and
visibility condition also requires a real 6k optimization/evaluation/replay
smoke before the main allocation. A newly introduced failure must be preserved
and investigated. No method-specific hyperparameter or normalization tuning is
authorized by this plan.

## Controlled allocation

Train all combinations of two visibility conditions and four methods on the
same five reserved training seeds **20, 21, 22, 23, 24**: **40 policies and
24,000,000 collection frames**. Keep failed/missing policies in all tables.
Do not replace a difficult seed or add runs because an effect is small.

Visibility conditions are predator sensing radius **1.0** and **None** (globally
visible prey). Both retain the same 17-wide actor schema and the same separate
22-wide physical critic state. Physics, reward, prey script, initialization and
exogenous disturbances are unchanged. The global-prey Identity policy supplies
the privileged-prey-information reference for the restricted condition. It is
not a full-state actor: teammate private velocities remain potential information
and visible motion may provide implicit signals.

| Method | Communication | Purpose |
|---|---|---|
| Identity | none; parameterless slot | separately trained original local baseline |
| LocalCapacity | none; active local encoder/decoder branch | active parameter-capacity control for Broadcast |
| Broadcast K=0 | sender budget zero | separately trained zero-budget member of the tested communicating policy family |
| Broadcast K=3 | all three senders | communicating member with one 32-scalar message per sender per step |

LocalCapacity applies the same size bias-free 128-to-32-to-128 bottleneck to
each receiver's own hidden state. Broadcast uses the same branch parameter
count but aggregates other agents' messages, with self excluded. All other
encoder, head and critic sizes match. Every local branch parameter is active;
the K=0 Broadcast branch is dormant and explicitly **does not** substitute for
the active capacity control. Record actor, branch, critic and total parameters
separately. These are architecture controls, not an equality claim about their
optimization landscapes or initial policies.

Budgets are the two endpoints **{0, 3} senders**. There is no scheduler, preview
key exchange, learned sender ranking, quantization, recurrence, additional
round or normalization. K=0 must return its local hidden input exactly, charge
zero payload, consume no channel RNG and dominate supplied replay masks. K=3
must match the existing Broadcast operation under identical inputs and weights.
Identity stays numerically unchanged and parameterless.

## Outcomes and held-out intervention evaluation

Training evaluation retains the full confirmed schedule: 51 points from 6k to
600k, with 128 deterministic complete 100-step episodes per point. Report the
last six evaluation means (540k–600k), normalized trapezoidal AUC on 6k–600k,
all finite-health diagnostics and complete learning curves per training seed.
No checkpoint is selected by its held-out performance; audit the final 600k actor.

Before the final comparison, define the task target as **probability of first
rewarded contact by transition 50 at least 0.80**. This deadline target concerns
contact, not distinct capture or simultaneous cooperation. Retain repeated
contact return, any-contact probability by step 100, first-contact censoring,
contact duration/pairs, simultaneous contact, distances, boundary occupancy,
collisions and visibility opportunity metrics as secondary outcomes.

Evaluate final policies on **256 complete deterministic episodes**, seeds
**200000–200255**, independent of the confirmation audit bank and shared across
methods, visibility conditions, training seeds and intervention arms. Pair by
episode ID, seed, initial physical state including wander, and exogenous noise
identity. Visibility changes actor observations, so physical-state equality is
the cross-visibility matching criterion. Always verify all 100 transitions and
separate true termination from time-limit truncation.

For K=3, evaluate six arms: live communication, complete severing, suppression
of sender 0, 1 or 2 separately, and semantic cross-episode message substitution.
For the three zero-communication controls, evaluate live and severed arms and
require exact null domain/return outcomes. Severing and suppression dominate
normal availability/replay without modifying policy parameters or consuming RNG.

For semantic substitution, first record the final policy's live messages on the
whole declared bank. At transition t of episode j, substitute sender messages
from transition t of cyclic donor episode (j+1) mod 256, preserving sender
identity and receiver availability. Archive the frozen live message tensor,
donor map and hashes. This changes semantic content even for a permutation-
invariant broadcast mean. It also induces a distribution shift from live donor
trajectories into receiver closed-loop trajectories; report that limitation.
It is an intervention on the learned policy, not a retrained policy or a proof
that communication-free behavior cannot solve the task.

Audit action, location and scale finiteness, positive scales, saturation and
reward accounting on every held-out episode. Also retain a separate 32-episode
RANDOM final action audit on seeds 100000–100031. No action saturation percentage
or negative entropy alone is a newly invented automatic failure threshold.

## Payload accounting and uncertainty

The declared encoding is float32, 32 scalars per sender packet, one round.
At K=3 live: 3 emitted packets = **3,072 sender payload bits per transition**;
six directed receiver deliveries = **6,144 delivery-equivalent payload bits**.
Suppressing one sender gives 2,048 and 4,096 bits; severing/K=0 gives zero.
Sum actual observed channel masks on every transition and cross-check the
module's recorded costs. Semantic substitution retains the same payload cost.
These are payload accounting models, excluding packet headers and transport
overhead; they are not measurements of physical network traffic. Training replay
compute is not deployed communication cost.

The primary contrast is K=3 minus LocalCapacity in the restricted condition on
held-out deadline-contact probability. Also report its paired difference across
visibility conditions, the K=3 minus K=0 budget contrast, and each reference
policy. These comparisons share trained policies; do not treat them as independent.

Average held-out episodes within each trained policy first. Then report all five
seed values and paired seed-level mean differences with a deterministic 10,000-
resample percentile bootstrap (seed 73191). Report single-policy episode
uncertainty separately if used; never pool 1,280 episodes as independent training
replicates. Five seeds give limited interval resolution and power; intervals are
descriptive, with no unplanned significance or multiplicity-adjusted claims.

Within each visibility condition, identify the smallest **tested** budget in
{0,3} whose across-seed mean deadline-contact probability has a bootstrap 95%
lower bound at least 0.80. If neither qualifies, report **not reached**. If K=0
qualifies, report zero within this feed-forward family, seed allocation and
compute budget. Separately report point estimates and uncertainty; the coarse
endpoint grid cannot identify an untested K=1 or K=2 threshold. This target and
rule will not be changed after seeing the held-out results.

## Stopping and evidence

No source, protocol, factors, plan, seed list or statistical rule may change
during comparison execution. On a numerical or replay failure, stop queue
expansion, let active workers close, preserve attempts and diagnose. Do not run
an automatic retry-until-success loop. All artifacts, failure summaries, source
snapshot, environment lock, manifest, audits, plots and compute accounting must
be archived with read-back verification. A complete paired contrast plus valid
controls, accounting and uncertainty closes deliverable 4 even if the result
is null or the chosen task target is not reached. Incomplete or failed cells
limit the result and must not be silently excluded.

MAPDN, custom MAPPO losses, the historical unconfirmed CUDA protocol and optional
mechanism expansion remain outside this bounded deliverable.
