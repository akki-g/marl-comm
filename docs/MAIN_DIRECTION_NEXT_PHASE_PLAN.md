# Prospective main-direction diagnostic phase, September 13, 2026

This is new evidence motivated by completed D4, not a revision of its original
predictions. D4's negative/uncertain communication advantage and opposite
supplementary reliance ordering remain intact. D5 demonstrated bounded Identity
learning with voltage/efficiency tradeoffs, not communication benefit.

## Scope and questions

Retain the research overview's Simple Spread, PCP and MAPDN families and fixed
BenchMARL MAPPO. This phase tests whether private information can change useful
decisions within PCP/MAPDN, evaluates baseline training-budget stability, and
measures MAPDN learning effects separately from same-checkpoint reliance.
The historical Simple Spread study remains a benchmark; it is not a full-state
null control. No QMIX, extra environment, learned reward or search for a positive
communication result is authorized by this protocol. Electrical Graph, Gated
and Attention rankings belong to the later core comparison; this bounded phase
does not purport to complete that ranking.

## Preserve and freeze before collection

Keep original D4/D5 sources, runtimes, plans, banks, negative results and failed
operational attempts. New development is isolated in
`../.codex-worktrees/main-direction-next-phase-20260913`. Before any scientific
collection, retain this plan, the separate private-information diagnostic plan,
all new source/scripts/configs, dependency versions, data hashes, explicit bank
identities and resolved specifications in a hash-checked preparation packet.
Engineering test fixtures are labeled separately and never pooled with science.
Changes after collection require a new named attempt with a reason; all earlier
evidence survives. A failed integrity gate stops expansion. A zero or negative
communication effect does not fail integrity and never triggers retuning.

## Private-information experiments

The companion PRIVATE_INFORMATION_DIAGNOSTIC_PLAN.md fixes constructed states,
native-observation equality checks, action grids, horizons and seeds before
execution. Finite-action counterfactual value is conditional on those pairs and
the fixed continuation policy. It is not an optimal-controller or population
claim. If physically valid MAPDN interventions change the receiver's local
observation, they are non-twins and cannot establish useful *private* information.
Both negative findings and failed twin construction are reportable outcomes.

## MAPDN baseline calibration

Use unchanged case33 task semantics: distributed inverters, zone-local actor
observations, centralized critic, 239-step horizon, history 1, no measurement
noise or random reset actions, original voltage/reward weights. Fit the shared
normalizer from sixteen new zero-reactive-action training episodes only. Keep
the chronological 60/20/20 data split and validate full history/terminal footprints.
Fresh banks exclude the sixteen previously inspected D5 windows in each split.
Calibration validation has eight windows; comparison validation has eight other
windows; final comparison test has sixteen windows. Bank seeds start at 940000,
950000, 960000 and 970000 respectively. Fixed rows/seeds are shared across methods.
New test windows lie in the same historical test period: temporal dependence
limits external-generalization claims even when episode footprints do not overlap.

Train Identity only, seeds 40 and 41, to 32,768 frames. Record deterministic
validation at 0, 8,192, 16,384, 24,576 and 32,768 frames. Retain native actor/critic
checkpoints at every nonzero listed frame. CPU MAPPO uses LR 5e-5, gamma .99,
lambda .9, entropy 0, 256-frame batches, one environment, minibatches 64 and four
passes, gradient-norm clipping 5, Adam epsilon 1e-6. These match D5's training
settings except the prospective longer frame allocation and measurement schedule.

After *both* complete trajectories, select a common comparison budget from
16,384/24,576/32,768 using Identity validation alone. Both seeds must show two
successive changes no greater than: return max(1, .05 × absolute initial return),
mean voltage violation 0.00005 p.u., violating-bus fraction .005 and reactive
effort .02. Use the earliest qualifying budget. If none qualifies, use the
32,768-frame cap and explicitly mark training adequacy unresolved. No optimizer,
reward, seed or frame extension is chosen from communication or test outcomes.
This stability diagnostic cannot establish convergence, useful learning or
optimality by itself; show initial-to-final learning and efficiency tradeoffs too.

## MAPDN paired learning and execution comparison

Allocate seeds 50–54 to Identity, active LocalCapacity and one-round Broadcast
at the selected common budget (15 policies; maximum 491,520 training frames).
Use shared actor encoder width 128, two Tanh layers, message/bottleneck width 32,
no role embedding, no communication-path normalization or extra loss. The active
LocalCapacity residual branch matches Broadcast's 8,192 branch parameters but
exchanges zero information. Broadcast excludes self messages, with an identity
channel. Record actual actor/critic/communication counts and delivered costs.

All training and final checkpoint hashes must close before opening any final
test window. Retain training/validation evidence and reload each final actor
strictly. Evaluate the same final checkpoint with live and severed communication,
using identical start rows, seeds, horizon, normalization and exogenous inputs.
Preserve full actions/rewards/domain trajectories, reset and exogenous hashes,
RNG state and model state. Identity and LocalCapacity must give exact nulls;
Broadcast need not be nonzero to pass. Repeat live/severed evaluation on the first
two windows of the separate comparison validation bank and require exact replay
before closing each training row. Do not select final checkpoints by test performance.

Primary learning estimand is LocalCapacity minus Broadcast mean voltage-violation
magnitude (positive means Broadcast reduces violation). Primary reliance is
Broadcast severed minus live voltage magnitude. Report Identity comparisons,
violating-bus fraction, reactive effort, line losses, return, solver failures
and action saturation as secondary outcomes without selecting favorable metrics.
Use attempted transitions for solver failures, explicit bus-transition counts
for frequency, and report valid physical-observation denominators separately.
Never turn failed physics into zero voltage violation. Incomplete banks or
integrity failures block a complete-cohort inference and are retained explicitly.

Average episodes within each trained policy, pair methods/interventions by the
five training seeds, then show every seed effect and a descriptive 95% bootstrap
interval (10,000 draws, NumPy RNG 9132026). No episode-level pseudo-replication,
multiplicity-adjusted significance claim, universal saliency-percent threshold
or pooled cross-family reward. Small seed allocation and same-feeder temporal
dependence limit precision and generalization. A negative effect is a valid result.

## PCP budget calibration and decision

Use a separately frozen extension protocol for fresh Identity seeds 40/41 in
both existing prey-visibility conditions, unchanged D4 optimization/task settings,
up to 1.2M frames. Inspect baseline-only validation/checkpoint curves at declared
intervals and preserve all four runs. Do not resume D4 policies as if exact
optimizer/RNG checkpoint resumption existed. Bind the extension through the
existing protocol launch checks before launch. Final preparation will spell out
the validation bank, checkpoints and stability rule; no PCP calibration starts
until that addendum and source/runtime binding are frozen. The unchanged D4
held-out outcomes are never used to choose a winning communication configuration.

The phase report will answer each question with positive, null, negative or
inconclusive evidence and state remaining precision/budget limits. Figures show
constructed-pair information effects, baseline validation curves, and paired
MAPDN voltage/reliance effects with Q/loss tradeoffs on separate axes. Broad
mechanism ranking remains a subsequent decision rather than an automatic sweep.
At most two scientific workers run, each with all four numerical thread settings
equal to one; durable supervisor logs must survive the tool connection closing.
