# Prospective private-information diagnostic, version 1

This is a finite, constructed-state experiment within the existing PCP and
MAPDN dynamics. It asks whether a fixed recipient can benefit from knowing
which of two declared physical states is present. It is not a trained-policy
comparison, a globally optimal controller, an oracle, a reachability proof, or
an estimate of how frequently these states occur during training. The rewards,
observations, physical constants and prey controller are unchanged.

Freeze this document, the JSON plan, and all recorded source/input hashes before
execution. Do not search over states, action grids, horizons or objectives after
seeing these results. Preserve every failed, zero and negative finding. A later
design must have a new prospective version. No old D4/D5 test episodes are used.

## PCP

Use the existing default D4 task parameters, including the 100-step environment
limit, with radius-1 and global prey visibility separately. Recipient is predator
0, donor predator 1; the third predator receives zero commanded action. Seeds
910101 and 910102 fix exogenous prey noise, independently for each state/action
replay. Rotate every position and velocity by 0 or pi/2. The base positions are
recipient (-0.35, 0), donor (0.55, 0), third predator (-0.65, 0.65), prey
(0.70, +/-0.35), and landmarks (-0.75, -0.70), (0.20, -0.75). These form the
prey-position pair, with all initial velocities zero.

The second pair holds prey at (0.65, 0.35) and donor at (0.35, 0), and changes
only the donor's initial velocity between (0, -0.8) and (0, +0.8). Other positions
and zero velocities remain as above. Native bounds, speed limits, finite state
and initial absence of collisions are checked. Nonzero velocity is an admissible
simulator state, not a sample from the zero-velocity reset distribution; dynamic
reachability of a constructed joint state is not asserted.

There are 16 pairs: 2 interventions x 2 visibility conditions x 2 rotations x
2 noise seeds. For each of the two states, evaluate exactly nine recipient
commands: zero and eight unit compass directions (angles k*pi/4). Repeat the
chosen command for 25 native transitions; all other predator commands remain
zero. Commands remain in the environment action box. The state changes are
privileged experimental setup, never inputs to a learned actor. This is a
bounded open-loop action-plan comparison. It is not a one-step action optimality
or unrestricted value-of-information theorem.

Before each rollout, directly record the recipient's and donor's native actor
observations. Exact equality is required for the recipient; donor observations
must differ. Prey-position pairs in the global condition are an observed-state
control and cannot demonstrate private information. Teammate velocities remain
private in both prey-visibility conditions. Record the prey noise schedule ID,
physical-state fingerprint, and observation vectors. Repeating the same state
across actions must reproduce exactly the initial observations and noise schedule.

Primary utility is native cumulative mean-predator reward, with any team contact
within 25 steps as a separate domain endpoint. Secondary utility is negative
mean recipient-to-prey distance over the same transitions; it is only an approach
diagnostic and cannot substitute for a reward/contact result. Report all action
tables, rather than selecting the objective with a favorable result.

## MAPDN case33

Use the existing aligned transition task, 239-step horizon, history 1, no
measurement noise, zero reset action and default voltage/reactive-power objective
(matching the earlier confirmation settings). Create a fresh content-addressed
manifest of the real case33 data with its normal chronological 60/20/20 split.
The diagnostic uses four training starts at evenly spaced indices across the
declared training starts (including both endpoints). Seeds are 920101 through
920104. It does not fit a normalizer or evaluate a learned policy; raw float64
and native unnormalized float32 observations are both recorded. Any future
normalized-actor claim requires rechecking that actor's exact preprocessing.

Recipient is inverter 0; donor is the lowest-index inverter in a different
declared zone. Multiply active and reactive loads at every bus in the donor's
zone by 0.8 or 1.2, leaving the recipient zone's injections unchanged. These are
constructed exogenous counterfactuals, not natural held-out profile samples.
Require nonnegative finite load values and a successful native power-flow solve.
Record all changed load indices/buses and donor/recipient zone/bus identities.
The interventions may change local voltage through physical coupling: exact
observation twins are a measured feasibility condition, not an assumption.

For each state, recipient actions are [-0.8, -0.4, 0, 0.4, 0.8]; all other
inverters receive zero. Evaluate the native one-step transition and retain
action-solve and advance-solve failures in the attempted-transition denominator.
Every action is replayed from the identical reset and perturbed state. Primary
utility is native reward; secondary endpoint is negative voltage-violation
magnitude, with reactive effort and line loss retained as tradeoffs. An action
solver failure invalidates its physical voltage endpoint, even though the
native failure-penalized reward remains a defined outcome.

Non-identical local observations imply **inconclusive for private information**,
even when the preferred recipient action changes. Report the local L-infinity
distance, raw and float32 exact equality, and donor distance. Do not invoke a
numerical closeness threshold to turn a coupled-state result into an exact twin.
If no donor zone exists, reset/solve fails, or an action comparison is incomplete,
retain the failure and denominator; do not replace that pair.

## Common estimand and interpretation

For complete finite utility table U[state, action], under the explicitly
constructed equally weighted two-state mixture, compute

`mean_state(max_action U) - max_action(mean_state U)`.

This nonnegative finite-grid information value is attributable to the donor's
information only if both states are valid, recipient observations are exactly
identical, and the donor observations differ. Otherwise expose it only as a
conditional state-information arithmetic quantity with the attribution gate
false. Equal utility is not a preferred-action reversal: compare complete
maximizer sets using a fixed absolute utility tolerance of 1e-10. Report a
strict preference conflict only when the sets are disjoint. Expose utility
scale and both constituent maxima; no normalized cross-environment score.

No confidence interval or population p-value is computed from these deliberately
selected geometries/rows. Noise seeds describe robustness of these examples,
not training-seed uncertainty. A positive constructed example establishes a
limited possibility in this action class; zero does not establish impossibility.
These diagnostics do not change the existing negative D4 findings and cannot
replace the forthcoming fixed-budget learned-policy experiment.
