# MAPDN case33 bounded confirmation — September 9, 2026

The fixed two-seed protocol is **CONFIRMED for its predefined voltage criterion**. Both seeds completed exactly 8,192 MAPPO frames, then reduced held-out test mean voltage-violation magnitude by at least 5% without increasing the number of violating bus observations. This is evidence of bounded voltage-control learning for these two seeds, one feeder and the declared temporal banks. It does not establish a generally optimal controller, zero voltage violations, communication benefits, or population-level robustness.

Seed 31 has a material tradeoff: its reactive effort and line losses increased and its reward/return worsened in both test and validation. Upper-limit violation frequency also increased, and fewer transitions were entirely free of bus voltage violations. Its average voltage criterion still passed under the unchanged prospective plan; these descriptive regressions were not used to replace or alter that criterion.

## Fixed protocol and evidence

Seeds 30 and 31 ran sequentially on one CPU thread, with 239-step episodes and history 1. Each used 32 batches of 256 frames, minibatches of 64, four MAPPO passes and the frozen learning rate 5e-5. The full resolved specifications and optimizer settings are retained in the preparation. The shared normalizer fitted exactly 23,040 observations from 16 training episodes only. The 16 validation and 16 test row/seed banks were fixed before training. Both trainings, pretraining actors, final native checkpoints and their archives completed before the first test rollout. Final actors were selected at the fixed 8,192-frame endpoint. There was no seed replacement, retry, tuning or checkpoint selection using test results.

Case33 has 33 buses, 32 loads and six inverter agents. Each actor receives its zone-local observation features; the critic concatenates those observations. A source row spans 180 seconds. Timestamp labels run from 2012-01-01T00:03 to 2015-01-01T00:00 and are normalized to UTC without establishing a geographic time zone. Observation/reward alignment is versioned as `mapdn_aligned_transition_v1`: apply and reward the action at the observed row, then solve the next exogenous row for the next observation.

## Primary test result

Each initial/final test bank has 16 complete episodes, 3,824 transitions and 126,192 bus observations (33 buses × 3,824 transitions). Counts represent unique environment transitions, without multiplying repeated agent copies. Voltage magnitude is the mean per-bus excursion outside [0.95,1.05] pu, pooled over the bank.

| Seed | Initial magnitude (pu) | Final magnitude (pu) | Reduction | Violating bus observations, initial → final | Fraction, initial → final | Gate |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 30 | 0.00344166051 | 0.000229530906 | 93.33% | 18,993 → 3,677 / 126,192 | 15.0509% → 2.9138% | PASS |
| 31 | 0.000383047457 | 0.000235632597 | 38.48% | 3,962 → 3,373 / 126,192 | 3.1397% → 2.6729% | PASS |

Residual violations remain: the final test fractions are 2.91% and 2.67%. Successful power-flow solves are numerical feasibility; they do not mean every voltage limit was satisfied.

[Standalone test figure (PNG)](/Users/akki/Desktop/Thesis/marl-comm/results/mapdn_confirmation_figures_20260909_v2/mapdn_test_outcomes.png) · [PDF](/Users/akki/Desktop/Thesis/marl-comm/results/mapdn_confirmation_figures_20260909_v2/mapdn_test_outcomes.pdf). The figure packet contains its source, data CSV and input hashes. It shows the two fixed seeds without confidence intervals.

## Descriptive tradeoffs and validation

Reactive effort is mean absolute reactive power per inverter in MVAr. Line loss is total feeder line active-power loss in MW. Returns are undiscounted sums over 239 transitions. These metrics are descriptive under this protocol.

| Seed / bank | Reactive effort, initial → final (MVAr/inverter) | Line loss, initial → final (MW) | Episode return, initial → final |
| --- | --- | --- | --- |
| 30 / test | 0.247919 → 0.106221 | 0.105711 → 0.062753 | -11.673224 → -7.054149 |
| 30 / validation | 0.238236 → 0.107499 | 0.120937 → 0.078480 | -11.951997 → -7.573548 |
| 31 / test | 0.261591 → 0.562727 | 0.077415 → 0.177465 | -10.489993 → -17.152317 |
| 31 / validation | 0.262757 → 0.566387 | 0.090354 → 0.192999 | -10.820726 → -17.491625 |

Seed 30 improved all three means. Seed 31 increased test reactive effort by about 115% and line loss by about 129%; its average test return fell from −10.49 to −17.15. The same direction of regression occurs on validation. Both seeds improved aggregate validation voltage magnitude and frequency, but effects vary across episodes.

| Seed / bank | Episodes with lower / equal / higher mean voltage excursion |
| --- | --- |
| 30 / test | 16 / 0 / 0 |
| 30 / validation | 16 / 0 / 0 |
| 31 / test | 9 / 0 / 7 |
| 31 / validation | 9 / 1 / 6 |

All 16 episode identities were matched exactly by ID, absolute start row and environment seed for each comparison. The separate analysis JSON retains every paired episode effect; no episode was removed or replaced.

## Additional voltage descriptors

Cells show initial → final. Mean voltage and mean absolute deviation from the reference are in pu. Worst lower/upper deviations are the largest recorded bus-limit excursions in each bank.

| Descriptor | Seed 30 test | Seed 30 validation | Seed 31 test | Seed 31 validation |
| --- | --- | --- | --- | --- |
| Mean voltage (pu) | 0.98203506 → 0.98564517 | 0.98224392 → 0.98566242 | 0.98881262 → 0.99609889 | 0.98990817 → 0.99721525 |
| Mean absolute reference deviation (pu) | 0.02405 → 0.018893194 | 0.026184778 → 0.020938554 | 0.017732085 → 0.015494321 | 0.01899932 → 0.016547988 |
| Mean voltage excursion (pu) | 0.0034416605 → 0.00022953091 | 0.0042305895 → 0.00064583143 | 0.00038304746 → 0.0002356326 | 0.00092096144 → 0.00057125965 |
| Lower-limit violation fraction | 0.13669646 → 0.020175605 | 0.14732313 → 0.030421897 | 0.014937555 → 0.0082017878 | 0.016839419 → 0.007528211 |
| Upper-limit violation fraction | 0.013812286 → 0.0089625333 | 0.026451756 → 0.020698618 | 0.016459047 → 0.018527323 | 0.02869437 → 0.029756244 |
| Transitions without bus voltage violations | 0.5248431 → 0.81799163 | 0.4748954 → 0.72437238 | 0.7541841 → 0.71548117 | 0.71051255 → 0.69822176 |
| Worst lower-limit excursion (pu) | 0.080135213 → 0.018185916 | 0.088126006 → 0.020843326 | 0.018481491 → 0.014238144 | 0.021860769 → 0.016984065 |
| Worst upper-limit excursion (pu) | 0.074807725 → 0.057023687 | 0.081657208 → 0.065246578 | 0.068315711 → 0.047999385 | 0.078107369 → 0.058194294 |

## Feasibility, actions and training health

All eight full initial/final validation/test banks completed all 3,824 transitions with valid physical measurements, numerical feasibility 1, no action/advance solver failures and zero curtailed PV. Thus physical-valid and all-attempted denominators both equal 3,824 here. Missing and nonfinite domain values are zero; failed-transition accounting is tested separately in the integration suite. Power metrics are sampled MW/MVAr values, not energy integrals.

Actions are normalized dimensionless inverter commands bounded by ±0.8. The physical-bound saturation rule is `abs(action)/0.8 > 0.999`, evaluated from retained action scalars with denominator 22,944 per full bank (3,824 × six inverters). No evaluated bank saturated under this rule. The generic training statistic using absolute threshold 0.999 cannot establish absence of MAPDN saturation; per-transition training actions were not retained, and the analysis states that limitation.

Both runs retain sequential frame evidence for all 8,192 transitions and complete health series for all 32 updates. Required losses, gradients, actions, distribution parameters, advantages, targets and values were finite. Collection replay used all 256 transitions per batch with no shape fallback. The maximum accepted log-probability replay-tolerance ratios were 0.155842 and 0.134414 (below 1). Actor parameters changed and remained finite. Solver failures were zero throughout training. Strict native checkpoint reloads had no compatibility overrides or mismatches, and the first two final validation episodes replayed exactly for each seed.

## Retention and review

Execution exited 0 after 1,615.41 seconds; its scoped sleep assertion exited with the worker. Five archives verified their member bytes against retained inputs. An independent standard-library audit recomputed the gates from raw transitions and passed 36 checks, including bank completeness, row alignment, physical masks, exact replay, frozen source hashes and archive read-back. The descriptive analyzer found no consistency issues or missing health series.

- [Frozen protocol report](../results/mapdn_case33_bounded_confirmation_v1_20260909/report.json)
- [Full descriptive and paired-episode analysis](../results/mapdn_case33_bounded_confirmation_analysis_20260909.json)
- [Independent final review](../results/mapdn_case33_bounded_confirmation_v1_20260909/independent_final_review.json)
- [Frozen plan](MAPDN_CONFIRMATION_PLAN.md)
- [Engineering and task implementation](MAPDN_DEVELOPMENT.md)

Preparation content SHA256: `a2949222b9774f8c4fff5560eece7d947d01be00f61f0492e0edf704ba094327`. Protocol report file SHA256: `541e47e7b77f903f91d19e9c3f3465b94c5670c1b6c82b331415a6f1ffbb6a27`. Descriptive analysis file SHA256: `0686210ef9fc213d866b7c50f995ede3fee41be87ed40ca7d8444b944b263174`. Final evidence archive SHA256: `34968ea29e28696f150091f65f2be5fe3232a140313c373d460e7c031b90b42a`.

The isolated development source and the original user-owned MAPDN work remain preserved. Main D4 code remains frozen independently. The [separate local environment bootstrap](MAPDN_LOCAL_BOOTSTRAP.md) passed exact dependency, import and runtime-contract checks with zero rollouts or training. That engineering proof does not alter or rerun this allocation.
