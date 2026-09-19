# MAPDN implementation and bounded voltage-learning result

**Current experimental status — September 14, 2026.** The bounded main-direction MAPDN comparison is now complete: 15 policies, five seeds × Identity/LocalCapacity/Broadcast, 32,768 frames each and 480 paired-arm test trajectories. Broadcast has no demonstrated primary voltage advantage over LocalCapacity; same-checkpoint reliance is mixed and both descriptive intervals cross zero. Baseline training adequacy remains unresolved. See the [final scientific readout](../results/next_phase_20260913/PHASE_READOUT.md) and [learning/reliance figure](../results/next_phase_20260913/mapdn_final_analysis_v1/mapdn_learning_and_reliance.pdf). The original serialized-path launch failure and reviewed evaluation-only adapter are preserved; the frozen original CLI still needs that narrow fix incorporated into a future source version. All phase execution/analysis is closed. Electrical Graph wiring and the broad mechanism comparison remain subsequent work.

**Historical integration and D5 assessment — September 13, 2026.** The account below preserves the original bounded Identity-only evidence and implementation state; its next-phase-pending statements are superseded by the completed readout above.

September 13, 2026. **MAPDN case33 is integrated and validated in main.**
The exact reviewed 95-file D5 delta was applied after the completed D4 study,
strict preserved-runtime replay, final evidence archive and candidate validation.
Main now exposes Simple Spread, PCP and MAPDN through the task registry. That
historical integration has package fingerprint
`3b66c1f28cf8b11e3456f27d1e52861b1d348c4badf574b606df7f8f11e11b65`.
Original user-owned MAPDN files and the original main `.venv` remain unchanged.

The subsequent fourteen-file additive next-phase integration supplies the
paired evaluator, private-information diagnostics, allocation runner and
supervisor, associated tests, and prospective plans. Current main has package
fingerprint `90571c6c7229bdae2e663e1c7465a28d53a3eb722f01e2cc407191b8a8ff9821`.
Its [combined regression](../results/next_phase_20260913/new_phase_main_regression_v1/report.json)
passed **1,330 tests and 106 subtests**, with two optional fixture skips.
All fourteen additions match the reviewed snapshot; existing source files and
both main runtime inventories were unchanged. The preserved development and
candidate roots retain the historical `3b66c1f2…` source for D5 reconstruction.

The September 9 predefined two-seed voltage-improvement result remains
**CONFIRMED** within its original bounded scope. The
[isolated development snapshot](../../.codex-worktrees/mapdn-development/)
continues to preserve that original scientific source and evidence. This
integration does not add evidence for a communication advantage.

The integration supplies the BenchMARL task/registry binding, zone-local actor
observations, a centralized critic, chronological train/validation/test windows,
training-only normalization, aligned action/load/reward transitions, and
per-transition diagnostics including solver failures. Real managed MAPPO,
checkpoint retention, strict reconstruction and exact trajectory replay passed.

Both fixed seeds trained for 8,192 frames. Each initial/final test bank contains
16 complete 239-step episodes. The declared voltage branch requires at least
5% lower mean voltage-violation magnitude and no increase in violated-bus count.

| Test outcome | Seed 30, initial → final | Seed 31, initial → final |
|---|---:|---:|
| Mean violation magnitude, pu | 0.00344166 → 0.000229531 | 0.000383047 → 0.000235633 |
| Violated bus-transitions | 15.0509% → 2.9138% | 3.1397% → 2.6729% |
| Mean absolute inverter Q, MVAr | 0.247919 → 0.106221 | 0.261591 → 0.562727 |
| Mean total line loss, MW | 0.105711 → 0.062753 | 0.077415 → 0.177465 |
| Mean episode return | -11.6732 → -7.0541 | -10.4900 → -17.1523 |

Both seeds pass the voltage criterion. Seed 31's reactive-power use, line losses
and return worsen, including on validation data. Successful power-flow solves
on every evaluated transition do not imply voltage-limit compliance: residual
violations remain in both seeds. This is a fixed two-seed, one-feeder result;
it does not establish convergence, general safety, improved efficiency or a
communication-necessity comparison.

The September 13 candidate and main suites each passed **1,278 tests and
106 subtests**, with two optional retained-smoke-fixture skips. The candidate
also completed a fresh real 64-frame engineering smoke with all seven gates:
finite changed actor parameters, exact frames, complete validation episodes,
strict reload and exact action/reward/domain replay. Both candidate and main
reconstructed the original seed30/31 native checkpoints with exact actor/critic
and rollout-policy parity, and exactly replayed four original validation
episodes (956 transitions total per source root). No original test-bank episode
was rerun and no scientific confirmation policy was retrained. Earlier failed
checks remain preserved in their original receipts.

A fresh main-root macOS arm64 CPython 3.12.13 bootstrap verified the original
67-distribution versions, wheel tags and artifact hashes offline. Its environment
is `.bootstrap-envs/mapdn-cp312-main-20260913`; select main's `src` and
`vendor/mapdn-source` explicitly in `PYTHONPATH`, disable user site and bytecode,
and keep the four numerical thread variables at one. The
[main bootstrap proof](../results/mapdn_bootstrap_main_macos_arm64_cp312_20260913/proof.json)
is bound to main's actual source root at the historical `3b66c1f2…` stage.
The later `90571c6c…` regression used that unchanged dedicated environment.
Main's original `.venv` was neither
repointed nor synchronized. The [bootstrap guide](MAPDN_LOCAL_BOOTSTRAP.md)
retains the original reproducible capture/lock/install/verify procedure.

The [integration journal](../results/continuation_20260909/recovery_20260913/main_integration/report.json)
binds all 95 files, five replacement backups and unchanged nonselected sources.
The [complete main validation](../results/continuation_20260909/recovery_20260913/main_validation/report.json)
passes all seven stages and confirms unchanged original/new runtime inventories.
The [preserved D4 source/runtime](PCP_FROZEN_RUNTIME.md) remains independently
usable: its strict four-policy replay and post-integration source/runtime
isolation checks passed. The complete 40-policy D4 archive has verified member
readback and preserved runtime bindings.

Held-out communication interventions, baseline budget calibration and checks
that private information matters for decisions are separate next-phase work;
their implementation and engineering validation do not supply scientific
outcomes. See the [prospective plan](MAIN_DIRECTION_NEXT_PHASE_PLAN.md).
The Identity-only learning confirmation, engineering smoke and historical
replays do not establish a Broadcast benefit, a learned-mechanism ranking,
electrical GraphComm wiring, or readiness for a broad experimental sweep.

See the [detailed result and provenance](../../.codex-worktrees/mapdn-development/docs/MAPDN_CONFIRMATION_RESULTS.md),
[development guide](../../.codex-worktrees/mapdn-development/docs/MAPDN_DEVELOPMENT.md),
[original confirmation report](../../.codex-worktrees/mapdn-development/results/mapdn_case33_bounded_confirmation_v1_20260909/report.json),
and [main agent log](../agents.md). The reviewed figure is available as
[PDF](../results/mapdn_confirmation_figures_20260909_v2/mapdn_test_outcomes.pdf)
and [PNG](../results/mapdn_confirmation_figures_20260909_v2/mapdn_test_outcomes.png),
with its CSV, plotting source and input/output hashes alongside it.
