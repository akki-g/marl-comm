# Simplification audit

The active workflow is one standalone YAML per task. It expands directly to the
five communication methods and the declared seeds. Algorithms remain in
BenchMARL; orchestration contains no research-phase approvals or study supervisors.

| Area | Before | After |
| --- | --- | --- |
| Python entrypoints | 39 scripts, 12,614 lines | 2 thin scripts, 10 lines |
| Configs | 75 YAML files, 7,470 lines | 2 standalone YAML files, 144 lines |
| Experiment package | 17 modules, 6,192 lines | 10 modules, about 2,620 lines |
| Analysis package | 12 modules, 6,143 lines | Plotting and package initializer, 154 lines |
| Core package | 17,359 Python lines, excluding vendor | About 9,500 lines, including required MAPDN simulator |
| Default comparison | Up to 90 policies across tasks/conditions | Exactly 25 per task config |
| Tracked scheduler output | 526 files | Zero; local contents preserved |

## What changed

- PCP scenario, MAPDN physics, and their environment bindings now live under
  `environments/` and `adapters/`. Task defaults and domain measurements belong
  to `tasks/`; installed code no longer searches repository YAML or imports scripts.
- The five retained communication implementations and shared actor are unchanged.
  LocalCapacity, budgeted broadcast and study-specific workflows are retired to
  the [historical revision](HISTORY.md).
- MAPDN uses the corrected implementation inside this package. Its license and
  source-origin record are unchanged. The independent Git dependency and unused
  vendored algorithms, agents, trainers and renderer are removed.
- A fresh bounded process runs each method/seed with one numerical thread. The
  coordinator owns combined CSVs, locking, failure summaries and restart decisions.
  It rejects incompatible config, source, dependency or data reuse.
- Each attempt records task-correct metrics to both CSV and `.out`; PCP returns
  measure predators rather than the zero-sum predator/prey average. Both collection
  and evaluation episode returns are retained. Final checkpoints contain actors
  and critics, without claiming optimizer continuation.
- BenchMARL's internal JSON logging remains enabled because version 1.5.2 gates
  scheduled evaluation on it. Native mixed-group progress is suppressed. A final
  evaluation also runs when the budget ends between scheduled checks.
- MAPDN preparation validates the four local files, selects chronological 60/20/20
  windows and fits normalization using 16 deterministic, nonoverlapping training
  episodes with zero actions. It records the windows, seeds, returns and hashes.
- Plotting averages observations within seed first, then gives completed seeds
  equal weight. Only shared recorded frames enter each aggregate. Missing/failed
  seeds remain visible in labels and individual trajectories.

## Verification

Final checks: **437 tests passed** (`uv run --locked pytest -q`); Ruff passed;
`uv lock --check`, `git diff --check`, and Slurm `bash -n` passed. Upstream
TorchRL deprecation/version warnings remain visible in test and worker output.

The retained regression suite checks communication masks and gradients, finite
training values, PPO behavior replay (including its saved numerical roundoff
fixture), evaluation RNG isolation, actor/critic information separation, task
returns, MAPDN chronological splits and physical failure handling.

The new workflow checks cover:

- 25-job defaults, five-job single-seed configs and invalid allocations;
- optimization, finite metrics, scheduled/final evaluation, changed actor/critic
  weights and strict checkpoint reloads for all five methods on PCP and a
  synthetic MAPDN feeder;
- exact numerical agreement between sequential and parallel execution;
- CSV/`.out` agreement and predator-only PCP returns;
- repeated invocation, preserved failed attempts, incompatible reuse, simultaneous
  launches, coordinator interruption and a killed worker while other jobs finish;
- equal seed weighting, missing-seed labels, aggregate CSVs and six figures per metric;
- discrete PCP QMIX optimization and early rejection of MAPDN QMIX.

A separate before/after check against `pre-streamline-2026-09-24` found **exact
agreement in 2,058 tensors** across three fixed-seed transitions for all ten
PCP/MAPDN × method combinations, including policy outputs. The reference used
its original source and the same installed numerical dependencies; only its
unused interactive-renderer import was stubbed.

A fresh, non-editable installation from the lockfile ran all five PCP and all
five synthetic MAPDN methods from outside the repository, without `PYTHONPATH`
or sibling repositories. The MAPDN license and source record match their original
bytes. All 526 historical scheduler outputs also match their preserved bytes.

The supplied real Case33 dataset also passed validation: 526,080 aligned
three-minute rows, 33 buses and six inverter agents. All five methods completed
512 training frames with seed 1100, the full 239-step horizon and unchanged model
and optimizer settings. Evaluation ran at frames 256 and 512 with two episodes
each. All metrics were finite, CSV and `.out` records agreed, no solver failures
occurred, actor and critic weights changed, and every checkpoint reloaded exactly.
All 16 training-only normalization episodes completed their full horizons. The
real logs produced exactly 12 figures and two correct aggregate CSVs for training
and evaluation returns. The complete 437-test regression passed again.

Both full configs pass dry-run validation and allocate 25 policies each. The
single Slurm wrapper submits them as array indices 0 (PCP) and 1 (MAPDN). Shell
syntax and both command routes were checked locally; actual scheduler execution
requires a Slurm host. No full-budget learning claim or positive communication
effect follows from these engineering checks.
