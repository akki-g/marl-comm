# Deliverable 4: implementation and execution record

A separate [frozen source/runtime copy](PCP_FROZEN_RUNTIME.md) is preserved for
strict historical reconstruction after future MAPDN integration. Its import
proof passed; policy replay remains a separate pending check.

**Continuation — September 9, 2026:** the workspace moved to `/Users/akki/Desktop/Thesis`; historical process IDs are absent. Source/evidence hashes and the exact CPU runtime have been revalidated, with a separate path-relocation packet preserving all original scientific records. Both original saved policies have completed full held-out banks under `results/continuation_20260909/initial_evaluations/`; strict reviews, exact live/severed null checks and pre-migration action/domain audit equivalence passed. The corrected two-worker continuation started at 21:49 EDT, after both original rows passed full evaluation and archive-preservation gates. Radius1/seed20 K0 and K3 then began training, with 36 rows still pending. D4 remains scientifically incomplete. The failed first continuation and its archive-provenance mismatch are preserved separately. The historical hold descriptions below are preserved; follow the latest entry in [the main agent log](../agents.md) for current execution status.

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

September 8, 2026. The controlled experiment is implemented and the corrected
revision-2 independent launch review passed. **The 40-row comparison queue
started at 21:33:45 UTC with two pinned CPU workers. Two rows completed and were
archived, with full training-integrity, action/domain and checkpoint audits passed.
Their held-out evaluations are suspended incomplete; 38 rows remain unstarted.
Deliverable 4's scientific completion gate remains pending.**
CPU v3 is CONFIRMED and its baseline promotion is complete through a separate
frozen copy with only its status changed. The actual initial queue storage check
passed at 9.6 GiB free. Historical failed preflight and storage-blocked records
remain intact. The [launch record](../results/pcp_visibility_budget_cpu_v1/launch_record.json)
binds the reviewed manifest, operational inputs and verified preparation archive.

The [final independent launch review](../results/pcp_visibility_budget_cpu_v1/final_launch_independent_review.json)
verified the typed promotion, all 40 configurations, all eight smoke conditions,
and the corrected 763-file source snapshot. It records launch eligibility only.

The unchanged [prospective plan](PCP_VISIBILITY_BUDGET_PLAN.md) declares two
visibility conditions, four methods and five reserved training seeds. It fixes
the outcome, deadline, target and uncertainty calculation before study outcomes.

| Item | Declared scope |
|---|---|
| Visibility | Prey visible within radius 1.0, or globally visible prey |
| Methods | Identity, active LocalCapacity, Broadcast K=0, Broadcast K=3 |
| Training | Seeds 20–24, 600k frames each, 40 rows / 24M frames |
| Actor/critic inputs | 17 local actor features; identical 22-feature physical critic state |
| Predator actor capacity | Identity 19,332; LocalCapacity/K0/K3 27,524; active local or learned broadcast branch 8,192 |
| Target | First rewarded contact by step 50 with probability at least 0.80 |
| Held-out bank | 256 complete episodes, seeds 200000–200255 |
| K3 interventions | Live, severed, each single sender suppressed, cyclic cross-episode message substitution |
| Null controls | Live/severed outcomes and action health must match exactly for Identity, LocalCapacity and K0 |
| Uncertainty | Average episodes within each policy, then pair the five training seeds; 10,000 bootstrap resamples, RNG 73191 |
| Budget claim | Smallest tested K in {0,3} whose 95% lower bound reaches the target, or “not reached” |

LocalCapacity has an active local bottleneck matching the broadcast branch's
parameter count. The K0 branch is dormant and is explicitly a separate zero
budget control. Global-prey Identity is a privileged-prey-information reference,
not a full-state oracle. Scope is the tested feed-forward policy family; no
untested K1/K2, recurrent-policy, convergence or communication-impossibility
claim follows.

At K3, the float32 payload model charges 3,072 sender bits and 6,144 directed
delivery-equivalent bits per transition. Suppressing one sender charges
2,048/4,096; zero/severed charges zero. The evaluator independently counts
actual masks on every transition, checks the module's costs and archives the
live donor packet bank. These figures exclude transport overhead and are not
measurements of physical network traffic.

## Engineering evidence

The package fingerprint is
`69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e`.
An explicit reviewed source bridge links it to the confirmed CPU v3 package.
Both versions have identical first-batch Identity metrics: 4,112 non-timing
scalars, including 18 setup scalars, plus exactly matching native actor/critic
state. A separate sidecar binds that smoke to the completed v3 trajectory.

All eight method/visibility combinations completed real 6k optimization,
evaluation and replay smokes on seed 10. Strict final-checkpoint deserialization
and both-group reloads passed, as did actor/critic dimensions, parameter counts,
finite health, complete episodes and payload costs. Reserved seeds 20–24 were
not used for validation training or held-out evaluation.

The complete source suite returned **985 passed and one failed**: the failing
Slurm test assumed all CPU suites were confirmation-only. Its test-only
correction now checks the explicit CPU comparison scope while keeping it out
of the CUDA analyzer. The entire Slurm regression then passed **55 tests**.
This is a composite validation record, not a fabricated fresh full-suite total.
Additional focused controls, evaluator, analyzer, queue and archive regressions
are recorded separately with hashes under
[`results/pcp_visibility_budget_cpu_v1`](../results/pcp_visibility_budget_cpu_v1/).
Ruff, compilation and whitespace checks pass. The analyzer retains invalid or
missing rows and suppresses qualified intervals/frontiers when the evidence
packet is incomplete.

The final actual launcher preflight also caught three missing named YAML
presets: direct protocol construction had worked, but CLI reconstruction first
loads the selected model name. The presets now exactly match the frozen method
definitions. Eleven new regressions passed, covering all 40 CLI-resolved
scientific configurations and all eight live model contracts. The earlier
failed invocation and withdrawn eligibility record are preserved; the revised
eligibility packet binds the corrected configuration snapshot. No scientific
setting or package source changed, and no full comparison row ran during repair.

## Launch and evidence preservation

Use the existing pinned environment; do not synchronize or upgrade it during
this cohort. The frozen protocol requires CPU sampling/training/buffering,
Python 3.12.13, Torch 2.13.0, BenchMARL 1.5.2, TorchRL 0.11.1, TensorDict 0.11.0
and VMAS 1.5.2. From the repository root:

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

.venv/bin/python scripts/protocol.py check \
  --manifest runs/pcp_visibility_budget_cpu_v1/manifest.csv \
  --stage comparison --runtime

caffeinate -i .venv/bin/python scripts/run_pcp_budget_queue.py \
  --manifest runs/pcp_visibility_budget_cpu_v1/manifest.csv \
  --out-dir results/pcp_visibility_budget_cpu_v1/training_queue \
  --archive-dir results/pcp_visibility_budget_cpu_v1/run_archives \
  --jobs 2
```

The typed launch certificate binds the confirmed baseline review and report,
the exact added-source bridge, final scientific protocol, plan and engineering
evidence. Candidate files remain unchanged; only the separately frozen child
can authorize comparison rows. Each worker must match one of the exact 40
declared conditions. Unknown stages, factors, stale manifests and source or
protocol changes fail before training.

The queue requires a fresh allocation and at least **8 GiB free** on each output
filesystem. This is an operational storage estimate, not a scientific success
threshold. The observed v3 run was 76.6 MB raw and 34.3 MB as a verified gzip
tar; 40 raw runs plus compressed copies already exceed 4.1 GiB, before larger
communication artifacts, audit outputs and system headroom. If another drive
is used, all output paths and the final manifest must be reviewed before launch.
No user data is deleted to make room.

After each worker closes, the queue validates completion at 600k/100 iterations,
archives every run file and its closed worker log, and reads back every archived
file hash before submitting another row. All five native checkpoints, the final
managed actor and any diagnostic snapshots remain in the original run as well.
An archive failure stops new submissions and preserves the completed training
attempt. Training/replay failures also stop expansion; no automatic retry or
seed replacement is allowed. A changed manifest, script, plan or protocol stops
the queue. Run status files are authoritative while the worker manifest is fixed.

Use two workers initially on this 8 GiB machine and avoid concurrent full-suite
tests or extra audit workers. Per-run wall times are contended and cannot serve
as clean method-overhead benchmarks. The observed v3 durations imply roughly
23.4 worker-hours for 40 Identity-like rows, before communication overhead and
final audits; this is a planning estimate, not measured D4 performance.

## Final evaluation and analysis

After successful training, run the following for each of the 40 manifest IDs,
using a fresh per-run output directory:

```bash
.venv/bin/python scripts/evaluate_pcp_budget.py \
  --manifest runs/pcp_visibility_budget_cpu_v1/manifest.csv \
  --run-id "$run_id" \
  --out-dir "results/pcp_visibility_budget_cpu_v1/evaluations/$run_id"
```

The evaluator validates every retained native checkpoint and binds the final
native actor exactly to the managed actor being evaluated. It performs the
separate 32-episode deterministic/stochastic action and domain audits, checks
full training integrity for that exact visibility/method condition, then runs
the held-out intervention bank and preserves packet donors and provenance.
Do not shorten the declared bank or omit failed rows.

```bash
.venv/bin/python scripts/analyze_pcp_budget.py \
  --manifest runs/pcp_visibility_budget_cpu_v1/manifest.csv \
  --protocol configs/protocols/pcp_visibility_budget_cpu_v1_frozen.yaml \
  --evaluations-dir results/pcp_visibility_budget_cpu_v1/evaluations \
  --out-dir results/pcp_visibility_budget_cpu_v1/final_analysis
```

The complete comparison allocates 26,112,000 online evaluation transitions,
3,072,000 held-out intervention transitions and 512,000 separate action/domain
audit transitions in addition to 24M training frames. Archive the final reports,
banks, plots, source/locks, manifests and compute ledger with read-back checks.
Only a complete valid paired contrast with controls, costs and seed uncertainty
closes deliverable 4. A null effect or an unreached target is a valid result.
The two local controls provide only interim descriptive training evidence;
their held-out banks and the remaining 38 rows are incomplete. No full Broadcast
comparison row or primary contrast is available. MAPDN remains deliverable 5.
