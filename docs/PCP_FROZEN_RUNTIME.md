# Preserved D4 source and runtime

The September 9 continuation preserves a separate, verified D4 source and
runtime for historical checkpoint reconstruction. Main now includes the exact
validated MAPDN integration and uses a separate MAPDN bootstrap environment;
its original restored `.venv` remains unchanged. This preserved D4 copy is
independent of those MAPDN package changes in main.

The source root is
`/Users/akki/Desktop/Thesis/.codex-worktrees/pcp-d4-frozen-69f202bf/repository`.
Its 54 Python package files reproduce D4 fingerprint
`69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e`.
All 764 entries of the original prelaunch ZIP were verified and materialized
with their original topology. The ZIP predates the September 9 relocation,
continuation and final outcomes; those are separate companion evidence.

The sibling `runtime_preservation_v1/` contains a copy of the exact environment
and complete CPython 3.12.13 base. All 33,864 environment entries and 1,904
interpreter entries were compared with their originals. Only 34 copied text
paths and three copied interpreter symlinks changed, with backups and hashes.
No original runtime files or dependencies were modified.

Use the preserved interpreter and select its matching source explicitly:

```sh
cd /Users/akki/Desktop/Thesis/.codex-worktrees/pcp-d4-frozen-69f202bf/repository
export PYTHONPATH=src
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
../runtime_preservation_v1/venv/bin/python -c 'import commstudy, torch; print(commstudy.__file__, torch.__version__)'
```

The verified imports use Torch 2.13.0, BenchMARL 1.5.2, TorchRL 0.11.1,
TensorDict 0.11.0 and VMAS 1.5.2 from the copied environment, plus commstudy
from the saved source. Its base interpreter is the copied CPython installation;
main site-packages are absent. The one-thread and tensor arithmetic checks
passed. All copied runtime entries remained unchanged after that import proof.
This is a local macOS arm64 preservation result; relocating the copy again
requires another explicit path verification.

The initial import proof constructed no environment or policy and performed
no rollout. Strict reconstruction and action replay for representative retained
D4 checkpoints subsequently passed on September 13, before MAPDN package
changes were applied to main, as recorded below.
Keep compatibility overrides disabled. Old D4 checkpoints hash the full
package and cannot be interpreted as historical replay from MAPDN's different
package fingerprint. The earlier D3 source remains separately preserved in
`pcp-retention-confirmation` with fingerprint `7124976a…`.

Evidence under `results/continuation_20260909/`:

- [Source materialization](../results/continuation_20260909/frozen_d4_source_materialization.json)
- [Complete runtime inventory and path changes](../results/continuation_20260909/frozen_d4_runtime_preservation.json)
- [Import and arithmetic proof](../results/continuation_20260909/frozen_d4_runtime_import_proof.json)
- [Post-import runtime inventory](../results/continuation_20260909/frozen_d4_runtime_post_import_inventory.json)
- [Later MAPDN integration plan](../results/continuation_20260909/mapdn_post_d4_integration_plan.md)

Original archives, metadata, approvals, relocation receipts, failed recovery
attempts and current row evidence remain intact. The [main agent log](../agents.md)
records the current experiment and remaining verification work.

## September 13 recovery: strict historical replay passed

The original September 10 supervisor failed when its tool stdout closed and
stopped its child before any replay rollout. Those failure receipts remain
unchanged. A fresh supervisor with durable output handling completed the
unchanged, pinned replay driver on September 13, 2026 in 260.74 seconds.

All four radius1 / seed20 policies (Identity, LocalCapacity, BudgetedBroadcast
K0 and K3) passed exact action and domain comparison on the original 32-episode
random and deterministic audit banks: 512 audit rollouts total, zero training
frames, and no rerun of the 200000 held-out outcome bank. Strict checkpoint
compatibility remained enabled. Original source, raw inputs, and the complete
copied runtime passed their before/after preservation checks.

The [replay report](../results/continuation_20260909/frozen_d4_replay_v1/report.json)
has SHA256 `c12b77e8fd3de306ffb33aa5d52ab99fd961da81ab12a898c0fe90c53f42f9af`.
The [recovered supervisor exit](../results/continuation_20260909/recovery_20260913/frozen_d4_replay_supervisor_exit.json)
records exit 0 and binds the completed queue, independent audit, exact replay
report and log. This closes the historical replay requirement described above;
it does not change the completed D4 scientific outcomes or establish a positive
communication effect. Final archive and MAPDN integration are separate gates.

The recovered final compound archive subsequently passed all 1,123 member
readbacks and gzip/footer/source/runtime checks. Its archive SHA256 is
`b8fec4e588e9293f1d3ca8ef4c43d61bceded57fb2a0bcaaf76f70b2513006f0`;
see the [verification receipt](../results/pcp_d4_final_compound_evidence_20260913/verification.json).
After the exact MAPDN delta and successful main bootstrap/regression/historical
D5 replay, the [post-integration D4 isolation proof](../results/continuation_20260909/recovery_20260913/main_validation/preserved_d4_isolation.json)
rechecked all 759 saved source files and 35,768 cloned runtime entries before
and after imports. The original strict replay report remained unchanged. This
last check constructed no environment or policy and performed no rollout.

Main subsequently received fourteen additive next-phase files and now has
package fingerprint `90571c6c7229bdae2e663e1c7465a28d53a3eb722f01e2cc407191b8a8ff9821`.
Its [combined regression](../results/next_phase_20260913/new_phase_main_regression_v1/report.json)
passed 1,330 tests and 106 subtests with two optional fixture skips, using the
unchanged dedicated MAPDN environment. The earlier `3b66c1f2…` bootstrap and
D5 replay receipts remain historical evidence for that source, which is retained
in the development and candidate roots. Historical D4 reconstruction and the
separately reviewed baseline calibration continue to select the `69f202bf…`
source and preserved runtime described here explicitly; main's current package
is not substituted for either preserved historical source.
