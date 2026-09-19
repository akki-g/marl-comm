# Deliverable 4: initial completed controls, incomplete cohort

**Continuation — September 9, 2026:** the workspace moved to `/Users/akki/Desktop/Thesis`; historical process IDs are absent. Source/evidence hashes and the exact CPU runtime have been revalidated, with a separate path-relocation packet preserving all original scientific records. Both original saved policies have completed full held-out banks under `results/continuation_20260909/initial_evaluations/`; strict reviews, exact live/severed null checks and pre-migration action/domain audit equivalence passed. Training remains 2 completed / 38 unstarted. The historical hold descriptions below are preserved; follow the latest entry in [the main agent log](../agents.md) for current execution status.

**Historical operational status, September 8, 22:16 UTC (superseded):** **2 of 40 D4 training
rows completed and were archived; 38 remain unstarted.** Both completed rows
passed the full training-integrity, final action/domain and retained-checkpoint
audits. Their held-out evaluators are now suspended and incomplete; neither
`heldout_evaluation.json` exists. The queue coordinator is also suspended.
The hold recorded 783,843,328 free bytes (about 0.73 GiB). Restore at least
8 GiB and verify frozen inputs before resuming: evaluators first, then the queue
only after audits close, with no more than two active CPU workers. D4 remains
incomplete. See the [evaluation resource hold](../results/pcp_visibility_budget_cpu_v1/evaluation_resource_hold_01.json).

September 8, 2026. This record covers the first two full D4 training rows:
Identity and LocalCapacity, both with predator sensing radius 1.0 and training
seed 20. Each completed 600,000 frames and 100 iterations. **The other 38 rows
remain unstarted under an operational storage hold. Deliverable 4 is incomplete.**
No full-horizon Broadcast K=0 or K=3 comparison row has started, and neither the
primary contrast nor a complete five-seed comparison is available.

The queue started at 21:33:45 UTC with about 9.6 GiB free. Its coordinator was
suspended at 21:43:50 UTC as capacity declined; the two active training workers
continued uninterrupted to completion. An earlier observation was 4.613 GiB free
at 22:07:57 UTC. Capacity then fell below 1 GiB, and both held-out evaluators were
suspended at 22:16:46 UTC with about 0.73 GiB free. At least 8 GiB is required
before resuming the evaluators and, after they close, the queue. This is a resource hold;
the two completed training records passed their checks. The coordinator's
queue status remains stale while suspended, so per-run status files are
authoritative.

## Descriptive training record

These are separate single-seed training summaries, without a method-ranking or
paired statistical conclusion. Each point averages 128 deterministic episodes.
The early and late windows each contain six points, `ceil(10% of 51)`.
Normalized AUC is trapezoidal return against frames from 6k through 600k,
divided by the 594k-frame span.

| Completed row | First evaluation | First-six mean, 6k–60k | Last-six mean, 540k–600k | Final evaluation | Normalized AUC |
|---|---:|---:|---:|---:|---:|
| Identity, radius 1.0, seed 20 | 0.312500 | 0.494792 | 27.916667 | 55.390625 | 12.680713 |
| LocalCapacity, radius 1.0, seed 20 | 0.234375 | 1.236979 | 635.638021 | 680.468750 | 271.175426 |

The [independent training-record review](../results/pcp_visibility_budget_cpu_v1/interim_training_review.json)
checked all 220,580 Identity and 221,786 LocalCapacity logged scalar values:
all were finite, with no duplicate metric keys. Each run retained 51 × 128
complete 100-step evaluation episodes, all ending through time-limit truncation.
Per-episode returns exactly reproduce the logged means, and domain reward
accounting error is zero. Metadata, parameters, task contracts, the 17-feature
predator actor input, 22-feature physical critic input and pinned CPU runtime
match the declared manifest. Both local controls recorded zero network payload.

Identity required no replay fallback in either group. LocalCapacity used the
existing collection-shape replay fallback in 27 predator batches, beginning at
432k; all accepted fallback errors were zero. Neither run required prey-group
fallback. The maximum accepted predator tolerance ratios were 0.313975 for
Identity and 0.995601 for LocalCapacity, within the unchanged combined absolute
and relative tolerances of 1e-5. Absolute error alone is not the acceptance rule.
The first LocalCapacity fallback diagnostic is retained for later inspection;
this lightweight review did not rerun a Torch or checkpoint replay audit.

## Preserved artifacts and ongoing evaluation

Both runs retain native checkpoints at 120k, 240k, 360k, 480k and 600k, plus their
final managed policy states. The unchanged archiver validated checkpoint headers
and exact final native/managed actor agreement for both groups. All 216 Identity
and 226 LocalCapacity raw files match their archived inventories and remained
unchanged throughout the independent review.

While the coordinator remained suspended, closed copies of the completed worker
logs were made under `interim_hold_logs/<run_id>.log`, with original-before,
original-after and copied hashes matching. Additional verified archives were
written under `interim_hold_archives/<run_id>/`; their compressed sizes are
29,232,520 bytes and 34,797,414 bytes respectively. The
[preservation completion record](../results/pcp_visibility_budget_cpu_v1/interim_preservation_completion.json)
binds the receipts and archive hashes. Original runs and logs remain intact.
These copies are separate from the future queue destinations in
`run_archives/<run_id>/`, which remain available for normal coordinator archival
before another row is submitted.

**Both full training-integrity validators passed with empty issue lists, and
final action/domain and retained-checkpoint audits are saved. The held-out
evaluations are suspended and incomplete.** Neither `heldout_evaluation.json`
exists, so no completed held-out bank or intervention outcome is claimed.
The evaluators use the final 600k actors, separate 32-episode action/domain
audits and the complete 256-episode bank with seeds 200000–200255. Partial
outputs remain in the intended `evaluations/<run_id>/` directories. No bank,
deadline, intervention or threshold has been shortened or changed. The
[interim workflow review](../results/pcp_visibility_budget_cpu_v1/interim_hold_workflow_review.json)
records the capacity checks, separate preservation paths and maximum of two
active CPU workers. After at least 8 GiB is restored and process identities and
frozen inputs are verified, resume the evaluators first. Keep the coordinator
suspended until those audits close. If an evaluator process is lost, preserve
its partial outputs and review a fresh output path for the same complete bank;
never retrain either completed policy.

## Unchanged scope and limits

The complete CPU v3 confirmation and status-only baseline promotion remain
preserved in the [v3 results](PCP_CONFIRMATION_CPU_V3_RESULTS.md). CPU v1 remains
failed and CPU v2 remains NOT CONFIRMED. The current comparison uses the
[frozen D4 protocol](../configs/protocols/pcp_visibility_budget_cpu_v1_frozen.yaml),
canonical SHA256
`ec7c109413625f270b79bffc1753adf5b7691d008eb90a4297d844b020898b0a`,
and unchanged package source
`69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e`.

The [prospective plan](PCP_VISIBILITY_BUDGET_PLAN.md) still requires four methods,
two visibility conditions, seeds 20–24 and 600k frames per row. All remaining
conditions and analysis rules remain unchanged. These initial observations do
not select hyperparameters, replace seeds or decide which remaining rows run.
Repeated-contact return does not establish distinct capture, the held-out
deadline target, convergence or communication necessity. A complete valid
contrast, held-out controls, costs and training-seed uncertainty remain required
before deliverable 4 can close.

The [verified small hold packet](../results/pcp_d4_first_pair_hold_evidence_20260908.tar.gz.verification.json)
preserves 132 files of passed audits and hold bookkeeping (2,481,771 compressed
bytes). Its dependency record binds the four separate larger archives. It does
not contain a completed held-out bank or a saved evaluator resume checkpoint.
The task-owned sleep assertion was released while all computations are held.

The [final independent hold review](../results/pcp_visibility_budget_cpu_v1/final_resource_hold_review.json)
verified the small packet, four sibling archive hashes, saved audits, unchanged
training inputs and stopped process states. It confirms preservation and the
resource hold; it does not mark the held-out evaluations or D4 complete.
