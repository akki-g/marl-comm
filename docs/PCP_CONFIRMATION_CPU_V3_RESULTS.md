# PCP CPU v3 confirmation and promotion evidence

September 8, 2026. **The complete CPU v3 confirmation gate passed.** Both
Identity policies completed 600,000 frames and 100 optimization iterations;
the required retained checkpoints, exact trajectory comparisons, replay
attribution, learning checks and final policy audits passed independent review.
The scientific claim is learning under a fixed 600k budget on two reused seeds.

This is a separately declared retention repair. CPU v1 remains failed at its
564k replay guard. CPU v2 remains **NOT CONFIRMED** because the assistant failed
to preserve its mandatory 480k checkpoint evidence. V3 neither recovers those
deleted files nor changes the earlier verdict. The immutable
[retention plan](PCP_CONFIRMATION_RETENTION_PLAN.md) required the additional
1.2M frames and every check below before promotion.

## Learning and health

| Training seed | First-six return | Last-six return | Normalized AUC | Last-six any-contact probability |
|---|---:|---:|---:|---:|
| 10 | 0.208333 | 489.674479 | 148.538905 | 0.924479 |
| 11 | 0.572917 | 550.638021 | 179.148516 | 0.920573 |

Each run has 51 deterministic evaluations of 128 complete 100-step episodes,
from 6k through 600k. The final window is 540k–600k; AUC uses the common
6k–600k interval. All 220,579 logged scalar values per run are finite, including
both optimized groups. Returns count repeated predator–prey contact, not unique
captures. Contact does not terminate these episodes; the horizon truncates them.

Late diagnostic windows continue improving. The 408k–492k and 504k–600k mean
returns are 316.729 and 459.479 for seed 10, and 356.719 and 526.050 for seed 11.
These curves do not establish convergence. Final predator entropy is -0.949
and -1.205; final critic losses are 975.900 and 1568.301, with pre-clipping
critic gradient norms 287.972 and 399.737. Those finite values are reported
alongside action health; negative differential entropy alone is not a failure.

The final 600k policies were strictly reconstructed under the saved CPU runtime.
Each policy received 32 complete episodes per deterministic/stochastic mode on
seeds 100000–100031. All 76,800 measured predator action scalars, locations and
scales were finite; scales were positive and observed saturation above
`|action| > 0.999` was zero. Every audited episode has exact reward accounting.
Deterministic mean returns are 588.4375 and 675.625; stochastic means are
571.5625 and 512.5. Boundary occupancy and repeated/simultaneous contact are
recorded in the full reports and do not establish a cooperation mechanism.

## Retention, numerical invariance and provenance

The only resolved execution changes from CPU v2 are output placement and
`keep_checkpoints_num: null` instead of 2. Retaining every native checkpoint
survived BenchMARL's periodic and end-of-run saves. Both completed runs retain
120k, 240k, 360k, 480k and 600k native checkpoints, their final managed policies,
and first predator replay snapshots at 564k. Each run's 13-file independent
archive and the suite descriptors were read back and hash-verified. The 480k
files also have an earlier independent verified copy.

For each seed, the independent comparisons verified:

- **201,665 scientific scalar values** match CPU v1 exactly through 558k,
  under the original declared exclusions.
- **220,228 non-timing scalar values** match the entire CPU v2 600k trajectory
  exactly, including every replay diagnostic. No keys are missing or extra.
- **30 tensors and eight TensorDict metadata values** match exactly at both
  comparisons: original v1 versus v3 at 480k, and v2 versus v3 at 600k.
  Tensor keys, dtype, shape and bytes are checked without coercion.
- First predator fallback is exactly frame 564000, CSV iteration 93, with no
  earlier fallback in either group. Saved whole-batch errors reproduce, and
  replay with the actual collection shape is bitwise exact at unchanged
  `rtol=atol=1e-5`. Parameters, RNG and input artifacts remain unchanged.

The original v1 failing 564k tensors were never saved. Agreement with its rounded
error summary supports attribution but does not reconstruct those lost tensors.
The preserved v3 snapshots support the exact offline replay result themselves.

Execution used the isolated retention worktree and package fingerprint
`7124976af366d28d40a3ccf0ebce62aa2b4019783d13974b9f98dd1c316af262`.
Python 3.12.13, Torch 2.13.0, BenchMARL 1.5.2, TorchRL 0.11.1, TensorDict 0.11.0
and VMAS 1.5.2 were pinned on CPU with one thread. Identity remains parameterless;
BenchMARL MAPPO and the physical-state centralized MLP critic are unchanged.
Both groups remain optimized, with the learned prey action discarded by its
script. Native checkpoints are not exact optimizer/RNG resume points.

## Review and next experiment

The [strict confirmation report](../results/pcp_candidate_confirmation_cpu_v3/confirmation_report.json),
[retention comparison](../results/pcp_candidate_confirmation_cpu_v3/retention_invariance.json),
[saved replay audit](../results/pcp_candidate_confirmation_cpu_v3/replay_snapshot_audit.json),
and [independent review](../results/pcp_candidate_confirmation_cpu_v3/independent_review.json)
provide separate evidence for integrity, invariance, mechanism and learning.
Reviews are attributed to Codex assistant agents; no human approval is claimed.

Promotion is limited to this CPU baseline and fixed compute. It does not confirm
CUDA, population-wide seed stability, communication necessity or MAPDN. The
[deliverable-4 plan](PCP_VISIBILITY_BUDGET_PLAN.md) separately controls visibility,
active local capacity and sender budgets on reserved seeds 20–24. Its added
source requires the exact Identity bridge and all eight real optimization
smokes. Its scientific completion still requires the full paired comparison,
held-out interventions, cost accounting and training-seed uncertainty.
