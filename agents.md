# Agent Handoff: VMAS Simple Spread Communication Study

Last updated: 2026-08-30 23:42 America/New_York

## Objective and stopping point

Complete the controlled VMAS Simple Spread communication study on the existing
BenchMARL infrastructure: rigorous bookkeeping, swappable communication
modules, validation, matched-seed experiments, focused ablations, and
analysis-ready outputs. Preserve the validated execution path and do **not**
implement PCP, PP, custom VMAS environments, a forked MAPPO, or
communication-specific PPO losses.

The intended actor path remains:

```text
local observation -> shared local encoder -> CommModule -> action/output head
```

`CommPolicyModel` is actor-only. Its critic remains BenchMARL's centralized
MLP. `IdentityComm` must remain parameterless and numerically unchanged.

## Repository state at task start

- Workspace: `/Users/akshatguduru/Desktop/Thesis/marl-comm`
- Branch: `main`
- Starting commit: `87e24207bfba16d52a5f4d8e2ec2b79d8af436cc`
- Starting worktree: clean
- Existing communication code: `CommModule`, `CommContext`, and
  `IdentityComm` only.
- Existing model configs: `benchmarl_mlp.yaml` and `comm_identity.yaml`.
- Existing experiment path: YAML/CLI -> `ExperimentSpec` -> registries ->
  `build_experiment()` -> BenchMARL.
- Existing tests cover the registries, config loading, actor construction,
  Identity behavior, and framework/Identity actor parameter matching.
- The root `README.md` was empty at task start.

## User requirements added during execution

- Keep this `agents.md` file current with everything researched,
  implemented, learned, tested, failed, and still pending so another agent can
  continue without reconstructing context.
- Stop before PCP/PP.

## Active work plan and status

1. **Complete:** audit this repository and the sibling `gnn_experiments`.
2. **Complete:** implement and unit-test Broadcast, Gated, Attention, Graph,
   shared channels, masks, replay semantics, stats, and cost accounting.
3. **Complete:** add reproducible bookkeeping, manifests, retry/status logic,
   tidy metrics, checkpoints, and analysis.
4. **Complete:** add model/sweep configs, real MAPPO/QMIX integration tests,
   README, and research-method documentation.
5. **Complete:** the post-hardening full suite passes 235 tests, and Ruff,
   compileall, `git diff --check`, and all real 6,000-frame optimization checks
   pass.
6. **Complete:** execute and analyze the 120k/600k seed-0 Identity pilots.
   The 600k run diverged rather than remaining merely unconverged, so extending
   this configuration to 1.2M or launching the main suite is not justified.
7. **Complete:** the bounded three-variant, 240k seed-0 Identity stability
   diagnostic completed without runtime/data failures. Gamma 0.9 prevented the
   collapse; entropy 0.01 was the safer of the two gamma-0.9 variants.
8. **Superseded calibration evidence:** a fresh managed 600k Identity seed-0
   confirmation at gamma 0.9, entropy 0.01, and ten minibatch iterations was
   stable, but the same provisional protocol failed to generalize to MLP seeds
   2--4. It is not the frozen study protocol.
9. **Historical/paused:** the provisional V1 main comparison exposed the
   baseline-wide multi-seed instability. Its queue was stopped at six completed,
   one failed, and 23 pending; no learned-communication row started. Never resume
   V1 or reinterpret its mixed state as the final comparison.
10. **Complete:** the nine-row, 360k MLP-only 2x2 stability diagnostic finished
    all rows and was synchronized/analyzed. Entropy 0.1 with five minibatch
    iterations is the leading candidate; full checkpoint saturation results are
    still being computed before the choice is frozen.
11. **In progress:** after the saturation audit, create a clean V2 suite and use
    its five 600k MLP rows as the full-horizon/all-seed confirmation gate.
12. **Pending experiments:** only after that gate, execute the five communication
    methods, required staged ablations, final aggregation, documentation, and
    exact readiness verdict.

## Authoritative current workspace snapshot

- `HEAD` remains the starting commit
  `87e24207bfba16d52a5f4d8e2ec2b79d8af436cc`; all work in this handoff is
  intentionally still uncommitted and the worktree contains modified and
  untracked implementation/documentation files.
- `runs/` and `results/` are Git-ignored local artifacts. They are present in
  this workspace but will not travel with a normal Git commit or clone.
- The last complete source gate was 235 passing tests plus Ruff, compileall,
  and `git diff --check`. Additional analysis hardening is currently being
  implemented and requires a new complete gate afterward.
- `simple_spread_comm_v1` is historical/provisional only. Its 0.01-entropy,
  ten-pass protocol is unstable across seeds and its 23 pending rows must not
  be launched.
- `simple_spread_mappo_multiseed_stability` is terminal: 9/9 completed, 0
  failed. Raw managed artifacts are in `runs/` and initial aggregate artifacts
  are in `results/simple_spread_mappo_multiseed_stability/`.

## Published-research verification

Primary sources checked:

- Sukhbaatar, Szlam, Fergus (NeurIPS 2016), *Learning Multiagent
  Communication with Backpropagation*: CommNet uses a continuous learned
  channel and aggregates other agents' transmissions. Our Broadcast module is
  an adaptation at one explicit injection point, not the paper's complete
  layer-interleaved controller.
- Foerster et al. (NeurIPS 2016), *Learning to Communicate with Deep
  Multi-Agent Reinforcement Learning*: DIAL motivates an end-to-end
  differentiable noisy/discretizable channel. We do not reproduce RIAL/DIAL's
  recurrent Q-learning algorithm.
- Jiang and Lu (NeurIPS 2018), *Learning Attentional Communication for
  Multi-Agent Cooperation*: motivates selective communication/integration.
- Singh, Jain, Sukhbaatar (ICLR 2019), *Learning When to Communicate at Scale
  in Multiagent Cooperative and Competitive Tasks*: IC3Net learns when to
  communicate with a gate and also uses individualized rewards. Our Gated
  module keeps the gate idea but uses the common MAPPO/shared-task setup.
- Das et al. (ICML 2019), *TarMAC: Targeted Multi-Agent Communication*:
  sender signatures/keys and values are matched against receiver queries by
  scaled dot-product soft attention; the paper also motivates multiple rounds.
  The paper uses recurrent policies and includes self-attention. Our default is
  feed-forward, one round, and excludes self because the local residual already
  carries the receiver state.
- Jiang et al. (ICLR 2020), *Graph Convolutional Reinforcement Learning*:
  DGN models agents as graph nodes with multi-head relation kernels. Our Graph
  module will use graph-restricted message passing, not DGN's DQN backbone or
  temporal relation regularization.
- Veličković et al. (ICLR 2018), *Graph Attention Networks*: motivates masked
  learned attention over graph neighborhoods.
- Kim et al. (ICLR 2019 / arXiv:1902.01554), *Learning to Schedule
  Communication in Multi-agent Reinforcement Learning*: motivates learned
  top-K sender budgets and a random-K control. We do not reproduce SchedNet's
  separate scheduler/actor/critic algorithm.

All method names therefore remain scientifically conservative:
`BroadcastComm`, `GatedComm`, `AttentionComm`, and `GraphComm`, described as
inspired by the cited methods rather than exact reproductions.

## Sibling `gnn_experiments` review

Only `../gnn_experiments` exists; the misspelled `../gnn_experiements` does
not. The sibling was read only. Its worktree was already dirty (modified
`.gitignore`, deleted run artifacts, and untracked thesis documents), so
untracked notes were treated as working material rather than canonical code.

Deeply inspected reference files:

- `comm/{base,utils,identity,broadcast,attention,graph}.py`
- `experiments/train_vmas_mappo.py`
- `backbone/{mappo,ppo_update,rollout_buffer,q_agent}.py`
- communication interface, identity, broadcast, attention, graph, rounds,
  severing, permutation, learning-smoke, class-ID, buffer, dissociation,
  shapes, environment-integration, plotting, and training-utility tests
- `experiments/{plot_run_metrics,plot_alpha_sweep}.py`
- `scripts/{pcp_eval_sweep,pcp_probe_sweep}.py`
- modern communication/MAPPO/QMIX/Simple Spread configs
- legacy `old/algos/gnn/*`, `old/algos/HetNet/*`, and radius-graph environment
  sections
- relevant `ARCHITECTURE.md`, `README.md`, thesis-review notes, Slurm scripts,
  and representative run configs/metrics/summaries

Ideas being reused:

- Receiver-row/sender-column convention: `mask[..., i, j]` means receiver
  `i` may consume sender `j`.
- Safe masked mean and masked softmax, full off-diagonal masks, and clone-before
  self-loop changes.
- Scaled multi-head attention, graph-restricted aggregation, shared rounds,
  and tests for permutation equivariance, severed edges, all-masked rows,
  mask replay, and round effects.
- k-nearest-neighbor and radius-graph construction concepts for future dynamic
  topology work, not for a custom Simple Spread environment.
- The invariant that a rollout's realized topology/failure mask must be stored
  and replayed during PPO updates.
- Recursive run discovery, config-derived labels, paired-bootstrap ideas,
  explicit failed/missing rows, and descriptive run directories.

Behavior intentionally not ported:

- Old Broadcast averages raw hidden vectors and transforms the local state;
  the new module needs a learned narrow message encoder/decoder and residual.
- Old modules only support 3-D/4-D input; the new contract is arbitrary
  `[..., N, D]` including unbatched input.
- Old Attention/Graph fix Q/K/V capacity to hidden width and use different
  fusion/normalization; the new study controls total message/key dimensions.
- Old Graph always inserts self-loops and defaults to two layers; the new
  default is one round, full graph minus self, with explicit self behavior.
- PCP class embeddings, PCP zero-message switches, custom MAPPO/QMIX loops,
  and environment-specific types are excluded.
- Legacy attention-coefficient dropout is not a communication-failure model.
- Existing plots use standard deviation/min-max, infer seeds from paths, can
  mispair missing seeds, and do not compute final-10% return or AUC; the new
  analysis uses structured metadata and bootstrap intervals.

Critical old defect: deterministic evaluation in
`experiments/train_vmas_mappo.py` calls the model without the configured
kNN/radius mask, silently evaluating sparse-topology policies under full
connectivity. The new framework must use identical topology semantics in
collection, optimization, and evaluation.

## Audited architecture decisions

- `CommPolicyModel` must partition configured context keys from encoder keys.
  At task start, a declared `[N,N]` mask would be both concatenated to the
  observation and passed as context, while `[N]` class IDs would fail generic
  feature validation. Keep BenchMARL `in_keys` intact, but compute/concatenate
  only non-context leaves and validate mask/class leaves separately.
- Context masks take precedence. In their absence, communication modules use
  full off-diagonal connectivity; Graph can supply explicitly named static
  directed-ring or seeded Erdős–Rényi controls without modifying VMAS.
- Masked/dropout senders must be removed from Attention/Graph softmax logits;
  zeroing only values incorrectly leaves probability mass on failed senders.
- Total `key_dim` and `message_dim` are across all heads, so the head ablation
  does not multiply channel capacity.
- Attention rounds share parameters. Graph's implementation must state whether
  rounds share parameters; default remains one.
- Stats are detached scalar accumulators only. Do not retain rollout tensors or
  full attention matrices in normal operation.
- For Attention/Graph nominal payload, count transmitted keys plus values;
  receiver queries and attention computation are local compute, not network
  bandwidth. Report sender emissions and edge-delivery-equivalent cost
  separately where broadcast semantics are ambiguous.
- Stochastic train-time failure is a PPO correctness hazard: independently
  resampling channel masks between behavior rollout and loss recomputation
  corrupts probability ratios. A scientific train-time dropout run must carry
  the realized mask in the TensorDict/replay path. Otherwise name dropout as a
  post-training evaluation robustness intervention. Never rely implicitly on
  `module.training` to define the experimental intervention.

## Experiment/bookkeeping audit

BenchMARL 1.5.2 creates `save_folder/generated_name` with `parents=False`, so
the outer parent must exist before experiment construction. Its CSV path is
doubly nested (`.../generated_name/generated_name/scalars/*.csv`), and every
custom communication actor is named `commpolicymodel`; method identity must
come from our metadata, never folder inference.

BenchMARL callbacks provide safe MAPPO-independent hooks after collection,
training, and evaluation. Native checkpoints omit Adam optimizer state and
restart seeding, so they are not exact scientific resume points. The initial
policy is to skip completed runs, preserve failures, and make explicit
`__retryNN` runs rather than claim exact resume.

The managed layout is:

```text
runs/<suite_id>/<run_id>/
  resolved_config.yaml
  metadata.json
  status.json
  metrics.csv
  summary.json
  git.patch (when dirty)
  benchmarl/<generated BenchMARL output>
```

Per-run status files are authoritative; a suite manifest can be rebuilt from
them safely. Analysis groups by metadata, uses the last 10% of evaluation
points for final performance, trapezoidal return-vs-frame AUC, and a fixed-RNG
seed-level bootstrap interval. Incomplete and failed runs are reported, not
silently discarded.

Verified live MAPPO parameter counts before learned modules:

- BenchMARL MLP actor: 18,948 trainable parameters.
- `CommPolicyModel + IdentityComm`: 18,948 trainable parameters.
- Centralized BenchMARL critic: 22,145 trainable parameters.
- Identity communication parameters: 0.

The live actor for both MAPPO and QMIX is discoverable by traversing
`experiment.group_policies[group].modules()` and deduplicating module/parameter
identities. The functionalized loss tree does not expose it as a normal child,
although optimizer parameter IDs overlap.

## MAPPO baseline research after the Identity pilot

The failed learning curve prompted a configuration audit before spending the
matched-seed budget. This is baseline-wide research; no communication method
was or will be tuned separately.

- The installed BenchMARL 1.5.2 MAPPO implementation constructs TorchRL's
  `ClipPPOLoss` with advantage normalization disabled. This is a property of
  the validated package path, not a reason to fork or modify MAPPO.
- BenchMARL's upstream base experiment configuration uses `gamma=0.99`,
  `lr=5e-5`, a 6,000-frame on-policy batch, 45 minibatch iterations, and a
  3M-frame default horizon. Source:
  `https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/experiment/base_experiment.yaml`.
- BenchMARL's upstream fine-tuned generic VMAS configuration uses
  `gamma=0.9`, `lr=5e-5`, a 60,000-frame batch from 600 environments, 45
  minibatch iterations, a 10M-frame horizon, and 200 evaluation episodes.
  It is a generic VMAS setting, not evidence of a Simple Spread optimum.
  Source:
  `https://github.com/facebookresearch/BenchMARL/blob/main/fine_tuned/vmas/conf/config.yaml`.
- The completed pilot used `gamma=0.99`, `lr=5e-5`, 6,000 frames per batch,
  10 environments, minibatches of 400, 10 minibatch iterations, five
  deterministic evaluation episodes, and zero entropy coefficient. Thus it
  matched several base defaults but used fewer optimizer passes and a much
  smaller experimental/evaluation budget than the upstream fine-tuned VMAS
  protocol.
- The read-only sibling's previously validated custom Simple Spread setup used
  `gamma=0.99`, lambda 0.95, an entropy schedule starting at 0.01, ten epochs,
  advantage normalization, and value normalization. It cannot be copied
  directly because it is a custom implementation, but it provides a concrete
  reason to test nonzero entropy in this repository's unchanged BenchMARL
  path.

The completed diagnostic deliberately used only three 240k-frame Identity runs,
all seed 0 and all retaining the pilot's learning rate, batch size,
environment count, minibatch size, and ten minibatch iterations:

| Variant | Gamma | Entropy coefficient | Isolated question |
|---|---:|---:|---|
| A: gamma only | 0.9 | 0.0 | Does the upstream VMAS discount prevent the late collapse? |
| B: entropy only | 0.99 | 0.01 | Does exploration regularization prevent boundary saturation? |
| C: both | 0.9 | 0.01 | Is the combination required? |

The 240k horizon crossed the observed failure onset near 192k while keeping
this a focused diagnostic rather than a broad hyperparameter search. The
existing failed configuration supplies the control trajectory. Compare
evaluation return, policy entropy, critic loss, explained variance, and
pre-clipping critic gradient norm; do not select from return alone. If none of
the three is stable, the next bounded diagnostic is the upstream 45 minibatch
iterations, not arbitrary communication-specific tuning. Only after a stable
common choice is identified may that exact choice be frozen for MLP, Identity,
all learned communication methods, the main seeds, and ablations.

The exact suite is `configs/sweeps/simple_spread_identity_stability.yaml`; its
three-row managed manifest is
`runs/simple_spread_identity_stability/manifest.csv`. All three workers ran
concurrently on 2026-08-30 local time and completed normally. Formal results
are under `results/simple_spread_identity_stability/`.

| Gamma / entropy | Final-3 eval return | Normalized AUC | Entropy first -> last | Critic loss first -> last | Critic grad first -> last | Final EV |
|---|---:|---:|---:|---:|---:|---:|
| 0.99 / 0.01 | -1545.368 | -779.744 | 1.311 -> -14.341 | 2462.98 -> 2378.29 | 155.68 -> 980.53 | 0.000003 |
| 0.9 / 0 | -632.385 | -620.158 | 1.311 -> -0.721 | 859.87 -> 101.03 | 92.72 -> 26.79 | 0.011914 |
| 0.9 / 0.01 | **-602.157** | **-610.901** | **1.311 -> -0.241** | 859.87 -> 101.36 | 92.72 -> 27.03 | 0.010670 |

All 2,655 logged values per run were finite, the failed-run table is empty,
and resolved-config differences contain only the intended gamma/entropy and
output-path changes. Gamma 0.99 still collapsed despite entropy. Both gamma
0.9 variants had bounded, nearly identical loss/gradient traces; entropy 0.01
beat entropy 0 at every evaluation from 120k onward and retained more policy
entropy. Because this evidence is one seed with five evaluation episodes and
explained variance remains weak, gamma 0.9 / entropy 0.01 is provisional until
a fresh 600k Identity run confirms stability beyond the original failure
window. Test 45 minibatch iterations only if that confirmation fails.

The confirmation passed. Suite
`runs/simple_spread_identity_stable_horizon/` completed with no failed rows and
was analyzed to `results/simple_spread_identity_stable_horizon/`. It produced
100 training iterations and 51 evaluation points in 249.70 seconds. The final
six evaluation means averaged **-502.3880** and normalized AUC was
**-586.1588**. Collection return improved from -552.31 initially to -446.08
at 600k. The last evaluation mean was -491.21; evaluations stayed within
[-885.10, -409.24] rather than entering the prior late collapse.

Optimizer diagnostics also improved through 600k: entropy went from 1.3113 to
0.1057 (minimum -0.3903), critic loss from 859.87 to 84.84, critic gradient
norm from 92.72 to 47.86 (maximum 414.06), and explained variance from 0.0036
to a run-best 0.2228. All values were finite and status was `completed`.
That seed-0 evidence ruled out a 45-minibatch diagnostic or 1.2M extension at
the time. The then-provisional common protocol was gamma 0.9, entropy coefficient 0.01, ten minibatch
iterations, 6,000 frames per collection, evaluation every 12,000 frames, and
a 600,000-frame horizon.

During the main suite, MLP seed 0 completed successfully. After excluding
timestamps, wall-clock rows, and Identity's explicit zero-communication metric
rows, its complete 600k metrics stream was byte-for-byte identical to the
fresh stable-horizon Identity seed-0 stream. This reconfirms the expected
framework/Identity numerical equivalence under that matched provisional protocol; it is not
a reason to modify Identity.

The seed-0 confirmation did not generalize cleanly across MLP seeds. Under the
same provisional gamma 0.9, entropy 0.01, ten-minibatch-iteration protocol,
seed 4 encountered a NaN action-distribution failure at 336k frames, while
seed 2 and seed 3 showed severe transformed-policy entropy contraction. Seed
3 completed but had final-window return -1571.08 and normalized AUC -1233.51;
seed 4's managed status is `failed` with the full VMAS NaN-action traceback.
The main manifest was synchronized at six completed, one failed, and 23
pending; partial analysis is under `results/simple_spread_comm_v1/`. This
is a common MAPPO baseline issue rather than evidence about any communication
module. Queue expansion was stopped before any learned-communication row
started, all already launched baseline rows closed, and the existing main MLP
seed-2/3/4 rows remain intact as controls.

Independent checkpoint/implementation audit corrected the mechanism label:
this is TanhNormal boundary saturation, not raw Normal variance collapse.
BenchMARL applies `NormalParamExtractor(biased_softplus_1.0)` and then a
bounded `TanhNormal`. On affected seeds the raw scale and location grow very
large, and tanh compresses sampled actions toward +/-1. Latest behavior-action
saturation (`|a| > .999`) was 5.32%, 7.22%, 48.25%, and 34.88% for seeds 0--3;
seed 4 was already 83.61% saturated at its 240k checkpoint, with mean/max scale
19.107/25.401 and mean/max location magnitude 5.793/11.244. Its last finite
entropy was -15.018 at 330k; at 336k collection remained finite, then actor
objective, KL, entropy, ESS, and actor grad became NaN while critic loss and
critic grad remained finite. Deterministic evaluation then emitted NaN action.

The local cause chain is documented rather than patched: installed BenchMARL
disables advantage normalization and constructs TanhNormal; TensorDict only
lower-bounds scale, while TanhNormal's default leaves location unbounded;
TorchRL estimates transformed entropy from one reparameterized Monte Carlo
sample and applies `-coefficient * entropy`; ten minibatch passes mean 150
updates per 6,000-frame collection. PyTorch grad clipping is invoked without
`error_if_nonfinite`, so a NaN norm can propagate rather than raising early.
No MAPPO/TorchRL fork was introduced. Entropy crossed -2/-5 at 228k/258k for
seed 2, 198k/228k for seed 3, and 150k/168k for seed 4 (below -10 at 204k),
which justifies the focused stress seeds and horizon below.

The bounded follow-up in
`configs/sweeps/simple_spread_mappo_multiseed_stability.yaml` completed all
nine new MLP runs. It crossed seeds 2, 3, and 4 with three 360k-frame variants:
entropy 0.1 with ten minibatch iterations, entropy 0.01 with five iterations,
and entropy 0.1 with five iterations. Together with the preserved
entropy-0.01/ten-iteration V1 controls, these form a clean 2x2 test of entropy
regularization and PPO data-reuse passes. All variants retain gamma 0.9 and a
12k evaluation interval. The new run IDs are disjoint from V1.

The synchronized manifest is 9 completed / 0 failed / 0 pending, and initial
analysis is under `results/simple_spread_mappo_multiseed_stability/`. For each
new run, final performance is the mean of the last `ceil(10%) = 4` of 31
evaluation points and AUC is trapezoidal return divided by its frame span:

| Entropy / passes | Seed | Final return | Normalized AUC | Final entropy | Final critic loss | Final actor grad | Final EV |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.10 / 10 | 2 | -510.554 | -645.793 | 1.046 | 119.103 | 1.996 | 0.0447 |
| 0.10 / 10 | 3 | -580.731 | -632.078 | 1.010 | 139.478 | 2.197 | 0.0853 |
| 0.10 / 10 | 4 | -500.655 | -606.811 | 0.976 | 101.892 | 1.589 | 0.2213 |
| 0.10 / 5 | 2 | -482.357 | -614.526 | 1.072 | 153.586 | 2.133 | -0.00015 |
| 0.10 / 5 | 3 | -563.470 | -576.683 | 1.072 | 186.095 | 2.478 | -0.00006 |
| 0.10 / 5 | 4 | -443.290 | -620.595 | 1.027 | 138.539 | 1.926 | -0.00038 |
| 0.01 / 5 | 2 | -548.447 | -674.939 | 0.464 | 169.578 | 2.062 | -0.00134 |
| 0.01 / 5 | 3 | -572.594 | -590.645 | 0.805 | 191.153 | 2.395 | -0.00028 |
| 0.01 / 5 | 4 | -481.234 | -677.708 | 0.431 | 147.766 | 1.847 | -0.00128 |

Across stress seeds, group mean final return / normalized AUC was
`-530.646 / -628.227` for 0.10/10, `-496.372 / -603.935` for 0.10/5, and
`-534.092 / -647.764` for 0.01/5. All logged optimizer values stayed finite
and all new runs crossed the former 336k failure boundary. For the preserved
0.01/10 controls on the common prefix, seeds 2 and 3 reached 360k with final
returns `-595.411` and `-1607.025`, AUCs `-783.910` and `-1027.595`, and final
entropies `-5.300` and `-7.304`; seed 4 failed at 336k after its last available
324k evaluation window reached `-2029.579` and prefix AUC `-1361.584`.

Thus both interventions help, their combination is strongest on primary
return/AUC and retains entropy near 1. The five-pass candidates have near-zero
critic explained variance at 360k, whereas 0.10/10 reaches positive EV; this
is recorded as a selection caution.

A read-only isolated checkpoint audit then rebuilt the exact nine specs into a
temporary output tree, strictly loaded each managed 360k final actor, reset the
framework/VMAS seed per mode, and ran five vectorized 100-step episodes in both
RANDOM and DETERMINISTIC exploration. Each row covered 3,000 action scalars.
All action, location, and scale finite rates were 100%; deterministic
`|action| > .999` saturation was 0% for all nine. Cross-seed RANDOM saturation
and scale means were 2.711% / 1.644 for 0.01/5, 0.811% / 1.340 for 0.10/10,
and 0.478% / 1.293 for 0.10/5. Per-seed RANDOM saturation for 0.10/5 was
0.400%, 0.300%, and 0.733%; maximum observed scales were 1.501, 1.494, and
1.555. The temporary tree was removed and original run artifacts were never
opened for writing.

| Entropy / passes | Seed | RANDOM sat. % | RANDOM mean `|loc|` | RANDOM max `|loc|` | RANDOM mean scale | RANDOM max scale |
|---|---:|---:|---:|---:|---:|---:|
| 0.01 / 5 | 2 | 3.1667 | 0.21672 | 0.71289 | 1.68920 | 2.16653 |
| 0.01 / 5 | 3 | 1.1000 | 0.21633 | 1.22657 | 1.46670 | 1.75401 |
| 0.01 / 5 | 4 | 3.8667 | 0.14708 | 0.41705 | 1.77726 | 2.07360 |
| 0.10 / 10 | 2 | 0.5000 | 0.12428 | 0.40213 | 1.29730 | 1.52540 |
| 0.10 / 10 | 3 | 0.3667 | 0.15171 | 0.59777 | 1.30554 | 1.56050 |
| 0.10 / 10 | 4 | 1.5667 | 0.16345 | 0.61615 | 1.41669 | 1.91886 |
| 0.10 / 5 | 2 | 0.4000 | 0.13033 | 0.53261 | 1.27449 | 1.50062 |
| 0.10 / 5 | 3 | 0.3000 | 0.18113 | 0.97242 | 1.26642 | 1.49398 |
| 0.10 / 5 | 4 | 0.7333 | 0.09427 | 0.38829 | 1.33696 | 1.55548 |

The jointly selected candidate is therefore gamma 0.9, entropy coefficient
0.1, five minibatch iterations, 6,000 frames per collection, deterministic
evaluation every 12,000 frames, and a 600,000-frame main horizon. This is a
baseline-wide protocol choice, not method-specific tuning.

## FROZEN STUDY PROTOCOL (V2 MLP gate passed 2026-08-31)

The required fresh 600k five-seed MLP confirmation gate was executed and
**passed**. The protocol below is now frozen and must be used unchanged for
every V2 main and ablation row. Do not re-tune it per method.

```text
gamma                             0.9
entropy_coef                      0.1
on_policy_n_minibatch_iters       5
on_policy_collected_frames_per_batch  6000
on_policy_n_envs_per_worker       10
on_policy_minibatch_size          400
lr                                5e-5
lmbda                             0.9
max_n_frames                      600000
evaluation_interval               12000
evaluation_episodes               5 (deterministic)
checkpoint_interval               120000, at end, keep 2
```

Gate rows are `simple_spread__mappo__benchmarl_mlp__v2__seed00{0..4}` in
`runs/simple_spread_comm_v2/`. They are simultaneously the main suite's MLP
reference rows, so the gate consumed no extra budget. Evidence, produced by
`scripts/audit_runs.py` and saved to
`results/simple_spread_comm_v2/mlp_gate_audit.csv`:

| Seed | Status | All finite | Logged values | Eval points | Final eval return | Entropy first/min/last | Critic loss last | EV last | Max actor grad |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|
| 0 | completed | yes | 1816 | 51 | -516.8 | 1.315 / 1.104 / 1.256 | 98.72 | 0.0360 | 6.52 |
| 1 | completed | yes | 1816 | 51 | -555.7 | 1.279 / 1.033 / 1.199 | 110.4 | 0.0027 | 6.96 |
| 2 | completed | yes | 1816 | 51 | -533.5 | 1.297 / 0.925 / 1.247 | 119.6 | 0.1002 | 7.19 |
| 3 | completed | yes | 1816 | 51 | -515.7 | 1.312 / 1.069 / 1.171 | 113.5 | 0.0384 | 7.65 |
| 4 | completed | yes | 1816 | 51 | -428.5 | 1.284 / 0.832 / 1.251 | 100.1 | 0.0220 | 7.49 |

Frozen-policy saturation audit of each final 600k actor, five vectorized
100-step episodes per exploration mode, environment reseeded per mode:

| Seed | DET saturated % | DET max abs action | RANDOM saturated % | RANDOM mean abs loc | RANDOM max abs loc | RANDOM mean scale | RANDOM max scale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.718 | 0.100 | 0.153 | 0.916 | 1.060 | 1.336 |
| 1 | 0.000 | 0.685 | 0.067 | 0.154 | 0.851 | 1.129 | 1.316 |
| 2 | 0.000 | 0.741 | 0.033 | 0.215 | 0.977 | 0.968 | 1.192 |
| 3 | 0.000 | 0.564 | 0.233 | 0.164 | 0.631 | 1.165 | 1.529 |
| 4 | 0.000 | 0.472 | 0.200 | 0.134 | 0.512 | 1.059 | 1.365 |

This is a decisive pass rather than a marginal one. Under V1's protocol seed 4
failed with a non-finite TanhNormal action at 336k and was 83.6% saturated at
its 240k checkpoint; seeds 2--3 reached entropy below -5. Under the frozen
protocol every seed crossed 600k, entropy never left [0.83, 1.32], no logged
value was non-finite, deterministic saturation is exactly zero everywhere, and
final returns are tightly clustered (spread -428.5 to -555.7). Seed 4 is now
the best seed. No 45-minibatch-iteration diagnostic or 1.2M extension is
justified.

## Execution environment for the V2 suites

All V2 rows execute with a pinned single-thread compute environment
(`OMP_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMEXPR_NUM_THREADS=1`)
on CPU, several rows at a time as isolated single-row array workers
(`scripts/sweep.py --manifest ... --run-id ... --run`). Pinning threads keeps
floating-point reduction order fixed across every row of the study rather than
letting it vary with machine load.

Two consequences must be stated rather than hidden:

- Wall-clock time recorded inside a V2 run is **contended**. Rows share eight
  cores, so in-suite training time measures scheduling as much as method cost.
  The dedicated, strictly sequential
  `configs/sweeps/simple_spread_comm_v2_wallclock.yaml` suite exists to give a
  clean per-method compute-overhead number and is labelled with its own
  `wallclock_benchmark` ablation so it can never be mixed into the comparison.
- The MLP gate rows were executed before `analysis/diagnostics.py`,
  `scripts/audit_runs.py`, the wall-clock suite YAML, and their tests were
  added, so their recorded `git.patch` differs from later V2 rows. None of
  those files is on the training path and no file under `configs/algorithms`,
  `configs/tasks`, `configs/models`, `configs/experiments`,
  `configs/sweeps/simple_spread_comm_v2*.yaml` (other than the additive
  wall-clock file), or `src/commstudy/{communication,models,experiments}` was
  touched, so every V2 row resolves a bit-identical scientific configuration.

Working rule for the rest of this phase: while any suite is executing, do not
modify training-path source or existing configs. Documentation-only edits are
provenance-visible but scientifically inert.

## Implementation log

Communication implementation:

- Added `communication/utils.py`: arbitrary-leading-dimension validation,
  receiver-row/sender-column masks, safe all-masked softmax/mean, sender-mask
  projection, activity metrics, and separate sender-emission versus directed
  edge-delivery communication costs.
- Added `communication/channel.py`: identity, whole-sender dropout, Gaussian,
  straight-through quantization, and sequential composition. Explicit sender
  masks are authoritative replay realizations. Gaussian `always`/`training`
  mode is rejected when embedded in a policy because payload noise is not
  reproduced by the persisted availability mask; evaluation-only Gaussian is
  supported.
- Extended `CommModule` with detached CPU scalar accumulation, true
  logging-window reduction for `max_message_norm`, reset/query methods, and
  nominal message bits. Unknown communication options now fail instead of
  silently falling back after a YAML typo.
- Preserved parameterless, object/value/gradient-identical `IdentityComm` and
  added explicit zero cost/activity statistics.
- Added CommNet-inspired `BroadcastComm`: bias-free learned message encoder and
  decoder, safe masked other-agent mean, residual/local path, optional channel,
  and one-round cost/activity metrics.
- Added IC3Net-inspired `GatedComm`: learned sigmoid or hard-ST gates,
  learned/random top-K, topology-aware candidates, raw learned gate metrics
  kept distinct from effective post-budget/failure metrics, and local residual.
- Added TarMAC-inspired `AttentionComm`: total-across-head Q/K/V widths,
  scaled masked attention, 1--3 shared-parameter rounds, sender budgets,
  fixed-seed random controls, finite fp16 diagnostics, and optional detached
  debug attention. Hard learned top-K uses an ST softmax surrogate so Q/K still
  receive gradients at K=1.
- Added DGN/GAT-inspired `GraphComm`: GATv2-style relation attention restricted
  by context/fallback topology, full/directed-ring/seeded-Erdos-Renyi controls,
  shared rounds, sender budgets, finite fp16 diagnostics, and graph stats.
- Attention/Graph reuse one sampled failure mask over all internal rounds, so a
  single stored per-step realization exactly reproduces the policy forward.

Actor/context integration:

- `CommPolicyModel` partitions task-agnostic context leaves from local encoder
  features and validates mask/class/sender shapes independently.
- A learned module writes its realized sender mask plus an internal generated
  marker into the rollout TensorDict. TorchRL retains arbitrary policy outputs
  into the next collector input, so stale generated masks are ignored on fresh
  no-gradient behavior/evaluation forwards and honored on gradient-enabled
  stored-action policy replay. Real MAPPO instrumentation confirmed all eight
  possible 3-agent masks occur temporally at p=0.5 and every optimizer forward
  receives its exact stored mask.
- BenchMARL 1.5.2 does not switch the actor to `.eval()` during evaluation.
  Channel evaluation mode therefore also recognizes BenchMARL's no-gradient
  deterministic interaction context. This is correct for the frozen protocol
  (`random` behavior collection, deterministic evaluation); a no-gradient
  deterministic behavior collector would be ambiguous, and nondeterministic
  evaluation must use an explicit `always` intervention or a future phase key.
- The critic remains a centralized BenchMARL `Mlp`; communication stays only
  in the decentralized actor and no MAPPO/QMIX/task/VMAS fork was introduced.

Experiment system:

- Added `experiments/bookkeeping.py`: atomic lifecycle files, descriptive run
  IDs, Git SHA/dirty status, dirty patch including safe untracked source,
  versions/devices, fully resolved project+BenchMARL configs, deduplicated
  actor/communication/critic parameter counts, failure tracebacks, final policy
  checkpoint, and summaries.
- Added `experiments/metrics.py`: stable long-form CSV covering collection,
  evaluation, loss/entropy/grad norms, returns, wall time, and drained detached
  communication statistics.
- Added `experiments/sweeps.py` plus `scripts/sweep.py`: deterministic manifest
  expansion, dry-run, local/array row execution, authoritative per-run status,
  completed-run skip, preserved failure, and explicit retry directories.
- Added `analysis/aggregate.py` plus `scripts/analyze.py`: metadata-driven
  per-run/grouped/failed CSVs, mean final ceil(10%) evaluation return,
  frame-normalized trapezoidal AUC, 95% fixed-RNG seed bootstrap intervals,
  matched-seed coverage, numeric-aware ablation order, and standardized plots.
  Communication-efficiency plots prefer realized per-step directed-delivery
  bits and retain sender-emission cost separately.
- `build_experiment()` now creates the outer save parent and accepts callbacks;
  `scripts/train.py` now runs through managed bookkeeping. These are generic
  experiment concerns, not communication algorithm logic.
- Added model configs for all four learned methods. Added sweep configs for the
  two-run pilot; 30-run main suite (MLP + five comm models, seeds 0--4); 48-run
  message-dimension; 48-run dropout; 18-run rounds; 24-run heads; 9-run graph
  topology; 36-run sender budget; and 18-run self-communication suites.
- After the seed-0 stable-horizon confirmation, all eight V1 main/ablation
  suites were updated to pin the then-provisional gamma 0.9, entropy 0.01, ten
  minibatch iterations, 600k horizon, and 12k evaluation interval explicitly.
  Those configs are now historical and must not be launched. Tests resolve and
  validate their 231 plans. The three-agent fixed ER
  graph control now uses p=0.5 seed 1, whose undirected degrees are [2,1,1],
  instead of seed 0, which isolated agent 1 and unnecessarily confounded the
  topology diagnostic with disconnection.
- Added the staged `simple_spread_mappo_multiseed_stability` suite: BenchMARL
  MLP only, seeds 2--4, 360k frames, and three baseline-wide optimizer
  variants (entropy 0.1/ten iterations, entropy 0.01/five iterations, and
  entropy 0.1/five iterations). With the preserved entropy-0.01/ten-iteration
  main controls this is a clean 2x2, and 360k crosses the observed 336k
  seed-4 failure boundary. Its nine exact resolved plans are regression tested,
  including non-collision with and preservation of the existing main controls.
  Its nine-row managed manifest was created under
  `runs/simple_spread_mappo_multiseed_stability/`; three isolated workers ran
  the rows in three batches on 2026-08-30 local time. All nine completed and
  were synchronized/analyzed as recorded above.
- Added a separate eight-file `simple_spread_comm_v2*` suite family rather
  than mutating V1. All 231 V2 plans pin the selected candidate: 600k frames,
  gamma 0.9, entropy 0.1, five minibatch iterations, deterministic evaluation
  every 12k, and 120k/end/keep-two checkpoints. V1 file SHA-256 values were
  unchanged. An optional generic `run_namespace` is now incorporated only into
  run-ID construction; V2 uses `v2`, so all V2 IDs contain `__v2__`, remain
  globally disjoint from all V1 IDs, and retain truthful `ablation=main`
  metadata. Legacy suites omit the option and keep identical IDs. Tests cover
  exact V1/V2 protocols, design parity, 231 plans per generation, and global
  uniqueness across all 462 plans. The focused sweep gate passed **50 tests**;
  Ruff and `git diff --check` passed. No V2 manifest had been created at that
  setup gate.
- Added `configs/sweeps/simple_spread_identity_stable_horizon.yaml` as a
  one-run horizon confirmation for Identity seed 0.
  It pins the selected common candidate (`gamma=0.9`, `entropy_coef=0.01`, ten
  minibatch iterations) for 600k frames while retaining the diagnostic/pilot
  12k evaluation interval and checkpoint policy. Its metadata uses
  `stable_horizon=600000`, yielding the exact run ID
  `simple_spread__mappo__identity__stable_horizon__600000__seed000`.
  Its one-row manifest was created under
  `runs/simple_spread_identity_stable_horizon/`; it completed and was analyzed
  as recorded in the calibration section above.

API input hardening applied after the 218-test gate and now focused-validated:

- `communication/utils.py` now intends to reject integer/complex hidden-state
  inputs, require boolean communication and sender masks instead of coercing
  numeric masks (including NaN-bearing masks), and require boolean masks when
  projecting senders to edges.
- `communication/channel.py` now has a shared message check for real floating
  tensors shaped at least `[..., N, M]` with nonempty sender/message axes. It
  requires boolean explicit sender masks, clones the recorded last mask to
  prevent caller mutation, requires finite nonnegative Gaussian standard
  deviation, and requires an integral quantization level count of at least two
  plus a finite positive clip.
- `IdentityComm` now calls the common input validator while returning the exact
  same tensor object for valid input, preserving the established Identity
  contract.
- Added regressions for integer/complex Identity and learned-module input,
  numeric and NaN masks, rank-1/integer channel messages, mutation of
  `last_sender_mask`, nonfinite Gaussian standard deviations, and fractional
  quantization levels. The combined focused command over Identity, Broadcast,
  communication utilities, channels, and sweep tests passed **86 tests**;
  `uv run ruff check src scripts`, compileall, and `git diff --check` also
  passed. Existing real integration tests use boolean masks. The complete
  pytest rerun remains pending until the active CPU-heavy diagnostic ends.

Documentation:

- Replaced the empty `README.md` with setup, individual commands, sweeps,
  analysis, layout, method table, reproducibility rules, and interpretation
  cautions.
- Added `docs/communication_methods.md` with primary citations, equations,
  implementation differences, exact config keys, channel/replay semantics,
  cost definitions, scheduler-overhead caveat, and Simple Spread limitations.
- Top-K Attention/Graph cost is explicitly an idealized scheduled-payload
  convention: candidate scores are evaluated before hard selection, and a
  separate physical scheduling/control exchange is not modeled or charged.

## Validation and experiment log

Initial clean-state baseline (2026-08-30):

- `uv run pytest -q` -> **58 passed**, 15 upstream deprecation warnings, in
  3.85 seconds.
- `uv run ruff check src scripts` -> **passed**.
- Installed versions: Python 3.12.13, PyTorch 2.13.0, TorchRL 0.11.1,
  TensorDict 0.11.0, BenchMARL 1.5.2, VMAS 1.5.2.

Implementation validation (2026-08-30):

- Real `build_experiment()` MAPPO tests ran one 6,000-frame optimization
  iteration for Broadcast, Gated, Attention, and Graph. Each constructed the
  real VMAS collector, produced finite rollout/training tensors, had nonzero
  actor and communication gradients, and changed communication parameters.
- Real 6,000-frame QMIX checks passed for Identity, Broadcast, Attention, and
  Graph with discrete action values, gradients, and parameter updates.
- A dedicated real MAPPO p=0.5 dropout test verifies temporal resampling across
  all 600 collector steps per environment plus mask presence in optimizer data.
- Direct replay tests verify identical outputs under changed RNG, including
  three communication rounds. Real instrumentation saw 15/15 PPO minibatch
  recomputations use the exact stored `(400,3)` masks.
- Real p=1 `mode=evaluation` instrumentation recorded requested rate 1.0 and
  realized rate 0.0 while BenchMARL left actor/channel `training=True`, proving
  the explicit evaluation lifecycle works.
- Exact MLP/Identity regression: six parameter tensors equal, direct logits
  equal bit-for-bit, and deterministic policy actions equal under seed 17.
- Final pre-pilot gate: `uv run pytest -q` -> **218 passed**, 27 upstream
  deprecation warnings, in 34.26 seconds; `uv run ruff check src scripts` ->
  **passed**; `uv run python -m compileall -q src scripts` -> **passed**.
- The independent reviewer completed a final audit of that pre-hardening tree
  and found no release-blocking Simple Spread/MAPPO protocol defect. Its
  remaining findings were limitations already recorded here (phase inference,
  idealized scheduler cost, and input-boundary hardening), not authorization to
  skip pilots or claim experimental readiness.
- Post-hardening focused gate: `uv run pytest -q src/tests/test_identity.py
  src/tests/test_broadcast.py src/tests/test_communication_utils.py
  src/tests/test_channel.py src/tests/test_sweeps.py` -> **86 passed**, 6
  upstream warnings, in 12.31 seconds; Ruff, compileall, and diff checks passed.
- Stable-horizon suite setup gate: `uv run pytest -q
  src/tests/test_sweeps.py` -> **17 passed**, 6 upstream warnings, in 11.07
  seconds; `uv run ruff check src/tests/test_sweeps.py` and `git diff --check`
  passed. The new exact-expansion regression resolves the real experiment spec
  and checks its sole run ID, suite/task/algorithm/model/seed, ablation fields,
  600k horizon, gamma, entropy coefficient, optimizer iterations, evaluation
  interval, and checkpoint settings. This setup statement is historical: the
  suite was subsequently launched, completed, and analyzed.
- Multi-seed stability setup gate: `uv run pytest -q
  src/tests/test_sweeps.py` -> **27 passed**, 6 upstream warnings, in 15.80
  seconds; `uv run ruff check src/tests/test_sweeps.py` and `git diff --check`
  passed. The new regression resolves all nine real experiment specs, checks
  every exact run ID and optimizer variant, verifies the common task/gamma/
  360k-horizon/evaluation/checkpoint protocol, and proves the preserved 600k
  main MLP seed-2/3/4 control IDs are disjoint and still resolve to entropy
  0.01 with ten minibatch iterations. With those controls, the tested arms
  form the intended 2x2. This setup statement is historical: the directory and
  manifest were subsequently created, all nine rows completed, and the suite
  was synchronized/analyzed.
- Post-hardening complete gate: `uv run pytest -q` -> **235 passed**, 27
  upstream warnings, in 34.10 seconds.
- Literal module-boundary follow-up added explicit CPU device-preservation
  tests for Identity/Broadcast/Gated/Attention/Graph and communicating-path
  input-unchanged/no-alias assertions for Broadcast and Gated. No implementation
  defect surfaced. The focused communication suite passed **104 tests** with
  six upstream warnings in 2.93 seconds; Ruff and `git diff --check` passed.

Identity training-budget pilot (completed and analyzed 2026-08-30 local time):

- Suite root and manifest: `runs/simple_spread_identity_pilot/`, including
  `manifest.csv` and `suite_config.yaml`.
- 120k run:
  `runs/simple_spread_identity_pilot/simple_spread__mappo__identity__training_budget__120000__seed000/`.
- 600k run:
  `runs/simple_spread_identity_pilot/simple_spread__mappo__identity__training_budget__600000__seed000/`.
- Each run is marked `completed` and contains `resolved_config.yaml`,
  `metadata.json`, `status.json`, `metrics.csv`, `summary.json`, the dirty-tree
  `git.patch`, the nested BenchMARL output, and
  `checkpoints/policy_state.pt`. There were no runtime failures.
- Analysis root: `results/simple_spread_identity_pilot/`. The canonical tables
  are `simple_spread_identity_pilot_per_run.csv`,
  `simple_spread_identity_pilot_summary.csv`, and
  `simple_spread_identity_pilot_failed_runs.csv`; the directory also contains
  learning, final-performance, sample-efficiency, wall-clock,
  communication-efficiency, and return-vs-budget plots. The failed-run table
  has no failed run records.

| Budget | Iterations | Eval points | Final-10% points | Mean final return | Normalized AUC | Train seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 120,000 | 20 | 11 | 2 | -663.16448 | -596.97538 | 57.3119 |
| 600,000 | 100 | 51 | 6 | -2100.83195 | -1559.53481 | 255.3281 |

The pilot rejects the original horizon decision: the 600k policy did not
merely need more frames; it became substantially worse. Collection mean return
fell from about -552.31 at 6k to -1316.42 at 600k. Evaluation began around
-479.53, crossed to -1155.68 at 192k and -1262.74 at 204k, and remained badly
degraded; the last evaluation mean was -1954.45 and the final-six mean was
-2100.83. Policy entropy fell from 1.3107 to -0.8631 by 120k, reached roughly
-14.88 around 300k, and ended at -14.3092, indicating severe continuous-action
TanhNormal boundary saturation. Over the 600k run, critic loss rose from 2462.98 to
13007.31, explained variance stayed effectively zero (0.00175 initially and
0.0000118 finally), and the logged pre-clipping critic gradient norm rose from
155.68 to 2217.35 despite configured clipping at 5. The policy objective also
rose from about 46.4 to 96.7.

All logged tensors/scalars remained finite, there were no NaNs, both status
files completed normally, and analysis consumed all expected evaluation
points. This is therefore evidence of baseline optimization collapse under the
pilot configuration, not a system crash, corrupted artifact, or a failure
specific to a learned communication module. Do not extend this configuration
to 1.2M and do not begin the main or ablation suites until the common Identity
baseline is stabilized.

Current live default parameter/cost table:

| Model | Actor params | Comm params | Critic params | Bits/sender/step |
|---|---:|---:|---:|---:|
| BenchMARL MLP | 18,948 | 0 | 22,145 | 0 |
| Identity | 18,948 | 0 | 22,145 | 0 |
| Broadcast | 27,140 | 8,192 | 22,145 | 1,024 |
| Gated | 27,269 | 8,321 | 22,145 | 1,024 |
| Attention | 35,332 | 16,384 | 22,145 | 2,048 |
| Graph | 35,364 | 16,416 | 22,145 | 2,048 |

Failures found and fixed during independent audit:

- Initial sender-mask persistence latched one initial dropout mask for all 600
  collector steps because both arbitrary policy leaves and the previous action
  survive in BenchMARL's behavior TensorDict. Replaced the action-key-only
  heuristic with verified gradient/no-gradient lifecycle logic and added a real
  temporal regression.
- Evaluation-only dropout originally never activated because BenchMARL does not
  call `.eval()`; added deterministic no-gradient evaluation recognition.
- fp16 masked entropy used an underflowing `1e-12` floor and produced NaN stats;
  diagnostics now calculate in fp32 with finite floors.
- Per-sender bits/fractions alone could not measure dropout/topology/budget
  savings; added per-step sender and directed-delivery scalar/bit counts.
- Gated gate metrics were erased by dropout/budget; raw and effective values are
  now separate. Sparse topologies can no longer waste a K slot on a sender with
  no receiver.
- Hard Attention K=1 yielded zero Q/K scheduler gradient; added a hard-forward,
  straight-through softmax surrogate and explicit gradient regression.
- Train-time Gaussian payload noise was silently resampled on PPO replay; the
  policy now rejects unsafe modes and documents evaluation-only support.
- Unknown YAML option typos were silently discarded; constructors now fail.
- `max_message_norm` averaged per-forward maxima; it now reduces as a true
  maximum across rounds and logging-window forwards.
- Numeric ablations plotted lexically (`16,32,64,8`); sorting is numeric-aware.

## Known constraints and cautions

- BenchMARL may fail if the parent of `save_folder` does not exist; all run
  paths must be created before handing them to BenchMARL.
- Preserve existing/user changes in a dirty worktree. The tree was clean at
  task start; any later dirty state must be attributed and recorded.
- Main comparisons must keep task, MAPPO, critic, actor encoder width/depth,
  seeds, frame budget, and evaluation protocol fixed.
- Learned-module capacity and communication cost must be reported separately.
- Never treat static Simple Spread graph controls as physical radio models.
- Never invent a VMAS success rate unless a stable task definition is added
  explicitly.
- Dropout replay lifecycle currently relies on the verified BenchMARL setup:
  `collect_with_grad=false` and gradient-enabled PPO policy loss. A future
  collector with gradients or no-gradient policy-loss recomputation requires
  an explicit phase signal instead of this heuristic.
- `mode=evaluation` detection assumes deterministic evaluation, as frozen in
  the configs. BenchMARL evaluation with random actions is indistinguishable
  from ordinary random collection without a phase signal.
- Gaussian train-time noise is intentionally blocked until its payload/noise
  realization is persisted, not merely its all-true sender mask.
- Attention/Graph top-K evaluates all candidate scores; scheduled delivery cost
  is an idealized accounting convention, not proof of a free real scheduler.
- The strict input/channel validation changes pass both focused and complete
  pytest invocations; no caller in the fixed Simple Spread protocol relied on
  numeric mask coercion.
- The generated sender-mask marker is safe for the current VMAS task, which
  does not supply its own sender availability mask. A future environment that
  owns that same context field needs an explicit provenance/phase design so an
  environment mask cannot be mistaken for a stale policy-generated mask.

## Direction change: execution moves to UCF ARCC Newton (2026-08-31)

Local CPU execution was abandoned mid-Stage-1 by user decision in favour of the
Newton GPU cluster. Local throughput was ~4,400 frames/second aggregate, which
put the remaining 226 rows at roughly nine hours.

What survives from local execution, as CPU validation evidence only:

- Five completed `benchmarl_mlp` 600k rows — the passed protocol gate.
- Completed `comm_identity` seed 0 and `comm_broadcast` seed 0.
- Five rows interrupted mid-flight when the session ended.

These must **not** be mixed with cluster results: device changes floating-point
numerics. `runs/` is Git-ignored, so a fresh clone on Newton starts with zero
completed rows and the whole suite executes on one device automatically. Do not
copy `runs/` to the cluster.

Honest note on GPU: a single row will not go faster on a V100/H100. The actors
are 19k--35k parameters on 10 vectorised environments, far too small to
saturate a GPU. The cluster benefit is horizontal parallelism across 231
independent rows. Making the GPU genuinely worthwhile would require raising
`on_policy_n_envs_per_worker` well above 10, which is a protocol change that
would invalidate the frozen protocol and the passed MLP gate, so it was not
done.

### Cluster facts established by research

Sources: ARCC user guide (`arcc.ist.ucf.edu/docs`) and the Newton resource
pages. Verify against `sinfo` and `module avail` on the account, since the
published docs are thin.

- Queues: `normal` (default) and `preemptable` (after the monthly allocation is
  spent). GPUs requested with `--gres=gpu:N`.
- Modules follow `<family>/<name-version>`: `python/python-3.11.4-gcc-12.2.0`,
  `cuda/cuda-12.4.0`, `anaconda/anaconda-2023.09`.
- `$HOME` is on `/lustre/fs1`, quota **1 TB and 1,000,000 files**, **not backed
  up**. One completed run is ~108 files / ~5 MB, so 231 rows is ~25k files and
  ~1.2 GB. File count, not size, is the binding constraint on future expansion.
- Default allocation 80,000 Dedicated Processor Hours per group per month.

### Infrastructure added for the cluster

The user cannot execute helper shell scripts on the cluster, so the workflow is
four numbered sbatch files and nothing else. `sbatch` is the only command.

- `slurm/01_setup.sbatch` — venv build, hard CUDA visibility check (fails the
  job rather than silently training on CPU), and writes **all** manifests so
  the training arrays never race to create them. Prints login-node fallback
  commands in case compute nodes have no outbound network.
- `slurm/02_main_comparison.sbatch` — `--array=0-29%15`, 30 rows.
- `slurm/03_ablations.sbatch` — `--array=0-200%20`, 201 rows across all seven
  ablation suites via one **combined manifest**.
- `slurm/04_analyze.sbatch` — per suite: status sync, protocol-gate audit,
  saliency, aggregate, report. Tolerates suites with no finished checkpoints.
- `slurm/newton_env.sh` — sourced (never executed, so no exec bit needed):
  modules, venv, pinned threads, device.
- `slurm/README.md` — what each experiment is, module verification, monitoring,
  preemption recovery, cost, storage, output layout.

`create_combined_manifest()` expands several suites into one manifest. Each row
already carries its own `suite_id` and `output_root`, so runs still land in
their own suite directories and per-suite analysis is untouched; the point is
that a seven-suite study becomes one contiguous array index range instead of
seven ranges tracked by hand. Duplicate run IDs across combined suites raise.

`src/tests/test_slurm_scripts.py` (15 tests) pins the invariant that the
hand-written `#SBATCH --array=0-N` ranges equal what the suite YAMLs actually
expand to. Without it, adding a seed or ablation value would silently leave the
extra rows unexecuted with no error anywhere.

### Resumability defect found and fixed

Interrupted rows exposed a real gap: `execute_plan` skipped any status that was
not `completed`/`failed`/absent, so a row abandoned by a killed worker stayed
`running` forever — never retried, never reported as failed. On a preemptable
queue that would silently produce suites full of dead rows.

Fix: `RunRecorder.heartbeat()` refreshes `status.json` once per collection
iteration and records the owning pid/host/Slurm ids; `is_stale_running()`
declares a row dead once it stops reporting; `execute_plan(...,
reclaim_stale_after=)` and `sweep.py --reclaim-stale` retry it under an explicit
`__retryNN` id while preserving the abandoned directory. A live worker is never
stolen from, because reclamation requires heartbeat silence. Covered by
`src/tests/test_resumability.py` (13 tests).

`select_plans(..., count=)` and `sweep.py --index/--count` add array chunking,
because one 600k row is far too small to justify a whole GPU. `sweep.py` also
accepts trailing `KEY=VALUE` machine-level overrides (the device above all),
which are appended to each row's recorded overrides so provenance stays
truthful.

### Communication saliency (new metric, user-requested)

Activity metrics cannot show that communication *helps*. `analysis/saliency.py`
measures the causal contribution on a frozen trained policy: evaluate normally,
evaluate again with every sender suppressed, report the paired return
difference. Suppression uses a `p=1` dropout channel; since every module decodes
through a bias-free projection, an empty neighbourhood gives an exactly zero
delta, so each learned module collapses to `h' = h`, i.e. exactly
`IdentityComm`. The intervention is therefore exact, identical across modules,
and leaves learned weights untouched.

Companion fields `saliency_action_shift_mean` and `saliency_policy_kl_mean`
separate behavioural influence from task benefit: a large action shift with a
near-zero return delta means the module communicates without coordinating.

Measured on the completed seed-0 baselines:

| Model | With comm | Severed | Saliency | Action shift |
|---|---:|---:|---:|---:|
| BenchMARL MLP (5 seeds) | -449.8 | -449.8 | **+0.00** | 0.000 |
| Identity seed 0 | -440.7 | -440.7 | **+0.00** | 0.000 |
| Broadcast seed 0 | -407.5 | -578.3 | **+170.82** | 0.166 |

Both controls are exactly zero, which is the correct value and a live wiring
check. Broadcast's channel is genuinely load-bearing (+29.5% relative). MLP and
Identity seed 0 both landing on -440.70 also re-confirms the architecture
equivalence.

Saliency is measured post-hoc from saved checkpoints, never inside a training
callback: extra in-training rollouts would consume RNG draws and perturb
trajectories, breaking comparability with rows already collected.

Results are written to `<run_dir>/saliency.json`, aggregated across seeds with
bootstrap intervals by `analysis/aggregate.py`, plotted as
`communication_saliency.png`, and tabulated in `REPORT.md`.

### Reporting

`analysis/report.py` plus `scripts/report.py` render one suite as
`results/<suite>/REPORT.md`: run inventory with an explicit warning when rows
are incomplete, provenance (commit, dirty state, device, GPU, versions, frames)
with a warning when a suite mixes commits or devices, the main comparison,
per-seed returns, matched-seed differences versus Identity, the saliency table,
numeric-ordered ablation sections, failed rows, and embedded figures.

### Source gate after these changes

`uv run pytest -q` -> **331 passed**; Ruff, compileall, `bash -n` on all five
shell scripts, and `git diff --check` pass.

## STUDY COMPLETE (2026-09-02) — results, verdict, and resume point

All eight V2 suites executed on Newton at commit `49b2ad43`. **231 planned rows,
224 completed (97%), ~43.2 GPU-hours.** Consolidated write-up with full metric
definitions and interpretation lives in
**`docs/RESULTS_v2_simple_spread.md`**; per-suite artifacts in `results/*/`
(gitignored, so keep the cluster copy).

### Verdict: READY TO IMPLEMENT PCP/PP, with one required fix

Required before launching the new environments:

1. **Comm-stack numerical instability: guards IMPLEMENTED (2026-09-02), not yet
   validated at full scale.** See "Divergence root cause and the two guards"
   below. Both default to off, so the completed suites reproduce exactly.
2. **Keep `exclude_self=true` and keep saliency as a first-class metric.**
   Enabling self-communication drops saliency to ~0 while leaving return
   unchanged -- a silently non-communicating study that no return-based metric
   detects.

### Divergence root cause and the two guards

The original diagnosis ("unnormalized residual stacking, fix with LayerNorm
and/or gradient clipping") was right about the mechanism and too loose about the
remedy. Measured on the diverging configuration:

- The failure is a **forward-pass amplification**, not a gradient-magnitude
  problem. Rounds share parameters and compose residually with no
  normalization, so activation magnitude grows *multiplicatively with depth*.
  Scaling the comm weights 100x gives activation norms of 1.3e3 / 1.3e5 / 1.4e7
  at 1 / 2 / 3 rounds; at 1000x it reaches 1.4e13, which is the magnitude the
  failed runs' `grad_norm_loss_objective` actually reported.
- **Gradient clipping alone does not fix this.** BenchMARL trains with Adam,
  whose per-parameter second-moment normalization makes updates largely
  invariant to gradient scale. Injecting 1e6 gradient spikes into the diverging
  configuration produced identical final weights with the clip off, at 100, at
  10, and at 1.0. Clipping is a useful spike limiter and diagnostic, not the
  remedy.
- **`normalize_comm_path` is the remedy.** LayerNorm on the contribution makes
  the output invariant to weight scale entirely (activation norm 36.1 at every
  weight gain from 10x to 1000x, versus 1.4e13 unnormalized).
- Why only attention and graph ever failed: broadcast and gated apply `tanh` to
  their messages, which bounds the payload. Attention and graph do not.

Both options live on `CommModule` and reach every module through
`comm_kwargs`:

```bash
# the guard that addresses the divergence
model_config.params.comm_kwargs.normalize_comm_path=true
# optional spike limiter; set it ABOVE the typical norm, not below
model_config.params.comm_kwargs.grad_clip=10.0
```

Implementation notes that matter for interpreting future runs:

- `normalize_comm_path` adds **exactly one parameter** (`comm_path_scale`, a
  LayerScale-style scalar gain initialized to 0.01). It is not cosmetic:
  normalizing to unit RMS alone destroys the modules' near-identity
  initialization and moved the first-iteration return from -552 to -1361. With
  the gain, the first iteration matches baseline (-551.5 vs -552.1).
- Clipping is applied to the message-derived contribution, *not* the module
  output, so the residual skip carrying the encoder's gradient is untouched.
- The clip unit is one environment transition, so the threshold means the same
  thing during rollout collection and during a PPO minibatch update.
- Clipping leaves the forward pass bit-identical (it is a backward hook), so it
  cannot perturb PPO replay or saliency. **Normalization does change the forward
  pass**, so runs with it enabled are not comparable to the completed V2 suites.
- New diagnostics surface through the existing stats pipeline as
  `comm_grad_clip_fraction`, `comm_grad_clip_observed_max_norm`, and
  `comm_comm_path_scale`. Calibrate `grad_clip` from the observed max norm
  rather than guessing.
- Covered by `src/tests/test_grad_clip.py` (78 tests, mutation-checked against
  four wrong implementations: clip disabled, clip on the output instead of the
  contribution, tensor-global instead of per-transition norm, and normalization
  disabled).

**Still outstanding:** none of this has been validated on a full 600k-frame run.
Re-run the 7 diverged cells (`message_dim=64`, `communication_rounds=2,3` for
attention and graph) with `normalize_comm_path=true` and confirm they complete
finite before trusting the guard at scale.

### CUDA MLP protocol gate: PASSED

The gate the previous handoff demanded is satisfied by the main-suite audit
(`results/simple_spread_comm_v2/simple_spread_comm_v2_run_audit.csv`), no
separate run needed. All 5 CUDA MLP seeds: all-finite, entropy last 1.184-1.237,
deterministic action saturation exactly 0.0000, returns -458.4 .. -501.8 (inside
the required -430..-560 band). The V2 protocol transfers from CPU to CUDA.

### Headline results

- Communication helps but only by **+6 to +12 return points on a -474 baseline
  (1.3-2.5%)**, matched-seed, all CIs excluding zero, dz 1.45-3.95.
- **Broadcast wins** on final return and AUC at *half* the bandwidth and comm
  parameters of attention/graph.
- Attention converged to **99.6% of uniform** (entropy 0.6905 vs ln2 = 0.6931),
  i.e. it rediscovered mean-pooling at 2x the cost. Head count 1-8 has no effect.
  Expected: Simple Spread is fully observable (`obs_agents: True`), so no sender
  is more informative than another and there is no selection problem to solve.
- **Learned top-1 sender selection matches full communication at 1/3 the
  bandwidth** (-466.0 vs -468.4); random top-1 at the same bandwidth is 10.5
  points worse and nearly non-salient. The soft weights are flat but the
  *ranking* is informative -- the one transferable win for attention.
- **Multi-round communication is a hard failure**: rounds=3 gives -785 to -807
  with *negative* saliency (severing comm helps by 158-325 points).
- Channel dropout p=0.25 improves AUC substantially (-556 -> -519) and yields the
  highest saliency of any setting, at 25% lower bandwidth.

### Reproducibility check: PASSED

The intended duplicate-baseline check (each ablation suite re-runs its default
point as a matched in-suite control) worked. Twelve configurations reached from
different ablation axes agree **to 13 significant figures across up to 8
independently scheduled Slurm jobs**. This also shows the "mixed GPU models"
warning in every report is a false positive here: the two models are V100-PCIE
16GB and 32GB, the same `sm_70` architecture differing only in memory.

### Known caveats carried forward

- Learning curves are **still rising at 600k frames** -- these are fixed-budget
  rankings, not asymptotic ones.
- Critic **explained variance is ~0.02** (median over 30 runs). Plausible for
  gamma=0.9 on this task, but the value signal is weak and contributes to the
  small effect sizes. Revisit gamma for PP/PCP, which have longer horizons.
- Ablations used 3 seeds; several ablation CIs span 30+ return points.
- The `switch to h100` commit did not take effect -- everything ran on V100s.
  Harmless here (see reproducibility above) but fix the submission before
  generating numbers to compare against these.

### Suggested first move in the new phase

Port `identity` and `broadcast` only, run the matched-seed protocol gate on PP,
and check that **saliency is materially larger than the ~32 points seen here**.
If communication still contributes only ~6% of return under partial
observability, the problem is the protocol, not the mechanisms -- and that is
much cheaper to learn from two modules than from five plus a full ablation grid.

## Historical resume point (superseded by the section above)

1. **Done, superseded as a multi-seed decision:** the 600k stable-horizon
   Identity seed-0 worker completed and analyzed successfully, but V1 showed
   that its optimizer protocol was not robust across seeds.
2. **Done:** the post-hardening complete pytest suite passes 235 tests.
3. **Done:** all launched V1 rows closed, its scheduler terminated, its
   manifest synchronized, and partial results generated. Preserve V1 as
   historical evidence; never resume its pending rows.
4. **Done:** the nine-row multi-seed diagnostic completed 9/9, synchronized,
   and analyzed. The leading protocol candidate is gamma 0.9, entropy 0.1,
   five minibatch iterations, 600k horizon, and 12k evaluation interval.
5. **In progress:** finish isolated checkpoint saturation evaluation, document
   the selection, and create clean V2 main/ablation configs. First execute only
   the five V2 MLP rows as the full-horizon all-seed gate. If any is nonfinite
   or again saturates pathologically, stop and diagnose before communication.
6. If the MLP gate passes, complete V2 Identity/Broadcast/Gated/Attention/Graph
   seeds 0--4 under exactly the same protocol and analyze Stage 1 before
   allocating ablation compute.
7. Execute at minimum message dimension, dropout, rounds, and graph topology;
   execute sender budget if the tested scheduler remains stable. Heads and
   self-communication are lower-priority staged diagnostics.
8. Save aggregate/raw/failed tables and all requested plots under `results/`,
   append individual and aggregate findings here, and give the required exact
   readiness verdict. PCP/PP remain untouched until then.
