# PCP and MAPDN: full communication-mechanism comparison
## Implementation specification for the coding agent

**Project:** `akki-g/marl-comm`  
**Specification date:** September 19, 2026  
**Inspected repository revision:** `9eb028ec5ad66bf4dd105623206ffd1c8da7561b`  
**Status:** Proposed implementation contract. No new training, Slurm submission, or result is represented as completed.  
**Primary deliverable:** An implemented, tested, manifest-driven experimental suite and the Slurm scripts/runbook needed to run it on the user's cluster.

> Implement the missing cross-mechanism comparison on the existing PCP and MAPDN tasks. Do not invent a new architecture, replace MAPPO, silently change the environments, or return only another plan. Return actual configuration files, executable entry points, tests, and Slurm submission scripts. Do not submit the full scientific suite during this implementation handoff unless the user separately authorizes submission.

**Reading guide.** Sections 1–6 define the science; 7–9 define evaluation and implementation; 10–12 define the CLI, Slurm, and artifacts; 13–15 define tests, work order, and the final response. `RESEARCH_REFERENCES.md` supplies primary references and source-to-requirement mappings. `[R…]` denotes inspected project evidence, `[P…]` a research source, and `[S…]` official systems documentation. Numerical allocations introduced here are proposed design choices, not recommendations quoted from those papers.

---

## 1. Mandate, scope, and evidence boundaries

### 1.1 What the user is asking for

Compare these **five requested configurations** on both task families:

1. Identity: local-only actor with a pass-through communication slot.
2. Broadcast: learned continuous messages with mean aggregation.
3. Gated: the project's learned sender-gating mechanism.
4. Attention: the project's sender/receiver attention mechanism.
5. Graph: the project's graph-restricted relation-attention mechanism.

Retain **LocalCapacity** as a sixth, separately identified, noncommunicating control. Identity is not an untrained policy; its encoder and action head learn. LocalCapacity is not Identity and is not Broadcast with its messages turned off. See Section 4.

The task families are:

- PCP with restricted prey visibility, radius 1.0;
- PCP with globally visible prey, represented by the existing `None`/YAML `null` setting;
- MAPDN case33 voltage control, using the integrated research adapter.

Thus there are **two environment families and three experimental conditions**. Do not describe them as three independent domains. Simple Spread remains a historical reference and lightweight regression task; a new full Simple Spread sweep is not part of this request.

### 1.2 What the available evidence supports

At the inspected revision, Simple Spread has the broad mechanism comparison. The completed PCP D4 study used Identity, active LocalCapacity, Broadcast K=0, and Broadcast K=3 across two visibility conditions. MAPDN's completed bounded comparison used Identity, LocalCapacity, and Broadcast. Its broader mechanism ranking and electrical Graph wiring remain subsequent work. These are repository-reported results, not fresh reanalysis of raw data by the author of this specification. [R0–R5]

The attached strategy report advised postponing a broad sweep in favor of a more diagnostic direction. **The user's subsequent instruction now requests this bounded coverage expansion.** Preserve the report as historical advice; do not rewrite it as though it recommended the present allocation. Its methodological cautions—capacity, information contracts, training adequacy, intervention interpretation, and preservation—still apply. [R0, pp. 3–7, 15–19]

An existing YAML, supported class, software test, or smoke run does not establish a completed scientific comparison. The agent's audit must distinguish **implemented**, **smoke-tested**, **trained**, **evaluated**, and **analyzed** for every cell.

### 1.3 What is out of scope

No QMIX/IPPO/MADDPG comparison, new communication architecture, new reward, new prey behavior, recurrent policy, role embedding, learned scheduler, delay queue, adversarial disturbance, variable team size, or new feeder is required. Do not discretize MAPDN actions to force another algorithm into the suite. No broad Cartesian product of heads × widths × rounds × dropout × topology.

This is a **controlled study of project communication modules inside a common MAPPO actor**, not a faithful reproduction tournament among the complete CommNet, IC3Net, TarMAC, and DGN algorithms. The differences must remain visible in the report. [R6; P1–P7]

---

## 2. Scientific questions and permitted conclusions

### 2.1 Main question

**Does the measured value of communication on PCP and MAPDN depend on the communication mechanism, when task, training procedure, and evaluation conditions are held fixed?**

### 2.2 Prespecified supporting questions

**PCP information interaction:** Is a mechanism's advantage over Identity different when prey visibility is restricted versus global?

**Learning versus reliance:** Does a method that uses messages after training also outperform a separately trained local-only policy?

**Resource interpretation:** What active network capacity and message payload accompany each outcome?

Write competing possibilities into the experiment document: all communicators may behave similarly; a selective method may improve on Broadcast; communication may hurt; or training variability/budget limitations may prevent a useful ordering. The suite must remain scientifically valid when the intended positive effect does not appear.

### 2.3 Do not overclaim

A successful comparison supports fixed-task, fixed-budget, tested-policy-family statements. It does not establish communication necessity, information-theoretic impossibility, convergence, general grid safety, universal algorithm rankings, or topology transfer. A frozen-policy message-removal loss measures reliance of that policy and also creates an input-distribution shift. It is not the performance of a separately retrained no-message policy. [R0, §7.4; P9, P11]

A method beating Identity is initially a **combined architecture/information effect**. The current LocalCapacity control strengthens the Broadcast comparison but does not perfectly isolate information for every larger mechanism. Section 4.4 defines the optional additional capacity sensitivity.

---

## 3. Suite allocation: fresh cohorts by default

### 3.1 Required core

| Suite ID | Conditions | Configurations per condition | Independent training seeds | Training policies |
|---|---:|---:|---:|---:|
| `pcp_mechanisms_v1` | Restricted and global prey visibility | 6 | 5 | 60 |
| `mapdn_mechanisms_v1` | case33 | 6 | 5 | 30 |
| **Core total** | **3** | **6** | **5** | **90** |

Configuration order must be explicit and stable: `identity`, `local_capacity`, `broadcast`, `gated`, `attention`, `graph_full`.

Proposed main training seeds: **1100, 1101, 1102, 1103, 1104**. Development/qualification seeds: **1000, 1001**. Short engineering smoke seed: **900**. These are new allocations; inspect the local evidence index for collisions before freezing. If a collision is found, replace the entire proposed stage namespace before collection and record the reason, rather than selectively replacing difficult seeds.

Five seeds is the planned resource allocation, not proof of sufficient power. The code must support an explicit larger seed list. Any increase must apply to the entire prespecified comparison before final evaluation, not only a promising method. Store the requested seed list in the frozen protocol, not only in the Slurm array range. [P10]

### 3.2 Fresh means fresh

The default core trains all 90 policies from initialization under the new Linux/cluster runtime. Do not quietly merge old macOS CPU policies with new CUDA policies or rerun only a historical losing method on new hardware. Existing results remain a separate historical table.

Historical reuse is an optional later amendment, not the default implementation. It requires actual artifacts, compatible observation/dynamics versions, identical scientific settings and runtime/backend conditions, verified checkpoint reconstruction, and a documented new evaluation plan. A tracked narrative or broken link into ignored `runs/` is not enough. Same seed integers do not establish identical training trajectories. [R0, p. 4; R4]

### 3.3 Separately selectable supplements

Implement optional manifest generation and scheduling hooks, without silently adding these rows to the core:

| Supplement | Additional training | Purpose |
|---|---:|---|
| `capacity` | LocalCapacity with bottleneck 64: 3 conditions × 5 seeds = **15** | Active local-capacity sensitivity near Attention/Graph branch sizes. |
| `mapdn_topology` | Two sparse Graph variants × 5 seeds = **10** | Feeder-informed versus matched-random communication masks; full Graph reuses core. |

Core plus both supplements is **115 trained policies**, excluding qualification/smokes. `full` means the full **core** unless `--include capacity,mapdn_topology` is explicit. If optional topology cannot be validated with real feeder mappings, mark that supplement blocked; do not substitute a random or full mask while retaining the feeder label. Its unavailability must not prevent the complete five-entry core comparison.

Do not automatically add Broadcast K=0 to every main cell. Retain it in engineering tests and historical discussion; it is a useful dormant-branch null, not a fifth communicator.

---

## 4. Exact model contract

### 4.1 Shared actor and critic structure

Use the existing actor-side insertion point:

```text
local observation -> shared local encoder -> selected module -> action head
```

Keep the existing task-specific encoder/head, action distribution, parameter-sharing setting, critic, MAPPO losses, and observation allowlist. Initial proposed common choices: hidden width 128, two encoder layers, Tanh, no role embeddings, no added communication normalization or auxiliary loss, self communication excluded, one communication round where applicable.

All core communicators use instantaneous, clean delivery with no sender budget, no stochastic dropout, no compression/noise, and no message cache. An `identity` **channel transformation** means delivery without alteration; it does not mean an Identity **actor module**.

Module input/output is `[..., N, D]`. Masks are oriented `[..., receiver, sender]`. A supplied context mask may override the Graph fallback; test the **effective mask**, not merely a YAML topology string. The mask is communication metadata, not an extra observation feature to concatenate into the actor encoder. [R6]

### 4.2 Methods

| ID | Existing implementation | Required main setting |
|---|---|---|
| `identity` | `communication/identity.py::IdentityComm` | Exact pass-through; zero branch parameters and zero payload. |
| `local_capacity` | `communication/local_control.py::LocalCapacityComm` | Bottleneck 32; no cross-agent inputs. |
| `broadcast` | `communication/broadcast.py::BroadcastComm` | Message width 32, residual enabled, all other senders. |
| `gated` | `communication/gated.py::GatedComm` | Message width 32; existing soft gate, `hard=false`; no top-K. |
| `attention` | `communication/attention.py::AttentionComm` | Total key width 32, total value width 32, 4 heads, 1 round. |
| `graph_full` | `communication/graph.py::GraphComm` | Total key/value widths 32, 4 heads, 1 round, full off-diagonal connectivity. |

Use the repository's actual constructor/schema names. Verify all defaults and materialize them in resolved configs. If HEAD has changed, document the mapping; do not silently invent an unsupported key. Keep existing initialization schemes, record them, and do not independently optimize a method's learning rate after seeing its test score.

The default Graph core is intentionally **not electrical-topology-aware**. It compares the existing relation-scoring operation without simultaneously restricting its communication budget. It is also not guaranteed numerically equivalent to the dot-product Attention module. [R6–R7; P5–P7]

### 4.3 Identity, LocalCapacity, and severed Broadcast

For local hidden state `h_i`, the existing LocalCapacity operation is:

```text
h_i + decoder(tanh(encoder(h_i)))
```

At hidden width 128 and bottleneck 32, the two bias-free matrices contain 8,192 active parameters. One-round Broadcast uses same-sized projections but aggregates **other agents'** encoded messages. Identity has no such branch. Broadcast with zero incoming messages contributes zero through its bias-free decoder; the unused branch does not become active local computation. [R8]

Tests must verify the distinction, including cross-agent gradients: changing another agent's hidden state must not affect the LocalCapacity output for receiver `i`. Constructed nondegenerate tests may demonstrate communication influence, but trained policies are not required to exhibit a positive influence or benefit to pass implementation checks.

### 4.4 Parameter and cost controls

Record actual actor encoder/head, communication branch, critic, and total trainable parameters **per trained group**. For PCP, separate predator parameters from the tiny optimized-but-overridden prey policy. Never count dormant parameters as active capacity merely to claim fairness.

The reviewed standard branch counts are useful expectations, not substitutes for measuring the constructed models: Broadcast 8,192; Gated approximately 8,321; Attention 16,384; Graph approximately 16,416. Resolve discrepancies against the actual code and optional guards. LocalCapacity-32 exactly matches the basic Broadcast branch, not all four communicators. [R6, R8, R9]

The optional LocalCapacity-64 branch has 16,384 active projection parameters at hidden width 128. It is an exact projection-count match to the stated Attention setup and an approximate match to Graph—not an identical computation or optimization landscape. Do not add unused scalar parameters to force equality.

Main communication costs are explicit float32 payload equivalents. Broadcast/Gated transmit 32-scalar payloads; Attention/Graph use their actual key-plus-value representation. Queries computed locally are not automatically transmitted. Count emitted packets/scalars separately from directed receiver deliveries. Count actual mask effects, but **a small soft gate is not a dropped packet**. Preserve transport-overhead exclusions. Equal `message_dim` does not mean equal bandwidth. [R6; P12]

---

## 5. Environment and training contracts

### 5.1 PCP: preserve the corrected task

Use `vmas_predator_capture_prey` and the existing scripted-prey scenario. VMAS is the simulation framework, not a new environment proposed here. [P13, R1] Preserve:

- Three trained homogeneous predators, one scripted prey, two landmarks, 100-step horizon.
- Predator actor observation width 17 in the reviewed layout; physical critic state width 22.
- Restricted `predator_sensing_radius=1.0` and global `predator_sensing_radius=null`.
- Identical physics, reset distributions, prey script, reward settings, action bounds, and exogenous noise between visibility conditions.
- No detector/capturer roles, no extra observation IDs, and no learned prey behavior.
- `return_groups=[adversary]`; the prey group's optimized output remains overridden as in the existing implementation. Do not change that behavior in this comparison.

Verify the corrected RNG/visibility versions and the actual `runtime_contract`. Global prey visibility is not a full-state oracle: the protocol itself notes possible teammate-private velocities and implicit movement signals. [R1, R4]

Copy scientific settings into a **new** suite protocol, rather than marking the historical candidate protocol approved:

| Setting | PCP core |
|---|---:|
| Learning algorithm | Existing BenchMARL MAPPO |
| Frames per policy | **600,000** |
| Gamma / GAE lambda | 0.95 / 0.9 |
| Learning rate / Adam epsilon | 5e-5 / 1e-6 |
| Entropy coefficient / PPO clipping | 0.1 / 0.2 |
| Collected frames per batch | 6,000 |
| Parallel environments per worker | 10 |
| Minibatch size / PPO passes | 400 / 10 |
| Global gradient-norm clipping | 5 |
| Retained native checkpoint frames | 120k, 240k, 360k, 480k, 600k |
| Final scientific checkpoint | Exactly 600k; not best-on-test |

Retain the task-specific critic and all other validated settings. Preserve the existing development-evaluation cadence unless a new common schedule is explicitly frozen before any scientific run; the reviewed protocol uses interval 12,000 with 128 deterministic full episodes, and its initial evaluation behavior must be read from the implementation. Log actual evaluation frame positions. Evaluation can be a substantial part of runtime; include it in profiling. Do not lower evaluation count for one method alone. [R1, R10]

Fix checkpoint retention so the five mandatory checkpoints survive native cleanup. `keep_checkpoints_num=2` in a historical candidate is not adequate without the existing retention mechanism. Prefer a new protocol with explicit sufficient retention and verify files after training, not only while the run is alive. Never edit the old protocol's retention value retroactively. [R11]

### 5.2 MAPDN: preserve research-adapter semantics

Use `mapdn_voltage_control`, `data/case33_3min_final`, and the vendored research adapter. Resolve paths through explicit user configuration; do not assume the data are committed to GitHub.

Preserve the already integrated alignment: the action and reward refer to the same input row, and the next observation is solved for the next row. Keep horizon **239**, history **1**, distributed **one-inverter-per-agent** mode, measurement noise **false** for this scientific suite, reset actions **false**, and the historical voltage/reactive-power objective. These suite settings override any generic task-class default. [R2–R3, R12]

Important implementation detail: the inspected `PowerGridTaskClass.state_spec()` returns `None`, and its runtime contract identifies critic information as **concatenated zone-local observations**. Do not give it a newly introduced full physical state and describe that as unchanged MAPPO. Keep the centralized critic interface currently used by this task. [R12]

The task configuration includes voltage limits 0.95–1.05 p.u., L1 voltage barrier, voltage weight 1.0, reactive-effort weight 0.1, and action scale 0.8. Confirm the resolved research settings, units, inverter capability calculation, and bus population against the vendored adapter. Do not replace these with upstream defaults. Actual agent count and agent-to-inverter/bus mapping must be exported from loaded case33 data, not inferred from the name “33.” [R12–R13; P8]

Proposed main training budget is **131,072 frames for every MAPDN method**, larger than the previous 32,768-frame diagnostic. This is a declared finite allocation, not a convergence claim or a literature-derived minimum.

| Setting | MAPDN core |
|---|---:|
| Learning algorithm | Existing BenchMARL MAPPO |
| Frames per policy | **131,072** |
| Gamma / GAE lambda | 0.99 / 0.9 |
| Learning rate / Adam epsilon | 5e-5 / 1e-6 |
| Entropy coefficient / PPO clipping | 0 / existing 0.2, verified in resolved config |
| Collection | One environment, 256 frames per batch |
| Minibatch size / PPO passes | 64 / 4 |
| Global gradient-norm clipping | 5 |
| Primary backend | CPU, one numerical thread, unless a new separately validated profile is frozen |
| Retained reporting checkpoints | 32,768; 65,536; 98,304; 131,072 |
| Final scientific checkpoint | Exactly the frozen budget |

Read `_spec()` in `scripts/run_mapdn_confirmation.py` and materialize inherited algorithm parameters. The table is not permission to omit unlisted parameters or use whatever a newer dependency happens to default to. The current task binding ignores the requested `num_envs` when constructing the single environment; do not advertise vectorized MAPDN throughput without implementing and validating it separately. [R13]

### 5.3 Qualification and budget adequacy

Generate separate qualification rows for Identity on seeds 1000 and 1001: both PCP visibility conditions at 600k, and MAPDN at 131,072. This is **six qualification policies**, not part of the 90 main policies. These runs supply target-runtime learning-health and throughput evidence; they do not require communication to win.

For MAPDN, retain initial and 32,768/65,536/98,304/131,072 validation measurements. Report changes in voltage magnitude, violating-bus fraction, reactive effort, and return. The previous stability tolerances—5e-5 p.u., 0.005 fraction, 0.02 reactive-effort units as verified, and `max(1, 0.05*abs(initial return))`—may be retained as a labeled descriptive diagnostic across two successive checkpoints, but do not import the old seed-locked allocation function unchanged. The main budget here is fixed, not selected for a favorable outcome. [R3, R14]

A numerically valid policy that still improves at the budget is **budget-adequacy unresolved**, not an engineering failure. A flat but poor curve is not evidence of optimality. If measured runtime makes the allocation infeasible, produce a proposed budget amendment before main training. Never silently shorten runs, stop a method when it looks good, or keep extending only losing communicators.

Core training allocation at the stated budgets is **36,000,000 PCP frames + 3,932,160 MAPDN frames**. Qualification adds **2,400,000 PCP + 262,144 MAPDN frames**. Short smokes and evaluation add further cost. These counts are arithmetic, not a wall-clock forecast.

---

## 6. Data, seeds, and held-out banks

### 6.1 PCP banks

Use a new explicit main bank of **256 complete 100-step episodes**, initially proposed environment seeds `3100000..3100255`, shared across methods and both visibility conditions. Use a distinct development bank and a separate random-action audit bank; verify no accidental reuse with retained historical banks before freezing.

Paired visibility episodes must share initial physical state and exogenous prey disturbances. Their actor observations are deliberately different. Channel/module RNG must not shift the environment's disturbance stream. Preserve the current reset and evaluation-isolation protections. [R1, R4]

### 6.2 MAPDN banks

Keep chronological **60/20/20** train/validation/test blocks, training-only normalization, and training-derived inverter capacity. Generate manifests from actual timestamps/row indices with complete history and terminal footprints. Different random seeds selecting the same rows do not create independent data.

Proposed banks: 16 training-only normalization episodes, 8 qualification-validation windows, 8 separate main-validation windows, and 16 main-test windows. Preserve nonoverlap within the intended bank structure and exclude previously inspected windows when their exact identifiers are available. Fit one normalizer per task contract, not one per method; freeze it before training. [R3, R14]

Audit **all available** old bank files, not only D5's first bank. If old banks cannot be accessed, record `historical_overlap_verification=unavailable`; do not claim the new windows are wholly uninspected. Still enforce the chronological split and within-new-suite footprint checks. If insufficient legal windows remain, fail with an explicit diagnostic and seek a precollection amendment; never silently shrink, overlap, or move the test bank into training.

Do not normalize using the whole dataset, recompute inverter limits using test maxima, or rebalance windows after seeing voltage results. Protect data identity with hashes, source/license attribution, and vendored-adapter provenance. Nonoverlapping windows can still be temporally dependent; this remains evaluation within the same historical feeder process, not independent-grid generalization. [R12–R15; P8]

### 6.3 Paired randomness and immutable test use

Separate namespaces for initialization/training, validation, test, channel interventions, topology generation, and bootstrap resampling. Record algorithms and seeds, not only a master integer. Same training-seed labels provide a declared experimental block; they do not guarantee identical stochastic training paths between architectures.

Seal all core final checkpoints, training failures, and protocol decisions before opening final test episodes. Test results must not select checkpoints, rewards, methods, hyperparameters, or banks. If final results motivate another experiment, create a separately labeled follow-up.

---

## 7. Evaluation and statistical reporting

### 7.1 Required final arms

For every evaluable main policy run **live** and **fully severed** communication on identical held-out initial conditions. Each arm evolves its own trajectory; do not hold later observations artificially equal after actions diverge. Freeze weights in both arms and use deterministic actions.

For Identity and LocalCapacity, require exact within-runtime live/severed equality in actions, domain outcomes, and returns. For learned communicators, zero, positive, or negative reliance is a scientific outcome—not a pass criterion. Message intervention must override runtime masks and any replay metadata, must be restored even on exception, and must not consume environment RNG. [R1, R6; P9]

Count core final episodes explicitly:

- PCP: 60 policies × 2 arms × 256 = **30,720 episodes**, or 3,072,000 full-horizon transitions when all episodes complete.
- MAPDN: 30 policies × 2 arms × 16 = **960 trajectories**, or 229,440 full-horizon transitions when complete.

Strict replay runs and development evaluation are additional compute, not extra independent scientific samples. The existing MAPDN evaluator repeats the full result for exact replay; account for that cost if retained. Never advertise repeat evaluations as extra training seeds. [R16]

Single-sender suppression and cross-episode message substitution may be exposed as opt-in diagnostics but are not required for the core count. Substitution requires method-correct packet handling, including keys and values for attention methods, and documented distribution shift. Do not reuse a Broadcast-only donor-packet routine for another mechanism without tests.

### 7.2 PCP measurements

Primary endpoint: probability of **first rewarded contact at or before transition 50**. Retain the exact existing transition convention and test the 49/50/51 boundary with deterministic fixtures. Do not redefine contact as simultaneous capture or change the endpoint because another metric looks better.

Secondary outcomes: adversary-group return, contact by 100, first-contact time with noncontacts retained as censored, contact duration/pairs, collisions, boundary occupancy, distances, and visibility opportunities. Keep the full 100-step episode after contact unless the existing task itself terminates; do not invent early stopping. Never average the zero-sum predator and prey group returns into the study outcome. [R1]

### 7.3 MAPDN measurements

Primary endpoint: mean voltage-violation magnitude in p.u. at the existing action/reward-aligned physical observation. Preserve the adapter's bus set and averaging convention; independently check the intended expression on fixtures:

```text
per_bus_violation(v) = max(v_lower - v, 0) + max(v - v_upper, 0)
```

Export bus identities, valid bus-transition denominators, and the adapter's actual reduction. Treat any discrepancy as a measurement-contract issue, not a reason to quietly change the metric.

Secondary outcomes: violating-bus fraction from integer counts; mean absolute inverter reactive effort with units; total line loss; native return; action saturation; actual power-flow failures; unobserved physical transitions; and curtailed active power where present.

Team diagnostics are repeated across agents for transport. Count **one copy per environment transition**, not N copies. Solver success is not voltage safety. Solver failure must never become zero violation or a missing row silently removed from the comparison. [R12–R13; P8]

If a scientific rollout encounters physical solver failure, retain the event, attempted-transition denominator, valid-observation denominator, and episode status. Report observed voltage statistics as conditional on valid physical observations, alongside failure frequency. Do not claim a clean voltage ranking for incomplete physical banks. Do not invent a worst-case imputation after seeing which method fails.

### 7.4 Primary and secondary contrasts

Prespecify four primary **method-versus-Identity** contrasts within each of the three task conditions: 12 descriptive effects total. Orient effects so positive means improvement:

```text
PCP benefit_m = contact_success_m - contact_success_identity
MAPDN benefit_m = voltage_violation_identity - voltage_violation_m
```

Separately report method-versus-LocalCapacity effects, with an exact-capacity qualification only where supported. Prespecified secondary comparisons include Gated/Attention/Graph versus Broadcast, and PCP information interactions:

```text
interaction_m = (success_m - success_identity)_restricted
              - (success_m - success_identity)_global
```

Reliance is a separate within-policy contrast:

```text
PCP reliance_m   = success_live - success_severed
MAPDN reliance_m = violation_severed - violation_live
```

Keep native units. PCP absolute probability differences may be displayed as percentage points. Do not divide by negative returns to claim universal percentage gains. Never pool contact probability and voltage violation into one raw cross-domain leaderboard. [R0, pp. 15–18]

### 7.5 Uncertainty, failures, and multiplicity

Average episodes within each trained policy first, then compare the independent training-seed blocks. Show every seed and the paired differences. Use a fixed, recorded RNG with 10,000 seed-level bootstrap draws for descriptive 95% intervals; resample shared seed blocks jointly when deriving visibility interactions or contrasts sharing a baseline.

Five-seed pointwise intervals are not simultaneous guarantees across 12 comparisons. Label them descriptive; do not add significance stars or claim equivalence merely because an interval crosses zero. A confirmatory hypothesis-testing or equivalence plan would require a separately justified family/margin/sample-size design. The statistical reference motivates uncertainty reporting; it does not certify that this particular allocation is sufficient. [P10]

Keep the full requested denominator in completion tables. Numerical divergence is an observed failed method/seed, not a seed to replace. Infrastructure failure can lead to a user-authorized retry, with the original attempt preserved and no best-of-attempts selection. If a cell lacks a complete planned seed cohort, label its inferential output incomplete; conditional partial summaries may be shown, but not advertised as the completed suite ranking.

### 7.6 Required figures and tables

Produce, from retained machine-readable data: (1) per-method primary outcomes with individual seeds for each condition; (2) seed-paired effect plots; (3) benefit-versus-reliance plots; (4) complete validation learning curves with the fixed endpoint; (5) parameter and emitted/delivered-payload tables; (6) MAPDN voltage/effort/loss tradeoff plots; and (7) the planned/completed/failed/missing matrix.

The plotting program must not fabricate data when real outputs are absent. Tiny synthetic fixtures used in tests must be prominently labeled and placed outside scientific results. Report generation must work headlessly.

---

## 8. Repository-specific implementation work

### 8.1 Audit before editing

Read local `agents.md` instructions, the newest status entry, task registry, corrected PCP protocol, D4 plan, MAPDN implementation status/phase plan, and current communication definitions. Compare local HEAD to the inspected revision. Inspect `git status`; preserve unrelated and uncommitted user work. Work on a new branch/worktree if appropriate, without deleting or resetting existing worktrees.

Create `docs/experiments/pcp_mapdn_mechanisms_v1/IMPLEMENTATION_AUDIT.md` with observed HEAD, local modifications, actual available artifacts/runtime/data, and a proposed file map. Do not let stale README wording override later dated completion records. Do not claim access to Newton or historical archives that are not mounted. [R0–R5]

### 8.2 Known extension points and blockers

| Existing path | What to inspect/reuse | Required change or constraint |
|---|---|---|
| `src/commstudy/communication/*.py` | Existing module forward contracts and statistics | Reuse mechanisms; add tests/helpers only where required. |
| `src/commstudy/models/model.py` | Actor allowlist, context keys, grouped models | Keep injection point and no privileged actor inputs. |
| `src/commstudy/experiments/runner.py` | Managed BenchMARL training | Reuse; do not write another PPO loop. |
| `experiments/config.py`, `protocols.py`, `sweeps.py` | Strict configs, frozen specs, plans | Extend with new suite identity/schema without bypassing old guards. |
| `experiments/bookkeeping.py` | JSON/path loading, atomic records | Normalize public path-like inputs and test serialized round trips. |
| `experiments/next_phase.py` | Old 3-method, seed/budget-locked constants | Do not mutate historical constants to reinterpret completed results. |
| `scripts/run_main_direction_phase.py::phase_spec` | MAPDN method construction | It currently permits only Identity/LocalCapacity/Broadcast; new construction must cover all required methods. |
| `scripts/evaluate_pcp_budget.py` | Audit and retention helpers | CLI is tied to D4 visibility/Broadcast protocol; a generic new-suite evaluator is required. |
| `analysis/pcp_budget.py`, `pcp_domain.py`, `saliency.py`, `diagnostics.py` | Rollouts, interventions, domain metrics, frozen loading | Reuse primitives; remove no guards globally merely to accept new rows. |
| `scripts/evaluate_mapdn_paired.py`, `analysis/mapdn_paired.py` | Paired test/replay and checkpoint validation | Verify every module works; parameterize new bank/method/frame contracts through a new suite path. |
| `tasks/torchrl/power_grids.py`, `mapdn_data.py` | Task, physical metrics, splits, normalization | Preserve semantics and explicit counts; validate real data. |
| `vendor/mapdn-source/` | Versioned research adapter | Import this exact implementation, not an unrelated pip-installed package. |
| `slurm/pcp_03_main_comparison.sbatch` | Historical 30-row manifest workflow | Do not relabel or repoint it to pretend this new suite already exists. |
| `slurm/newton_env.sh` | Existing environment detection | Its CUDA/thread defaults are not the new suite's scientific runtime contract. |

Paths prefixed only with `experiments/`, `analysis/`, or `tasks/` above are under `src/commstudy/`. Inspect the actual local APIs before applying changes. These paths are grounded in the inspected tree and imports; not every helper's complete behavior was re-audited for this document. [R7–R16]

### 8.3 Proposed new files

Use equivalent established project abstractions where available, but return a concrete mapping if names differ:

```text
configs/experiments/pcp_mapdn_mechanisms_v1.yaml
configs/protocols/pcp_mechanisms_v1.yaml
configs/protocols/mapdn_mechanisms_v1.yaml
configs/sweeps/pcp_mechanisms_v1.yaml
configs/sweeps/mapdn_mechanisms_v1.yaml
configs/sweeps/mapdn_topology_v1.yaml                 # opt-in
configs/sweeps/local_capacity_sensitivity_v1.yaml     # opt-in
configs/models/pcp_... and mapdn_... presets as needed

src/commstudy/experiments/mechanism_suite.py
src/commstudy/analysis/mechanism_suite.py
scripts/communication_suite.py

slurm/communication_comparison_v1/                   # Section 11
src/tests/test_mechanism_suite_*.py
docs/experiments/pcp_mapdn_mechanisms_v1/
```

The proposed top-level suite YAML is a **new schema to implement**, not a claim that the current config parser already understands arbitrary study fields. Validate unknown keys strictly and translate the study design into existing resolved `ExperimentSpec` objects.

Expose method registry metadata: module/class path, communication capability, expected packet representation, required context keys, and null-control status. Do not infer method identity only from substrings in run IDs. A separately serialized MAPDN `phase_gated` name must resolve on a fresh interpreter; earlier launcher bugs show why an in-memory spec alone is insufficient.

### 8.4 Do not bypass scientific launch guards

PCP currently restricts ad hoc scientific training and binds manifests to protocol/source/runtime evidence. Add a legitimately new supported protocol/stage or a narrowly scoped new suite adapter. Do not change every run to `--validation-run`, delete approval checks, promote an old candidate without evidence, or replace strict checkpoint loads with `strict=False`.

The known serialized-path issue must be repaired in a new source version. Test a JSON record containing a string path through the actual post-training-to-evaluation route, not only `Path` construction in isolation. Preserve the old failure/workaround as history. [R0, p. 7; R11, R14]

### 8.5 Runtime qualification

The historical PCP scientific confirmation is CPU-based; a GPU request in an old Slurm script does not certify the corrected task on CUDA. The new Linux CPU profile also needs real execution checks. Default to explicit CPU profiles until a target profile passes tests and qualification; alternatively qualify CUDA before freezing a fresh cohort. Do not mix backends within a comparison or silently fall back from CUDA to CPU.

MAPDN's power-flow simulator is CPU work. Request a GPU only when an explicitly profiled actor/backend plan justifies it or a documented site allocation policy requires one. Distinguish requested resources from actual compute devices. The public Newton inventory includes multiple GPU generations, so inspect the allocated hardware and pin a documented family where needed rather than assuming V100 or H100 from a job name. [R17–R18; S4]

---

## 9. Optional electrical-topology study

Implement only as a separately enabled supplement. Do not make it a condition for returning all five core mechanisms.

Export the real agent-to-inverter-to-bus mapping, line/transformer/switch status, and the graph-construction rule. Electrical buses and controllable-inverter agents are different node sets. A bare 33×33 bus adjacency is generally not an agent message mask.

For a first explicit rule, construct a static undirected **electrical-hop k-nearest-agent graph** from the in-service feeder graph, with k=2 (or `min(2,N-1)`), deterministic tie-breaking, and symmetric union. A same-bus distance is zero. Freeze the rule, bus mapping, and generated mask before collection. Name it `electrical_hop_knn_k2`, not “direct electrical adjacency.” It is a proposed communication design using public topology, not proof of actual telecommunications links.

Build a matched-random control with the same node count, no self edges, and the same edge count; preserve degree sequence and connectivity where feasible. Verify the graph differs from the electrical graph. If constraints make a distinct control impossible at the loaded agent count, mark the comparison degenerate instead of looping indefinitely or silently relaxing requirements. Export realized degree/connectivity statistics and all generation seeds. Use a predeclared graph bank so one fortunate random mask does not become a structural conclusion.

Train the two sparse Graph variants for the same main budget and training seeds; reuse core full Graph only when the shared settings are identical. Keep the physical feeder, observations, reward, and message processor fixed. This compares **communication topology on case33**, not changing physical grid topology. Use runtime effective-mask assertions so an all-ones adapter mask cannot accidentally override the sparse graph. [R0, pp. 7, 21; R7]

---

## 10. Command-line contract the agent must implement

Prefer one tested Python entry point with subcommands wrapping existing project machinery. The following names are the requested interface; equivalent interfaces require a clear crosswalk in the runbook.

```text
python scripts/communication_suite.py plan          --suite CONFIG --out RUN_ROOT
python scripts/communication_suite.py preflight     --run-root RUN_ROOT --profile PROFILE
python scripts/communication_suite.py smoke         --run-root RUN_ROOT --task TASK --index I
python scripts/communication_suite.py qualify       --run-root RUN_ROOT --task TASK --index I
python scripts/communication_suite.py freeze        --run-root RUN_ROOT
python scripts/communication_suite.py train-row     --run-root RUN_ROOT --task TASK --index I
python scripts/communication_suite.py close-training --run-root RUN_ROOT --task TASK
python scripts/communication_suite.py evaluate-row  --run-root RUN_ROOT --task TASK --index I
python scripts/communication_suite.py analyze       --run-root RUN_ROOT
python scripts/communication_suite.py status        --run-root RUN_ROOT
python scripts/communication_suite.py retry-plan   --run-root RUN_ROOT --task TASK --reason TEXT
```

`plan` only resolves deterministic configs and writes a new preparation directory; it must not train, open held-out trajectories, or call `sbatch`. Plan output must show all 90 core rows and separate optional/qualification/smoke counts. `preflight` validates real environment/data availability; expensive checks run inside an allocation, not as login-node training.

`freeze` consumes recorded qualification/preflight evidence, captures budgets/runtime/source/banks and writes the main approval artifact. It must not require a positive reward effect. If qualification integrity is incomplete, return an explicit pending/blocked status rather than fabricating approval. New main manifests cannot launch without that artifact.

A `train-row` resolves exactly one immutable manifest row by dense zero-based index. It refuses negative/out-of-range indices, altered scientific hashes, accidental test split use, incompatible runtime, existing success overwrites, or a concurrently owned run. It records the Slurm job/array identity and actual devices.

`close-training` seals every planned row's final checkpoint or explicit terminal failure status. Unsafe provenance or missing unresolved jobs blocks closure. An explicitly recorded scientific training failure remains a failed planned row and can permit an incomplete-cohort evaluation plan; do not require it to magically become a success. After closure, evaluations of valid rows may proceed, while failed rows emit `not_evaluable` receipts. The final report remains incomplete wherever a planned scientific cell failed.

`retry-plan` is a nonexecuting command that writes a separate list of eligible failed/infrastructure-interrupted rows and the reason. It must not relabel numerical failure as infrastructure failure automatically. Actual retries require an explicit user action, preserve attempts, and do not select the highest scoring attempt.

The runbook must contain commands validated against implemented `--help` output. Do not return sample commands with flags the implementation ignores or has not defined.

---

## 11. Required Slurm deliverables

### 11.1 Return these actual files

Under `slurm/communication_comparison_v1/`:

```text
README.md
cluster.env.example
common_env.sh
00_probe.sbatch
01_bootstrap.sbatch
02_smoke.sbatch
03_qualify.sbatch
04_train_pcp.sbatch
05_train_mapdn.sbatch
06_close_training.sbatch
07_evaluate_pcp.sbatch
08_evaluate_mapdn.sbatch
09_analyze.sbatch
submit_full.sh
status.sh
retry_failed.sh
```

The agent may reuse a generic worker behind these task-specific wrappers, but both PCP and MAPDN launch paths must actually exist and be exercised in tests. Scripts must be executable or explicitly invoked through `bash` as appropriate. `submit_full.sh` is the end-user entry point; scientific settings come from manifests, not a list of fragile shell overrides.

### 11.2 Cluster configuration

Require an explicit profile containing absolute source snapshot/run root, task-specific interpreter/venv paths, account, partition, allowed resource shape, time/memory/CPU/GPU requests, and concurrency caps. The old repository uses `normal` and `--gres=gpu:1` and provides module candidates, but the user's current account entitlements, installed module names, and accessible paths must be probed rather than assumed. [R17–R18]

Use a new isolated environment for this suite. Do not overwrite `.venv`, `.venv-newton`, `.resolved_env`, or the preserved MAPDN environment. Do not reuse a macOS arm64 wheel lock as a Linux artifact lock. Resolve a compatible target-runtime package set in setup, test it, and freeze its versions/artifact identities before scientific rows.

Runtime checks must verify import origins for `commstudy` and vendored `mapdn`, plus actual versions of PyTorch, TorchRL, TensorDict, BenchMARL, VMAS, NumPy, pandas, pandapower, PettingZoo, and Gymnasium. Scientific workers never `pip install --upgrade`, rebuild the environment, or download data. Missing imports are fatal with a clear setup remedy; a module warning followed by an accidental fallback is not acceptable.

Set OMP, MKL, VECLIB, and NUMEXPR numerical threads to **1** and validate `torch.get_num_threads()` for the frozen CPU protocol, regardless of allocated `--cpus-per-task`. More allocated CPUs may cover overhead; they do not silently change the numerical protocol. Explicitly configure PyTorch inter-op threads if required and record the decision. Avoid nested multiprocessing oversubscription.

Because compute-node network access is not assured, support a pre-staged Linux wheelhouse/data path. If internet access or credentials are missing, return a precise staging blocker and usable scripts; never substitute fabricated scientific data. Do not print private tokens or environment secrets into provenance.

### 11.3 Stage graph

Support separate safe phases:

```text
probe/setup -> real smoke checks -> qualification -> freeze
                                                  |
                       +--------------------------+------------------------+
                       |                                                   |
                 PCP train array                                    MAPDN train array
                       |                                                   |
                 PCP closure audit                                 MAPDN closure audit
                       |                                                   |
                 PCP eval array                                     MAPDN eval array
                       +--------------------------+------------------------+
                                                  |
                                      final/partial analysis report
```

Freeze is the boundary between development and scientific execution. The user should be able to run `prepare` and inspect qualification/compute reports before `main`. A convenience command may submit the stage DAG after explicit authorization, but the workers still enforce evidence gates.

Generate train and eval array bounds from the immutable manifest counts: core PCP indices **0–59**, MAPDN **0–29**. Never retain the old hardcoded `0–29` PCP array and call it the two-visibility six-method suite. Concurrency caps are configurable; do not launch 90 jobs with unbounded parallelism by default.

Use `afterany` for closure/status collectors that must report failed predecessors, and `afterok` for stages that require a successful integrity gate. Analysis should run after evaluation arrays terminate so it can report failed/missing cells rather than remain invisible forever. Its output must state whether the scientific suite is complete. These choices follow Slurm's dependency meanings; receipts determine scientific readiness, not scheduler completion alone. [S1–S3]

### 11.4 Submission behavior and safety

`submit_full.sh` must default to **dry-run**. Only an explicit `--execute` sends jobs to Slurm. The interface must let the user select `--phase prepare` or `--phase main`, an existing run root, a cluster profile, and optional supplements. A main-phase dry-run must show every command, manifest hash, row count, resource request, dependency, and estimated cost without calling `sbatch`.

Construct commands as argument arrays with proper quoting; do not use shell `eval` or interpolate untrusted JSON into executable shell. Slurm directives do not expand shell variables, so place variable resource/account/output values in the `sbatch` argument array or generate literal validated directives. Put sbatch options before the script path; later words are script arguments. [S2]

Create absolute log directories **before** submission because Slurm opens logs before the worker can create them. Use `%A_%a` for arrays and `%j` for singleton jobs. Capture `sbatch --parsable` output, correctly handling an optional `;cluster` suffix, and atomically append submitted IDs to a submission ledger. A nonzero submission result aborts further dependent submission while preserving already submitted job IDs. [S1–S2]

Inside every worker: `set -euo pipefail`; explicit working directory/source snapshot; exact interpreter; frozen runtime/profile checks; `srun` or the site's verified equivalent; correct propagation of Python exit status. Do not start scientific Python in the background and let the shell report success. Separate row-local artifacts prevent concurrent writers from corrupting a shared CSV.

Prevent duplicate full submissions using a run-root lock and inspect the saved job ledger. Scheduler status and artifact status are separate. Never reclaim a running row solely because a timestamp looks old; check job ownership and require an explicit recovery path.

### 11.5 Walltime, preemption, and retry semantics

Profile actual PCP and MAPDN training plus evaluation on the selected nodes. Estimate training, checkpoint, evaluation/replay, and archive time separately, with an explicitly labeled operational margin. Never extrapolate the old Simple Spread V100 average into a MAPDN forecast. [R0, §8.5]

Default to no automatic requeue for scientific rows unless full optimizer, scheduler, RNG, environment, and collector state are demonstrably saved and restored. Native actor/critic checkpoints alone are not exact training resume points. A timeout retry should ordinarily restart from initialization in a new preserved attempt with an explicit reason, not silently continue from a policy-only checkpoint. Do not implement exact-resume machinery as a new large project for this request.

Use termination handling to write best-effort interruption receipts, but do not claim shell traps recover from every node failure or SIGKILL. `status.sh` must reconcile Slurm terminal state with artifact state so absent final receipts are visible. A walltime problem should produce a resource-plan adjustment, not secret scientific-budget truncation.

### 11.6 Returned command sequence

Document an actual verified sequence equivalent to:

```bash
# New interfaces to be implemented by the agent, not commands already present today.
python scripts/communication_suite.py plan \
  --suite configs/experiments/pcp_mapdn_mechanisms_v1.yaml \
  --out /absolute/path/to/study-root

bash slurm/communication_comparison_v1/submit_full.sh \
  --phase prepare --run-root /absolute/path/to/study-root \
  --profile /absolute/path/to/cluster.env

# Re-run with --execute after reviewing the printed preparation plan.
# After setup/smoke/qualification, freeze the scientific suite:
python scripts/communication_suite.py freeze --run-root /absolute/path/to/study-root

bash slurm/communication_comparison_v1/submit_full.sh \
  --phase main --run-root /absolute/path/to/study-root \
  --profile /absolute/path/to/cluster.env

# Add --execute to actually submit the inspected main DAG.
```

The runbook must also show single-task-only submission, monitoring, inspection of failed rows, and a dry-run retry plan. Fill actual account/module/path values only when observed or supplied; otherwise identify each required user value and keep `cluster_ready=false` until preflight passes.

---

## 12. Artifact and provenance contract

Use one study root outside the immutable source snapshot:

```text
study-root/
  preparation/
    suite.yaml
    source_inventory.json
    runtime_profiles/
    data_inventory.json
    observation_and_graph_contracts.json
    pcp_manifest.csv
    mapdn_manifest.csv
    evaluation_banks/
    normalizer.json
    qualification_manifest.csv
  gates/
    preflight.json
    smoke_report.json
    qualification_report.json
    main_approval.json
    training_closure_pcp.json
    training_closure_mapdn.json
  runs/<task>/<run_id>/<attempt_id>/
  evaluation/<task>/<run_id>/<evaluation_attempt>/
  analysis/
  slurm_logs/
  submissions.jsonl
```

Integrate with the project's existing `RunContext` structure rather than duplicating its contents unnecessarily. A manifest row must bind suite/task/condition/method/training seed, frame budget, fully resolved model and critic settings, graph/channel identity, runtime/backend profile, source and data hashes, normalization/bank digests, output identity, and stage. Actual names may differ but the information must be explicit.

Separate scientific identity from operational placement. Moving a read-only artifact should not rewrite its scientific hash; record relocation mappings independently. Changing devices, numerical libraries, task settings, or selected frames is not merely a path relocation.

Native checkpoints and final managed policies must agree on retained model states. Retain all required actor and critic groups and trained-frame counters. Distinguish inference reconstruction from exact training resumption. Serialize paths portably and normalize path-like values at API boundaries.

Write JSON atomically with finite-number enforcement; use explicit nullable/missing fields where values are undefined rather than NaN pretending to be a number. Per-row failures must retain exception type, phase, traceback, source/runtime hashes, actual frames, and available checkpoints. No automatic deletion of failed attempts, logs, old outcomes, or user data.

Keep raw trajectories sufficient to reconstruct primary endpoints and interventions, plus action/model hashes and exogenous identities. Do not require enormous hidden-tensor dumps for every time step unless needed for a declared diagnostic. Estimate bytes and file counts from a representative run, verify quotas/free space, and archive with read-back checks. Log paths and expected storage are part of preflight.

Source snapshots must exclude generated logs, runs, caches, and virtual environments while including actual source/config/vendor/runtime-lock files used by the study. Do not keep editing a shared checkout while queued workers import it. A code commit name by itself does not capture dirty changes; include a clean snapshot or a hash-bound diff and inventory.

---

## 13. Test plan and acceptance criteria

### 13.1 Unit and contract tests

| Test area | Required evidence |
|---|---|
| Coverage | Exactly 60 PCP and 30 MAPDN core rows; every method/condition/seed present once. |
| Schema | Unknown keys and unsupported methods fail; all method presets reconstruct in a fresh interpreter. |
| Scope | MAPPO unchanged; task and critic differ only where explicitly prescribed. |
| Identity | Same local tensor behavior, zero payload, no extra trainable branch. |
| LocalCapacity | Active branch gradients; no cross-agent dependency; unaffected by message interventions. |
| Mechanisms | Gated/Attention/Graph actually constructed, used by the predator/agent actor, and present in saved weights. |
| Null masks | No incoming messages gives finite outputs and zero remote contribution. |
| Mask semantics | Receiver/sender orientation, self exclusion, runtime precedence, padding, and suppression correctness. |
| Payload | Soft gating is not mistaken for packet saving; attention keys/values counted; directed deliveries separated. |
| PCP | 17/22 reviewed dimensions, physical-state matching, scripted prey, return groups, contact deadline boundaries. |
| MAPDN | Correct import origin, agent mapping, action/reward row alignment, terminal footprint, physical denominators. |
| Data | Chronological split; no train/test normalization leakage; full-window exclusion; duplicate-window failures. |
| Checkpoints | Mandatory native files retained after cleanup; strict final reload; string-path JSON round-trip works. |
| Evaluation | Independent arm trajectories, reset/RNG isolation, exact local nulls, interventions restored after failure. |
| Statistics | Seed-level aggregation; known-sign fixtures; paired interaction arithmetic; incomplete cells clearly labeled. |
| No result fishing | Test inputs inaccessible before closure; no best-on-test checkpoint; no method-dependent early stopping. |
| Slurm mapping | First/last/all row indices map correctly; optional rows do not shift core identity silently. |
| Shell | `bash -n`; ShellCheck where available; all command quoting checked with paths containing spaces. |
| Scheduler mock | Dry-run makes zero submissions; dependency parsing, semicolon job IDs, failed submissions, duplicate calls tested. |
| Failure recovery | Timeouts/node failures/numerical failures retain distinct statuses; no automatic best-of retries or fake resume. |

### 13.2 Real engineering smokes

After unit tests, produce **12 PCP smokes** (six configurations × two visibility conditions) of 6,000 frames each, and **six MAPDN real-data smokes** of 512 frames each. Use development-only seeds. These amounts are engineering allocations, not proof that a method learns effectively.

Each must execute real optimization, save and strictly reload a checkpoint, and evaluate a small **development** paired bank. Verify the actor parameters change and remain finite, frame accounting is exact, the environment emits the promised observations/metrics, and source/runtime are preserved. Main test windows remain unopened.

Run on the actual chosen compute profile before marking it cluster-qualified. A local CPU smoke does not certify CUDA. Tests may be skipped only with an explicit missing-dependency/data/hardware reason; a skipped real MAPDN smoke is `pending`, not `passed`. Tiny mocked grids prove code paths, not real case33 integration.

### 13.3 Readiness is layered

Return separate booleans/statuses for:

- `implementation_complete`: required code/configs/scripts/tests are present;
- `local_checks_passed`: exact commands and outcomes attached;
- `real_pcp_smokes_passed` and `real_mapdn_smokes_passed`;
- `cluster_profile_verified`;
- `qualification_complete`;
- `main_ready`: all required evidence and budget/data/runtime freezes exist;
- `scientific_suite_executed`: false unless genuinely run with authorization.

Do not collapse these into a vague “everything works.” If access limits prevent target execution, still finish the implementation and scripts, return the blocked checks and exact next commands, and avoid claiming cluster validation.

---

## 14. Implementation milestones

**Milestone A — audit and executable plan.** Inspect the repo, preserve old artifacts, implement/extend the suite schema, produce all 90 resolved core rows, and document exact unchanged versus changed settings. Add the source/paper reference crosswalk. No Slurm submission.

**Milestone B — model and evaluator coverage.** Wire all five requested methods plus LocalCapacity on both tasks, correct the serialization path, parameterize intervention evaluation, and pass contract/statistical fixtures. Keep historical protocols intact.

**Milestone C — local and real-task checks.** Run permitted lightweight tests and smokes where resources/data exist. Explicitly record unavailable real dependencies. Do not spend the full scientific training budget merely to claim completion of implementation.

**Milestone D — Slurm package.** Implement the named scripts, actual dry-run output, row-index tests, dependency/failure handling, isolated runtime setup, and resource profiling/qualification stages. Create the user's runbook with no imaginary CLI flags.

**Milestone E — review and return.** Re-run formatting/lint/tests appropriate to the changes, inspect the diff for scope creep and historical modifications, and return the implemented file list plus the copy-paste launch commands. Do not push changes or submit the full suite unless specifically authorized.

Keep implementation compact. Reuse the existing framework, provenance, and analysis helpers; do not create a second general workflow framework with dozens of redundant receipts. The necessary complexity is the scientific contract and reliable stage boundaries, not another platform rewrite.

---

## 15. Required final response from the coding agent

The agent must return:

1. **Implementation summary:** actual files changed/created, baseline HEAD, new source identity, and any divergence from this spec.
2. **Experiment table:** all methods and conditions, seeds, budgets, core/supplement counts, qualification/smoke counts, and expected final-evaluation volume.
3. **Actual Slurm scripts:** the files under `slurm/communication_comparison_v1/`, with exact dry-run and execute commands for preparation and main training/evaluation/analysis.
4. **Validation evidence:** commands, real results, skips/blockers, tested backend, and mock versus real distinctions.
5. **Resource report:** measured profile or explicitly unmeasured values; per-task time/memory/storage estimates with assumptions; supported concurrency and walltime.
6. **Research interpretation contract:** planned effects, source citations, adaptation labels, capacity caveats, and the fact that negative outcomes are valid.
7. **Operational next action:** the exact first command the user should run, and only genuinely unresolved account/path/data values.

A response containing only YAML sketches, a generic `sbatch` example, or another high-level experimental recommendation is not complete. A script that calls an absent runner is not complete. A script whose first meaningful action is to weaken the old protocol guards is not acceptable.

**Definition of success:** the user can inspect a complete, honest 90-policy plan, qualify the actual runtime/data, and then launch every requested mechanism on PCP and MAPDN through a controlled Slurm workflow—with preserved evidence and interpretable results, whichever methods win or lose.
