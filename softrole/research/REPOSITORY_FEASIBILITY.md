# Repository feasibility: dynamic-team SoftRole-HetGAT

Audit date: 2026-09-25. Scope: local design documents, current implementation, installed locked dependencies, and the supplied Guide 3. This is a feasibility assessment, not a new learning result. No training or source-code changes were made.

## Decision

The communication and optimization infrastructure is reusable, but dynamic-team SoftRole is **a new task and actor family**, not a small replacement of a type lookup. Three changes are essential: a size-independent observation representation, a meaningful capability/team-change environment, and explicit agent-lifecycle handling. Recurrent inference is additionally needed if hidden capability changes are identifiable only through history.

The narrow first experiment should ask whether learned role-conditioned communication improves **handover after a capability change**, compared with an ordinary shared policy given the same observations and event training. Once that works, extend to agents entering/leaving and held-out team sizes. These are distinct tests; episode-level type randomization alone does not answer the user's question.

A fixed-size training critic does not logically prevent deploying an otherwise size-compatible actor on a larger team. However, the *current actor inputs and task/model specifications* are not size compatible, and a set-based critic makes mixed-size training and consistent checkpoint reconstruction much cleaner.

## What was reviewed

- Both supplied SoftRole documents, including the mathematical propositions, proposed categorical mixture actor, decentralized protocol, old implementation roadmap, and dynamic-population discussion: [architecture](../SOFTROLE_HETGAT_ARCHITECTURE.md), [critical review](../SOFTROLE_HETGAT_CRITICAL_REVIEW.md).
- [Guide 3](../../../03_Team_Changes_and_Information_Role_Recovery.pdf), all five pages, extracted using `pdftotext -layout`. Its three-agent known-failure fixture is a useful initial environment. Its instruction to avoid new architectures described that guide's previous scope; the user's present request explicitly expands the research scope.
- Current [agents.md](../../agents.md), [README](../../README.md), [cleanup audit](../../docs/AUDIT.md), [historical index](../../docs/HISTORY.md), [communication definitions](../../docs/communication_methods.md), and relevant sections of the older [overview](../../../RESEARCH_OVERVIEW.md), [assessment](../../../RESEARCH_ASSESSMENT.md), [directions](../../../RESEARCH_DIRECTIONS.md), and [roadmap](../../../RESEARCH_CODE_ROADMAP.md).
- Current `src/commstudy` model, communication, PCP environment/adapter/state, task configuration, algorithm registry, experiment construction/schema/health paths; relevant tests; and installed BenchMARL/VMAS source in `.venv`.

The old research records are historical evidence, not authoritative descriptions of the September 25 source. The cleanup index identifies the retained historical revision `pre-streamline-2026-09-24`. This audit did not reproduce historical learning outcomes or rerun the full test suite.

## Current implementation, with evidence

| Item | What exists now | Consequence |
|---|---|---|
| Repository layout | Package implementation lives in `src/commstudy`; scripts are thin entrypoints. `agents.md:3–18` and `README.md:123–139`. | Do not implement the July roadmap literally under `envs/`, `backbone/`, or `comm/`. |
| Policy | Shared MLP encoder, communication injection, shared linear output head. `models/model.py:272–361,490–530`; `experiments/config.py:317`. | Useful actor scaffold, but no learned router or recurrent history currently exists. |
| Local inputs | Strict actor allowlist; `state`, `info`, `reward`, and `episode_reward` are forbidden as actor input names. `models/model.py:46–78`; PCP adapter selects only observations at `adapters/pcp.py:64–74`. | Capability labels and current privileged state must stay out of learned actor inputs unless an announced-capability condition explicitly declares them. A performance estimator needs legitimate local outcomes, not a hidden shortcut through diagnostics. |
| Physical agents | Current PCP subclasses VMAS Simple Tag. It has three predator actors by default and one scripted prey; there is no detector/capturer split or capture-authority role. `environments/pcp.py:15–40,124–174`; `tasks/pcp.py:15–34`; `configs/pcp.yaml:23–43`. | The heterogeneous task in the July documents has not been carried into this active PCP environment. |
| Actor actions | Managed MAPPO prefers continuous actions; PCP action vectors have two components. `experiments/config.py:315–318`; installed `benchmarl/algorithms/mappo.py:115–176`. | The old five-action categorical mixture recipe is not a drop-in for this workflow. |
| Type hooks | `CommContext` has `class_id`, and optional hard type biases exist. Current experiment context contains only graph and sender masks. `communication/base.py:24–37`; `experiments/config.py:278–290`. | Hooks are available, but neither a latent posterior nor a PCP type label is supplied. |
| Existing typed behavior | Attention/Graph use a hard class-pair bias with a shared value projection. `communication/attention.py:292–305,324–344`; `communication/graph.py:366–379,398–422`. | These are not the complete original HetNet architecture or a bank of role-pair message operators. |
| Optional role embedding | The hard `nn.Embedding` is added **after** communication. `models/model.py:116–130,517–518`. | A SoftRole router that conditions messages must run before the relevant communication pass; simply enabling this option does not do that. |
| Critic | Flat concatenation of physical state in fixed world order; no current capability state. `adapters/critic_state.py:1–7,22–36`; standard MLP selected by `experiments/config.py:292–295`. | Add current capabilities and activity when the environment gains these state variables. Use set pooling for mixed-size training. |
| Graph masks | Receiver-first convention; safe all-masked softmax; sender mask removes outgoing sender columns. `communication/base.py:3–9`; `communication/utils.py:109–142,175–198`. | Good primitives; a sender failure is not a complete agent departure. |
| PPO replay | Realized channel masks are retained and reused during differentiable likelihood recomputation. `models/model.py:380–422,453–488`; `experiments/health.py:105–113,126–212`. | Preserve this behavior for event masks, recurrent context, and any stochastic router. |
| Recurrent support | Current `CommPolicyConfig` does not override recurrent state/spec behavior. Installed BenchMARL `models/common.py:315–350` returns `is_rnn=False` and empty state by default; `experiment/experiment.py:737–742` flattens nonrecurrent batches. | A GRU is a real integration change: sequence sampling, history replay, and per-agent reset semantics must be defined. |

The references in this table are relative to `marl-comm/src/commstudy/`, except where another root is stated. The source files can be opened through the [current package](../../src/commstudy/).

### Measured size obstruction

A read-only environment construction/reset probe using the locked environment produced:

| Predator count | Actor observation tensor | Privileged state | Continuous action specification |
|---:|---|---|---|
| 3 | `[1, 3, 17]` | `[1, 22]` | `[1, 3, 2]` |
| 5 | `[1, 5, 21]` | `[1, 30]` | `[1, 5, 2]` |

This used `resolve_task("vmas_predator_capture_prey", {"num_adversaries": n})` and `get_env_fun(1, True, 31, "cpu")()`, for `n in (3,5)`. It inspected reset tensors, action specifications, actor keys, and group maps, then closed each environment. It did not train a policy. The public experiment alias `pcp` is translated by the config layer; the task registry's internal name is `vmas_predator_capture_prey`.

The source explains the result. PCP inherits a vector that concatenates teammate positions in world order (`environments/pcp.py:130–174`; installed `vmas/scenarios/mpe/simple_tag.py:203–234`). Changing the population changes the vector width, so the learned first-layer matrix cannot load unchanged. `models/model.py:85–90,231–269` also binds the model's specification to a construction-time agent count.

There is a second symmetry issue: permuting rows of an already constructed observation tensor is not the same as relabeling physical agents and reconstructing every ego-centric observation. The latter permutes teammate slots *inside* each row. A generic MLP is not invariant to that internal permutation. Shared weights and an equivariant communication block do not fix an order-sensitive observation frontend.

## Corrections to the earlier SoftRole documents

1. **Paths and training ownership are stale.** Architecture §20 and critical-review §16 refer to `backbone/mappo.py`, `ppo_update.py`, `rollout_buffer.py`, `envs/asymmetry.py`, and `experiments/train_vmas_mappo.py`. Current MAPPO/QMIX are owned by BenchMARL (`algorithms/registry.py:1–47`), and configuration/building occurs in `experiments/`. Prefer supported framework interfaces and project-owned subclasses/configurations over editing installed packages.
2. **The old PCP capability model is stale.** Statements about three detectors, two capturers, `alpha`, an index-4 visibility cue, recent-detection capture windows, and a static environment `[N]` class vector describe the old proposed environment, not current PCP. Current visibility is distance based and appended at the observation tail. Add explicit heterogeneous capabilities in a new task version.
3. **The old five-action actor is stale.** Current default MAPPO uses continuous movement. One can intentionally choose a discrete task version and retain the categorical mixture, or implement a proper continuous mixture distribution. Mixing Gaussian means/scales gives one Gaussian, generally not the mixture of distributions described in the old equations. Continuous mixture entropy also lacks the simple categorical sum used there.
4. **Some previous bugs are not current findings.** The old review's reset, evaluator-mask, missing-checkpoint, and test-environment observations must not be copied as present defects. Current PCP has per-world reset state, isolated disturbances, a dedicated evaluator isolation class, replay health checks, and a later cleanup audit. Their adequacy for a *new* lifecycle model still requires testing.
5. **“Exact HetNet recovery” remains too strong.** One-hot gates recover the proposed implementation's hard typed operator bank. They do not reproduce original class-specific heterogeneous input/output spaces, binary channels, or every HetNet attention/aggregation choice. The old critical review §6.3 already gives the correct narrower statement.
6. **“Oracle upper bound” is incorrect.** Architecture §16.1 calls physical-type gating an upper bound; critical review §8.5 corrects this. A physical class need not be the best behavioral partition, and learned soft mixtures may outperform it. Use “privileged type-gated reference.”
7. **Episode-fixed hidden capability inference is not dynamic reassignment.** The original architecture's proposed experiment randomizes labels at reset only. It is a diagnostic of capability-conditioned computation, not a test of adapting to entry, exit, or capability loss during an episode.
8. **The earlier sequencing is advice tied to an old objective.** Requiring all budgeted-topology/Sinkhorn phases before dynamic populations would unnecessarily delay the present question. Dynamic capability and membership tests can precede hard topology learning and endogenous tool allocation.
9. **Old effort/compute estimates are not current benchmarks.** July person-week/GPU-hour estimates concern a different implementation and experiment grid. Do not present them as measured cost of this extension.

## What “changing teams” must mean in the implementation

Keep the following dimensions separate in the protocol and result tables:

| Condition | What changes | What a positive result supports |
|---|---|---|
| Pure permutation | Array order changes; physical state, graph, capabilities, and histories are permuted consistently. | Correct equivariance; a null engineering test. |
| Episode-level composition variation | Team size/capabilities sampled at reset, fixed for the episode. | Generalization across starting compositions. |
| Within-episode capability loss | An agent stays present but loses sensing, strength, speed, or another capability. | Diagnosis/reorganization with the same membership. |
| Within-episode departure | Agent loses physical, sensing, acting, and communication participation. | Membership adaptation; surviving-agent behavior may need to change. |
| Within-episode entry | A new actor-controlled agent becomes present with declared initial local state. | Integration of a new same-policy teammate. |
| Unfamiliar partner policy | Replacement follows a separately trained or otherwise unknown policy. | Ad hoc teamwork, a stronger and different claim. |

The user's current proposal can start with all entering agents executing the same frozen shared policy. This tests membership and capability generalization without adding unknown-policy adaptation. A larger set of known capability values is different from a new sensor modality, action schema, or semantic class.

### Activity and lifecycle contract

For an initial fixed maximum number of slots, let `active[b,i]` identify physical presence. The legal communication mask is

\[
A^{\rm live}_{b,ij}=A^0_{b,ij}\,\ell_{b,i}\,\ell_{b,j},
\]

where \(\ell\) is the active indicator and indexing is receiver first. The current sender-mask utility supplies only the \(\ell_j\) part. A departure additionally requires:

- No physical force, collision, reward contribution, sensor observation, message emission, or message receipt from that absent entity.
- No inactive-slot term in policy loss, entropy, role occupancy, auxiliary losses, or actor metrics. Merely forcing a zero action still lets a dummy slot influence learning unless its loss is masked.
- An explicit choice of whether value targets are team values or individual continuation values, including bootstrapping at departure and arrival. Do not equate an agent's exit with team episode termination.
- Recurrent state maintained for persistent entities, cleared on a genuinely new entrant, and moved consistently when slots are permuted. Returning-agent memory retention must be declared. Identity may be used for lifecycle bookkeeping without becoming an actor embedding.
- Padding neutrality: adding an inactive slot must leave every active agent's distribution unchanged. Test outputs, gradients, cost, and loss normalization.
- Current activity and capability state included in privileged critic inputs; no future failure schedule, future noise, or private event seed leaked into the actor.

Sensor loss is deliberately different: the agent may still move, transmit, relay, and contribute other skills. Mark its sensor capability/measurement validity, not its entire activity as false.

Fixed-capacity padding is the least disruptive first implementation because TensorDict and environment specifications remain rectangular. It supports real changes in the active population within a declared capacity. It does **not** alone prove arbitrary-cardinality execution. Later evaluation beyond the training slot count must reconstruct compatible specs around shared weights and use an entity encoder whose feature width is independent of slot count.

## Actor and communication changes

### 1. Represent agents and observed entities as sets

Replace concatenated teammate slots with a shared entity encoder plus masked pooling/attention. For each agent, separate ego fields from an unordered set of visible entity records:

\[
h_{t,i}=f_{\rm ego}\!\left(x_{t,i},\;\operatorname{Pool}_{e\in\mathcal E_{t,i}}\phi(x_{t,i},x_{t,e},v_{t,e})\right).
\]

Visibility/validity flags are explicit. Only physically observed records are included. Pooling perceived teammate positions is sensing, not a free delivery of hidden teammate capabilities. A common canonical sensor/action interface or suitable adapters is required for heterogeneous modalities; raw feature widths cannot simply vary by type.

Mean/softmax aggregation can lose cardinality information. Supply an execution-available neighbor/active count when the task needs counts, or use a sum/count summary. A privileged global team census must not be silently inserted into a locally observed actor. Full communication can exchange such information, but its availability and cost are part of the protocol.

### 2. Separate inferred capability from current behavioral role

An agent can retain sensing capability but change from scout to relay or responder depending on who else is present. Accordingly, useful inputs are a local history representation \(u_{t,i}\), a capability estimate \(\hat c_{t,i}\), and a team-context message \(g_{t,i}\):

\[
p_{t,i}=\operatorname{softmax}f_\phi(u_{t,i},\hat c_{t,i},g_{t,i}).
\]

The capability vector and role posterior need not coincide. If the router is purely local and two contexts produce exactly the same local history, it must emit the same posterior in both; it cannot react to an unobserved teammate departure. Provide a legitimate notification or communication path, or accept a detection delay.

Start with a declared untyped negotiation pass, then compute roles, then role-conditioned task messages. This avoids circular use of a posterior inferred from the very message that supposedly required it. A previous-step role token is another valid protocol, but it introduces explicit latency and stale-state semantics. Count all rounds and role-token payloads.

### 3. Add deterministic soft role-pair values first

With `role_probs[...,N,K]`, use the ordered relation bank

\[
v_{ij}=\sum_{r,s}p_{ir}p_{js}M_{r\leftarrow s}(h_j),
\quad
e_{ij}=e^{\rm content}_{ij}+p_i^\top Gp_j.
\]

Keep shared query/key maps initially. Apply legal/activity/channel masks before normalization and preserve exact finite no-neighbor behavior. Send the sender role token along an allowed channel; a receiver cannot read every posterior merely because they share a simulator tensor.

Deterministic gates avoid new latent sampling during PPO. Under a role-conditioned single Gaussian head, describe the actor accurately as a conditional policy. If true expert-distribution mixing is the scientific claim, the distribution and likelihood must implement it explicitly.

### 4. Introduce memory only with a well-defined hidden-change test

The announced-failure experiment can begin with the present feedforward design or one fixed history window used by all baselines. Hidden diagnosis requires history when a snapshot cannot identify the failure. For example,

\[
u_{t,i}=\operatorname{GRU}(u_{t-1,i},o_{t,i},a_{t-1,i},y^{\rm local}_{t,i},m^{\rm received}_{t-1,i}).
\]

Here \(y^{\rm local}\) must be an outcome observable by the physical agent, such as achieved displacement or sensor validity. The current code's actor-information contract excludes global reward/diagnostic shortcuts. If success/failure feedback is newly exposed, document it and give it to every baseline.

A hidden fault that induces no change in any available observation-history distribution is unidentifiable. With equal-prior hypotheses that induce identical histories, every detector has average error at least one half. No role architecture can remove this information obstruction. Design diagnostic evidence before claiming that a GRU can infer lost capabilities.

## Concrete extension map

All proposed package additions remain under `src/commstudy/`; the names below are suggestions, not existing files.

| Purpose | Current integration point | Suggested bounded addition/change |
|---|---|---|
| Versioned recovery environment | `environments/pcp.py`, `tasks/registry.py`, `adapters/pcp.py` | New `environments/role_recovery.py`, typed `tasks/role_recovery.py`, and adapter; preserve current PCP as its own task. |
| Event state and paired randomness | `adapters/randomness.py`, `environments/pcp.py:63–79,96–122` | Seeded episode event manifest with separate streams for requests, capability changes, joins/leaves; partial-reset correctness. |
| Entity observations | `models/model.py:85–109,490–502` | `models/entity_encoder.py`, with ego/entity masks and constant feature width. |
| Role/capability inference | `models/model.py:490–518`, `models/config.py` | `models/softrole.py`; shared deterministic router, optional history state, declared gate overrides. |
| Typed messages | `communication/base.py`, `attention.py`, `graph.py`, `utils.py` | `communication/softrole.py`; explicit `role_probs` and active mask, receiver-first role-pair bank, metadata accounting. |
| Model configuration | `experiments/schema.py:124`, `experiments/build.py:63–110` | Register the new actor family and context fields; preserve current public five-method configs and defaults. |
| Variable-size critic | `adapters/critic_state.py`, `experiments/build.py` | `models/set_critic.py` plus an entity-state adapter; activity/capability-aware permutation-invariant team value. |
| Policy/loss lifecycle | `algorithms/registry.py`, installed BenchMARL MAPPO interfaces | Project-owned supported integration for inactive loss masks; custom distribution only if continuous expert mixtures are used. Do not patch `.venv`. |
| Recurrent rollout state | `models/config.py`, BenchMARL model-state specs and RNN collector support | Add `is_rnn`, state specs, sequence sampling/burn-in, entrant reset rules, and replay tests when introducing GRU inference. |
| Behavioral diagnostics | `tasks/pcp_metrics.py`, `experiments/metrics.py`, `experiments/evaluation.py` | New task-owned event/recovery metrics and a paired evaluation manifest, logged outside actor inputs. |
| Provenance and health | `experiments/provenance.py`, `health.py`, `bookkeeping.py` | Record protocol versions, observation schema, activity/capability/event settings, recurrent reset convention, payload and round cost. |
| Tests | `tests/test_information_contracts.py`, `test_comm_policy_model.py`, `test_comm_context.py`, `test_training_health.py` | Add focused task, lifecycle, full-pipeline equivariance, padding-neutrality, no-leak, and replay tests for changed behavior. |

The current runner fixes a five-method study, and LocalCapacity was retired by the cleanup (`docs/AUDIT.md:22–24`). A future experiment needs an explicit configuration extension for the SoftRole and capacity-matched baseline family. Restore a simple local-capacity baseline through ordinary model/config support if needed; do not revive the retired study supervisors or approval workflows.

## Staged experiment that addresses the new question

### Stage A — establish a solvable recovery task

Use Guide 3's two potential scouts and one responder. Fresh requests identify one of two service sites; the responder needs communicated information. Both scouts can sense, but only one initially obtains the cue. At a declared event, the active scout loses its sensor while retaining movement and radio. The backup remains capable.

First test controlled handover: the backup receives the next cue without additional travel, with the number of informative sources and packet opportunities held constant. Then test physical recovery: the backup must move to sense. A scripted surviving-capability controller must succeed, establishing feasibility rather than an optimal upper bound. Keep request generation fresh after failure so remembered old information cannot solve later rounds.

Do not add anonymous ID embeddings to manufacture a failure. Physical source transfer must differ from a pure relabeling. All methods receive the same announced-failure signals in the first stage.

### Stage B — distinguish data coverage from architecture

Compare ordinary shared Attention (and a widened version), a uniform-gate model with the same role/expert bank, and learned SoftRole. A privileged type/capability-conditioned reference helps diagnose task/headroom issues. Use the same history, negotiation rounds, payload width plus metadata, optimizer, training steps, and event exposure.

Cross or stage training distributions explicitly:

1. Fixed starting source, no within-episode failures.
2. Random starting source, no within-episode failures.
3. Random starting source with within-episode capability changes.

Condition 2 evaluated on a mid-episode handover tests temporal-transition generalization; condition 3 tests learned recovery under familiar event classes. A SoftRole model trained with failures cannot be fairly credited for outperforming a baseline trained without them. Packet dropout is a separate exposure factor, not a substitute for capability loss.

Start with a small development cohort, then choose confirmatory seed allocation from pilot variability and a practical effect threshold. Do not inherit a fixed July run count or a claimed compute estimate. One primary comparison with disjoint development/test seeds is more useful than an immediate all-baseline grid.

### Stage C — test actual membership changes

At fixed maximum capacity, remove a scout or responder while a physically suitable survivor/entrant remains. Then add an agent during an episode. Train on a limited set of counts and compositions and hold out specified larger counts, entry times, and capability combinations. Freeze the checkpoint throughout evaluation.

Report size/composition extrapolation separately from in-distribution event adaptation. A policy trained at several counts and tested at an intermediate count is a different claim from training only on small counts and testing above the maximum. State whether the test includes more agents, more targets, different density, and a changed communication degree; these alter the task in different ways.

For slightly differing capabilities, vary interpretable continuous axes such as sensing radius, sensing reliability, speed, or actuation strength. Distinguish interpolation within the training range, extrapolation beyond it, and a new modality/action schema. Permutation symmetry guarantees none of these performance transfers.

### Stage D — hidden capability loss and inference

Remove the explicit fault flag while preserving observable local diagnostic evidence. Compare the same recurrent baseline family against recurrent SoftRole. Evaluate announced versus hidden versions to separate organization difficulty from diagnosis difficulty. Hold out change times and capability severities rather than immediately adding arbitrary new roles.

Use frozen-router, frozen-capability-belief, and history-reset interventions, together with role/message shuffles, to determine whether online inference contributes. A role switch alone is not evidence of useful recovery.

## Measurements and decision rules

The main outcome should be task completion after the event, e.g. post-failure requests served before deadline, with intact performance reported alongside it. Define a recovery penalty for each trained cohort:

\[
\mathcal L_{\rm rec}=Y_{\rm intact}-Y_{\rm event}.
\]

The primary architecture contrast can be the difference in these penalties between SoftRole and a matched shared-attention baseline, while retaining both raw outcomes. This distinguishes a smaller failure penalty from merely higher general competence.

Log three times separately: first useful backup observation, first useful delivered information, and first successful post-event action. Also record request-aligned success curves, duplicate travel, inactive-slot violations, communication cost including role tokens/rounds, and recovery latency with nonrecovering episodes retained as failures/censored observations. An average over successful recoveries alone is biased toward easy cases.

Capability classification accuracy is meaningful only when the evaluation defines a true physical capability label. Behavioral roles have no mandatory one-to-one ground truth. Fit any latent-label alignment on validation episodes, then fix it for test. Population entropy by itself does not establish differentiated agents; report individual entropy and the difference between population and mean individual entropy, as already noted in critical review §8.3.

The mechanistic question is whether the model reallocates a needed function when another agent changes. Test a surviving agent in matched contexts with the same capability and controlled local state, but different team composition, using legal context messages. Include same-checkpoint gate freezes/shuffles, message shuffles, and directional edge/expert interventions. Interpret them as interventions on the learned controller; out-of-distribution disruption is still a possible explanation and should be checked with complementary controls.

Stop escalating architecture complexity if a matched ordinary shared policy already recovers well. That is an informative result: the tested setting supports generalization through shared policy/entity structure without extra latent roles. If role posteriors track capability but add no task benefit, report capability inference rather than role-learning advantage. If SoftRole helps only with known failure flags, limit the claim to reorganization after known changes.

## Validation needed before interpreting learning results

- Full-pipeline relabeling test: physical entities, entity features, observation sets, graphs, activity, capabilities, and histories all transform consistently; action distributions and roles permute accordingly.
- Padding-neutrality test and both-sided departure masks; entrants cannot inherit another entity's history or cached message.
- Random partial reset test across vectorized worlds; events/capabilities reset only in the selected world.
- Hidden-state nonleak tests, including absence of future event schedules from actor inputs and role messages.
- Correct probability distribution and PPO behavior replay under the same historical observations, role temperature, event/channel masks, and recurrent state/sequence. Preserve current numerical tolerance logic instead of imposing an unsupported universal `1e-6` threshold.
- Active-only actor loss/entropy/role statistics and declared value/GAE semantics at exits and entries.
- One-hot hard typed reduction, all-masked rows, receiver/sender orientation, forbidden-edge isolation, nonzero router/expert gradients, exact checkpoint reconstruction.
- A tiny end-to-end training smoke establishing finite updates and event correctness, followed by learning experiments with independent seeds. A successful smoke is not evidence that communication or roles improve performance.

When code work begins, current `agents.md` requests `uv run --locked pytest` and `uv run --locked ruff check src scripts tests`. No such full-suite result is claimed by this documentation-only audit.

## Feasibility conclusion

Feasibility is high for deterministic soft routing at fixed team capacity, moderate for a carefully specified membership-changing task and entity-based policy, and materially harder for hidden capability diagnosis with recurrent PPO. The strongest near-term contribution is the controlled link between inferred capability, team context, role-conditioned communication, and measured recovery after changes. Neither an invariant aggregator nor an interpretable latent label is sufficient evidence of this link.

The shortest useful path is: **versioned handover task → size-independent entity inputs → matched ordinary-policy/SoftRole comparison → within-episode membership events → hidden-change inference**. Hard communication-budget optimization, Sinkhorn quotas, learned role counts, new action modalities, and unfamiliar-policy teammates can remain separate extensions.
