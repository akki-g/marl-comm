# Communication methods and study conventions

This document defines the communication mechanisms used in the controlled
VMAS Simple Spread study. The implementations deliberately share one
BenchMARL policy shell and one MAPPO backbone. They adopt mathematical ideas
from published systems; they are not complete reproductions of those systems.

## Common policy and tensor contract

Every method occupies the same actor-only injection point:

```text
local observation -> shared local encoder -> communication -> action head
```

For hidden width `D` and `N` agents, every `CommModule` accepts and returns:

```text
h: [..., N, D] -> h_prime: [..., N, D]
```

All dimensions before `N, D` are arbitrary leading dimensions. A context mask
uses receiver-row/sender-column orientation:

```text
mask[..., i, j] = true  iff receiver i may consume sender j
```

The context mask always takes precedence over a module's fallback topology.
When no mask is available in stock Simple Spread, the default learned methods
use a fully connected directed graph without self edges. Local action ability
does not depend on the channel: the residual path retains each agent's own
hidden state even if a receiver has no incoming edge.

The critic is always BenchMARL's centralized MLP. Communication is never added
to the critic, MAPPO loss, task, or environment.

## IdentityComm

### Operation

\[
h'_i = h_i.
\]

Identity is the primary no-communication control inside `CommPolicyModel`. It
is parameterless, returns the same tensor object, preserves gradients, and
reports zero communication activity and cost. Its numerical behavior must not
be changed to distinguish it artificially from BenchMARL's parameter-matched
MLP reference.

## BroadcastComm

### Published inspiration

[Sukhbaatar, Szlam, and Fergus, “Learning Multiagent Communication with
Backpropagation,” NeurIPS 2016](https://papers.nips.cc/paper_files/paper/2016/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html)
introduced CommNet, in which continuous transmissions from other agents are
aggregated and the protocol is learned end to end.

### Project operation

For message width `M`:

\[
m_j = E(h_j),\qquad
c_i = \frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}m_j,\qquad
h'_i = h_i + D(c_i).
\]

`E` and `D` are learned message encoder/decoder projections. Empty
neighborhoods produce an exactly zero communication update. Self messages are
excluded by default.

### Difference from CommNet

CommNet interleaves communication with controller layers and was evaluated
with its own training/controller design. This project uses one explicit
communication slot inside a feed-forward MAPPO actor, adds a narrow learned
message representation, and leaves the BenchMARL algorithm unchanged. The
name `BroadcastComm` reflects that adaptation.

Primary parameters are `message_dim`, `exclude_self`, `residual`, channel
configuration, and optional externally imposed sender budget.

## GatedComm

### Published inspiration

[Singh, Jain, and Sukhbaatar, “Learning When to Communicate at Scale in
Multiagent Cooperative and Competitive Tasks,” ICLR
2019](https://openreview.net/forum?id=rye7knCqK7) introduced IC3Net and a
learned communication gate.

### Project operation

\[
m_j=E(h_j),\qquad
g_j=\sigma(G(h_j)),\qquad
\widetilde m_j=g_jm_j,
\]

followed by the same masked other-agent mean and residual decoder used by
Broadcast. The default gate is differentiable and soft. A hard
straight-through mode can use a binary forward decision while retaining the
sigmoid surrogate gradient. A learned top-K budget ranks senders by their gate
scores; random-K is an external scheduling control.

### Difference from IC3Net

This module does not reproduce IC3Net's recurrent controller, individualized
reward formulation, or original gate-learning procedure. It isolates only the
continuous sender-gating idea under the common MAPPO/shared-task experiment.
It is therefore described as IC3Net-inspired.

Primary parameters are `message_dim`, `hard`, `gate_threshold`,
`exclude_self`, `residual`, channel configuration, and sender-budget options.

## AttentionComm

### Published inspiration

[Das et al., “TarMAC: Targeted Multi-Agent Communication,” ICML
2019](https://proceedings.mlr.press/v97/das19a.html) learns sender signatures
and message values plus receiver queries, then aggregates values using
sender-receiver soft attention. [Jiang and Lu, “Learning Attentional
Communication for Multi-Agent Cooperation,” NeurIPS
2018](https://papers.nips.cc/paper/7956-learning-attentional-communication-for-multi-agent-cooperation)
is a secondary basis for selective communication.

### Project operation

For each attention head:

\[
q_i=W_qh_i,\quad k_j=W_kh_j,\quad v_j=W_vh_j,
\]

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}\,\tau},\quad
\alpha_{ij}=\operatorname{maskedSoftmax}_j(s_{ij}),\quad
c_i=\sum_j\alpha_{ij}v_j,
\]

\[
h'_i=h_i+W_oc_i.
\]

Masked probabilities are exactly zero. An all-masked row has zero aggregate
and no NaN. `key_dim` and `message_dim` denote totals across all heads, so
varying `num_heads` does not silently multiply channel capacity. Optional
rounds reuse projections to avoid a parameter-count confound. Learned sender
budgets use hard top-K selection in the forward pass and a straight-through
softmax surrogate, so query/key scheduling scores still receive gradients at
`K=1`. Random-K has no learned surrogate.

### Difference from TarMAC

TarMAC uses recurrent policies, messages exchanged across time, and supports
self-attention; its paper's multi-round update is integrated with the recurrent
state. This project performs synchronous attention on encoded current
observations inside a feed-forward MAPPO actor. The default excludes self
because the residual already supplies local information. It is a
TarMAC-inspired targeted continuous channel, not an exact TarMAC reproduction.

Primary parameters are `message_dim`, `key_dim`, `num_heads`, `temperature`,
`rounds`, `exclude_self`, `residual`, channel configuration,
and sender-budget options.

## GraphComm

### Published inspiration

[Jiang et al., “Graph Convolutional Reinforcement Learning,” ICLR
2020](https://openreview.net/forum?id=HkxdQkSYDB) models agents as graph nodes
and uses multi-head learned relation kernels over dynamic neighborhoods.
[Veličković et al., “Graph Attention Networks,” ICLR
2018](https://openreview.net/forum?id=rJXMpikCZ) introduced masked attention
over graph neighborhoods.

### Project operation

Agents are nodes, hidden representations are node features, and only edges in
`CommContext.mask` may deliver a message. Learned graph-relation attention
weights the permitted neighbors before projecting the aggregate back to `D`.
The no-context Simple Spread fallback is full directed connectivity without
self edges. Static directed-ring and seeded Erdős–Rényi fallbacks are topology
controls only; they are not claimed to model physical radio connectivity.

An all-false receiver row produces no communication delta. Explicit self-loop
behavior is respected rather than silently rewriting a supplied graph.

### Difference from DGN and GAT

DGN was instantiated with deep Q-learning and adds temporal relation
regularization; neither is used here. This module runs under unchanged MAPPO
and implements only graph-restricted learned message passing. It also preserves
the project's residual/local path and generic context interface. It is called
DGN/GAT-inspired `GraphComm`, not DGN.

Primary parameters are `message_dim`, `key_dim`, `num_heads`,
`rounds`, `exclude_self`, `residual`, fallback topology and
topology seed/probability, plus channel options.

## Channel transformations and failure semantics

[Foerster et al., “Learning to Communicate with Deep Multi-Agent Reinforcement
Learning,” NeurIPS
2016](https://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning)
motivates differentiable noisy/discretizable channels through DIAL. The common
channel layer supports identity and whole-sender dropout; Gaussian noise and
straight-through quantization may be enabled as secondary interventions.

Failure drops communication payloads only. It never drops the local hidden
state. In attention-based methods, a dropped sender is removed from the
softmax mask rather than merely assigned a zero value.

Stochastic channel failure needs special care under PPO. If failure masks are
resampled between the behavior rollout and the PPO loss recomputation, policy
ratios no longer compare the same conditional policy. A run that trains with
stochastic failure must therefore store and replay each realized sender mask.
Otherwise dropout is run and named as a post-training evaluation robustness
intervention. Channel activation mode and requested/realized rate are recorded
explicitly; ordinary module train/eval mode is not treated as an unstated
scientific definition.

`CommPolicyModel` writes each sampled dropout mask into the rollout and reuses
it during the gradient-enabled PPO policy recomputation. Fresh no-gradient
collector/evaluator forwards ignore the stale policy output carried by
TorchRL and sample a new per-environment-step mask. Multi-round modules reuse
one failure realization across their internal rounds. The configured study
uses random behavior collection and deterministic evaluation; BenchMARL's
no-gradient deterministic interaction context activates `mode: evaluation`
even though BenchMARL does not call `policy.eval()`.

Gaussian noise changes payload values rather than only sender availability, so
a sender mask cannot replay its realization. The policy model therefore rejects
Gaussian `mode: always` or `mode: training`; Gaussian intervention is
evaluation-only until payload noise itself is persisted. Quantization is
deterministic and dropout is replayed through its mask.

## Sender-budget convention

[Kim et al., “Learning to Schedule Communication in Multi-agent Reinforcement
Learning,” ICLR 2019](https://openreview.net/forum?id=HUAnBToP_a) motivates
bandwidth-constrained scheduling. With `N=3`, budgets `K=1,2,3` cap active
senders per round.

For Gated, learned scheduling ranks gate scores. For Attention, learned
scheduling derives sender importance from learned attention. `random-K` is an
externally imposed control. These adaptations are not complete SchedNet
reproductions and are labeled accordingly.

## Communication metrics and nominal cost

Metrics are detached scalar summaries. Full attention matrices and rollout
graphs are not retained by default.

Common fields:

- `message_dim`
- `message_bits_per_sender`
- `active_sender_fraction`
- `active_edge_fraction`
- `mean_message_norm` and `max_message_norm`
- `communication_rounds`
- `messages_per_step` and `active_edges_per_step`
- `realized_sender_bits_per_step` for broadcast-style sender emissions
- `realized_bits_per_step` for directed edge-delivery-equivalent payload
- matching nominal sender and edge scalar/bit fields before failure/scheduling

Attention fields:

- entropy \(-\sum_j\alpha_{ij}\log\alpha_{ij}\) over valid rows
- maximum attention probability
- effective neighbor count `exp(entropy)` (attention perplexity)

Gated fields include mean gate, fraction above the declared threshold, and
effective active senders. Graph fields include topology density and mean
incoming degree.

For float32 Broadcast/Gated payloads, the nominal per-sender payload is:

\[
32\times \text{message\_dim}\times \text{rounds}\quad\text{bits}.
\]

Attention/Graph additionally transmit sender keys/signatures under this study's
accounting convention:

\[
32\times(\text{key\_dim}+\text{message\_dim})\times\text{rounds}.
\]

Queries are receiver-local. Attention computation is compute cost, not network
communication. Parameter count, wall-clock compute, sender emission payload,
and edge-delivery-equivalent payload are reported separately.

Top-K scheduling costs are an idealized scheduled-payload convention. The
Attention/Graph scheduler evaluates all candidate relation scores before hard
selection; this study does not model or charge a separate scheduling/control
exchange. Consequently, lower selected edge-delivery cost is not evidence of
an implemented physical network protocol with free scheduling. The thesis must
report that assumption alongside sender-emission and edge-delivery metrics.

## Common-backbone adaptation

Every method in this study is trained with the same unmodified BenchMARL MAPPO
and the same centralized MLP critic, regardless of the algorithm its source
paper used. CommNet and IC3Net were introduced with their own controllers and
training procedures, TarMAC with recurrent actor-critic policies, and DGN with
deep Q-learning. Holding the RL algorithm fixed is a deliberate methodological
choice: it isolates the communication mechanism as the single manipulated
variable, at the cost of not reproducing any original system end to end.

This means a weak result here is evidence about *that mechanism under this
backbone on this task*, never evidence that the published method is weak. No
communication-specific loss term, MAPPO fork, or per-method hyperparameter
tuning is used, and the optimizer protocol below was selected from
baseline-only runs before any communication row executed.

## Frozen experimental protocol

Selected from MLP/Identity baseline diagnostics alone and confirmed by a
five-seed 600,000-frame MLP gate:

| Setting | Value |
|---|---|
| discount `gamma` | `0.9` |
| `entropy_coef` | `0.1` |
| PPO minibatch iterations | `5` |
| collected frames per batch | `6000` (10 environments) |
| minibatch size | `400` |
| learning rate | `5e-5`, GAE `lambda` `0.9` |
| training horizon | `600000` frames |
| evaluation | deterministic, 5 episodes, every `12000` frames |
| seeds | `0,1,2,3,4` main; `0,1,2` ablations |

Final performance is the mean of the final `ceil(10%)` evaluation points.
Sample efficiency is the trapezoidal return-vs-frames AUC divided by its frame
span. Group uncertainty is a fixed-RNG 95% bootstrap over seeds.

### Compute cost is measured separately

Rows execute as concurrent single-threaded workers, so wall-clock time recorded
inside a study suite is contended and reflects scheduling as much as method
cost. `configs/sweeps/simple_spread_comm_v2_wallclock.yaml` runs one short
60,000-frame row per method strictly sequentially to obtain a comparable
compute-overhead number. It carries its own `wallclock_benchmark` ablation
label and is never aggregated into the scientific comparison.

## Communication saliency

Every metric above measures communication *activity*: how many messages, how
wide, how concentrated the attention. None of them establishes that the channel
contributes to the task. A module can transmit maximally and coordinate
nothing, and on Simple Spread this is a live possibility because the
observation already exposes relative information about other agents.

Saliency measures the causal contribution by intervening on a frozen, fully
trained policy:

\[
\mathrm{saliency} = J(\pi) - J(\pi_{\varnothing}),
\]

where \(\pi_{\varnothing}\) is the same policy with every sender suppressed.
Suppression is implemented with the existing channel abstraction (`p=1`
dropout). Because each module decodes communication through a bias-free
projection, an empty neighbourhood yields an exactly zero communication delta,
so every learned module reduces to \(h'_i = h_i\) — exactly `IdentityComm`.
Learned weights are never modified; only the channel is removed. This makes the
intervention exact rather than approximate, and identical across modules.

Both arms are rolled out from the same environment seed, so the comparison is
paired and reflects the intervention rather than episode variance.

Reported fields:

- `saliency_return_delta` — return lost when the channel is severed; the
  headline task-benefit number
- `saliency_return_delta_fraction` — the same, relative to the severed return's
  magnitude, so the sign survives Simple Spread's negative-cost returns
- `saliency_action_shift_mean` — mean action displacement on identical states
- `saliency_policy_kl_mean` — KL between the pre-tanh action distributions of
  the two arms

The last two separate *behavioural influence* from *task benefit*. A large
action shift with a near-zero return delta means the module is communicating
without coordinating. That distinction follows Lowe et al., *On the Pitfalls of
Measuring Emergent Communication* (2019), which argues that message statistics
alone cannot establish that communication is used; saliency here is measured by
intervention rather than by their mutual-information estimators.

`BenchMARL MLP` and `IdentityComm` must return exactly zero saliency. That is
the correct control value and doubles as a wiring check on the intervention.

Saliency is measured **after** training from the saved final actor, never
during it. Running extra rollouts inside a training callback would consume RNG
draws and perturb the trajectory, breaking comparability with runs already
collected.

## Simple Spread interpretation limit

Stock VMAS Simple Spread is retained deliberately as an integration and
ablation environment. Its observations may already expose relative information
about other agents, so learned messages are not guaranteed to add task-relevant
information. This phase establishes correctness, learning behavior, and
controlled channel/topology responses; it does not claim that Simple Spread is
the project's definitive communication-necessity benchmark.
