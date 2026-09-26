# Critical Review of SoftRole-HetGAT

*Literature audit, mathematical novelty assessment, revised research
formulation, and dependency-ordered implementation roadmap.*

*Review date: 2026-07-23. Literature cutoff: 2026-07-23.*

> This is a standalone review of
> [SOFTROLE_HETGAT_ARCHITECTURE.md](SOFTROLE_HETGAT_ARCHITECTURE.md). It does
> not replace [THESIS_DIRECTION_GUIDE.md](THESIS_DIRECTION_GUIDE.md). The
> matched causal communication audit in that guide remains the safest primary
> thesis. This document asks whether the proposed learned-role extension is
> feasible, what is already known, what its equations actually contribute, and
> what would have to change for a stronger algorithmic contribution.
>
> Novelty is a claim about the complete prior-art record, not about how unusual
> an equation looks. This review uses primary papers and publisher records where
> possible, including work available through July 2026. It can identify a
> defensible candidate contribution; it cannot guarantee that no uncatalogued,
> concurrent, or differently named method overlaps it. Any eventual “first” or
> “to our knowledge” claim still requires a formal search protocol and advisor
> review immediately before submission.

## 1. Executive verdict

### 1.1 Direct answer

The proposed system is technically possible:

\[
\text{infer a role mixture}
\longrightarrow
\text{use it to route communication experts}
\longrightarrow
\text{use it to mix action experts}.
\]

It is also a sensible continuation of the thesis question. Once the current
work establishes that communication matters when sensing and action authority
are split, the natural next question is:

> Can a team infer or allocate the functional structure that determines who
> should process which information, who should communicate with whom, and
> which behavior each agent should execute?

The current SoftRole-HetGAT formulation is coherent and implementable, but it
is **not broadly novel as presently framed**. Nearly every primitive has a
strong precedent:

- learned and dynamic roles in MARL;
- soft role distributions and role-specific action/Q experts;
- learned grouping and role-aware communication;
- intra-role and inter-role communication;
- directed class-pair message transformations;
- mixed-membership bilinear edge scores;
- graph attention;
- mixture-of-experts gating;
- load balancing and temporal role regularization.

Two recent papers are particularly close. [ROCO](https://doi.org/10.1016/j.eswa.2025.129421)
already combines learned action-oriented roles, role-specific decision modules,
attention-based inter-role communication, and mutual-information-guided
intra-role communication. [GRDC](https://ojs.aaai.org/index.php/AAAI/article/view/40185)
jointly learns graph-derived roles and role-restricted communication.
[SCoUT](https://arxiv.org/abs/2603.04833) uses soft temporal groups as a
recipient-selection prior and reuses the same assignments in its critic.

The narrow distinction that remains plausible is:

> One deterministic soft role posterior is used simultaneously as a true
> probability mixture over action policies and, through
> \(p_i\otimes p_j\), as the interpolation weight over a complete ordered bank
> of receiver-role-by-sender-role message operators, while cross-role
> communication is retained.

That is a specific synthesis, not an entirely new family of ideas. It may be
enough for a strong undergraduate thesis extension if evaluated rigorously. It
is not yet a safe publication-level novelty claim.

### 1.2 Recommended decision

There are two legitimate routes, and they should not be blurred together.

| Route | Scientific claim | Novelty level | Added effort | Recommendation |
|---|---|---:|---:|---|
| **A. Characterized SoftRole-HetGAT** | A shared soft latent variable couples typed communication and action specialization; its use is tested causally | Modest architectural synthesis; stronger mathematical characterization and experimental design | About 5.5–8.25 person-weeks after the causal audit | Feasible appendix or follow-on study |
| **B. Joint role allocation and budgeted topology** | The team jointly allocates scarce functional roles and a directed communication graph under explicit capacity constraints as task demand and team size change | More defensible candidate algorithmic novelty, but every ingredient still has precedents | About 12–18 additional person-weeks, plus substantial compute | Separate paper-scale project |

For the current thesis:

1. finish the matched causal communication audit first;
2. do not make thesis completion depend on SoftRole;
3. if time remains, implement Route A through a three-seed decision gate;
4. pursue Route B only if there is enough time to add a task that actually
   requires collective assignment rather than trivial capability decoding.

### 1.3 What not to do

Do not add more notation, losses, or graph layers merely to make the method
look mathematically novel. That increases complexity without moving the
prior-art boundary. In particular:

- \(p_i^\top Gp_j\) is not a new topology model;
- \(p_i p_j^\top\) is not a new joint-role posterior;
- a soft role gate is not automatic group formation;
- attention weights on a dense graph are not learned sparse connectivity;
- floating-point vector width is not a Shannon bit rate;
- a role label cannot change physical sensing or action authority unless the
  environment makes that assignment an explicit action;
- a regular graph alone cannot make identical anonymous agents choose distinct
  deterministic roles.

## 2. Clarifying the research object

The original question combines four different objects. They need separate
names because each creates a different research problem.

| Object | Meaning | Who controls it? | Present PCP status |
|---|---|---|---|
| **Physical capability** | What an agent can sense or physically execute | Environment, unless explicitly made an allocation action | Fixed detector/capturer capability |
| **Behavioral or computational role** | Which expert or strategy an agent uses | Learned policy | Proposed soft role posterior |
| **Group membership** | Which agents form a team or subteam | Independent local gates or a joint allocator | Independent gates do not enforce a partition |
| **Communication topology** | Which directed links carry payloads | Physical candidate network plus a learned scheduler | Current model only weights permitted dense edges |

A model can infer that an agent is detector-like without giving that agent a
sensor. It can route messages as if an agent is a capturer without giving that
agent capture authority. Conversely, if the environment already reveals the
capability through a local observation bit, the model is classifying an
existing type rather than inventing a role.

The thesis should use the following language precisely:

- **role inference** when the role already exists in the environment and the
  policy estimates it;
- **behavioral specialization** when latent modes affect only computation;
- **group formation** only when assignments are coupled across agents;
- **capability allocation** only when a policy action changes sensors, tools,
  or admissible actions;
- **topology learning** only when links are selected or removed before payload
  transmission, not merely assigned attention after every message is already
  available.

### 2.1 Why a regular graph cannot create distinct roles by itself

Let a deterministic graph policy \(F\) be permutation equivariant:

\[
F(QX,QAQ^\top)=QF(X,A)
\]

for every agent-permutation matrix \(Q\). Suppose \(Q\) is an automorphism of
the graph and the node features are also unchanged:

\[
QX=X,\qquad QAQ^\top=A.
\]

Then:

\[
F(X,A)=QF(X,A).
\]

All nodes in the same automorphism orbit must receive the same output. On an
anonymous regular graph with identical histories, a shared deterministic
router therefore cannot elect one agent as a detector, another as a capturer,
and so on. This is not a GNN-specific accident; it is part of the older
symmetry problem in distributed computing, exemplified by
[Angluin's work on anonymous processor networks](https://doi.org/10.1145/800141.804655).

At least one legitimate symmetry breaker is required:

- different observations, positions, histories, or capabilities;
- private random seeds or stochastic role samples;
- persistent agent identifiers, if identity dependence is scientifically
  intended;
- a task signal that distinguishes role suitability;
- or a coupled assignment protocol with randomness or an external tie-break.

Even a balanced Sinkhorn allocator does not solve exact symmetry. If every
agent has the same score vector and the entropy-regularized solution is unique,
every row receives the same fractional role mixture. Hardening that mixture
still needs a tie-break.

## 3. The proposed architecture, stripped to its essential mathematics

Let \(h_{t,i}\) be agent \(i\)'s locally available representation and let
\(K\) be the number of latent roles. The router outputs:

\[
p_{t,i}
=
\operatorname{softmax}\!\left(
\frac{f_\phi(h_{t,i})}{\tau}
\right)
\in\Delta^{K-1}.
\]

The directed role-pair attention perturbation is:

\[
\beta_{t,ij}
=
p_{t,i}^{\top}G p_{t,j}.
\]

The role-conditioned message is:

\[
v_{t,ij}
=
\sum_{r=1}^{K}\sum_{s=1}^{K}
p_{t,i,r}p_{t,j,s}
M_{r\leftarrow s}(h_{t,j}).
\]

After masked attention and aggregation produce \(y_{t,i}\), the action policy
is a true mixture:

\[
\pi(a_{t,i}\mid y_{t,i},p_{t,i})
=
\sum_{r=1}^{K}
p_{t,i,r}\pi_r(a_{t,i}\mid y_{t,i}).
\]

The attractive property is the shared gate: \(p_{t,i}\) affects both the
information-processing path and the action path. If \(p_{t,i}\) is one-hot,
the model reduces to a hard typed policy such as HetNet. If it is uniform, the
model becomes an equal mixture over every role and relation expert.

That is a clean architecture. The novelty question is not whether the
equations are coherent; it is which parts have already appeared and what new
scientific proposition their combination tests.

## 4. Historical and cross-disciplinary foundation

### 4.1 Decentralized decisions came before deep MARL

The core difficulty is older than modern reinforcement learning. In
decentralized control, each decision maker acts from a different information
set. [Witsenhausen's counterexample](https://doi.org/10.1137/0306011) showed
that seemingly simple decentralized linear-quadratic-Gaussian problems can
require nonlinear strategies; an action can simultaneously control the system
and signal information to a later decision maker.
[Ho and Chu](https://doi.org/10.1109/TAC.1972.1099850) formalized how
information structures affect team decision problems.

This matters here for two reasons:

1. communication is part of an information structure, not simply another
   neural layer;
2. an environment action can itself communicate implicitly, so “no explicit
   messages” does not mean “no signaling.”

The current PCP design is valuable because it makes the relevant information
path explicit and experimentally interruptible. The role extension should
retain that causal clarity.

### 4.2 Early MARL and Dec-POMDPs

[Tan (1993)](https://www.cs.utexas.edu/~shivaram/readings/b2hd-Tan1993.html)
compared independent and cooperative learning agents, including experience and
sensation sharing. [Littman (1994)](https://www.eecs.harvard.edu/cs286r/courses/spring06/papers/littman94markov.pdf)
used Markov games as a multi-agent learning framework. The cooperative,
partially observed problem is naturally represented as a Dec-POMDP.
[Bernstein et al. (2002)](https://pubsonline.informs.org/doi/abs/10.1287/moor.27.4.819.297)
showed that finite-horizon decentralized control is NEXP-hard even with two
agents.

Deep CTDE methods make the problem tractable in practice without removing its
information constraints. [MADDPG](https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)
made centralized critics a standard deep-MARL pattern;
[QMIX](https://proceedings.mlr.press/v80/rashid18a.html) factorized a joint
value monotonically; and
[MAPPO](https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)
showed that a carefully implemented PPO baseline is highly competitive.

The conceptual lesson is that a centralized training tensor is not
automatically a legal execution-time signal. Every role posterior, neighbor
key, edge choice, and payload used by a decentralized actor needs an explicit
local source or communication protocol.

The tuple used in the architecture note should also be described as a
simplified Dec-POMDP specification rather than a complete formal definition:
it suppresses at least the initial-state distribution and horizon or
termination convention, and writes each observation as a deterministic
\(O_i(s)\). None of those omissions invalidates the implementation, but the
thesis formulation should distinguish a deterministic observation function
from the more general observation kernel
\(O(o_{1:N}\mid s,a_{1:N})\).

### 4.3 Learned communication: from broadcast to selective structure

The modern learned-communication line already covers most axes needed by the
proposal:

| Development | Representative primary work | Relevance |
|---|---|---|
| Continuous differentiable broadcast | [CommNet, 2016](https://arxiv.org/abs/1605.07736) | Messages and policy trained end to end |
| Discrete or differentiable protocols | [RIAL/DIAL, 2016](https://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning) | Centralized gradient path with decentralized execution |
| Learned decision to communicate | [IC3Net, 2019](https://iclr.cc/virtual/2019/poster/770) | Gating who speaks |
| Learned recipient attention | [TarMAC, 2019](https://proceedings.mlr.press/v97/das19a.html) | What to send and whom to address |
| Explicit bandwidth scheduling | [SchedNet, 2019](https://iclr.cc/virtual/2019/poster/931) | Restricted number of transmitters |
| Adaptive hierarchical grouping | [Learning Structured Communication, 2020](https://openreview.net/forum?id=BklWt24tvH) | Learned groups plus inter-/intra-group message passing |
| Content and connection compression | [IMAC, 2020](https://proceedings.mlr.press/v119/wang20i.html) | Information bottleneck and a scheduler under limited bandwidth |
| Hard and soft attention | [G2ANet](https://arxiv.org/abs/1911.10715) | Select agents, then weight selected messages |
| Graph message passing | [DGN, 2020](https://openreview.net/forum?id=S8icDSeqfvy) | Multi-round graph convolution in MARL |
| Low-degree symbolic protocols | [Neurosymbolic Transformers, 2020](https://proceedings.neurips.cc/paper/2020/hash/9d740bd0f36aaa312c8d504e28c42163-Abstract.html) | Explicitly sparse, interpretable graph communication |
| Expressivity of neural protocols | [Universally Expressive Communication, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d8a19c815a8bef25e6094e87f963d28e-Abstract-Conference.html) | Relates common protocols to MPNNs and formalizes symmetry-breaking needs |

SoftRole-HetGAT therefore should not claim to invent targeting, grouping,
communication scheduling, attention, or differentiable protocols.

### 4.4 Roles, groups, and heterogeneous policies in MARL

The role literature has also moved well beyond fixed labels:

- [ROMA](https://proceedings.mlr.press/v119/wang20f.html) learns stochastic,
  dynamic, identifiable role embeddings and role-conditioned policies.
- [RODE](https://iclr.cc/virtual/2021/poster/2717) decomposes action spaces and
  assigns agents to learned roles.
- [LDSA](https://papers.nips.cc/paper_files/paper/2022/hash/0b4145b562cc22fb7fa50a2cd17c191d-Abstract-Conference.html)
  learns subtask assignments and subtask policies.
- [SORA](https://doi.org/10.1007/978-981-99-8079-6_25) already uses a soft role
  distribution with role-specific Q-networks.
- [role transfer across team sizes](https://arxiv.org/abs/2204.12937) directly
  studies learned assignment and transfer to larger teams.
- [HetNet](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf) uses
  supplied classes, class-wise policies, and ordered inter-class message
  transformations.
- [ROCO](https://www.sciencedirect.com/science/article/pii/S0957417425030374)
  combines action-oriented roles, role-specific decision modules, inter-role
  attention, and intra-role message filtering.
- [GRDC](https://ojs.aaai.org/index.php/AAAI/article/download/40185/44146)
  builds interaction graphs, infers prototype roles, and restricts attention
  communication to the inferred role.
- [SCoUT](https://arxiv.org/abs/2603.04833) repeatedly infers soft groups,
  biases recipient selection with group affinity, reuses assignments in the
  critic, and derives counterfactual communication advantages.
- [GIRA](https://doi.org/10.1016/j.neucom.2026.134437), available online in
  July 2026, infers roles from group-interaction signals and targets adaptation
  to changing populations and unseen team sizes.
- [g-MAT](https://doi.org/10.1016/j.knosys.2026.116612), also available online
  in July 2026, learns role-labelled groups from an action-effect graph and
  combines local within-group with compressed cross-group attention.

The rapid 2025–2026 convergence of role learning, grouping, communication, and
population adaptation makes broad novelty claims especially unsafe.

### 4.5 Graph learning and mixed-membership networks

The graph equations have direct non-MARL ancestors:

- [MPNNs](https://proceedings.mlr.press/v70/gilmer17a.html) provide the generic
  learned message-and-aggregate framework.
- [GAT](https://openreview.net/pdf?id=rJXMpikCZ) supplies masked neighbor
  attention.
- [R-GCN](https://doi.org/10.1007/978-3-319-93417-4_38) uses
  relation-specific message matrices.
- [HGT](https://arxiv.org/abs/2003.01332) uses node- and edge-type-dependent
  attention and transformations.
- [Neural Relational Inference](https://proceedings.mlr.press/v80/kipf18a.html)
  infers latent directed interaction types and uses a GNN decoder.

Most importantly,
[mixed-membership stochastic blockmodels](https://jmlr.org/beta/papers/v9/airoldi08a.html)
give each node a simplex-valued membership vector and combine endpoint roles
through a compatibility matrix. The expected pair relation has the same core
form as:

\[
p_i^\top Gp_j.
\]

[Dynamic MMSB](https://icml.cc/Conferences/2009/papers/478.pdf) already tracks
time-varying mixed roles and a role-compatibility matrix. Bilinear relation
models such as [RESCAL](https://icml.cc/2011/papers/438_icmlpaper.pdf) likewise
score directed relations with an asymmetric matrix.

Thus \(PGP^\top\) should be described as a learned mixed-membership block
operator, not as a new kind of graph.

### 4.6 Mixture-of-experts and assignment

The action policy:

\[
\pi(a\mid x)=\sum_r p_r(x)\pi_r(a\mid x)
\]

is classical mixture-of-experts mathematics. The gating and posterior
responsibility equations go back at least to
[adaptive mixtures of local experts](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf).
Modern sparse MoE work makes expert collapse, load balancing, and sparse
routing familiar concerns; see
[Shazeer et al.](https://research.google/pubs/outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer/)
and the [Switch Transformer](https://www.jmlr.org/beta/papers/v23/21-0998.html).

Joint balanced assignment also has established machinery:

- [Sinkhorn and Knopp](https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf)
  formalized iterative matrix scaling;
- [Gumbel-Sinkhorn](https://openreview.net/pdf?id=Byt3oJ-0W) provides a
  differentiable matching relaxation;
- [SwAV](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)
  uses balanced online cluster assignments;
- [dynamic multi-agent assignment via optimal transport](https://arxiv.org/abs/1910.10748)
  couples task assignment and agent dynamics;
- [state-augmented multi-agent assignment RL](https://proceedings.mlr.press/v242/agorio24a.html)
  uses distributed dual-variable communication and gives feasibility
  guarantees.

Using Sinkhorn would be a useful design change. Sinkhorn itself would not be
the novelty.

### 4.7 Topology, networking, and information

Several foundational distinctions prevent overclaiming.

**Connectivity.** [Fiedler's algebraic connectivity](https://dml.cz/handle/10338.dmlcz/101168)
uses the second-smallest Laplacian eigenvalue of an undirected graph.
[Olfati-Saber and Murray](https://www.cds.caltech.edu/~murray/papers/2003c_om03-cdc.html)
connect graph structure and spectral quantities to consensus under specific
linear protocols and switching topologies. These results do not automatically
apply to a directed nonlinear attention layer. A graph can be useful for
task-directed information flow without being optimal for consensus.

**Capacity.** [Shannon's communication theory](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x)
concerns information transmission under a specified channel.
[Gupta and Kumar](https://web.stanford.edu/class/cs244/papers/WirelessCapacity.pdf)
show that throughput in wireless networks scales nontrivially with population
and interference. Dense all-to-all floating-point exchange silently assumes
away those constraints.

**Coding and routing.** [Network information flow](https://doi.org/10.1109/18.850663)
and linear network coding concern flows, edge capacities, intermediate coding,
and destination decoding. A neural linear transform of a neighbor feature is
not automatically a network code.

**Message-passing bottlenecks.** Sparse multi-hop GNNs may over-squash
long-range information; see
[Alon and Yahav](https://openreview.net/references/pdf?id=Jac2LI7GYF) and
[Topping et al.](https://openreview.net/pdf?id=7UmjRGzp-A). This is a relevant
future scaling concern, but it is not a convincing motivation for the current
five-agent, nearly complete graph.

The practical conclusion is that the project must distinguish:

\[
\text{physical availability}
\neq
\text{edge scheduling}
\neq
\text{attention after receipt}
\neq
\text{message compression}
\neq
\text{information rate}.
\]

## 5. Closest-work comparison

The following table is the decisive novelty audit.

| Method | Learned roles/groups | Action specialization | Communication structure | Soft memberships | Cross-role flow | Variable team size | Collision with proposal |
|---|---|---|---|---|---|---|---|
| HetNet (2022) | No; classes supplied | Class-specific policies | Ordered class-pair transformations and heterogeneous attention | No | Yes | Team composition varies, class set fixed | Already contains the hard \(K^2\) typed endpoint operators |
| ROMA (2020) | Dynamic stochastic roles | Role-conditioned policies | Not the central contribution | Role embeddings | N/A | Fixed in main setting | Learned roles and role-conditioned behavior are established |
| SORA (2023) | Soft role assignment | Role-specific Q networks | No role-pair routing shown | Yes | N/A | Not central | Soft action-expert gating is established |
| Learning Structured Communication (2020) | Adaptive groups | Shared downstream policy | Hierarchical inter- and intra-group GNN | Group assignment | Yes | Designed for scale | Learned groups and both communication levels are established |
| ROCO (2025/2026) | Dynamic action-oriented hard roles | Role-specific Q/action modules | Attention across roles; MI filtering within roles | Hard periodic selection | Yes | Evaluated on larger teams | Broad combination of roles, action modules, and inter/intra-role communication already exists |
| GRDC (2026) | Prototype roles from local interaction graphs | Role-conditioned coordination | Attention restricted to same-role agents | Gumbel/discrete | No | Prototype pruning | Role discovery and communication are already unified |
| SCoUT (2026 preprint) | Soft temporal groups | Ordinary environment-action head | Send decision and recipient selection biased by group affinity | Yes | Yes, via recipient policy | Explicit scaling target | Same soft assignment already routes communication and another learner component |
| GIRA (July 2026) | Roles from group interaction signals | Role-specific policy differentiation | Interaction signals inform roles | Learned representation | Not a \(K^2\) bank | Dynamic and unseen populations | Undercuts group-informed roles and population-adaptation claims |
| g-MAT (July 2026) | Action-effect graph clustering | Grouped autoregressive policy | Local within-group and compressed global cross-group attention | Primarily clustered groups | Yes | Efficiency target | Learned groups determine hybrid communication topology |
| SoftRole-HetGAT | Local or negotiated posterior | True mixture over role policies | Full directed \(K^2\) role-pair message bank and attention bias | Yes | Yes | Actor can be size-equivariant | Remaining distinction is continuous shared routing across both complete operator banks |

The proposal's current Section 27 should therefore not call GRDC the sole
closest bridge between roles and communication. ROCO is the most direct
collision, while SCoUT is the closest collision on reusing one soft assignment.

One still earlier historical warning is
[Stone and Veloso (1999)](https://www.cs.cmu.edu/~coral/old/publications/b2hd-AIJ99.html),
which joined task decomposition, dynamic role assignment, and low-bandwidth
communication for real-time teamwork before deep MARL. The neural
implementation is new; the organizational question is not.

## 6. Equation-by-equation novelty audit

| Proposed object | Correct mathematical interpretation | Closest established family | Novel by itself? |
|---|---|---|---:|
| \(p_i=\operatorname{softmax}(u_i/\tau)\) | Simplex-valued local gate | Latent-role MARL and MoE | No |
| \(p_i^\top E\) | Convex mixture of role embeddings before downstream nonlinearities | Soft embeddings and conditional computation | No |
| \(p_ip_j^\top\) | Independent or mean-field coupling of endpoint memberships | Mixed-membership blockmodels | No |
| \(p_i^\top Gp_j\) | Expected bilinear role compatibility under that coupling | MMSB, RESCAL, relational embeddings | No |
| \(\sum_{r,s}p_{ir}p_{js}M_{r\leftarrow s}(h_j)\) | Multilinear extension of a hard typed-relation operator bank | R-GCN, HGT, HetNet, Meneghetti–Bianchi | No |
| Masked softmax attention | Content-dependent normalization over permitted senders | GAT, TarMAC, DGN | No |
| \(\sum_rp_{ir}\pi_r(a)\) | Probability-space mixture of categorical experts | Classical MoE and SORA | No |
| Gate entropy and population load terms | Sharpness and expert-utilization regularization | Clustering, latent roles, sparse MoE | No |
| \(D_{\mathrm{KL}}(p_{t-1}\|p_t)\) | Temporal smoothing of a latent assignment | Dynamic latent-variable models and role MARL | No |
| One posterior routes both paths | Coupled conditional computation | Close to ROCO/SCoUT, but exact operator use differs | Possibly as a narrow synthesis |
| Same-checkpoint gate/message/edge/expert interventions | Causal test of whether the learned routing variable is functionally used | Related ablation and counterfactual communication work | Strong experimental contribution |

### 6.1 The outer product is not a learned joint posterior

The gate:

\[
\Gamma_{ij,rs}=p_{i,r}p_{j,s}
\]

assumes the endpoint roles factorize at the point where an edge operator is
selected:

\[
\Gamma_{ij}=p_i\otimes p_j.
\]

It is a valid coupling with the desired marginals, and it is the
maximum-entropy coupling when no endpoint dependence is modeled. It is not a
separately inferred:

\[
q(r_i,r_j\mid h_i,h_j).
\]

The agents' marginal vectors may still be statistically dependent because
their inputs share an environment or negotiation history. The edge gate itself,
however, contains no residual interaction beyond the product. Calling it an
independent endpoint coupling or mean-field routing gate is accurate.

### 6.2 The soft model is not exact Bayesian marginalization

For a hard role pair \(R_i,R_j\), the expected pre-attention bias is:

\[
\mathbb E[G_{R_iR_j}]
=
p_i^\top Gp_j.
\]

The expected message expert is likewise the proposed soft mixture. But:

\[
\operatorname{softmax}(\mathbb E[e])
\ne
\mathbb E[\operatorname{softmax}(e)]
\]

in general, and:

\[
f\!\left(\sum_rp_r x_r\right)
\ne
\sum_rp_r f(x_r)
\]

for a nonlinear \(f\). The network is a differentiable relaxation of a hard
typed graph, not exact inference over every hard role-labelled graph.

### 6.3 “HetNet generalization” needs narrower wording

One-hot \(p_i\) selects a hard role and an ordered relation operator in the
proposed implementation. That proves reduction to **the project's own hard
typed model**. It does not reproduce every component of the original HetNet,
which includes supplied class-specific encoders, decoders, policies,
heterogeneous spaces, and binarized communication.

Use:

> a continuous relaxation of an ordered typed-relation architecture inspired
> by HetNet

rather than:

> an exact soft generalization of the complete HetNet algorithm.

## 7. Mathematics worth keeping in Route A

The present formulation can be made much sharper without pretending these are
deep new theorems. These properties define exactly what the architecture does
and give falsifiable implementation tests.

Define the soft typed operator:

\[
\mathcal T_{p,q}(x)
=
\sum_{r=1}^{K}\sum_{s=1}^{K}
p_rq_sM_{rs}(x),
\qquad p,q\in\Delta^{K-1}.
\]

Assume:

\[
\max_{r,s}\|M_{rs}(x)\|\le B(x).
\]

### Proposition 1 — exact vertex recovery

For one-hot basis vectors \(e_a,e_b\):

\[
\mathcal T_{e_a,e_b}(x)=M_{ab}(x).
\]

This is the precise hard-role reduction that should be unit-tested.

### Proposition 2 — convex-hull containment

Because every coefficient is nonnegative and:

\[
\sum_{r,s}p_rq_s=1,
\]

then:

\[
\mathcal T_{p,q}(x)
\in
\operatorname{conv}\{M_{rs}(x):r,s\in[K]\}.
\]

Before later nonlinearities, soft routing cannot create a relation output
outside the convex hull of the hard experts.

### Proposition 3 — gate stability

For \(p,p',q,q'\in\Delta^{K-1}\):

\[
\left\|
\mathcal T_{p,q}(x)-\mathcal T_{p',q'}(x)
\right\|
\le
B(x)
\left(
\|p-p'\|_1+\|q-q'\|_1
\right).
\]

A proof adds and subtracts \(\mathcal T_{p',q}(x)\) and applies the triangle
inequality. This provides a local sensitivity certificate for posterior
perturbations.

### Proposition 4 — hardening bound

Let \(a=\arg\max_r p_r\) and \(b=\arg\max_s q_s\). Then:

\[
\left\|
\mathcal T_{p,q}(x)-M_{ab}(x)
\right\|
\le
2B(x)(1-p_aq_b).
\]

When both endpoint posteriors are confident, replacing the soft pair with its
argmax pair changes the pre-attention message by a bounded amount. A
whole-policy bound would additionally require Lipschitz constants for
attention, fusion, recurrence, and the actor.

### Proposition 5 — action-mixture hardening bound

For:

\[
\pi_p=\sum_rp_r\pi_r
\]

and \(a=\arg\max_rp_r\):

\[
\operatorname{TV}(\pi_p,\pi_a)
\le
\sum_{r\ne a}p_r\operatorname{TV}(\pi_r,\pi_a)
\le
1-p_a.
\]

This links gate confidence to a directly interpretable change in the action
distribution.

### Proposition 6 — low-rank role logit structure

Stack memberships into \(P\in\mathbb R^{N\times K}\). The role contribution to
the pairwise attention logits is:

\[
B_{\mathrm{role}}=PGP^\top,
\]

so:

\[
\operatorname{rank}(B_{\mathrm{role}})\le K.
\]

For one-hot \(P\), \(B_{\mathrm{role}}\) is block-constant. An asymmetric \(G\)
creates directed blocks. The final softmax attention matrix need not remain
low-rank; the property applies to the additive role-logit term.

### Proposition 7 — agent-permutation equivariance

If the encoder, router, message aggregation, and actor share parameters across
agents, and the mask transforms as \(A\mapsto QAQ^\top\), then:

\[
\Pi(QH,QAQ^\top)=Q\Pi(H,A).
\]

This should be tested for role probabilities, messages, attention, and action
probabilities—not merely for the encoder.

### Proposition 8 — role-label gauge invariance

Let \(S\) be a \(K\times K\) role-permutation matrix. Transform:

\[
P'=PS,\qquad G'=S^\top GS,
\]

and reindex every role and role-pair expert consistently. The represented
policy is unchanged. Latent role numbers are therefore not identifiable; they
are coordinates with a label-permutation symmetry.

The non-identifiability can be worse than discrete label switching. If experts
are equal or linearly redundant, multiple continuous combinations of gates and
expert parameters can implement the same policy. Role alignment is therefore
a diagnostic, not proof that the latent variable is unique or semantically
true.

### Proposition 9 — deterministic symmetry limitation

If two nodes remain in the same automorphism orbit with identical
execution-available histories, a deterministic equivariant router assigns the
same posterior to both. Stochastic sampling may produce different realized
roles, but it does not by itself create stable semantics or coordinated
symmetry breaking.

These nine properties are a useful theorem-and-test package. Their value is
precision and auditability. They should not be marketed as a new mathematical
field.

## 8. Hostile scientific review of the proposed experiment

### 8.1 The hidden-role PCP task mostly tests cue decoding

The current observation layout explicitly contains:

\[
o_i[4]=\texttt{target\_visible}.
\]

At \(\alpha=0\), full-sight agents normally receive \(1\), while hard-blind
agents receive \(0\); see
[envs/asymmetry.py](envs/asymmetry.py). Randomizing agent indices prevents an
index shortcut, but the one-bit capability cue remains. A feedforward router
can recover detector versus capturer without team negotiation.

The resulting positive claim would be:

> The policy converted an execution-available local capability cue into
> capability-conditioned action and communication computation.

It would not be:

> Anonymous agents collectively invented roles from an otherwise regular
> graph.

At \(\alpha=1\), the observation semantics become the same while physical
capture authority may remain different. If the hidden assignment is
independent of position and history, a shared local router cannot infer it
above chance. Better-than-chance recovery would indicate leakage or another
cue, not surprising intelligence.

### 8.2 Fixed physical classes do not test dynamic role reassignment

Resampling a detector/capturer permutation at episode reset tests
generalization over hidden type assignments. Keeping the assignment fixed
within an episode does not test:

- agents switching roles as task demand changes;
- a team changing the number of agents in each role;
- reassignment after an agent fails;
- new role creation;
- or endogenous capability allocation.

If “dynamic roles” is central, the environment must contain a within-episode
event that changes role demand or capability and makes reassignment beneficial.

### 8.3 Independent local softmaxes do not form a team partition

For:

\[
p_i=\operatorname{softmax}(u_i),
\]

each row sums to one, but there is no constraint across rows. Every agent may
select the same role. The following are all possible:

\[
\sum_i p_{i,1}=N,\qquad
\sum_i p_{i,2}=0,
\]

or:

\[
p_i=[1/K,\ldots,1/K]\quad\forall i.
\]

The latter yields:

\[
K_{\mathrm{eff}}
=
\exp H\!\left(\frac1N\sum_i p_i\right)
=K
\]

despite there being no differentiated agents. Report both the mean individual
entropy and the role-to-agent mutual-information diagnostic:

\[
I(I;R)
=
H\!\left(\frac1N\sum_i p_i\right)
-
\frac1N\sum_iH(p_i).
\]

High population entropy with low individual entropy supports differentiated
occupancy. Neither metric establishes useful semantics; causal interventions
must do that.

### 8.4 The task may not need role-specialized computation

The most important feasibility question is not whether the router can classify
roles. It is whether separate experts add value after ordinary shared
communication. If an oracle type gate and a uniform equal-capacity gate perform
similarly, the environment cannot adjudicate the proposed mechanism.

This motivates an early stop rule:

> Do not train learned gates at scale unless a privileged type-gated reference
> beats a uniform gate by a preregistered practical margin.

Ten absolute episodic-success percentage points is a reasonable development
threshold, but it should be chosen before those runs.

### 8.5 “Oracle” is not an upper bound

The physical detector/capturer partition may not be the best computational
partition. A learned soft policy can combine experts or form a different
behavioral decomposition and outperform the type-gated model. Call the
condition:

> privileged physical-type-gated reference

rather than:

> architectural upper bound.

Only compute an oracle-gap recovery ratio when the privileged-reference minus
uniform denominator is itself materially nonzero.

## 9. Hostile mathematical and systems review

### 9.1 Decentralized execution is underspecified

The relation:

\[
v_{ij}
=
\sum_{r,s}p_{ir}p_{js}M_{r\leftarrow s}(h_j)
\]

requires receiver \(i\)'s posterior, sender \(j\)'s posterior, and sender
content. A vectorized simulator can access all three in one global tensor.
That does not define a decentralized protocol.

Three legitimate implementations are:

1. sender transmits \(h_j\) and \(p_j\), costing \(D+K\) real scalars; receiver
   applies its own role-conditioned transform;
2. factor:

   \[
   M_{r\leftarrow s}=A_rB_s,
   \]

   let the sender compute a sender-conditioned basis and the receiver apply
   its own decoder;
3. sender unicasts receiver-specific messages, which increases payload and
   requires the sender to know receiver context.

The first is simplest and honest. The role token is communication overhead and
must be supplied equivalently to controls. If it is hardened, the token can be
encoded in approximately \(\lceil\log_2K\rceil\) bits; if it remains a
floating-point vector, report scalars and precision rather than claiming a bit
rate.

### 9.2 Attention is not bandwidth reduction

If every sender payload is available before attention assigns weight, then all
candidate edges communicated. A small \(\alpha_{ij}\) reduces computational
influence, not transmitted bytes.

To claim communication efficiency, the architecture needs:

- a discrete send or edge decision before payload transmission;
- an explicit header, scheduling, and payload protocol;
- a message alphabet or numerical precision;
- and a cost that includes role metadata and negotiation rounds.

Without those pieces, use “selective aggregation” or “routing weight,” not
“bandwidth-efficient topology.”

### 9.3 A supplied mask is not learned connectivity

The proposed \(A_t^0\) says which edges are allowed. The role posterior changes
biases and values on those edges but cannot create a forbidden path or avoid
transmitting on an allowed one. This differs from methods such as
[MAGIC](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf),
G2ANet, and SCoUT that explicitly learn communication decisions or graph
structure.

At the current PCP setting, \(N=5,k=4\) produces a complete graph. It cannot
support claims about sparsity, multi-hop routing, scaling, graph rewiring, or
topological bottlenecks.

### 9.4 Spectral connectivity is not a universal coordination objective

For a symmetric nonnegative negotiation graph with Laplacian \(L\), a linear
lazy-consensus step:

\[
C=I-\eta L
\]

can contract disagreement at a rate governed by \(\lambda_2(L)\) under the
usual step-size conditions. That is useful if the negotiation objective is
consensus.

The task path detector \(\rightarrow\) capturer is directed and semantic.
Maximizing the algebraic connectivity of a symmetrized graph may waste budget
on reverse or within-role edges, homogenize useful specialist information, and
conflict with sparsity. A task-flow reachability or causal value criterion is
more appropriate than adding \(-\lambda_2\) because it looks sophisticated.

### 9.5 Communication cost must use an explicit unit

Possible units are:

- number of active directed payload edges;
- number of transmitted real scalars at a specified precision;
- number of discrete symbols;
- coded bits under an explicitly defined source/channel model;
- latency or number of synchronous rounds;
- energy under a specified radio or network model.

These units are not interchangeable. A deterministic 64-dimensional float
message has no intrinsic Shannon rate unless quantization, noise, and coding
are specified.

### 9.6 Role and expert optimization is degenerate

If all action experts are identical:

\[
\pi_r(a)=\pi_s(a)\quad\forall r,s,
\]

then the direct action-mixture gradient into the router vanishes. If the gate
becomes sharp too early, unused experts receive little data and remain bad,
which further sharpens the gate. This is the familiar rich-get-richer MoE
failure.

Initialize experts with small controlled differences. Begin without sharpness,
load, or switching losses. Add at most one prespecified stabilization after
observing a diagnosed failure. A 3:2 load target injects knowledge of the
hidden capability counts and must be reported as weak supervision.

The PCP dense reward is detector-oriented while the critical capturer behavior
depends more on sparse capture events. That creates an additional route to
detector-like expert collapse.

### 9.7 Sharing one posterior may be harmful coupling

Using one \(p_i\) for both action and communication is the remaining
architectural distinction, but it is an empirical hypothesis, not an
unqualified benefit. The behaviorally useful partition may differ from the
information-routing partition.

The essential factorial is:

| Action gate | Communication gate | Question |
|---|---|---|
| Uniform | Uniform | Equal-capacity null |
| Learned | Uniform | Action specialization only |
| Uniform | Learned | Communication specialization only |
| Shared learned posterior | Shared learned posterior | Coupled model |
| Separate learned posteriors | Separate learned posteriors | Is coupling itself useful? |

If separate gates outperform one shared gate, the proposed coupling is a
constraint rather than a contribution.

### 9.8 PPO bookkeeping crosses agent boundaries

For deterministic soft roles, the environment-action distribution factorizes
conditional on the complete communication forward pass. However, \(p_j\) can
change several receivers' messages and therefore several agents' action
likelihoods. PPO evaluation must reconstruct the same mask, temperature,
posteriors, role metadata, and message computation from the stored rollout
inputs, without detaching cross-agent gradients.

The initial pre-update ratio should be one within numerical tolerance. If hard
roles or communication graphs are sampled, their likelihoods become part of
the team latent decision. Per-agent clipping is then an approximation because
one sampled routing action changes teammates' action distributions.

### 9.9 A randomized assignment changes the critic's state

The current privileged state contains positions, target position, and
optionally timestep, but not the physical detector/capturer assignment; see
[envs/pursuit.py](envs/pursuit.py). After assignments vary by environment,
the same state vector can induce different observation and reward dynamics.

Append the true assignment to the centralized critic state for every
experimental condition, disclose it as privileged CTDE information, and never
pass it to the learned or uniform actor. Keeping critic inputs identical is
more important than making the critic superficially “role agnostic.”

### 9.10 The repository requires structural environment changes

Current class identifiers are static \([N]\) tensors:

- [envs/asymmetry.py](envs/asymmetry.py) asserts \([N]\);
- [envs/pursuit.py](envs/pursuit.py) constructs one fixed vector;
- adapters cache it;
- [experiments/train_vmas_mappo.py](experiments/train_vmas_mappo.py) caches it
  before the rollout.

Randomized parallel assignments require \([B,N]\) masks throughout observation,
reward, diagnostics, critic state, trainer, and evaluation. The lockstep
auto-reset in `PursuitEnv.step` samples new positions directly rather than
calling `PCPEnv.reset`, so sampling roles only in `reset()` would silently fail
to resample them during ordinary training. Introduce one episode-reset hook
used by both explicit and automatic resets.

The deterministic MAPPO evaluator currently calls the actor without rebuilding
the training communication mask. That must be repaired before topology or
directional-edge claims.

At the review date, the focused existing-suite command:

```text
expenv/bin/python -m pytest -q tests --ignore=tests/test_plot_run_metrics.py
```

reported 206 passing tests. Treat that as a preservation baseline, not as a
clean repository-wide result: unrestricted test collection also reaches
legacy tests and a local Matplotlib/Pyexpat environment problem. No
checkpoints were present under `runs/`; only smoke-era checkpoints were found
under `old_runs/`. The existing-checkpoint causal audit in the timeline
therefore depends on recovering the intended trained checkpoint or retraining
it from a frozen configuration.

### 9.11 Capacity matching needs stronger controls

Uniform, learned, and privileged hard gates have equal nominal parameter counts
but different function classes and gradient paths:

- uniform routing trains every expert on every sample;
- hard routing gives disjoint samples to selected experts;
- learned soft routing jointly optimizes the router and all activated experts;
- the overridden router is unused in uniform and privileged conditions.

Add:

- a widened shared model matched on trainable parameters, approximate FLOPs,
  and transmitted scalars;
- random balanced gates unrelated to capability;
- same-checkpoint gate shuffles;
- and the action/communication factorial.

The same-checkpoint interventions are more causally interpretable than a
between-training comparison alone.

### 9.12 Statistical risks remain

Use the training seed as the independent unit. Evaluation episodes are nested
observations, not independent replications of training.

Required practices:

- use the final checkpoint or a frozen validation-selection rule;
- evaluate at least 500 held-out episodes per trained seed;
- use identical episode manifests across interventions;
- show raw seed effects and a paired seed-level or hierarchical bootstrap;
- fit role-label alignment on separate validation episodes and lock it before
  test;
- split probes by whole episode or trajectory, never by individual correlated
  rows;
- predeclare one primary estimand and a practical effect threshold;
- gate secondary tests or correct for multiplicity.

## 10. Overall feasibility assessment

| Dimension | Assessment | Reason |
|---|---|---|
| Basic soft role router | High | Small shared MLP with clear tensor shapes |
| Probability-mixture actor | High | Standard categorical MoE, provided PPO log probabilities are exact |
| Full \(K^2\) message bank at \(N=5,K=2\) | High computationally | Small relation bank, though decentralization must be specified |
| Randomized \([B,N]\) physical types | Medium | Requires environment, reset, state, adapter, and trainer refactor |
| Scientifically meaningful hidden-role recovery in current PCP | Low-to-medium | \(\alpha=0\) is nearly one-bit capability classification |
| Dynamic role discovery | Low in current task | Roles do not change within episodes |
| Learned sparse topology claim | Unsupported | Complete five-node candidate graph and post-receipt attention |
| Variable-team generalization | Medium-to-low in current learner | Actor can be equivariant; critic, storage, evaluation, and QMIX mixer remain fixed-\(N\) |
| Route A thesis extension | Feasible | 5.5–8.25 person-weeks after causal audit |
| Route B algorithmic contribution | Feasible but high-risk | Needs new task, coupled allocation, explicit protocol, and much larger experiment matrix |

## 11. Novelty options, ranked

### Option 1 — Current SoftRole-HetGAT plus rigorous characterization

Keep:

\[
p_i,\qquad
p_i^\top Gp_j,\qquad
\sum_{r,s}p_{ir}p_{js}M_{r\leftarrow s},\qquad
\sum_rp_{ir}\pi_r.
\]

Add the propositions in Section 7, a fully explicit execution protocol, and the
factorial causal evaluation.

**Benefit:** lowest implementation risk; directly connected to the current
code and causal thesis.

**Limitation:** the contribution is a precisely characterized synthesis.
ROCO, SCoUT, SORA, HetNet, mixed-membership models, and MoE make a broad
architecture-novelty claim weak.

**Recommendation:** this is Route A and the only reasonable architecture
extension on a short thesis schedule.

### Option 2 — Joint constrained role assignment

Replace independent per-agent softmaxes with one team allocation matrix whose
row and column sums are controlled. This changes the question from:

> What role distribution does each agent independently predict?

to:

> How should the team distribute a finite set of role responsibilities among
> its members?

**Benefit:** the model now actually forms a team allocation, prevents all-agent
collapse by construction, and can represent changing role demand.

**Limitation:** optimal transport and balanced assignment are established;
centralized Sinkhorn is not decentralized; fixed quotas inject task knowledge;
and the current PCP task does not need joint allocation because the local
capability cue is almost deterministic.

**Recommendation:** use only with a variable-demand task and explicit
decentralization statement.

### Option 3 — Joint assignment plus hard budgeted directed communication

Couple the team allocation to a communication scheduler that selects payload
edges under an explicit cost. This asks whether inferred organizational
structure improves the return-versus-communication Pareto frontier.

**Benefit:** it unifies the thesis's information-asymmetry result with a real
network constraint. It also distinguishes “who should do what” from “who needs
whose information.”

**Limitation:** SchedNet, IMAC, MAGIC, SCoUT, ROCO, and networking theory all
cover pieces of this problem. Hard routing adds a difficult stochastic or
combinatorial decision, protocol overhead, and cross-agent credit assignment.

**Recommendation:** combine with Option 2 as the strongest candidate for a
separate algorithmic paper, not a small add-on.

### Option 4 — Role-conditioned rate-distortion messages

Let:

\[
M\sim q_\theta(m\mid X,R_i,R_j)
\]

and penalize a conditional variational rate:

\[
\mathcal R
=
\mathbb E
D_{\mathrm{KL}}
\left(
q_\theta(M\mid X,C)
\|
r_\psi(M\mid C)
\right)
\ge
I(X;M\mid C),
\]

where \(C=(R_i,R_j)\). Sweep reward or action distortion against
\(\mathcal R/\ln2\).

**Benefit:** gives a principled notion of content compression by role pair.

**Limitation:** IMAC already applies the information bottleneck to MARL
communication; continuous KL is not an actual bit count without a coding
model; sequential return is not automatically Shannon distortion.

**Recommendation:** future work, not core.

### Option 5 — Spectrally constrained negotiation

Use a symmetric linear negotiation layer with a communication budget and:

\[
\lambda_2(L)\ge\delta.
\]

This gives a real consensus contraction statement under restricted assumptions.

**Benefit:** defensible mathematics for a role-negotiation phase.

**Limitation:** it does not justify directed nonlinear task attention, may
erase specialist differences, and is uninformative on the present complete
five-node graph.

**Recommendation:** a separate control/topology study.

### Option 6 — Graphon, mean-field, open-population, or endogenous tools

Graphon policies, mean-field MARL, learned role counts, agents joining and
leaving, and online capability allocation are each large research directions.
GIRA and role-transfer work already cover important parts of population
adaptation.

**Recommendation:** do not combine all of them. Endogenous tool allocation is
the one future extension most closely aligned with this thesis, because it makes
the sensing/action split a decision rather than a supplied fact.

## 12. Recommended higher-novelty formulation

The working name below is descriptive, not a novelty claim:

> **Constrained Online Role and Topology Allocation (CORTA).**

The candidate contribution is the complete constrained control problem, not
the use of Sinkhorn, MoE, or graph attention individually.

### 12.1 Augmented decentralized control problem

At time \(t\), let:

- \(H_t=[h_{t,1},\ldots,h_{t,N}]^\top\) contain execution-available agent
  histories;
- \(A_t^0\in\{0,1\}^{N\times N}\) be the permitted directed candidate graph,
  using receiver-first indexing;
- \(P_t\in[0,1]^{N\times K}\) be the team role allocation;
- \(Z_t\in\{0,1\}^{N\times N}\) be the payload-edge schedule;
- \(b_t\in\mathbb N^K\), \(\sum_rb_{t,r}=N\), be current role demand;
- \(B_t\) be the communication budget.

The policy chooses organizational variables and ordinary actions:

\[
(P_t,Z_t,\mathbf a_t)
\sim
\Pi_\theta(
\cdot\mid
\boldsymbol\tau_t,A_t^0,b_t,B_t
).
\]

For computational roles, \(P_t\) only routes neural modules. For physical
capability allocation, a hard assignment changes the environment's observation
or action functions:

\[
o_{t,i}\sim O^{z_{t,i}}(\cdot\mid s_t),
\qquad
a_{t,i}\in\mathcal A^{z_{t,i}}.
\]

That second case is a hierarchical constrained Dec-POMDP, not just an actor
architecture. It requires an explicit allocation phase and resource rules.

### 12.2 Team-level entropic role allocation

Let a shared scorer produce suitability:

\[
U_{t,ir}
=
f_{\mathrm{score}}
(h_{t,i},c_{t,i},r),
\]

where \(c_{t,i}\) is only context legally available after any declared
negotiation round.

Define:

\[
\mathcal U(b_t)
=
\left\{
P\ge0:
P\mathbf 1_K=\mathbf 1_N,\;
P^\top\mathbf 1_N=b_t
\right\}.
\]

Then:

\[
P_t^\star
=
\arg\max_{P\in\mathcal U(b_t)}
\left[
\langle U_t,P\rangle
+
\varepsilon H(P)
-
\frac{\kappa}{2}
\|P-\widetilde P_{t-1}\|_F^2
\right].
\]

Here:

\[
H(P)=-\sum_{i,r}P_{ir}\log P_{ir}.
\]

\(\widetilde P_{t-1}\) is the previous assignment aligned only across agents
that persist. Omit the temporal term in the first implementation; it
complicates join/leave semantics and is unnecessary for episode-level
allocation.

There are three scientifically different demand variants:

1. **Known demand:** \(b_t\) is supplied by the task. This cleanly tests agent
   allocation, not role-count inference.
2. **Learned demand:** a permutation-invariant team summary predicts
   \(\rho_t\), followed by \(b_t\approx N\rho_t\). This requires a common
   execution-time signal or negotiation.
3. **Unbalanced allocation:** penalize, rather than fix, column totals. This
   permits unused roles but weakens feasibility guarantees.

Use known demand first. Do not quietly describe a supplied 3:2 quota as
unsupervised discovery.

For \(\varepsilon>0\) and positive feasible marginals, the entropy term gives a
unique soft optimum with a Sinkhorn-scaled exponential form. If agent rows in
\(U_t\) are permuted by \(Q\):

\[
P^\star(QU_t)=QP^\star(U_t).
\]

With integer marginals and a unique linear optimum, the
\(\varepsilon\rightarrow0\) transport solution approaches an integral
assignment vertex. This follows from the assignment polytope's integral
structure; it is not a new RL convergence theorem.

### 12.3 Role-conditioned edge utility

For every permitted receiver \(i\), sender \(j\):

\[
u_{t,ij}
=
\frac{q(h_{t,i})^\top k(h_{t,j})}{\sqrt d}
+
p_{t,i}^\top C p_{t,j}
+
\chi_{t,ij},
\]

where \(\chi_{t,ij}\) may contain legal physical-link features such as distance,
latency, or channel quality.

The mixed-membership term:

\[
P_tCP_t^\top
\]

is intentionally acknowledged as an MMSB-like low-rank block operator. The new
question is whether it helps choose a constrained task communication graph.

### 12.4 Hard communication feasibility set

Let \(c_{ij}\ge0\) be the cost of one payload edge. Define:

\[
\mathcal G(A_t^0,B_t)
=
\left\{
Z\in\{0,1\}^{N\times N}:
Z\le A_t^0,\;
Z_{ii}=0,\;
\sum_{i,j}c_{ij}Z_{ij}\le B_t
\right\}.
\]

Optionally impose per-sender and per-receiver limits:

\[
\sum_iZ_{ij}\le b_j^{\mathrm{send}},
\qquad
\sum_jZ_{ij}\le b_i^{\mathrm{recv}}.
\]

Then:

\[
Z_t^\star
\in
\arg\max_{Z\in\mathcal G(A_t^0,B_t)}
\sum_{i,j}Z_{ij}u_{t,ij}.
\]

At small \(N\), an exact top-budget or matching solver is possible. At larger
\(N\), use local per-receiver top-\(k\), Gumbel-top-\(k\), or another
differentiable relaxation during training. The evaluation graph must be hard
if the claim concerns transmitted payloads.

A continuous \(w_{ij}\in[0,1]\) satisfying an expected budget is acceptable as
a training relaxation. If all payloads are transmitted to compute it, report
only soft routing—not bandwidth savings.

### 12.5 Explicit two-stage network protocol

A deployable version must say who knows what:

1. each sender exchanges a small header with permitted neighbors;
2. receivers or a local scheduler select payload edges;
3. accepted senders transmit role metadata and content;
4. receivers apply their own role-conditioned decoder.

For a bounded-degree candidate graph with \(E_t^0\) permitted edges:

\[
C_t^{\mathrm{bits}}
=
E_t^0 b_{\mathrm{header}}
+
|Z_t|
\left(
b_{\mathrm{ack}}+b_{\mathrm{role}}+b_{\mathrm{payload}}
\right).
\]

This exposes a subtle scaling fact: a payload budget can be \(O(N)\) while
dense all-pairs header scoring remains \(O(N^2)\). To claim both communication
and compute scalability, use a connected bounded-degree candidate graph or a
subquadratic neighbor-discovery mechanism.

### 12.6 Factorized role-pair message operator

For a practical decentralized message, factor:

\[
M_{r\leftarrow s}=A_rB_s.
\]

The sender computes:

\[
\zeta_{t,j}
=
\sum_s p_{t,j,s}B_sh_{t,j}
\]

and transmits \((p_{t,j},\zeta_{t,j})\) on a selected edge. Receiver \(i\)
computes:

\[
v_{t,ij}
=
\sum_rp_{t,i,r}A_r\zeta_{t,j}.
\]

This yields:

\[
v_{t,ij}
=
\sum_{r,s}
p_{t,i,r}p_{t,j,s}
A_rB_sh_{t,j},
\]

which is the original pair mixture with a low-rank relation bank. It keeps
receiver context local and avoids sending \(K\) full payloads.

If the complete unfactorized \(M_{rs}\) bank is scientifically essential,
transmit \(h_j\) and \(p_j\), then let the receiver evaluate every pair expert.
Report that receiver-side cost.

### 12.7 Role-conditioned aggregation and action

For selected edges:

\[
\alpha_{t,ij}
=
\frac{
Z_{t,ij}\exp(u_{t,ij})
}{
\sum_{\ell}Z_{t,i\ell}\exp(u_{t,i\ell})
}.
\]

Define finite fallback behavior for a receiver with no selected incoming edge.
Then:

\[
m_{t,i}
=
\sum_j\alpha_{t,ij}v_{t,ij},
\qquad
y_{t,i}=f_{\mathrm{fuse}}(h_{t,i},m_{t,i}),
\]

and:

\[
\pi_i(a\mid y_{t,i},p_{t,i})
=
\sum_rp_{t,i,r}\pi_r(a\mid y_{t,i}).
\]

For hard physical capability masks, probability mixing over infeasible actions
is not meaningful. Either:

- allocate one hard capability before task execution and mask its action set;
- or retain a common physical action set and call the experts computational
  roles.

### 12.8 Constrained learning objective

Let:

\[
J_R(\theta)
=
\mathbb E_{\Pi_\theta}
\left[
\sum_t\gamma^tR_t
\right],
\]

and:

\[
J_C(\theta)
=
\mathbb E_{\Pi_\theta}
\left[
\sum_t\gamma^tC_t
\right].
\]

The principled problem is:

\[
\max_\theta J_R(\theta)
\quad\text{subject to}\quad
J_C(\theta)\le\overline C.
\]

One practical Lagrangian is:

\[
\mathcal L(\theta,\lambda)
=
J_R(\theta)
-
\lambda
\left(
J_C(\theta)-\overline C
\right),
\qquad
\lambda\ge0,
\]

with:

\[
\lambda
\leftarrow
\left[
\lambda+\eta_\lambda
\left(
\widehat J_C-\overline C
\right)
\right]_+.
\]

This is preferable to choosing an arbitrary communication \(L_1\) coefficient
and later calling the outcome “budgeted.” Do not claim global primal-dual
convergence for a nonconvex neural MARL system. The exact per-step feasible
edge set can guarantee the hard link budget even when policy optimization has
no global guarantee.

### 12.9 What is plausibly new in the complete formulation

The candidate contribution is:

> A permutation-equivariant constrained policy that jointly allocates
> team-level role mass and a hard directed payload graph, uses the same
> assignment to route action and ordered endpoint-role message operators, and
> optimizes task return under an explicit communication budget as role demand
> and team size vary.

Each ingredient has precedents. The conjunction and controlled causal study
may be new. It must be compared directly with:

- independent SoftRole gates;
- hard type-gated HetNet-style relations;
- ROCO-like hard roles with separate intra-/inter-role communication;
- GRDC-like same-role-only communication;
- SCoUT-like group-affinity recipient routing;
- ordinary content attention under the same hard budget;
- and transport allocation without role-conditioned communication.

## 13. Formal properties for Route B

These are realistic theorem targets. They are architecture properties, not
claims of optimal MARL.

### 13.1 Unique entropic allocation

With positive feasible marginals and \(\varepsilon>0\), the transport objective
without the temporal term is strictly concave in \(P\), so it has a unique
solution.

### 13.2 Permutation equivariance

For agent permutation \(Q\):

\[
U_t\mapsto QU_t,\quad
A_t^0\mapsto QA_t^0Q^\top.
\]

Uniqueness of the transport solution gives:

\[
P_t^\star\mapsto QP_t^\star.
\]

If the edge optimizer is permutation equivariant:

\[
Z_t^\star\mapsto QZ_t^\star Q^\top,
\]

then shared message passing and action experts give a permuted joint policy.

### 13.3 Exact occupancy feasibility

By construction:

\[
P_t^\top\mathbf1_N=b_t.
\]

This is stronger than a load-balancing penalty. It is also stronger prior
information and must be disclosed.

### 13.4 Exact edge-budget feasibility

For every returned hard graph:

\[
\sum_{ij}c_{ij}Z_{t,ij}\le B_t.
\]

This is an architectural guarantee independent of whether RL training reaches
an optimal return.

### 13.5 Hard typed-model reduction

When \(P_t\) is integral and \(Z_t=A_t^0\), the model selects one role expert
and one ordered relation per endpoint pair. It reduces to the project's hard
typed policy.

### 13.6 Complexity distinction

For \(L_{\mathrm{SK}}\) Sinkhorn iterations, \(E_0=|A_t^0|\) candidate edges,
and \(B=|Z_t|\) selected payload edges:

\[
\text{assignment compute}=O(L_{\mathrm{SK}}NK),
\]

\[
\text{edge scoring}=O(E_0(d+K)),
\]

\[
\text{payload aggregation}=O(BD).
\]

If \(E_0=O(N^2)\), scoring is still quadratic even when \(B=O(N)\). If the
candidate graph has bounded degree, all three can scale linearly in \(N\) for
fixed \(K,D,L_{\mathrm{SK}}\).

### 13.7 Symmetry limitation remains

Identical anonymous inputs on a symmetric graph still do not yield a unique
semantically differentiated hard assignment without a legitimate tie-break.
The constrained allocator coordinates scarcity; it does not repeal
distributed-computing symmetry.

## 14. Task design required to support the new claim

### 14.1 Keep randomized hidden-capability PCP for Route A only

This task is useful for verifying:

- batched hidden type permutations;
- one-hot versus soft typed routing;
- capability-cue generalization;
- and causal use of a learned gate.

It is not sufficient for joint allocation or dynamic role formation.

### 14.2 Add Variable-Demand Tool PCP for Route B

A clean next environment would start agents physically homogeneous and make
scarce tools an allocation decision.

At an allocation macro-step, assign each agent one mutually exclusive tool:

- **sensor:** sees the target but cannot execute capture;
- **capturer:** can execute capture but does not see the target;
- optional **relay:** can forward farther or at lower cost but has neither end
  capability.

Resource demand \(b_t\) varies across episodes, and later across phases within
an episode. The task should vary:

- number and location of targets;
- number of available tools of each type;
- team size \(N\);
- candidate communication degree;
- payload budget;
- and occasional tool or agent failure.

Agent positions and local conditions should make suitability unequal; otherwise
every balanced assignment is equivalent. A mid-episode demand switch or failure
creates a genuine role-reallocation test.

This task follows directly from the thesis:

\[
\text{split sensing/action creates communication value}
\]

becomes:

\[
\text{the team decides the split and then builds the information path}.
\]

The additional complexity is real. Capability allocation changes the
environment transition, observation, and action functions and should be
presented as a separate hierarchical problem.

### 14.3 Controlled task ladder

Use a ladder rather than unrelated benchmarks:

1. **Information-sufficient PCP:** every agent sees and can act; roles should
   add little.
2. **Fixed split PCP:** supplied detector/capturer capabilities; communication
   should matter.
3. **Random hidden split PCP:** capabilities fixed per episode but labels
   hidden; tests inference from cues.
4. **Allocatable split PCP:** the policy assigns scarce tools; tests joint
   allocation.
5. **Dynamic allocatable PCP:** demand or failures change within an episode;
   tests reassignment.
6. **Variable-\(N\), bounded-degree PCP:** tests scaling and topology.

Every added level changes one conceptual factor. This is more interpretable
than immediately moving to an unrelated StarCraft benchmark.

## 15. Experimental design

### 15.1 Primary estimands

For Route A:

\[
\Delta_{\mathrm{shared}}
=
Y_{\mathrm{shared\ soft\ gate}}
-
Y_{\mathrm{uniform\ gate}}.
\]

For Route B:

\[
\Delta_{\mathrm{joint}}
=
Y_{\mathrm{joint\ allocation}}
-
Y_{\mathrm{independent\ gates}},
\]

under the same demand and communication budget.

For communication efficiency, compare Pareto frontiers rather than one
arbitrary penalty:

\[
\mathcal P
=
\left\{
\left(
\mathbb E[C],\mathbb E[Y]
\right)
\right\}.
\]

A useful result is that one method dominates another over a range of budgets,
not merely that it wins at one tuned multiplier.

### 15.2 Minimum baseline set

| Baseline | Purpose |
|---|---|
| Shared actor + no communication | Behavioral lower reference |
| Shared actor + content attention | Standard communication without roles |
| Widened shared actor/attention | Parameter/FLOP/payload capacity control |
| Privileged hard physical-type gate | Tests whether supplied types make specialized computation useful |
| Random balanced hard gate | Controls partitioning and expert utilization without correct semantics |
| Uniform soft gate | Equal-capacity routing null |
| Independent SoftRole | Tests ordinary local role inference |
| Separate action and communication posteriors | Tests whether sharing the posterior helps |
| Joint transport roles only | Isolates team allocation |
| Budgeted topology only | Isolates hard scheduling |
| Full joint allocation + topology | Proposed Route B |

Mechanism-matched ROCO-, GRDC-, and SCoUT-like variants should be implemented
inside the common codebase if faithful external reproduction is impractical.
State clearly that they are matched variants, not exact paper reproductions.

### 15.3 Intervention set

For the same checkpoint and common episode manifest:

- zero payload content;
- shuffle payloads across states while preserving marginal distribution;
- shuffle role posteriors across agents;
- replace roles with uniform, random balanced, privileged type, or a separate
  posterior;
- disable each ordered role-pair operator;
- block detector-to-capturer, reverse, and within-role edges;
- preserve edges but shuffle message content;
- preserve content but replace the learned graph with a degree-matched random
  graph;
- sweep hard communication budgets;
- inject agent failure or role-demand changes.

Correct messages versus distribution-preserving shuffles is more informative
than zeroing alone because zero activations may be out of distribution.

### 15.4 Metrics

Report four levels separately.

**Task**

- episodic capture success;
- return;
- time to first valid capture;
- target-directed progress;
- invalid capability-action rate.

**Role allocation**

- occupancy feasibility;
- individual entropy;
- population entropy;
- \(I(I;R)\);
- assignment switch rate;
- Hungarian-aligned capability accuracy only where ground truth exists;
- response time and regret after demand changes.

**Communication**

- active payload edges;
- header, metadata, and payload scalars or bits;
- rounds and latency;
- directed role-flow mass;
- performance versus budget;
- same-checkpoint shuffle and edge-removal effects.

**Optimization**

- router and expert gradient norms;
- expert utilization;
- action-policy and message-output divergence;
- PPO pre-update ratio error;
- constraint violation and dual variable;
- wall-clock and memory.

### 15.5 Statistical protocol

- Use untouched confirmatory training seeds after development.
- Treat training seed as the independent unit.
- Use at least 500 held-out episodes per seed for low-variance intervention
  estimates.
- Pair every intervention by episode manifest.
- Report all seed points, paired effects, confidence intervals, and the
  prespecified practical threshold.
- Do not tune role temperature, loss weights, or budget on confirmatory seeds.
- Use hierarchical gatekeeping: test the primary effect before interpreting
  role alignment, expert-specific, or topology-specific hypotheses.

## 16. Dependency-ordered implementation guide

The order below protects the existing thesis result and places cheap
falsification tests before expensive architecture work.

### Phase 0 — preserve and repair the causal thesis

1. Create a checkpoint/config/result manifest.
2. Repair episodic evaluation and reconstruct the training communication mask.
3. Add same-checkpoint zero, shuffle, noise, and directional-edge
   interventions.
4. Split probes by whole episode.
5. Complete the blind-versus-sighted matched causal comparison.

Primary files:

- `experiments/train_vmas_mappo.py`;
- a new `experiments/evaluate_interventions.py`;
- a new `comm/interventions.py`;
- `probes/predictability.py`;
- evaluation and manifest tests.

**Gate 0:** communication content must have a practically meaningful
same-checkpoint effect under blind PCP. If it does not, do not expand into role
routing to rescue the thesis.

### Phase 1 — refactor environment types correctly

Prefer a new randomized environment mode so fixed-role PCP remains a regression
target.

1. Change capability labels from \([N]\) to \([B,N]\).
2. Replace fixed index slicing with batched Boolean reductions.
3. create one episode-reset hook used by `reset()` and auto-reset;
4. use separate role and position random-number streams;
5. append hidden capability assignment to the centralized critic state;
6. expose labels only to privileged evaluation and the type-gated reference.

Primary files:

- new `envs/randomized_pcp.py`;
- `envs/asymmetry.py`;
- `envs/pursuit.py`;
- `envs/pcp.py`;
- `envs/pcp_adapter.py`.

Required tests:

- every batch row has the requested role count;
- assignments differ across parallel environments;
- labels stay fixed within an episode and resample on auto-reset;
- index does not predict type;
- actor tensors contain no private labels;
- legacy fixed-PCP trajectories remain unchanged under fixed seeds;
- critic state is Markov with respect to the randomized capability dynamics.

### Phase 2 — implement the smallest action-side SoftRole model

1. Add `roles/router.py`.
2. Add `backbone/role_actor.py`.
3. Keep communication shared and unchanged.
4. Implement learned, uniform, random-balanced, supervised-classifier, and
   privileged-type gate modes.
5. Verify exact probability-mixture log probabilities and entropy.

Required tests:

- \([B,N,K]\) rows are finite and normalized;
- one-hot and uniform equivalence;
- brute-force and implementation action likelihood agree within \(10^{-6}\);
- gradients reach the router and every active expert;
- agent and role permutations behave as predicted;
- save/load reconstructs the gate and experts exactly.

**Gate 1:** the privileged type gate must materially beat uniform. Otherwise
stop: the task has no useful action-specialization signal.

### Phase 3 — add explicit soft role communication

1. Add `role_probs` to `comm/base.py`; do not overload `class_id`.
2. Add `comm/soft_role_attention.py`.
3. Begin with shared Q/K and value-only role specialization.
4. Choose and document the receiver-side protocol.
5. Log payload width, metadata width, attention, and ordered expert
   utilization.

Required tests:

- one-hot typed-message equivalence;
- receiver-first/sender-second orientation;
- prohibited edges remain prohibited;
- all-masked rows are finite;
- no disconnected sender affects a receiver;
- gate perturbations satisfy the local stability tests;
- sender metadata available at execution matches the documented payload.

**Gate 2:** learned routing must beat uniform on development seeds, use both
roles, and show predicted same-checkpoint intervention effects. Alignment alone
does not pass.

### Phase 4 — integrate MAPPO and frozen evaluation

Update:

- `backbone/mappo.py`;
- `backbone/ppo_update.py`;
- `backbone/factory.py`;
- `experiments/train_vmas_mappo.py`;
- checkpoint configuration and analysis.

Only update `backbone/rollout_buffer.py` if ordered temporal losses or sampled
hard routing truly require new stored variables.

Required tests:

- maximum pre-update old/new log-probability difference below \(10^{-6}\);
- rollout and update use the same temperature and mask;
- mixture entropy is the entropy of the final categorical distribution;
- router and communication modules are in the actor optimizer and gradient
  clipping;
- deterministic evaluation uses the same topology construction;
- no NaNs under uniform, sharp, or all-masked edge cases.

### Phase 5 — run Route A decision experiment

Development:

- privileged type gate;
- learned soft gate;
- uniform gate;
- three seeds each.

Proceed only if:

- privileged minus uniform clears the practical threshold;
- learned has a positive median advantage over uniform;
- no more than one learned run collapses;
- both roles and critical experts receive meaningful gradients;
- role and message shuffles harm the checkpoint in the predicted direction.

Then run ten untouched seeds for the surviving conditions and the minimal
factorial needed to attribute action versus communication specialization.

### Phase 6 — build the variable-demand allocation task

Do not implement Sinkhorn first. Build and validate the environment and simple
heuristics first:

- random valid hard assignment;
- nearest-suitable assignment;
- privileged optimal or search-based small-\(N\) assignment;
- fixed type split.

If the privileged allocator does not beat random allocation, the task does not
test assignment quality.

Primary additions:

- `envs/variable_demand_pcp.py`;
- `roles/allocation_types.py`;
- task-specific tests and oracle heuristics.

**Gate 3:** assignment quality must create a material performance gap under a
fixed communication protocol.

### Phase 7 — implement joint transport allocation

Add:

- `roles/transport.py`;
- `roles/demand.py`;
- Sinkhorn convergence and marginal diagnostics;
- optional hard rounding for physical tool assignment.

Required tests:

- exact row and column marginals within tolerance;
- agent-permutation equivariance;
- agreement with brute-force assignment on small matrices as
  \(\varepsilon\to0\);
- stable gradients for development temperatures;
- exact disclosure of whether the operation is centralized, coordinator-based,
  or implemented through communication.

**Gate 4:** joint allocation must beat independent local gates specifically
when role demand varies or local evidence is ambiguous. No gain in that regime
means the constraint is unnecessary.

### Phase 8 — implement hard budgeted communication

Add:

- `comm/budgeted_router.py`;
- explicit header/selection/payload accounting;
- budget schedules and Pareto analysis;
- hard graph evaluation.

Required tests:

- no payload edge violates the candidate graph;
- hard cost never exceeds the budget;
- degree limits hold;
- random degree-matched graphs preserve edge count;
- changing a forbidden edge score cannot affect behavior;
- header and role-token costs appear in logged communication;
- selected-edge compute and all-pair scoring compute are reported separately.

**Gate 5:** the role-aware method must improve the return-cost Pareto frontier,
not merely return at a larger hidden communication cost.

### Phase 9 — variable team size and dynamic reassignment

Only after all earlier gates:

- make the centralized critic permutation invariant and size agnostic;
- remove fixed-\(N\) assumptions from storage and analysis;
- train on at least two team sizes and test an unseen size;
- add a within-episode demand change or failure;
- measure reassignment latency and transient performance.

Do not add QMIX until MAPPO passes the complete causal and constraint audit.
The current QMIX mixer and replay pipeline have additional fixed-\(N\)
assumptions.

## 17. Timeline and resource estimate

These are person-weeks, not promises about cluster queue time. A week assumes
one primary implementer with normal thesis-writing interruptions.

### 17.1 Thesis-safe path

| Phase | Work | Person-weeks | Compute |
|---|---|---:|---:|
| T0 | Reproducibility manifest and frozen test command | 0.5 | None |
| T1 | Evaluation metric, mask, and intervention repair | 1.25–1.75 | Tiny |
| T2 | Episode-grouped probe repair | 0.5 | CPU |
| T3 | Existing-checkpoint causal audit | 0.5 | Roughly 4–5M inference steps |
| T4 | Matched blind/sighted confirmation and write-up | 0.75–1.75 | Roughly 120 GPU-hours for a minimal MAPPO endpoint study; more if retraining all controls |

**Total:** approximately 3.5–5 person-weeks plus queue time.

This should be completed before an architecture extension.

### 17.2 Route A: characterized SoftRole-HetGAT

| Week | Deliverable | Exit criterion |
|---:|---|---|
| S1 | Randomized batched capability environment and reset hook | All environment and no-leak tests pass |
| S2 | Local router and probability-mixture actor | Likelihood, entropy, gradient, and permutation tests pass |
| S3–S4 | Explicit role-conditioned communication protocol | One-hot relation tests and bandwidth accounting pass |
| S5 | MAPPO/checkpoint/evaluator integration | Pre-update ratio error below \(10^{-6}\); deterministic evaluation matched |
| S6 | Three-seed, three-condition smoke study | Gates 1 and 2 pass |
| S7–S8 | Untouched confirmation, causal interventions, analysis | Primary learned-minus-uniform effect and same-checkpoint mechanism agree |

**Engineering total:** approximately 5.5–8.25 person-weeks.

Development smoke:

\[
3\text{ conditions}\times3\text{ seeds}=9\text{ runs},
\]

roughly 54–81 GPU-hours at the current estimated run length.

Untouched confirmation:

\[
3\text{ conditions}\times10\text{ seeds}=30\text{ runs},
\]

roughly 180–270 GPU-hours, followed by approximately 6M inference steps for a
ten-checkpoint, 12-intervention, 500-episode audit.

The widened capacity baseline and full action/communication factorial add
substantial runs. They are necessary for a publication claim, but can be
scoped after the three-seed gate.

### 17.3 Route B: joint allocation and budgeted topology

Route B begins only after Route A is stable.

| Week | Deliverable | Exit criterion |
|---:|---|---|
| R1–R2 | Variable-Demand Tool PCP plus random, heuristic, and privileged allocation | Assignment quality demonstrably matters |
| R3 | Entropic transport allocator and hard rounding | Marginal, equivariance, and brute-force tests pass |
| R4–R5 | Hard edge scheduler and explicit protocol | Per-step budgets and accounting are exact |
| R6 | Integrate joint role/topology policy | Stable training on one small task |
| R7 | Three-seed baseline elimination study | Joint method beats independent roles in variable-demand regime |
| R8–R9 | Dynamic demand/failure and bounded-degree graph | Reassignment and topology hypotheses are testable |
| R10–R12 | Untouched confirmation over surviving baselines and budget sweep | Pareto and causal claims pass |
| R13–R14 | Analysis, theorem write-up, limitations, reproduction package | Every claim maps to a test or estimand |

**Additional total:** approximately 12–14 person-weeks in a favorable case;
allow 16–18 if hard routing, variable-\(N\) storage, or optimization is unstable.

Do not launch every baseline at ten seeds. Use three development seeds to
eliminate mechanisms, freeze the final comparison, then use new seeds.

### 17.4 Suggested calendar if begun on 2026-07-27

| Date window | Target |
|---|---|
| Jul 27–Aug 30 | Finish thesis-safe causal audit |
| Aug 31–Oct 25 | Route A implementation and smoke gates |
| Oct 26–Nov 29 | Route A confirmation and write-up |
| Nov 30–Mar 7, 2027 | Route B implementation, staged experiments, and analysis |

This calendar is intentionally sequential. Parallel cluster runs may shorten
elapsed time; failed scientific gates should shorten the project by stopping
it.

## 18. Go/no-go decision tree

```text
Does same-checkpoint message content matter in blind PCP?
|
+-- no --> finish the causal thesis as a negative/qualified result;
|          do not add roles to rescue it
|
+-- yes
    |
    +-- Does privileged type gating beat uniform routing?
        |
        +-- no --> role-specialized computation is unnecessary in this task;
        |          keep roles as future work
        |
        +-- yes
            |
            +-- Does learned SoftRole beat uniform and survive shuffles?
                |
                +-- no --> useful specialization exists, but unsupervised
                |          routing failed; report that boundary
                |
                +-- yes
                    |
                    +-- Does one shared gate beat separate action/comm gates?
                        |
                        +-- no --> shared coupling is not supported
                        |
                        +-- yes --> Route A succeeds
                                     |
                                     +-- Does variable task demand make joint
                                         assignment beat independent gates?
                                         |
                                         +-- no --> stop before topology
                                         |
                                         +-- yes
                                             |
                                             +-- Does role-aware hard routing
                                                 improve the return-cost frontier?
                                                 |
                                                 +-- no --> assignment result only
                                                 |
                                                 +-- yes --> Route B succeeds
```

Negative branches are interpretable research results, not engineering excuses.

## 19. Claim language

### 19.1 Defensible Route A headline

> Prior work separately and jointly studies learned roles, role-specific
> policies, and role-aware communication. We test a narrower continuous-routing
> hypothesis: whether one execution-time role posterior can simultaneously mix
> action policies and interpolate every ordered endpoint-role message operator.
> Randomized hidden-capability PCP and same-checkpoint interventions test
> whether that routing variable is inferred, used, and behaviorally relevant.

### 19.2 Defensible positive result

> Under randomized hidden physical types, the learned gate converted
> execution-available capability cues into specialized action and directed
> message computation, outperformed an equal-capacity uniform gate, and showed
> the predicted sensitivity to gate, content, direction, and expert
> interventions.

### 19.3 Defensible Route B headline

> We formulate role formation as a constrained team allocation problem and
> communication as a hard directed resource-allocation problem. A shared
> assignment routes both typed action/message operators and payload edges while
> satisfying explicit occupancy and communication budgets under changing task
> demand.

### 19.4 Claims to prohibit

- first emergent-role MARL method;
- first method combining roles and communication;
- first inter-role/intra-role communication architecture;
- automatic discovery of arbitrary classes from a regular graph;
- learned capabilities when the environment supplied them;
- dynamic role reassignment from episode-fixed labels;
- topology learning from attention over a fixed complete mask;
- bandwidth reduction without pre-transmission hard gating;
- information-theoretic optimality from float-vector width;
- exact equivalence to the full original HetNet;
- unique semantic roles from latent-label alignment;
- universal team-size generalization from a size-equivariant actor;
- algorithmic novelty from MAPPO or Sinkhorn alone;
- global convergence of the neural constrained MARL objective.

## 20. Final recommendation

The current thesis has a strong and unusually clear scientific center:

> communication matters when one agent has the information and another has the
> authority to act on it.

SoftRole-HetGAT is a logical next layer:

> can the team infer the computational structure that should connect
> information to authority?

That extension is worth doing only with precise claims. In the current hidden
PCP formulation, it is best understood as capability-cue-conditioned soft typed
computation. Its exact shared posterior and full directed role-pair operator
bank are distinctive, but the surrounding territory is crowded by ROCO, GRDC,
SCoUT, GIRA, SORA, HetNet, typed GNNs, mixed-membership networks, and MoE.

The most defensible near-term contribution is therefore:

1. complete the causal communication audit;
2. implement and mathematically characterize the minimal shared-gate model;
3. establish value with privileged, uniform, random, widened, factorial, and
   same-checkpoint controls;
4. stop if the early oracle-room or learned-gate tests fail.

If a larger research project is desired, the meaningful change is not another
attention equation. It is to let the team **jointly allocate scarce
sensor/action roles and select payload edges under explicit budgets while task
demand changes**. That formulation turns independent classification into
organization, makes networking constraints real, and directly extends the
thesis's sensing-versus-authority insight. It is also a separate paper-scale
project and should be scheduled accordingly.
