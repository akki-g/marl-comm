# SoftRole-HetGAT: Learned Role-Conditioned Communication and Action Policies

*Standalone architecture specification, mathematical formulation, training plan,
and experimental design.*

*Last updated: 2026-07-22.*

> This document is intentionally separate from
> [THESIS_DIRECTION_GUIDE.md](THESIS_DIRECTION_GUIDE.md). The causal audit in
> that guide remains the primary thesis direction. This document specifies a
> possible extension in which agents infer latent roles and use those roles to
> route both communication and action-policy computation.

## 1. Central idea

Fixed-class heterogeneous MARL architectures such as HetNet are given an agent's
class and use that class to select:

- a class-specific observation encoder;
- a class-specific action policy;
- a sender-class-to-receiver-class communication transformation;
- class-aware attention and aggregation parameters.

SoftRole-HetGAT replaces the supplied class label with a learned role posterior.
Each agent infers a probability distribution over latent roles and uses that
same distribution to blend:

1. role-specific action-policy experts;
2. directed sender-role-to-receiver-role message experts;
3. intra-role and inter-role attention preferences.

The key object is therefore not a hard class identifier but

\[
p_{t,i}
=
q_\phi(z_{t,i}\mid \tau_{t,i},\mathcal N_{t,i})
\in \Delta^{K-1},
\]

where:

- \(i\) is an agent;
- \(t\) is a timestep;
- \(K\) is the maximum number of latent roles;
- \(\tau_{t,i}\) is the agent's local observation-action history;
- \(\mathcal N_{t,i}\) is optional locally available graph context;
- \(p_{t,i,r}\) is the inferred probability that agent \(i\) occupies role \(r\).

For example, with two roles:

\[
p_{t,i}=[0.85,\;0.15]
\]

means that the agent currently behaves mostly like role 0 while retaining a
small contribution from role 1.

The architecture is a soft generalization of fixed-class HetNet. When every
\(p_{t,i}\) becomes a supplied one-hot class vector, every mixture below
collapses to the corresponding fixed class or fixed class-pair module.

## 2. What the architecture does and does not assign

The word “role” can mean three different things. They must remain separate.

| Meaning | Learned object | Environment effect | Scope |
|---|---|---|---|
| Computational or behavioral role | Which policy and communication experts process an agent | No change to physical sensors or actions | Core SoftRole-HetGAT |
| Hidden capability role | Which fixed environment capability an agent already has | Environment owns the capability; model infers it | Recommended PCP experiment |
| Endogenous capability assignment | Which agent receives a sensor, tool, or admissible action set | Policy changes environment capability state | Separate hierarchical MARL project |

SoftRole-HetGAT directly learns computational roles. It can infer hidden
detector/capturer capabilities when those capabilities are randomized and
withheld from the policy. It does not create physical sensors or capture
authority unless the environment explicitly treats role allocation as a
high-level action.

This distinction matters for causal interpretation. A policy may infer that an
agent is detector-like and route detector-like messages without having the
authority to change that agent's physical observation function.

## 3. Relationship to fixed-class HetNet

HetNet represents heterogeneous agents as typed graph nodes and uses ordered
class pairs as edge types. For two fixed classes, its communication relations
have the form:

\[
0\leftarrow0,\quad
0\leftarrow1,\quad
1\leftarrow0,\quad
1\leftarrow1.
\]

The first index denotes the receiver and the second denotes the sender. Diagonal
relations are intra-class; off-diagonal relations are inter-class.

HetNet is supplied the class label \(c_i\). A class-pair message transformation
is therefore selected as:

\[
m_{ij}=M_{c_i\leftarrow c_j}(h_j).
\]

SoftRole-HetGAT replaces that hard selection with its expectation under the
receiver and sender role posteriors:

\[
m_{ij}
=
\sum_{r=1}^{K}\sum_{s=1}^{K}
p_{i,r}p_{j,s}
M_{r\leftarrow s}(h_j).
\]

If \(p_i=e_{c_i}\) and \(p_j=e_{c_j}\), where \(e_c\) is a one-hot basis vector,
then every term except \(r=c_i,s=c_j\) vanishes:

\[
m_{ij}=M_{c_i\leftarrow c_j}(h_j).
\]

The fixed-class architecture is therefore a limiting case of the proposed
soft-role architecture.

## 4. Formal problem formulation

### 4.1 Cooperative partially observable task

Let the cooperative task be a Dec-POMDP:

\[
\mathcal G
=
\langle
\mathcal I,\mathcal S,
\{\mathcal O_i\}_{i\in\mathcal I},
\{\mathcal A_i\}_{i\in\mathcal I},
P,R,\gamma
\rangle.
\]

- \(\mathcal I=\{1,\ldots,N\}\) is the agent set.
- \(s_t\in\mathcal S\) is the environment state.
- \(o_{t,i}=O_i(s_t)\) is agent \(i\)'s local observation.
- \(a_{t,i}\in\mathcal A_i\) is its action.
- \(P(s_{t+1}\mid s_t,\mathbf a_t)\) is the transition function.
- \(R(s_t,\mathbf a_t)\) is the shared team reward.
- \(\gamma\in[0,1)\) is the discount factor.

The decentralized actor may use only information available through its local
history and allowed communication edges. A centralized critic may use the
privileged state during training.

### 4.2 Candidate interaction graph

At time \(t\), agents are nodes in a candidate graph:

\[
G_t^0=(V,E_t^0,A_t^0),
\qquad V=\mathcal I.
\]

\(A_{t,ij}^0\in\{0,1\}\) says whether receiver \(i\) is physically or
administratively permitted to receive from sender \(j\). This graph may be:

- fully connected off-diagonal;
- spatial k-nearest-neighbor;
- radius limited;
- a supplied regular graph;
- another externally valid candidate topology.

SoftRole-HetGAT does not need to invent physical connectivity. It learns how to
weight and transform communication over the permitted edges.

### 4.3 Shared local encoder

Each agent uses one shared encoder:

\[
h_{t,i}^{0}=f_{\mathrm{enc}}(\tau_{t,i};\theta_{\mathrm{enc}})
\in\mathbb R^D.
\]

For the current feedforward repository,
\(\tau_{t,i}\) is simply \(o_{t,i}\). A recurrent extension could replace the
MLP with a GRU and use the complete local history.

The shared encoder preserves permutation equivariance and prevents agent
identity from automatically implying a separate policy network.

### 4.4 Symmetry limitation

A bare regular graph does not by itself produce different roles.

Let \(F\) be a permutation-equivariant graph network and \(P\) a permutation
matrix:

\[
F(PX,PAP^\top)=PF(X,A).
\]

If \(P\) is an automorphism of the graph and node features are identical under
that permutation:

\[
PX=X,\qquad PAP^\top=A,
\]

then:

\[
F(X,A)=PF(X,A).
\]

Nodes in the same automorphism orbit receive identical outputs. Consequently, a
deterministic shared role selector assigns them identical role probabilities.

Role differentiation therefore requires at least one symmetry-breaking source:

- different local observations;
- positions or histories;
- hidden physical capabilities that affect observations;
- persistent private random variables;
- stochastic role sampling;
- or an explicit constrained assignment mechanism.

In PCP, positions and target-visibility information already break symmetry. In
an exactly symmetric synthetic graph, the architecture cannot deterministically
invent a unique partition without additional information.

## 5. Architecture overview

The recommended computation is:

~~~text
local observation or history
          |
          v
shared encoder: h0
          |
          +-----------------------+
          |                       |
          v                       |
optional untyped negotiation      |
          |                       |
          v                       |
role router: p(role)              |
          |                       |
          +----------+------------+
                     |
          +----------+-----------+
          |                      |
          v                      v
soft role embedding      role-pair routing weights
          |                      |
          +----------+-----------+
                     |
                     v
role-conditioned graph attention
                     |
                     v
post-communication representation
                     |
                     v
role-conditioned action experts
                     |
                     v
mixture action distribution
~~~

Let \(B_{\mathrm{env}}\) denote the number of parallel environments, \(T\) the
rollout length, \(N\) the number of agents, \(K\) the number of latent roles,
\(D\) the hidden size, \(H\) the number of attention heads, and
\(d_h=D/H\) the per-head size. Require \(D\bmod H=0\). For a minibatch after
flattening any time and environment axes, use:

| Symbol | Shape | Meaning |
|---|---|---|
| \(H^0\) | \(M\times N\times D\) | Pre-communication agent embeddings |
| \(P\) | \(M\times N\times K\) | Soft role posteriors |
| \(E\) | \(K\times D\) | Role embedding table |
| \(G\) | \(K\times K\) or \(H\times K\times K\) | Directed role-pair attention bias |
| \(A^0\) | \(M\times N\times N\) | Candidate communication mask |
| \(\alpha\) | \(M\times N\times N\times H\) | Multi-head attention coefficients |
| \(L\) | \(M\times N\times K\times |\mathcal A|\) | Role-expert action logits |

\(M=B_{\mathrm{env}}\) for ordinary inference and
\(M=T B_{\mathrm{env}}\) after flattening a rollout minibatch. Flatten \(P\)
and \(A^0\) along exactly the same leading dimensions as \(H^0\); a static
\([N,K]\) role tensor cannot represent different assignments in parallel
environments.

## 6. Role inference

### 6.1 Local pre-communication inference

The lowest-risk version infers roles from the pre-communication local
embedding:

\[
u_{t,i}=f_{\mathrm{role}}(h_{t,i}^{0};\phi)\in\mathbb R^K,
\]

\[
p_{t,i,r}
=
\frac{\exp(u_{t,i,r}/\tau)}
{\sum_{k=1}^{K}\exp(u_{t,i,k}/\tau)}.
\]

A minimal shared router is:

\[
u_{t,i}
=
W_2\,\sigma(W_1h_{t,i}^0+b_1)+b_2.
\]

Initialize the final logits near zero so that the initial posterior is close to
uniform, but initialize the downstream expert heads with small independent
perturbations. If every expert is exactly identical, the action-mixture path
provides no signal telling the router which role to favor.

\(\tau>0\) is a temperature:

- large \(\tau\) produces softer mixtures;
- small \(\tau\) approaches a one-hot assignment;
- \(\tau\to0\) approximates hard expert selection.

This version is fully decentralized because each agent computes its role from
its own locally available embedding. It also avoids a circular dependency
between role inference and role-conditioned communication.

### 6.2 Negotiated inference

If roles depend on the current team configuration rather than only local
capability cues, use a role-agnostic negotiation round:

\[
\tilde h_{t,i}
=
\operatorname{Comm}_{\mathrm{base}}
\left(h_{t,i}^{0},\{h_{t,j}^{0}:j\in\mathcal N_{t,i}\}\right),
\]

\[
p_{t,i}
=
q_\phi(z_{t,i}\mid h_{t,i}^{0},\tilde h_{t,i},p_{t-1,i}).
\]

The task communication pass then uses \(p_{t,i}\):

\[
h_{t,i}^{1}
=
\operatorname{Comm}_{\mathrm{role}}
\left(h_{t,i}^{0},P_t,A_t^0\right).
\]

This is a genuine two-stage protocol. It costs an additional communication
round and must be compared with a parameter- and round-matched baseline. It
would be invalid to infer a role from the output of a role-conditioned pass and
claim that the same pass used the newly inferred role without an explicit
fixed-point or iterative construction.

### 6.3 How neighbors know one another's roles

To compute a sender-role-to-receiver-role transformation in decentralized
execution:

- receiver \(i\) knows its own \(p_i\);
- sender \(j\) computes its own \(p_j\);
- sender \(j\) transmits \(p_j\), a compressed role token, or a role-conditioned
  message from which the relation is implemented.

Accessing every \(p_j\) through a centralized tensor is convenient in a
vectorized simulator, but the deployed communication protocol must account for
how the receiver obtains the sender role information. A soft role token costs
\(K\) real values per role update; a hardened token costs approximately
\(\lceil\log_2 K\rceil\) bits.

### 6.4 Fixed versus learned number of roles

The first implementation should fix \(K=2\). Automatically learning both the
partition and the number of roles adds a separate model-selection problem.

Later, one can choose \(K_{\max}\) and encourage unused prototypes to disappear
through sparsity, prototype pruning, or a nonparametric prior. A useful
diagnostic is the effective number of population roles:

\[
K_{\mathrm{eff}}
=
\exp\left(
H\left(\bar p\right)
\right),
\qquad
\bar p=\frac{1}{N}\sum_i p_i.
\]

The number of roles should depend on task structure, not mechanically on the
number of agents. Ten agents may still require only two functional roles.

## 7. Soft role conditioning

### 7.1 Role embedding

The current hard class embedding \(E[c_i]\) becomes:

\[
e_i
=
\sum_{r=1}^{K}p_{i,r}E_r
=
p_i^\top E.
\]

The role-conditioned node representation is:

\[
x_i=h_i^0+e_i.
\]

With batch axes:

~~~python
role_emb = torch.einsum("mnr,rd->mnd", role_probs, role_embeddings)
x = h0 + role_emb
~~~

When \(p_i=e_{c_i}\), this reduces to the hard embedding lookup
\(E[c_i]\).

### 7.2 Directed role-pair attention bias

Let \(G\in\mathbb R^{K\times K}\), with \(G_{rs}\) representing the prior
preference for receiver role \(r\) to attend to sender role \(s\). Define:

\[
\beta_{ij}=p_i^\top Gp_j
=
\sum_{r,s}p_{i,r}G_{rs}p_{j,s}.
\]

In tensor form:

~~~python
pair_bias = torch.einsum(
    "mir,rs,mjs->mij", role_probs, role_pair_bias, role_probs
)
~~~

For multi-head attention, either share \(G\) across heads or learn
\(G^{(h)}\) per head:

\[
\beta_{ij}^{(h)}=p_i^\top G^{(h)}p_j.
\]

Diagonal entries govern intra-role attention. Off-diagonal entries govern
directed inter-role attention. In PCP, a large capturer-receiver,
detector-sender entry corresponds to the expected detector-to-capturer route.

### 7.3 Role-pair message experts

Let:

\[
M_{r\leftarrow s}:\mathbb R^D\rightarrow\mathbb R^{D_v}
\]

be the message expert used when the receiver occupies role \(r\) and the sender
occupies role \(s\). The soft directed message is:

\[
v_{ij}
=
\sum_{r=1}^{K}\sum_{s=1}^{K}
p_{i,r}p_{j,s}
M_{r\leftarrow s}(x_j).
\]

This formulation lets each agent continuously weigh all intra-role and
inter-role communication policies. The gate weight on expert
\(M_{r\leftarrow s}\) is:

\[
g_{ij}^{rs}=p_{i,r}p_{j,s}.
\]

This outer product is a mean-field routing choice: the edge gate uses the
factorized endpoint posterior \(q(r_i)q(r_j)\), not a separately learned joint
posterior \(q(r_i,r_j)\). The endpoints can still be statistically dependent
through shared observations or a negotiation round, but the expert mixer is
conditioned on their two marginal vectors. A joint edge router would be more
expressive and would also add another latent object whose meaning must be
identified.

For \(K=2\), the four gate weights sum to one:

\[
\sum_{r,s}g_{ij}^{rs}
=
\left(\sum_r p_{i,r}\right)
\left(\sum_s p_{j,s}\right)
=1.
\]

The full expert can be a matrix:

\[
M_{r\leftarrow s}(x_j)
=
W_{r\leftarrow s}^{V}x_j+b_{r\leftarrow s}^{V}.
\]

This costs \(K^2D_vD\) weight parameters. With \(K=2,D=64\), the bank is
manageable, but capacity-matched controls remain essential.

### 7.4 Lower-risk value-only specialization

The recommended first version keeps shared query/key projections and
specializes only message values:

\[
q_i=W_Qx_i,\qquad k_j=W_Kx_j,
\]

\[
v_{ij}
=
\sum_{r,s}p_{i,r}p_{j,s}W_{r\leftarrow s}^{V}x_j.
\]

This preserves one common compatibility space while allowing different role
pairs to translate content differently. It is easier to diagnose than giving
every role and relation separate query, key, value, and fusion networks.

### 7.5 Full role-specific projections

A more expressive extension uses role-conditioned query and key banks:

\[
q_i=\sum_r p_{i,r}W_Q^r x_i,
\qquad
k_j=\sum_s p_{j,s}W_K^s x_j.
\]

Combined with relation-specific values, this approaches a complete soft
heterogeneous graph-attention network. It also multiplies capacity and makes it
harder to tell whether gains come from role structure or additional parameters.
It should not be the first experiment.

### 7.6 Low-rank or hypernetwork alternative

For larger \(K\), avoid \(K^2\) full matrices. Generate a modulation from role
embeddings:

\[
W_{r\leftarrow s}^{V}
=
W_0
+
U\,\operatorname{diag}
\left(g(e_r,e_s)\right)V^\top,
\]

or use FiLM:

\[
M_{r\leftarrow s}(x)
=
\gamma_{rs}\odot W_0x+\delta_{rs}.
\]

The role-pair hypernetwork shares most parameters and lets continuous role
embeddings generalize, but it adds another modeling choice. Full \(K=2\)
experts are simpler for the proposed PCP experiment.

## 8. Role-conditioned graph attention

For head \(h\), define:

\[
e_{ij}^{(h)}
=
\frac{
\left(q_i^{(h)}\right)^\top k_j^{(h)}
}{\sqrt{d_h}}
+
\beta_{ij}^{(h)}.
\]

Apply the candidate graph mask before normalizing over senders:

\[
\alpha_{ij}^{(h)}
=
\frac{
A_{ij}^0\exp(e_{ij}^{(h)})
}{
\sum_{\ell}
A_{i\ell}^0\exp(e_{i\ell}^{(h)})
}.
\]

The message aggregation is:

\[
m_i^{(h)}
=
\sum_{j}
\alpha_{ij}^{(h)}
v_{ij}^{(h)}.
\]

Concatenate heads and fuse with the receiver's representation:

\[
m_i=\operatorname{Concat}_{h=1}^{H}m_i^{(h)},
\]

\[
y_i
=
\operatorname{LayerNorm}
\left(
x_i+W_Om_i
\right).
\]

The existing repository uses either concatenation-and-projection or residual
fusion depending on the module. Either is compatible. The experiment should
fix one fusion rule across oracle, learned, and uniform role conditions.

The attention coefficient answers “which sender matters?” The relation expert
answers “how should this sender's representation be translated for this
receiver?” These are distinct functions and should be logged separately.

There is one important mathematical boundary. The soft pair score
\(p_i^\top Gp_j\) is the expectation of a hard role-pair bias under the
factorized role posterior, and the soft value \(v_{ij}\) is the corresponding
expert expectation. Because softmax and later neural layers are nonlinear,

\[
\operatorname{softmax}\!\left(\mathbb E[e]\right)
\neq
\mathbb E\!\left[\operatorname{softmax}(e)\right]
\]

in general. SoftRole-HetGAT is therefore a differentiable relaxation of the
hard-role graph, not exact Bayesian marginalization over all hard role-labeled
graphs. Likewise,

\[
f\!\left(\sum_r p_rE_r\right)
\neq
\sum_rp_rf(E_r)
\]

for nonlinear \(f\). A soft role embedding and a true mixture of experts are
different operations; this guide uses each deliberately.

## 9. Role-conditioned action mixture

### 9.1 Expert policies

Let \(g_{\mathrm{actor}}\) be a shared action trunk:

\[
a_i^{\mathrm{feat}}=g_{\mathrm{actor}}(y_i).
\]

Each role has a small expert head:

\[
\ell_i^r=W_\pi^r a_i^{\mathrm{feat}}+b_\pi^r
\in\mathbb R^{|\mathcal A|},
\]

\[
\pi_r(a\mid y_i)
=
\operatorname{softmax}(\ell_i^r)_a.
\]

The final action distribution is a true mixture:

\[
\pi(a\mid y_i,p_i)
=
\sum_{r=1}^{K}
p_{i,r}\pi_r(a\mid y_i).
\]

Implementation:

~~~python
expert_logits = role_heads(actor_features)       # [M, N, K, A]
expert_probs = expert_logits.softmax(dim=-1)     # [M, N, K, A]
action_probs = (
    role_probs.unsqueeze(-1) * expert_probs
).sum(dim=-2)                                    # [M, N, A]
dist = torch.distributions.Categorical(probs=action_probs)
~~~

For PPO likelihoods, evaluate the same mixture in log space:

~~~python
log_gate = torch.log_softmax(role_logits / temperature, dim=-1)
log_expert = torch.log_softmax(expert_logits, dim=-1)
log_action = torch.logsumexp(
    log_gate.unsqueeze(-1) + log_expert,
    dim=-2,
)                                                   # [M, N, A]

# actions: [M, N]
selected = log_expert.gather(
    -1,
    actions[..., None, None].expand(-1, -1, K, 1),
).squeeze(-1)                                       # [M, N, K]
executed_log_prob = torch.logsumexp(
    log_gate + selected,
    dim=-1,
)                                                   # [M, N]
~~~

This avoids taking \(\log\) after probabilities have underflowed and ensures
that the stored likelihood is the likelihood of the distribution that actually
sampled the action. The first line applies to learned gates. In oracle or
uniform mode, construct **log_gate** from the actual overridden role
probabilities, using negative infinity for exact zero entries; never evaluate
an oracle- or uniform-routed action under the unused learned gate.

Averaging logits and then applying softmax defines a valid gated policy, but it
is not the same mathematical object as a mixture of role policies. The
probability mixture above most directly matches the claim that the agent weighs
role-specific policies.

### 9.2 Mixture-policy gradient

For the executed action \(a\), define the posterior responsibility of expert
\(r\):

\[
\rho_r(a)
=
\frac{
p_r\pi_r(a)
}{
\sum_k p_k\pi_k(a)
}.
\]

Then:

\[
\nabla\log\pi(a)
=
\sum_r
\rho_r(a)
\nabla\log\left[p_r\pi_r(a)\right].
\]

This separates into role-router and expert gradients:

\[
\nabla\log\pi(a)
=
\sum_r\rho_r(a)\nabla\log p_r
+
\sum_r\rho_r(a)\nabla\log\pi_r(a).
\]

If role logits are \(u\) and \(p=\operatorname{softmax}(u)\), then:

\[
\frac{\partial\log\pi(a)}{\partial u_r}
=
\rho_r(a)-p_r.
\]

This is the **direct** derivative through the final mixture while holding the
expert distributions fixed. In the complete architecture, \(p\) also changes
role embeddings, communication biases, and message values; automatic
differentiation adds those indirect paths. The direct equation still explains
the action gate: an expert receives greater role probability when it assigns
relatively high probability to actions that obtain positive advantage. If all
expert policies are identical, then \(\rho_r=p_r\), this direct gradient is
zero, and the router needs asymmetry or a useful communication-mediated
gradient to specialize.

### 9.3 Hard-role limit

If \(p_i=e_{c_i}\), then:

\[
\pi(a\mid y_i,p_i)=\pi_{c_i}(a\mid y_i).
\]

The action mixture collapses exactly to a fixed class-specific action policy.

### 9.4 Different action sets

If roles have different admissible actions, define a union action set
\(\mathcal A=\cup_r\mathcal A_r\) and role-dependent masks
\(\mu_r(a)\in\{0,1\}\). The expert becomes:

\[
\pi_r(a)
\propto
\mu_r(a)\exp(\ell_r(a)).
\]

For soft roles, an action is feasible only under a clearly defined environment
rule. Blending incompatible physical action masks is ambiguous. The first
SoftRole-HetGAT experiment should retain the current common five-action movement
space and treat roles as computational specializations, not as changing
physical action authority.

## 10. Critic and CTDE structure

The clean MAPPO design keeps the critic role-agnostic:

\[
V_\psi(s_t).
\]

The actor path is:

\[
o_i
\rightarrow
h_i^0
\rightarrow
p_i
\rightarrow
\operatorname{RoleComm}
\rightarrow
\operatorname{RoleActor}
\rightarrow
a_i.
\]

The critic path remains:

\[
s_t\rightarrow V_\psi(s_t).
\]

This preserves the current CTDE separation and prevents a privileged critic
state from becoming an execution-time role signal. The critic may help train the
actor through advantages, but it does not supply role labels or messages during
execution.

A class- or role-conditioned critic is possible but unnecessary for the first
study. It would introduce an additional pathway through which inferred roles
affect optimization and make the actor-side mechanism harder to isolate.

## 11. MAPPO training formulation

### 11.1 Deterministic soft roles

The recommended first implementation makes \(p_i\) a deterministic,
differentiable function of the stored observation:

\[
p_i=q_\phi(h_i^0).
\]

No role sample is added to the environment action. During rollout, store the
movement action and its log probability under the complete mixture:

\[
\log\pi_{\theta_{\mathrm{old}},\phi_{\mathrm{old}}}
(a_{t,i}\mid o_{t,i}).
\]

During the PPO update, recompute:

- the encoder output;
- the current role posterior;
- role-conditioned communication;
- every action expert;
- the final action-mixture probability.

The PPO ratio remains:

\[
r_{t,i}(\theta,\phi)
=
\exp\left[
\log\pi_{\theta,\phi}(a_{t,i}\mid o_{t,i})
-
\log\pi_{\theta_{\mathrm{old}},\phi_{\mathrm{old}}}
(a_{t,i}\mid o_{t,i})
\right].
\]

The clipped actor objective is:

\[
L_{\mathrm{clip}}
=
-
\mathbb E_{t,i}
\left[
\min
\left(
r_{t,i}\hat A_{t,i},
\operatorname{clip}(r_{t,i},1-\epsilon,1+\epsilon)
\hat A_{t,i}
\right)
\right].
\]

Because the new mixture probability is differentiated end to end, gradients
flow into:

- the shared encoder;
- the role router;
- role embeddings and relation biases;
- role-pair message experts;
- attention and fusion layers;
- action-policy experts.

The old mixture log probability is sufficient for the PPO ratio. The old role
posterior does not need to be stored when roles are deterministic functions of
the rollout observation. It should still be logged for analysis.

Collection may run under **torch.no_grad()**, but the PPO **evaluate()** path
must recompute \(p\) without detaching the router, communication output, or
expert probabilities. The router and all role-conditioned modules must be
included in the actor parameter group so the existing actor-gradient clipping
covers them. Before the first optimizer step on a rollout, verify that the
recomputed old-versus-new mean ratio is approximately one; this catches
mismatched gates, masks, temperatures, or likelihood definitions.

### 11.2 Mixture entropy

The exploration entropy must be computed from the final action distribution:

\[
H[\pi_i]
=
-
\sum_a
\pi_i(a)\log\pi_i(a).
\]

It is not generally equal to:

\[
\sum_r p_{i,r}H[\pi_r].
\]

If \(R\sim p_i\) selects an expert and \(A\sim\pi_R\), then:

\[
H(A)
=
H(A\mid R)+I(A;R)
=
\sum_rp_{i,r}H[\pi_r]+I(A;R).
\]

The weighted expert entropy omits the nonnegative mutual-information term and
should not replace the actual mixture-policy entropy. Log action entropy and
role-gate entropy separately because they measure different uncertainties.

### 11.3 Total optimization objective

One possible minimized loss is:

\[
L_{\mathrm{total}}
=
L_{\mathrm{clip}}
+
c_VL_V
-
c_HH[\pi]
+
\lambda_{\mathrm{sharp}}L_{\mathrm{sharp}}
+
\lambda_{\mathrm{load}}L_{\mathrm{load}}
+
\lambda_{\mathrm{switch}}L_{\mathrm{switch}}
+
\lambda_{\mathrm{aux}}L_{\mathrm{aux}}.
\]

\(L_V\) is the ordinary critic loss. The role terms are described below. The
first experiment should use as few auxiliary terms as possible and fix their
coefficients before the full run.

## 12. Role and expert regularization

### 12.1 Assignment sharpness

To encourage interpretable assignments:

\[
L_{\mathrm{sharp}}
=
\frac{1}{MN}
\sum_{m,i}
H(p_{m,i})
=
-
\frac{1}{MN}
\sum_{m,i,r}
p_{m,i,r}\log p_{m,i,r}.
\]

Minimizing this term encourages each agent to place most probability on a small
number of roles.

Strong sharpness too early can lock random initial assignments into place.
Apply it only after a warm-up or increase it gradually while decreasing
temperature.

### 12.2 Population usage or load balancing

Define the mean role usage per environment:

\[
\bar p_m
=
\frac{1}{N}
\sum_i p_{m,i}.
\]

A collapse-prevention loss is:

\[
L_{\mathrm{load}}
=
\frac{1}{M}
\sum_m
D_{\mathrm{KL}}
\left(
\bar p_m\;\|\;\rho
\right),
\]

where \(\rho\) is a target usage prior.

For two equally expected roles, \(\rho=[0.5,0.5]\). For randomized PCP with
three detectors and two capturers, one could use \([0.6,0.4]\), but doing so
injects knowledge of the hidden class count and weakens the unsupervised claim.
It also arbitrarily names the larger latent role unless the loss is made
permutation invariant, for example:

\[
L_{\mathrm{load}}^{\mathrm{perm}}
=
\min_{\sigma\in S_K}
D_{\mathrm{KL}}\!\left(\bar p_m\;\|\;\rho_\sigma\right),
\]

where \(\rho_\sigma\) ranges over permutations of the occupancy prior.

The safest use is a weak, temporary anti-collapse term rather than a hard
requirement that every task use every role equally. A communication-unnecessary
task may legitimately collapse to one effective role.

### 12.3 Temporal consistency

If roles should remain stable:

\[
L_{\mathrm{switch}}
=
\frac{1}{(T-1)BN}
\sum_{t>0,b,i}
D_{\mathrm{KL}}
\left(
\operatorname{sg}[p_{t-1,b,i}]
\;\|\;
p_{t,b,i}
\right),
\]

where \(\operatorname{sg}\) stops gradients through the previous posterior.

The current PPO minibatches flatten time. Training this loss correctly requires
preserving adjacent observations or calculating it on ordered rollout
sequences. Logging role-switch rate does not require this change and should
precede adding the loss.

### 12.4 Expert diversity

If experts remain identical, a possible diagnostic is pairwise
Jensen-Shannon divergence:

\[
D_{\mathrm{JS}}(\pi_r,\pi_s)
=
\frac{1}{2}D_{\mathrm{KL}}(\pi_r\|m)
+
\frac{1}{2}D_{\mathrm{KL}}(\pi_s\|m),
\quad
m=\frac{\pi_r+\pi_s}{2}.
\]

One could maximize this value on visited states. However, forcing expert
difference can create arbitrary specialization unrelated to reward. Expert
diversity should first be measured rather than directly optimized.

Initialize expert heads from a common base plus small independent noise. This
preserves comparable initial behavior while providing enough asymmetry for the
mixture gate to receive nonzero gradients.

### 12.5 No supervised role loss in the primary condition

The learned-role condition should not minimize cross-entropy to hidden
detector/capturer labels. Those labels are reserved for evaluation. A
supervised role classifier is a useful engineering baseline, but it answers a
different question.

## 13. Hard or stochastic role selection

### 13.1 Why direct argmax is insufficient

\[
z_i=\operatorname{argmax}_r p_{i,r}
\]

has zero gradient almost everywhere. Using this operation during ordinary
backpropagation does not train the router.

### 13.2 Straight-through Gumbel-Softmax

A hard differentiable approximation is:

\[
\tilde z_{i,r}
=
\frac{
\exp((u_{i,r}+g_{i,r})/\tau)
}{
\sum_k\exp((u_{i,k}+g_{i,k})/\tau)
},
\quad
g_{i,r}\sim\operatorname{Gumbel}(0,1),
\]

with a one-hot forward value and soft backward gradient.

This estimator is biased and stochastic. If a sampled role changes the action
distribution, resampling different Gumbel noise during the PPO update makes the
new and old likelihoods refer to different latent decisions.

### 13.3 Treating role as a policy action

For a mathematically explicit hard-role policy:

\[
z_i\sim q_\phi(z_i\mid o_i),
\]

\[
a_i\sim\pi_\theta(a_i\mid o_i,z_i).
\]

The joint likelihood is:

\[
\pi_{\theta,\phi}(z_i,a_i\mid o_i)
=
q_\phi(z_i\mid o_i)
\pi_\theta(a_i\mid o_i,z_i).
\]

The stored log probability must be:

\[
\log q_\phi(z_i\mid o_i)
+
\log\pi_\theta(a_i\mid o_i,z_i).
\]

The role sample and both old likelihood components must be stored. PPO then
clips the joint probability ratio. This single-agent expression is sufficient
only when \(z_i\) gates agent \(i\)'s own action expert and has no effect on
other agents.

When sampled roles condition communication, one sampled role can change several
agents' action distributions. The latent decision is then the team role vector:

\[
q_\phi(\mathbf z\mid\mathbf o)
=
\prod_iq_\phi(z_i\mid o_i),
\]

\[
\pi_{\theta,\phi}(\mathbf z,\mathbf a\mid\mathbf o)
=
q_\phi(\mathbf z\mid\mathbf o)
\prod_i
\pi_\theta
\left(
a_i\mid o_i,\operatorname{Comm}(\mathbf o,\mathbf z)
\right).
\]

Its log likelihood is:

\[
\sum_i\log q_\phi(z_i\mid o_i)
+
\sum_i\log\pi_\theta
\left(
a_i\mid o_i,\operatorname{Comm}(\mathbf o,\mathbf z)
\right).
\]

The safest PPO treatment uses the same stored team role vector and a team-level
joint ratio. Independently clipping each agent's role-action ratio is not an
exact factorization once a role affects teammates through communication. This
is a substantially larger algorithmic change and another reason to begin with
deterministic soft routing.

For this reason, deterministic soft roles are strongly preferred for the first
study. Harden them only for diagnostic evaluation after the soft model works.

## 14. QMIX extension

QMIX is not part of the recommended first implementation, but the corresponding
agent utility is:

\[
Q_i(a)
=
\sum_r
p_{i,r}Q_i^r(a).
\]

The shared monotonic mixer consumes the selected per-agent utilities as usual.
The greedy action is selected after mixing the utilities:

\[
a_i^\star
=
\arg\max_a
\sum_rp_{i,r}Q_i^r(a),
\]

not by mixing the separate experts' greedy actions. For a Double-Q target, let
the online network—including its router and communication experts—select each
component \(a_i^\star\) by this equation and define:

\[
\mathbf a^\star
=
(a_1^\star,\ldots,a_N^\star).
\]

The target network then evaluates those selected actions:

\[
y
=
r
+
\gamma(1-d)\,
Q_{\mathrm{tot,target}}
\left(
\mathbf Q_{\mathrm{target}}
(\mathbf o',\mathbf a^\star;\mathbf p'_{\mathrm{target}}),
s'
\right).
\]

In implementation:

1. the online role router and online experts select the greedy next action;
2. the target role router and target experts evaluate that selected action;
3. router, communication experts, and Q heads are synchronized together during
   target updates.

Deterministic roles can be recomputed from replayed observations. Stochastic or
persistent role samples must be stored in replay. The current fixed-\(N\) mixer
and replay structure still prevent population-size generalization without
additional work.

Auxiliary role losses belong only to the online network; target role tensors
must remain gradient-free. Soft expert mixing preserves QMIX monotonicity with
respect to the final per-agent utility supplied to the mixer, but does not make
the mixer monotonic in each internal expert utility independently.

The MAPPO study should pass its causal and expert-utilization gates before any
QMIX replication is considered.

## 15. Randomized hidden-role PCP

### 15.1 Environment assignment

For each parallel environment \(b\) at reset, sample:

\[
c_b\in\{0,1\}^N
\]

uniformly from assignments containing exactly three detectors and two
capturers. The assignment is fixed within that episode and resampled at the
next reset.

The environment uses \(c_{b,i}\) internally for:

- target observation masking;
- detector-event computation;
- capture eligibility;
- role-dependent diagnostics.

The actor, role router, and communication module do not receive \(c_{b,i}\).
Agent ordering is randomized so an index cannot identify the hidden role.

### 15.2 Required tensor change

The current environment class identifier has shape \([N]\). Randomized
parallel episodes require:

\[
C\in\{0,1\}^{B\times N}.
\]

Observation and reward code must use per-environment Boolean masks rather than
slicing the same fixed agent indices for every batch element.

For oracle experiments, the hidden \(C\) is passed only to the gate interface.
For learned and uniform experiments, it remains environment-private.

### 15.3 Central critic state

Randomized capability assignment changes the reward and observation functions.
The privileged critic state may include the hidden assignment, provided:

- it is included identically for oracle, learned, and uniform conditions;
- it is unavailable to the actor and role router;
- the state definition is documented as training-only.

Alternatively, omit it for all conditions. The choice must be fixed before
comparison.

### 15.4 What the task demonstrates

At blind PCP, the target-visibility field strongly indicates whether an agent is
a detector or capturer. Therefore, success supports:

> The network learned to convert local capability cues into coordinated
> computational specialization and directed information routing.

It does not support:

> Roles emerged without observable evidence or prior task structure.

At \(\alpha=1\), the two environment classes may be locally indistinguishable
to a feedforward router even though capture authority remains different. Do not
use that setting for the first hidden-role experiment unless a capability cue,
history-based inference mechanism, or explicit role-negotiation problem is
defined.

## 16. Experimental conditions

### 16.1 Core trained conditions

All three primary conditions instantiate the same action heads, message
experts, attention module, and nominal parameter count.

#### Oracle hard gate

\[
p_i=e_{c_i}.
\]

The ground-truth hidden environment role selects the mixture components. This
is the architectural upper bound.

#### Learned soft gate

\[
p_i=q_\phi(h_i^0).
\]

No role label is supplied and no supervised role loss is used.

#### Uniform gate

\[
p_i=\left[\frac{1}{K},\ldots,\frac{1}{K}\right].
\]

All experts remain present, but role information cannot route computation. This
is the primary capacity-matched null.

These conditions are nominally parameter matched but not gradient-path matched:
the fixed oracle and uniform gates do not train a useful role router. This
limitation should be stated. Same-checkpoint gate interventions complement the
between-training comparison.

### 16.2 Repository reference

The existing shared attention actor without latent-role experts is a useful
reference. It is not the primary capacity-matched control because it contains
fewer specialized modules.

### 16.3 Inference-time shuffled gate

For a learned checkpoint, permute role posteriors across agents or parallel
environments:

\[
\tilde p_{b,i}=p_{\sigma(b,i)}.
\]

Choose a permutation that preserves the marginal distribution of gate values
while breaking their correspondence to the current agent capability. This tests
whether the semantic placement of roles matters, rather than only their global
distribution.

### 16.4 Component factorial

After the core experiment succeeds, separate action specialization and
communication specialization:

| Action expert gate | Communication expert gate | Interpretation |
|---|---|---|
| Uniform | Uniform | Capacity-matched null |
| Learned | Uniform | Action specialization only |
| Uniform | Learned | Communication specialization only |
| Learned | Learned | Full SoftRole-HetGAT |

All experts remain instantiated in every cell. “Uniform” averages them rather
than removing parameters.

A factorial interaction can be estimated as:

\[
I_{\mathrm{A\times C}}
=
\left(
Y_{\mathrm{learnedA,learnedC}}
-
Y_{\mathrm{uniformA,learnedC}}
\right)
-
\left(
Y_{\mathrm{learnedA,uniformC}}
-
Y_{\mathrm{uniformA,uniformC}}
\right).
\]

A positive \(I_{\mathrm{A\times C}}\) supports synergy. If only one component
matters, credit that component rather than the full architecture.

## 17. Hypotheses and estimands

### H1 — specialization value

\[
\Delta_{\mathrm{role}}
=
Y_{\mathrm{learned}}
-
Y_{\mathrm{uniform}}
>0.
\]

Meaningful learned routing improves episodic capture beyond the capacity of an
equally sized uniform mixture.

### H2 — oracle recovery

\[
R_{\mathrm{oracle}}
=
\frac{
Y_{\mathrm{learned}}-Y_{\mathrm{uniform}}
}{
Y_{\mathrm{oracle}}-Y_{\mathrm{uniform}}
}.
\]

Predefine a target such as \(R_{\mathrm{oracle}}\ge0.75\). Do not compute this
ratio when the oracle-minus-uniform denominator is too small to be meaningful.

### H3 — semantic role recovery

On unseen role assignments, inferred roles align with hidden
detector/capturer classes after label-permutation matching.

### H4 — action specialization

Learned action gating improves over uniform action gating while communication
gating is held fixed.

### H5 — communication specialization

Learned communication gating improves over uniform communication gating while
action gating is held fixed.

### H6 — joint specialization

The action-by-communication interaction \(I_{\mathrm{A\times C}}\) is positive.

### H7 — directional mechanism

Blocking detector-sender-to-capturer-receiver flow or the corresponding latent
expert \(M_{\mathrm{C}\leftarrow\mathrm{D}}\) causes a larger performance
decline than blocking reverse or within-role relations.

### H8 — semantic reliance

Distribution-preserving message and gate shuffles reduce same-checkpoint
performance.

### H9 — allocation consistency

Assignments are stable during an episode with fixed capabilities and change
appropriately when the environment resamples capabilities.

## 18. Same-checkpoint causal interventions

Evaluate identical learned checkpoints on common episode seeds under:

1. Correct messages and correct learned role posteriors.
2. Zero teammate message content.
3. Messages shuffled across parallel target states.
4. Role posteriors shuffled across agents or environments.
5. True detector-to-capturer edges blocked.
6. Capturer-to-detector edges blocked.
7. Within-hidden-role edges blocked.
8. Latent \(M_{\mathrm{C}\leftarrow\mathrm{D}}\) expert disabled.
9. Reverse or diagonal latent experts disabled.
10. Action experts swapped.
11. Action mixture forced to uniform.
12. Oracle gates substituted into the learned checkpoint when the interface is
    compatible.

For each intervention, hold constant:

- policy weights;
- environment initial state;
- target state;
- hidden capability assignment;
- evaluation episode seed;
- candidate communication mask.

Distinguish two kinds of directional test:

- hidden-role edge blocking uses ground-truth environment roles only to select
  which communication edges to disable;
- latent-expert blocking disables the role-pair operator selected by the
  learned representation.

Agreement between them supports semantic alignment between latent and physical
roles.

Message shuffling remains necessary even if role shuffling is performed.
Otherwise an action-expert gain could be mistaken for evidence that messages
carry task-relevant content.

## 19. Evaluation and statistics

### 19.1 Primary outcome

Use deterministic episodic capture success:

\[
Y
=
\frac{1}{E}
\sum_{e=1}^{E}
\mathbf 1
\left[
\text{episode }e\text{ contains at least one valid capture}
\right].
\]

This differs from the fraction of timesteps during which a capture condition is
true.

### 19.2 Secondary behavioral outcomes

- time to first valid capture;
- episodic return;
- target-directed action alignment;
- capturer distance reduction after detector communication;
- detection-event rate;
- detection-window-open rate;
- invalid-capture rate, which must remain zero.

### 19.3 Latent-role metrics

Latent role labels are exchangeable. Role 0 in one seed may mean detector while
role 1 means detector in another. Align labels using a mapping learned on a
validation set, then apply that mapping to held-out evaluation data.

Report:

- balanced accuracy after Hungarian matching;
- adjusted mutual information;
- per-role precision and recall;
- confusion matrices;
- gate entropy;
- effective number of used roles;
- role occupancy;
- role-switch rate;
- accuracy after the episode capability assignment changes.

Do not choose the label mapping separately on every test episode; that would
inflate apparent recovery.

### 19.4 Expert and routing metrics

- attention mass by hidden receiver/sender class;
- gate-weighted utilization of every relation expert;
- action-head utilization;
- gradient norm for every expert;
- pairwise action-distribution divergence;
- pairwise message-output divergence;
- performance effect of each expert intervention;
- message-shuffle and gate-shuffle effects.

An expert with nonzero gate mass but zero gradient may be functionally dead. An
expert with gradient but outputs identical to another expert has not developed
meaningful specialization.

### 19.5 Statistical unit

The training seed is the top-level independent unit. Evaluation episodes are
nested within a trained seed.

- use common episode seeds across interventions;
- compute paired seed-level effects;
- show all raw seed points;
- report confidence intervals in natural episodic-success units;
- predefine the oracle-recovery target and any equivalence margin.

## 20. Repository implementation blueprint

### 20.1 Suggested new modules

#### roles/router.py

Responsibilities:

- map pre-communication embeddings to role logits;
- apply the temperature schedule;
- return role probabilities;
- expose entropy and utilization diagnostics;
- optionally support oracle and uniform override modes.

Interface:

~~~python
role_logits, role_probs = router(h_pre, context=None)
~~~

#### comm/soft_role_attention.py

Responsibilities:

- consume \(H^0,P,A^0\);
- apply soft role embeddings;
- compute role-pair biases;
- compute shared Q/K attention;
- mix role-pair value experts;
- return post-communication features and routing diagnostics.

#### backbone/role_actor.py

Responsibilities:

- apply a shared trunk;
- compute \(K\) expert action distributions;
- mix them using role probabilities;
- return the final categorical distribution and expert diagnostics.

### 20.2 MAPPO integration

Update [backbone/mappo.py](backbone/mappo.py):

- construct and register the router and role actor;
- include their parameters in the actor parameter group;
- compute roles before communication;
- return optional diagnostics without altering rollout actions;
- save and load router/expert state and configuration.

Update [backbone/ppo_update.py](backbone/ppo_update.py):

- use final mixture log probabilities;
- compute entropy from the mixture distribution;
- add prespecified role regularizers;
- log router and expert gradients.

Update [backbone/rollout_buffer.py](backbone/rollout_buffer.py) only if needed:

- deterministic per-step roles need no new field for PPO correctness;
- role diagnostics may be stored for analysis;
- temporal regularization requires ordered previous/current observations;
- sampled hard roles require role samples and old role log probabilities.

Update [backbone/factory.py](backbone/factory.py) so checkpoints reconstruct
the router, role count, temperature configuration, action experts, and
communication experts before loading weights. Registering modules is enough to
put their tensors in a state dictionary, but the reconstruction metadata still
has to be saved.

### 20.3 Communication integration

The current [comm/attention.py](comm/attention.py) and
[comm/graph.py](comm/graph.py) use:

- a hard class embedding lookup;
- a hard class-pair bias lookup.

Do not overload the existing environment **class_id** argument with inferred
roles. Physical capability class and latent computational role are different
variables. Add an explicit **role_probs** input.

Extend [comm/base.py](comm/base.py) with that explicit optional argument rather
than changing the meaning of **class_id**. Let
[comm/identity.py](comm/identity.py) accept and ignore **role_probs** so the
same router and action experts can be evaluated in a no-communication control.
If [comm/broadcast.py](comm/broadcast.py) remains a comparison condition, retain
its hard capability path and either add an explicit soft-role variant or state
that it is outside the new architecture; do not silently coerce a dynamic
\([M,N,K]\) posterior into its static \([N]\) class interface.

If both known capabilities and inferred computational roles are retained:

\[
x_i
=
h_i
+
E_{\mathrm{cap}}[c_i]
+
p_i^\top E_{\mathrm{role}},
\]

\[
e_{ij}
=
\operatorname{contentScore}(i,j)
+
G_{\mathrm{cap}}[c_i,c_j]
+
p_i^\top G_{\mathrm{role}}p_j.
\]

The hidden-role experiment intentionally withholds \(c_i\) from the actor.

### 20.4 Environment integration

Update [envs/asymmetry.py](envs/asymmetry.py) and
[envs/pcp.py](envs/pcp.py):

- represent hidden classes as \([B,N]\);
- sample seeded per-episode assignments;
- build observation visibility per environment and agent;
- compute detector/capturer distance masks without fixed index slices;
- preserve the capture-window invariant;
- expose hidden labels only through privileged diagnostic/evaluation channels.

Update [experiments/train_vmas_mappo.py](experiments/train_vmas_mappo.py):

- stop precomputing one global actor class vector for the learned condition;
- construct the role modules from configuration;
- log all role and expert diagnostics;
- preserve the existing physical communication mask.

### 20.5 Configuration

Suggested configuration schema:

~~~yaml
roles:
  enabled: true
  num_roles: 2
  mode: learned          # learned | oracle | uniform
  inference: local       # local | negotiated
  temperature_start: 1.5
  temperature_final: 0.3
  anneal_steps: 1000000
  sharp_coef_final: 0.001
  load_coef_initial: 0.001
  load_coef_final: 0.0

comm:
  module: soft_role_attention
  num_heads: 1
  role_pair_values: true
  role_pair_queries: false
  role_pair_keys: false
  topology: full

model:
  action_role_experts: true
  expert_head_layers: 1
~~~

These values are starting hypotheses, not final tuned recommendations. Freeze
them after a bounded smoke study.

## 21. Required tests

### 21.1 Mathematical equivalence

- One-hot role probabilities exactly reproduce hard role embedding lookup.
- One-hot receiver/sender probabilities select exactly one role-pair expert.
- One-hot action gate selects exactly one action expert.
- Uniform probabilities create the exact equal mixture.
- Mixture action probabilities sum to one.

### 21.2 Shape handling

- \([B,N,D]\) inference works.
- \([T,B,N,D]\) PPO evaluation works.
- Role-pair bias orientation is receiver-first, sender-second.
- Masks remain effective after adding role-pair biases.
- All-masked rows remain finite.

### 21.3 Gradient behavior

- Task loss sends gradients to the role router.
- Every action head receives gradient under uniform soft gates.
- Every relation expert receives gradient under uniform soft gates.
- Oracle one-hot gates send gradient only to the selected experts.
- The initial PPO ratio remains approximately one before an update.
- Expert and router gradients obey clipping.

### 21.4 Permutation equivariance

For an agent permutation \(P\):

\[
\operatorname{Model}(PX,PAP^\top)
=
P\operatorname{Model}(X,A).
\]

Test role posteriors, messages, action probabilities, and environment outputs.

### 21.5 Environment correctness

- Every reset produces exactly three detectors and two capturers.
- Assignments vary across resets and are reproducible by seed.
- Agent indices do not predict roles over many resets.
- Detector observations contain the intended target information.
- Capturer observations remain hard-blind at \(\alpha=0\).
- Only hidden capturers can satisfy capture eligibility.
- Capture without recent detection remains impossible.
- Parallel environments may have different role permutations simultaneously.

### 21.6 Causal intervention integrity

- Message shuffle preserves shapes and marginal message distributions.
- Gate shuffle preserves role-probability marginals.
- Edge blocking changes only the requested direction.
- Expert blocking changes only the requested role-pair contribution.
- Common evaluation seeds produce identical environment trajectories up to
  policy-action consequences.

### 21.7 Numerical and semantic traps

- Mixing categorical probabilities is not the same as mixing their logits.
- A soft embedding is not a mixture of nonlinear expert networks.
- A soft expected attention score is not the exact expectation of hard-role
  attention distributions.
- Receiver role is always the first relation index and sender role the second.
- Add relation bias before graph masking; no learned bias may resurrect a
  prohibited edge.
- Normalize attention over the sender axis.
- Preserve the existing self-loop convention separately for each communication
  implementation.
- Define finite behavior for an all-masked receiver row.
- Never detach the router on PPO evaluation.
- Do not compare latent labels across seeds without permutation alignment.
- Test both 3-D inference tensors and flattened 4-D rollout tensors.
- For QMIX, verify that auxiliary losses send no gradient into target modules.

## 22. Computational complexity

Shared dense attention costs approximately:

\[
O(MND^2+MN^2D),
\]

with attention-weight memory \(O(MN^2H)\). A shared-head role-pair bias can be
computed as:

~~~python
pair_bias = (role_probs @ role_pair_bias) @ role_probs.transpose(-1, -2)
~~~

This association costs approximately:

\[
O\!\left(MNK^2+MN^2K\right),
\]

rather than materializing a tensor of shape \([M,N,N,K,K]\). Head-specific
biases multiply this role-routing cost by \(H\).

Naively evaluating a full matrix expert for every edge and role pair costs:

\[
O(MN^2K^2D^2).
\]

For \(N=5,K=2,D=64\), this is feasible but not elegant. Efficient
implementations can:

- precompute every sender-role transform \(W^s x_j\);
- use receiver-role FiLM modulation;
- factor \(W_{r\leftarrow s}\) into sender and receiver components;
- use low-rank adapters around one shared \(W_0\);
- specialize values only.

Parameter counts should be reported explicitly. A full relation bank contains:

\[
K^2D^2
\]

matrix parameters, while \(K\) action readout heads contain approximately:

\[
KD|\mathcal A|
\]

parameters after a shared trunk.

For a router hidden width \(P\), the router adds roughly \(DP+PK\) parameters.
Role embeddings add \(KD\); shared-head pair bias adds \(K^2\); full
role-specific query/key/value banks would add roughly \(3KD^2\). Treat that
full-projection version and the \(K^2D^2\) relation bank as high-capacity
ablations, not free architectural changes.

Population-size invariance of the actor does not imply population-size
invariance of the whole learner. The current centralized critic, rollout
storage, and QMIX mixer assume fixed \(N\).

## 23. Failure modes and interpretations

| Failure | Observable symptom | Likely cause | Response |
|---|---|---|---|
| One-role collapse | \(K_{\mathrm{eff}}\approx1\) | Team reward gives no specialization pressure | Check oracle-uniform gap; use weak temporary load balancing |
| Permanently uniform roles | Gate entropy remains maximal | Experts are identical or router gradients vanish | Inspect expert disagreement and router gradients |
| Dead experts | Near-zero gate mass and gradients | Early gate saturation | Warm-up with softer temperature and balanced initialization |
| Arbitrary expert diversity | Experts differ but roles do not align or improve return | Diversity regularizer dominates task objective | Remove forced diversity and rely on causal metrics |
| Role flicker | High timestep switching | Feedforward cue noise or no persistence cost | Add history or a correctly implemented temporal term |
| Identity shortcut | Same indices receive same role | Fixed capability assignment or ID leakage | Randomize hidden roles and remove identity inputs |
| Spatial-bin shortcut | Roles track location rather than function | Position is easiest discriminative cue | Evaluate on role permutations and varied initial positions |
| Extra-capacity gain | Learned and uniform MoE both beat shared actor equally | More heads, not meaningful gating | Use learned-minus-uniform as primary comparison |
| Alignment without value | High role accuracy, learned ≈ uniform | Roles are inferable but unnecessary computationally | Report classification, not coordination improvement |
| Value without alignment | Learned > uniform, low hidden-role agreement | Latents encode other behavioral modes | Call them behavioral modes, not detector/capturer recovery |
| Zero-only sensitivity | Zero messages hurt but shuffle does not | Out-of-distribution zero activation | Do not claim semantic message use |
| Negotiation leakage | Router uses privileged global tensors | Centralized execution hidden in implementation | Enforce local message interfaces |

## 24. Cross-task extension

The larger research program is:

> How does learned communication and computational specialization change as
> task-induced information dependencies change?

Do not compare unrelated tasks and attribute every difference to role
structure. Prefer a controlled family:

1. **Hidden-role blind PCP:** sensing and capture authority are split.
2. **Sighted or information-sufficient variant:** target information is locally
   available, with capability cues defined so roles remain inferable if that is
   required.
3. **Homogeneous coordination task:** no fixed physical class split.
4. **Multi-target allocation task:** several simultaneous subgoals create
   pressure for multiple subteams.

Across tasks, measure:

- effective number of roles;
- role stability;
- action-expert divergence;
- intra-role and inter-role routing mass;
- causal effect of each directed relation;
- message-shuffle effect;
- whether a one-role model becomes sufficient.

If a task provides no node-level asymmetry, a deterministic equivariant router
cannot break symmetry. Introduce legitimate task features or stochastic
assignment rather than hidden agent identifiers.

## 25. Endogenous capability allocation

A future environment could let the team allocate scarce capabilities. This
requires a hierarchical action before ordinary task execution:

\[
\mathbf c_0
\sim
\Pi_{\mathrm{alloc}}
(\cdot\mid G_0,\mathbf o_0),
\]

followed by:

\[
\mathbf a_t
\sim
\Pi_{\mathrm{task}}
(\cdot\mid \mathbf o_t,\mathbf c_0).
\]

The allocation must obey a resource constraint, for example:

\[
\sum_i\mathbf 1[c_i=\mathrm{sensor}]=3,
\qquad
\sum_i\mathbf 1[c_i=\mathrm{capturer}]=2.
\]

Without a budget or incompatibility, every agent may choose the dominant
capability or combine sensing and capture, eliminating the information
asymmetry that motivates communication.

A centralized Sinkhorn or top-k allocator enforces counts but is itself a
centralized coordination mechanism. A decentralized exact quota requires
negotiation or distributed matching. Capability allocation is therefore beyond
the first SoftRole-HetGAT study.

## 26. Staged experiment plan

### Stage 0 — prerequisite

Complete the primary matched communication audit first. SoftRole-HetGAT should
not become necessary for the thesis to be defensible.

### Stage 1 — unit and synthetic tests

- implement deterministic soft roles;
- verify one-hot equivalence;
- verify mixture probabilities and gradients;
- test permutation equivariance;
- test randomized hidden-role PCP.

### Stage 2 — three-seed smoke

Run:

- oracle hard gate;
- learned soft gate;
- uniform gate.

Three seeds each gives nine runs.

Proceed only if:

- oracle materially exceeds uniform;
- learned improves over uniform in the expected direction;
- both roles are used;
- critical relation and action experts receive gradients;
- held-out balanced role accuracy is approximately at least 0.80;
- no more than one prespecified stabilization adjustment is required.

If oracle and uniform are similar, stop. Role-specific computation then has too
little behavioral value for this task.

### Stage 3 — core ten-seed study

Run the three primary conditions at ten seeds: 30 MAPPO training runs.

Require:

- learned-minus-uniform paired confidence interval above zero;
- a meaningful oracle-gap recovery ratio;
- role alignment above chance;
- no expert collapse;
- message and role-gate shuffle sensitivity;
- directional detector-to-capturer intervention effect.

### Stage 4 — component attribution

Only after Stage 3 succeeds, run:

- action-specialization-only;
- communication-specialization-only;
- full specialization;
- uniform null.

Use the smallest seed count that can reliably decide whether a full ten-seed
expansion is warranted.

### Stage 5 — optional depth

- negotiated role inference;
- recurrent role persistence;
- more than two roles;
- variable team size;
- another task family;
- QMIX replication;
- endogenous capability allocation.

Do not add these to rescue a failed core experiment.

## 27. Related work and novelty boundary

[HetNet](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1173.pdf)
(Seraj et al., AAMAS 2022) is the direct architectural starting point. It
assumes known agent classes. Agents with the same state, observation, and
action spaces belong to one class; communication edges are typed by ordered
sender-receiver class pairs; and class-specific encoders, decoders, attention
transformations, and policies are learned. HetNet's Gumbel-Softmax operation
binarizes message content. It does not infer or reassign classes.

[ROMA](https://proceedings.mlr.press/v119/wang20f.html)
(Wang et al., ICML 2020) learns stochastic latent role embeddings and
role-conditioned policies, with regularizers intended to produce specialized
and identifiable roles.

[SORA](https://link.springer.com/chapter/10.1007/978-981-99-8079-6_25)
(Zhou et al., ICONIP 2023) is a close action-side precedent. It produces a soft
role distribution and uses role-specific Q-networks, allowing an agent to
express multiple roles simultaneously.

[LDSA](https://proceedings.neurips.cc/paper_files/paper/2022/file/0b4145b562cc22fb7fa50a2cd17c191d-Paper-Conference.pdf)
(Yang et al., NeurIPS 2022) derives categorical subtask assignments and learns
distinct subtask policies whose parameters are generated from subtask
representations.

[GRDC](https://ojs.aaai.org/index.php/AAAI/article/view/40185)
(Gao et al., AAAI 2026) most closely connects graph-derived roles and
communication. It constructs local interaction graphs, matches agents to role
prototypes, and restricts attention communication to agents assigned the same
role. Its roles determine who may communicate, but it does not softly
interpolate a bank of ordered cross-role message transformations.

These papers establish that latent roles, soft role distributions,
role-specific policies, graph-derived grouping, and role-aware communication
are not independently novel.

The defensible architectural distinction is:

> SoftRole-HetGAT uses one online role posterior to route both role-specific
> action experts and every directed sender-role-to-receiver-role message
> operator, while preserving cross-role communication and allowing agents to
> blend or change roles.

The empirical contribution should emphasize:

- recovery of useful hidden functional roles;
- parameter-matched action and communication attribution;
- causal gate, message, edge, and expert interventions;
- the relation between inferred roles and task-required information flow.

It should be presented as a soft latent-role generalization of HetNet, not as
the first emergent-role MARL method.

## 28. Permissible and prohibited claims

### Permissible after full positive evidence

> SoftRole-HetGAT inferred the randomized detector/capturer partition from
> local capability cues and used that partition to specialize action
> computation and directed information routing, recovering a substantial
> fraction of the oracle-gated model's benefit.

### Conditional interpretations

- Learned exceeds uniform, but roles do not align: the model learned useful
  behavioral modes, not the detector/capturer partition.
- Roles align, but learned equals uniform: roles are inferable but
  role-specialized computation adds no behavioral value.
- Oracle exceeds uniform, but learned fails: specialization is useful, but the
  unsupervised router did not recover it.
- Oracle equals uniform: this PCP variant does not justify role-specific
  mixture computation.
- Only action specialization works: attribute the gain to action experts.
- Only communication specialization works: attribute the gain to relation
  routing.
- Full specialization lacks a positive factorial interaction: do not claim
  synergy.

### Prohibited claims

- first emergent-role MARL method;
- arbitrary role invention;
- roles emerged without observable cues;
- automatic discovery of any number of roles;
- learned sparse graph topology;
- universal generalization across tasks or team sizes;
- algorithm invariance from MAPPO;
- causal role use from alignment alone;
- communication use from decodability alone;
- dynamic role reassignment when hidden roles are fixed within episodes;
- role benefit without capacity-matched and same-checkpoint controls.

## 29. End-to-end pseudocode

~~~python
def forward_actor(obs, candidate_mask, mode, hidden_role=None):
    # Shared local representation
    h0 = encoder(obs)                              # [B, N, D]

    # Role gate
    role_logits = role_router(h0)                 # [B, N, K]
    learned_p = softmax(role_logits / temperature, dim=-1)

    if mode == "learned":
        p = learned_p
    elif mode == "oracle":
        p = one_hot(hidden_role, num_classes=K)
    elif mode == "uniform":
        p = full_like(learned_p, 1.0 / K)
    else:
        raise ValueError(mode)

    # Soft role embedding
    role_emb = einsum("bnr,rd->bnd", p, E_role)
    x = h0 + role_emb

    # Shared content-based attention
    q = W_q(x).reshape(B, N, H, d_head)
    k = W_k(x).reshape(B, N, H, d_head)
    content_score = pairwise_dot(q, k) / sqrt(d_head)

    # Directed soft role-pair bias
    role_bias = einsum("bir,hrs,bjs->bijh", p, G_role, p)
    score = content_score + role_bias
    alpha = masked_softmax(score, candidate_mask, sender_dim=2)

    # Directed soft role-pair messages
    # Each expert maps sender x_j for receiver role r and sender role s.
    expert_value = apply_all_relation_experts(x)
    # conceptual shape: [B, receiver_i, sender_j, r, s, H, d_head]
    edge_gate = einsum("bir,bjs->bijrs", p, p)
    edge_value = (
        edge_gate[..., None, None] * expert_value
    ).sum(dim=(3, 4))                              # [B, i, j, H, d_head]

    message = (
        alpha.unsqueeze(-1) * edge_value
    ).sum(dim=2)                                   # [B, i, H, d_head]
    message = message.reshape(B, N, H * d_head)    # [B, i, D]
    h_post = layer_norm(x + W_o(message))

    # Role-specific action experts
    actor_features = actor_trunk(h_post)
    expert_logits = action_heads(actor_features)  # [B, N, K, A]
    expert_probs = softmax(expert_logits, dim=-1)

    # True mixture of expert policies
    action_probs = (
        p.unsqueeze(-1) * expert_probs
    ).sum(dim=-2)                                 # [B, N, A]

    dist = Categorical(probs=action_probs)
    diagnostics = {
        "role_probs": p,
        "learned_role_probs": learned_p,
        "attention": alpha,
        "edge_gate": edge_gate,
        "expert_probs": expert_probs,
    }
    return dist, diagnostics
~~~

The actual implementation should avoid materializing the full conceptual
\([B,N,N,K,K,H,d_h]\) tensor when possible. Precompute transformed sender
features or use a factored relation expert.

## 30. Final recommendation

The proposed architecture is coherent and implementable:

\[
\boxed{
\text{infer roles}
\rightarrow
\text{mix directed communication experts}
\rightarrow
\text{mix role action policies}
}
\]

Its strongest property is that the same latent posterior controls both
information routing and behavior. Its strongest scientific test is not whether
it obtains a higher return, but whether:

1. learned gates outperform an equal-capacity uniform mixture;
2. inferred roles align with hidden task functions;
3. the architecture recovers much of an oracle gate's advantage;
4. role, edge, message, and expert interventions cause the predicted behavioral
   effects;
5. action and communication specialization can be attributed separately.

The matched causal communication audit should remain the primary thesis. This
architecture is a rigorous extension answering the next question:

> Once communication is known to matter, can agents infer the functional
> structure needed to decide how information and policies should specialize?
