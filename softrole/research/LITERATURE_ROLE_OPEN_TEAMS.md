# Literature audit: learned roles, changing teams, and capability recovery

Audit date: **2026-09-25**. This is a targeted primary-source audit, not a proof that all prior art has been found. It supplements the original architecture and critical-review notes without changing them. File inventory and access limitations are recorded in [manifest_roles.json](../papers/manifest_roles.json). PDF page numbers below count from the beginning of the indicated file; printed page numbers are supplied where useful. Text extractions in `role_pdf_text/` support reproducible searching; they are not substitutes for the source PDFs.

## Assessment

The proposed research question is viable, but **learned roles + graph communication + changing team sizes is already established territory**. The defensible experiment is narrower: after unannounced loss or change of a *physical capability*, can a frozen, decentralized recurrent policy recover the team's information and action responsibilities, including when teammates arrive or depart? The role variable must causally improve recovery beyond a capability-conditioned recurrent graph policy with equal training coverage and capacity.

Three distinctions are essential:

1. **Fixed neural parameters versus fixed internal state.** Frozen deployment can still update recurrent states, capability beliefs and role assignments. Training a new encoder or doing gradient updates during deployment is a different adaptation regime.
2. **Composition at reset versus membership during a trajectory.** Testing different team compositions between episodes does not establish recovery from arrivals or failures during an episode.
3. **Controlled entrants versus unfamiliar policies.** Replicating the same learned actor on new robots is open-team cooperative MARL. Working with independently trained teammates is ad hoc teamwork and requires a broader evaluation.

## ROMA: precise comparison

Primary sources: [ICML 2020 paper](https://proceedings.mlr.press/v119/wang20f.html), [main PDF](../papers/2020_ROMA_Wang_Emergent_Roles.pdf), [supplement](../papers/2020_ROMA_Wang_Supplement.pdf).

**Verified mechanism.** Main pp. 3–4, Eqs. (1)–(7):

\[
(\mu_i,\sigma_i)=f_\phi(o_i),\quad
\rho_i\sim\mathcal N(\mu_i,\sigma_i),\quad
\theta_i=g_\psi(\rho_i),\quad Q_i(\tau_i,a_i;\theta_i).
\]

\[
\mathcal L=\mathcal L_{TD}+\lambda_I\mathcal L_I+\lambda_D\mathcal L_D,
\quad
\mathcal L_I=\mathbb E\,D_{KL}\big[p(\rho_i^t\mid o_i^t)\Vert q_\xi(\rho_i^t\mid\tau_i^{t-1},o_i^t)\big].
\]

ROMA learns continuous stochastic role embeddings and uses a hypernetwork to generate policy parameters. This is different from a finite simplex gate mixing action distributions and ordered message experts. Its demonstrated roles are already dynamic: §5.1, Fig. 3, pp. 6–7 show adaptation to position, health and death; §5.2 includes heterogeneous unit types. Consequently, “ROMA has fixed roles” or “does not react to changing health” would be false.

The audited experiments do not establish unannounced entrant handling or frozen transfer to new capability distributions. ROMA is a necessary learned-role comparator, but it is insufficient as the only novelty comparator. Its regularizers are learning objectives, not guarantees of recovery time, optimal role allocation or generalization.

**Mathematical caution, independently derived.** Supplement A.1 derives a variational lower bound for \(I(\rho_i^t;\tau_i^{t-1}\mid o_i^t)\), then explicitly assumes \(\rho_i^t\perp\tau_i^{t-1}\mid o_i^t\). Under the literal observation-conditioned sampler with independent sampling noise, that conditional mutual information is zero. The KL training loss remains well-defined, but the formal lower-bound argument does not establish nontrivial information about history beyond the observation. For SoftRole, use an explicit history-conditioned belief or predictive objective rather than importing a claim of guaranteed semantic identification.

## Closest role and communication papers

### RODE: an important correction to the existing framing

[ICLR 2021 record](https://openreview.net/forum?id=TTUVg6vkNjK); [author-hosted final PDF](../papers/2021_RODE_Wang_Roles_Decompose_Tasks_Author_Final.pdf). An earlier arXiv version is also archived, but its pagination differs.

RODE clusters actions by learned effects, defines restricted role action spaces, and periodically selects roles. §3.2–3.3, pp. 4–5 uses dot-product action and role scoring, schematically
\(Q_i(\tau_i,a)=h_{\tau_i}^{\top}z_a\), with a role selector and role-conditioned action support.

**Transfer is real but qualified.** Final §4.3, pp. 8–9 and Fig. 5 train on six allies and evaluate up to eighteen with unchanged role policies. However, p. 8 explicitly collects **50k new-task samples to train an action encoder** for the expanded action set. Appendix C.2, p. 19 confirms this step. Observations use the nearest five allies and twenty-four enemies.

Thus RODE is precedent for role-policy transfer to larger teams. It is not a clean example of absolutely no deployment-time fitting when new actions arise. SoftRole should report both “no policy updates” and the stricter “no parameter updates anywhere,” and should retain a common action interface when testing size transfer alone.

### Learning to Transfer Role Assignment Across Team Sizes

[Nguyen et al., AAMAS 2022](https://arxiv.org/abs/2204.12937); [PDF](../papers/2022_Nguyen_Transfer_Role_Assignment_Team_Sizes.pdf).

§3.1–3.3, p. 3 gives a role-based value mixer:

\[
Q_{tot}=b^{(2)}+\sum_kW_k^{(2)}\sigma\!\left(b_k^{(1)}+\sum_iW_{ik}^{(1)}Q_i\right),
\quad W_{ik}^{(1)}\ge0,\quad\sum_iW_{ik}^{(1)}=1.
\]

Shared observation-conditioned networks generate agent-to-role mixing weights. These normalize **over agents for each role**, unlike a per-agent simplex distribution over roles. A reward-horizon regularizer encourages different role contributions. §3.3 explicitly uses a small-team demonstration stage followed by continued reinforcement learning on larger teams. The work blocks novelty claims about role transfer across team sizes; its protocol leaves room for a stricter frozen deployment study. Its enriched predator-prey task includes tools and defending/attacking responsibilities and is useful environment precedent.

### SORA

[Publisher chapter](https://link.springer.com/chapter/10.1007/978-981-99-8079-6_25). ICONIP 2023, first online 14 November 2023; Springer citation/print year 2024. Full text was not accessible through the available public routes.

The publisher abstract verifies attention-generated soft role distributions, simultaneous multiple-role membership, and role-specific Q networks. It therefore establishes precedent for soft role specialization. **It does not, from the available text, verify the precise probability-mixture equation** \(\pi=\sum_rp_r\pi_r\). The original critical review's equation table should cite classical mixture-of-experts for that exact equation and describe SORA more conservatively. No claim about SORA's churn or capability-loss experiments is justified by this abstract-only audit.

### ROCO

[Publisher record](https://www.sciencedirect.com/science/article/pii/S0957417425030374), [author preprint record](https://ssrn.com/abstract=5060074), [author code](https://github.com/surebo/ROCO). Expert Systems with Applications 297, 129421 (February 2026); DOI registered August 2025; preprint posted December 2024.

Publisher sections verify learned roles, inter-role attention, intra-role mutual-information message filtering, and evaluation on SMAC/GRF, including larger maps. This is a close architectural collision with role-conditioned communication. Full PDF requests were blocked, so exact equations and all transfer protocols remain unverified here. Evaluating a method on a separately trained large map must not be relabelled frozen-policy transfer. The proposed complete directed role-pair operator bank remains a specific architectural distinction to test, not evidence that the broader role/communication idea is new.

### GRDC

[Gao et al., AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40185); [PDF](../papers/2026_GRDC_Gao_Role_Discovery_Communication.pdf).

Method pp. 3–4 (printed 29443–29444), Eqs. (1)–(3): neighborhood attention augments each trajectory embedding to \(z_i\); prototype matching yields
\[
q_i=\operatorname{GumbelSoftmax}(z_i^\top P_1,\ldots,z_i^\top P_K),\qquad
\beta_i=\sum_{j\in G_{r_i}\setminus\{i\}}\alpha_{ij}h_j.
\]

Messages are restricted to agents sharing the inferred role. Entropy, prototype decorrelation and pruning prevent redundant roles. This directly precedes graph-derived role discovery coupled to communication. SoftRole can differ by preserving complementary cross-role information flow, but must show why that matters in the environment. The stated linear complexity requires bounded neighborhood and same-role group sizes; it is not an unconditional scaling result. The audited evaluation does not provide a frozen unseen-size/capability-loss recovery guarantee.

### SCoUT

[Vora et al., March 2026 preprint](https://arxiv.org/abs/2603.04833); [PDF](../papers/2026_SCoUT_Vora_Temporal_Grouping.pdf).

§4.2–4.4, PDF pp. 6–8: soft Gumbel group assignments \(Y\) produce affinity \(G=YY^\top\); a \(\log(G_{ij}+\epsilon)\) term biases recipient logits, and the same assignments map group values to per-agent baselines. §4.5 uses leave-one-message-out value differences for communication credit.

**Execution distinction:** introduction and §4.6 explicitly discard the centralized group sampler at test time. It is precedent for reusing soft groups to structure communication learning and a critic, not automatically precedent for a persistent execution-time role posterior routing both policy and message operators. §5 studies training at large populations; large-scale training does not itself establish small-to-large frozen transfer. This audit treats the work as a preprint, not a verified conference publication.

### GIRA

[Li et al., publisher record](https://www.sciencedirect.com/science/article/pii/S0925231226018357). Neurocomputing 700, 134437, issue date November 2026; accessible record and DOI registration predate this audit (July 2026).

Publisher abstract/method/experiment previews explicitly cover **homogeneous agents with common observation/action spaces**, trajectory-aligned replay for joining/leaving agents, an agent-count encoder, and interaction-driven roles. Experiment headings distinguish dynamic populations (§5.3) from zero-shot unseen team sizes (§5.5). These are direct overlaps with the proposed framing. A potential distinction is latent physical capability degradation in heterogeneous teams with frozen belief updates. Full PDF access was blocked, so the detailed churn schedule, role equations and test leakage controls need an eventual full-text check; they cannot be assumed from the abstract.

### g-MAT

[Wang et al., publisher record](https://www.sciencedirect.com/science/article/pii/S0950705126013389). *Learning group collaboration for efficient multi-agent reinforcement learning*, Knowledge-Based Systems 351, 116612 (2026); DOI registered 11 July 2026.

The primary abstract and introduction verify hierarchical clustering of an action-effect graph into role-labelled groups, followed by hybrid attention combining local within-group and global cross-group information in an autoregressive policy. This is additional direct precedent for learned roles determining communication structure. It reinforces the existing novelty boundary without establishing the proposed frozen capability-recovery mechanism. The publisher PDF returned HTTP 403, and no legal public author copy was located. Exact equations, execution information requirements, and event/transfer protocols therefore remain unverified.

## Open teams and ad hoc teamwork

### GPL (ICML 2021)

[Rahman et al.](https://proceedings.mlr.press/v139/rahman21a.html); [PDF](../papers/2021_GPL_Rahman_Open_Ad_Hoc_Teamwork.pdf).

§3 formalizes open stochastic Bayesian games. §4.2, pp. 4–5 combines recurrent inferred teammate types, an agent-action model, and a coordination-graph value function:
\[
Q(s,a)=\sum_{j\in N_t}Q_j(a_j\mid s)+\sum_{j\ne k\in N_t}Q_{jk}(a_j,a_k\mid s),
\qquad
\bar Q(s,a_i)=\mathbb E_{a_{-i}\sim\hat\pi_{-i}}Q(s,a).
\]

§5.2–5.5 describes actual entrance/exit processes, training with up to three agents and evaluation up to five. This already demonstrates graph-based adaptation to unknown team composition. The learner controls one agent while teammates follow fixed policies; that differs from jointly deploying a shared SoftRole actor. Recurrent type inference is nevertheless a strong precedent for capability/performance inference. Generalization is empirical; the architecture alone does not prove success on arbitrary unseen populations.

### GPL general framework (JMLR 2023)

[Rahman et al.](https://jmlr.org/papers/v24/22-099.html); [PDF](../papers/2023_GPL_Rahman_General_Open_Ad_Hoc_Framework.pdf).

This 74-page extension formalizes partial observability as PO-OSBG (§3.3, p. 9) and develops belief representations over environment state, teammate existence and teammate type (§6–7). These distinctions matter for dropout: an agent missing from view need not have departed. The paper should guide how a recurrent SoftRole system separates membership uncertainty, capability uncertainty and task role. A belief about a teammate's latent behavior is not itself an assigned functional role. GPL is a useful conceptual comparator, but its single controlled learner and independent teammates differ from a fully cooperative policy trained jointly.

### CIAO: what its theorem establishes

[Wang et al., ICML 2024](https://proceedings.mlr.press/v235/wang24an.html); [PDF](../papers/2024_CIAO_Wang_Open_Ad_Hoc_Cooperative_Game_Theory.pdf).

§3 defines an open stochastic Bayesian coalitional-affinity game. Theorem 2, p. 5, justifies a pairwise-plus-individual Q decomposition under an additive preference structure and Assumption 2, which zeroes an agent's future contribution after departure. It supports a specific modeling choice in GPL-like systems. It does not establish unrestricted GNN transfer: Appendix D's Assumption 7 explicitly assumes generalizability to agent types. Other assumptions include fixed teammate policies for the convergence argument and sufficient teammate residence time. Thus it is relevant mathematical prior art, but its guarantees cannot simply be attached to a SoftRole-HetGAT actor or to arbitrary capability shocks.

### NAHT / POAM

[Wang et al., NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/cabf611498431ad89a85ace75f790d93-Abstract-Conference.html); [PDF](../papers/2024_NAHT_Wang_N_Agent_Ad_Hoc_Teamwork.pdf).

§3, p. 3 optimizes a controlled subset against sampled uncontrolled teammates:
\[
\max_\theta\;\mathbb E_{\pi^{(M)}\sim X(U,C(\theta))}\left[\sum_t\gamma^t r_t\right].
\]

The team sampler runs **at the beginning of each episode**. The abstract's dynamic-team wording should therefore not be used as evidence for unannounced within-trajectory additions. §4 gives a counterexample showing that copying a policy optimized for one controlled agent can be suboptimal when several agents are controlled. POAM combines shared independent PPO with a recurrent teammate model; §6.3 tests held-out teammate policies. This is the appropriate comparison if entrants may use independently learned conventions, a harder problem than copies of a common actor.

### Additional 2025 NAHT developments

[MAT-NAHT](https://arxiv.org/abs/2506.05527), [PDF](../papers/2025_MAT_NAHT_Sequence_Modeling.pdf): the inspected October 2025 preprint uses a **centralized transformer** over controlled agents' historical observations/actions and evaluates one StarCraft task. It is relevant to variable controlled subsets, but it is not decentralized execution or a latent capability-recovery result.

[Shapley Machine](https://arxiv.org/abs/2506.11285), [PDF](../papers/2025_Shapley_Machine_N_Agent_Ad_Hoc.pdf): the inspected June 2025 preprint models NAHT through cooperative games and derives TD(λ)-like credit learning satisfying Shapley axioms. It is additional prior art for open-team credit assignment, rather than evidence for role-aware communication. These two papers were checked at abstract/method level; their proofs and all experimental results were not independently audited.

## Capability failures and newer openness results

### Collaborative Adaptation (CDC 2024)

[Findik, Hasenfus and Azadeh, author PDF](https://cs.uml.edu/~reza/pdf/CDC_2024.pdf); [local PDF](../papers/2024_Findik_Collaborative_Adaptation_Malfunctions.pdf).

The paper studies immobilized grid agents and leg malfunctions in a multi-agent ant. §VI, p. 5 assumes a mechanism that detects the malfunction time and affected agent, resets exploration, and continues learning; Fig. 4 reports training after a malfunction at episode 30,000. It is direct precedent for cooperative recovery from lost capabilities, but it addresses **post-failure learning**, not frozen-policy inference. This distinction supports a meaningful SoftRole evaluation axis: recovery without a fault identity label, exploration reset or parameter updates.

### PLATO (July 2026 preprint)

[Saleh Abadi et al.](https://arxiv.org/abs/2607.25082); [PDF](../papers/2026_PLATO_Agent_Task_Openness.pdf).

PLATO combines a pointer actor over current tasks with a graph critic. §2, p. 3 distinguishes agent and task openness from **fundamental openness**, including changing capabilities, and leaves the latter for future work. Lemma 1/Theorem 1 (pp. 5–6; Appendix A) establish permutation properties and well-defined processing of varying sets; empirical zero-shot results are separate. It is especially relevant if SoftRole actions address a changing task set. Correct technical language is important: task-indexed probability vectors permute with task order, while the represented distribution over task identities remains unchanged. These facts do not imply a return guarantee for new tasks or team sizes.

### SubMAPG (May 2026 preprint)

[Liu et al.](https://arxiv.org/abs/2605.13269); [PDF](../papers/2026_SubMAPG_Submodular_Open_Team_Allocation.pdf).

This work covers open-agent/open-target allocation with monotone submodular stage utilities. Lemmas 3.1–3.2 connect categorical joint policies to a partition multilinear extension and marginal-contribution gradients. Theorems 5.1–5.2 prove stagewise half-approximation and dynamic-regret results for **projected stochastic ascent in marginal space**. §5 explicitly excludes general neural parameter-space convergence guarantees.

It offers a theorem-backed alternative if the thesis is recast as submodular coverage/allocation. It does not prove a long-horizon neural HetGAT policy optimal or generalizing. In particular, complementary detector-and-capturer tasks can violate diminishing returns; submodularity must be verified for the actual utility before importing the theorem.

## Corrections and implications for the proposed study

The original critical review is broadly right that the primitive architecture is crowded. This audit sharpens several claims:

- Treat role adaptation to health/death as already demonstrated by ROMA.
- Include RODE's qualified larger-team transfer, not only Nguyen's curriculum transfer.
- Distinguish SCoUT's training-only grouping from an execution-time router.
- Do not claim SORA implements an exact probability mixture without its full text.
- Treat GIRA as especially close for dynamic population changes, while preserving its stated homogeneous-agent scope and current access limitations.
- Include open-team and capability-awareness baselines, not just role-learning baselines. The separate main report covers capability-aware shared policies and CASH.
- Preserve the scientific distinction between performance inference, physical capability inference, learned behavioral roles and communication responsibilities.

A concrete contribution candidate is: **a recurrent capability-belief-conditioned soft relation operator, deployed with frozen parameters, that recovers complementary information/action responsibilities after hidden capability shocks under agent entry/exit, with measured recovery latency and causal role interventions.** This is a hypothesis and experimental program, not a confirmed novelty claim or a performance theorem.

Useful matched comparisons are: common recurrent graph actor without roles; roles only in the actor; roles only in messages; coupled actor/message roles; observed capability versus inferred capability; adaptive versus frozen role posterior; and equal-capacity dense relation experts. Hold the training distribution, observations, physical communication graph and reward fixed. Train on several team sizes as well as a fixed-size diagnostic; otherwise failure may reflect absent experience rather than an architectural defect.

For identifiability, a policy must receive evidence of failure (action outcomes, sensor consistency, announced capability status, or informative messages). If two capability states induce the same accessible histories but require different actions, no observation-based controller can always choose correctly. For recovery, compare return with a capability-aware oracle on the **same post-shock task**: if all capturers disappear, no router can recreate capture authority. For size transfer, separate changing count, changing density, changing load per agent, new capability combinations and out-of-range capability values.

## Search and verification limitations

Primary records were checked through PMLR, conference proceedings, JMLR, publisher pages, arXiv, author PDFs and author code links. Queries included learned/soft roles, role transfer, open ad hoc teamwork, dynamic teams, capability failure, and 2025–2026 work. The audit prioritizes direct overlaps over all historical communication papers; foundational symmetry/GNN/HetNet theory is handled elsewhere in this project. Four especially relevant papers (SORA, ROCO, GIRA, g-MAT) have verified primary metadata and accessible descriptions but unavailable full PDFs. This limitation is represented explicitly in the manifest, rather than replacing blocked sources with unrelated files or asserting unchecked equations.
