# V2 Simple Spread communication study — consolidated results

**Scope.** Eight suites, 231 planned training runs, 224 completed, ~43 GPU-hours on UCF ARCC Newton
(Tesla V100). Commit `49b2ad43`, BenchMARL 1.5.2 / TorchRL 0.11.1 / VMAS 1.5.2 / PyTorch 2.13.0+cu126.
Every run is 600,000 environment frames of MAPPO on VMAS `simple_spread`.

This document consolidates the eight per-suite `results/*/REPORT.md` files, explains what each number
means, assesses whether the results are what the environment should have produced, and gives the
readiness verdict for the predator–prey (PP) and predator–capture–prey (PCP) phase.

---

## 1. What the environment actually is

This matters more than anything else in this report, so it goes first.

**VMAS `simple_spread`, 3 agents, 3 landmarks, 100 steps, continuous 2-D force actions.**

**Reward (shared, identical scalar for all agents, every step):**

```
r_t = -3 · Σ_landmarks  min_agents ‖pos_agent − pos_landmark‖    −   collision penalty
```

Two details that are easy to miss and that shape every result below:

1. The landmark-coverage term is **multiplied by the agent count** (the VMAS implementation loops
   over agents around a term that does not depend on the agent), so with 3 agents the coverage
   penalty is counted 3×. This is why returns sit near −460 rather than −150; it is a scale factor,
   not a bug in our code.
2. Collisions cost −1 per *ordered* overlapping pair, so each overlapping pair of agents costs −2
   per step.

**Observation (14-D per agent):** own position (2), own velocity (2), relative position of all 3
landmarks (6), relative position of both other agents (4). The BenchMARL task config sets
`obs_agents: True`.

> **Every agent already observes every other agent's position and every landmark. The task is fully
> observable and there is no information asymmetry.**

That single fact is the lens for the whole study. Communication here cannot transmit information a
receiver lacks — it can only transmit *computation* (an encoding of another agent's internal state
or implicit landmark assignment). So we should expect communication to help a little, and we should
expect mechanisms that are good at *selecting whom to listen to* to show no advantage, because no
sender is more informative than another. Both expectations are borne out below.

### The methods compared

| Name | What it does | Comm params | Realized bandwidth |
|---|---|---:|---:|
| `benchmarl_mlp` | Stock BenchMARL MLP actor. No comm layer at all. | 0 | — |
| `identity` | Our comm scaffold with a pass-through module. Architecturally identical to the MLP. | 0 | 0 |
| `broadcast` | Every agent sends one message to every other; receiver mean-pools them. | 8,192 | 6,144 bits/step |
| `gated` | Broadcast plus a learned scalar gate per edge that scales each message. | 8,321 | 6,144 bits/step |
| `attention` | Multi-head dot-product attention over senders (4 heads). | 16,384 | 12,288 bits/step |
| `graph` | Attention restricted to a topology mask (default: fully connected). | 16,416 | 12,288 bits/step |

All methods share one frozen protocol (γ=0.9, entropy coef 0.1, lr 5e-5, λ=0.9, 6000 frames/batch,
5 minibatch iters, message_dim 32, self-communication excluded). **No method was tuned separately** —
that is the point of the study. Attention and graph cost 2× the bandwidth because they transmit a
key and a value per edge; broadcast and gated transmit only a value.

---

## 2. Metric glossary

Read this before the tables.

| Metric | Definition | How to read it |
|---|---|---|
| **Final return** | Mean episode return over the last 10% of evaluation points (5 deterministic episodes every 12k frames). | Less negative is better. Range here ≈ −450 to −500. |
| **AUC** (normalized) | Trapezoidal integral of return vs. frames, divided by the frame span. | A *sample-efficiency* measure: rewards learning early, not just ending well. A method can have better AUC and worse final return. |
| **95% CI** | Fixed-RNG bootstrap over seeds (not over episodes). | Spread across random initializations. Wide here because seed variance dominates. |
| **Matched-seed difference** | Per-seed difference vs. the Identity control, then bootstrapped. | The load-bearing statistic. Pairing cancels the shared seed-level noise that swamps the raw curves. |
| **dz** | Cohen's d for paired differences (mean diff / SD of diffs). | 0.2 small, 0.5 medium, 0.8 large. Says how *consistent* the effect is across seeds, not how big it is in return units. |
| **Communication saliency** | Take the frozen trained policy, force every message to zero (channel dropout p=1, bias-free decoders → exactly the Identity computation), re-evaluate on the same environment seeds, report the return lost. | **The causal test of whether the channel is doing anything.** Must be exactly 0.0 for MLP and Identity — that is the built-in control. |
| **Relative saliency** | Saliency ÷ \|return with comm\|. | Fraction of achieved performance that depends on messages. |
| **Action shift** | Mean L2 change in the deterministic action when messages are severed. | Large shift + small saliency = the channel changes behaviour without helping the objective. |
| **Realized bits/step** | Message scalars actually delivered per environment step × 32 bits, after dropout / topology / budget masking. | The real bandwidth cost. Distinct from *nominal* bits, which ignores masking. |
| **Attention entropy** | Entropy of the attention distribution over senders. | With self excluded and 3 agents there are 2 senders, so the maximum is **ln 2 = 0.693**. At the max, attention is uniform and equivalent to mean-pooling. |
| **ESS** | Effective sample size of PPO importance weights. | Near 1.0 = the policy is not drifting far from the behaviour policy. Healthy. |
| **Clip fraction** | Share of samples hitting the PPO ratio clip. | 0.01–0.1 is healthy; approaching 1.0 means the update is being fully clipped, i.e. divergence. |
| **Explained variance** | How much return variance the critic predicts. | 1.0 = perfect critic, 0 = no better than predicting the mean. |

---

## 3. Validation: can these numbers be trusted?

Four independent checks, all passed.

**3.1 The saliency control is exact.** `benchmarl_mlp` and `identity` both report saliency of
**exactly 0.000** with a [0.0, 0.0] interval. Severing a channel that carries nothing must change
nothing, and it does. This simultaneously proves the severing machinery is wired to the right
modules and that the two controls really are non-communicating.

**3.2 The comm scaffold is a no-op when it should be.** Identity has *exactly* the same parameter
count as the stock MLP (18,948) and its matched-seed difference vs. the MLP is **−2.1 return points,
CI [−15.1, 10.4], dz = −0.13** — statistically indistinguishable. Inserting our communication layer
between encoder and action head does not itself change the agent. (Per-seed they are no longer
bit-identical as they were on CPU; on GPU the extra tensor ops reorder floating-point reductions and
the two trajectories diverge chaotically over 600k frames. The paired test confirms the divergence
is unbiased, which is the correct expectation.)

**3.3 Runs reproduce exactly across independent jobs.** Twelve configurations appear in multiple
suites under different run IDs (e.g. `attention` at message_dim 32 = attention heads 4 = rounds 1 =
dropout p0.00 = self-comm excluded = sender budget k3 — these are all the same configuration reached
from different ablation axes). All of them agree **to 13 significant figures across up to 8
separately scheduled Slurm jobs**. Determinism holds end-to-end.

That also resolves the "mixed GPU models" warning that every per-suite report emits: the two models
are Tesla V100-PCIE-**16GB** and V100-PCIE-**32GB** — the same Volta `sm_70` architecture differing
only in memory capacity. Eight independent jobs across that pool produced identical numbers, so the
split is numerically immaterial here. The warning is keying on the device *name string* and is a
false positive in this case. It would be a real warning for a V100/H100 mix. (The commit is titled
`switch to h100`, but the jobs still landed on V100s — the pin did not take effect. It did not
matter, but it is worth fixing before any run you intend to compare against these.)

**3.4 Training is numerically healthy.** Across all 30 main-comparison runs:

| Diagnostic | Observed range | Verdict |
|---|---|---|
| Final entropy | 1.12 – 1.24 | Stable, no collapse |
| Clip fraction (max) | 0.017 – 0.056 | Healthy PPO |
| ESS (min) | 0.983 – 0.995 | Near-perfect importance sampling |
| Grad norm (max) | 6.1 – 8.0 | Tame |
| Deterministic action saturation | **0.0000 everywhere** | The V1 TanhNormal failure mode is gone |
| Non-finite values | 0 of 30 runs | Clean |

This is the V2 protocol doing its job. The V1 study died of TanhNormal boundary saturation; there is
now zero saturation in deterministic rollouts across every baseline run.

**One honest caveat:** critic **explained variance is only ~0.02** (median across the 30 runs, max
0.10). The value function barely beats predicting the mean. With γ=0.9 the effective horizon is ~10
steps and returns are dominated by random spawn geometry, so this is plausible for the task rather
than a defect — but it means advantage estimates are noisy, the policy-gradient signal is weak, and
that is part of why all methods land within ~14 return points of each other.

---

## 4. Main comparison (6 methods × 5 seeds)

| Method | Final return | 95% CI | AUC | Saliency | Bandwidth |
|---|---:|---:|---:|---:|---:|
| **broadcast** | **−462.4** | [−472.4, −452.3] | −552.2 | 31.6 | 6,144 |
| attention | −464.9 | [−476.4, −453.4] | −558.0 | 32.2 | 12,288 |
| gated | −465.6 | [−475.8, −455.5] | −558.7 | 10.8 | 6,144 |
| graph | −468.4 | [−478.5, −457.4] | −560.0 | 34.1 | 12,288 |
| identity *(control)* | −474.3 | [−482.9, −465.8] | −559.5 | **0.0** | 0 |
| benchmarl_mlp *(control)* | −476.4 | [−490.5, −463.7] | −556.4 | **0.0** | — |

### Matched-seed difference vs. the no-communication control

| Method | Mean difference | 95% CI | dz |
|---|---:|---:|---:|
| broadcast | **+11.9** | [9.6, 14.4] | **3.95** |
| attention | +9.4 | [5.2, 13.7] | 1.69 |
| gated | +8.7 | [6.3, 11.7] | 2.46 |
| graph | +5.9 | [2.8, 9.0] | 1.45 |
| benchmarl_mlp | −2.1 | [−15.1, 10.4] | −0.13 |

**Findings.**

- **Communication helps, reliably but slightly.** All four comm methods beat the no-comm control with
  intervals excluding zero and large paired effect sizes (dz 1.45–3.95). But the magnitude is
  **+6 to +12 return points on a −474 baseline — about 1.3% to 2.5%**. Consistent, not dramatic.
- **Saliency agrees with the comparison and is larger than it.** Severing messages on a trained
  broadcast/attention/graph policy costs ~32 points (~6% of return). Communication contributes more
  to the *trained policy* than the training comparison suggests, which means part of the benefit is
  absorbed into learning dynamics rather than showing up as final-score separation.
- **Complexity buys nothing.** Broadcast — the simplest mechanism, plain mean-pooling — wins on final
  return, wins on AUC, and does it at **half the bandwidth and half the comm parameters** of
  attention and graph. Attention's extra machinery is not paying for itself.
- **Why attention doesn't help:** measured attention entropy is **0.6905 against a maximum of
  0.6931** — 99.6% of uniform, with max attention probability 0.524 ≈ 0.5. **The attention layer
  learned to attend uniformly, i.e. it degenerated into exactly the mean-pooling that broadcast does
  for free.** In a fully observable task with 3 homogeneous agents there is no selection problem to
  solve, so there is nothing for attention to learn. Graph is marginally more selective (entropy
  0.668, max prob 0.582) and performs slightly worse.
- **Gating suppresses dependence.** Gated has mean gate 0.53 with gates "open" 81% of the time, and
  the **lowest saliency of any comm method (10.8 vs ~32)** — yet nearly the same return. The gate
  damps message magnitude, the policy learns to rely on messages less, and it loses almost nothing.
  Another sign the messages are near-redundant.

**A caveat on reading the learning curves:** in `results/simple_spread_comm_v2/learning_curves.png`
the six methods are visually indistinguishable and the per-seed bands overlap completely. That is
not a failure — between-seed variance is an order of magnitude larger than between-method variance.
The signal is only recoverable because the design is matched-seed and the analysis is paired. Also,
**the curves are still rising at 600k frames**: nothing here is converged, and these are rankings at
a fixed budget, not asymptotic rankings.

---

## 5. Ablations (201 rows across 7 suites)

### 5.1 Message dimension — more capacity helps, and destabilizes

| dim | broadcast return | broadcast saliency | attention return | attention saliency |
|---:|---:|---:|---:|---:|
| 8 | −471.2 | 26.1 | −474.4 | 5.6 |
| 16 | −470.2 | 30.2 | −470.0 | 14.5 |
| 32 | −467.7 | 42.5 | −468.4 | 39.8 |
| 64 | −465.0 | **75.8** | −458.2 † | **53.2** |

† 2 of 3 seeds; see §6.

Monotone in both return and saliency. Wider messages carry more and the policy depends on them more —
saliency roughly triples from dim 8 to dim 64. Returns improve only ~6 points across a 3× parameter
increase, so the capacity is being used but the task can't cash it in. **Message_dim 64 is also where
runs start diverging.**

### 5.2 Channel dropout — a regularizer, with an inverted-U in saliency

| p | broadcast return | broadcast AUC | broadcast saliency | realized bits/step |
|---:|---:|---:|---:|---:|
| 0.00 | −467.7 | −556.5 | 42.5 | 6,144 |
| 0.25 | −466.6 | **−519.4** | **65.4** | 4,599 |
| 0.50 | −472.6 | −520.3 | 42.8 | 3,073 |
| 0.75 | −473.0 | −524.8 | 17.5 | 1,537 |

Two clean effects:

- **AUC improves sharply with any dropout** (−556 → −519) while final return is flat-to-slightly-worse.
  Dropout speeds up *early* learning. The plausible mechanism: early in training all messages are
  uninformative noise, and forcing the policy not to depend on them prevents an unproductive early
  reliance. The effect is driven by the seeds that otherwise learn slowly.
- **Saliency is an inverted U.** Moderate dropout (p=0.25) makes the policy depend on messages
  *more* than no dropout (65.4 vs 42.5) — it learns a robust, genuinely-used channel. Heavy dropout
  (p=0.75) makes it depend on them much less (17.5) — it learns to ignore an unreliable channel.
  p=0.25 delivers the highest saliency at **25% lower bandwidth** than p=0.

### 5.3 Communication rounds — hard failure at ≥2

| rounds | attention return | graph return | saliency |
|---:|---:|---:|---:|
| 1 | −468.4 | −470.8 | ≈ +39 |
| 2 | −551.2 (2/3 seeds) | −476.2 (1/3 seeds) | +17 / +70 |
| 3 | −785.0 (1/3 seeds) | −807.1 | **−325 / −158** |

**This is the clearest negative result in the study.** Multi-round communication does not just fail
to help — it destroys the policy, and 5 of 18 runs diverged outright. At 3 rounds saliency goes
**negative**: severing all communication *improves* return by 158–325 points. The comm stack has
become an active liability. Only rounds=1 is usable. Details in §6.

### 5.4 Attention heads — no effect

| heads | attention | graph |
|---:|---:|---:|
| 1 | −467.1 | −476.9 |
| 2 | −467.7 | −474.3 |
| 4 | −468.4 | −470.8 |
| 8 | −468.1 | −469.2 |

Attention is flat across 1–8 heads (spread 1.3 points, far inside the CIs). Consistent with §4:
the attention distribution is uniform, so head count is irrelevant. Graph improves mildly and
monotonically with heads (−476.9 → −469.2), the only place head count matters at all.

### 5.5 Graph topology — connectivity is what matters

| topology | return | saliency |
|---|---:|---:|
| full | −470.8 | 39.2 |
| Erdős–Rényi p0.5 (connected) | −478.2 | 26.9 |
| directed ring | −474.7 | **−1.2** |

Sparsifying the graph costs both return and saliency. The directed ring — each agent hears exactly
one other — has saliency ≈ **0**: the policy became functionally non-communicating. A single
neighbour's message was not worth learning to use.

### 5.6 Sender budget — the one place learned selection demonstrably works

| budget | attention return | saliency | realized bits/step |
|---|---:|---:|---:|
| learned top-1 | **−466.0** | 37.0 | 4,096 |
| learned top-2 | −468.9 | 31.5 | 8,192 |
| learned top-3 (all) | −468.4 | 39.8 | 12,288 |
| random 1 | −476.5 | **5.5** | 4,096 |
| random 2 | −471.2 | 49.1 | 8,192 |
| random 3 (all) | −468.4 | 39.8 | 12,288 |

**Learned top-1 matches full communication at one third the bandwidth** (−466.0 vs −468.4). At the
same 4,096 bits/step, *random* selection is 10.5 points worse and nearly non-salient (5.5).

This is the important nuance to §4: the soft attention weights are near-uniform, but the underlying
scores still **rank** senders usefully. Attention isn't useless — its *soft averaging* is what
adds nothing; its *ranking* is worth 10.5 return points once you force a hard choice. That is a
directly transferable result for bandwidth-constrained settings.

### 5.7 Self-communication — must stay excluded

| method | excluded | included | saliency excluded → included |
|---|---:|---:|---|
| attention | −468.4 | −469.7 | 39.8 → **5.1** |
| broadcast | −467.7 | −470.4 | 42.5 → **−0.5** |
| graph | −470.8 | −472.5 | 39.2 → **12.8** |

**Returns barely move; saliency collapses to ~zero.** When an agent may attend to its own message, it
does — and stops using anyone else's. Its own observation is already available through the encoder,
so a self-message is redundant but far easier to learn, and it crowds out genuine inter-agent
communication. Broadcast goes slightly *negative*: the channel became pure overhead.

This is a methodological trap, not a hyperparameter: **with self-communication enabled you can run a
whole "communication" study in which no agent ever communicates, and the return column will not tell
you.** Only saliency catches it. Keep self-communication excluded, and keep measuring saliency.

---

## 6. What failed, and why

7 of 231 runs (3.0%) failed. All 7 have an **identical signature** and all 7 are the same root cause.

| Suite | Config | Failed | Diverged at |
|---|---|---:|---|
| message_dim | attention dim 64 | 1/3 seeds | 438k frames |
| message_dim | graph dim 64 | 1/3 seeds | 348k frames |
| rounds | attention rounds 2 | 1/3 seeds | 408k frames |
| rounds | attention rounds 3 | 2/3 seeds | 324k / 342k frames |
| rounds | graph rounds 2 | 2/3 seeds | 270k / 390k frames |

Every failure is a **numerical divergence in the communication stack**, surfaced as an
`AssertionError` from BenchMARL when a non-finite value reaches the loss. The audit shows the causal
chain, and `comm_max_message_norm` / `comm_mean_message_norm` go non-finite *in the same step* as
everything else:

```
comm message norm ──▶ TanhNormal location explodes ──▶ PPO ratio explodes
   ──▶ entropy → 5.8e5 … 2.5e6      (healthy: ~1.2)
   ──▶ kl_approx → up to 2.2e6      (healthy: <0.2)
   ──▶ clip_fraction → 0.987        (healthy: ~0.03)
   ──▶ ESS → 0.016                  (healthy: ~0.99)
   ──▶ grad_norm → up to 1.7e13     (healthy: ~7)
   ──▶ NaN
```

**The pattern is completely consistent:**

- It only ever hits **attention and graph** — the two modules with a softmax-normalized aggregation
  and a stacked residual path. **Broadcast and gated never failed once**, in any configuration.
- It only appears at the **high-capacity / deep end**: message_dim 64, or rounds ≥ 2. All 224 rows at
  the default (dim 32, rounds 1) are finite.
- The runs that *didn't* NaN at rounds ≥ 2 were still catastrophically bad (−551, −785, −807), so
  this is not bad luck on a few seeds — multi-round composition is genuinely unstable under this
  protocol.

**Diagnosis:** the comm stack has no normalization and no gradient clipping on the message path, so
repeated attention-weighted aggregation compounds activation magnitude. One round at dim 32 stays
in range; two or three rounds, or a 64-D message, does not.

**This is a contained, well-understood boundary, not a mystery.** The default configuration is
stable in all 224 completed rows. No completed row was silently corrupted — the audit checks every
logged value for finiteness and all 224 passed.

---

## 7. Are these results what we should have expected?

**Yes — and the ways they are underwhelming are informative rather than alarming.**

| Observation | Expected? | Why |
|---|---|---|
| Communication helps by only 1–3% | **Yes** | The task is fully observable. Every agent already sees every other agent and every landmark, so messages carry no *information* the receiver lacks — only computation. A small gain is the correct outcome; a large one would have been suspicious. |
| Broadcast beats attention and graph | **Yes** | With 3 homogeneous agents and no information asymmetry, no sender is more relevant than another, so selection has no purchase. Attention converged to 99.6% of uniform — it *rediscovered* mean-pooling at 2× the cost. |
| Head count doesn't matter | **Yes** | Follows directly: you cannot usefully split a uniform attention distribution across heads. |
| Wider messages → higher saliency, flat return | **Yes** | Capacity is used (saliency triples) but the task has no more value to extract. |
| Sparser topology → lower saliency | **Yes** | Fewer senders, less to gain, and at one sender the policy stops bothering. |
| Self-communication zeroes out saliency | **Yes**, and it is the most useful methodological finding here | Self-messages are redundant-but-easy, so they win the credit-assignment race against real communication. |
| Dropout improves AUC | **Mildly surprising, coherent** | Early messages are noise; forcing independence early is a useful regularizer. Worth keeping as a technique. |
| Multi-round diverges | **Not expected, and it is a real defect to fix** | Unnormalized residual stacking. See §8. |
| Critic explained variance ≈ 0.02 | **Plausible, worth watching** | γ=0.9 gives a ~10-step horizon and returns are dominated by spawn geometry. But it does mean the learning signal is weak, which contributes to the small effect sizes. |

**The honest summary:** Simple Spread validated the *infrastructure* thoroughly and told us relatively
little about the *methods*, because the task cannot distinguish them. That is exactly the outcome a
controlled baseline is supposed to produce before moving to a task that can. The five modules all
train, all produce measurable and correctly-signed saliency, and rank in a way fully explained by
the environment's structure.

**What this study genuinely established:**

1. The comm-injection scaffold is a verified no-op when the module is Identity (§3.2).
2. Saliency is a working causal metric with exact zero on both controls (§3.1).
3. The V2 protocol eliminated the V1 TanhNormal saturation failure (§3.4).
4. The full pipeline is deterministic and reproducible across independent jobs (§3.3).
5. Two transferable design rules: **exclude self-communication**, and **learned top-k selection buys
   a 3× bandwidth reduction for free**.
6. A known stability boundary: **rounds ≥ 2 and message_dim ≥ 64 diverge for attention/graph**.

---

## 8. Readiness verdict

### **READY TO IMPLEMENT PCP / PP** — with one required fix first.

**Why ready.** Everything this phase was supposed to de-risk is de-risked. 97% of runs completed;
the failures are explained, contained, and confined to non-default settings. Controls behave exactly
correctly. The harness is deterministic, resumable, audited, and produces analysis-ready output at
scale. All five communication modules are implemented, validated, and characterized.

**Why moving on is not merely permitted but *necessary*.** Simple Spread is fully observable with
homogeneous agents. It structurally cannot discriminate between communication mechanisms — the
uniform attention result is proof that the task poses no selection problem. Running more ablations
here will not produce more signal. PP and PCP supply exactly what is missing: **partial observability**
and **heterogeneous roles** (in PCP, capture agents cannot subdue prey alone and observing agents
carry information the captors lack). That is genuine information asymmetry — the precondition for
communication to matter, and the setting where broadcast, gated, attention, and graph should finally
separate.

**Required before launching PCP/PP** — because those tasks will push exactly the settings that broke:

1. **Fix the comm-stack instability (§6).** Add normalization (LayerNorm on the message path) and/or
   gradient clipping on the comm module. PP/PCP have more agents and richer observations, so a larger
   `message_dim` will be tempting, and 64 is already where attention and graph diverge. Re-verify
   with the same finiteness audit before running a full suite.
2. **Keep self-communication excluded** and **keep saliency as a first-class metric** — it is the only
   thing that would have caught the self-communication trap, and in a partially observable task a
   silently non-communicating baseline is a much more expensive mistake.

**Recommended, not blocking:**

3. **Re-check the protocol on the new tasks.** γ=0.9 and 600k frames were frozen for Simple Spread.
   PP/PCP have longer credit-assignment horizons; γ=0.9 (≈10-step horizon) is likely too short, and
   the ~0.02 explained variance suggests the critic is already strained here.
4. **Pin the GPU model.** The `switch to h100` commit did not take effect and everything ran on V100s.
   Harmless this time (§3.3), but fix the submission before generating numbers you want to compare.
5. **Budget more seeds or more frames.** Effect sizes will hopefully be larger in PP/PCP, but 3 seeds
   per ablation cell was thin — several ablation CIs here span 30+ return points.

### Suggested first move in the new phase

Port `identity` and `broadcast` only, run the same matched-seed protocol gate on PP, and check that
**saliency is materially larger than the ~32 points seen here**. If communication still contributes
only ~6% of return in a partially observable task, the problem is the protocol, not the mechanisms —
and it is much cheaper to learn that from two modules than from five plus a full ablation grid.

---

## Appendix — suite inventory

| Suite | Rows | Completed | GPU-h | Artifacts |
|---|---:|---:|---:|---|
| `simple_spread_comm_v2` (main) | 30 | 30 (100%) | 5.2 | `results/simple_spread_comm_v2/` |
| `..._stage2_message_dim` | 48 | 46 (96%) | 8.7 | 2 diverged @ dim 64 |
| `..._stage2_dropout` | 48 | 48 (100%) | 9.2 | |
| `..._stage3_rounds` | 18 | 13 (72%) | 3.0 | 5 diverged @ rounds ≥ 2 |
| `..._stage3_heads` | 24 | 24 (100%) | 4.8 | |
| `..._stage4_graph_topology` | 9 | 9 (100%) | 1.8 | |
| `..._stage4_sender_budget` | 36 | 36 (100%) | 6.9 | |
| `..._stage4_self_communication` | 18 | 18 (100%) | 3.5 | |
| **Total** | **231** | **224 (97%)** | **43.2** | |

Each suite directory contains `REPORT.md`, six or seven figures, and six CSVs
(`_summary`, `_per_run`, `_saliency`, `_paired_comparisons`, `_run_audit`, `_failed_runs`).
