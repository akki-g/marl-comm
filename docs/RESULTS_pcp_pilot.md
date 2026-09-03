# PCP identity pilot — results and verdict (2026-09-03)

Suite `pcp_identity_pilot`, 2 rows, both `completed`, run on one V100 under array job
793465 at commit `3bb11aa`. Raw output in `runs/pcp/`.

**Verdict: do not launch `pcp_comm_main` or the ablations.** The runs are healthy and the
harness is correct, but three things must change first, and one of them is a property of
the environment rather than of the configuration. The pilot answered the question it was
asked — *does anything learn on PCP?* — and the answer is *yes, weakly*. It also surfaced a
larger problem it was not asked about.

---

## 1. What the two rows say

| | 60,000 frames | 120,000 frames |
|---|---|---|
| status | completed | completed |
| wall clock | 232 s | 471 s |
| eval points | 6 | 11 |
| `mean_final_return` (corrected) | 0.00 | **7.00** |
| `mean_final_return` (as shipped) | 0.00 | 2.33 |
| comm parameters | 0 | 0 |

Throughput is **255 frames/s** including evaluation — 23.5 s per 6,000-frame iteration.
That is the single most useful number for planning: budget is not the binding constraint
here.

### Learning is real but weak and nowhere near a plateau

The five-episode deterministic evaluation is too noisy to read (§2), so the learning signal
comes from the collection returns, which average ~60 episodes per iteration:

```
frames    6k   12k   18k   24k   30k   36k   42k   48k   54k   60k
adv ret 0.67  1.33  0.67  1.67  3.17  1.00  1.00  2.50  2.00  1.17
frames   66k   72k   78k   84k   90k   96k  102k  108k  114k  120k
adv ret 1.00  2.83  1.17  3.00  4.17  1.67  2.00  2.00  3.50  3.67
```

- First half **1.52 ± 0.26**, second half **2.50 ± 0.35** (Welch *t* = 2.28, *p* = 0.036).
- Random-policy baseline, measured over 300 episodes: **0.83**.
- Trend **+1.75 return per 100k frames**, with no sign of flattening.
- Policy entropy fell only 1.304 → 1.146 over the whole run; `ESS` ≈ 0.99 and
  `kl_approx` ≈ 0.003 every iteration. The policy barely moved.

The reason is arithmetic: 120,000 frames ÷ 6,000 per batch = 20 iterations, × 10 minibatch
epochs = **200 optimizer steps** at `lr = 5e-5`. Critic `explained_variance` did climb
0.04 → 0.5, so the value function is learning; the policy simply has not had enough
updates. **120k frames is roughly a tenth of what this task needs.**

---

## 2. Three defects, in order of cost

### 2.1 The environment has no information asymmetry — *this is the expensive one*

`docs/RESULTS_v2_simple_spread.md:400` states the reason for moving to PCP:

> PP and PCP supply exactly what is missing: **partial observability** and **heterogeneous
> roles** (in PCP, capture agents cannot subdue prey alone and observing agents carry
> information the captors lack).

**That environment does not exist.** `PredatorCapturePreyScenario` overrides
`process_action` only; it inherits `observation()` unchanged from VMAS `simple_tag`.
Measured directly:

```
adversary_0  obs_dim=16  u_mult=3.0  max_speed=1.0  radius=0.075
adversary_1  obs_dim=16  ...
adversary_2  obs_dim=16  ...
agent_0      obs_dim=14  u_mult=4.0  max_speed=1.3  radius=0.05
```

Those 16 dimensions are: own velocity (2), own position (2), both landmarks' relative
positions (4), **the prey's relative position (2)**, both teammates' relative positions
(4), and **the prey's velocity (2)**. Every predator already knows exactly where the prey
is, how fast it is moving, and where both teammates are, at every step.

A communication channel can only re-transmit what the receiver already has. This is the
*same* structural condition that produced the Simple Spread null result, and that document
is explicit about what follows (`:360`): *"The task is fully observable... messages carry
no information the receiver lacks — only computation. A small gain is the correct outcome."*

Running 231 ablation rows on this task as configured would reproduce that null at higher
cost. The predators' problem here is **coordination**, not information — and MAPPO's
centralised critic already handles coordination without a channel.

Note this is not a regression; the repo says so in two places already
(`README.md:174`, `agents.md:1267`), both deferring the role split as "a modeling decision,
not a fix." The pilot is what makes it *blocking*: the study cannot measure what it is for
until that decision is made.

### 2.2 Five evaluation episodes cannot measure anything

`experiment.evaluation_episodes: 5`. Measured over 300 random-policy episodes:

| statistic | value |
|---|---|
| mean per-episode return | 0.833 |
| std | 3.789 |
| episodes scoring exactly 0 | **94.0%** |
| mean contact-steps per episode | 0.083 / 100 |

| episodes | standard error |
|---|---|
| 5 (current) | **1.69** |
| 32 | 0.67 |
| 128 | 0.34 |
| 256 | 0.24 |

The effect sizes in play are 1–3 return points. At n = 5 the standard error is **larger
than the entire effect**, and because 94% of episodes are exactly zero the estimator is a
rare-event counter whose 5-draw distribution is nowhere near Gaussian. This is visible in
the raw pilot data — the entire 120k eval curve is
`0, 0, 0, 2, 0, 0, 0, 8, 2, 12, 2`, where every non-zero point is one or two lucky
episodes out of five.

**This is nearly free to fix.** For a batched environment BenchMARL passes
`evaluation_episodes` straight to the eval env's `num_envs`
(`benchmarl/experiment/experiment.py:449`) and does a *single* batched rollout of
`max_steps` (`:932`). Going 5 → 128 widens the batch but runs the same 100 sequential
steps. Observed cost at n=5 is 5.7 s per evaluation; at n=128 it should be within a small
factor, against a 23.5 s iteration.

### 2.3 The pilot ran a protocol no PCP suite uses — and `pcp_comm_main`'s γ is wrong for this task

The alignment fix was written but **never committed**; the cluster pulled `3bb11aa` without
it (`git diff configs/sweeps/pcp_identity_pilot.yaml` still shows it pending). So:

| | pilot (as run) | `pcp_comm_main` |
|---|---|---|
| `experiment.gamma` | **0.99** | **0.9** |
| `entropy_coef` | **0.0** | **0.1** |
| `on_policy_n_minibatch_iters` | **10** | **5** |

The budget question was answered for a configuration no suite runs. Worse, the direction is
unfavourable: `pcp_comm_main` inherits Simple Spread's frozen γ = 0.9, and the Simple Spread
report itself flags this (`:419`): *"PP/PCP have longer credit-assignment horizons; γ=0.9
(≈10-step horizon) is likely too short."* The geometry confirms it:

- The prey's `max_speed` is **1.3 vs the predators' 1.0** — it is strictly faster, so a
  direct one-predator chase *never converges*. Capture requires a multi-predator pincer.
- Nearest predator starts 0.64 away; ~6.4 steps to close at full speed *if the prey held
  still*. Setting up a pincer takes far longer.
- γ = 0.9 discounts a reward 20 steps out to **0.12**; γ = 0.99 leaves it at **0.82**.

The manoeuvres that make a capture possible are effectively invisible to a γ = 0.9 critic.
The pilot learned *because* it accidentally ran γ = 0.99.

### 2.4 Fixed already: `summary.json` divided PCP returns by three

`summarize_metrics_csv` in `experiments/bookkeeping.py` collected every evaluation
`return_mean` row regardless of group. PCP writes three per evaluation — the ungrouped
study figure plus one per BenchMARL group — and its two groups are exactly zero-sum, so the
average was `(x + x + −x)/3 = x/3`. The 120k row's true `mean_final_return` is **7.00**, not
the 2.33 it reported.

The analysis path was never affected: `_evaluation_curve` in `analysis/aggregate.py` already
filters `not row.get("group")`, so everything under `results/` is correct. A single-group
task cannot expose the bug — there the extra row equals the study figure — which is why
Simple Spread never caught it, and why its numbers are bit-identical after the fix.

Fixed, with `test_summary_ignores_per_group_returns_on_a_two_group_task` pinning the
two code paths together. **503 tests pass.**

### Not a defect

`comm_*` metrics are all exactly 0.0 and `communication_total` is 0 — correct for the
`identity` control, and the expected wiring-check value.

---

## 3. Plan — Gates A and B are implemented, C is configured, D is blocked on C

### Gate A — information asymmetry — **DONE**

`predator_sensing_radius` on `PredatorCapturePreyScenario`. Beyond that distance a
predator's view of the prey's position *and* velocity is zeroed and one visibility flag per
prey is appended, taking the predator observation from 16 dims to 17. Predators still see
each other and the landmarks, so the only thing a message can carry that its receiver lacks
is **where the prey is** — and at any step only some predators can answer that, and which
ones changes as the chase moves. That is dynamic, state-dependent sender relevance: the
selection problem Simple Spread could not pose, and the condition under which attention and
gating should beat broadcast rather than converge to it (Simple Spread: attention reached
99.6% of uniform at 2× the cost).

The masked block is located by counting prey, not by a hardcoded index —
`SimpleTagScenario.observation` walks `world.agents` in order and `make_world` adds every
adversary before every good agent, so for a predator the tail is always all prey positions
followed by all prey velocities. Verified against the stock observation: the head is
bit-identical, the prey block is bit-identical where visible and exactly zero where not, and
the layout generalises to 2 prey / 4 predators / 3 landmarks.

**Radius calibrated to 1.0** against the random-policy geometry, where mean predator–prey
distance is 1.08:

| R | mean seers / 3 | P(none see) | P(exactly one sees) | P(all three see) |
|---|---|---|---|---|
| 0.6 | 0.49 | 0.583 | 0.345 | 0.003 |
| 0.8 | 0.95 | 0.309 | 0.460 | 0.028 |
| **1.0** | **1.41** | **0.142** | **0.409** | **0.103** |
| 1.2 | 1.85 | 0.051 | 0.298 | 0.246 |
| 1.6 | 2.51 | 0.004 | 0.076 | 0.595 |

At R = 1.0 the team is blind only 14% of steps — low enough to bootstrap — exactly-one-seer
is the **modal** case at 41%, and it collapses to full observability only 10% of the time.
R = 0.6 (the first guess) is too harsh: nobody sees the prey 58% of the time, and
information no one holds cannot be communicated. R = 1.2 is the conservative alternative.

The scenario default stays `None` (stock, fully observable) so the pilot's numbers still
describe a reproducible configuration; `configs/tasks/vmas_predator_capture_prey.yaml` sets
1.0. Verified end to end: `pcp_comm_attention` builds and trains against the 17-dim
observation.

**Nothing measured before 2026-09-03 is comparable to post-Gate-A rows**, including the
0.83 random baseline in §2.2. Re-measure it.

### Gate B — fix the measurement — **DONE**

`experiment.evaluation_episodes: 128` on every PCP comparison sweep and on the protocol
gate. The claim that this is nearly free was measured, not assumed:

| episodes | evaluation wall-clock | ms/episode |
|---|---|---|
| 5 | 0.99 s | 198.8 |
| 32 | 1.04 s | 32.4 |
| **128** | **1.23 s** | **9.6** |
| 256 | 1.63 s | 6.4 |

**+24% for 25× the samples**, taking the standard error from 1.69 to 0.34. Even 256 is only
+65%. Against a 23.5 s iteration this is noise. Do not lower it.

Saliency stays first-class — `RESULTS_v2_simple_spread.md:413` is right that it is the only
thing that catches a silently non-communicating baseline, and under real partial
observability that mistake gets much more expensive.

### Gate C — recalibrate the protocol — **CONFIGURED, NOT RUN**

`configs/sweeps/pcp_protocol_gate.yaml` + `slurm/pcp_02b_protocol.sbatch`. The
no-communication control at γ ∈ {0.9, 0.95, 0.99} × `entropy_coef` ∈ {0, 0.01, 0.1}, three
seeds, 600k frames: **27 rows, ~18 GPU-hours**, under a tenth of the comparison it protects.
At 600k the final-10% window averages 5 evaluation points rather than the pilot's 1.

This supersedes `pcp_identity_pilot`, whose question (60k or 120k?) is answered: neither.

Read off (a) the winning cell, and (b) where its curve flattens — that frame count, not
600k, is what the comparison sweeps should carry. If γ = 0.9 wins, something in §2.3 is
wrong; find out what before Gate D.

*Contingency, not a default:* if learning still fails, `shape_adversary_rew: True` adds a
dense `−0.1 × min-distance` term. Note the confound before using it — dense distance shaping
hands every predator a private gradient toward the prey, which partially substitutes for the
very messages the study is measuring. Under Gate A it would also leak the prey's bearing
past the sensing radius through the reward, undoing the asymmetry.

### Gate D — run the suite, main first — **BLOCKED on C**

All eight `pcp_comm_*.yaml` now carry a `BLOCKED` header naming the three placeholder
settings (`max_n_frames` 60000, γ 0.9, `entropy_coef` 0.1) and pointing at the gate that
decides them; a test asserts the header is there. Take all three from the gate's winner,
delete the header, rerun `pcp_01_setup`, then submit.

Run the 30-row main comparison first and gate the 201 ablation rows on whether it separates
the mechanisms at all. If broadcast, gated, attention and graph land on top of each other
again *after* Gate A, the ablation grid will not rescue it — that is the same lesson
`RESULTS_v2_simple_spread.md:430` draws, and V1 is preserved in this repo as evidence of
what launching a full grid under an unvalidated protocol costs. At a 600k–1M budget the
full 231 rows are roughly **190–250 GPU-hours**.

---

## 4. Submit sequence

```bash
sbatch slurm/pcp_01_setup.sbatch       # rebuilds manifests, now incl. the gate
sbatch slurm/pcp_02b_protocol.sbatch   # 27 rows, ~18 GPU-h  <- READ THIS
# then freeze gamma/entropy/budget into configs/sweeps/pcp_comm_*.yaml,
# rerun pcp_01_setup, and only then:
sbatch slurm/pcp_03_main_comparison.sbatch
```

`slurm/pcp_02_pilot.sbatch` is superseded and should not be resubmitted.

## 5. Files changed

- `src/commstudy/tasks/vmas/scenarios/predator_capture_prey.py` — `observation()` override
  and the `predator_sensing_radius` knob (Gate A).
- `configs/tasks/vmas_predator_capture_prey.yaml`, `configs/tasks/defaults/predator_capture_prey.yaml`
  — radius 1.0 with the calibration recorded.
- `configs/sweeps/pcp_protocol_gate.yaml`, `slurm/pcp_02b_protocol.sbatch` — Gate C.
- `configs/sweeps/pcp_comm_*.yaml` (8) — `evaluation_episodes: 128`, `BLOCKED` header.
- `slurm/pcp_01_setup.sbatch` — builds the gate manifest.
- `src/commstudy/experiments/bookkeeping.py` — group filter in `summarize_metrics_csv`.
- `src/tests/test_bookkeeping.py`, `test_pcp_integration.py`, `test_slurm_scripts.py` — tests
  for all of the above.
