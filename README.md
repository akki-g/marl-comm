# marl-comm

Controlled learned-communication experiments for cooperative multi-agent
reinforcement learning. Every row pairs one VMAS task, BenchMARL MAPPO, one
centralized MLP critic, and a swappable actor-side `CommModule`; only the
communication mechanism changes.

Two tasks are available. Stock VMAS Simple Spread is the reference task and
carries the completed architecture comparison
([docs/RESULTS_v2_simple_spread.md](docs/RESULTS_v2_simple_spread.md)).
Predator-Capture-Prey (PCP) is a second task family for the same comparison:
three trained predators pursuing one scripted prey, described under
[Predator-Capture-Prey](#predator-capture-prey). Role/class-id conditioning on
PCP and the PP task both remain out of scope.

The infrastructure underneath is used unmodified: the simulator is VMAS
([Bettini et al., DARS 2022](https://arxiv.org/abs/2207.03530)), the
experiment/config layer is BenchMARL (Bettini, Prorok, and Moens, *JMLR*
25(217), 2024), and every row trains with BenchMARL's MAPPO
([Yu et al., NeurIPS 2022 Datasets and Benchmarks](https://arxiv.org/abs/2103.01955)).

## Setup

Python 3.11 or newer is required. The lockfile is the reproducible source for
the environment:

```bash
uv sync --extra dev --extra analysis
```

Run the quality gates before experiments:

```bash
uv run pytest -q
uv run ruff check src scripts
uv run python -m compileall -q src scripts
```

## Communication methods

| Model config | Method | Published basis | Default channel |
|---|---|---|---|
| `benchmarl_mlp` | Framework reference MLP | BenchMARL | none |
| `comm_identity` | Exact no-communication control | identity | zero messages |
| `comm_broadcast` | learned mean broadcast | CommNet-inspired | 32-float message |
| `comm_gated` | learned sender gate | IC3Net-inspired | 32-float gated message |
| `comm_attention` | targeted sender/receiver attention | TarMAC-inspired | 32-float key + 32-float value |
| `comm_graph` | graph-restricted learned attention | DGN/GAT-inspired | 32-float key + 32-float value |

All learned modules accept `[..., N, D]` and preserve that shape. A
communication mask uses `mask[..., receiver, sender]`. Self communication is
off by default; the local residual always remains available. Full definitions,
paper links, equations, differences from the original algorithms, and cost
conventions are in [docs/communication_methods.md](docs/communication_methods.md).

## One managed run

Every CLI training run is managed: it receives a descriptive run ID, captures
resolved configuration and source provenance, writes tidy metrics, and ends in
an explicit completed or failed state.

Framework/reference baseline:

```bash
uv run python scripts/train.py \
  --suite-id simple_spread_smoke \
  model=benchmarl_mlp seed=0 experiment.max_n_frames=6000 \
  experiment.evaluation=false
```

No-communication actor baseline:

```bash
uv run python scripts/train.py \
  --suite-id simple_spread_smoke \
  model=comm_identity seed=0 experiment.max_n_frames=6000 \
  experiment.evaluation=false
```

Learned modules use the same command with one model selection:

```bash
uv run python scripts/train.py --suite-id simple_spread_smoke \
  model=comm_broadcast seed=0 experiment.max_n_frames=6000

uv run python scripts/train.py --suite-id simple_spread_smoke \
  model=comm_gated seed=0 experiment.max_n_frames=6000

uv run python scripts/train.py --suite-id simple_spread_smoke \
  model=comm_attention seed=0 experiment.max_n_frames=6000

uv run python scripts/train.py --suite-id simple_spread_smoke \
  model=comm_graph seed=0 experiment.max_n_frames=6000
```

Any project/BenchMARL setting can be overridden with an OmegaConf dot key,
for example:

```bash
uv run python scripts/train.py --suite-id diagnostic \
  model=comm_attention seed=2 \
  model_config.params.comm_kwargs.message_dim=16 \
  experiment.max_n_frames=12000
```

Use `--dry-run` to resolve and print a run without creating its run directory
or training. A completed run is skipped. A failed or otherwise existing run is
preserved; use `--retry` to create a clearly named `__retryNN` attempt. Native
BenchMARL restore is not advertised as exact scientific resume because its
checkpoint does not include Adam state.

## Predator-Capture-Prey

PCP is the project's second task. It subclasses VMAS's port of the MPE
`simple_tag` predator-prey scenario
([Lowe et al., NeurIPS 2017](https://arxiv.org/abs/1706.02275)) and replaces the
prey's learned policy with a hand-written pursuit-evasion rule, so the only
trained group is the three predators. The scenario itself is unchanged
otherwise; see `src/commstudy/tasks/vmas/scenarios/predator_capture_prey.py`.

It exists to test whether a communication mechanism's advantage on Simple
Spread generalizes to a structurally different coordination problem. Simple
Spread rewards coverage, so messages mostly help agents avoid claiming the same
landmark. PCP rewards interception of an actively fleeing target, so the
plausible value of a message is pursuit coordination — one predator's view of
the prey is useful to a predator that currently has none. That is much closer
to what TarMAC, ATOC, and DGN were evaluated on. A mechanism that helps on both
is evidence about information sharing under coordination pressure rather than
about one task's geometry; a result that reverses on PCP is itself a finding,
not a bug.

Select the task and its grouped models:

```bash
uv run python scripts/train.py --suite-id pcp_smoke \
  task=vmas_predator_capture_prey \
  model=pcp_comm_attention critic_model=pcp_critic \
  seed=0 experiment.max_n_frames=6000
```

PCP runs two BenchMARL groups — `adversary` (predators) and `agent` (prey) — so
its actor and critic configs are group ensembles rather than flat ones:

| Config | Role |
|---|---|
| `pcp_actor` | framework reference MLP, the PCP equivalent of `benchmarl_mlp` |
| `pcp_comm_identity` … `pcp_comm_graph` | one per `comm_*` model, communication on the `adversary` group only |
| `pcp_critic` | centralized MLP critic for both groups; the critic never carries communication |

Two consequences worth knowing before launching anything:

- **Overrides are group-scoped.** What is `model_config.params.comm_kwargs.X`
  on Simple Spread is `model_config.groups.adversary.params.comm_kwargs.X` on
  PCP. The flat path does not fail — OmegaConf merges it in as a stray key that
  nothing reads — so an ablation written that way runs silently at its default.
- **The prey group trains a policy that is thrown away.** BenchMARL builds and
  optimizes an actor and critic for the `agent` group like any other, but
  `PredatorCapturePreyScenario.process_action` overwrites its action with the
  scripted force on every step. The `agent` blocks are kept deliberately tiny
  for that reason. Its loss and gradient rows in `metrics.csv` are expected
  waste, not a symptom.

**What counts as the return.** A task declares the groups whose reward is the
study's outcome, as `return_groups` beside `params` in
`configs/tasks/<task>.yaml`: `[agents]` for Simple Spread, `[adversary]` for
PCP. This matters because `simple_tag`'s two groups are exactly zero-sum — each
predator scores `+10` per capture and the prey `−10` — so averaging over both,
which is what the metrics and saliency code did before 2026-09-03, reported
`0.0` no matter how the predators performed. `metrics.csv` also records each
group's own return as a separate row; the analysis reads the ungrouped study
row. Omitting the key measures every group, which is what a single-group task
resolved to anyway, so Simple Spread's numbers are unchanged.

Role/class-id conditioning is deferred on this task. `use_role_embedding` stays
`false` in every PCP model config: the three predators are homogeneous, and the
scenario inherits stock `simple_tag` observations, which expose no role signal
to condition on. Giving them one is a modeling decision (asymmetric speed or
sensing radius), not a wiring change, and it is not part of this integration.

The PCP suites live in `configs/sweeps/pcp_*.yaml` and mirror the V2 Simple
Spread suites row for row, with one deliberate difference: `max_n_frames` is
60,000 rather than 600,000. PCP has no stability gate of its own yet, so that
is a cheap first pass rather than a calibrated horizon, and every file carries
a `TODO` saying so. Review them the usual way:

```bash
uv run python scripts/sweep.py configs/sweeps/pcp_identity_pilot.yaml --dry-run
uv run python scripts/sweep.py configs/sweeps/pcp_comm_main.yaml --dry-run
```

## Baseline calibration and sweep workflow

The original Identity budget pilot was completed before the learned-method
comparison. Its 600k run used `gamma=0.99` and zero entropy regularization and
collapsed late in training: the final-10% evaluation return was `-2100.83`,
policy entropy fell to about `-14.31`, and critic loss/gradient norms grew
sharply. Extending that optimization configuration to 1.2M frames was therefore
rejected.

A bounded three-run, seed-0 diagnostic then isolated discount and entropy at
240k frames. `gamma=0.9` prevented the late collapse, while
`entropy_coef=0.01` was the safer of the two gamma-0.9 variants. A fresh 600k
Identity confirmation with that combination remained stable and improved: its
final-10% evaluation return was `-502.39`, normalized AUC was `-586.16`, and
its final explained variance reached `0.223`.

That seed-0 confirmation did not generalize: under the same provisional
protocol, MLP seed 4 failed with a non-finite TanhNormal action at 336k and
seeds 2--3 developed severe boundary saturation. The V1 main/ablation configs
are therefore historical and must not be launched. A 2x2 stress-seed diagnostic
then favored `gamma=0.9`, `entropy_coef=0.1`, and five minibatch iterations,
and a fresh 600k five-seed MLP gate confirmed that choice.

### Frozen protocol

The gate passed on all five seeds, so this protocol is frozen for every V2
main and ablation row and is not re-tuned per method:

| Setting | Value |
|---|---|
| `experiment.gamma` | `0.9` |
| `algorithm_config.params.entropy_coef` | `0.1` |
| `experiment.on_policy_n_minibatch_iters` | `5` |
| `experiment.on_policy_collected_frames_per_batch` | `6000` |
| `experiment.max_n_frames` | `600000` |
| `experiment.evaluation_interval` | `12000` (5 deterministic episodes) |

Every gate run completed with all 1,816 logged values finite, policy entropy
confined to `[0.83, 1.32]`, final returns between `-428.5` and `-555.7`, zero
deterministic action saturation, and at most `0.23%` random-exploration
saturation. Under V1 the same seed 4 produced a non-finite action at 336k and
was 83.6% saturated at 240k, so this is a decisive rather than marginal pass.

Audit any finished run the same way. The cheap half reads only `metrics.csv`;
the full form additionally rebuilds the run from its recorded overrides,
strictly loads its saved actor, and measures TanhNormal saturation:

```bash
uv run python scripts/audit_runs.py runs/simple_spread_comm_v2 \
  --model benchmarl_mlp --no-policy

uv run python scripts/audit_runs.py runs/simple_spread_comm_v2 \
  --model benchmarl_mlp --out results/simple_spread_comm_v2/mlp_gate_audit.csv
```

Review the exact managed calibration suites:

```bash
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_identity_pilot.yaml --dry-run

uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_identity_stability.yaml --dry-run

uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_identity_stable_horizon.yaml --dry-run
```

The corresponding completed run and analysis roots are:

```text
runs/simple_spread_identity_pilot/
results/simple_spread_identity_pilot/
runs/simple_spread_identity_stability/
results/simple_spread_identity_stability/
runs/simple_spread_identity_stable_horizon/
results/simple_spread_identity_stable_horizon/
```

The managed execution commands remain available for reproduction; completed
rows in an existing suite are skipped automatically:

```bash
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_identity_pilot.yaml --run

uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_identity_stability.yaml --run

uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_identity_stable_horizon.yaml --run
```

Regenerate each calibration analysis from its managed artifacts with:

```bash
uv run python scripts/analyze.py \
  runs/simple_spread_identity_pilot \
  --results-dir results/simple_spread_identity_pilot

uv run python scripts/analyze.py \
  runs/simple_spread_identity_stability \
  --results-dir results/simple_spread_identity_stability

uv run python scripts/analyze.py \
  runs/simple_spread_identity_stable_horizon \
  --results-dir results/simple_spread_identity_stable_horizon
```

The pilot and diagnostic sweep configs are intentionally retained as
historical calibration protocols, not rewritten to later settings. Their
managed run directories preserve the fully resolved configuration, source
provenance, metrics, and policy checkpoint needed to reproduce and audit why
the final protocol changed.

Review the historical V1 main manifest only; do not run it:

```bash
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v1.yaml --dry-run
```

V1 stopped at six completed, one failed, and 23 pending rows before any learned
communication model started. It is preserved as failure evidence.

### V2 study suites

V2 is the live study. Every suite pins the frozen protocol above. Review any
of them first with `--dry-run`, then execute:

```bash
# Stage 1: MLP + five communication models, seeds 0-4.
uv run python scripts/sweep.py configs/sweeps/simple_spread_comm_v2.yaml --run

# Stage 2: bandwidth and channel reliability.
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_stage2_message_dim.yaml --run
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_stage2_dropout.yaml --run

# Stage 3: communication depth and attention heads.
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_stage3_rounds.yaml --run
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_stage3_heads.yaml --run

# Stage 4: topology, sender scheduling, self communication.
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_stage4_graph_topology.yaml --run
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_stage4_sender_budget.yaml --run
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_stage4_self_communication.yaml --run

# Instrumentation only: uncontended per-method compute overhead.
uv run python scripts/sweep.py \
  configs/sweeps/simple_spread_comm_v2_wallclock.yaml --run
```

Rows are independent, so a suite can also be executed as a Slurm array or as
local parallel workers by selecting one row at a time from its manifest. Pin
the thread count so every row shares one numerical environment:

```bash
OMP_NUM_THREADS=1 uv run python scripts/sweep.py \
  --manifest runs/simple_spread_comm_v2/manifest.csv \
  --run-id simple_spread__mappo__attention__v2__seed000 --run
```

A row selected by `--index`/`--run-id` treats the manifest as immutable input
and writes only its own `status.json`; a later invocation without a selection
rewrites the manifest from those authoritative files. Wall-clock time recorded
inside a suite run concurrently is contended and is not a clean method
comparison; use the dedicated sequential wall-clock suite for that.

In any managed suite, per-run `status.json` files are authoritative; completed
rows are not rerun, failed outputs remain in place, and `--retry-failed`
creates a new retry ID.

## Running on a cluster (UCF ARCC Newton)

Rows are fully independent, so each experiment runs as a Slurm job array. Four
numbered scripts, submitted in order — `sbatch` is the only command needed:

```bash
cd ~/marl-comm
sbatch slurm/01_setup.sbatch            # once: venv, CUDA check, manifests
sbatch slurm/02_main_comparison.sbatch  # 30 rows  — MLP + 5 comm methods x 5 seeds
sbatch slurm/03_ablations.sbatch        # 201 rows — all seven ablations
sbatch slurm/04_analyze.sbatch          # CSVs, plots, REPORT.md
```

Wait for `01` before submitting `02`/`03`; they read the manifests it writes.
`02` and `03` are independent and can run concurrently. Re-submitting a training
script is the recovery path after preemption: completed rows are skipped and
abandoned rows are retried. See [slurm/README.md](slurm/README.md) for module
overrides, monitoring, preemption, cost, and storage.

PCP has its own parallel family — `slurm/pcp_01_setup.sbatch` through
`slurm/pcp_05_analyze.sbatch`, reusing the same virtualenv. Run the pilot and
read it before the main comparison: PCP's training budget has never been
calibrated, and at the budget measured so far a deterministic policy never
catches the prey, which leaves the evaluation return a five-sample rare-event
count. The evidence is in
[slurm/README.md](slurm/README.md#predator-capture-prey).

Two things to be deliberate about:

- **A GPU does not make a single row faster.** These actors are 19k--35k
  parameters on 10 vectorised environments; that is far too small to saturate a
  V100 or H100. The cluster win is running many independent rows at once, not
  per-row acceleration. Making a GPU worthwhile would mean raising
  `on_policy_n_envs_per_worker`, which changes the frozen protocol and
  invalidates the baseline gate.
- **Device changes numerics.** A suite must execute entirely on one device.
  `runs/` is Git-ignored, so a fresh clone on the cluster starts with no
  completed rows and the whole suite runs on one device automatically. If you
  copy `runs/` across machines you will silently mix devices; `REPORT.md` warns
  when it detects this.

### Interrupted runs

A worker killed by preemption or a walltime limit cannot write a terminal
status, so its row would sit at `running` forever -- never retried, never
reported as failed. Each run therefore writes a heartbeat every collection
iteration, and rows that stop reporting can be reclaimed:

```bash
uv run python scripts/sweep.py --manifest runs/<suite>/manifest.csv \
  --reclaim-stale 3600 --status
```

Reclaimed rows are retried under an explicit `__retryNN` id; the abandoned
directory is preserved as evidence. A live worker is never stolen from, because
reclamation requires the heartbeat to have gone silent.

## Outputs

Managed runs use this layout:

```text
runs/
└── <suite_id>/
    ├── suite_config.yaml
    ├── manifest.csv
    └── <run_id>/
        ├── resolved_config.yaml
        ├── metadata.json
        ├── status.json
        ├── metrics.csv
        ├── summary.json
        ├── saliency.json             # when communication saliency was measured
        ├── git.patch                 # when the source tree is dirty
        ├── checkpoints/
        └── benchmarl/
            └── <generated BenchMARL output>
```

Analysis products, regenerated from the above and never edited by hand:

```text
results/
└── <suite_id>/
    ├── REPORT.md                     # start here
    ├── <suite>_per_run.csv
    ├── <suite>_summary.csv
    ├── <suite>_failed_runs.csv
    ├── <suite>_paired_comparisons.csv
    ├── <suite>_saliency.csv
    └── *.png
```

`status.json` additionally carries a `heartbeat` and the owning process/Slurm
job, which is what makes abandoned runs detectable.

`metadata.json` records the seed, method, task/algorithm, timestamps, devices,
library versions, Git SHA/dirty state, parameter counts, command, ablation, and
actual BenchMARL output path. `resolved_config.yaml` preserves both the project
specification and BenchMARL defaults after construction. `metrics.csv` is a
tidy long-form table covering collection, training, evaluation, gradient/loss,
timer, and communication statistics where available.

Raw `runs/` and regenerated `results/` are ignored by Git. Each dirty run saves
the source diff/provenance needed to identify the exact code used.

## Analysis

Three passes, in order. Saliency is optional but should be run before
aggregation so its numbers reach the summary tables and report:

```bash
# 1. Communication saliency from each frozen final actor.
uv run python scripts/saliency.py runs/simple_spread_comm_v2 \
  --episodes 32 --out results/simple_spread_comm_v2/simple_spread_comm_v2_saliency.csv

# 2. Aggregate CSVs and plots.
uv run python scripts/analyze.py runs/simple_spread_comm_v2 \
  --results-dir results/simple_spread_comm_v2

# 3. Single readable rollup: results/<suite>/REPORT.md
uv run python scripts/report.py runs/simple_spread_comm_v2 \
  --results-dir results/simple_spread_comm_v2
```

### Communication saliency

Activity metrics say how much a module transmits, not whether transmitting
helps. Saliency answers the causal question on a frozen, already-trained
policy:

```text
saliency = evaluation return  -  evaluation return with every message suppressed
```

Suppression reuses the channel abstraction (`p=1` dropout). Because every module
decodes communication through a bias-free projection, an empty neighbourhood
gives an exactly zero communication delta, so each learned module collapses to
`h' = h` -- precisely `IdentityComm`. The learned weights are untouched; only
the channel is removed. Both arms are evaluated from the same environment seed,
so the difference is the intervention rather than episode luck.

`BenchMARL MLP` and `IdentityComm` must score exactly `0.0`; that is the control
value, and it is a live check that the intervention is wired correctly. Measured
on the seed-0 baselines: MLP and Identity both `+0.00`, `BroadcastComm`
`+170.82` (`-407.5` with communication versus `-578.3` severed, a 29.5%
improvement) -- the channel is genuinely load-bearing.

Two companion numbers separate *influence* from *benefit*: `action_shift` is the
mean action displacement on identical states, and `policy_kl` the divergence of
the pre-tanh action distributions. A large action shift with a small return
delta means the module changes behaviour without improving the shared
objective, which raw activity metrics cannot reveal.

Aggregating a suite directly from managed run directories also works for the
historical suites:

```bash
uv run python scripts/analyze.py \
  runs/simple_spread_comm_v1 \
  --results-dir results/simple_spread_comm_v1
```

The analysis ignores incomplete runs in aggregates but reports them separately.
It writes raw per-run and grouped CSVs, failed-run information, learning curves,
final-return comparisons, AUC/sample-efficiency summaries, communication-cost
plots, and wall-clock comparisons. Final performance is the mean of the final
10% of evaluation points. AUC uses environment frames on the x-axis. Group
uncertainty is a fixed-seed 95% bootstrap confidence interval over independent
runs; matched seed coverage is checked rather than inferred from directory
names.

## Research protocol

The first comparison holds fixed:

- stock VMAS Simple Spread and its task configuration;
- BenchMARL MAPPO and centralized critic;
- actor encoder width/depth and activation;
- seeds `0,1,2,3,4`;
- one common 600,000-frame/gamma/entropy/update-reuse protocol selected by the
  baseline stability gate (the exact V2 values are not yet frozen);
- deterministic evaluation every 12,000 frames;
- all other collection, optimization, and evaluation settings.

Only the communication mechanism changes. Actor capacity and communication
payload are reported separately. The MLP remains a framework sanity reference;
Identity is the primary no-communication control because every learned method
shares its policy shell. Identical MLP/Identity results are expected under the
matched architecture and seed.

Communication dropout is not silently implemented as ordinary neural-network
dropout. PPO requires the realized failure mask used for behavior collection to
be replayed during policy-loss recomputation. The actor persists that mask,
samples a fresh one at every no-gradient environment step, and reuses the exact
realization during the gradient-enabled PPO forward. Train-time Gaussian noise
is rejected because a sender mask cannot replay payload noise; use its explicit
evaluation mode instead. Static ring/random graphs on Simple Spread are
topology diagnostics, not models of physical radio links.

PCP now runs the same comparison on a second task; its protocol is not yet
calibrated (see [Predator-Capture-Prey](#predator-capture-prey)). PP, forking
MAPPO, and placing communication logic in the algorithm, runner, task, or VMAS
environment all remain out of scope.
