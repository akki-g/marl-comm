# Running the study on UCF ARCC Newton

PCP uses the evidence-gated workflow below. Its corrected protocol is still a
candidate, so no PCP training stage is launch-ready. See
[the PCP protocol guide](../docs/PCP_PROTOCOL.md) for review artifacts and exact
scientific settings.

## Historical Simple Spread workflow

The following `00`–`04` scripts preserve the completed Simple Spread V2
infrastructure workflow. They describe that study's seeds, settings, and runtime
experience; they do not authorize a new PCP experiment.

```bash
cd ~/marl-comm

sbatch slurm/00_probe.sbatch            # optional: report what this cluster has
sbatch slurm/01_setup.sbatch            # once: venv, CUDA check, manifests
sbatch slurm/02_main_comparison.sbatch  # 30 rows  — MLP + 5 comm methods x 5 seeds
sbatch slurm/03_ablations.sbatch        # 201 rows — full ablation study
sbatch slurm/04_analyze.sbatch          # CSVs, plots, REPORT.md
```

Wait for `01` to finish before submitting `02` or `03`; they need the manifests
it writes. `02` and `03` are independent and can run at the same time. Chain the
analysis so it starts automatically:

```bash
sbatch --dependency=afterany:<main_jobid>:<ablation_jobid> slurm/04_analyze.sbatch
```

`slurm/newton_env.sh` is *sourced* by the job scripts, never executed, so it
needs no execute permission.

## Historical Simple Spread experiments

**`02_main_comparison.sbatch` — 30 rows, the headline result.** Six models x
five seeds at 600,000 frames on stock VMAS Simple Spread:

| Model | Method |
|---|---|
| `benchmarl_mlp` | framework reference MLP |
| `comm_identity` | no-communication control, same actor shell |
| `comm_broadcast` | CommNet-inspired mean broadcast |
| `comm_gated` | IC3Net-inspired sender gating |
| `comm_attention` | TarMAC-inspired targeted attention |
| `comm_graph` | DGN/GAT-inspired graph message passing |

**`03_ablations.sbatch` — 201 rows, seeds 0-2, same frozen protocol** so every
ablation is directly comparable to the main comparison:

| Ablation | Rows | Values |
|---|---:|---|
| `message_dim` | 48 | 8 / 16 / 32 / 64, all four learned modules |
| `communication_dropout` | 48 | p = 0.00 / 0.25 / 0.50 / 0.75 |
| `communication_rounds` | 18 | 1 / 2 / 3, attention + graph |
| `attention_heads` | 24 | 1 / 2 / 4 / 8 at fixed total width |
| `graph_topology` | 9 | full / directed ring / Erdos-Renyi |
| `sender_budget` | 36 | learned-K vs seeded random-K control |
| `self_communication` | 18 | `exclude_self` true / false |

All seven share one combined manifest, so this is a single contiguous array
rather than seven ranges to track by hand. Each row still writes into its own
suite directory, so per-suite analysis is unchanged.

Every ablation re-runs its own default point (`message_dim=32`, `rounds=1`,
`topology=full`, `heads=4`, `exclude_self=true`). Those are matched in-suite
controls that should reproduce the main-suite rows on seeds 0-2 — a free
reproducibility check.

## Predator-Capture-Prey

**The corrected candidate is not launch-ready.** The old 60k/120k pilot and
27-row gamma/entropy grid are retired calibration evidence. Array 795630
completed all 27 rows at 600k and demonstrated learning at several gamma values;
it does not confirm the repaired reset/RNG, observation, critic, and training
health semantics. Preserve those historical artifacts under their original IDs.

The single source for new scientific settings is
[`configs/protocols/pcp_corrected_v1.yaml`](../configs/protocols/pcp_corrected_v1.yaml).
The candidate uses gamma 0.95, entropy 0.1, **ten** PPO minibatch passes, 600k
frames, and 128 deterministic evaluation episodes per point. The budget is a
fixed-compute choice, not a convergence claim. Full settings, semantic versions,
runtime requirements, evidence requirements, and the reviewer workflow are in
[the PCP protocol guide](../docs/PCP_PROTOCOL.md).

### Plan first; submit only after the stage check passes

Use the existing virtualenv prepared by `01_setup.sbatch`, or select a compatible
interpreter through `COMMSTUDY_PYTHON`. Planning constructs and checks the model
and task contracts and writes inspectable manifests. It submits no training and
grants no approval:

```bash
cd ~/marl-comm
bash slurm/pcp_01_setup.sbatch
```

| Script | Corrected workload | Manifest |
|---|---|---|
| `pcp_02b_protocol.sbatch` | 2 Identity confirmation rows, new seeds 10 and 11 | `runs/pcp_candidate_confirmation_corrected_v1/manifest.csv` |
| `pcp_03_main_comparison.sbatch` | 30 rows: MLP reference plus five communication models, seeds 20–24 | `runs/pcp_comm_main_corrected_v1/manifest.csv` |
| `pcp_04_ablations.sbatch` | 201 rows across seven ablations, seeds 20–22 | `runs/_manifests/pcp_ablations_corrected_v1.csv` |

Main and ablation YAML filenames remain familiar, but their suite IDs and run
namespace are new. They derive their scientific settings from the versioned
protocol. Existing manifests cannot silently authorize changed rows.
`pcp_02_pilot.sbatch` and the old protocol-grid YAML are retired; do not use them
to launch more calibration rows.

A stage review must name its reviewer and rationale and bind the exact protocol
hash, research-source fingerprint, stage, and hashed evidence files. Confirmation
requires a candidate review. Main comparison additionally requires the protocol
to be frozen with corrected confirmation evidence. Ablations require a reviewed
main comparison, with extra evidence for guarded width/round factors. An
approval-template command creates a **pending** review, not launch permission.
Changes to bound sources, settings, or evidence invalidate the corresponding
checks. The currently missing reviewed evidence must be supplied before launch.

After the relevant review is complete, invoke the wrappers from a login node:

```bash
bash slurm/pcp_02b_protocol.sbatch       # confirmation, after candidate review
bash slurm/pcp_03_main_comparison.sbatch # main, after corrected confirmation and freezing
bash slurm/pcp_04_ablations.sbatch       # ablations, after main review
```

Each wrapper checks its manifest and stage **before calling `sbatch`**. The
allocated worker checks the same binding again, plus package/device/thread
requirements, before `srun` and training. Calling `sbatch` directly bypasses the
login-node pre-submission check and can consume an allocation before rejection;
it does not bypass the worker or managed-run checks. Environment setup success
is not evidence that the candidate has been scientifically confirmed.

### Measure the predators and preserve historical semantics

PCP's measured group is explicitly `adversary`. Reward is repeated +10 per
predator–prey contact step; it is not a count of distinct captures or a
first-capture termination objective. The scripted prey still has an optimized
policy whose actions the scenario discards. Its metrics remain separately
inspectable, and its diagnostics must not be mixed into predator health.

Frozen-policy channel reliance compares matched complete episodes. Same-input
influence evaluates both communication conditions on identical recorded inputs.
Neither measurement alone establishes that communication-free control is
impossible. `scripts/saliency.py --episodes 32` and
`scripts/audit_runs.py --episodes 32` now measure exactly 32 episodes, regardless
of the training/test environment batch size. This differs from the retired
batched evaluator. Complete paired Identity episodes must remain an exact null.

```bash
sbatch slurm/pcp_05_analyze.sbatch              # the 9 corrected suites
sbatch slurm/pcp_05_analyze.sbatch --historical # retired suites, saved metrics only
```

The default analysis covers corrected confirmation, main comparison, and all
seven corrected ablation suites. It synchronizes run status, audits frozen
predator policies, records reliance/influence measurements, and regenerates
reports. It writes status/saliency artifacts beside runs as well as derived
files under `results/`; it is not read-only.

Historical mode skips checkpoint rollouts and new saliency measurements. Loading
an old policy under current source/task semantics is not historical trajectory
reproduction. Direct audit/saliency CLIs reject missing or incompatible runtime
provenance by default. Any deliberate cross-runtime re-evaluation requires
explicit analysis-device/mismatch options and records those differences; it must
not replace the historical calibration claim or serve as corrected confirmation.

## Historical Simple Spread runtime notes

These notes record the Simple Spread infrastructure experience. For PCP, use
the runtime declared in its protocol and the worker checks; changing devices,
thread counts, packages, or batch sizes requires a reviewed protocol decision.

**Small Simple Spread rows did not establish a GPU speedup.** The actors are 19k-35k
parameters and the task runs 10 vectorised VMAS environments with three agents.
That is far too small to saturate a V100 or H100; per-row time is dominated by
Python and kernel-launch overhead, and a single row may well be *slower* on a
GPU than on one CPU core. Locally each 600k-frame row takes 6-14 minutes on one
core.

The cluster win is **horizontal**: 231 rows are fully independent, so the array
finishes in roughly the time of the slowest wave rather than the sum of all
rows.

- Making a GPU genuinely worthwhile would mean raising
  `experiment.on_policy_n_envs_per_worker` (BenchMARL's own fine-tuned VMAS
  config uses 600 against our 10). That is a **protocol change**: it invalidates
  the frozen protocol, the passed five-seed MLP gate, and every collected row.
  Do not do it mid-study.
- If you only want the results fast with the protocol intact, edit the two
  training scripts to drop `--gres=gpu:1`, raise `--cpus-per-task`, and let
  `newton_env.sh` default `COMMSTUDY_DEVICE=cpu`. It costs far fewer DPH.

**Device changes numerics.** CPU and CUDA runs are not directly comparable, so a
suite must execute entirely on one device. `runs/` is Git-ignored, so a fresh
clone on Newton starts with zero completed rows and the whole study runs on one
device automatically — **do not copy `runs/` from your laptop**. `REPORT.md`
warns if it detects a suite that mixes devices or commits.

**The CPU protocol gate does not automatically transfer.** After `02` finishes,
check the MLP rows on CUDA before trusting any cross-method comparison:

```bash
grep -A3 benchmarl_mlp results/simple_spread_comm_v2/REPORT.md
```

Expect all-finite metrics, entropy near 1, zero deterministic action saturation,
and final returns clustered near −430..−560. If CUDA reproduces that, the
protocol transfers; if not, stop and diagnose before interpreting communication
rows.

## Troubleshooting setup

`00_probe.sbatch` is read-only and prints the ground truth for your account:
available python/cuda/anaconda modules, the GPU, the partitions, and — most
usefully — which candidate python module actually yields a working interpreter.
Run it whenever setup fails:

```bash
sbatch slurm/00_probe.sbatch
cat slurm/logs/probe_<jobid>.out
```

**`No module named venv`.** The Spack-built modules here provide `python3` but
not always `python`, so a bare `python` silently falls through to
`/usr/bin/python`, which has no `venv`. `01_setup.sbatch` now resolves the
interpreter explicitly and rejects anything under `/usr/bin` or `/bin`, anything
below 3.11, and anything whose `venv` module does not actually run. It tries
these in order and reports which it used:

```text
python/python-3.11.4-gcc-12.2.0
python/python-3.11.4-oneapi-2023.1.0
python-3.11.4-gcc-12.2.0
anaconda/anaconda-2024.10
anaconda/anaconda-2023.09
```

Force a specific one without editing files:

```bash
sbatch --export=ALL,COMMSTUDY_PYTHON_MODULE=anaconda/anaconda-2024.10 \
  slurm/01_setup.sbatch
```

The module it settles on is written to `slurm/.resolved_env`, which the training
jobs then source, so every job uses the interpreter setup actually verified
rather than re-guessing.

**`No matching distribution found for torch>=2.7`.** The `cu124` index does not
carry PyTorch 2.7 or newer — 2.7 dropped CUDA 12.4 and ships `cu118`, `cu126`,
and `cu128` only. Use **cu126**, which is now the default.

cu126 is chosen deliberately over cu128: its wheels cover both architectures on
this cluster, Volta `sm_70` (V100) and Hopper `sm_90` (H100), so a job is safe
wherever it lands. cu128 targets newer architectures, so pair it with an H100
pin or a V100 allocation may fail at runtime with `no kernel image is available
for execution on the device`.

The wheel bundles its own CUDA runtime, so the index need not match the loaded
`cuda` module; only the node driver has to be new enough.

**`No module named 'torch'`, or `cannot reach pypi.org`.** If compute nodes have
no outbound internet, `pip` cannot run inside a job. `01_setup.sbatch` detects
this before building anything and stops with instructions rather than leaving a
half-built venv behind.

Install from a **login node**, then resubmit `01`. It detects the completed
environment, skips the install, and just builds the manifests:

```bash
cd ~/marl-comm
module load python/python-3.11.4-gcc-12.2.0 cuda/cuda-12.4.0
python3 -m venv .venv-newton                      # note: python3, not python
source .venv-newton/bin/activate
pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu126 'torch>=2.7'
pip install -e '.[dev,analysis]'
```

**Do not verify the install by importing torch on the login node.** Verify with
a job instead:

```bash
sbatch slurm/00_probe.sbatch
cat slurm/logs/probe_<jobid>.out     # reports every package version
```

**`ImportError: libtorch_cuda.so: failed to map segment from shared object`.**
This nearly always means the *login node*, not a broken install.
`libtorch_cuda.so` is 1–2 GB, and login nodes commonly cap address space
(`ulimit -v`) to stop heavy work there, so the mmap fails. Compute nodes
normally have no such cap.

Confirm by importing on a compute node — if it works there, the install is fine
and nothing needs fixing:

```bash
srun --gres=gpu:1 --pty .venv-newton/bin/python \
  -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The same error can also come from a **truncated** library if the install hit a
quota limit. `00_probe.sbatch` reports `ulimit -a`, your Lustre quota, and the
actual size of `libtorch_cuda.so`, which separates the two cases: a healthy file
is on the order of gigabytes.

A venv that exists but is incomplete is now caught once, in `newton_env.sh`,
with the remedy printed — instead of all 231 array tasks each dying inside an
import traceback.

**`virtualenv not found`** from `02`/`03`/`04` simply means `01` has not
succeeded yet. Those jobs exit immediately without touching `runs/`, so a failed
setup costs nothing and creates no partial results.

Other defaults baked into the scripts: partition `normal`, `--gres=gpu:1`,
`cuda/cuda-12.4.0`.

## Choosing V100 vs H100

Newton mixes both: roughly 42 V100s (21 nodes, dual V100 16/32 GB) and 90 H100s
(29 nodes with dual H100 80 GB, plus 4 nodes with eight each). A bare
`--gres=gpu:1` takes whatever is free, which for this study means rows may land
on different architectures.

**That matters scientifically.** Kernel selection and floating-point reduction
order differ between Volta and Hopper, so a study split across both is not a
clean comparison — the same hazard as mixing CPU and CUDA. `REPORT.md` now
detects and warns about it, but pinning is better than discovering it later.

The scripts do not hardcode a GPU type, because an unconfigured GRES type makes
every submission fail outright. Find the real string first:

```bash
sbatch slurm/00_probe.sbatch
cat slurm/logs/probe_<jobid>.out
```

The probe prints the configured GRES types, the node features, and the GPU model
per node. Then pin on the command line — sbatch flags override the `#SBATCH`
headers, so no file editing is needed:

```bash
# If the probe shows a GRES type such as gpu:h100
sbatch --gres=gpu:h100:1 slurm/02_main_comparison.sbatch
sbatch --gres=gpu:h100:1 slurm/03_ablations.sbatch

# If it instead shows a node feature such as h100
sbatch --constraint=h100 slurm/02_main_comparison.sbatch
```

H100 nodes are both faster and far more numerous here, so pinning to them
usually improves queue time as well as consistency. Use the same pin for every
suite in the study, including `04_analyze.sbatch`, so the saliency rollouts run
on the architecture the policies were trained on.

Check partitions, time limits, GRES, and features directly with:

```bash
sinfo -o "%20P %10l %24G %20f %6D %t"
```

## Monitoring

```bash
squeue -u "$USER"
tail -f slurm/logs/main/<jobid>_0.out

source .venv-newton/bin/activate
python scripts/sweep.py --manifest runs/simple_spread_comm_v2/manifest.csv --status
python scripts/sweep.py --manifest runs/_manifests/ablations.csv --status
```

`--status` reads the authoritative per-run `status.json` files, not the
scheduler, so it stays correct across resubmissions.

## Preemption, timeouts, and retries

A worker killed by preemption or a walltime limit cannot write a terminal
status, so its row would otherwise sit at `running` forever — never retried and
never reported as failed. Every run writes a heartbeat once per collection
iteration, and the training scripts already pass `--reclaim-stale 3600`.

**To recover from any interruption, just resubmit the same script.** Completed
rows are skipped, abandoned rows are retried under an explicit `__retryNN` id
with the original directory preserved as evidence, and a still-live worker is
never stolen from.

```bash
sbatch slurm/02_main_comparison.sbatch    # resubmitting is the recovery path
```

To resubmit only the rows that failed, use the array's index list:

```bash
sbatch --array=3,17,22 slurm/02_main_comparison.sbatch
```

## Cost and storage

Default allocation is 80,000 Dedicated Processor Hours per group per month.
Switch to the preemptable queue once depleted — heartbeat reclamation makes
preemption recoverable:

```bash
sbatch --partition=preemptable slurm/03_ablations.sbatch
```

Concurrency is capped in the scripts (`%15` for the main comparison, `%20` for
ablations). Lower it if you are sharing the allocation.

`$HOME` is on `/lustre/fs1` with a **1 TB / 1,000,000-file quota** and **no
backups**. One completed run is ~108 files and ~5 MB, so the full 231-row study
is roughly 25k files and 1.2 GB — comfortable, but the file count is the binding
constraint if you later expand seeds or suites. Back up `runs/` and `results/`
yourself.

## Output layout

```text
runs/<suite_id>/                  managed run artifacts (git-ignored)
  manifest.csv                    planned rows + last synced status
  suite_config.yaml               exact suite definition used
  <run_id>/
    resolved_config.yaml          project + BenchMARL config after construction
    metadata.json                 seed, git SHA, versions, device, params, timings
    status.json                   authoritative state + heartbeat + owning job
    metrics.csv                   tidy long-form metrics
    summary.json                  final return, AUC, timings
    saliency.json                 communication saliency, when measured
    checkpoints/policy_state.pt   final actor
    benchmarl/                    BenchMARL's own output tree

runs/_manifests/ablations.csv     combined 201-row manifest for step 3

results/<suite_id>/               regenerated analysis (git-ignored)
  REPORT.md                       start here
  *_per_run.csv  *_summary.csv  *_failed_runs.csv
  *_paired_comparisons.csv  *_saliency.csv  *_run_audit.csv
  *.png

slurm/logs/main/<jobid>_<task>.out|.err
slurm/logs/ablations/<jobid>_<task>.out|.err
slurm/logs/setup_<jobid>.out|.err
slurm/logs/analyze_<jobid>.out|.err
```
