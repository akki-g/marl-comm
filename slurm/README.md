# Running the study on UCF ARCC Newton

Four numbered scripts, submitted with `sbatch` in order. No helper shell scripts
to execute — the only command you ever run is `sbatch`.

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

## What each experiment is

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

## Read this before you submit

**A GPU will not make an individual row faster.** The actors are 19k-35k
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

Verify before resubmitting — this must print versions, not a traceback:

```bash
python -c 'import torch, benchmarl, torchrl, tensordict, vmas; print(torch.__version__)'
```

`torch.cuda.is_available()` will be `False` on a login node with no GPU; that is
expected. `01_setup.sbatch` checks it properly from a compute node.

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
