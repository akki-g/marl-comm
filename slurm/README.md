# Running the study on UCF ARCC Newton

Newton is UCF's GPU cluster (V100 and H100 nodes) scheduled with Slurm. These
scripts run the full V2 communication study there as job arrays.

## Read this before you submit

**A GPU will not make an individual row faster.** The actors in this study are
19k–35k parameters and the task runs 10 vectorised VMAS environments with three
agents. That workload is far too small to saturate a V100 or H100; per-row time
is dominated by Python and kernel-launch overhead, and a single row may well be
*slower* on a GPU than on one CPU core. Locally each 600k-frame row takes about
6–14 minutes on a single core.

The cluster win is **horizontal**, not per-row: 231 rows are fully independent,
so an array finishes in roughly the time of the slowest chunk instead of the sum
of all rows.

Two consequences worth deciding on deliberately:

- If you want GPU utilisation to actually pay off, the lever is
  `experiment.on_policy_n_envs_per_worker` (BenchMARL's own fine-tuned VMAS
  config uses 600 environments against our 10). That is a **protocol change**:
  it invalidates the frozen protocol, the passed five-seed MLP gate, and every
  row already collected. Do not do it mid-study.
- **Device changes numerics.** CPU and CUDA runs are not directly comparable.
  Run a suite entirely on one device. Each run records its device in
  `metadata.json`, and `REPORT.md` warns when a suite mixes devices.

If you simply want the results as fast as possible with the existing protocol
intact, request several CPUs and no GPU (`--gpus 0`) and raise `--chunk`; the
scripts support it and it costs far fewer DPH.

## One-time setup (login node)

```bash
git clone <your remote> ~/marl-comm
cd ~/marl-comm
bash slurm/bootstrap_newton.sh
```

This loads `python/python-3.11.4-gcc-12.2.0` and `cuda/cuda-12.4.0`, creates
`.venv-newton`, installs the CUDA build of PyTorch first so the CPU wheel is not
pulled in transitively, then installs the project. It prints a verification
block; confirm `cuda available` on a compute node rather than the login node:

```bash
srun --gres=gpu:1 --pty nvidia-smi
```

Verify the module names against your account before trusting the defaults —
they are the values most likely to drift:

```bash
module avail python
module avail cuda
sinfo -o "%P %l %G %D"     # partitions, time limits, GRES, node counts
```

Override any of them without editing files:

```bash
COMMSTUDY_PYTHON_MODULE=python/python-3.11.4-oneapi-2023.1.0 \
COMMSTUDY_CUDA_MODULE=cuda/cuda-12.6.0 \
  bash slurm/bootstrap_newton.sh
```

## Submitting

```bash
# One suite.
bash slurm/submit.sh configs/sweeps/simple_spread_comm_v2.yaml

# See the exact sbatch command without submitting.
bash slurm/submit.sh configs/sweeps/simple_spread_comm_v2.yaml --dry-run

# Every V2 suite (231 rows total).
bash slurm/submit.sh --all --chunk 4 --max-concurrent 25
```

`submit.sh` builds each manifest, derives the array range from the actual row
count so it can never drift, names the job after the suite, and creates the log
directory. Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--chunk N` | `4` | Manifest rows per array task, run sequentially |
| `--max-concurrent N` | `25` | Array throttle (`--array=0-M%N`) |
| `--partition NAME` | `normal` | Use `preemptable` when out of monthly DPH |
| `--time HH:MM:SS` | `08:00:00` | Walltime per array task |
| `--gpus N` | `1` | `0` requests no GPU |
| `--cpus N` | `4` | Also sets the pinned thread count |
| `--dependency SPEC` | – | Passed through to `sbatch` |

Size the walltime against the chunk: one row is ~15 minutes of GPU time, so
`--chunk 4` fits comfortably inside 8 hours with a wide safety margin.

## Monitoring

```bash
squeue -u "$USER"
python scripts/sweep.py --manifest runs/simple_spread_comm_v2/manifest.csv --status
tail -f slurm/logs/simple_spread_comm_v2/<jobid>_0.out
```

`--status` reads the authoritative per-run `status.json` files, not the
scheduler, so it stays correct across resubmissions.

## Preemption, timeouts, and retries

A worker killed by preemption or a walltime limit cannot write a terminal
status, so its row would otherwise sit at `running` forever — never retried and
never reported as failed. Every run therefore writes a heartbeat once per
collection iteration, and rows are reclaimable once they stop reporting:

```bash
# Report abandoned rows as failed.
python scripts/sweep.py --manifest runs/<suite>/manifest.csv --reclaim-stale 3600 --status

# Resubmit; abandoned rows are retried under an explicit __retryNN id.
bash slurm/submit.sh configs/sweeps/<suite>.yaml
```

The array workers already pass `--reclaim-stale 3600`, so simply resubmitting a
suite picks up whatever the previous attempt abandoned. Completed rows are
never rerun, and failed output is preserved rather than deleted.

## Analysis

```bash
sbatch --dependency=afterany:<training_job_id> \
  slurm/analyze_suite.sbatch simple_spread_comm_v2
```

This syncs status, measures communication saliency from each frozen checkpoint,
aggregates, plots, and writes `results/<suite>/REPORT.md`. Run it locally too:

```bash
python scripts/sweep.py --manifest runs/<suite>/manifest.csv --reclaim-stale 3600 --status
python scripts/saliency.py runs/<suite> --out results/<suite>/<suite>_saliency.csv
python scripts/analyze.py runs/<suite> --results-dir results/<suite>
python scripts/report.py  runs/<suite> --results-dir results/<suite>
```

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

results/<suite_id>/               regenerated analysis (git-ignored)
  REPORT.md                       start here
  *_per_run.csv  *_summary.csv  *_failed_runs.csv
  *_paired_comparisons.csv  *_saliency.csv
  *.png

slurm/logs/<suite_id>/<jobid>_<task>.out|.err
```

## Storage

`$HOME` is on `/lustre/fs1` with a **1 TB / 1,000,000-file quota** and **no
backups**. One completed run is ~108 files and ~5 MB, so the full 231-row study
is roughly 25k files and 1.2 GB — comfortable, but the file count is the
binding constraint if you later expand seeds or suites. Back up `runs/` and
`results/` yourself; nothing on the cluster is backed up for you.

## Cost

The default group allocation is 80,000 Dedicated Processor Hours per month.
Prefer `--partition preemptable` once depleted; combined with heartbeat-based
reclamation, preempted rows are retried automatically on resubmission.
