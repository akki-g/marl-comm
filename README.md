# commstudy

Two reproducible starting experiments: PCP predator capture and MAPDN voltage
control. Every config runs **Identity, Broadcast, Gated, Attention and Graph**
with one to five seeds. BenchMARL owns MAPPO and QMIX; this package supplies the
communication models, tasks, recording and execution.

## Install

For local development, use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

This installs both tasks and plotting. MAPDN's required simulator is included
in the package; no sibling repository or `PYTHONPATH` setting is needed.
On UCF Newton, use the Slurm command below: the batch job performs installation
on a compute node, so `uv` does not need to be available in your login shell.

MAPDN requires your local `case33_3min_final` directory containing `model.p`,
`pv_active.csv`, `load_active.csv`, and `load_reactive.csv`. The starting config
points to `../MAPDN/mapdn/environments/var_voltage_control/data/case33_3min_final`
from the repository root. Keep that dataset layout on the cluster or change
`data_path` in `configs/mapdn.yaml`. Paths inside YAML are relative to the config's directory.
The runner validates all inputs and prepares a chronological 60/20/20 split and
training-only normalization from 16 fixed zero-action episodes automatically.
Validation is used for scheduled evaluation; the test split stays unused.

## Run

```bash
uv run --locked python scripts/run_experiment.py configs/pcp.yaml --dry-run
uv run --locked python scripts/run_experiment.py configs/pcp.yaml
uv run --locked python scripts/run_experiment.py configs/mapdn.yaml
```

Both starting configs have five seeds (1100–1104), so each runs 25 policies.
PCP uses sensing radius 1.0 and 600,000 frames per policy; MAPDN uses 131,072.
A frame is one joint environment transition, summed across vectorized environments.
Evaluation frames do not consume the training budget. Collection batch sizes must
divide the budget; the runner rejects settings that would silently exceed it.

Edit the single YAML to change the task, algorithm, seeds, budget, optimization,
model dimensions, evaluation or execution. Five CPU workers are the default,
capped by the available allocation. Each job starts in a fresh process with one
numerical thread. For a quick check, reduce seeds to one and use a smaller budget
that is a multiple of `training.batch_frames`; use a new `output_dir`.

Repeating the command skips compatible completed jobs. Failed/interrupted jobs
restart from the beginning, keeping their previous files under `attempts/`.
Changes to scientific settings, code, dependencies or input data require a new
output directory. A directory lock prevents concurrent launches. Independent
jobs finish after another job fails; the command then reports failures and exits
nonzero. This is restart support, not exact optimizer/checkpoint continuation.

On **UCF Newton**, submit from the repository root on the login node:

```bash
sbatch slurm/run_experiment.sbatch
```

The wrapper loads `anaconda/anaconda-2024.10`, activates its base Python 3.12,
installs pinned `uv` 0.12.5 under `.tools/`, and creates/syncs `.venv-newton`
from `uv.lock`. It activates that environment before running the experiment.
Both array tasks serialize environment setup; subsequent submissions reuse it.
Initial setup needs access to Astral/GitHub and the package indexes from the compute node.

This follows UCF's [module and environment instructions](https://arcc.ist.ucf.edu/docs/software/anaconda/),
which require environment creation/package installation on compute nodes, and its
[batch submission guide](https://arcc.ist.ucf.edu/docs/scheduler/scripts/).
The published [module list](https://arcc.ist.ucf.edu/docs/software/availableModules/)
includes this Anaconda version but does not list `uv`; installation uses
[Astral's standalone installer](https://docs.astral.sh/uv/getting-started/installation/)
without changing shell profiles. CUDA/MPI modules are unnecessary for these CPU configs.

This submits a two-task array on UCF's [normal partition](https://arcc.ist.ucf.edu/docs/scheduler/limitations/):
index **0 runs PCP**, index **1 runs MAPDN**.
Each task runs its config's 25 policies with five CPU workers, 24 GB of memory
and a 48-hour time limit. Scheduler output goes to `slurm-<job>_<index>.out`;
experiment outputs go to the directories configured in YAML.
For just one experiment, use `sbatch --array=0 slurm/run_experiment.sbatch`
(PCP) or `--array=1` (MAPDN). Check progress with `squeue -u "$USER"`.
Use `sbatch --account=<your-allocation> ...` if your account requires explicit
allocation selection; resource requests can also be overridden with `sbatch` options.
The public guide does not specify your account's wall-time limit; inspect
`scontrol show partition normal` and adjust `--time` if needed.

## Results and plots

`logs/pcp/` and `logs/mapdn/` contain the resolved `config.yaml`, source/dependency
`provenance.json`, readable `experiment.out`, `runs.csv` and combined `metrics.csv`.
Each `<method>/seed_<seed>/` contains its own `metrics.csv`, `run.out`, `status.json`
and final actor/critic `checkpoint.pt`. MAPDN also records preparation under `data/`.
Modified source is captured in `git.patch`. Internal `.framework/` JSON files keep
BenchMARL's scheduled evaluation enabled.

Metrics use long-form rows: frames, phase, group, metric, value, and optional
sample/episode. The combined CSV adds method and seed. Empty group means the
experiment's measured return: **predators only for PCP**, shared team return for
MAPDN. Per-group and individual training/evaluation episode records remain available.

```bash
uv run --locked python scripts/plot.py logs/pcp --list-metrics
uv run --locked python scripts/plot.py logs/pcp \
  --metrics evaluation/return_mean collection/return_mean
```

On Newton, load the same Anaconda module inside a compute allocation and use
`.venv-newton/bin/python scripts/plot.py ...` instead of the local `uv run` prefix.

Each requested metric creates six PNGs under `plots/`: one five-method comparison
(mean ± 1 sample standard deviation) and one individual-seed figure per method.
The plotted aggregates are also exported to CSV. Episodes are averaged within
seed first, then completed seeds receive equal weight. Curves use the intersection
of recorded frames, without smoothing or interpolation. Legends show available
seed counts; incomplete individual trajectories are labeled. With one seed,
the standard deviation band is zero.

## Develop

Core code is under `src/commstudy/`: `communication`, `environments`, `adapters`,
`tasks`, `models`, `algorithms`, `experiments` and `analysis`. Scripts only call the
package. `experiments/` contains generic execution, configuration, recording and
training checks, with no study approval or supervisor workflows.

```bash
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked ruff check src scripts tests
```

QMIX remains available for discrete PCP: set `algorithm: qmix`, replace
`algorithm_params` with its QMIX parameters (or `{}`), and replace MAPPO's
`epochs`/`minibatch_size` with `optimizer_steps`, `train_batch_size`, `memory_size`
and `init_random_frames`. MAPDN rejects QMIX because its actions are continuous.

See [the cleanup audit](docs/AUDIT.md), [method definitions](docs/communication_methods.md)
and [historical revision index](docs/HISTORY.md). MAPDN source attribution and its
MIT license are retained in `src/commstudy/environments/mapdn/`. Regression and
short real Case33 checks establish integration correctness; full-budget learning
comparisons are separate experiments.
