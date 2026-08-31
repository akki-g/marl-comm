from __future__ import annotations

import csv
import dataclasses
import json
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from commstudy.experiments.bookkeeping import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    RunContext,
    atomic_write_text,
    current_status,
    is_stale_running,
    make_run_id,
    retry_context,
)
from commstudy.experiments.config import load_experiment_spec
from commstudy.experiments.runner import run_managed_experiment


#: A run is presumed dead if it has not reported progress for this long.
#: One collection iteration is seconds to a couple of minutes, so an hour of
#: silence means the worker is gone, not slow.
DEFAULT_STALE_AFTER_SECONDS = 3_600.0

MANIFEST_COLUMNS = (
    "run_id",
    "suite_id",
    "task",
    "algorithm",
    "model",
    "seed",
    "ablation",
    "ablation_value",
    "max_n_frames",
    "status",
    "attempt",
    "retry_of",
    "output_root",
    "overrides_json",
    "command",
)


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    suite_id: str
    task: str
    algorithm: str
    model: str
    seed: int
    ablation: str
    ablation_value: str | None
    max_n_frames: int
    status: str
    output_root: Path
    overrides: tuple[str, ...]
    command: str
    attempt: int = 0
    retry_of: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "suite_id": self.suite_id,
            "task": self.task,
            "algorithm": self.algorithm,
            "model": self.model,
            "seed": self.seed,
            "ablation": self.ablation,
            "ablation_value": self.ablation_value or "",
            "max_n_frames": self.max_n_frames,
            "status": self.status,
            "attempt": self.attempt,
            "retry_of": self.retry_of or "",
            "output_root": str(self.output_root),
            "overrides_json": json.dumps(list(self.overrides)),
            "command": self.command,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> RunPlan:
        return cls(
            run_id=row["run_id"],
            suite_id=row["suite_id"],
            task=row["task"],
            algorithm=row["algorithm"],
            model=row["model"],
            seed=int(row["seed"]),
            ablation=row.get("ablation") or "main",
            ablation_value=row.get("ablation_value") or None,
            max_n_frames=int(row["max_n_frames"]),
            status=row.get("status") or STATUS_PENDING,
            output_root=Path(row["output_root"]),
            overrides=tuple(json.loads(row["overrides_json"])),
            command=row.get("command", ""),
            attempt=int(row.get("attempt") or 0),
            retry_of=row.get("retry_of") or None,
        )


def _plain_config(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Sweep configuration must be a mapping: {path}")
    return loaded


def _override(key: str, value: Any) -> str:
    if isinstance(value, str):
        encoded = json.dumps(value)
    else:
        encoded = json.dumps(value, separators=(",", ":"))
    return f"{key}={encoded}"


def _merged_block(defaults: Mapping[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update({key: value for key, value in block.items() if key != "overrides"})
    merged["overrides"] = {
        **dict(defaults.get("overrides", {})),
        **dict(block.get("overrides", {})),
    }
    return merged


def expand_suite_config(
    suite_config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[RunPlan]:
    suite_id = str(suite_config["suite_id"])
    run_namespace = suite_config.get("run_namespace")
    configured_root = Path(str(suite_config.get("output_root", "runs")))
    output_root = configured_root if configured_root.is_absolute() else repo_root / configured_root

    defaults = {
        key: value
        for key, value in suite_config.items()
        if key not in {"suite_id", "run_namespace", "output_root", "runs"}
    }
    blocks = suite_config.get("runs") or [{}]
    plans: list[RunPlan] = []
    seen_ids: set[str] = set()

    for raw_block in blocks:
        block = _merged_block(defaults, raw_block)
        algorithm = str(block.get("algorithm", "mappo"))
        task = str(block.get("task", "vmas_simple_spread"))
        models = block.get("models", [block.get("model", "comm_identity")])
        seeds = block.get("seeds", [0])
        ablation = str(block.get("ablation", "main"))
        ablation_value = block.get("ablation_value")
        max_n_frames = int(block.get("max_n_frames", 120_000))

        for model in models:
            for seed in seeds:
                run_id = make_run_id(
                    task=task,
                    algorithm=algorithm,
                    model=str(model),
                    seed=int(seed),
                    namespace=run_namespace,
                    ablation=ablation,
                    ablation_value=ablation_value,
                )
                if run_id in seen_ids:
                    raise ValueError(f"Duplicate run_id generated by suite: {run_id}")
                seen_ids.add(run_id)

                overrides = [
                    f"algorithm={algorithm}",
                    f"task={task}",
                    f"model={model}",
                    f"seed={int(seed)}",
                    f"experiment.max_n_frames={max_n_frames}",
                ]
                for key, value in block.get("overrides", {}).items():
                    overrides.append(_override(str(key), value))

                command_parts = [
                    "uv",
                    "run",
                    "python",
                    "scripts/train.py",
                    "--suite-id",
                    suite_id,
                    "--run-id",
                    run_id,
                    "--output-root",
                    str(output_root),
                    "--ablation",
                    ablation,
                ]
                if ablation_value is not None:
                    command_parts.extend(["--ablation-value", str(ablation_value)])
                command_parts.extend(overrides)

                plans.append(
                    RunPlan(
                        run_id=run_id,
                        suite_id=suite_id,
                        task=task,
                        algorithm=algorithm,
                        model=str(model),
                        seed=int(seed),
                        ablation=ablation,
                        ablation_value=(
                            None if ablation_value is None else str(ablation_value)
                        ),
                        max_n_frames=max_n_frames,
                        status=STATUS_PENDING,
                        output_root=output_root.resolve(),
                        overrides=tuple(overrides),
                        command=shlex.join(command_parts),
                    )
                )
    return plans


def write_manifest(path: Path, plans: Sequence[RunPlan]) -> None:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
    writer.writeheader()
    writer.writerows(plan.as_row() for plan in plans)
    atomic_write_text(path, stream.getvalue())


def read_manifest(path: Path) -> list[RunPlan]:
    with path.open(encoding="utf-8", newline="") as file:
        return [RunPlan.from_row(row) for row in csv.DictReader(file)]


def create_manifest(suite_path: Path, *, repo_root: Path) -> tuple[Path, list[RunPlan]]:
    suite_config = _plain_config(suite_path)
    plans = expand_suite_config(suite_config, repo_root=repo_root)
    if not plans:
        raise ValueError("Sweep expanded to zero runs.")
    suite_dir = plans[0].output_root / plans[0].suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(suite_dir / "suite_config.yaml", suite_path.read_text(encoding="utf-8"))
    manifest_path = suite_dir / "manifest.csv"
    write_manifest(manifest_path, plans)
    return manifest_path, plans


def sync_manifest_status(
    path: Path,
    plans: Sequence[RunPlan] | None = None,
    *,
    stale_after: float | None = None,
) -> list[RunPlan]:
    """Rebuild manifest status from authoritative per-run status files.

    With ``stale_after`` set, a ``running`` row whose worker has stopped
    reporting is recorded as ``failed`` so it appears in failure tables and
    becomes eligible for retry instead of silently blocking its slot.
    """

    current = list(plans) if plans is not None else read_manifest(path)
    synced = []
    for plan in current:
        run_dir = plan.output_root / plan.suite_id / plan.run_id
        status = current_status(run_dir)
        if (
            status == STATUS_RUNNING
            and stale_after is not None
            and is_stale_running(run_dir, stale_after)
        ):
            status = STATUS_FAILED
        synced.append(dataclasses.replace(plan, status=status or plan.status))
    write_manifest(path, synced)
    return synced


def manifest_status_counts(plans: Sequence[RunPlan]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
    return dict(sorted(counts.items()))


def select_plans(
    plans: Sequence[RunPlan],
    *,
    index: int | None = None,
    run_id: str | None = None,
    count: int = 1,
) -> list[RunPlan]:
    """Select rows by array index (optionally a chunk of ``count``) or run ID.

    Chunking lets one Slurm array task own several consecutive rows, which
    matters when each row is far too small to justify a whole GPU.
    """

    if index is not None and run_id is not None:
        raise ValueError("Select by either index or run_id, not both.")
    if count < 1:
        raise ValueError("count must be at least one")
    if index is not None:
        if index < 0:
            raise IndexError(f"Manifest index {index} is negative.")
        if index >= len(plans):
            raise IndexError(f"Manifest index {index} is out of range.")
        return list(plans[index : index + count])
    if run_id is not None:
        selected = [plan for plan in plans if plan.run_id == run_id]
        if not selected:
            raise KeyError(f"run_id not found in manifest: {run_id}")
        return selected
    return list(plans)


def execute_plan(
    plan: RunPlan,
    *,
    config_root: Path,
    repo_root: Path,
    retry_failed: bool = False,
    reclaim_stale_after: float | None = None,
    extra_overrides: Sequence[str] = (),
) -> tuple[RunPlan, Any | None]:
    """Execute one manifest row.

    ``extra_overrides`` carries machine-level settings that must not live in a
    suite YAML because they are properties of where the job runs, not of the
    experiment design -- the compute device above all. They are appended to the
    row's own overrides so the executed configuration and the recorded
    provenance stay identical.
    """

    run_dir = plan.output_root / plan.suite_id / plan.run_id
    existing = current_status(run_dir)
    if existing == STATUS_COMPLETED:
        return dataclasses.replace(plan, status=STATUS_COMPLETED), None

    # A row abandoned by a preempted or walltime-killed worker still says
    # "running". Treat it like a failure so it can be retried, but only after
    # it has demonstrably stopped reporting progress, so a live concurrent
    # worker is never stolen from.
    stale = (
        existing == STATUS_RUNNING
        and reclaim_stale_after is not None
        and is_stale_running(run_dir, reclaim_stale_after)
    )
    if existing not in {None, STATUS_FAILED} and not stale:
        return dataclasses.replace(plan, status=existing), None
    if existing is not None and not (retry_failed or stale):
        return dataclasses.replace(plan, status=existing), None

    if extra_overrides:
        plan = dataclasses.replace(
            plan,
            overrides=(*plan.overrides, *extra_overrides),
            command=shlex.join([*shlex.split(plan.command), *extra_overrides]),
        )

    context = RunContext(
        suite_id=plan.suite_id,
        run_id=plan.run_id,
        output_root=plan.output_root,
        command=tuple(shlex.split(plan.command)),
        ablation=plan.ablation,
        ablation_value=plan.ablation_value,
        attempt=plan.attempt,
        retry_of=plan.retry_of,
    )
    if existing is not None:
        context = retry_context(context)
        plan = dataclasses.replace(
            plan,
            run_id=context.run_id,
            attempt=context.attempt,
            retry_of=context.retry_of,
            status=STATUS_PENDING,
        )

    spec = load_experiment_spec(config_root, plan.overrides)
    try:
        experiment = run_managed_experiment(
            spec,
            context,
            repo_root=repo_root,
            overrides=plan.overrides,
        )
    except Exception:
        return dataclasses.replace(plan, status=STATUS_FAILED), None
    return dataclasses.replace(plan, status=STATUS_COMPLETED), experiment


def format_plan_table(plans: Iterable[RunPlan]) -> str:
    return "\n".join(f"[{index:03d}] {plan.command}" for index, plan in enumerate(plans))
