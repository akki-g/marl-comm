from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from commstudy.communication.base import CommModule
from commstudy.experiments.config import ExperimentSpec


STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
VALID_STATUSES = {
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
}


class RunDirectoryExistsError(FileExistsError):
    """Raised when a managed run would overwrite existing evidence."""


class RunAlreadyCompletedError(RunDirectoryExistsError):
    """Raised when a completed run is selected without an explicit retry."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: object) -> str:
    text = str(value).strip().lower().replace("vmas_", "")
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-_")
    return text or "none"


def model_label(model: str) -> str:
    return model.removeprefix("comm_")


def make_run_id(
    *,
    task: str,
    algorithm: str,
    model: str,
    seed: int,
    namespace: object | None = None,
    ablation: str = "main",
    ablation_value: object | None = None,
) -> str:
    parts = [_slug(task), _slug(algorithm), _slug(model_label(model))]
    if namespace is not None and str(namespace).strip().lower() not in {"", "none"}:
        parts.append(_slug(namespace))
    if ablation not in {"", "main", "none"}:
        parts.append(_slug(ablation))
        if ablation_value is not None:
            parts.append(_slug(ablation_value))
    parts.append(f"seed{int(seed):03d}")
    return "__".join(parts)


@dataclass(frozen=True)
class RunContext:
    suite_id: str
    run_id: str
    output_root: Path
    command: tuple[str, ...] = ()
    ablation: str = "main"
    ablation_value: str | None = None
    attempt: int = 0
    retry_of: str | None = None

    @property
    def suite_dir(self) -> Path:
        return self.output_root / self.suite_id

    @property
    def run_dir(self) -> Path:
        return self.suite_dir / self.run_id

    @property
    def benchmarl_dir(self) -> Path:
        return self.run_dir / "benchmarl"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def capture_git_state(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    status = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _run_git(repo_root, "diff", "--binary", "HEAD", "--")

    untracked_process = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    excluded_roots = {".venv", "runs", "results", "outputs"}
    untracked = []
    if untracked_process.returncode == 0:
        for raw_path in untracked_process.stdout.split(b"\0"):
            if not raw_path:
                continue
            relative = Path(os.fsdecode(raw_path))
            if relative.parts and relative.parts[0] not in excluded_roots:
                candidate = (repo_root / relative).resolve()
                try:
                    candidate.relative_to(repo_root.resolve())
                except ValueError:
                    continue
                if candidate.is_file():
                    untracked.append(relative)

    status_lines = []
    for line in status.stdout.splitlines():
        path_text = line[3:].split(" -> ")[-1].strip('"') if len(line) > 3 else ""
        path = Path(path_text)
        if path.parts and path.parts[0] in excluded_roots:
            continue
        status_lines.append(line)
    status_text = "\n".join(status_lines)
    patch_parts = [diff.stdout]
    for relative in untracked:
        untracked_diff = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", str(relative)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        # git diff --no-index returns 1 when it successfully found a difference.
        if untracked_diff.returncode in {0, 1}:
            patch_parts.append(untracked_diff.stdout)
    patch_text = "".join(patch_parts)
    dirty = bool(status_text) or commit.returncode != 0
    result: dict[str, Any] = {
        "commit_sha": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": dirty,
        "status": status_text.splitlines(),
        "untracked_files": [str(path) for path in untracked],
    }

    if dirty:
        patch_path = run_dir / "git.patch"
        atomic_write_text(patch_path, patch_text)
        result["patch_file"] = patch_path.name
        result["patch_sha256"] = hashlib.sha256(patch_text.encode()).hexdigest()

    return result


def capture_versions() -> dict[str, str | None]:
    names = ("benchmarl", "torch", "torchrl", "tensordict", "vmas")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = platform.python_version()
    return versions


def to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_serializable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _unique_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    parameters: dict[int, torch.nn.Parameter] = {}
    for module in modules:
        for parameter in module.parameters():
            parameters.setdefault(id(parameter), parameter)
    return list(parameters.values())


def parameter_counts(experiment: Any) -> dict[str, int]:
    actor_modules = list(experiment.group_policies.values())
    actor_parameters = _unique_parameters(actor_modules)

    comm_modules: dict[int, CommModule] = {}
    for actor in actor_modules:
        for module in actor.modules():
            if isinstance(module, CommModule):
                comm_modules.setdefault(id(module), module)
    comm_parameters = _unique_parameters(comm_modules.values())

    critic_modules = []
    for loss in experiment.losses.values():
        critic = getattr(loss, "critic_network", None)
        if isinstance(critic, torch.nn.Module):
            critic_modules.append(critic)
    critic_parameters = _unique_parameters(critic_modules)

    return {
        "actor_total": sum(parameter.numel() for parameter in actor_parameters),
        "actor_trainable": sum(
            parameter.numel() for parameter in actor_parameters if parameter.requires_grad
        ),
        "communication_total": sum(parameter.numel() for parameter in comm_parameters),
        "communication_trainable": sum(
            parameter.numel() for parameter in comm_parameters if parameter.requires_grad
        ),
        "critic_total": sum(parameter.numel() for parameter in critic_parameters),
        "critic_trainable": sum(
            parameter.numel() for parameter in critic_parameters if parameter.requires_grad
        ),
    }


def _resolved_document(spec: ExperimentSpec, experiment: Any | None = None) -> dict[str, Any]:
    document: dict[str, Any] = {"commstudy": to_serializable(spec)}
    if experiment is not None:
        document["benchmarl"] = {
            "experiment": to_serializable(vars(experiment.config)),
            "algorithm": to_serializable(vars(experiment.algorithm_config)),
            "task": to_serializable(experiment.task.config),
            "actor_model": to_serializable(vars(experiment.model_config)),
            "critic_model": to_serializable(vars(experiment.critic_model_config)),
        }
    return document


def write_resolved_config(
    context: RunContext,
    spec: ExperimentSpec,
    experiment: Any | None = None,
) -> None:
    document = _resolved_document(spec, experiment)
    yaml_text = OmegaConf.to_yaml(OmegaConf.create(document), resolve=True, sort_keys=True)
    atomic_write_text(context.run_dir / "resolved_config.yaml", yaml_text)


def current_status(run_dir: Path) -> str | None:
    status = read_json(run_dir / "status.json", {})
    value = status.get("status") if isinstance(status, dict) else None
    return value if value in VALID_STATUSES else None


def scheduler_identity() -> dict[str, str | int | None]:
    """Identify the process/job that owns a run, for stale-run reclamation."""

    return {
        "pid": os.getpid(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def seconds_since_heartbeat(run_dir: Path) -> float | None:
    """Age in seconds of a run's last liveness signal, or ``None`` if absent."""

    status = read_json(run_dir / "status.json", {})
    if not isinstance(status, dict):
        return None
    stamp = status.get("heartbeat") or status.get("timestamp")
    if not stamp:
        return None
    try:
        recorded = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - recorded).total_seconds())


def is_stale_running(run_dir: Path, stale_after_seconds: float) -> bool:
    """True when a run claims to be running but has stopped reporting.

    A worker killed by preemption, a walltime limit, or a crashed node leaves
    ``status.json`` at ``running`` forever. Without this check such a row is
    never retried and never reported as failed, so a whole suite can quietly
    finish "incomplete" with no failure record anywhere.
    """

    if current_status(run_dir) != STATUS_RUNNING:
        return False
    age = seconds_since_heartbeat(run_dir)
    return age is None or age > float(stale_after_seconds)


def retry_context(context: RunContext) -> RunContext:
    base_id = context.retry_of or context.run_id.split("__retry", maxsplit=1)[0]
    index = 1
    while (context.suite_dir / f"{base_id}__retry{index:02d}").exists():
        index += 1
    return dataclasses.replace(
        context,
        run_id=f"{base_id}__retry{index:02d}",
        attempt=index,
        retry_of=base_id,
    )


class RunRecorder:
    """Owns the durable state transitions for one managed run."""

    def __init__(
        self,
        context: RunContext,
        spec: ExperimentSpec,
        *,
        repo_root: Path,
        overrides: Sequence[str] = (),
    ) -> None:
        self.context = context
        self.spec = spec
        self.repo_root = repo_root.resolve()
        self.overrides = tuple(overrides)

    def start(self) -> None:
        self.context.suite_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.context.run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            if current_status(self.context.run_dir) == STATUS_COMPLETED:
                raise RunAlreadyCompletedError(self.context.run_id) from error
            raise RunDirectoryExistsError(self.context.run_id) from error

        self.context.benchmarl_dir.mkdir(parents=False, exist_ok=False)
        started = utc_now()
        metadata = {
            "schema_version": 1,
            "run_id": self.context.run_id,
            "suite_id": self.context.suite_id,
            "status": STATUS_RUNNING,
            "attempt": self.context.attempt,
            "retry_of": self.context.retry_of,
            "start_timestamp": started,
            "end_timestamp": None,
            "algorithm": self.spec.algorithm,
            "task": self.spec.task,
            "model": self.spec.model,
            "communication_module": self.spec.model_config.get("params", {}).get(
                "comm_class_path"
            ),
            "seed": self.spec.seed,
            "ablation": self.context.ablation,
            "ablation_value": self.context.ablation_value,
            "command": list(self.context.command),
            "overrides": list(self.overrides),
            "git": capture_git_state(self.repo_root, self.context.run_dir),
            "versions": capture_versions(),
            "runtime": {
                "platform": platform.platform(),
                "hostname": platform.node(),
                "executable": sys.executable,
                "device": self.spec.experiment.get("train_device"),
                "scheduler": scheduler_identity(),
                "torch_num_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
            "parameters": None,
            "benchmarl_output": None,
        }
        atomic_write_json(self.context.metadata_path, metadata)
        atomic_write_json(
            self.context.status_path,
            {
                "status": STATUS_RUNNING,
                "timestamp": started,
                "heartbeat": started,
                "owner": scheduler_identity(),
                "error": None,
            },
        )
        write_resolved_config(self.context, self.spec)

    def heartbeat(self) -> None:
        """Refresh the liveness stamp of a still-running run.

        Called from the metrics callback once per collection iteration, which
        is frequent enough to detect a killed worker and rare enough to cost
        nothing. It never downgrades a terminal status.
        """

        status = read_json(self.context.status_path, {})
        if not isinstance(status, dict) or status.get("status") != STATUS_RUNNING:
            return
        status["heartbeat"] = utc_now()
        atomic_write_json(self.context.status_path, status)

    def record_experiment(self, experiment: Any) -> None:
        metadata = read_json(self.context.metadata_path, {})
        metadata["parameters"] = parameter_counts(experiment)
        metadata["benchmarl_output"] = str(Path(experiment.folder_name).resolve())
        metadata["runtime"]["sampling_device"] = str(experiment.config.sampling_device)
        metadata["runtime"]["train_device"] = str(experiment.config.train_device)
        metadata["runtime"]["buffer_device"] = str(experiment.config.buffer_device)
        atomic_write_json(self.context.metadata_path, metadata)
        write_resolved_config(self.context, self.spec, experiment)

    def complete(self, experiment: Any) -> dict[str, Any]:
        ended = utc_now()
        checkpoint_dir = self.context.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy_path = checkpoint_dir / "policy_state.pt"
        temporary_policy_path = checkpoint_dir / f".policy_state.{os.getpid()}.tmp"
        torch.save(
            {
                "schema_version": 1,
                "run_id": self.context.run_id,
                "suite_id": self.context.suite_id,
                "algorithm": self.spec.algorithm,
                "task": self.spec.task,
                "model": self.spec.model,
                "seed": self.spec.seed,
                "frames": int(experiment.total_frames),
                "group_policies": {
                    group: policy.state_dict()
                    for group, policy in experiment.group_policies.items()
                },
            },
            temporary_policy_path,
        )
        os.replace(temporary_policy_path, policy_path)
        metadata = read_json(self.context.metadata_path, {})
        metadata.update(
            {
                "status": STATUS_COMPLETED,
                "end_timestamp": ended,
                "frames": int(experiment.total_frames),
                "iterations": int(experiment.n_iters_performed),
                "benchmarl_total_time_seconds": float(experiment.total_time),
                "policy_checkpoint": str(policy_path.relative_to(self.context.run_dir)),
            }
        )
        atomic_write_json(self.context.metadata_path, metadata)
        atomic_write_json(
            self.context.status_path,
            {"status": STATUS_COMPLETED, "timestamp": ended, "error": None},
        )
        summary = summarize_metrics_csv(self.context.run_dir / "metrics.csv")
        summary.update(
            {
                "schema_version": 1,
                "run_id": self.context.run_id,
                "suite_id": self.context.suite_id,
                "status": STATUS_COMPLETED,
                "frames": int(experiment.total_frames),
                "iterations": int(experiment.n_iters_performed),
                "training_time_seconds": float(experiment.total_time),
                "parameters": metadata.get("parameters"),
                "policy_checkpoint": metadata["policy_checkpoint"],
            }
        )
        atomic_write_json(self.context.run_dir / "summary.json", summary)
        return summary

    def fail(self, error: BaseException) -> None:
        ended = utc_now()
        error_record = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        metadata = read_json(self.context.metadata_path, {})
        metadata.update(
            {"status": STATUS_FAILED, "end_timestamp": ended, "error": error_record}
        )
        atomic_write_json(self.context.metadata_path, metadata)
        atomic_write_json(
            self.context.status_path,
            {"status": STATUS_FAILED, "timestamp": ended, "error": error_record},
        )
        atomic_write_json(
            self.context.run_dir / "summary.json",
            {
                "schema_version": 1,
                "run_id": self.context.run_id,
                "suite_id": self.context.suite_id,
                "status": STATUS_FAILED,
                "error": error_record,
            },
        )


def summarize_metrics_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "evaluation_points": 0,
            "mean_final_return": None,
            "return_auc": None,
            "normalized_return_auc": None,
        }

    points: dict[int, list[float]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if (
                row.get("phase") == "evaluation"
                and row.get("metric") == "return_mean"
                # The study figure is written without a group; per-group returns
                # are recorded alongside it and must not be averaged into the
                # curve. On a single-group task the two agree, so omitting this
                # is silently harmless; on PCP it divides the predators' return
                # by three. `_evaluation_curve` in analysis/aggregate.py applies
                # the same filter and the two must stay in step.
                and not row.get("group")
                and not row.get("sample")
            ):
                points.setdefault(int(row["frames"]), []).append(float(row["value"]))

    curve = sorted((frame, sum(values) / len(values)) for frame, values in points.items())
    if not curve:
        return {
            "evaluation_points": 0,
            "mean_final_return": None,
            "return_auc": None,
            "normalized_return_auc": None,
        }

    final_count = max(1, math.ceil(len(curve) * 0.10))
    final_return = sum(value for _, value in curve[-final_count:]) / final_count
    auc = sum(
        (right_frame - left_frame) * (left_value + right_value) / 2
        for (left_frame, left_value), (right_frame, right_value) in zip(
            curve, curve[1:], strict=False
        )
    )
    span = curve[-1][0] - curve[0][0]
    return {
        "evaluation_points": len(curve),
        "final_window_points": final_count,
        "final_window_definition": "mean of final ceil(10%) evaluation points",
        "mean_final_return": final_return,
        "return_auc": auc,
        "normalized_return_auc": auc / span if span > 0 else final_return,
    }
