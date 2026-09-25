"""Small durable I/O and provenance helpers shared by the runner and metrics."""

from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import torch
from commstudy.communication.base import CommModule


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        Path(path), json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
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
    excluded_roots = {".venv", "runs", "results", "outputs", "logs", "data", "build", "dist"}
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
    names = (
        "benchmarl",
        "torch",
        "torchrl",
        "tensordict",
        "vmas",
        "numpy",
        "scipy",
        "pandas",
        "pandapower",
        "gymnasium",
        "pettingzoo",
        "matplotlib",
        "omegaconf",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = platform.python_version()
    return versions


def _unique_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    parameters: dict[int, torch.nn.Parameter] = {}
    for module in modules:
        for parameter in module.parameters():
            parameters.setdefault(id(parameter), parameter)
    return list(parameters.values())


def _model_parameter_counts(actor_modules, critic_modules) -> dict[str, int]:
    actor_parameters = _unique_parameters(actor_modules)

    comm_modules: dict[int, CommModule] = {}
    for actor in actor_modules:
        for module in actor.modules():
            if isinstance(module, CommModule):
                comm_modules.setdefault(id(module), module)
    comm_parameters = _unique_parameters(comm_modules.values())

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


def parameter_counts(experiment: Any) -> dict[str, int]:
    """Keep aggregate counts and expose the same accounting for each group.

    ``actor_total`` includes the communication parameters; communication is
    reported separately as a subset, never added a second time. All counts
    deduplicate parameter identities, including framework-shared modules.
    Flat group keys remain compatible with tidy scalar setup metrics.
    """
    critics = {
        group: critic
        for group, loss in experiment.losses.items()
        if isinstance(critic := getattr(loss, "critic_network", None), torch.nn.Module)
    }
    counts = _model_parameter_counts(
        list(experiment.group_policies.values()), list(critics.values())
    )
    for group, actor in experiment.group_policies.items():
        critic_modules = [critics[group]] if group in critics else []
        for name, count in _model_parameter_counts([actor], critic_modules).items():
            counts[f"group/{group}/{name}"] = count
    return counts


def task_runtime_contract(experiment: Any) -> dict[str, Any]:
    """Runtime semantic versions shared by managed runs and analysis artifacts.

    These name implemented semantics, not an approved scientific configuration
    or a launch authorization. The evaluation guard also affects stock tasks,
    whose historical trajectories therefore retain their original provenance.
    """
    contract = {
        "schema_version": 1,
        "evaluation_randomness_protocol": "evaluation_rng_v2",
        "channel_randomness_protocol": "channel_rng_v2",
        "training_health_protocol": "training_health_v1",
    }
    task_contract = getattr(getattr(experiment, "task", None), "runtime_contract", None)
    if callable(task_contract):
        contract.update(task_contract(experiment.test_env))
    return contract
