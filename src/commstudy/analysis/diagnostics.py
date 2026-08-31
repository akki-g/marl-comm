"""Run-level health diagnostics used as experiment protocol gates.

Two independent questions are answered here.

``finiteness_report`` and ``metric_trace`` read only the durable ``metrics.csv``
of a finished run. They answer "did anything logged during training become
non-finite, and how did the optimizer diagnostics evolve?".

``policy_action_diagnostics`` rebuilds one finished run from its recorded
overrides, strictly loads its saved actor, and measures the *behaviour* of that
frozen policy. It exists because the MAPPO baseline on VMAS Simple Spread can
fail through TanhNormal boundary saturation: the raw location/scale grow large
and ``tanh`` compresses sampled actions onto +/-1 long before any value becomes
non-finite. A finite metrics stream is therefore not sufficient evidence that a
training protocol is healthy.

Nothing here writes into the audited run directory.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.experiments.config import load_experiment_spec
from commstudy.experiments.runner import build_experiment


DEFAULT_SATURATION_THRESHOLD = 0.999

#: Optimizer/critic diagnostics inspected by the standard protocol gate.
GATE_TRAINING_METRICS = (
    "entropy",
    "loss_objective",
    "loss_critic",
    "explained_variance",
    "grad_norm_loss_objective",
    "grad_norm_loss_critic",
    "kl_approx",
    "clip_fraction",
    "ESS",
)


@dataclass(frozen=True)
class MetricTrace:
    """First/last/extreme values of one logged metric over a whole run."""

    phase: str
    metric: str
    count: int
    nonfinite_count: int
    first_frames: int | None
    last_frames: int | None
    first: float | None
    last: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class ActionDiagnostics:
    """Behaviour of a frozen actor under one exploration mode."""

    exploration: str
    episodes: int
    steps: int
    action_scalars: int
    finite_action_fraction: float
    saturated_action_fraction: float
    saturation_threshold: float
    mean_abs_action: float
    max_abs_action: float
    finite_location_fraction: float | None
    mean_abs_location: float | None
    max_abs_location: float | None
    finite_scale_fraction: float | None
    mean_scale: float | None
    max_scale: float | None


def _metric_rows(run_dir: Path) -> Iterator[dict[str, str]]:
    path = Path(run_dir) / "metrics.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def _value(row: Mapping[str, str]) -> float | None:
    try:
        return float(row["value"])
    except (KeyError, TypeError, ValueError):
        return None


def metric_trace(run_dir: Path, phase: str, metric: str) -> MetricTrace | None:
    """Summarize one ``(phase, metric)`` series, ignoring per-episode samples."""

    points: list[tuple[int, float]] = []
    nonfinite = 0
    for row in _metric_rows(run_dir):
        if row.get("phase") != phase or row.get("metric") != metric or row.get("sample"):
            continue
        value = _value(row)
        if value is None:
            continue
        if not math.isfinite(value):
            nonfinite += 1
            continue
        points.append((int(row["frames"]), value))

    if not points and not nonfinite:
        return None

    points.sort(key=lambda point: point[0])
    values = [value for _, value in points]
    return MetricTrace(
        phase=phase,
        metric=metric,
        count=len(points) + nonfinite,
        nonfinite_count=nonfinite,
        first_frames=points[0][0] if points else None,
        last_frames=points[-1][0] if points else None,
        first=values[0] if values else None,
        last=values[-1] if values else None,
        minimum=min(values) if values else None,
        maximum=max(values) if values else None,
    )


def finiteness_report(run_dir: Path) -> dict[str, Any]:
    """Count every logged scalar and locate any non-finite value."""

    total = 0
    nonfinite = 0
    by_metric: dict[str, int] = defaultdict(int)
    first_frames: int | None = None
    for row in _metric_rows(run_dir):
        value = _value(row)
        if value is None:
            continue
        total += 1
        if math.isfinite(value):
            continue
        nonfinite += 1
        by_metric[f"{row.get('phase')}/{row.get('metric')}"] += 1
        frames = int(row["frames"])
        first_frames = frames if first_frames is None else min(first_frames, frames)

    return {
        "run_id": Path(run_dir).name,
        "total_values": total,
        "nonfinite_values": nonfinite,
        "all_finite": nonfinite == 0,
        "nonfinite_metrics": dict(sorted(by_metric.items())),
        "first_nonfinite_frames": first_frames,
    }


def _named_leaves(batch: Any, name: str) -> list[torch.Tensor]:
    """Collect policy-written leaves called ``name``, skipping ``next`` copies."""

    found: list[torch.Tensor] = []
    for key, value in batch.items(include_nested=True, leaves_only=True):
        parts = key if isinstance(key, tuple) else (key,)
        if parts[0] == "next" or parts[-1] != name:
            continue
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            found.append(value.detach().reshape(-1).to(torch.float64))
    return found


def _finite_stats(tensors: Sequence[torch.Tensor], *, absolute: bool) -> tuple[float, float, float]:
    """Return (finite fraction, mean, max) over the finite entries only."""

    if not tensors:
        return math.nan, math.nan, math.nan
    values = torch.cat(tensors)
    if absolute:
        values = values.abs()
    finite = torch.isfinite(values)
    fraction = float(finite.to(torch.float64).mean())
    kept = values[finite]
    if kept.numel() == 0:
        return fraction, math.nan, math.nan
    return fraction, float(kept.mean()), float(kept.max())


def rebuild_spec(run_dir: Path, config_root: Path) -> Any:
    """Re-resolve the exact ExperimentSpec a finished run was launched with."""

    metadata = json.loads((Path(run_dir) / "metadata.json").read_text(encoding="utf-8"))
    return load_experiment_spec(config_root, list(metadata.get("overrides") or ()))


def load_frozen_experiment(
    run_dir: Path,
    *,
    config_root: Path,
    scratch_root: Path,
) -> Any:
    """Rebuild a run's experiment and strictly load its saved final actor.

    ``scratch_root`` receives every BenchMARL side effect, so the audited run
    directory is only ever read.
    """

    run_dir = Path(run_dir)
    spec = rebuild_spec(run_dir, config_root)
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    audit_spec = dataclasses.replace(
        spec,
        experiment={
            **spec.experiment,
            "save_folder": str(scratch_root.resolve()),
            "loggers": [],
            "create_json": False,
            "checkpoint_interval": 0,
            "checkpoint_at_end": False,
        },
    )
    experiment = build_experiment(audit_spec)

    checkpoint = torch.load(
        run_dir / "checkpoints" / "policy_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    for group, state_dict in checkpoint["group_policies"].items():
        experiment.group_policies[group].load_state_dict(state_dict, strict=True)
    return experiment


def policy_action_diagnostics(
    experiment: Any,
    *,
    exploration: str = "DETERMINISTIC",
    episodes: int = 5,
    steps: int | None = None,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    seed: int | None = None,
) -> ActionDiagnostics:
    """Roll a frozen actor out and measure action/location/scale saturation.

    The environment seed is reset per call so RANDOM and DETERMINISTIC modes
    are measured on the same starting states.
    """

    mode = ExplorationType[exploration.upper()]
    max_steps = int(steps if steps is not None else experiment.max_steps)
    if seed is not None:
        experiment.test_env.set_seed(int(seed))

    batched = experiment.test_env.batch_size != ()
    rollouts = []
    with torch.no_grad(), set_exploration_type(mode):
        if batched:
            rollout = experiment.test_env.rollout(
                max_steps=max_steps,
                policy=experiment.policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
            rollouts = list(rollout.unbind(0))
        else:
            for _ in range(int(episodes)):
                rollouts.append(
                    experiment.test_env.rollout(
                        max_steps=max_steps,
                        policy=experiment.policy,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                )

    actions = [tensor for rollout in rollouts for tensor in _named_leaves(rollout, "action")]
    locations = [tensor for rollout in rollouts for tensor in _named_leaves(rollout, "loc")]
    scales = [tensor for rollout in rollouts for tensor in _named_leaves(rollout, "scale")]

    action_values = torch.cat(actions) if actions else torch.empty(0, dtype=torch.float64)
    finite_actions = torch.isfinite(action_values)
    kept_actions = action_values[finite_actions].abs()
    saturated = (
        float((kept_actions > saturation_threshold).to(torch.float64).mean())
        if kept_actions.numel()
        else math.nan
    )

    location_fraction, location_mean, location_max = _finite_stats(locations, absolute=True)
    scale_fraction, scale_mean, scale_max = _finite_stats(scales, absolute=False)

    return ActionDiagnostics(
        exploration=mode.name,
        episodes=len(rollouts),
        steps=max_steps,
        action_scalars=int(action_values.numel()),
        finite_action_fraction=(
            float(finite_actions.to(torch.float64).mean()) if action_values.numel() else math.nan
        ),
        saturated_action_fraction=saturated,
        saturation_threshold=float(saturation_threshold),
        mean_abs_action=float(kept_actions.mean()) if kept_actions.numel() else math.nan,
        max_abs_action=float(kept_actions.max()) if kept_actions.numel() else math.nan,
        finite_location_fraction=None if not locations else location_fraction,
        mean_abs_location=None if not locations else location_mean,
        max_abs_location=None if not locations else location_max,
        finite_scale_fraction=None if not scales else scale_fraction,
        mean_scale=None if not scales else scale_mean,
        max_scale=None if not scales else scale_max,
    )


def audit_run(
    run_dir: Path,
    *,
    config_root: Path,
    scratch_root: Path | None = None,
    explorations: Sequence[str] = ("DETERMINISTIC", "RANDOM"),
    episodes: int = 5,
    steps: int | None = None,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    seed: int | None = 0,
    metrics: Sequence[str] = GATE_TRAINING_METRICS,
) -> dict[str, Any]:
    """Full protocol gate for one finished run.

    The policy rollout half is skipped when ``scratch_root`` is ``None`` or the
    run has no saved actor, so the cheap metrics half stays usable on its own.
    """

    run_dir = Path(run_dir)
    report: dict[str, Any] = finiteness_report(run_dir)
    status_path = run_dir / "status.json"
    report["status"] = (
        json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if status_path.exists()
        else None
    )

    evaluation = metric_trace(run_dir, "evaluation", "return_mean")
    report["evaluation_points"] = evaluation.count if evaluation else 0
    report["final_evaluation_return"] = evaluation.last if evaluation else None

    for metric in metrics:
        trace = metric_trace(run_dir, "training", metric)
        if trace is None:
            continue
        report[f"{metric}_first"] = trace.first
        report[f"{metric}_last"] = trace.last
        report[f"{metric}_min"] = trace.minimum
        report[f"{metric}_max"] = trace.maximum

    checkpoint = run_dir / "checkpoints" / "policy_state.pt"
    if scratch_root is None or not checkpoint.exists():
        report["policy_audited"] = False
        return report

    experiment = load_frozen_experiment(
        run_dir,
        config_root=config_root,
        scratch_root=Path(scratch_root),
    )
    try:
        for exploration in explorations:
            diagnostics = policy_action_diagnostics(
                experiment,
                exploration=exploration,
                episodes=episodes,
                steps=steps,
                saturation_threshold=saturation_threshold,
                seed=seed,
            )
            prefix = diagnostics.exploration.lower()
            for field, value in dataclasses.asdict(diagnostics).items():
                if field == "exploration":
                    continue
                report[f"{prefix}_{field}"] = value
    finally:
        close = getattr(experiment.test_env, "close", None)
        if callable(close):
            close()
    report["policy_audited"] = True
    return report
