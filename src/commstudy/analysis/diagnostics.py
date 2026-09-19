"""Run-level health diagnostics used as experiment protocol gates.

Two independent questions are answered here.

``finiteness_report`` and ``metric_trace`` read only the durable ``metrics.csv``
of a finished run. They answer "did anything logged during training become
non-finite, and how did the optimizer diagnostics evolve?".

``policy_action_diagnostics`` rebuilds one finished run from its saved resolved
specification, strictly loads its saved actor, and measures the *behaviour* of that
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
import hashlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.analysis.saliency import _fresh_env, _measurement_context
from commstudy.experiments.bookkeeping import capture_versions, task_runtime_contract
from commstudy.experiments.runner import build_experiment
from commstudy.experiments.returns import RETURN_GROUPS_KEY
from commstudy.utils.rng import preserve_rng_state


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
    group: str
    episode_ids: tuple[str, ...]
    episode_transitions: tuple[int, ...]
    complete_episodes: int
    finite_action_scalars: int
    saturation_denominator: str
    action_abs_quantiles: dict[str, float]
    location_abs_quantiles: dict[str, float] | None
    scale_quantiles: dict[str, float] | None


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


def metric_trace(
    run_dir: Path,
    phase: str,
    metric: str,
    group: str | None = None,
) -> MetricTrace | None:
    """Summarize one ``(phase, metric)`` series, ignoring per-episode samples.

    ``group`` selects one group's rows. Omitting it is allowed only when the
    requested metric contains one group; mixed groups raise. Pass
    ``""`` for a metric that also has a per-group breakdown, such as the
    evaluation return, to read the study figure rather than a mixture of it and
    its own components.
    """

    points: list[tuple[int, float]] = []
    nonfinite = 0
    found_groups = set()
    for row in _metric_rows(run_dir):
        if row.get("phase") != phase or row.get("metric") != metric or row.get("sample"):
            continue
        if group is not None and (row.get("group") or "") != group:
            continue
        found_groups.add(row.get("group") or "")
        value = _value(row)
        if value is None:
            continue
        if not math.isfinite(value):
            nonfinite += 1
            continue
        points.append((int(row["frames"]), value))

    if group is None and len(found_groups) > 1:
        raise ValueError(f"Metric {phase}/{metric} has multiple groups; select group explicitly.")

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


def finiteness_report(run_dir: Path, group: str | None = None) -> dict[str, Any]:
    """Count scalars, optionally selecting one group plus unscoped study rows."""

    total = 0
    nonfinite = 0
    by_metric: dict[str, int] = defaultdict(int)
    first_frames: int | None = None
    for row in _metric_rows(run_dir):
        if group is not None and row.get("group") not in (None, "", group):
            continue
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


def _named_leaves(batch: Any, name: str, group: str) -> list[torch.Tensor]:
    """Read exactly the declared group's policy leaf, never info/next copies."""
    value = batch.get((group, name), None)
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return [value.detach().reshape(-1).to(device="cpu", dtype=torch.float64)]
    return []


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


def rebuild_spec(run_dir: Path, config_root: Path | None = None) -> Any:
    """Read the saved spec without consulting today's project YAML defaults.

    ``config_root`` is retained for callers of the previous API and is unused.
    A missing snapshot is an error; recorded override strings are not an exact
    reconstruction mechanism. Saved effective BenchMARL defaults fill only
    fields absent from the saved commstudy specification.
    """
    from commstudy.experiments.config import experiment_spec_from_dict, scientific_config_sha256

    del config_root
    path = Path(run_dir) / "resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing resolved specification: {path}")
    document = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(document, Mapping) or not isinstance(document.get("commstudy"), Mapping):
        raise ValueError(f"{path} must contain a complete commstudy specification.")
    raw = dict(document["commstudy"])
    effective = document.get("benchmarl") or {}
    if isinstance(effective, Mapping):
        if isinstance(effective.get("experiment"), Mapping):
            raw["experiment"] = {**effective["experiment"], **raw.get("experiment", {})}
        for component in ("task", "algorithm"):
            saved = effective.get(component)
            if isinstance(saved, Mapping):
                key = f"{component}_config"
                values = dict(raw.get(key) or {})
                values["params"] = {**saved, **values.get("params", {})}
                raw[key] = values
    spec = experiment_spec_from_dict(raw)
    document_hash = document.get("scientific_config_sha256")
    if document_hash is not None and document_hash != scientific_config_sha256(spec):
        raise ValueError(
            "Resolved-document scientific configuration hash does not match its snapshot."
        )
    return spec


_ANALYSIS_OVERRIDE_KEYS = {
    "sampling_device",
    "train_device",
    "buffer_device",
    "evaluation_episodes",
    "render",
}


def _analysis_spec(spec, scratch_root, overrides):
    values = dict(overrides or {})
    unknown = set(values) - _ANALYSIS_OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"Unsupported analysis-only experiment overrides: {sorted(unknown)}")
    automatic = {
        "save_folder": str(scratch_root.resolve()),
        "loggers": [],
        "create_json": False,
        "checkpoint_interval": 0,
        "checkpoint_at_end": False,
        "restore_file": None,
        "restore_map_location": None,
        "render": False,
    }
    changed = {**automatic, **values}
    return dataclasses.replace(spec, experiment={**spec.experiment, **changed}), changed


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_compatibility(metadata, spec):
    from commstudy.experiments.provenance import source_fingerprint

    current_versions = capture_versions()
    saved_versions = metadata.get("versions") or {}
    mismatches = {
        f"versions.{name}": {"saved": saved_versions.get(name), "current": version}
        for name, version in current_versions.items()
        if saved_versions.get(name) != version
    }
    source = source_fingerprint(Path(__file__).resolve().parents[3])
    if metadata.get("source_sha256") != source:
        mismatches["source_sha256"] = {"saved": metadata.get("source_sha256"), "current": source}
    # Device changes must be deliberate even when compatible weights can load.
    for key in ("sampling_device", "train_device", "buffer_device"):
        saved = (metadata.get("runtime") or {}).get(key)
        current = spec.experiment.get(key)
        if saved is not None and saved != current:
            mismatches[f"runtime.{key}"] = {"saved": saved, "current": current}
    threads = (metadata.get("runtime") or {}).get("torch_num_threads")
    if threads is not None and threads != torch.get_num_threads():
        mismatches["runtime.torch_num_threads"] = {
            "saved": threads,
            "current": torch.get_num_threads(),
        }
    return mismatches, current_versions, source


def load_frozen_experiment(
    run_dir: Path,
    *,
    config_root: Path | None = None,
    scratch_root: Path,
    analysis_overrides: Mapping[str, Any] | None = None,
    allow_runtime_mismatch: bool = False,
) -> Any:
    """Rebuild a run's experiment and strictly load its saved final actor.

    ``scratch_root`` receives every BenchMARL side effect, so the audited run
    directory is only ever read. Missing/mismatched source, software, device,
    or task semantics fail by default. An explicit allow_runtime_mismatch
    acknowledgement records a labelled re-evaluation under current semantics;
    it does not claim historical trajectory reproduction.
    """

    from commstudy.experiments.config import scientific_config_sha256

    run_dir = Path(run_dir).resolve()
    spec = rebuild_spec(run_dir, config_root)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    saved_hash = metadata.get("scientific_config_sha256")
    scientific_hash = scientific_config_sha256(spec)
    if saved_hash is not None and saved_hash != scientific_hash:
        raise ValueError(
            "Saved scientific configuration hash does not match the resolved snapshot."
        )
    scratch_root = Path(scratch_root).resolve()
    if scratch_root == run_dir or run_dir in scratch_root.parents:
        raise ValueError("scratch_root must be outside the audited run directory.")
    audit_spec, applied_overrides = _analysis_spec(spec, scratch_root, analysis_overrides)
    mismatches, versions, source = _runtime_compatibility(metadata, audit_spec)
    if mismatches and not allow_runtime_mismatch:
        raise ValueError(
            "Saved source/runtime is missing or incompatible; use an explicit "
            f"allow_runtime_mismatch acknowledgement for a labelled re-evaluation: {mismatches}"
        )
    scratch_root.mkdir(parents=True, exist_ok=True)
    with preserve_rng_state():
        experiment = build_experiment(audit_spec)
        try:
            contract = task_runtime_contract(experiment)
            if metadata.get("task_runtime_contract") != contract:
                mismatches["task_runtime_contract"] = {
                    "saved": metadata.get("task_runtime_contract"),
                    "current": contract,
                }
                if not allow_runtime_mismatch:
                    raise ValueError("Saved task runtime contract is missing or incompatible.")
            path = run_dir / "checkpoints" / "policy_state.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            policies = checkpoint.get("group_policies")
            if not isinstance(policies, Mapping) or set(policies) != set(experiment.group_policies):
                raise ValueError(
                    "Checkpoint policy group set must exactly match the resolved task."
                )
            for group, state_dict in policies.items():
                experiment.group_policies[group].load_state_dict(state_dict, strict=True)
            experiment.analysis_provenance = {
                "source_run": str(run_dir),
                "resolved_config_sha256": _file_sha256(run_dir / "resolved_config.yaml"),
                "checkpoint_sha256": _file_sha256(path),
                "scientific_config_sha256": scientific_hash,
                "source_sha256": source,
                "versions": versions,
                "analysis_overrides": applied_overrides,
                "allow_runtime_mismatch": allow_runtime_mismatch,
                "compatibility_mismatches": mismatches,
                "task_runtime_contract": contract,
            }
            return experiment
        except BaseException:
            experiment.close()
            raise


def policy_action_diagnostics(
    experiment: Any,
    *,
    exploration: str = "DETERMINISTIC",
    episodes: int = 5,
    steps: int | None = None,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    seed: int | None = None,
    group: str | None = None,
) -> ActionDiagnostics:
    """Roll a frozen actor out and measure action/location/scale saturation.

    Each requested episode gets a fresh one-world environment and its own seed.
    Batched training/evaluation environment sizes cannot change the requested
    sample count. RANDOM and DETERMINISTIC modes use the same episode seed bank.
    A shortened horizon is reported explicitly through complete_episodes.
    """
    if group is None:
        if len(experiment.group_map) != 1:
            raise ValueError("Action diagnostics require an explicit group for a multi-group task.")
        group = next(iter(experiment.group_map))
    if group not in experiment.group_map:
        raise ValueError(f"Unknown measured group {group!r}.")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer.")
    max_steps = steps if steps is not None else experiment.max_steps
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("steps must be a positive integer.")
    if not math.isfinite(saturation_threshold) or not 0 < saturation_threshold < 1:
        raise ValueError("saturation_threshold must be finite and between zero and one.")
    if experiment.model_config.is_rnn:
        raise ValueError("Recurrent action diagnostics require a validated history/reset contract.")
    base_seed = experiment.seed if seed is None else int(seed)
    mode = ExplorationType[exploration.upper()]
    rollouts = []
    completed = 0
    transitions = []
    episode_ids = []
    for index in range(episodes):
        episode_seed = base_seed + index
        with _measurement_context(experiment, episode_seed), set_exploration_type(mode):
            env = _fresh_env(experiment, episode_seed)
            try:
                rollout = env.rollout(
                    max_steps=max_steps,
                    policy=experiment.policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                if env.batch_size:
                    if env.batch_size != torch.Size([1]):
                        raise ValueError(
                            "Action audit requires one simulator instance per episode."
                        )
                    rollout = rollout[0]
                transitions.append(int(rollout.shape[0]))
                done = rollout.get(("next", "done"), None)
                completed += int(done is not None and bool(done[-1].all()))
                rollouts.append(rollout.cpu())
                episode_ids.append(f"seed:{base_seed}/episode:{index}")
            finally:
                env.close()

    actions = [tensor for rollout in rollouts for tensor in _named_leaves(rollout, "action", group)]
    locations = [tensor for rollout in rollouts for tensor in _named_leaves(rollout, "loc", group)]
    scales = [tensor for rollout in rollouts for tensor in _named_leaves(rollout, "scale", group)]
    if not actions:
        raise ValueError(f"No floating action leaf for measured group {group!r}.")

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
        group=group,
        episode_ids=tuple(episode_ids),
        episode_transitions=tuple(transitions),
        complete_episodes=completed,
        finite_action_scalars=int(finite_actions.sum()),
        saturation_denominator="finite_action_scalars",
        action_abs_quantiles=_quantiles(actions, absolute=True),
        location_abs_quantiles=_quantiles(locations, absolute=True) if locations else None,
        scale_quantiles=_quantiles(scales, absolute=False) if scales else None,
    )


def _quantiles(tensors, *, absolute):
    values = torch.cat(tensors)
    values = values[torch.isfinite(values)]
    if absolute:
        values = values.abs()
    names = ("p05", "p50", "p95", "p99")
    if not values.numel():
        return dict.fromkeys(names, math.nan)
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95, 0.99], dtype=values.dtype))
    return dict(zip(names, quantiles.tolist(), strict=True))


def audit_run(
    run_dir: Path,
    *,
    config_root: Path | None = None,
    scratch_root: Path | None = None,
    explorations: Sequence[str] = ("DETERMINISTIC", "RANDOM"),
    episodes: int = 5,
    steps: int | None = None,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    seed: int | None = 0,
    metrics: Sequence[str] = GATE_TRAINING_METRICS,
    group: str | None = None,
    analysis_overrides: Mapping[str, Any] | None = None,
    allow_runtime_mismatch: bool = False,
) -> dict[str, Any]:
    """Full protocol gate for one finished run.

    The policy rollout half is skipped when ``scratch_root`` is ``None`` or the
    run has no saved actor, so the cheap metrics half stays usable on its own.
    """

    run_dir = Path(run_dir)
    spec = rebuild_spec(run_dir, config_root)
    from commstudy.experiments.config import scientific_config_sha256

    scientific_hash = scientific_config_sha256(spec)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    recorded_hash = metadata.get("scientific_config_sha256")
    if recorded_hash is not None and recorded_hash != scientific_hash:
        raise ValueError(
            "Saved scientific configuration hash does not match the resolved snapshot."
        )
    available = {
        row["group"]
        for row in _metric_rows(run_dir)
        if row.get("phase") == "training" and row.get("group")
    }
    configured = spec.task_config.get(RETURN_GROUPS_KEY)
    if group is None:
        candidates = tuple(configured) if configured is not None else tuple(sorted(available))
        if len(candidates) != 1:
            raise ValueError("Run audit requires an explicit measured group for ambiguous tasks.")
        group = candidates[0]
    if available and group not in available:
        raise ValueError(
            f"Measured group {group!r} has no training rows; available: {sorted(available)}"
        )
    report: dict[str, Any] = finiteness_report(run_dir, group=group)
    report["measured_group"] = group
    report["scientific_config_sha256"] = scientific_hash
    report["all_groups_finiteness"] = finiteness_report(run_dir)
    status_path = run_dir / "status.json"
    report["status"] = (
        json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if status_path.exists()
        else None
    )

    evaluation = metric_trace(run_dir, "evaluation", "return_mean", group=group)
    if evaluation is None and (
        tuple(configured or ()) == (group,) or (configured is None and available == {group})
    ):
        evaluation = metric_trace(run_dir, "evaluation", "return_mean", group="")
    report["evaluation_points"] = evaluation.count if evaluation else 0
    report["final_evaluation_return"] = evaluation.last if evaluation else None

    missing_metrics = []
    for metric in metrics:
        trace = metric_trace(run_dir, "training", metric, group=group)
        if trace is None:
            missing_metrics.append(metric)
            continue
        report[f"{metric}_first"] = trace.first
        report[f"{metric}_last"] = trace.last
        report[f"{metric}_min"] = trace.minimum
        report[f"{metric}_max"] = trace.maximum
    report["missing_training_metrics"] = missing_metrics
    report["required_training_metrics_present"] = not missing_metrics

    checkpoint = run_dir / "checkpoints" / "policy_state.pt"
    if scratch_root is None or not checkpoint.exists():
        report["policy_audited"] = False
        return report

    experiment = load_frozen_experiment(
        run_dir,
        config_root=config_root,
        scratch_root=Path(scratch_root),
        analysis_overrides=analysis_overrides,
        allow_runtime_mismatch=allow_runtime_mismatch,
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
                group=group,
            )
            prefix = diagnostics.exploration.lower()
            for field, value in dataclasses.asdict(diagnostics).items():
                if field == "exploration":
                    continue
                report[f"{prefix}_{field}"] = value
        report["analysis_provenance"] = experiment.analysis_provenance
    finally:
        experiment.close()
    report["policy_audited"] = True
    return report
