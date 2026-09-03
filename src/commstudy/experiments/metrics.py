from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from benchmarl.experiment.callback import Callback
from tensordict import TensorDict

from commstudy.communication.base import CommModule
from commstudy.experiments.bookkeeping import parameter_counts
from commstudy.experiments.returns import (
    group_collection_returns,
    group_rollout_returns,
    mean_over_groups,
    resolve_return_groups,
)


METRIC_COLUMNS = (
    "timestamp",
    "frames",
    "iteration",
    "phase",
    "group",
    "metric",
    "value",
    "sample",
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().to(torch.float).mean().cpu().item()
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else scalar


class TidyMetricsWriter:
    """Append-only, long-form metrics writer for one isolated run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / "metrics.csv"

    def _ensure_header(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=METRIC_COLUMNS).writeheader()

    def write(
        self,
        *,
        frames: int,
        iteration: int,
        phase: str,
        metrics: Mapping[str, Any],
        group: str = "",
        sample: int | str | None = None,
    ) -> None:
        self._ensure_header()
        timestamp = _timestamp()
        with self.path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
            for metric, raw_value in metrics.items():
                value = _scalar(raw_value)
                if value is None:
                    continue
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "frames": int(frames),
                        "iteration": int(iteration),
                        "phase": phase,
                        "group": group,
                        "metric": metric,
                        "value": repr(value),
                        "sample": "" if sample is None else sample,
                    }
                )
            file.flush()


class ExperimentMetricsCallback(Callback):
    """Bridge BenchMARL's lifecycle to stable commstudy metrics."""

    def __init__(
        self,
        run_dir: str | Path,
        heartbeat: Callable[[], None] | None = None,
        return_groups: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.writer = TidyMetricsWriter(run_dir)
        self._heartbeat = heartbeat
        self._wall_start: float | None = None
        # Which groups' reward is the study's return. None means every group,
        # which is what a single-group task resolves to anyway.
        self._configured_return_groups = (
            None if return_groups is None else tuple(str(g) for g in return_groups)
        )
        self._return_groups: tuple[str, ...] = ()

    def _elapsed(self) -> float:
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        return time.perf_counter() - self._wall_start

    def _comm_modules(self, group: str | None = None) -> list[CommModule]:
        policies = self.experiment.group_policies
        selected = policies.items() if group is None else [(group, policies[group])]
        modules: dict[int, CommModule] = {}
        for _, policy in selected:
            for module in policy.modules():
                if isinstance(module, CommModule):
                    modules.setdefault(id(module), module)
        return list(modules.values())

    def _drain_comm_stats(self, group: str | None = None) -> dict[str, float]:
        collected: dict[str, list[float]] = {}
        for module in self._comm_modules(group):
            get_stats = getattr(module, "communication_stats", None)
            if callable(get_stats):
                stats = get_stats()
                for key, value in stats.items():
                    scalar = _scalar(value)
                    if scalar is not None:
                        collected.setdefault(str(key), []).append(scalar)
            reset = getattr(module, "reset_stats", None)
            if callable(reset):
                reset()
        return {
            f"comm_{key}": sum(values) / len(values)
            for key, values in collected.items()
            if values
        }

    def _forward_to_benchmarl(self, metrics: Mapping[str, float], prefix: str) -> None:
        if not metrics:
            return
        self.experiment.logger.log(
            {f"{prefix}/{key}": value for key, value in metrics.items()},
            # BenchMARL invokes on_setup before it initializes these counters.
            step=getattr(self.experiment, "n_iters_performed", 0),
        )

    def on_setup(self) -> None:
        self._wall_start = time.perf_counter()
        # Resolved here so a mistyped group name fails at run start rather than
        # producing a study whose headline metric is quietly absent.
        self._return_groups = resolve_return_groups(
            self.experiment.group_map,
            self._configured_return_groups,
        )
        counts = parameter_counts(self.experiment)
        metrics = {f"parameters_{key}": value for key, value in counts.items()}
        self.writer.write(
            frames=0,
            iteration=0,
            phase="setup",
            metrics=metrics,
        )
        self._forward_to_benchmarl(metrics, "commstudy")
        # Do not mix any setup/spec-check forwards into first collection statistics.
        self._drain_comm_stats()

    def on_batch_collected(self, batch) -> None:
        if self._heartbeat is not None:
            self._heartbeat()
        # Not experiment.mean_return: BenchMARL averages over every group, which
        # on a two-group zero-sum task such as PCP is identically zero. This is
        # the same per-group figure BenchMARL logs, restricted to the groups the
        # study measures, so a single-group task is unchanged.
        per_group = group_collection_returns(batch, tuple(self.experiment.group_map))
        study_return = mean_over_groups(self._measured(per_group))

        metrics: dict[str, Any] = {"wall_time_seconds": self._elapsed()}
        if study_return is not None:
            metrics["return_mean"] = study_return
        comm_stats = self._drain_comm_stats()
        metrics.update(comm_stats)
        self.writer.write(
            frames=self.experiment.total_frames,
            iteration=self.experiment.n_iters_performed,
            phase="collection",
            metrics=metrics,
        )
        self._write_group_returns("collection", per_group)
        self._forward_to_benchmarl(comm_stats, "collection/communication")

    def on_train_step(self, batch, group: str) -> TensorDict | None:
        del batch
        comm_stats = self._drain_comm_stats(group)
        if not comm_stats:
            return None
        device = torch.device(self.experiment.config.train_device)
        return TensorDict(
            {key: torch.tensor(value, device=device) for key, value in comm_stats.items()},
            batch_size=[],
            device=device,
        )

    def on_train_end(self, training_td, group: str) -> None:
        metrics: dict[str, float] = {}
        for key, value in training_td.items(include_nested=True, leaves_only=True):
            name = "/".join(key) if isinstance(key, tuple) else str(key)
            scalar = _scalar(value)
            if scalar is not None:
                metrics[name] = scalar
        metrics["wall_time_seconds"] = self._elapsed()
        self.writer.write(
            frames=self.experiment.total_frames,
            iteration=self.experiment.n_iters_performed,
            phase="training",
            group=group,
            metrics=metrics,
        )

    def _measured(self, per_group: Mapping[str, float]) -> dict[str, float]:
        """Keep only the groups whose reward the study reports."""

        return {
            group: value
            for group, value in per_group.items()
            if group in self._return_groups
        }

    def _write_group_returns(self, phase: str, per_group: Mapping[str, float]) -> None:
        """Record each measured group's own return beside the study figure.

        The aggregate at group "" is what the analysis reads. These rows exist
        so a mis-declared ``return_groups`` is visible in the data rather than
        only in the config, and so a scripted or otherwise excluded group can
        still be inspected.
        """

        for group, value in per_group.items():
            self.writer.write(
                frames=self.experiment.total_frames,
                iteration=self.experiment.n_iters_performed,
                phase=phase,
                group=group,
                metrics={"return_mean": value},
            )

    def on_evaluation_end(self, rollouts) -> None:
        episode_returns: list[float] = []
        per_group_totals: dict[str, list[float]] = {}
        for rollout in rollouts:
            # Restricted to the measured groups: averaging PCP's predators with
            # its scripted prey cancels to exactly zero.
            group_returns = group_rollout_returns(
                rollout, tuple(self.experiment.group_map)
            )
            for group, value in group_returns.items():
                per_group_totals.setdefault(group, []).append(value)
            episode_return = mean_over_groups(self._measured(group_returns))
            if episode_return is not None:
                episode_returns.append(episode_return)

        metrics: dict[str, float] = {}
        if episode_returns:
            metrics.update(
                {
                    "return_mean": sum(episode_returns) / len(episode_returns),
                    "return_min": min(episode_returns),
                    "return_max": max(episode_returns),
                    "episode_length_mean": sum(
                        rollout.batch_size[0] for rollout in rollouts
                    )
                    / len(rollouts),
                }
            )
        metrics["wall_time_seconds"] = self._elapsed()
        comm_stats = self._drain_comm_stats()
        metrics.update(comm_stats)
        self.writer.write(
            frames=self.experiment.total_frames,
            iteration=self.experiment.n_iters_performed,
            phase="evaluation",
            metrics=metrics,
        )
        self._write_group_returns(
            "evaluation",
            {
                group: sum(values) / len(values)
                for group, values in per_group_totals.items()
                if values
            },
        )
        for index, value in enumerate(episode_returns):
            self.writer.write(
                frames=self.experiment.total_frames,
                iteration=self.experiment.n_iters_performed,
                phase="evaluation",
                metrics={"return_episode": value},
                sample=index,
            )
        self._forward_to_benchmarl(comm_stats, "eval/communication")
