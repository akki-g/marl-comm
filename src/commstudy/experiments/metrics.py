from __future__ import annotations

import csv
import json
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
from commstudy.experiments.health import TrainingHealthCallback
from commstudy.tasks.returns import (
    group_collection_returns,
    collection_episode_returns,
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


def tensor_health_metrics(value: torch.Tensor, prefix: str) -> dict[str, float]:
    """Describe actual scalars, keeping the denominator and finite rate explicit."""
    values = value.detach().reshape(-1).to(dtype=torch.float32)
    if not values.numel():
        return {}
    finite = torch.isfinite(values)
    metrics = {
        f"{prefix}_scalar_count": float(values.numel()),
        f"{prefix}_finite_fraction": float(finite.float().mean().cpu()),
    }
    values = values[finite]
    if values.numel():
        quantiles = torch.quantile(values, values.new_tensor([0.01, 0.5, 0.99]))
        metrics.update(
            {
                f"{prefix}_mean": float(values.mean().cpu()),
                f"{prefix}_std": float(values.std(correction=0).cpu()),
                f"{prefix}_min": float(values.min().cpu()),
                f"{prefix}_max": float(values.max().cpu()),
                **{
                    f"{prefix}_{name}": float(v.cpu())
                    for name, v in zip(("q01", "q50", "q99"), quantiles, strict=True)
                },
            }
        )
    return metrics


def group_health_metrics(batch, group: str) -> dict[str, float]:
    """Never discover or concatenate another group's policy output leaves."""
    metrics: dict[str, float] = {"health_transition_count": float(batch.numel())}
    for key in ("loc", "scale", "action", "value_target", "advantage", "state_value"):
        value = batch.get((group, key), None)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            metrics.update(tensor_health_metrics(value, key))
            if key == "action":
                metrics["action_saturated_fraction"] = float(
                    (value.detach().abs() > 0.999).float().mean().cpu()
                )
    return metrics


class CommunicationHealthObserver:
    """Bounded detached sums; each forward is weighted by environment transitions.

    Norms describe the net residual contribution of all configured rounds.
    Gate saturation has its own sender-round denominator. Optimization metrics
    describe replay compute and are kept in training rows, never deployed cost.
    """

    def __init__(self, module: CommModule, checking_replay: Callable[[], bool]):
        self.module = module
        self.checking_replay = lambda: (
            checking_replay() or getattr(module, "_health_replay_check", False)
        )
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.transitions = 0
        self.agent_transitions = 0
        self.handles = [
            module.register_forward_pre_hook(self._before),
            module.register_forward_hook(self._after),
        ]
        gate = getattr(module, "gate_network", None)
        if isinstance(gate, torch.nn.Module):
            self.handles.append(gate.register_forward_hook(self._gate))

    def _before(self, module, args):
        if not self.checking_replay():
            h = args[0]
            module._stats_transition_weight = h.numel() // (h.shape[-2] * h.shape[-1])

    def _add(self, key: str, value: torch.Tensor):
        value = value.detach()
        self.sums[key] = self.sums.get(key, 0.0) + float(value.sum().cpu())
        self.counts[key] = self.counts.get(key, 0) + value.numel()

    def _after(self, module, args, output):
        if self.checking_replay():
            return
        h = args[0].detach()
        encoder = h.norm(dim=-1)
        delta = (output.detach() - h).norm(dim=-1)
        self.transitions += h.numel() // (h.shape[-2] * h.shape[-1])
        self.agent_transitions += encoder.numel()
        self._add("encoder_norm_mean", encoder)
        self._add("contribution_norm_mean", delta)
        positive = encoder > 0
        self._add("zero_encoder_fraction", (~positive).float())
        self._add("contribution_to_encoder_norm_ratio", delta[positive] / encoder[positive])
        if module.normalize_comm_path:
            self._add("normalization_gain", module.comm_path_scale.detach().expand_as(encoder))

    def _gate(self, module, args, output):
        del module, args
        if self.checking_replay():
            return
        gate = output.detach().sigmoid()
        self._add("gate_saturated_low_fraction", (gate < 0.01).float())
        self._add("gate_saturated_high_fraction", (gate > 0.99).float())

    def drain(self) -> dict[str, float]:
        result = {
            key: value / self.counts[key] for key, value in self.sums.items() if self.counts[key]
        }
        result["transition_count"] = float(self.transitions)
        result["agent_transition_count"] = float(self.agent_transitions)
        for name in ("contribution_to_encoder_norm_ratio", "gate_saturated_low_fraction"):
            if name in self.counts:
                result[f"{name}_count"] = float(self.counts[name])
        self.sums.clear()
        self.counts.clear()
        self.transitions = self.agent_transitions = 0
        return result


class TidyMetricsWriter:
    """Append-only, long-form metrics writer for one isolated run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / "metrics.csv"
        self.on_record = None

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
        with (
            self.path.open("a", encoding="utf-8", newline="") as file,
            self.path.with_name("run.out").open("a", encoding="utf-8") as out,
        ):
            writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
            for metric, raw_value in metrics.items():
                value = _scalar(raw_value)
                if value is None:
                    continue
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite metric {phase}/{group}/{metric}: {value}")
                record = {
                    "timestamp": timestamp,
                    "frames": int(frames),
                    "iteration": int(iteration),
                    "phase": phase,
                    "group": group,
                    "metric": metric,
                    "value": repr(value),
                    "sample": "" if sample is None else sample,
                }
                writer.writerow(record)
                out.write("METRIC " + json.dumps(record, sort_keys=True) + "\n")
                if self.on_record is not None:
                    self.on_record(record)
            file.flush()


class ExperimentMetricsCallback(Callback):
    """Bridge BenchMARL's lifecycle to stable commstudy metrics."""

    def __init__(
        self,
        run_dir: str | Path,
        return_groups: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.writer = TidyMetricsWriter(run_dir)
        self._wall_start: float | None = None
        # Which groups' reward is the study's return. None means every group,
        # which is what a single-group task resolves to anyway.
        self._configured_return_groups = (
            None if return_groups is None else tuple(str(g) for g in return_groups)
        )
        self._return_groups: tuple[str, ...] = ()
        self._comm_observers: dict[int, CommunicationHealthObserver] = {}
        self._pcp_domain = False
        self._mapdn_domain = False

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
                observer = self._comm_observers.get(id(module))
                if observer is not None:
                    stats.update(observer.drain())
                for key, value in stats.items():
                    scalar = _scalar(value)
                    if scalar is not None:
                        collected.setdefault(str(key), []).append(scalar)
            reset = getattr(module, "reset_stats", None)
            if callable(reset):
                reset()
        return {
            f"comm_{key}": sum(values) / len(values) for key, values in collected.items() if values
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
        self._pcp_domain = (
            getattr(getattr(self.experiment, "task", None), "name", "").lower()
            == "predator_capture_prey"
        )
        env_name = getattr(getattr(self.experiment, "task", None), "env_name", lambda: "")
        self._mapdn_domain = env_name() == "mapdn"
        # Resolved here so a mistyped group name fails at run start rather than
        # producing a study whose headline metric is quietly absent.
        self._return_groups = resolve_return_groups(
            self.experiment.group_map,
            self._configured_return_groups,
        )
        self._comm_observers = {
            id(module): CommunicationHealthObserver(
                module, lambda: getattr(self.experiment, "_checking_policy_replay", False)
            )
            for module in self._comm_modules()
        }
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
        # Not experiment.mean_return: BenchMARL averages over every group, which
        # on a two-group zero-sum task such as PCP is identically zero. This is
        # the same per-group figure BenchMARL logs, restricted to the groups the
        # study measures, so a single-group task is unchanged.
        per_group = group_collection_returns(batch, tuple(self.experiment.group_map))
        study_return = mean_over_groups(self._measured(per_group))

        metrics: dict[str, Any] = {"wall_time_seconds": self._elapsed()}
        if study_return is not None:
            metrics["return_mean"] = study_return
        per_group_comm = self._write_group_health("collection", batch)
        comm_stats = self._study_comm_metrics(per_group_comm)
        metrics.update(comm_stats)
        self.writer.write(
            frames=self.experiment.total_frames,
            iteration=self.experiment.n_iters_performed,
            phase="collection",
            metrics=metrics,
        )
        self._write_group_returns("collection", per_group)
        for index, value in enumerate(collection_episode_returns(batch, self._return_groups)):
            self.writer.write(
                frames=self.experiment.total_frames,
                iteration=self.experiment.n_iters_performed,
                phase="collection",
                metrics={"return_episode": value},
                sample=index,
            )
        if self._mapdn_domain:
            self.writer.write(
                frames=self.experiment.total_frames,
                iteration=self.experiment.n_iters_performed,
                phase="collection_domain",
                group="agents",
                metrics=self.experiment.task.log_info(batch),
            )
        if self._pcp_domain:
            from commstudy.tasks.pcp_statistics import domain_transition_series

            domain = domain_transition_series(batch)
            self.writer.write(
                frames=self.experiment.total_frames,
                iteration=self.experiment.n_iters_performed,
                phase="collection_domain",
                group="adversary",
                metrics={
                    **{name + "_mean": float(value.mean().cpu()) for name, value in domain.items()},
                    "transition_count": float(batch.numel()),
                    "reward_accounting_max_abs_error": float(
                        (domain["predator_reward"] - domain["expected_predator_reward"])
                        .abs()
                        .max()
                        .cpu()
                    ),
                },
            )
        self._forward_to_benchmarl(comm_stats, "collection/communication")

    def on_train_step(self, batch, group: str) -> TensorDict | None:
        comm_stats = self._drain_comm_stats(group)
        # These are the sampled optimization transitions, including repeated
        # replay passes. They are not extra deployed environment interactions.
        comm_stats["optimization_transition_count"] = float(batch.numel())
        for key in ("value_target", "advantage", "state_value"):
            value = batch.get((group, key), None)
            if isinstance(value, torch.Tensor):
                comm_stats.update(
                    {
                        name: scalar
                        for name, scalar in tensor_health_metrics(value, key).items()
                        if not name.endswith(("_q01", "_q50", "_q99"))
                    }
                )
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
        # Framework losses/KL/ESS retain their upstream minibatch summaries.
        # Our transition denominators count every replayed transition, while
        # moments are pooled using their actual scalar/transition counts.
        for name in list(metrics):
            value = training_td.get(name, None)
            if isinstance(value, torch.Tensor) and name.endswith("_count"):
                metrics[name] = float(value.sum().cpu())
            elif isinstance(value, torch.Tensor) and name.startswith("comm_"):
                if name.removeprefix("comm_") in CommModule._MAX_REDUCED_STATS:
                    metrics[name] = float(value.max().cpu())
                    continue
                count_key = (
                    f"{name}_count"
                    if name.endswith("norm_ratio")
                    else "comm_gate_saturated_low_fraction_count"
                    if "gate_saturated_" in name
                    else "comm_transition_count"
                )
                weights = training_td.get(count_key, None)
                if isinstance(weights, torch.Tensor) and bool(weights.sum() > 0):
                    metrics[name] = float((value * weights).sum().cpu() / weights.sum().cpu())
        for prefix in ("value_target", "advantage", "state_value"):
            weights = training_td.get(f"{prefix}_scalar_count", None)
            if weights is None:
                continue
            mean = training_td.get(f"{prefix}_mean")
            std = training_td.get(f"{prefix}_std")
            pooled_mean = (mean * weights).sum() / weights.sum()
            variance = ((std.square() + mean.square()) * weights).sum() / weights.sum()
            metrics[f"{prefix}_mean"] = float(pooled_mean.cpu())
            metrics[f"{prefix}_std"] = float(
                (variance - pooled_mean.square()).clamp_min(0).sqrt().cpu()
            )
            metrics[f"{prefix}_min"] = float(training_td.get(f"{prefix}_min").min().cpu())
            metrics[f"{prefix}_max"] = float(training_td.get(f"{prefix}_max").max().cpu())
        metrics["wall_time_seconds"] = self._elapsed()
        self.writer.write(
            frames=self.experiment.total_frames,
            iteration=self.experiment.n_iters_performed,
            phase="training",
            group=group,
            metrics=metrics,
        )

    def _study_comm_metrics(self, per_group: Mapping[str, dict[str, float]]) -> dict[str, float]:
        measured = [per_group[g] for g in self._return_groups if g in per_group]
        # Preserve existing single measured-group analysis keys without mixing
        # dimensions, parameter counts, or costs from unrelated groups.
        return measured[0] if len(measured) == 1 else {}

    def _write_group_health(self, phase: str, batch) -> dict[str, dict[str, float]]:
        per_group_comm = {}
        for group in self.experiment.group_map:
            metrics = group_health_metrics(batch, group)
            comm = self._drain_comm_stats(group)
            per_group_comm[group] = comm
            metrics.update(comm)
            if phase == "collection":
                for callback in self.experiment.callbacks:
                    if isinstance(callback, TrainingHealthCallback):
                        metrics.update(callback.replay_metrics.get(group, {}))
            self.writer.write(
                frames=self.experiment.total_frames,
                iteration=self.experiment.n_iters_performed,
                phase=phase,
                group=group,
                metrics=metrics,
            )
        return per_group_comm

    def _measured(self, per_group: Mapping[str, float]) -> dict[str, float]:
        """Keep only the groups whose reward the study reports."""

        return {group: value for group, value in per_group.items() if group in self._return_groups}

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
            group_returns = group_rollout_returns(rollout, tuple(self.experiment.group_map))
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
                    "episode_length_mean": sum(rollout.batch_size[0] for rollout in rollouts)
                    / len(rollouts),
                }
            )
        metrics["wall_time_seconds"] = self._elapsed()
        # BenchMARL supplies one complete TensorDict per episode, possibly of
        # different lengths. Flatten and concatenate for transition-weighted
        # moments/quantiles rather than averaging episode summaries.
        flat_rollouts = [rollout.reshape(-1) for rollout in rollouts]
        per_group_comm = (
            self._write_group_health("evaluation", torch.cat(flat_rollouts, dim=0))
            if flat_rollouts
            else {}
        )
        comm_stats = self._study_comm_metrics(per_group_comm)
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
        if self._mapdn_domain:
            for index, rollout in enumerate(rollouts):
                self.writer.write(
                    frames=self.experiment.total_frames,
                    iteration=self.experiment.n_iters_performed,
                    phase="evaluation_domain_episode",
                    group="agents",
                    sample=index,
                    metrics=self.experiment.task.log_info(rollout),
                )
            if flat_rollouts:
                self.writer.write(
                    frames=self.experiment.total_frames,
                    iteration=self.experiment.n_iters_performed,
                    phase="evaluation_domain",
                    group="agents",
                    metrics=self.experiment.task.log_info(torch.cat(flat_rollouts, dim=0)),
                )
        if self._pcp_domain:
            from commstudy.tasks.pcp_statistics import (
                domain_episode_metrics,
                summarize_domain_episodes,
            )

            domain_episodes = []
            for index, rollout in enumerate(rollouts):
                episode_metrics = domain_episode_metrics(rollout)
                domain_episodes.append({"metrics": episode_metrics})
                self.writer.write(
                    frames=self.experiment.total_frames,
                    iteration=self.experiment.n_iters_performed,
                    phase="evaluation_domain_episode",
                    group="adversary",
                    sample=index,
                    metrics=episode_metrics,
                )
            summary = summarize_domain_episodes(domain_episodes)
            self.writer.write(
                frames=self.experiment.total_frames,
                iteration=self.experiment.n_iters_performed,
                phase="evaluation_domain",
                group="adversary",
                metrics={
                    **summary["means"],
                    **{key: value for key, value in summary.items() if key != "means"},
                },
            )
        self._forward_to_benchmarl(comm_stats, "eval/communication")
