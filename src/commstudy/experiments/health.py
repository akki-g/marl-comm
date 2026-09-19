"""Finite-value and PPO replay guards using supported PyTorch/BenchMARL hooks.

No optimizer, algorithm loss, or training loop is reimplemented here. A failed
check raises before the affected optimizer step; managed runners retain their
normal failed status, traceback, and artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import warnings
from contextlib import contextmanager

import torch
from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase

from commstudy.communication.base import CommModule
from commstudy.communication.channel import CommChannel, channel_phase, preserve_channel_rng
from commstudy.utils.rng import preserve_rng_state


class TrainingHealthError(FloatingPointError):
    """An optimizer update was rejected because its scientific inputs are invalid."""

    def __init__(self, message, *, details=None):
        self.details = details or {}
        if self.details:
            message += " Details: " + json.dumps(self.details, sort_keys=True)
        super().__init__(message)


def require_finite(value, label: str) -> None:
    """Inspect tensors recursively without mutating them or their gradients."""
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise TrainingHealthError(f"Non-finite {label}; optimizer step rejected.")
    elif isinstance(value, (Mapping, TensorDictBase)):
        for key, leaf in value.items():
            require_finite(leaf, f"{label}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, leaf in enumerate(value):
            require_finite(leaf, f"{label}/{index}")


def guard_optimizer_step(optimizer, args, kwargs, *, label: str = "optimizer") -> None:
    """Public optimizer pre-hook: reject parameter, gradient, or Adam-state poison."""
    del args, kwargs
    for group_index, group in enumerate(optimizer.param_groups):
        for index, parameter in enumerate(group["params"]):
            prefix = f"{label}/parameter[{group_index},{index}]"
            require_finite(parameter, prefix)
            require_finite(parameter.grad, f"{prefix}/gradient")
            require_finite(optimizer.state.get(parameter, {}), f"{prefix}/optimizer_state")


@contextmanager
def _preserve_communication_stats(policy):
    modules = [module for module in policy.modules() if isinstance(module, CommModule)]
    snapshots = [
        (
            dict(module._stat_sums),
            dict(module._stat_counts),
            getattr(module, "_health_replay_check", False),
        )
        for module in modules
    ]
    try:
        for module in modules:
            module._health_replay_check = True
        yield
    finally:
        for module, (sums, counts, checking) in zip(modules, snapshots, strict=True):
            module._stat_sums = sums
            module._stat_counts = counts
            module._health_replay_check = checking


def _replay_comparison(actual, expected):
    error = (actual - expected).abs()
    ratio = error / (1e-5 + 1e-5 * expected.abs())
    violations = ratio > 1
    indices = torch.nonzero(violations.reshape(-1), as_tuple=False).flatten()[:5]
    return {
        "maximum_absolute_error": float(error.max().cpu()),
        "maximum_tolerance_ratio": float(ratio.max().cpu()),
        "violation_count": int(violations.sum().cpu()),
        "examples": [
            {
                "flat_index": int(index),
                "stored_log_prob": float(expected.reshape(-1)[index]),
                "replayed_log_prob": float(actual.reshape(-1)[index]),
                "tolerance_ratio": float(ratio.reshape(-1)[index]),
            }
            for index in indices
        ],
    }


def _policy_replay(policy, batch, group):
    replay = batch.clone()
    distribution = policy.get_dist(replay)
    actual = distribution.log_prob(batch.get((group, "action"))).detach()
    expected = batch.get((group, "log_prob")).detach()
    require_finite(actual, f"{group}/replayed_log_prob")
    if actual.numel() != expected.numel():
        raise TrainingHealthError(f"{group}: behavior/replay log-probability shapes disagree.")
    return actual.reshape(expected.shape), replay


def _captured_outputs(replay, actual, group):
    values = {
        name: replay.get((group, name)).detach().cpu()
        for name in ("loc", "scale")
        if replay.get((group, name), None) is not None
    }
    values["log_prob"] = actual.detach().cpu()
    return values


def check_behavior_replay(
    policy, batch, group: str, *, diagnostics=None, capture=None
) -> dict[str, float]:
    """Verify stored MAPPO probabilities before any update with unchanged tolerance.

    Full-batch actor GEMMs can round differently from collection-width GEMMs.
    If the fast comparison fails, an explicitly named time axis permits exact
    collection-shape verification. This fallback never relaxes rtol or atol.
    Both paths restore RNG and communication statistics and replay stored masks.
    ``diagnostics`` receives JSON-safe checks; optional ``capture`` receives
    detached evidence only when the full-batch comparison fails.
    """
    stored = batch.get((group, "log_prob"), None)
    if stored is None or not hasattr(policy, "get_dist"):
        return {}
    expected = stored.detach()
    require_finite(expected, f"{group}/behavior_log_prob")
    details = {
        "group": group,
        "batch_shape": list(batch.batch_size),
        "batch_names": list(batch.names),
        "rtol": 1e-5,
        "atol": 1e-5,
    }
    if diagnostics is not None:
        diagnostics.update(details)
        details = diagnostics
    channels = [module for module in policy.modules() if isinstance(module, CommChannel)]
    fallback_slices = 0
    with (
        preserve_rng_state(),
        preserve_channel_rng(channels),
        _preserve_communication_stats(policy),
        channel_phase("optimization"),
        torch.enable_grad(),
    ):
        actual, replay = _policy_replay(policy, batch, group)
        full = _replay_comparison(actual, expected)
        details["batched"] = full
        if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
            if capture is not None:
                capture["batched"] = _captured_outputs(replay, actual, group)
            time_axes = [index for index, name in enumerate(batch.names) if name == "time"]
            if len(time_axes) != 1 or not batch.batch_size[time_axes[0]]:
                details["fallback_unavailable"] = "requires exactly one nonempty named time axis"
                raise TrainingHealthError(
                    f"{group}: behavior log-probability replay mismatch before optimization; "
                    "collection-shape verification unavailable; optimizer step rejected.",
                    details=details,
                )
            time_axis = time_axes[0]
            details["time_axis"] = time_axis
            details["collection_shape"] = [
                size for index, size in enumerate(batch.batch_size) if index != time_axis
            ]
            verified, outputs = [], []
            for step_batch in batch.unbind(time_axis):
                step_actual, step_replay = _policy_replay(policy, step_batch, group)
                verified.append(step_actual)
                if capture is not None:
                    outputs.append(_captured_outputs(step_replay, step_actual, group))
            actual = torch.stack(verified, dim=time_axis)
            fallback_slices = len(verified)
            accepted = _replay_comparison(actual, expected)
            details["collection_shaped"] = accepted
            if capture is not None:
                capture["collection_shaped"] = {
                    name: torch.stack([output[name] for output in outputs], dim=time_axis)
                    for name in outputs[0]
                }
            if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
                raise TrainingHealthError(
                    f"{group}: behavior log-probability replay mismatch before optimization "
                    "persists at collection shape; optimizer step rejected.",
                    details=details,
                )
        else:
            accepted = full
    return {
        "replay_log_prob_max_abs_error": accepted["maximum_absolute_error"],
        "replay_log_prob_scalar_count": float(expected.numel()),
        "replay_transition_count": float(batch.numel()),
        "replay_batched_log_prob_max_abs_error": full["maximum_absolute_error"],
        "replay_batched_log_prob_max_tolerance_ratio": full["maximum_tolerance_ratio"],
        "replay_accepted_log_prob_max_tolerance_ratio": accepted["maximum_tolerance_ratio"],
        "replay_collection_shape_fallback_count": float(fallback_slices > 0),
        "replay_collection_shape_fallback_slices": float(fallback_slices),
    }


class TrainingHealthCallback(Callback):
    """Check all trained groups without mixing their health or changing PPO."""

    def __init__(self) -> None:
        super().__init__()
        self._handles = []
        self.replay_metrics: dict[str, dict[str, float]] = {}
        self._fallback_snapshots = set()

    def on_setup(self) -> None:
        for group in self.experiment.train_group_map:
            loss = self.experiment.losses[group]

            def loss_guard(module, inputs, output, *, group=group):
                del module, inputs
                for key, value in output.items():
                    if str(key).startswith("loss_"):
                        require_finite(value, f"{group}/{key}")

            self._handles.append(loss.register_forward_hook(loss_guard))
            parameters = {}
            for name, optimizer in self.experiment.optimizers[group].items():

                def optimizer_guard(optimizer, args, kwargs, *, label=f"{group}/{name}"):
                    guard_optimizer_step(optimizer, args, kwargs, label=label)

                self._handles.append(optimizer.register_step_pre_hook(optimizer_guard))
                for entry in optimizer.param_groups:
                    for parameter in entry["params"]:
                        parameters.setdefault(id(parameter), parameter)
            for index, parameter in enumerate(parameters.values()):
                require_finite(parameter, f"{group}/initial_parameter[{index}]")

                def gradient_guard(gradient, *, label=f"{group}/gradient[{index}]"):
                    require_finite(gradient, label)
                    return gradient

                self._handles.append(parameter.register_hook(gradient_guard))

    def _save_replay_diagnostic(self, batch, group, details, capture, reason):
        folder = getattr(self.experiment, "folder_name", None)
        if folder is None:
            return False
        temporary = None
        try:
            from commstudy.experiments.provenance import source_fingerprint

            directory = Path(folder) / "replay_diagnostics"
            directory.mkdir(parents=True, exist_ok=True)
            frames = int(getattr(self.experiment, "total_frames", 0))
            safe_group = "".join(c if c.isalnum() or c in "_-" else "_" for c in group)
            stem = f"frame{frames:09d}_{safe_group}_{reason}"
            path = directory / f"{stem}.pt"
            suffix = 0
            while path.exists():
                suffix += 1
                path = directory / f"{stem}_{suffix}.pt"
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(
                {
                    "schema_version": 1,
                    "diagnostic_only": True,
                    "not_resume": True,
                    "reason": reason,
                    "frames": frames,
                    "group": group,
                    "details": details,
                    "batch": batch.detach().cpu(),
                    "replay_tensors": capture,
                    "group_policies": {
                        name: policy.state_dict()
                        for name, policy in self.experiment.group_policies.items()
                    },
                    "source_sha256": source_fingerprint(Path(__file__).resolve().parents[3]),
                },
                temporary,
            )
            os.replace(temporary, path)
            return True
        except Exception as error:
            # Never replace a scientific replay failure with a diagnostics I/O error.
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                warnings.warn(
                    f"Could not save replay diagnostic: {error}", RuntimeWarning, stacklevel=2
                )
            except Exception:
                pass  # A warning-as-error policy must not replace the replay outcome.
            return False

    def on_batch_collected(self, batch) -> None:
        # The public lifecycle hook executes before any group is optimized.
        self.replay_metrics.clear()
        self.experiment._checking_policy_replay = True
        try:
            for group in self.experiment.train_group_map:
                require_finite(batch.get(group), f"{group}/collected_policy_data")
                policy = self.experiment.group_policies[group]
                details, capture = {}, {}
                try:
                    self.replay_metrics[group] = check_behavior_replay(
                        policy,
                        batch,
                        group,
                        diagnostics=details,
                        capture=capture,
                    )
                except TrainingHealthError:
                    self._save_replay_diagnostic(batch, group, details, capture, "failure")
                    raise
                if (
                    self.replay_metrics[group].get("replay_collection_shape_fallback_count")
                    and group not in self._fallback_snapshots
                ):
                    if self._save_replay_diagnostic(
                        batch, group, details, capture, "first_fallback"
                    ):
                        self._fallback_snapshots.add(group)
        finally:
            self.experiment._checking_policy_replay = False
