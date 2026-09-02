"""Shared interfaces for swappable inter-agent communication modules.

The communication slot receives local agent embeddings shaped ``[..., N, D]``
and must return the same shape. A mask follows the receiver/sender convention
``mask[..., i, j] == True`` when receiver ``i`` may consume sender ``j``'s
message.

Communication is part of the decentralized actor only. It must never be used
to construct the centralized critic representation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import torch
from torch import nn


@dataclass
class CommContext:
    """Optional runtime information supplied to a communication module.

    ``extras`` deliberately remains task-agnostic. In particular,
    ``extras["sender_mask"]`` may carry an explicitly realized sender
    availability mask shaped ``[..., N]``. Replaying that realization avoids
    silently resampling stochastic channel failures between rollout collection
    and an optimizer forward pass.
    """

    mask: torch.Tensor | None = None
    class_id: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class CommModule(nn.Module, ABC):
    """Base interface and lightweight scalar-stat accumulator.

    Subclasses record only detached scalar summaries through
    :meth:`_record_stats`; retaining full message or attention tensors here
    would unnecessarily keep training data alive. Recorded values are averaged
    over forward calls since :meth:`reset_stats`, except ``max_message_norm``,
    which is reduced as a true window maximum.
    """

    #: Stats reduced as a window maximum rather than averaged over calls.
    _MAX_REDUCED_STATS = frozenset(
        {"max_message_norm", "grad_clip_observed_max_norm"}
    )

    #: Initial gain on a normalized communication path. Matches the xavier gain
    #: the modules already use on their output projection, so enabling
    #: normalization does not change how loud communication is at step zero.
    _COMM_PATH_SCALE_INIT = 0.01

    def __init__(
        self,
        hidden_dim: int,
        grad_clip: float | None = None,
        normalize_comm_path: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown communication options: {unknown}")

        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.hidden_dim = int(hidden_dim)
        self.message_dim = 0
        self.communication_rounds = 0
        self.grad_clip = self._validate_grad_clip(grad_clip)
        if not isinstance(normalize_comm_path, bool):
            raise TypeError(
                "normalize_comm_path must be a bool, got "
                f"{type(normalize_comm_path).__name__}."
            )
        self.normalize_comm_path = normalize_comm_path
        if normalize_comm_path:
            # Normalizing to unit RMS on its own would destroy the deliberate
            # near-identity initialization: the modules init output_projection
            # at xavier gain 0.01 so communication starts as almost a no-op, and
            # forcing unit RMS made the first evaluation return -1361 against a
            # -552 baseline. A LayerScale-style scalar restores that init while
            # keeping normalization's scale invariance -- influence can now only
            # grow linearly in one monitorable number, never multiplicatively in
            # the weights.
            self.comm_path_scale = nn.Parameter(
                torch.full((), self._COMM_PATH_SCALE_INIT)
            )

        # Keep diagnostics outside TorchRL's functionalized parameter/buffer
        # state. Only detached CPU scalars are stored here.
        self._stat_sums: dict[str, torch.Tensor] = {}
        self._stat_counts: dict[str, int] = {}

    @staticmethod
    def _validate_grad_clip(grad_clip: float | None) -> float | None:
        if grad_clip is None:
            return None
        if isinstance(grad_clip, bool) or not isinstance(grad_clip, Real):
            raise TypeError(
                "grad_clip must be a positive number or None, got "
                f"{type(grad_clip).__name__}."
            )
        value = float(grad_clip)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"grad_clip must be positive and finite, got {value}."
            )
        return value

    def _stabilize_comm_path(self, contribution: torch.Tensor) -> torch.Tensor:
        """Bound the message-derived term before it enters the residual stream.

        Two independent, independently-enabled guards, in the order they act:

        ``normalize_comm_path`` rescales the contribution to unit RMS per agent
        and then applies one learned scalar gain. This is the guard that
        addresses the observed divergence. Rounds share parameters and compose
        residually, so the forward pass amplifies multiplicatively with depth:
        measured on the diverging configuration, a 100x weight scale produces
        1.3e3 / 1.3e5 / 1.4e7 activation norms at 1, 2 and 3 rounds. Normalizing
        makes the output invariant to weight scale entirely.

        The LayerNorm itself is elementwise-affine-free; the only added
        parameter is the single ``comm_path_scale``. That is what preserves the
        modules' near-identity initialization, and it means communication's
        influence can grow only linearly in one scalar you can log, never
        multiplicatively in the weight matrices.

        ``grad_clip`` bounds the gradient flowing back into the message
        encoders, attention scores and decoders. Note this is a spike limiter
        and not a fix for the amplification above: BenchMARL trains with Adam,
        whose per-parameter second-moment normalization makes updates largely
        invariant to gradient scale, so clipping every step is close to a no-op.
        Set it above the typical norm so it only catches outliers, and read
        ``grad_clip_fraction`` to confirm it is not firing constantly.

        Clipping alone leaves the forward pass bit-identical; normalization does
        not, and changes the architecture. Both default to off so the frozen V2
        protocol reproduces exactly.
        """

        if self.normalize_comm_path:
            contribution = self.comm_path_scale * nn.functional.layer_norm(
                contribution, (contribution.shape[-1],)
            )
            self._record_stats(comm_path_scale=self.comm_path_scale)
        return self._clip_comm_path(contribution)

    def _clip_comm_path(self, contribution: torch.Tensor) -> torch.Tensor:
        """Bound the gradient flowing back through the communication path.

        ``contribution`` is the message-derived term a module adds to its
        input, i.e. the ``delta`` in ``output = h + delta``. Clipping here and
        not on the module output is deliberate: the residual skip carries the
        encoder's gradient, which is healthy and must keep training at full
        scale. Only the path through the message encoders, attention scores and
        decoders is throttled.

        The clip unit is one environment transition (the trailing ``[N, D]``
        block), not the whole tensor. A tensor-global norm would make the same
        threshold mean different things during rollout collection and during a
        PPO minibatch update, because those call the module with different batch
        shapes.

        Registering a backward hook leaves the forward pass bit-identical, so
        enabling this cannot perturb stored-vs-recomputed log-probabilities
        during PPO replay, and cannot change any evaluation or saliency rollout.
        """

        if self.grad_clip is None or not contribution.requires_grad:
            return contribution
        if contribution.numel() == 0 or contribution.dim() < 2:
            return contribution

        max_norm = self.grad_clip

        def clip(grad: torch.Tensor) -> torch.Tensor:
            per_sample = grad.reshape(-1, grad.shape[-2] * grad.shape[-1])
            norms = per_sample.norm(dim=-1)
            # Report before clipping so the recorded maximum shows how hard the
            # clip is working, and stays meaningful once the clip is active.
            self._record_stats(
                grad_clip_observed_max_norm=norms.amax(),
                grad_clip_fraction=(norms > max_norm).to(torch.float32).mean(),
            )
            scale = (max_norm / norms.clamp_min(torch.finfo(norms.dtype).tiny)).clamp(
                max=1.0
            )
            return (per_sample * scale.unsqueeze(-1)).reshape(grad.shape)

        contribution.register_hook(clip)
        return contribution

    @abstractmethod
    def forward(
        self,
        h: torch.Tensor,
        context: CommContext | None = None,
    ) -> torch.Tensor:
        """Transform ``h`` while preserving its ``[..., N, D]`` shape."""

        raise NotImplementedError

    def message_bits(self) -> int:
        """Nominal float32 payload bits sent by one sender per environment step."""

        return self.message_dim * 32 * self.communication_rounds

    def _record_stats(
        self,
        **values: torch.Tensor | Real,
    ) -> None:
        """Accumulate detached scalar summaries for later logging."""

        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                scalar = value.detach()
                if scalar.numel() != 1:
                    scalar = scalar.to(dtype=torch.float32).mean()
                scalar = scalar.to(device="cpu", dtype=torch.float64).reshape(())
            elif isinstance(value, Real):
                scalar = torch.tensor(float(value), dtype=torch.float64)
            else:
                raise TypeError(
                    f"Communication stat '{name}' must be a scalar or Tensor, "
                    f"got {type(value).__name__}."
                )

            if name in self._stat_sums:
                if name in self._MAX_REDUCED_STATS:
                    self._stat_sums[name] = torch.maximum(
                        self._stat_sums[name], scalar
                    )
                else:
                    self._stat_sums[name] = self._stat_sums[name] + scalar
                self._stat_counts[name] += 1
            else:
                self._stat_sums[name] = scalar.clone()
                self._stat_counts[name] = 1

    def communication_stats(self) -> dict[str, float]:
        """Return small, non-differentiable averages since the last reset."""

        stats = {
            "message_dim": float(self.message_dim),
            "message_bits_per_sender": float(self.message_bits()),
            "communication_rounds": float(self.communication_rounds),
        }
        stats.update(
            {
                name: float(
                    total.item()
                    if name in self._MAX_REDUCED_STATS
                    else total.item() / self._stat_counts[name]
                )
                for name, total in self._stat_sums.items()
            }
        )
        return stats

    def reset_stats(self) -> None:
        """Discard all accumulated dynamic summaries."""

        self._stat_sums.clear()
        self._stat_counts.clear()
