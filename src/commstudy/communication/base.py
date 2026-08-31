"""Shared interfaces for swappable inter-agent communication modules.

The communication slot receives local agent embeddings shaped ``[..., N, D]``
and must return the same shape. A mask follows the receiver/sender convention
``mask[..., i, j] == True`` when receiver ``i`` may consume sender ``j``'s
message.

Communication is part of the decentralized actor only. It must never be used
to construct the centralized critic representation.
"""

from __future__ import annotations

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

    def __init__(
        self,
        hidden_dim: int,
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

        # Keep diagnostics outside TorchRL's functionalized parameter/buffer
        # state. Only detached CPU scalars are stored here.
        self._stat_sums: dict[str, torch.Tensor] = {}
        self._stat_counts: dict[str, int] = {}

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
                if name == "max_message_norm":
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
                    if name == "max_message_norm"
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
