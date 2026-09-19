"""Explicit zero-or-all sender budgets for the minimal Broadcast contrast.

Only K=0 and K=N are supported. There is no learned or randomized scheduler,
sender-order preference, scheduling preview, quantization, or extra payload.
The population-sized budget uses the existing Broadcast implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .base import CommContext
from .broadcast import BroadcastComm
from .channel import CommChannel, _resolve_sender_mask
from .utils import communication_costs_for_round, resolve_comm_mask, validate_comm_input


class BudgetedBroadcastComm(BroadcastComm):
    """Broadcast with a declared, dominant static zero-or-all sender budget.

    A zero-budget actor retains Broadcast-shaped parameters for a mechanistic
    zero-message control, but those parameters are dormant. It must not be
    called an active capacity control; ``LocalCapacityComm`` supplies that
    separate comparison. Realized costs count emitted packets and directed
    deliveries independently, with float32 payload semantics inherited from
    Broadcast. At N=3 and M=32, K=3 emits 96 scalars/3072 bits and delivers
    192 scalars/6144 bits per transition on a full graph excluding self.
    """

    def __init__(
        self,
        hidden_dim: int,
        sender_budget: int,
        message_dim: int = 32,
        exclude_self: bool = True,
        residual: bool = True,
        channel: CommChannel | Mapping[str, Any] | Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if type(sender_budget) is not int or sender_budget < 0:
            raise ValueError("sender_budget must be a non-negative integer")
        super().__init__(
            hidden_dim=hidden_dim,
            message_dim=message_dim,
            exclude_self=exclude_self,
            residual=residual,
            channel=channel,
            **kwargs,
        )
        self.sender_budget = sender_budget

    def forward(
        self,
        h: torch.Tensor,
        context: CommContext | None = None,
    ) -> torch.Tensor:
        validate_comm_input(h, self.hidden_dim, self.__class__.__name__)
        if self.sender_budget not in (0, h.shape[-2]):
            raise ValueError(
                "BudgetedBroadcastComm supports only sender_budget=0 or the "
                f"full sender population ({h.shape[-2]}), got {self.sender_budget}."
            )
        self._record_stats(configured_sender_budget=float(self.sender_budget))
        if self.sender_budget:
            return super().forward(h, context)

        context = context if context is not None else CommContext()
        # Still validate context contracts, even when no payload is permitted.
        base_edges = resolve_comm_mask(h, context.mask, exclude_self=self.exclude_self)
        messages = h.new_zeros((*h.shape[:-1], self.message_dim))
        _resolve_sender_mask(messages, context.extras.get("sender_mask"))
        unavailable = torch.zeros(h.shape[:-1], dtype=torch.bool, device=h.device)
        # Publish the actual all-false replay mask without invoking any channel
        # transform or consuming private/global RNG. _finish only masks the
        # zero payload and records availability; an intervention cannot reopen it.
        self.channel._finish(messages, unavailable)
        no_edges = torch.zeros_like(base_edges)
        costs = communication_costs_for_round(no_edges, no_edges, packet_dim=self.message_dim)
        self._record_stats(
            active_sender_fraction=0.0,
            active_edge_fraction=0.0,
            mean_message_norm=0.0,
            max_message_norm=0.0,
            requested_dropout_rate=self.channel.requested_dropout_rate,
            realized_communication_rate=0.0,
            **costs,
        )
        # Exact identity, including signed zero; dormant projections are not
        # evaluated. No broadcast bias or normalization can create a signal.
        return h if self.residual else torch.zeros_like(h)
