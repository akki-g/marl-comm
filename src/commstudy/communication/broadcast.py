"""CommNet-inspired learned continuous broadcast communication.

This is a project adaptation rather than a reproduction of the complete
CommNet architecture. Each agent learns a compact message, receivers average
messages from permitted senders, and a decoded communication delta is fused
with the receiver's local representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .base import CommContext, CommModule
from .channel import CommChannel, build_channel
from .utils import (
    active_edge_fraction,
    active_sender_fraction,
    communication_costs_for_round,
    masked_sender_mean,
    resolve_comm_mask,
    sender_mask_to_edge_mask,
    validate_comm_input,
)


class BroadcastComm(CommModule):
    """Learn, broadcast, and mean-aggregate compact continuous messages."""

    def __init__(
        self,
        hidden_dim: int,
        message_dim: int = 32,
        exclude_self: bool = True,
        residual: bool = True,
        channel: CommChannel | Mapping[str, Any] | Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, **kwargs)
        if message_dim < 1:
            raise ValueError("message_dim must be >= 1")

        self.message_dim = int(message_dim)
        self.communication_rounds = 1
        self.exclude_self = bool(exclude_self)
        self.residual = bool(residual)

        # Bias-free decoding guarantees a zero communication delta when a
        # receiver has no live incoming edge.
        self.message_encoder = nn.Linear(hidden_dim, message_dim, bias=False)
        self.message_decoder = nn.Linear(message_dim, hidden_dim, bias=False)
        self.channel = build_channel(channel)

    def forward(
        self,
        h: torch.Tensor,
        context: CommContext | None = None,
    ) -> torch.Tensor:
        validate_comm_input(h, self.hidden_dim, self.__class__.__name__)

        context = context if context is not None else CommContext()
        base_edges = resolve_comm_mask(
            h,
            context.mask,
            exclude_self=self.exclude_self,
        )

        messages = torch.tanh(self.message_encoder(h))
        explicit_sender_mask = context.extras.get("sender_mask")
        channel_output = self.channel(
            messages,
            sender_mask=explicit_sender_mask,
        )
        effective_edges = sender_mask_to_edge_mask(
            channel_output.sender_mask,
            base_edges,
        )

        aggregated = masked_sender_mean(channel_output.messages, effective_edges)
        delta = self._stabilize_comm_path(self.message_decoder(aggregated))
        output = h + delta if self.residual else delta

        message_norm = torch.linalg.vector_norm(channel_output.messages, dim=-1)
        sender_fraction = active_sender_fraction(effective_edges)
        channel_retention = channel_output.sender_mask.to(torch.float32).mean()
        costs = communication_costs_for_round(
            effective_edges,
            base_edges,
            packet_dim=self.message_dim,
            active_senders=channel_output.sender_mask,
        )
        self._record_stats(
            active_sender_fraction=sender_fraction,
            active_edge_fraction=active_edge_fraction(
                effective_edges,
                exclude_self=self.exclude_self,
            ),
            mean_message_norm=message_norm.mean(),
            max_message_norm=message_norm.max(),
            requested_dropout_rate=self.channel.requested_dropout_rate,
            # Channel retention is deliberately independent of graph topology:
            # p=0 retains all senders and p=1 retains none. Edge/sender activity
            # above separately captures topology and scheduling effects.
            realized_communication_rate=channel_retention,
            **costs,
        )

        return output
