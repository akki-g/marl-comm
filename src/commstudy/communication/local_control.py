"""Active local capacity control for one-round learned Broadcast.

The two bias-free projections have exactly the same shapes and activation as
Broadcast's message encoder/decoder. Every agent applies both projections to
its own embedding, with no aggregation, channel, or other-agent input. The
``message_dim`` constructor option denotes the matching bottleneck width; it
does not represent a transmitted message in this control.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import CommContext, CommModule
from .utils import validate_comm_input


class LocalCapacityComm(CommModule):
    """Compute ``h_i + decoder(tanh(encoder(h_i)))`` independently per agent.

    At hidden width 128 and bottleneck width 32, both matrices contain 8,192
    active parameters in total, exactly matching unnormalized Broadcast. The
    optional path guards match Broadcast's guards; they do not transmit data.
    Masks, class IDs, sender suppression and message interventions have no
    effect on this local computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        message_dim: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, **kwargs)
        if type(message_dim) is not int or message_dim < 1:
            raise ValueError("message_dim must be an integer >= 1")
        self.local_bottleneck_dim = message_dim
        # Keep projection names/shapes aligned for explicit capacity and
        # initialization comparisons. No communication channel is constructed.
        self.message_encoder = nn.Linear(hidden_dim, message_dim, bias=False)
        self.message_decoder = nn.Linear(message_dim, hidden_dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        context: CommContext | None = None,
    ) -> torch.Tensor:
        validate_comm_input(h, self.hidden_dim, self.__class__.__name__)
        local = torch.tanh(self.message_encoder(h))
        delta = self._stabilize_comm_path(self.message_decoder(local))
        return h + delta

    def communication_stats(self) -> dict[str, float]:
        """Zero network traffic, with local bottleneck capacity named explicitly."""
        stats = super().communication_stats()
        stats.update(
            local_bottleneck_dim=float(self.local_bottleneck_dim),
            active_sender_fraction=0.0,
            active_edge_fraction=0.0,
            mean_message_norm=0.0,
            max_message_norm=0.0,
            requested_dropout_rate=0.0,
            realized_communication_rate=0.0,
            messages_per_step=0.0,
            active_edges_per_step=0.0,
            nominal_messages_per_step=0.0,
            potential_edges_per_step=0.0,
            realized_sender_scalars_per_step=0.0,
            realized_sender_bits_per_step=0.0,
            nominal_sender_scalars_per_step=0.0,
            nominal_sender_bits_per_step=0.0,
            realized_scalar_transmissions_per_step=0.0,
            realized_bits_per_step=0.0,
            nominal_scalar_transmissions_per_step=0.0,
            nominal_bits_per_step=0.0,
        )
        return stats
