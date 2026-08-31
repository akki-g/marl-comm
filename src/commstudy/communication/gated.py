"""IC3Net-inspired differentiable sender-gated communication.

The full IC3Net algorithm includes recurrent policies and reward-design choices
that are intentionally outside this project. This module adopts its learned
"when to communicate" idea inside the common MAPPO/QMIX actor interface.
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


class GatedComm(CommModule):
    """Learn one sigmoid transmission gate per sender and communication round."""

    def __init__(
        self,
        hidden_dim: int,
        message_dim: int = 32,
        exclude_self: bool = True,
        residual: bool = True,
        hard: bool = False,
        gate_threshold: float = 0.5,
        sender_budget: int | None = None,
        sender_selection: str = "learned",
        sender_selection_seed: int = 0,
        channel: CommChannel | Mapping[str, Any] | Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, **kwargs)
        if message_dim < 1:
            raise ValueError("message_dim must be >= 1")
        if not 0.0 <= gate_threshold <= 1.0:
            raise ValueError("gate_threshold must be in [0, 1]")
        if sender_budget is not None and sender_budget < 0:
            raise ValueError("sender_budget must be >= 0 or None")
        if sender_selection not in {"learned", "random"}:
            raise ValueError("sender_selection must be 'learned' or 'random'")

        self.message_dim = int(message_dim)
        self.communication_rounds = 1
        self.exclude_self = bool(exclude_self)
        self.residual = bool(residual)
        self.hard = bool(hard)
        self.gate_threshold = float(gate_threshold)
        self.sender_budget = None if sender_budget is None else int(sender_budget)
        self.sender_selection = sender_selection
        self.sender_selection_seed = int(sender_selection_seed)

        self.message_encoder = nn.Linear(hidden_dim, message_dim, bias=False)
        self.gate_network = nn.Linear(hidden_dim, 1)
        self.message_decoder = nn.Linear(message_dim, hidden_dim, bias=False)
        self.channel = build_channel(channel)

    def _compute_gate(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate_network(h)
        soft_gate = torch.sigmoid(logits)
        if not self.hard:
            return soft_gate, logits

        hard_gate = (soft_gate >= self.gate_threshold).to(soft_gate.dtype)
        straight_through_gate = soft_gate + (hard_gate - soft_gate).detach()
        return straight_through_gate, logits

    def _budget_mask(
        self,
        gate_logits: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        """Select up to K available senders by learned gate score."""

        if self.sender_budget is None or self.sender_budget >= gate_logits.shape[-1]:
            return available
        if self.sender_budget == 0:
            return torch.zeros_like(available)

        if self.sender_selection == "learned":
            scores = gate_logits
        else:
            # A fixed seeded priority vector implements the random-K control
            # without resampling between rollout and optimizer forwards. It is
            # deterministic, device-independent, and fully recorded by config.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.sender_selection_seed)
            priorities = torch.rand(
                gate_logits.shape[-1],
                generator=generator,
                dtype=torch.float64,
            ).to(device=gate_logits.device, dtype=gate_logits.dtype)
            view_shape = (1,) * (gate_logits.dim() - 1) + (gate_logits.shape[-1],)
            scores = priorities.reshape(view_shape).expand_as(gate_logits)

        scores = scores.masked_fill(~available, torch.finfo(gate_logits.dtype).min)
        indices = scores.topk(self.sender_budget, dim=-1).indices
        selected = torch.zeros_like(available)
        selected.scatter_(-1, indices, True)
        return selected & available

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

        raw_messages = torch.tanh(self.message_encoder(h))
        gate, gate_logits = self._compute_gate(h)
        gated_messages = raw_messages * gate

        channel_output = self.channel(
            gated_messages,
            sender_mask=context.extras.get("sender_mask"),
        )
        topology_candidates = base_edges.any(dim=-2)
        budget_mask = self._budget_mask(
            gate_logits.squeeze(-1),
            channel_output.sender_mask & topology_candidates,
        )
        sent_messages = channel_output.messages * budget_mask.unsqueeze(-1).to(h.dtype)
        available_edges = sender_mask_to_edge_mask(budget_mask, base_edges)

        aggregated = masked_sender_mean(sent_messages, available_edges)
        delta = self.message_decoder(aggregated)
        output = h + delta if self.residual else delta

        learned_gate = torch.sigmoid(gate_logits).squeeze(-1)
        learned_active_gates = learned_gate >= self.gate_threshold
        effective_gate = gate.squeeze(-1) * budget_mask.to(gate.dtype)
        active_gate_senders = budget_mask & learned_active_gates
        message_norm = torch.linalg.vector_norm(sent_messages, dim=-1)
        scheduled_sender_fraction = active_sender_fraction(available_edges)
        costs = communication_costs_for_round(
            available_edges,
            base_edges,
            packet_dim=self.message_dim,
            active_senders=budget_mask,
        )
        stats: dict[str, torch.Tensor | float] = {
            # Soft gates generally leave a nonzero payload. Common cost metrics
            # therefore count channel- and budget-available senders/edges;
            # thresholded gate activity remains a separate diagnostic below.
            "active_sender_fraction": scheduled_sender_fraction,
            "active_edge_fraction": active_edge_fraction(
                available_edges,
                exclude_self=self.exclude_self,
            ),
            "mean_message_norm": message_norm.mean(),
            "max_message_norm": message_norm.max(),
            # Raw learned-gate diagnostics remain interpretable under channel
            # dropout and external budgets. Effective/scheduled quantities are
            # logged separately below.
            "mean_gate": learned_gate.mean(),
            "active_gate_fraction": learned_active_gates.to(torch.float32).mean(),
            "effective_mean_gate": effective_gate.mean(),
            "effective_active_gate_fraction": active_gate_senders.to(
                torch.float32
            ).mean(),
            "effective_active_senders": active_gate_senders.to(torch.float32)
            .sum(dim=-1)
            .mean(),
            "requested_dropout_rate": self.channel.requested_dropout_rate,
            "realized_communication_rate": channel_output.sender_mask.to(
                torch.float32
            ).mean(),
            **costs,
        }
        if self.sender_budget is not None:
            stats["sender_budget"] = float(self.sender_budget)
            stats["sender_selection_random"] = float(self.sender_selection == "random")
            stats["sender_selection_seed"] = float(self.sender_selection_seed)
        self._record_stats(**stats)

        return output
