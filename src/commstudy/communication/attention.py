"""TarMAC-inspired targeted multi-agent communication.

This module adopts TarMAC's sender key/value and receiver query mechanism while
fitting it into the project's stateless ``CommModule`` interface.  It is not a
full TarMAC reproduction: the actor remains BenchMARL's actor, communication
rounds share parameters, and the recurrent architecture from the paper is not
used here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .base import CommContext, CommModule
from .channel import CommChannel, build_channel
from .utils import (
    communication_costs_for_round,
    masked_softmax,
    resolve_comm_mask,
    sender_mask_to_edge_mask,
    validate_comm_input,
)


class AttentionComm(CommModule):
    """Scaled multi-head query/key communication over permitted senders.

    ``message_dim`` and ``key_dim`` are total dimensions across all heads, not
    per-head dimensions.  Both must therefore be divisible by ``num_heads``.
    A sender's key and value are transformed by the same communication channel
    so whole-sender failures remove the sender from the attention logits.

    Parameters are shared when ``rounds`` is greater than one.  Optional sender
    budgets can select senders from the current attention logits or from a
    fixed, seeded random ordering.  The latter is deliberately fixed so PPO
    rollout/update evaluation does not resample an unobserved channel state.
    """

    def __init__(
        self,
        hidden_dim: int,
        message_dim: int = 32,
        key_dim: int = 32,
        num_heads: int = 4,
        rounds: int = 1,
        temperature: float = 1.0,
        exclude_self: bool = True,
        residual: bool = True,
        channel: CommChannel | Mapping[str, Any] | Sequence[Any] | None = None,
        sender_budget: int | None = None,
        sender_selection: str = "attention",
        sender_selection_seed: int = 0,
        store_debug_attention: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, **kwargs)

        self._validate_configuration(
            message_dim=message_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            rounds=rounds,
            temperature=temperature,
            sender_budget=sender_budget,
            sender_selection=sender_selection,
        )

        self.message_dim = message_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.communication_rounds = rounds
        self.rounds = rounds
        self.temperature = float(temperature)
        self.exclude_self = exclude_self
        self.residual = residual
        self.sender_budget = sender_budget
        self.sender_selection = sender_selection
        self.sender_selection_seed = int(sender_selection_seed)
        self.store_debug_attention = store_debug_attention

        self.query_projection = nn.Linear(hidden_dim, key_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, key_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, message_dim, bias=False)
        self.output_projection = nn.Linear(message_dim, hidden_dim, bias=False)
        self.channel = build_channel(channel)

        self._reset_parameters()
        self._debug_attention: torch.Tensor | None = None
        self._random_priority_cache: dict[int, torch.Tensor] = {}

    @staticmethod
    def _validate_configuration(
        *,
        message_dim: int,
        key_dim: int,
        num_heads: int,
        rounds: int,
        temperature: float,
        sender_budget: int | None,
        sender_selection: str,
    ) -> None:
        if message_dim <= 0 or key_dim <= 0:
            raise ValueError("message_dim and key_dim must both be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if message_dim % num_heads != 0:
            raise ValueError("message_dim must be divisible by num_heads")
        if key_dim % num_heads != 0:
            raise ValueError("key_dim must be divisible by num_heads")
        if not 1 <= rounds <= 3:
            raise ValueError("rounds must be between 1 and 3")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sender_budget is not None and sender_budget < 0:
            raise ValueError("sender_budget must be non-negative")
        if sender_selection not in {"attention", "random"}:
            raise ValueError("sender_selection must be 'attention' or 'random'")

    def _reset_parameters(self) -> None:
        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)

    @property
    def debug_attention(self) -> torch.Tensor | None:
        """Last attention matrix, detached on CPU, when explicitly enabled."""

        return self._debug_attention

    def _fixed_random_priorities(self, n_agents: int, device: torch.device) -> torch.Tensor:
        cached = self._random_priority_cache.get(n_agents)
        if cached is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.sender_selection_seed)
            cached = torch.rand(n_agents, generator=generator)
            self._random_priority_cache[n_agents] = cached
        return cached.to(device=device)

    def _apply_sender_budget(
        self,
        scores: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.sender_budget is None:
            return edge_mask, None

        n_agents = edge_mask.shape[-1]
        budget = min(self.sender_budget, n_agents)
        if budget == n_agents:
            return edge_mask, None
        if budget == 0:
            return torch.zeros_like(edge_mask), None

        candidates = edge_mask.any(dim=-2)
        if self.sender_selection == "attention":
            valid_scores = scores.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))
            priorities = valid_scores.amax(dim=(-3, -1))
        else:
            priorities = self._fixed_random_priorities(n_agents, scores.device)
            priorities = priorities.expand_as(candidates)

        priorities = priorities.masked_fill(~candidates, float("-inf"))
        selected_indices = priorities.topk(budget, dim=-1).indices
        selected = torch.zeros_like(candidates)
        selected.scatter_(-1, selected_indices, True)
        selected &= candidates
        selected_edges = sender_mask_to_edge_mask(selected, edge_mask)

        if self.sender_selection == "attention":
            # Hard top-K defines the forward communication budget. A
            # straight-through softmax surrogate lets policy gradients train
            # the Q/K scores even for K=1, where hard masked attention itself
            # would have a constant probability of one and zero Q/K gradient.
            soft_selection = masked_softmax(priorities, candidates, dim=-1)
            sender_gate = (
                selected.to(dtype=soft_selection.dtype)
                + soft_selection
                - soft_selection.detach()
            )
        else:
            sender_gate = None
        return selected_edges, sender_gate

    @staticmethod
    def _possible_edge_count(h: torch.Tensor, exclude_self: bool) -> int:
        n_agents = h.shape[-2]
        samples = math.prod(h.shape[:-2]) if h.shape[:-2] else 1
        per_sample = n_agents * (n_agents - 1 if exclude_self else n_agents)
        return samples * per_sample

    @staticmethod
    def _round_statistics(
        *,
        h: torch.Tensor,
        values: torch.Tensor,
        base_edge_mask: torch.Tensor,
        active_edge_mask: torch.Tensor,
        sender_mask: torch.Tensor,
        attention: torch.Tensor,
        exclude_self: bool,
    ) -> dict[str, torch.Tensor]:
        valid_rows = active_edge_mask.any(dim=-1).unsqueeze(-1).expand_as(
            attention.sum(dim=-2)
        )
        # Compute diagnostics in float32: 1e-12 underflows to zero in fp16,
        # turning masked probabilities into 0 * log(0) = NaN even when the
        # actual communication output is finite.
        diagnostic_attention = attention.to(dtype=torch.float32)
        entropy = -(
            diagnostic_attention
            * diagnostic_attention.clamp_min(
                torch.finfo(diagnostic_attention.dtype).tiny
            ).log()
        ).sum(dim=-2)
        maximum = diagnostic_attention.amax(dim=-2)
        valid_count = valid_rows.sum().clamp_min(1)

        active_senders = active_edge_mask.any(dim=-2)
        message_norms = values.norm(dim=-1)
        active_sender_count = active_senders.sum().clamp_min(1)
        mean_message_norm = (
            message_norms * active_senders.to(message_norms.dtype)
        ).sum() / active_sender_count
        max_message_norm = (
            message_norms * active_senders.to(message_norms.dtype)
        ).amax()

        possible_edges = active_edge_mask.new_tensor(
            AttentionComm._possible_edge_count(h, exclude_self),
            dtype=torch.float32,
        ).clamp_min(1)
        return {
            "active_sender_fraction": active_senders.float().mean(),
            "active_edge_fraction": active_edge_mask.sum() / possible_edges,
            "realized_communication_rate": sender_mask.to(torch.float32).mean(),
            "mean_message_norm": mean_message_norm,
            "max_message_norm": max_message_norm,
            "attention_entropy": entropy.masked_fill(~valid_rows, 0.0).sum()
            / valid_count,
            "attention_max_probability": maximum.masked_fill(~valid_rows, 0.0).sum()
            / valid_count,
            "effective_neighbor_count": entropy.exp().masked_fill(~valid_rows, 0.0).sum()
            / valid_count,
        }

    def forward(
        self,
        h: torch.Tensor,
        context: CommContext | None = None,
    ) -> torch.Tensor:
        validate_comm_input(h, self.hidden_dim, self.__class__.__name__)
        base_edge_mask = resolve_comm_mask(
            h,
            context.mask if context is not None else None,
            exclude_self=self.exclude_self,
        )
        explicit_sender_mask = (
            context.extras.get("sender_mask") if context is not None else None
        )

        n_agents = h.shape[-2]
        key_head_dim = self.key_dim // self.num_heads
        value_head_dim = self.message_dim // self.num_heads
        result = h
        stats_by_round: list[dict[str, torch.Tensor]] = []
        costs_by_round: list[dict[str, torch.Tensor]] = []
        round_sender_mask = explicit_sender_mask

        for _ in range(self.rounds):
            queries = self.query_projection(result)
            sender_packet = torch.cat(
                (self.key_projection(result), self.value_projection(result)), dim=-1
            )
            channel_output = self.channel(sender_packet, sender_mask=round_sender_mask)
            # A channel failure is an environment-step realization, not a new
            # unobserved random variable for every internal communication
            # round. Reusing it also means the single stored sender mask fully
            # specifies the policy forward during PPO replay.
            if round_sender_mask is None:
                round_sender_mask = channel_output.sender_mask
            keys, values = channel_output.messages.split(
                (self.key_dim, self.message_dim), dim=-1
            )

            queries = queries.reshape(
                *result.shape[:-2], n_agents, self.num_heads, key_head_dim
            )
            keys = keys.reshape(
                *result.shape[:-2], n_agents, self.num_heads, key_head_dim
            )
            scores = (queries.unsqueeze(-3) * keys.unsqueeze(-4)).sum(dim=-1)
            scores = scores / (math.sqrt(key_head_dim) * self.temperature)

            active_edge_mask = sender_mask_to_edge_mask(
                channel_output.sender_mask, base_edge_mask
            )
            active_edge_mask, sender_gate = self._apply_sender_budget(
                scores,
                active_edge_mask,
            )
            if sender_gate is not None:
                values = values * sender_gate.unsqueeze(-1)
            head_values = values.reshape(
                *result.shape[:-2], n_agents, self.num_heads, value_head_dim
            )
            attention = masked_softmax(scores, active_edge_mask.unsqueeze(-1), dim=-2)

            aggregated = (
                attention.unsqueeze(-1) * head_values.unsqueeze(-4)
            ).sum(dim=-3)
            aggregated = aggregated.reshape(
                *result.shape[:-2], n_agents, self.message_dim
            )
            update = self.output_projection(aggregated)
            result = result + update if self.residual else update

            stats_by_round.append(
                self._round_statistics(
                    h=h,
                    values=values,
                    base_edge_mask=base_edge_mask,
                    active_edge_mask=active_edge_mask,
                    sender_mask=channel_output.sender_mask,
                    attention=attention,
                    exclude_self=self.exclude_self,
                )
            )
            costs_by_round.append(
                communication_costs_for_round(
                    active_edge_mask,
                    base_edge_mask,
                    packet_dim=self.key_dim + self.message_dim,
                )
            )
            if self.store_debug_attention:
                self._debug_attention = attention.detach().cpu()

        averaged_stats = {}
        for name in stats_by_round[0]:
            round_values = torch.stack(
                [round_stats[name] for round_stats in stats_by_round]
            )
            averaged_stats[name] = (
                round_values.max()
                if name == "max_message_norm"
                else round_values.mean()
            )
        step_costs = {
            name: torch.stack(
                [round_costs[name] for round_costs in costs_by_round]
            ).sum()
            for name in costs_by_round[0]
        }
        self._record_stats(
            **averaged_stats,
            **step_costs,
            requested_dropout_rate=self.channel.requested_dropout_rate,
        )
        return result

    def message_bits(self) -> int:
        """Nominal float32 sender payload: key plus value for every round."""

        return self.rounds * (self.key_dim + self.message_dim) * 32
