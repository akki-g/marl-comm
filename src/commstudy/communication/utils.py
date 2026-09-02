"""Shape-, mask-, and aggregation helpers shared by communication modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_comm_input(
    h: torch.Tensor,
    hidden_dim: int,
    module_name: str = "CommModule",
) -> None:
    """Validate the common ``[..., N, D]`` communication input contract."""

    if not isinstance(h, torch.Tensor):
        raise TypeError(f"{module_name} expected a Tensor, got {type(h).__name__}.")
    if h.dim() < 2:
        raise ValueError(f"{module_name} expected [..., N, D], got {tuple(h.shape)}.")
    if h.shape[-2] < 1:
        raise ValueError(f"{module_name} requires at least one agent.")
    if h.shape[-1] != hidden_dim:
        raise ValueError(
            f"{module_name} expected hidden dimension {hidden_dim}, got {h.shape[-1]}."
        )
    if not h.is_floating_point():
        raise TypeError(
            f"{module_name} requires real floating-point embeddings, got {h.dtype}."
        )


def full_comm_mask(
    n_agents: int,
    *,
    leading_shape: Sequence[int] = (),
    device: torch.device | str | None = None,
    exclude_self: bool = True,
) -> torch.Tensor:
    """Construct a full receiver/sender mask with optional self edges."""

    if n_agents < 1:
        raise ValueError("n_agents must be >= 1")

    mask = torch.ones((n_agents, n_agents), dtype=torch.bool, device=device)
    if exclude_self:
        mask.fill_diagonal_(False)

    shape = (*tuple(leading_shape), n_agents, n_agents)
    return torch.broadcast_to(mask, shape).clone()


def resolve_comm_mask(
    h: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    exclude_self: bool = True,
) -> torch.Tensor:
    """Return a bool edge mask exactly matching ``h``'s leading dimensions.

    A supplied mask is authoritative and may use ordinary right-aligned
    broadcasting, including ``[N, N]`` and ``[B, N, N]`` for a
    ``[T, B, N, D]`` input. ``exclude_self=True`` always clears the diagonal;
    when it is false, a supplied diagonal is preserved.
    """

    if h.dim() < 2:
        raise ValueError(f"Expected h shaped [..., N, D], got {tuple(h.shape)}.")

    n_agents = h.shape[-2]
    target_shape = (*h.shape[:-2], n_agents, n_agents)

    if mask is None:
        return full_comm_mask(
            n_agents,
            leading_shape=h.shape[:-2],
            device=h.device,
            exclude_self=exclude_self,
        )

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Communication mask must be a Tensor, got {type(mask).__name__}.")
    if mask.dtype is not torch.bool:
        raise TypeError(f"Communication mask must be boolean, got {mask.dtype}.")
    if mask.dim() < 2 or tuple(mask.shape[-2:]) != (n_agents, n_agents):
        raise ValueError(
            "Communication mask must end in "
            f"({n_agents}, {n_agents}), got {tuple(mask.shape)}."
        )

    try:
        resolved = torch.broadcast_to(
            mask.to(device=h.device),
            target_shape,
        ).clone()
    except RuntimeError as exc:
        raise ValueError(
            f"Communication mask shape {tuple(mask.shape)} is not broadcastable "
            f"to {target_shape}."
        ) from exc

    if exclude_self:
        diagonal = torch.arange(n_agents, device=h.device)
        resolved[..., diagonal, diagonal] = False

    return resolved


def masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Softmax over valid entries, with exact zeros for all-masked rows."""

    if not scores.is_floating_point():
        raise TypeError(f"masked_softmax requires floating scores, got {scores.dtype}.")

    try:
        valid = torch.broadcast_to(
            mask.to(device=scores.device, dtype=torch.bool),
            scores.shape,
        )
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} is not broadcastable to scores "
            f"shape {tuple(scores.shape)}."
        ) from exc

    # A finite minimum avoids NaNs from softmax([-inf, ...]) on an all-masked
    # row. Multiplication and renormalization make masked probabilities exactly
    # zero while leaving every valid row normalized to one.
    filled = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(filled, dim=dim)
    probabilities = probabilities * valid.to(dtype=probabilities.dtype)
    normalizer = probabilities.sum(dim=dim, keepdim=True)
    safe_normalizer = normalizer.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return torch.where(
        normalizer > 0,
        probabilities / safe_normalizer,
        torch.zeros_like(probabilities),
    )


def masked_sender_mean(
    messages: torch.Tensor,
    edge_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean sender messages for every receiver, safely returning zero if empty.

    ``messages`` is ``[..., sender, message_dim]`` and ``edge_mask`` is
    ``[..., receiver, sender]``. The result is
    ``[..., receiver, message_dim]``.
    """

    if messages.dim() < 2:
        raise ValueError(
            f"Messages must be shaped [..., N, M], got {tuple(messages.shape)}."
        )

    n_agents = messages.shape[-2]
    if edge_mask.dim() < 2 or tuple(edge_mask.shape[-2:]) != (n_agents, n_agents):
        raise ValueError(
            f"Edge mask must end in ({n_agents}, {n_agents}), "
            f"got {tuple(edge_mask.shape)}."
        )

    valid = edge_mask.to(device=messages.device, dtype=torch.bool)
    weighted = messages.unsqueeze(-3) * valid.unsqueeze(-1).to(messages.dtype)
    message_sum = weighted.sum(dim=-2)
    count = valid.sum(dim=-1, keepdim=True).to(messages.dtype)
    return message_sum / count.clamp_min(1)


def sender_mask_to_edge_mask(
    sender_mask: torch.Tensor,
    edge_mask: torch.Tensor,
) -> torch.Tensor:
    """Remove unavailable senders from every receiver's incoming edges."""

    if sender_mask.dim() < 1:
        raise ValueError("Sender mask must have at least an agent dimension.")
    if sender_mask.dtype is not torch.bool or edge_mask.dtype is not torch.bool:
        raise TypeError("Sender and edge masks must both be boolean tensors.")
    if edge_mask.dim() < 2 or sender_mask.shape[-1] != edge_mask.shape[-1]:
        raise ValueError(
            "Sender and edge masks disagree on the number of agents: "
            f"{tuple(sender_mask.shape)} vs {tuple(edge_mask.shape)}."
        )

    available = sender_mask.to(device=edge_mask.device).unsqueeze(-2)
    try:
        return edge_mask & available
    except RuntimeError as exc:
        raise ValueError(
            f"Sender mask shape {tuple(sender_mask.shape)} is not broadcastable "
            f"to edge mask shape {tuple(edge_mask.shape)}."
        ) from exc


def active_sender_fraction(edge_mask: torch.Tensor) -> torch.Tensor:
    """Fraction of senders with at least one live outgoing edge."""

    return edge_mask.any(dim=-2).to(dtype=torch.float32).mean()


def active_edge_fraction(
    edge_mask: torch.Tensor,
    *,
    exclude_self: bool,
) -> torch.Tensor:
    """Mean directed-edge density relative to the configured potential graph."""

    n_agents = edge_mask.shape[-1]
    possible = n_agents * (n_agents - 1 if exclude_self else n_agents)
    if possible == 0:
        return torch.zeros((), dtype=torch.float32, device=edge_mask.device)

    live_per_sample = edge_mask.to(dtype=torch.float32).sum(dim=(-2, -1))
    return (live_per_sample / possible).mean()


def communication_costs_for_round(
    active_edges: torch.Tensor,
    base_edges: torch.Tensor,
    *,
    packet_dim: int,
    active_senders: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return detached-compatible communication costs for one round.

    Sender emissions and directed edge deliveries are deliberately separate.
    A radio-style broadcast may emit one sender packet that is delivered over
    several directed edges, while a point-to-point accounting convention may
    charge every delivery. ``packet_dim`` is the transmitted sender payload
    width (message only for Broadcast/Gated, key plus value for
    Attention/Graph).
    """

    if packet_dim < 0:
        raise ValueError("packet_dim must be non-negative")
    if active_edges.shape[-2:] != base_edges.shape[-2:]:
        raise ValueError(
            "Active and base edge masks must end in the same receiver/sender shape."
        )

    active_edges = active_edges.to(dtype=torch.bool)
    base_edges = base_edges.to(device=active_edges.device, dtype=torch.bool)
    try:
        base_edges = torch.broadcast_to(base_edges, active_edges.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Base edge shape {tuple(base_edges.shape)} is not broadcastable to "
            f"active edge shape {tuple(active_edges.shape)}."
        ) from exc

    base_sender_mask = base_edges.any(dim=-2)
    if active_senders is None:
        active_sender_mask = active_edges.any(dim=-2)
    else:
        try:
            active_sender_mask = torch.broadcast_to(
                active_senders.to(device=active_edges.device, dtype=torch.bool),
                base_sender_mask.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Active sender shape {tuple(active_senders.shape)} is not "
                f"broadcastable to {tuple(base_sender_mask.shape)}."
            ) from exc
        # A sender with no configured receiver does not need to emit a packet.
        active_sender_mask = active_sender_mask & base_sender_mask

    active_messages = active_sender_mask.to(torch.float32).sum(dim=-1).mean()
    nominal_messages = base_sender_mask.to(torch.float32).sum(dim=-1).mean()
    active_deliveries = active_edges.to(torch.float32).sum(dim=(-2, -1)).mean()
    nominal_deliveries = base_edges.to(torch.float32).sum(dim=(-2, -1)).mean()
    scalar_width = active_deliveries.new_tensor(float(packet_dim))
    bits_per_scalar = active_deliveries.new_tensor(32.0)

    return {
        "messages_per_step": active_messages,
        "active_edges_per_step": active_deliveries,
        "nominal_messages_per_step": nominal_messages,
        "potential_edges_per_step": nominal_deliveries,
        "realized_sender_scalars_per_step": active_messages * scalar_width,
        "realized_sender_bits_per_step": (
            active_messages * scalar_width * bits_per_scalar
        ),
        "nominal_sender_scalars_per_step": nominal_messages * scalar_width,
        "nominal_sender_bits_per_step": (
            nominal_messages * scalar_width * bits_per_scalar
        ),
        "realized_scalar_transmissions_per_step": (
            active_deliveries * scalar_width
        ),
        "realized_bits_per_step": (
            active_deliveries * scalar_width * bits_per_scalar
        ),
        "nominal_scalar_transmissions_per_step": (
            nominal_deliveries * scalar_width
        ),
        "nominal_bits_per_step": (
            nominal_deliveries * scalar_width * bits_per_scalar
        ),
    }

def pairwise_class_bias(
    class_bias: torch.Tensor,
    class_id: torch.Tensor | None,
    module_name: str,
) -> torch.Tensor:
    """Expand ``[num_roles, num_roles, num_heads]`` to ``[..., N, N, num_heads]``.

    Indexed ``[receiver_class, sender_class]`` to match the score layout
    ``[..., receiver, sender, head]``.
    """

    if class_id is None:
        raise ValueError(
            f"{module_name} was built with role_aware=True but the CommContext "
            "carried no class_id."
        )
    cid = class_id.long()
    return class_bias[cid.unsqueeze(-1), cid.unsqueeze(-2)]