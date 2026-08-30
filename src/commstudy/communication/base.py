"""
Abstract base class for swappable communication modules.

Shape contract
--------------
h        : Tensor [B, N, D] or [T, B, N, D]
             per-agent embeddings produced by the shared encoder.
mask     : Optional Tensor [B, N, N] or [T, B, N, N]  bool
             True where agent i is allowed to receive from agent j.
class_id : Optional Tensor [N]  long
             agent-class identifier for heterogeneous comm modules.

return   : Tensor with identical shape to h.

The comm slot sits between the shared encoder and the actor head.
It must NOT be used as part of the centralized critic input path.
All future communication modules (Broadcast, Attention, Graph, …)
must subclass CommModule and implement forward().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

@dataclass
class CommContext:
    """
    Runtime info available to a comm module
    mask: 
        Communication adjacency/mask [..., N, N]
    
    class_id:
        optional agent role/class identifiers
    
    extras: 
        Additional task-specific communication metadata
    """
    mask: torch.Tensor | None = None

    class_id: torch.Tensor | None = None

    extras: dict[str, Any] = field(default_factory=dict)
    
class CommModule(nn.Module, ABC):
    """
    Base interface for all communication modules.

    Input:
        h: [..., N, D]

    Output:
        [..., N, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        h: torch.Tensor,
        context: CommContext | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def message_bits(self) -> int:
        return 0