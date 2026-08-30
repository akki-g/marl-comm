import torch

from .base import CommContext, CommModule


class IdentityComm(CommModule):
    """
    No-communication baseline.

    Returns each agent's local embedding unchanged.
    """

    def __init__(
        self,
        hidden_dim: int,
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            **kwargs,
        )

    def forward(
        self,
        h: torch.Tensor,
        context: CommContext | None = None,
    ) -> torch.Tensor:

        if h.dim() < 2:
            raise ValueError(
                f"Expected [..., N, D], got {tuple(h.shape)}"
            )

        if h.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}, "
                f"got {h.shape[-1]}"
            )

        return h

    def message_bits(self) -> int:
        return 0