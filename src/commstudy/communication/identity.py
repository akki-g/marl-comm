import torch

from .base import CommContext, CommModule
from .utils import validate_comm_input


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

        validate_comm_input(h, self.hidden_dim, self.__class__.__name__)

        return h

    def message_bits(self) -> int:
        return 0

    def communication_stats(self) -> dict[str, float]:
        """Report an explicit zero-communication control."""

        return {
            "message_dim": 0.0,
            "message_bits_per_sender": 0.0,
            "active_sender_fraction": 0.0,
            "active_edge_fraction": 0.0,
            "mean_message_norm": 0.0,
            "max_message_norm": 0.0,
            "communication_rounds": 0.0,
            "messages_per_step": 0.0,
            "active_edges_per_step": 0.0,
            "nominal_messages_per_step": 0.0,
            "potential_edges_per_step": 0.0,
            "realized_sender_scalars_per_step": 0.0,
            "realized_sender_bits_per_step": 0.0,
            "nominal_sender_scalars_per_step": 0.0,
            "nominal_sender_bits_per_step": 0.0,
            "realized_scalar_transmissions_per_step": 0.0,
            "realized_bits_per_step": 0.0,
            "nominal_scalar_transmissions_per_step": 0.0,
            "nominal_bits_per_step": 0.0,
        }
