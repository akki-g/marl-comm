from __future__ import annotations

from dataclasses import dataclass, field

from benchmarl.models.common import ModelConfig

from src.commstudy.models.model import CommPolicyModel


@dataclass
class CommPolicyConfig(ModelConfig):
    hidden_dim: int = 128
    num_encoder_layers: int = 2

    activation_class_path: str = "torch.nn.Tanh"

    comm_class_path: str = (
        "commstudy.communication.identity.IdentityComm"
    )

    comm_kwargs: dict = field(
        default_factory=dict
    )

    comm_context_keys: dict = field(
        default_factory=lambda: {
            "mask": "comm_mask",
            "class_id": "agent_type",
        }
    )

    @staticmethod
    def associated_class():
        return CommPolicyModel