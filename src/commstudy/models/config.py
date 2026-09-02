from __future__ import annotations

from dataclasses import dataclass, field

from benchmarl.models.common import ModelConfig

from commstudy.models.model import CommPolicyModel


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

    use_role_embedding: bool = False
    num_roles: int = 2 

    @staticmethod
    def associated_class():
        return CommPolicyModel
