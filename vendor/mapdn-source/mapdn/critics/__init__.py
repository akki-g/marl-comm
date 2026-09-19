"""Critic networks. Requires torch."""

from .maac_critic import AttentionCritic
from .mlp_critic import MLPCritic
from .qmix import QMixer
from .rnn_critic import RNNCritic

__all__ = [
    "AttentionCritic",
    "MLPCritic",
    "QMixer",
    "RNNCritic",
]
