"""Actor networks. Requires torch.

The deterministic and Gaussian variants share class names in their own
modules, so the Gaussian ones are re-exported under distinct aliases.
"""

from .mlp_agent import MLPAgent
from .mlp_agent_gaussian import MLPAgent as GaussianMLPAgent
from .rnn_agent import RNNAgent
from .rnn_agent_gaussian import RNNAgent as GaussianRNNAgent

__all__ = [
    "GaussianMLPAgent",
    "GaussianRNNAgent",
    "MLPAgent",
    "RNNAgent",
]
