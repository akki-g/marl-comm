"""MARL algorithms. Requires torch.

``Model`` and ``Strategy`` map an algorithm name (e.g. ``"maddpg"``) to its
class and to the training strategy it needs.
"""

from .coma import COMA
from .facmaddpg import FACMADDPG
from .iac import IAC
from .iddpg import IDDPG
from .ippo import IPPO
from .maac import MAAC
from .maddpg import MADDPG
from .mappo import MAPPO
from .matd3 import MATD3
from .model import Model as BaseModel
from .model_registry import Model, Strategy
from .random import RandomAgent
from .sqddpg import SQDDPG

__all__ = [
    "BaseModel",
    "COMA",
    "FACMADDPG",
    "IAC",
    "IDDPG",
    "IPPO",
    "MAAC",
    "MADDPG",
    "MAPPO",
    "MATD3",
    "Model",
    "RandomAgent",
    "SQDDPG",
    "Strategy",
]
