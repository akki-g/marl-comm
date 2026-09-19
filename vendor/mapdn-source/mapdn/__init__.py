"""MAPDN: Multi-Agent Power Distribution Networks.

The environments are the only thing exposed at the top level, so that
``import mapdn`` stays light and does not pull in torch::

    from mapdn import VoltageControl
    from mapdn.environments import VoltageControl

The MARL stack (which requires torch) lives in the subpackages and is
imported explicitly::

    from mapdn.models import Model, Strategy, MADDPG
    from mapdn.utilities import PGTrainer, PGTester
"""

from .environments import ActionSpace, MultiAgentEnv, VoltageBarrier, VoltageControl

__version__ = "0.1.0"

__all__ = [
    "ActionSpace",
    "MultiAgentEnv",
    "VoltageBarrier",
    "VoltageControl",
]
