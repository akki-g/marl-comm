"""Exception-safe runtime RNG isolation for simulation and evaluation.

VMAS 1.5.2 stores its CPU/Python/NumPy stream on the Environment *class*.
Saving only torch's global state therefore does not isolate two environments.
CUDA streams also need explicit protection: the upstream VMAS and BenchMARL
local_seed contexts save CPU torch state only. These contexts assume serialized
environment calls within each worker process, as in the managed collector.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import random
import sys
from typing import Any

import numpy as np
import torch


def _vmas_state():
    module = sys.modules.get("vmas.simulator.environment.environment")
    return None if module is None else module.Environment.vmas_random_state


@dataclass
class RNGState:
    python: Any
    numpy: Any
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None
    vmas: list | None

    @classmethod
    def capture(cls) -> RNGState:
        return cls(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch_cpu=torch.random.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            vmas=deepcopy(_vmas_state()),
        )

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.random.set_rng_state(self.torch_cpu)
        if self.torch_cuda is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda)
        shared = _vmas_state()
        if shared is not None and self.vmas is not None:
            shared[:] = deepcopy(self.vmas)

    @classmethod
    def seeded(cls, seed: int) -> RNGState:
        with preserve_rng_state(seed=seed):
            return cls.capture()


@contextmanager
def preserve_rng_state(seed: int | None = None):
    """Restore caller RNG streams even on failure; optionally seed the body.

    This protects random streams, not simulator tensors. Evaluation must use its
    own environment, and communication generators need their own state context.
    It promises within-runtime repeatability, not CPU/CUDA trajectory equivalence.
    """
    before = RNGState.capture()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            shared = _vmas_state()
            if shared is not None:
                shared[:] = [torch.random.get_rng_state(), np.random.get_state(), random.getstate()]
        yield
    finally:
        before.restore()
