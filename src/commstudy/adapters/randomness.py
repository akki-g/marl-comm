"""Per-instance VMAS randomness without changing simulator or MAPPO code."""

from contextlib import contextmanager

import torch
from torchrl.envs.libs.vmas import VmasEnv

from commstudy.utils.rng import RNGState, preserve_rng_state


class IsolatedVmasEnv(VmasEnv):
    """Give reset/dynamics a stream independent of actors and other envs.

    VMAS's own decorators use a class-shared stream; ordinary global RNG forking
    does not isolate it. Install and restore that stream along with CPU, CUDA,
    Python, and NumPy state at each environment boundary. The scenario separately
    indexes exogenous prey noise by episode and step, so different reset/dynamics
    branches cannot change noise for an already-running world.
    """

    def __init__(self, *args, seed=None, **kwargs):
        device = torch.device(kwargs.get("device") or "cpu")
        if device.type == "cuda":
            # Materialize the caller's CUDA state before the first snapshot;
            # otherwise simulator construction could initialize and seed CUDA
            # inside an isolation context with no prior state to restore.
            torch.cuda.get_rng_state(device)
        self._task_random_state = RNGState.seeded(0 if seed is None else seed)
        self._rng_depth = 0
        with self._isolated_rng():
            super().__init__(*args, seed=seed, **kwargs)

    @contextmanager
    def _isolated_rng(self):
        if self._rng_depth:
            yield
            return
        with preserve_rng_state():
            self._task_random_state.restore()
            self._rng_depth += 1
            try:
                yield
            finally:
                self._task_random_state = RNGState.capture()
                self._rng_depth -= 1

    def _set_seed(self, seed):
        # A new seed means a fresh episode schedule on the next reset, rather
        # than continuing any old wander direction or noise cursor.
        with self._isolated_rng():
            return super()._set_seed(seed)

    def _reset(self, tensordict=None, **kwargs):
        with self._isolated_rng():
            return super()._reset(tensordict, **kwargs)

    def _step(self, tensordict):
        with self._isolated_rng():
            return super()._step(tensordict)
