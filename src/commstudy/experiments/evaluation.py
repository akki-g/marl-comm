"""Evaluation lifecycle guards around the unchanged BenchMARL implementation."""

from contextlib import contextmanager

from benchmarl.experiment import Experiment

from commstudy.communication.channel import CommChannel, channel_phase, preserve_channel_rng
from commstudy.utils.rng import preserve_rng_state


class EvaluationIsolatedExperiment(Experiment):
    """Preserve training randomness and explicitly mark evaluation channels.

    BenchMARL's existing local_seed decorator preserves CPU globals on success.
    This adds VMAS's shared stream, initialized CUDA streams, failure restoration,
    and private communication generators. Collection and PPO remain upstream code.
    The separate test environment retains its own stream when evaluation_static
    is false and is reseeded by BenchMARL when evaluation_static is true.
    """

    def _setup_task(self):
        super()._setup_task()
        # The factory remains training-bound. The separately constructed test
        # environment receives its explicit held-out split before evaluation.
        configure = getattr(self.task, "configure_evaluation_env", None)
        if configure is not None:
            configure(self.test_env)

    @contextmanager
    def _evaluation_context(self):
        channels = [module for module in self.policy.modules() if isinstance(module, CommChannel)]
        index = getattr(self, "_evaluation_rng_index", 0)
        # Static evaluation repeats its declared draws; non-static evaluation
        # gets the next independent draw set, even for evaluation-only channels
        # that have never run in collection. Neither depends on training history.
        seed = (self.seed + 0x4556414C + (0 if self.config.evaluation_static else index)) % (
            2**63 - 1
        )
        with (
            preserve_rng_state(seed=seed),
            preserve_channel_rng(channels, seed=seed),
            channel_phase("evaluation"),
        ):
            yield
        self._evaluation_rng_index = index + 1

    def _collection_loop(self):
        # The upstream loop retains complete ownership of collection and PPO.
        # Nested optimizer/evaluation contexts override this collection default.
        with channel_phase("collection"):
            return super()._collection_loop()

    def _optimizer_loop(self, group):
        with channel_phase("optimization"):
            return super()._optimizer_loop(group)

    def _evaluation_loop(self):
        with self._evaluation_context():
            return super()._evaluation_loop()

    def evaluate(self):
        # Public evaluate() seeds before calling _evaluation_loop; protect that
        # outer seed as well as callbacks/logging and the actual rollout.
        with preserve_rng_state():
            return super().evaluate()
