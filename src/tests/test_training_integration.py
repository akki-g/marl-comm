from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from benchmarl.experiment.callback import Callback
from benchmarl.models.mlp import Mlp

from commstudy.communication.base import CommModule
from commstudy.experiments import build_experiment, load_experiment_spec
from commstudy.models import CommPolicyModel


GROUP = "agents"
LEARNED_MODELS = (
    "comm_broadcast",
    "comm_gated",
    "comm_attention",
    "comm_graph",
)


def _actor(experiment) -> CommPolicyModel:
    actors = [
        module
        for module in experiment.group_policies[GROUP].modules()
        if isinstance(module, CommPolicyModel)
    ]
    assert len(actors) == 1
    return actors[0]


class OptimizationProbe(Callback):
    """Capture evidence that BenchMARL really optimizes the custom actor."""

    def __init__(self, *, expect_sender_mask: bool) -> None:
        super().__init__()
        self.expect_sender_mask = expect_sender_mask
        self.before: dict[str, torch.Tensor] = {}
        self.actor_gradient_norms: list[float] = []
        self.comm_gradient_norms: list[float] = []
        self.training_values: list[torch.Tensor] = []
        self.collection_finite = False
        self.collection_sender_mask = False
        self.optimizer_sender_mask = False
        self._handles = []

    def on_setup(self) -> None:
        actor = _actor(self.experiment)
        self.before = {
            name: parameter.detach().clone()
            for name, parameter in actor.named_parameters()
        }
        comm_parameter_ids = {id(parameter) for parameter in actor.comm.parameters()}

        for parameter in actor.parameters():
            is_comm = id(parameter) in comm_parameter_ids

            def capture(gradient, *, communication=is_comm):
                norm = float(gradient.detach().norm().cpu())
                if communication:
                    self.comm_gradient_norms.append(norm)
                else:
                    self.actor_gradient_norms.append(norm)

            self._handles.append(parameter.register_hook(capture))

    def on_batch_collected(self, batch) -> None:
        floating = [
            value
            for value in batch.values(include_nested=True, leaves_only=True)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ]
        self.collection_finite = bool(floating) and all(
            torch.isfinite(value).all() for value in floating
        )
        self.collection_sender_mask = (
            GROUP,
            "comm_sender_mask",
        ) in set(batch.keys(include_nested=True, leaves_only=True))

    def on_train_step(self, batch, group: str):
        self.optimizer_sender_mask |= (
            group,
            "comm_sender_mask",
        ) in set(batch.keys(include_nested=True, leaves_only=True))
        return None

    def on_train_end(self, training_td, group: str) -> None:
        del group
        self.training_values.extend(
            value.detach().cpu()
            for value in training_td.values(include_nested=True, leaves_only=True)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        )

    def assert_completed(self, experiment, *, learned: bool) -> None:
        actor = _actor(experiment)
        after = dict(actor.named_parameters())
        changed = {
            name
            for name, before in self.before.items()
            if not torch.equal(before, after[name].detach())
        }
        comm_names = {f"comm.{name}" for name, _ in actor.comm.named_parameters()}

        assert self.collection_finite
        assert self.training_values
        assert all(torch.isfinite(value).all() for value in self.training_values)
        assert any(
            norm > 0 and torch.isfinite(torch.tensor(norm))
            for norm in self.actor_gradient_norms
        )
        assert changed
        if learned:
            assert comm_names
            assert changed & comm_names
            assert any(
                norm > 0 and torch.isfinite(torch.tensor(norm))
                for norm in self.comm_gradient_norms
            )
        if self.expect_sender_mask:
            assert self.collection_sender_mask
            assert self.optimizer_sender_mask

        for handle in self._handles:
            handle.remove()


def _build_short_experiment(
    config_root: Path,
    tmp_path: Path,
    *,
    algorithm: str,
    model: str,
    callback: Callback,
    extra_overrides: tuple[str, ...] = (),
):
    overrides = [
        f"algorithm={algorithm}",
        f"model={model}",
        "seed=0",
        "experiment.max_n_frames=6000",
        "experiment.evaluation=false",
        "experiment.loggers=[]",
        "experiment.create_json=false",
        "experiment.checkpoint_at_end=false",
    ]
    if algorithm == "mappo":
        overrides.append("experiment.on_policy_n_minibatch_iters=1")
    else:
        overrides.extend(
            [
                "experiment.off_policy_init_random_frames=0",
                "experiment.off_policy_n_optimizer_steps=1",
                "experiment.off_policy_memory_size=12000",
            ]
        )

    overrides.extend(extra_overrides)

    spec = load_experiment_spec(config_root, overrides)
    save_folder = tmp_path / f"{algorithm}_{model}"
    spec = dataclasses.replace(
        spec,
        experiment={**spec.experiment, "save_folder": str(save_folder)},
    )
    return build_experiment(spec, callbacks=[callback])


class DropoutLifecycleProbe(OptimizationProbe):
    """Retain the collected channel trace for temporal-lifecycle checks."""

    def __init__(self) -> None:
        super().__init__(expect_sender_mask=True)
        self.collected_mask: torch.Tensor | None = None
        self.collected_marker: torch.Tensor | None = None

    def on_batch_collected(self, batch) -> None:
        super().on_batch_collected(batch)
        self.collected_mask = batch.get(
            (GROUP, "comm_sender_mask")
        ).detach().cpu().clone()
        self.collected_marker = batch.get(
            (GROUP, "comm_sender_mask_generated")
        ).detach().cpu().clone()


@pytest.mark.parametrize("model", LEARNED_MODELS)
def test_mappo_real_iteration_updates_each_communication_module(
    config_root,
    tmp_path,
    model,
):
    probe = OptimizationProbe(expect_sender_mask=True)
    experiment = _build_short_experiment(
        config_root,
        tmp_path,
        algorithm="mappo",
        model=model,
        callback=probe,
    )

    actor = _actor(experiment)
    critics = [
        module
        for module in experiment.losses[GROUP].critic_network.modules()
        if isinstance(module, (Mlp, CommPolicyModel))
    ]
    assert isinstance(actor.comm, CommModule)
    assert len(critics) == 1
    assert isinstance(critics[0], Mlp)
    assert critics[0].centralised

    experiment.run()

    assert experiment.total_frames == 6000
    assert experiment.n_iters_performed == 1
    probe.assert_completed(experiment, learned=True)


def test_mappo_dropout_is_fresh_each_step_and_replayed_to_optimizer(
    config_root,
    tmp_path,
):
    probe = DropoutLifecycleProbe()
    experiment = _build_short_experiment(
        config_root,
        tmp_path,
        algorithm="mappo",
        model="comm_attention",
        callback=probe,
        extra_overrides=(
            "model_config.params.comm_kwargs.channel.type=dropout",
            "model_config.params.comm_kwargs.channel.p=0.5",
            "model_config.params.comm_kwargs.channel.mode=always",
        ),
    )

    experiment.run()

    probe.assert_completed(experiment, learned=True)
    assert probe.collected_mask is not None
    assert probe.collected_marker is not None
    assert probe.collected_marker.all()
    # VMAS is collected as [parallel_env, time, agent]. Every environment
    # must see more than one independent channel realization over 600 steps;
    # a fixed mask here is the stale-policy-output latching failure.
    assert probe.collected_mask.ndim == 3
    assert all(
        torch.unique(environment_trace, dim=0).shape[0] > 1
        for environment_trace in probe.collected_mask
    )


@pytest.mark.parametrize(
    "model",
    ("comm_identity", "comm_broadcast", "comm_attention", "comm_graph"),
)
def test_qmix_real_iteration_is_algorithm_independent(
    config_root,
    tmp_path,
    model,
):
    learned = model != "comm_identity"
    probe = OptimizationProbe(expect_sender_mask=learned)
    experiment = _build_short_experiment(
        config_root,
        tmp_path,
        algorithm="qmix",
        model=model,
        callback=probe,
    )

    assert not experiment.continuous_actions
    assert _actor(experiment).out_key == (GROUP, "action_value")

    experiment.run()

    assert experiment.total_frames == 6000
    assert experiment.n_iters_performed == 1
    probe.assert_completed(experiment, learned=learned)
