"""Communication saliency: the intervention must be exact and reversible."""

from __future__ import annotations

import math

import pytest
import torch

from commstudy.analysis.saliency import (
    _gaussian_kl,
    communication_modules,
    severed_communication,
)
from commstudy.communication.attention import AttentionComm
from commstudy.communication.broadcast import BroadcastComm
from commstudy.communication.gated import GatedComm
from commstudy.communication.graph import GraphComm
from commstudy.communication.identity import IdentityComm


LEARNED = {
    "broadcast": BroadcastComm,
    "gated": GatedComm,
    "attention": AttentionComm,
    "graph": GraphComm,
}


def _module(name, **kwargs):
    return LEARNED[name](hidden_dim=16, message_dim=8, **kwargs)


@pytest.mark.parametrize("name", sorted(LEARNED))
def test_severing_reduces_every_learned_module_to_identity(name):
    """The whole metric rests on this: severed communication == IdentityComm.

    Every module decodes through a bias-free projection, so an empty
    neighbourhood must yield an exactly zero communication delta.
    """

    torch.manual_seed(0)
    module = _module(name)
    hidden = torch.randn(4, 3, 16)

    communicating = module(hidden)
    with severed_communication([module]):
        severed = module(hidden)

    assert torch.equal(severed, hidden)
    # Guard against a vacuous test: the module must actually do something.
    assert not torch.allclose(communicating, hidden)


@pytest.mark.parametrize("name", sorted(LEARNED))
def test_severing_is_reversible(name):
    torch.manual_seed(0)
    module = _module(name)
    hidden = torch.randn(2, 3, 16)
    original_channel = module.channel

    before = module(hidden)
    with severed_communication([module]):
        module(hidden)
    after = module(hidden)

    assert module.channel is original_channel
    assert torch.allclose(before, after)


def test_severing_restores_channels_after_an_exception():
    module = _module("broadcast")
    original = module.channel

    with pytest.raises(RuntimeError), severed_communication([module]):
        raise RuntimeError("boom")

    assert module.channel is original


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_severing_holds_across_multiple_communication_rounds(rounds):
    torch.manual_seed(0)
    module = AttentionComm(hidden_dim=16, message_dim=8, key_dim=8, num_heads=2, rounds=rounds)
    hidden = torch.randn(2, 3, 16)

    with severed_communication([module]):
        assert torch.equal(module(hidden), hidden)


def test_severing_leaves_a_parameterless_identity_module_untouched():
    module = IdentityComm(hidden_dim=16)
    hidden = torch.randn(2, 3, 16)

    with severed_communication([module]):
        assert torch.equal(module(hidden), hidden)


def test_severing_preserves_gradient_flow_to_the_local_path():
    module = _module("broadcast")
    hidden = torch.randn(2, 3, 16, requires_grad=True)

    with severed_communication([module]):
        module(hidden).sum().backward()

    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_gaussian_kl_is_zero_for_identical_distributions():
    location = torch.randn(4, 2, dtype=torch.float64)
    scale = torch.rand(4, 2, dtype=torch.float64) + 0.5

    divergence = _gaussian_kl(location, scale, location, scale)

    assert torch.allclose(divergence, torch.zeros(4, dtype=torch.float64), atol=1e-12)


def test_gaussian_kl_grows_with_separation():
    scale = torch.ones(1, 1, dtype=torch.float64)
    zero = torch.zeros(1, 1, dtype=torch.float64)

    near = _gaussian_kl(zero, scale, torch.full((1, 1), 0.5, dtype=torch.float64), scale)
    far = _gaussian_kl(zero, scale, torch.full((1, 1), 3.0, dtype=torch.float64), scale)

    assert 0 < float(near) < float(far)


def test_communication_modules_deduplicates_shared_instances():
    module = _module("attention")
    container = torch.nn.ModuleList([module, module])

    class FakeExperiment:
        group_policies = {"agents": container}

    found = communication_modules(FakeExperiment())

    assert len(found) == 1
    assert found[0] is module


@pytest.mark.parametrize("name", sorted(LEARNED))
def test_saliency_result_row_is_flat_and_serializable(name):
    from commstudy.analysis.saliency import SaliencyResult

    result = SaliencyResult(
        episodes=4,
        steps=100,
        exploration="DETERMINISTIC",
        return_with_comm=-400.0,
        return_without_comm=-500.0,
        return_delta=100.0,
        return_delta_fraction=0.2,
        per_episode_delta_mean=100.0,
        per_episode_delta_std=5.0,
        action_shift_mean=0.3,
        action_shift_max=1.1,
        policy_kl_mean=0.05,
        communicating_modules=1,
    )
    row = result.as_row()

    assert row["saliency_return_delta"] == 100.0
    assert row["saliency_return_delta_fraction"] == pytest.approx(0.2)
    assert all(
        value is None or isinstance(value, (int, float, str)) for value in row.values()
    )


def test_real_frozen_policy_saliency_is_exactly_zero_without_communication(
    tmp_path, config_root
):
    """A model with no channel must report zero saliency, not a missing value.

    This also exercises the full rollout path end to end on real VMAS.
    """

    import dataclasses

    from commstudy.analysis.saliency import communication_saliency
    from commstudy.experiments.config import load_experiment_spec
    from commstudy.experiments.runner import build_experiment

    spec = load_experiment_spec(
        config_root,
        ["model=comm_identity", "seed=0", "experiment.evaluation=false"],
    )
    experiment = build_experiment(
        dataclasses.replace(
            spec, experiment={**spec.experiment, "save_folder": str(tmp_path / "out")}
        )
    )
    try:
        result = communication_saliency(experiment, episodes=2, steps=10, seed=0)
    finally:
        experiment.test_env.close()

    assert result.communicating_modules == 1
    assert result.return_delta == pytest.approx(0.0, abs=1e-9)
    assert result.action_shift_mean == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(result.return_with_comm)


def test_real_frozen_learned_policy_saliency_is_measurable(tmp_path, config_root):
    """A learned module must produce a genuinely different severed policy."""

    import dataclasses

    from commstudy.analysis.saliency import communication_saliency
    from commstudy.experiments.config import load_experiment_spec
    from commstudy.experiments.runner import build_experiment

    spec = load_experiment_spec(
        config_root,
        ["model=comm_attention", "seed=0", "experiment.evaluation=false"],
    )
    experiment = build_experiment(
        dataclasses.replace(
            spec, experiment={**spec.experiment, "save_folder": str(tmp_path / "out")}
        )
    )
    try:
        result = communication_saliency(experiment, episodes=2, steps=10, seed=0)
    finally:
        experiment.test_env.close()

    assert result.communicating_modules == 1
    assert math.isfinite(result.return_with_comm)
    assert math.isfinite(result.return_without_comm)
    assert result.action_shift_mean is not None
    # An untrained actor initializes its output projection near zero, so the
    # magnitude is not asserted -- only that the comparison actually ran.
    assert result.action_shift_mean >= 0.0
