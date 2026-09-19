import pytest
import torch
from tensordict.nn.probabilistic import (
    InteractionType,
    set_interaction_type,
)

from commstudy.communication.channel import (
    DropoutChannel,
    GaussianNoiseChannel,
    IdentityChannel,
    QuantizedChannel,
    SequentialChannel,
    build_channel,
)


def messages():
    return torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)


def test_identity_channel_enforces_explicit_sender_mask():
    value = messages()
    sender_mask = torch.tensor([True, False, True])

    result = IdentityChannel()(value, sender_mask=sender_mask)

    assert result.sender_mask.shape == (2, 3)
    assert torch.equal(result.messages[:, 1], torch.zeros(2, 4))
    assert torch.equal(result.messages[:, 0], value[:, 0])


@pytest.mark.parametrize(
    "value",
    [
        torch.ones(4),
        torch.ones(2, 3, 4, dtype=torch.int64),
    ],
    ids=["one_dimensional", "integer"],
)
def test_channel_rejects_invalid_message_payloads(value):
    expected = "shaped" if value.dim() == 1 else "real floating dtype"

    with pytest.raises((TypeError, ValueError), match=expected):
        IdentityChannel()(value)


@pytest.mark.parametrize(
    "sender_mask",
    [
        torch.ones(3, dtype=torch.int64),
        torch.tensor([1.0, float("nan"), 0.0]),
    ],
    ids=["integer", "nan_float"],
)
def test_channel_rejects_numeric_and_nan_sender_masks(sender_mask):
    with pytest.raises(TypeError, match="sender_mask must be boolean"):
        IdentityChannel()(messages(), sender_mask=sender_mask)


def test_last_sender_mask_is_isolated_from_forward_output_mutation():
    channel = IdentityChannel()
    expected = torch.tensor([[True, False, True], [False, True, False]])

    result = channel(messages(), sender_mask=expected)
    result.sender_mask.zero_()
    expected.zero_()

    assert torch.equal(
        channel.last_sender_mask,
        torch.tensor([[True, False, True], [False, True, False]]),
    )


def test_dropout_is_whole_sender_and_evaluation_only_by_default():
    channel = DropoutChannel(p=1.0)
    value = messages()

    channel.train()
    training = channel(value)
    channel.eval()
    evaluation = channel(value)

    assert training.sender_mask.all()
    assert torch.equal(training.messages, value)
    assert not evaluation.sender_mask.any()
    assert torch.equal(evaluation.messages, torch.zeros_like(value))


def test_explicit_sender_realization_overrides_random_dropout():
    channel = DropoutChannel(p=1.0, mode="always")
    explicit = torch.tensor([[True, False, True], [False, True, False]])

    result = channel(messages(), sender_mask=explicit)

    assert torch.equal(result.sender_mask, explicit)
    assert torch.equal(channel.last_sender_mask, explicit)
    assert channel.last_sender_mask.grad_fn is None


@pytest.mark.parametrize(
    ("mode", "training", "expected_drop"),
    [
        ("always", True, True),
        ("training", True, True),
        ("training", False, False),
        ("evaluation", True, False),
        ("evaluation", False, True),
        ("disabled", False, False),
    ],
)
def test_dropout_modes_are_explicit(mode, training, expected_drop):
    channel = DropoutChannel(p=1.0, mode=mode)
    channel.train(training)

    result = channel(messages())

    assert (not result.sender_mask.any()) is expected_drop


def test_evaluation_mode_recognizes_benchmarl_no_grad_lifecycle():
    evaluation_channel = DropoutChannel(p=1.0, mode="evaluation")
    training_channel = DropoutChannel(p=1.0, mode="training")
    evaluation_channel.train()
    training_channel.train()

    with (
        torch.no_grad(),
        set_interaction_type(InteractionType.DETERMINISTIC),
    ):
        evaluation_result = evaluation_channel(messages())
        training_result = training_channel(messages())

    assert not evaluation_result.sender_mask.any()
    assert training_result.sender_mask.all()


def test_gaussian_noise_cannot_resurrect_explicitly_dropped_sender():
    channel = GaussianNoiseChannel(std=1.0, mode="always")
    explicit = torch.tensor([True, False, True])

    result = channel(torch.zeros(2, 3, 4), sender_mask=explicit)

    assert torch.equal(result.messages[:, 1], torch.zeros(2, 4))
    assert not torch.equal(result.messages[:, 0], torch.zeros(2, 4))


def test_train_time_gaussian_noise_declares_replay_hazard():
    channel = GaussianNoiseChannel(std=0.5, mode="always")

    with pytest.raises(ValueError, match="not replay-safe"):
        channel.validate_policy_replay_safety()

    GaussianNoiseChannel(
        std=0.5,
        mode="evaluation",
    ).validate_policy_replay_safety()


@pytest.mark.parametrize("std", [float("nan"), float("inf"), float("-inf")])
def test_gaussian_noise_rejects_non_finite_standard_deviation(std):
    with pytest.raises(ValueError, match="must be finite"):
        GaussianNoiseChannel(std=std)


@pytest.mark.parametrize("levels", [3.5, True])
def test_quantized_channel_rejects_non_integer_levels(levels):
    with pytest.raises(ValueError, match="integer >= 2"):
        QuantizedChannel(levels=levels)


def test_quantized_channel_uses_straight_through_gradients():
    value = torch.tensor([[[-0.7, -0.2, 0.3, 0.8]]], requires_grad=True)
    channel = QuantizedChannel(
        levels=3,
        clip_value=1.0,
        straight_through=True,
        mode="always",
    )

    output = channel(value).messages
    output.sum().backward()

    assert set(output.detach().flatten().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.equal(value.grad, torch.ones_like(value))


def test_build_channel_supports_serializable_sequences():
    channel = build_channel(
        [
            {"type": "dropout", "p": 1.0, "mode": "always"},
            {"type": "gaussian", "std": 1.0, "mode": "always"},
        ]
    )

    assert isinstance(channel, SequentialChannel)
    result = channel(messages())
    assert torch.equal(result.messages, torch.zeros_like(messages()))
    assert channel.requested_dropout_rate == pytest.approx(1.0)


@pytest.mark.parametrize("phase", ["collection", "optimization", "evaluation"])
@pytest.mark.parametrize("exploration", [InteractionType.RANDOM, InteractionType.DETERMINISTIC])
def test_declared_phase_controls_channel_independently_of_policy_sampling(phase, exploration):
    from commstudy.communication.channel import channel_phase

    channel = DropoutChannel(p=1, mode="evaluation").train()
    with torch.no_grad(), set_interaction_type(exploration), channel_phase(phase):
        result = channel(messages())
    assert bool(result.sender_mask.any()) == (phase != "evaluation")


@pytest.mark.parametrize("kind", ["identity", "dropout", "gaussian", "sequential"])
def test_dominant_intervention_overrides_replay_without_consuming_rng(kind):
    from commstudy.communication.channel import channel_intervention

    channel = {
        "identity": IdentityChannel(),
        "dropout": DropoutChannel(p=0.5, mode="always"),
        "gaussian": GaussianNoiseChannel(std=1, mode="always"),
        "sequential": build_channel([
            {"type": "dropout", "p": 0.5, "mode": "always"},
            {"type": "gaussian", "std": 1, "mode": "always"},
        ]),
    }[kind]
    value = messages()
    supplied = torch.ones(value.shape[:-1], dtype=torch.bool)
    before = torch.get_rng_state().clone()
    with channel_intervention([channel], False):
        result = channel(value, sender_mask=supplied)
        assert not result.sender_mask.any()
        assert torch.equal(result.messages, torch.zeros_like(value))
        assert channel._rng == {}
    assert torch.equal(torch.get_rng_state(), before)
    assert supplied.all()


def test_sender_intervention_intersects_replay_and_nested_context_restores():
    from commstudy.communication.channel import channel_intervention

    channel = IdentityChannel()
    value = messages()
    mask = torch.tensor([False, True, False])
    with channel_intervention([channel], mask):
        with pytest.raises(RuntimeError), channel_intervention([channel], False):
            raise RuntimeError("interrupted")
        result = channel(value, sender_mask=torch.ones(3, dtype=torch.bool))
        assert torch.equal(result.sender_mask, mask.expand(2, 3))
    assert channel(value).sender_mask.all()


def test_channel_stream_does_not_advance_policy_rng_and_evaluation_restores_it():
    from commstudy.communication.channel import preserve_channel_rng

    channel = DropoutChannel(p=0.5, mode="always")
    value = torch.ones(100, 3, 4)
    before = torch.get_rng_state().clone()
    channel(value)
    with preserve_channel_rng([channel]):
        expected = channel(value).sender_mask
    with preserve_channel_rng([channel], seed=987):
        channel(value)
    actual = channel(value).sender_mask
    assert torch.equal(expected, actual)
    assert torch.equal(torch.get_rng_state(), before)


def test_composed_dropout_stages_have_independent_streams():
    channel = build_channel([
        {"type": "dropout", "p": 0.5, "mode": "always"},
        {"type": "dropout", "p": 0.5, "mode": "always"},
    ])
    result = channel(torch.ones(10000, 3, 1))
    assert float(result.sender_mask.float().mean()) == pytest.approx(0.25, abs=0.02)


def test_severing_does_not_hide_invalid_replay_mask():
    from commstudy.communication.channel import channel_intervention

    channel = DropoutChannel(p=0.5, mode="always")
    with channel_intervention([channel], False):
        with pytest.raises(TypeError, match="boolean"):
            channel(messages(), sender_mask=torch.ones(8))
        with pytest.raises(ValueError, match="must end"):
            channel(messages(), sender_mask=torch.ones(8, dtype=torch.bool))


def test_rng_preservation_restores_channel_runtime_cache():
    from commstudy.communication.channel import preserve_channel_rng

    channel = DropoutChannel(p=0.5, mode="always")
    channel(torch.ones(3, 4))
    before = channel.last_sender_mask.clone()
    with preserve_channel_rng([channel], seed=22):
        channel(messages())
        assert channel.last_sender_mask.shape == (2, 3)
    assert torch.equal(channel.last_sender_mask, before)
