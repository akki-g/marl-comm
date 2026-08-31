from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from commstudy.communication.attention import AttentionComm
from commstudy.communication.base import CommContext
from commstudy.experiments import build_model_config
from commstudy.models import CommPolicyConfig


def _attention(
    *,
    hidden_dim: int = 16,
    store_debug_attention: bool = True,
    **kwargs,
) -> AttentionComm:
    return AttentionComm(
        hidden_dim=hidden_dim,
        message_dim=8,
        key_dim=8,
        num_heads=2,
        store_debug_attention=store_debug_attention,
        **kwargs,
    )


def _zero_attention_scores(module: AttentionComm) -> None:
    with torch.no_grad():
        module.query_projection.weight.zero_()
        module.key_projection.weight.zero_()


@pytest.mark.parametrize(
    "shape",
    [
        (3, 16),
        (5, 3, 16),
        (4, 5, 3, 16),
        (2, 4, 5, 3, 16),
    ],
)
def test_attention_preserves_arbitrary_leading_dimensions_and_gradients(shape):
    module = _attention(store_debug_attention=False)
    h = torch.randn(*shape, requires_grad=True)
    original = h.detach().clone()

    output = module(h)
    output.square().mean().backward()

    assert output.shape == h.shape
    assert torch.equal(h.detach(), original)
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_attention_preserves_input_device():
    module = _attention(store_debug_attention=False).to("cpu")
    h = torch.randn(2, 3, 16, device="cpu")

    output = module(h)

    assert h.device.type == "cpu"
    assert output.device == h.device


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"message_dim": 7}, "message_dim"),
        ({"key_dim": 7}, "key_dim"),
        ({"num_heads": 0}, "num_heads"),
        ({"rounds": 0}, "rounds"),
        ({"rounds": 4}, "rounds"),
        ({"temperature": 0.0}, "temperature"),
        ({"sender_budget": -1}, "sender_budget"),
        ({"sender_selection": "unknown"}, "sender_selection"),
    ],
)
def test_attention_rejects_invalid_configuration(kwargs, message):
    defaults = {"hidden_dim": 16, "message_dim": 8, "key_dim": 8, "num_heads": 2}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        AttentionComm(**defaults)


def test_attention_default_excludes_self_and_normalizes_valid_rows():
    module = _attention()
    _zero_attention_scores(module)

    module(torch.randn(2, 3, 16))
    attention = module.debug_attention

    assert attention is not None
    assert attention.shape == (2, 3, 3, 2)
    for agent in range(3):
        assert torch.count_nonzero(attention[:, agent, agent, :]) == 0
    assert torch.allclose(attention.sum(dim=-2), torch.ones(2, 3, 2))


def test_attention_all_masked_rows_are_exact_local_residual_and_finite():
    module = _attention(rounds=3)
    h = torch.randn(2, 3, 16, requires_grad=True)
    mask = torch.zeros(3, 3, dtype=torch.bool)

    output = module(h, CommContext(mask=mask))
    output.sum().backward()

    assert torch.equal(output, h)
    assert module.debug_attention is not None
    assert torch.count_nonzero(module.debug_attention) == 0
    assert torch.isfinite(h.grad).all()


def test_attention_context_mask_takes_precedence_and_broadcasts():
    module = _attention()
    _zero_attention_scores(module)
    h = torch.randn(2, 4, 3, 16)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 1] = True
    original_mask = mask.clone()

    module(h, CommContext(mask=mask))
    attention = module.debug_attention

    assert attention is not None
    assert torch.equal(mask, original_mask)
    assert torch.allclose(attention[..., 0, 1, :], torch.ones(2, 4, 2))
    assert torch.count_nonzero(attention[..., 0, 0, :]) == 0
    assert torch.count_nonzero(attention[..., 1:, :, :]) == 0


def test_attention_explicit_sender_availability_masks_logits():
    module = _attention()
    _zero_attention_scores(module)
    sender_mask = torch.tensor([True, False, True])

    module(
        torch.randn(2, 3, 16),
        CommContext(extras={"sender_mask": sender_mask}),
    )
    attention = module.debug_attention

    assert attention is not None
    assert torch.count_nonzero(attention[..., 1, :]) == 0
    assert torch.allclose(attention[:, 1].sum(dim=-2), torch.ones(2, 2))


def test_explicit_sender_mask_overrides_dropout_realization():
    module = _attention(channel={"type": "dropout", "p": 1.0, "mode": "always"})
    _zero_attention_scores(module)
    h = torch.randn(2, 3, 16)

    without_override = module(h)
    assert torch.equal(without_override, h)

    module(h, CommContext(extras={"sender_mask": torch.ones(3, dtype=torch.bool)}))
    assert module.debug_attention is not None
    assert torch.allclose(module.debug_attention.sum(dim=-2), torch.ones(2, 3, 2))


def test_attention_budget_limits_whole_senders_and_random_budget_is_stable():
    module = _attention(
        sender_budget=1,
        sender_selection="random",
        sender_selection_seed=17,
    )
    _zero_attention_scores(module)
    h = torch.randn(2, 4, 16)

    module(h)
    first = module.debug_attention.clone()
    module(h)
    second = module.debug_attention

    assert second is not None
    assert torch.equal(first > 0, second > 0)
    active_senders = (second > 0).any(dim=(-3, -1))
    assert (active_senders.sum(dim=-1) <= 1).all()
    stats = module.communication_stats()
    assert stats["realized_communication_rate"] == pytest.approx(1.0)
    assert stats["active_sender_fraction"] < 1.0


def test_attention_learned_k1_scheduler_has_qk_gradients():
    module = _attention(
        sender_budget=1,
        sender_selection="attention",
        store_debug_attention=False,
    )
    output = module(torch.randn(4, 3, 16))

    output.square().mean().backward()

    assert module.query_projection.weight.grad is not None
    assert module.key_projection.weight.grad is not None
    assert torch.count_nonzero(module.query_projection.weight.grad) > 0
    assert torch.count_nonzero(module.key_projection.weight.grad) > 0


def test_attention_multiround_dropout_replays_one_exact_step_mask():
    module = _attention(
        rounds=3,
        channel={"type": "dropout", "p": 0.5, "mode": "always"},
        store_debug_attention=False,
    )
    h = torch.randn(5, 3, 16)

    torch.manual_seed(7)
    sampled_output = module(h)
    sampled_mask = module.channel.last_sender_mask.clone()
    torch.manual_seed(999)
    replayed_output = module(
        h,
        CommContext(extras={"sender_mask": sampled_mask}),
    )

    assert torch.equal(replayed_output, sampled_output)


def test_attention_reports_detached_scalar_stats_and_key_value_payload():
    module = _attention(rounds=2)
    module(torch.randn(2, 3, 16, requires_grad=True))

    stats = module.communication_stats()

    assert module.message_bits() == 2 * (8 + 8) * 32
    assert stats["message_bits_per_sender"] == module.message_bits()
    assert stats["communication_rounds"] == 2
    assert stats["messages_per_step"] == pytest.approx(6.0)
    assert stats["active_edges_per_step"] == pytest.approx(12.0)
    assert stats["realized_bits_per_step"] == pytest.approx(
        12 * (8 + 8) * 32
    )
    for key in (
        "active_sender_fraction",
        "active_edge_fraction",
        "mean_message_norm",
        "attention_entropy",
        "attention_max_probability",
        "effective_neighbor_count",
    ):
        assert isinstance(stats[key], float)
        assert torch.isfinite(torch.tensor(stats[key]))
    assert module.debug_attention is not None
    assert not module.debug_attention.requires_grad
    assert module.debug_attention.device.type == "cpu"


def test_attention_half_precision_statistics_remain_finite():
    module = _attention(store_debug_attention=False).half()
    module(torch.randn(2, 3, 16, dtype=torch.float16))

    stats = module.communication_stats()

    assert torch.isfinite(torch.tensor(stats["attention_entropy"]))
    assert torch.isfinite(torch.tensor(stats["effective_neighbor_count"]))


def test_attention_rounds_share_parameters():
    one_round = _attention(rounds=1, store_debug_attention=False)
    three_rounds = _attention(rounds=3, store_debug_attention=False)

    assert sum(parameter.numel() for parameter in one_round.parameters()) == sum(
        parameter.numel() for parameter in three_rounds.parameters()
    )


def test_attention_yaml_builds_the_expected_module(config_root):
    document = OmegaConf.to_container(
        OmegaConf.load(config_root / "models" / "comm_attention.yaml"),
        resolve=True,
    )
    config = build_model_config(document)
    module = AttentionComm(hidden_dim=config.hidden_dim, **config.comm_kwargs)

    assert isinstance(config, CommPolicyConfig)
    assert config.comm_class_path.endswith("AttentionComm")
    assert module.message_dim == 32
    assert module.key_dim == 32
    assert module.num_heads == 4
    assert module.rounds == 1
