from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from commstudy.communication.base import CommContext
from commstudy.communication.graph import GraphComm
from commstudy.experiments import build_model_config
from commstudy.models import CommPolicyConfig


def _graph(
    *,
    hidden_dim: int = 16,
    store_debug_attention: bool = True,
    **kwargs,
) -> GraphComm:
    return GraphComm(
        hidden_dim=hidden_dim,
        message_dim=8,
        key_dim=8,
        num_heads=2,
        store_debug_attention=store_debug_attention,
        **kwargs,
    )


def _zero_relation_scores(module: GraphComm) -> None:
    with torch.no_grad():
        module.receiver_projection.weight.zero_()
        module.sender_projection.weight.zero_()
        module.relation_vector.zero_()


@pytest.mark.parametrize(
    "shape",
    [
        (3, 16),
        (5, 3, 16),
        (4, 5, 3, 16),
        (2, 4, 5, 3, 16),
    ],
)
def test_graph_preserves_arbitrary_leading_dimensions_and_gradients(shape):
    module = _graph(store_debug_attention=False)
    h = torch.randn(*shape, requires_grad=True)
    original = h.detach().clone()

    output = module(h)
    output.square().mean().backward()

    assert output.shape == h.shape
    assert torch.equal(h.detach(), original)
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert module.relation_vector.grad is not None
    assert torch.isfinite(module.relation_vector.grad).all()


def test_graph_preserves_input_device():
    module = _graph(store_debug_attention=False).to("cpu")
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
        ({"negative_slope": -0.1}, "negative_slope"),
        ({"topology": "unknown"}, "topology"),
        ({"erdos_renyi_p": 1.1}, "erdos_renyi_p"),
        ({"sender_budget": -1}, "sender_budget"),
    ],
)
def test_graph_rejects_invalid_configuration(kwargs, message):
    defaults = {"hidden_dim": 16, "message_dim": 8, "key_dim": 8, "num_heads": 2}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        GraphComm(**defaults)


def test_graph_full_default_excludes_self_and_normalizes():
    module = _graph()
    _zero_relation_scores(module)

    module(torch.randn(2, 3, 16))
    attention = module.debug_attention

    assert attention is not None
    for agent in range(3):
        assert torch.count_nonzero(attention[:, agent, agent, :]) == 0
    assert torch.allclose(attention.sum(dim=-2), torch.ones(2, 3, 2))


def test_graph_directed_ring_has_one_predecessor_sender_per_receiver():
    module = _graph(topology="directed_ring")
    _zero_relation_scores(module)
    module(torch.randn(2, 4, 16))
    attention = module.debug_attention

    assert attention is not None
    for receiver in range(4):
        sender = (receiver - 1) % 4
        assert torch.allclose(attention[:, receiver, sender, :], torch.ones(2, 2))
        assert (attention[:, receiver, :, :] > 0).sum(dim=-2).eq(1).all()


def test_graph_runtime_context_mask_precedes_fallback_topology():
    module = _graph(topology="directed_ring", rounds=3)
    h = torch.randn(2, 4, 16, requires_grad=True)
    no_edges = torch.zeros(4, 4, dtype=torch.bool)

    output = module(h, CommContext(mask=no_edges))
    output.sum().backward()

    assert torch.equal(output, h)
    assert module.debug_attention is not None
    assert torch.count_nonzero(module.debug_attention) == 0
    assert torch.isfinite(h.grad).all()


def test_graph_seeded_erdos_renyi_topology_is_static_and_reproducible():
    first_module = _graph(topology="erdos_renyi", erdos_renyi_p=0.5, topology_seed=9)
    same_module = _graph(topology="erdos_renyi", erdos_renyi_p=0.5, topology_seed=9)
    other_module = _graph(topology="erdos_renyi", erdos_renyi_p=0.5, topology_seed=10)
    for module in (first_module, same_module, other_module):
        _zero_relation_scores(module)
    h = torch.randn(2, 8, 16)

    first_module(h)
    first = first_module.debug_attention > 0
    first_module(h)
    repeated = first_module.debug_attention > 0
    same_module(h)
    same = same_module.debug_attention > 0
    other_module(h)
    other = other_module.debug_attention > 0

    assert torch.equal(first, repeated)
    assert torch.equal(first, same)
    assert not torch.equal(first, other)


def test_graph_context_mask_direction_severs_unlisted_sender():
    module = _graph()
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 1] = True
    h = torch.randn(2, 3, 16)
    perturbed = h.clone()
    perturbed[:, 2] += 100.0

    first = module(h, CommContext(mask=mask))
    second = module(perturbed, CommContext(mask=mask))

    assert torch.equal(first[:, 0], second[:, 0])


def test_graph_explicit_sender_availability_masks_relation_logits():
    module = _graph()
    _zero_relation_scores(module)
    module(
        torch.randn(2, 3, 16),
        CommContext(extras={"sender_mask": torch.tensor([True, False, True])}),
    )

    assert module.debug_attention is not None
    assert torch.count_nonzero(module.debug_attention[..., 1, :]) == 0


def test_graph_can_explicitly_include_self_on_full_fallback():
    module = _graph(exclude_self=False)
    _zero_relation_scores(module)
    module(torch.randn(2, 3, 16))

    assert module.debug_attention is not None
    for agent in range(3):
        assert (module.debug_attention[:, agent, agent, :] > 0).all()


def test_graph_sender_budget_limits_whole_senders():
    module = _graph(
        sender_budget=1,
        sender_selection="random",
        sender_selection_seed=21,
    )
    _zero_relation_scores(module)
    module(torch.randn(2, 4, 16))

    assert module.debug_attention is not None
    active_senders = (module.debug_attention > 0).any(dim=(-3, -1))
    assert (active_senders.sum(dim=-1) <= 1).all()
    stats = module.communication_stats()
    assert stats["realized_communication_rate"] == pytest.approx(1.0)
    assert stats["active_sender_fraction"] < 1.0


def test_graph_learned_k1_scheduler_has_relation_gradients():
    module = _graph(
        sender_budget=1,
        sender_selection="attention",
        store_debug_attention=False,
    )
    output = module(torch.randn(4, 3, 16))

    output.square().mean().backward()

    assert module.receiver_projection.weight.grad is not None
    assert module.sender_projection.weight.grad is not None
    assert torch.count_nonzero(module.receiver_projection.weight.grad) > 0
    assert torch.count_nonzero(module.sender_projection.weight.grad) > 0


def test_graph_reports_topology_stats_and_key_value_payload():
    module = _graph(topology="directed_ring", rounds=2)
    module(torch.randn(2, 4, 16))
    stats = module.communication_stats()

    assert module.message_bits() == 2 * (8 + 8) * 32
    assert stats["message_bits_per_sender"] == module.message_bits()
    assert stats["communication_rounds"] == 2
    assert stats["messages_per_step"] == pytest.approx(8.0)
    assert stats["active_edges_per_step"] == pytest.approx(8.0)
    assert stats["realized_bits_per_step"] == pytest.approx(
        8 * (8 + 8) * 32
    )
    assert stats["edge_density"] == pytest.approx(1 / 3)
    assert stats["topology_edge_density"] == pytest.approx(1 / 3)
    assert stats["mean_degree"] == pytest.approx(1.0)
    for value in stats.values():
        assert isinstance(value, (float, int))


def test_graph_half_precision_statistics_remain_finite():
    module = _graph(store_debug_attention=False).half()
    module(torch.randn(2, 3, 16, dtype=torch.float16))

    stats = module.communication_stats()

    assert torch.isfinite(torch.tensor(stats["attention_entropy"]))
    assert torch.isfinite(torch.tensor(stats["effective_neighbor_count"]))


def test_graph_rounds_share_parameters():
    one_round = _graph(rounds=1, store_debug_attention=False)
    three_rounds = _graph(rounds=3, store_debug_attention=False)

    assert sum(parameter.numel() for parameter in one_round.parameters()) == sum(
        parameter.numel() for parameter in three_rounds.parameters()
    )


def test_graph_yaml_builds_the_expected_module(config_root):
    document = OmegaConf.to_container(
        OmegaConf.load(config_root / "models" / "comm_graph.yaml"),
        resolve=True,
    )
    config = build_model_config(document)
    module = GraphComm(hidden_dim=config.hidden_dim, **config.comm_kwargs)

    assert isinstance(config, CommPolicyConfig)
    assert config.comm_class_path.endswith("GraphComm")
    assert module.message_dim == 32
    assert module.key_dim == 32
    assert module.num_heads == 4
    assert module.rounds == 1
    assert module.topology == "full"
