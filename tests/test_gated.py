import pytest
import torch

from commstudy.communication.base import CommContext
from commstudy.communication.gated import GatedComm


N_AGENTS = 3
HIDDEN_DIM = 8
MESSAGE_DIM = 4


def test_gated_rejects_unknown_configuration_options():
    with pytest.raises(TypeError, match="gate_threshhold"):
        GatedComm(HIDDEN_DIM, gate_threshhold=0.9)


@pytest.mark.parametrize("leading_shape", [(), (5,), (2, 4), (2, 1, 4)])
def test_gated_preserves_arbitrary_leading_shape(leading_shape):
    module = GatedComm(HIDDEN_DIM, MESSAGE_DIM)
    h = torch.randn(*leading_shape, N_AGENTS, HIDDEN_DIM)

    output = module(h)

    assert output.shape == h.shape
    assert torch.isfinite(output).all()


def test_gated_preserves_input_device():
    module = GatedComm(HIDDEN_DIM, MESSAGE_DIM).to("cpu")
    h = torch.randn(2, N_AGENTS, HIDDEN_DIM, device="cpu")

    output = module(h)

    assert h.device.type == "cpu"
    assert output.device == h.device


def _identity_projection_module(*, hard=False, residual=False):
    module = GatedComm(
        2,
        message_dim=2,
        residual=residual,
        hard=hard,
    )
    with torch.no_grad():
        module.message_encoder.weight.copy_(torch.eye(2))
        module.message_decoder.weight.copy_(torch.eye(2))
        module.gate_network.weight.zero_()
    return module


def test_zero_gate_suppresses_all_communication():
    module = _identity_projection_module(hard=True)
    with torch.no_grad():
        module.gate_network.bias.fill_(-100.0)

    output = module(torch.randn(N_AGENTS, 2))

    assert torch.equal(output, torch.zeros_like(output))
    stats = module.communication_stats()
    # Soft/hard gate activity is diagnostic; nominal channel cost conservatively
    # counts available payload slots separately.
    assert stats["active_sender_fraction"] == pytest.approx(1.0)
    assert stats["active_gate_fraction"] == pytest.approx(0.0)
    assert stats["realized_communication_rate"] == pytest.approx(1.0)


def test_one_gate_allows_manual_mean_communication():
    module = _identity_projection_module(hard=True)
    with torch.no_grad():
        module.gate_network.bias.fill_(100.0)

    h = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    output = module(h)
    encoded = torch.tanh(h)
    expected = torch.stack(
        [
            encoded[[1, 2]].mean(dim=0),
            encoded[[0, 2]].mean(dim=0),
            encoded[[0, 1]].mean(dim=0),
        ]
    )

    assert torch.allclose(output, expected)


def test_normal_gated_communication_does_not_modify_input():
    module = _identity_projection_module(hard=True, residual=True)
    with torch.no_grad():
        module.gate_network.bias.fill_(100.0)
    h = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    original = h.clone()

    output = module(h)

    assert torch.equal(h, original)
    assert output.data_ptr() != h.data_ptr()
    assert not torch.equal(output, original)


def test_straight_through_hard_gate_has_gate_network_gradients():
    module = GatedComm(HIDDEN_DIM, MESSAGE_DIM, hard=True)
    h = torch.randn(5, N_AGENTS, HIDDEN_DIM, requires_grad=True)

    module(h).square().mean().backward()

    assert module.gate_network.weight.grad is not None
    assert torch.count_nonzero(module.gate_network.weight.grad) > 0


def test_soft_gate_gradients_reach_all_communication_parameters():
    module = GatedComm(HIDDEN_DIM, MESSAGE_DIM, hard=False)
    h = torch.randn(5, N_AGENTS, HIDDEN_DIM, requires_grad=True)

    module(h).square().mean().backward()

    assert torch.count_nonzero(module.message_encoder.weight.grad) > 0
    assert torch.count_nonzero(module.gate_network.weight.grad) > 0
    assert torch.count_nonzero(module.message_decoder.weight.grad) > 0


def test_explicit_sender_mask_and_context_edges_are_both_enforced():
    module = _identity_projection_module(hard=True)
    with torch.no_grad():
        module.gate_network.bias.fill_(100.0)

    h = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    edges = torch.zeros(3, 3, dtype=torch.bool)
    edges[0, 1] = True
    context = CommContext(
        mask=edges,
        extras={"sender_mask": torch.tensor([True, False, True])},
    )

    output = module(h, context)

    # The sole permitted sender is explicitly unavailable.
    assert torch.equal(output, torch.zeros_like(output))


def test_learned_sender_budget_limits_active_transmitters():
    module = GatedComm(
        HIDDEN_DIM,
        MESSAGE_DIM,
        hard=True,
        sender_budget=1,
    )
    with torch.no_grad():
        module.gate_network.weight.zero_()
        module.gate_network.bias.fill_(100.0)

    module(torch.randn(4, N_AGENTS, HIDDEN_DIM))
    stats = module.communication_stats()

    assert stats["sender_budget"] == 1.0
    assert stats["effective_active_senders"] == pytest.approx(1.0)
    assert stats["active_sender_fraction"] == pytest.approx(1 / N_AGENTS)


def test_random_sender_budget_uses_stable_seeded_priorities():
    module = GatedComm(
        HIDDEN_DIM,
        MESSAGE_DIM,
        sender_budget=1,
        sender_selection="random",
        sender_selection_seed=17,
    )
    available = torch.ones(4, N_AGENTS, dtype=torch.bool)
    logits_a = torch.randn(4, N_AGENTS)
    logits_b = torch.randn(4, N_AGENTS) * 100

    selected_a = module._budget_mask(logits_a, available)
    selected_b = module._budget_mask(logits_b, available)

    assert torch.equal(selected_a, selected_b)
    assert torch.equal(selected_a[0], selected_a[1])
    assert selected_a.sum(dim=-1).tolist() == [1, 1, 1, 1]

    module(torch.randn(2, N_AGENTS, HIDDEN_DIM))
    stats = module.communication_stats()
    assert stats["sender_selection_random"] == 1.0
    assert stats["sender_selection_seed"] == 17.0


def test_no_edges_produce_exact_zero_delta_with_residual():
    module = GatedComm(HIDDEN_DIM, MESSAGE_DIM, residual=True)
    h = torch.randn(4, N_AGENTS, HIDDEN_DIM)
    edges = torch.zeros(4, N_AGENTS, N_AGENTS, dtype=torch.bool)

    output = module(h, CommContext(mask=edges))

    assert torch.equal(output, h)


def test_gated_budget_uses_only_senders_with_a_permitted_receiver():
    module = GatedComm(HIDDEN_DIM, MESSAGE_DIM, sender_budget=1)
    with torch.no_grad():
        module.gate_network.weight.zero_()
        module.gate_network.bias.zero_()
    h = torch.randn(2, N_AGENTS, HIDDEN_DIM)
    edges = torch.zeros(N_AGENTS, N_AGENTS, dtype=torch.bool)
    edges[0, 2] = True

    module(h, CommContext(mask=edges))
    stats = module.communication_stats()

    assert stats["active_edges_per_step"] == pytest.approx(1.0)
    assert stats["messages_per_step"] == pytest.approx(1.0)


def test_gated_reports_activity_and_nominal_cost():
    module = GatedComm(HIDDEN_DIM, MESSAGE_DIM, hard=True)
    with torch.no_grad():
        module.gate_network.weight.zero_()
        module.gate_network.bias.fill_(100.0)

    module(torch.randn(2, N_AGENTS, HIDDEN_DIM))
    stats = module.communication_stats()

    assert stats["message_bits_per_sender"] == MESSAGE_DIM * 32
    assert stats["mean_gate"] == pytest.approx(1.0)
    assert stats["active_gate_fraction"] == pytest.approx(1.0)
    assert stats["active_edge_fraction"] == pytest.approx(1.0)


def test_gated_raw_gate_metrics_are_not_erased_by_channel_dropout():
    module = GatedComm(
        HIDDEN_DIM,
        MESSAGE_DIM,
        hard=True,
        channel={"type": "dropout", "p": 1.0, "mode": "always"},
    )
    with torch.no_grad():
        module.gate_network.weight.zero_()
        module.gate_network.bias.fill_(100.0)

    module(torch.randn(2, N_AGENTS, HIDDEN_DIM))
    stats = module.communication_stats()

    assert stats["mean_gate"] == pytest.approx(1.0)
    assert stats["active_gate_fraction"] == pytest.approx(1.0)
    assert stats["effective_mean_gate"] == pytest.approx(0.0)
    assert stats["effective_active_gate_fraction"] == pytest.approx(0.0)


def test_gated_supports_matching_float64_module_dtype():
    module = GatedComm(HIDDEN_DIM, MESSAGE_DIM).double()
    h = torch.randn(2, N_AGENTS, HIDDEN_DIM, dtype=torch.float64)

    assert module(h).dtype == torch.float64
