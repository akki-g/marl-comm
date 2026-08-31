import pytest
import torch

from commstudy.communication.base import CommContext
from commstudy.communication.broadcast import BroadcastComm


N_AGENTS = 3
HIDDEN_DIM = 8
MESSAGE_DIM = 4


def test_broadcast_rejects_unknown_configuration_options():
    with pytest.raises(TypeError, match="messsage_dim"):
        BroadcastComm(HIDDEN_DIM, messsage_dim=99)


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_broadcast_rejects_non_real_floating_embeddings(dtype):
    module = BroadcastComm(HIDDEN_DIM, MESSAGE_DIM)
    h = torch.ones(N_AGENTS, HIDDEN_DIM, dtype=dtype)

    with pytest.raises(TypeError, match="real floating-point embeddings"):
        module(h)


@pytest.mark.parametrize("leading_shape", [(), (5,), (2, 4), (2, 1, 4)])
def test_broadcast_preserves_arbitrary_leading_shape(leading_shape):
    module = BroadcastComm(HIDDEN_DIM, MESSAGE_DIM)
    h = torch.randn(*leading_shape, N_AGENTS, HIDDEN_DIM)

    output = module(h)

    assert output.shape == h.shape
    assert torch.isfinite(output).all()


def test_broadcast_preserves_input_device():
    module = BroadcastComm(HIDDEN_DIM, MESSAGE_DIM).to("cpu")
    h = torch.randn(2, N_AGENTS, HIDDEN_DIM, device="cpu")

    output = module(h)

    assert h.device.type == "cpu"
    assert output.device == h.device


def test_broadcast_excludes_self_and_matches_manual_neighbor_mean():
    module = BroadcastComm(2, message_dim=2, residual=False, exclude_self=True)
    with torch.no_grad():
        module.message_encoder.weight.copy_(torch.eye(2))
        module.message_decoder.weight.copy_(torch.eye(2))

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


def test_context_mask_takes_precedence_over_default_graph():
    module = BroadcastComm(2, message_dim=2, residual=False)
    with torch.no_grad():
        module.message_encoder.weight.copy_(torch.eye(2))
        module.message_decoder.weight.copy_(torch.eye(2))

    h = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 2] = True

    output = module(h, CommContext(mask=mask))

    assert torch.allclose(output[0], torch.tanh(h[2]))
    assert torch.equal(output[1:], torch.zeros(2, 2))


def test_explicit_sender_mask_removes_sender_from_every_receiver():
    module = BroadcastComm(2, message_dim=2, residual=False)
    with torch.no_grad():
        module.message_encoder.weight.copy_(torch.eye(2))
        module.message_decoder.weight.copy_(torch.eye(2))

    h = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    context = CommContext(
        extras={"sender_mask": torch.tensor([True, False, True])}
    )
    output = module(h, context)

    # Receiver 0 can now hear only sender 2. Receiver 2 hears only sender 0.
    assert torch.allclose(output[0], torch.tanh(h[2]))
    assert torch.allclose(output[2], torch.tanh(h[0]))


def test_no_edges_produce_exact_zero_delta_and_do_not_modify_input():
    module = BroadcastComm(HIDDEN_DIM, MESSAGE_DIM, residual=True)
    h = torch.randn(4, N_AGENTS, HIDDEN_DIM)
    original = h.clone()
    mask = torch.zeros(4, N_AGENTS, N_AGENTS, dtype=torch.bool)
    mask_original = mask.clone()

    output = module(h, CommContext(mask=mask))

    assert torch.equal(output, original)
    assert torch.equal(h, original)
    assert torch.equal(mask, mask_original)


def test_normal_broadcast_communication_does_not_modify_input():
    module = BroadcastComm(2, message_dim=2, residual=True)
    with torch.no_grad():
        module.message_encoder.weight.copy_(torch.eye(2))
        module.message_decoder.weight.copy_(torch.eye(2))
    h = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    original = h.clone()

    output = module(h)

    assert torch.equal(h, original)
    assert output.data_ptr() != h.data_ptr()
    assert not torch.equal(output, original)


def test_single_agent_default_is_safe_and_local():
    module = BroadcastComm(HIDDEN_DIM, MESSAGE_DIM)
    h = torch.randn(5, 1, HIDDEN_DIM)

    assert torch.equal(module(h), h)


def test_broadcast_gradients_reach_input_encoder_and_decoder():
    module = BroadcastComm(HIDDEN_DIM, MESSAGE_DIM)
    h = torch.randn(5, N_AGENTS, HIDDEN_DIM, requires_grad=True)

    module(h).square().mean().backward()

    assert h.grad is not None and torch.count_nonzero(h.grad) > 0
    assert torch.count_nonzero(module.message_encoder.weight.grad) > 0
    assert torch.count_nonzero(module.message_decoder.weight.grad) > 0


def test_broadcast_supports_matching_float64_module_dtype():
    module = BroadcastComm(HIDDEN_DIM, MESSAGE_DIM).double()
    h = torch.randn(2, N_AGENTS, HIDDEN_DIM, dtype=torch.float64)

    assert module(h).dtype == torch.float64


def test_broadcast_accumulates_detached_scalar_stats_and_resets():
    module = BroadcastComm(HIDDEN_DIM, MESSAGE_DIM)
    module(torch.randn(2, N_AGENTS, HIDDEN_DIM))
    module(torch.randn(2, N_AGENTS, HIDDEN_DIM))

    stats = module.communication_stats()

    assert stats["message_dim"] == MESSAGE_DIM
    assert stats["message_bits_per_sender"] == MESSAGE_DIM * 32
    assert stats["communication_rounds"] == 1
    assert stats["active_sender_fraction"] == pytest.approx(1.0)
    assert stats["messages_per_step"] == pytest.approx(3.0)
    assert stats["active_edges_per_step"] == pytest.approx(6.0)
    assert stats["realized_sender_bits_per_step"] == pytest.approx(
        3 * MESSAGE_DIM * 32
    )
    assert stats["realized_bits_per_step"] == pytest.approx(
        6 * MESSAGE_DIM * 32
    )
    assert all(isinstance(value, float) for value in stats.values())

    module.reset_stats()
    reset = module.communication_stats()
    assert "active_sender_fraction" not in reset


def test_max_message_norm_is_a_true_logging_window_maximum():
    module = BroadcastComm(2, 2)
    with torch.no_grad():
        module.message_encoder.weight.copy_(torch.eye(2))

    module(torch.full((3, 2), 2.0))
    expected_max = torch.tanh(torch.tensor(2.0)) * (2.0**0.5)
    module(torch.zeros(3, 2))

    assert module.communication_stats()["max_message_norm"] == pytest.approx(
        expected_max.item()
    )


@pytest.mark.parametrize(
    ("dropout_p", "expected_retention"),
    [(0.0, 1.0), (1.0, 0.0)],
)
def test_realized_channel_rate_is_requested_sender_retention(
    dropout_p,
    expected_retention,
):
    module = BroadcastComm(
        HIDDEN_DIM,
        MESSAGE_DIM,
        channel={"type": "dropout", "p": dropout_p, "mode": "always"},
    )
    h = torch.randn(2, N_AGENTS, HIDDEN_DIM)
    # Sparse topology must affect active edges, not channel retention itself.
    edges = torch.zeros(2, N_AGENTS, N_AGENTS, dtype=torch.bool)
    edges[:, 0, 1] = True

    module(h, CommContext(mask=edges))
    stats = module.communication_stats()

    assert stats["realized_communication_rate"] == expected_retention
    if dropout_p == 0.0:
        assert stats["active_edge_fraction"] == pytest.approx(1 / 6)
