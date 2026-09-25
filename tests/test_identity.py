import pytest
import torch

from commstudy.communication.identity import IdentityComm


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_identity_rejects_non_real_floating_embeddings(dtype):
    comm = IdentityComm(hidden_dim=16)
    h = torch.ones(3, 16, dtype=dtype)

    with pytest.raises(TypeError, match="real floating-point embeddings"):
        comm(h)


def test_identity_returns_same_tensor():
    comm = IdentityComm(hidden_dim=16)

    h = torch.randn(4, 3, 16)

    out = comm(h)

    # Identity should return the exact same Tensor object.
    assert out is h


def test_identity_preserves_input_device():
    comm = IdentityComm(hidden_dim=16)
    h = torch.randn(4, 3, 16, device="cpu")

    out = comm(h)

    assert h.device.type == "cpu"
    assert out.device == h.device


def test_identity_preserves_values():
    comm = IdentityComm(hidden_dim=16)

    h = torch.randn(4, 3, 16)
    original = h.clone()

    out = comm(h)

    assert torch.equal(out, original)


def test_identity_supports_unbatched_agents():
    comm = IdentityComm(hidden_dim=16)

    # [N, D]
    h = torch.randn(3, 16)

    out = comm(h)

    assert out.shape == (3, 16)
    assert out is h


def test_identity_supports_batched_agents():
    comm = IdentityComm(hidden_dim=16)

    # [B, N, D]
    h = torch.randn(8, 3, 16)

    out = comm(h)

    assert out.shape == h.shape
    assert out is h


def test_identity_supports_time_and_batch_dims():
    comm = IdentityComm(hidden_dim=16)

    # [T, B, N, D]
    h = torch.randn(10, 8, 3, 16)

    out = comm(h)

    assert out.shape == h.shape
    assert out is h


def test_identity_ignores_context():
    comm = IdentityComm(hidden_dim=16)

    h = torch.randn(4, 3, 16)

    # Identity should not care whether context exists.
    out = comm(h, context=None)

    assert out is h


def test_identity_preserves_gradients():
    comm = IdentityComm(hidden_dim=16)

    h = torch.randn(
        4,
        3,
        16,
        requires_grad=True,
    )

    out = comm(h)

    loss = out.sum()
    loss.backward()

    assert h.grad is not None
    assert torch.equal(
        h.grad,
        torch.ones_like(h),
    )


def test_identity_reports_zero_message_bits():
    comm = IdentityComm(hidden_dim=16)

    assert comm.message_bits() == 0


def test_identity_reports_explicit_zero_communication_stats():
    comm = IdentityComm(hidden_dim=16)
    comm(torch.randn(4, 3, 16))

    assert comm.communication_stats() == {
        "message_dim": 0.0,
        "message_bits_per_sender": 0.0,
        "active_sender_fraction": 0.0,
        "active_edge_fraction": 0.0,
        "mean_message_norm": 0.0,
        "max_message_norm": 0.0,
        "communication_rounds": 0.0,
        "messages_per_step": 0.0,
        "active_edges_per_step": 0.0,
        "nominal_messages_per_step": 0.0,
        "potential_edges_per_step": 0.0,
        "realized_sender_scalars_per_step": 0.0,
        "realized_sender_bits_per_step": 0.0,
        "nominal_sender_scalars_per_step": 0.0,
        "nominal_sender_bits_per_step": 0.0,
        "realized_scalar_transmissions_per_step": 0.0,
        "realized_bits_per_step": 0.0,
        "nominal_scalar_transmissions_per_step": 0.0,
        "nominal_bits_per_step": 0.0,
    }

    comm.reset_stats()
    assert comm.communication_stats()["active_sender_fraction"] == 0.0
