import torch

from commstudy.communication.identity import IdentityComm


def test_identity_returns_same_tensor():
    comm = IdentityComm(hidden_dim=16)

    h = torch.randn(4, 3, 16)

    out = comm(h)

    # Identity should return the exact same Tensor object.
    assert out is h


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