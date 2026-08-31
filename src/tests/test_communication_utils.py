import pytest
import torch

from commstudy.communication.utils import (
    active_edge_fraction,
    active_sender_fraction,
    communication_costs_for_round,
    full_comm_mask,
    masked_sender_mean,
    masked_softmax,
    resolve_comm_mask,
    sender_mask_to_edge_mask,
    validate_comm_input,
)


def test_validate_comm_input_accepts_arbitrary_leading_dimensions():
    for shape in ((3, 8), (4, 3, 8), (2, 4, 3, 8), (2, 1, 4, 3, 8)):
        validate_comm_input(torch.randn(shape), 8, "TestComm")


@pytest.mark.parametrize("shape", [(8,), (2, 3, 7)])
def test_validate_comm_input_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        validate_comm_input(torch.randn(shape), 8, "TestComm")


def test_full_mask_uses_receiver_sender_convention_and_excludes_self():
    mask = full_comm_mask(3, leading_shape=(2, 4), exclude_self=True)

    assert mask.shape == (2, 4, 3, 3)
    assert mask.dtype is torch.bool
    assert not mask.diagonal(dim1=-2, dim2=-1).any()
    assert mask.sum().item() == 2 * 4 * 3 * 2


def test_resolve_mask_broadcasts_without_mutating_input():
    h = torch.randn(2, 4, 3, 8)
    source = torch.ones(4, 3, 3, dtype=torch.bool)
    original = source.clone()

    resolved = resolve_comm_mask(h, source, exclude_self=True)

    assert resolved.shape == (2, 4, 3, 3)
    assert torch.equal(source, original)
    assert not resolved.diagonal(dim1=-2, dim2=-1).any()


def test_resolve_mask_preserves_supplied_diagonal_when_self_is_enabled():
    h = torch.randn(3, 8)
    mask = torch.eye(3, dtype=torch.bool)

    assert torch.equal(resolve_comm_mask(h, mask, exclude_self=False), mask)


def test_resolve_mask_rejects_nonbroadcastable_leading_shape():
    h = torch.randn(2, 4, 3, 8)
    mask = torch.ones(5, 3, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="not broadcastable"):
        resolve_comm_mask(h, mask)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(3, 3, dtype=torch.int64),
        torch.full((3, 3), float("nan")),
    ],
    ids=["integer", "nan_float"],
)
def test_resolve_mask_rejects_numeric_and_nan_masks(mask):
    with pytest.raises(TypeError, match="must be boolean"):
        resolve_comm_mask(torch.randn(3, 8), mask)


def test_masked_softmax_normalizes_valid_rows_and_zeros_masked_entries():
    scores = torch.tensor([[1.0, 2.0, 3.0], [5.0, 4.0, 3.0]])
    mask = torch.tensor([[True, False, True], [False, False, False]])

    probabilities = masked_softmax(scores, mask, dim=-1)

    assert probabilities[0, 1].item() == 0.0
    assert torch.allclose(probabilities[0].sum(), torch.tensor(1.0))
    assert torch.equal(probabilities[1], torch.zeros(3))
    assert torch.isfinite(probabilities).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_masked_softmax_preserves_dtype(dtype):
    scores = torch.randn(2, 3, dtype=dtype)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    output = masked_softmax(scores, mask, dim=-1)

    assert output.dtype == dtype
    assert torch.equal(output, torch.zeros_like(output))


def test_masked_sender_mean_is_safe_for_receivers_with_no_edges():
    messages = torch.tensor([[[1.0], [3.0], [8.0]]])
    # receiver 0 hears senders 1 and 2; receiver 1 hears nobody.
    edges = torch.zeros(1, 3, 3, dtype=torch.bool)
    edges[:, 0, 1:] = True

    aggregated = masked_sender_mean(messages, edges)

    assert aggregated[0, 0, 0].item() == pytest.approx(5.5)
    assert torch.equal(aggregated[0, 1], torch.zeros(1))
    assert torch.isfinite(aggregated).all()


def test_sender_mask_removes_sender_from_all_receiver_edges():
    edges = full_comm_mask(3, leading_shape=(2,))
    senders = torch.tensor([[True, False, True], [False, False, True]])

    effective = sender_mask_to_edge_mask(senders, edges)

    assert not effective[..., 1].any()
    assert not effective[1, ..., 0].any()
    assert effective[0, 1, 0]


def test_activity_metrics_have_explicit_denominators():
    edges = torch.zeros(1, 3, 3, dtype=torch.bool)
    edges[0, 0, 1] = True
    edges[0, 2, 1] = True

    assert active_sender_fraction(edges).item() == pytest.approx(1 / 3)
    assert active_edge_fraction(edges, exclude_self=True).item() == pytest.approx(2 / 6)


def test_cost_metrics_separate_sender_emissions_from_edge_deliveries():
    base = full_comm_mask(3, leading_shape=(2,))
    active_senders = torch.tensor([True, False, True])
    active = sender_mask_to_edge_mask(active_senders, base)

    costs = communication_costs_for_round(
        active,
        base,
        packet_dim=4,
        active_senders=active_senders,
    )

    assert costs["messages_per_step"] == pytest.approx(2.0)
    assert costs["active_edges_per_step"] == pytest.approx(4.0)
    assert costs["realized_sender_bits_per_step"] == pytest.approx(2 * 4 * 32)
    assert costs["realized_bits_per_step"] == pytest.approx(4 * 4 * 32)
    assert costs["nominal_sender_bits_per_step"] == pytest.approx(3 * 4 * 32)
    assert costs["nominal_bits_per_step"] == pytest.approx(6 * 4 * 32)
