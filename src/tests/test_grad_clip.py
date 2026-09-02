"""Gradient clipping on the communication path.

Seven of 231 V2 runs diverged, all of them attention or graph at
``message_dim=64`` or ``rounds>=2``, and all with the same signature: the
message norm goes non-finite in the same step as the loss, after
``grad_norm_loss_objective`` reaches 1e5..1e13. The clip bounds the gradient
entering the message encoders, attention scores and decoders, which is the
feedback loop that drives the runaway.

Two properties matter as much as the clipping itself and are tested here:

* the forward pass must stay bit-identical, or PPO replay would compare
  log-probabilities from two different policies; and
* the residual skip must stay unclipped, or the clip would throttle the
  encoder, which was never the problem.
"""

from __future__ import annotations

import pytest
import torch

from commstudy.communication.attention import AttentionComm
from commstudy.communication.broadcast import BroadcastComm
from commstudy.communication.gated import GatedComm
from commstudy.communication.graph import GraphComm
from commstudy.communication.identity import IdentityComm


HIDDEN = 16
AGENTS = 3


def _module(kind: str, **kwargs):
    common = {"hidden_dim": HIDDEN, **kwargs}
    if kind == "broadcast":
        return BroadcastComm(message_dim=8, **common)
    if kind == "gated":
        return GatedComm(message_dim=8, **common)
    if kind == "attention":
        return AttentionComm(message_dim=8, key_dim=8, num_heads=2, **common)
    if kind == "graph":
        return GraphComm(message_dim=8, key_dim=8, num_heads=2, **common)
    raise AssertionError(kind)


KINDS = ("broadcast", "gated", "attention", "graph")


def _seeded(kind: str, *, seed: int = 0, **kwargs):
    torch.manual_seed(seed)
    return _module(kind, **kwargs)


def _input(batch: int = 4, *, scale: float = 1.0) -> torch.Tensor:
    torch.manual_seed(123)
    return (torch.randn(batch, AGENTS, HIDDEN) * scale).requires_grad_(True)


# --------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_grad_clip_defaults_to_disabled(kind):
    """The frozen protocol ran without it; the default must not change."""

    assert _module(kind).grad_clip is None


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan")])
def test_grad_clip_rejects_non_positive_and_non_finite(bad):
    with pytest.raises(ValueError, match="positive and finite"):
        _module("attention", grad_clip=bad)


@pytest.mark.parametrize("bad", ["1.0", True, None.__class__])
def test_grad_clip_rejects_non_numeric(bad):
    with pytest.raises(TypeError, match="grad_clip must be"):
        _module("attention", grad_clip=bad)


@pytest.mark.parametrize("kind", KINDS)
def test_grad_clip_is_accepted_through_comm_kwargs(kind):
    """It must reach the module the same way every other comm option does."""

    assert _module(kind, grad_clip=2.5).grad_clip == pytest.approx(2.5)


def test_identity_accepts_grad_clip_without_changing_its_output():
    """IdentityComm's numerical behaviour is fixed by the study protocol."""

    inputs = _input()
    plain = IdentityComm(hidden_dim=HIDDEN)(inputs)
    clipped = IdentityComm(hidden_dim=HIDDEN, grad_clip=0.001)(inputs)

    assert torch.equal(plain, clipped)
    assert torch.equal(clipped, inputs)


# --------------------------------------------------------------------------
# The forward pass must not move
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_forward_is_bit_identical_with_and_without_clipping(kind):
    """A backward hook must not perturb PPO's stored log-probabilities.

    If enabling the clip changed the forward at all, the log-probability stored
    during rollout collection and the one recomputed during the update would come
    from different policies, silently corrupting the PPO ratio.
    """

    inputs = _input()
    plain = _seeded(kind)(inputs)
    clipped = _seeded(kind, grad_clip=1e-6)(inputs)

    assert torch.equal(plain, clipped)


@pytest.mark.parametrize("kind", KINDS)
def test_clipping_does_not_run_under_no_grad(kind):
    """Evaluation and saliency rollouts are inference-only; hooks cannot attach."""

    module = _module(kind, grad_clip=0.5)
    with torch.no_grad():
        output = module(torch.randn(4, AGENTS, HIDDEN))

    assert output.shape == (4, AGENTS, HIDDEN)
    assert torch.isfinite(output).all()


# --------------------------------------------------------------------------
# The clip actually bounds the gradient
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_clip_bounds_the_comm_parameter_gradients(kind):
    """Large upstream gradients must not reach the comm weights unattenuated."""

    def comm_grad_norm(grad_clip):
        module = _seeded(kind, grad_clip=grad_clip)
        inputs = _input()
        # A deliberately huge upstream gradient, standing in for the exploding
        # PPO objective seen in the diverged runs.
        (module(inputs).sum() * 1e6).backward()
        comm_params = [
            p for name, p in module.named_parameters()
            if p.grad is not None and "encoder" not in name
        ]
        assert comm_params, "expected communication parameters to receive gradient"
        return torch.stack([p.grad.norm() for p in comm_params]).sum()

    unclipped = comm_grad_norm(None)
    clipped = comm_grad_norm(1.0)

    assert clipped < unclipped
    # The clip is meant to be decisive, not cosmetic.
    assert clipped < unclipped / 100


@pytest.mark.parametrize("kind", KINDS)
def test_clip_leaves_the_residual_skip_path_at_full_scale(kind):
    """The encoder gradient travels the skip connection and must not shrink.

    Clipping the module output instead of the contribution would throttle the
    whole actor. The observed failure was in the message path only.
    """

    def input_grad(grad_clip):
        module = _seeded(kind, grad_clip=grad_clip)
        inputs = _input()
        (module(inputs).sum() * 1e6).backward()
        return inputs.grad.detach().clone()

    unclipped = input_grad(None)
    clipped = input_grad(0.001)

    # The identity term contributes a constant 1e6 per element regardless of the
    # clip; only the message-path component of the input gradient is attenuated.
    assert clipped.abs().sum() > 0
    assert torch.allclose(clipped, torch.full_like(clipped, 1e6), rtol=1e-3)
    assert not torch.allclose(unclipped, clipped)


@pytest.mark.parametrize("kind", KINDS)
def test_clip_is_a_no_op_when_gradients_are_already_small(kind):
    """A generous threshold must leave healthy training untouched."""

    def comm_grads(grad_clip):
        module = _seeded(kind, grad_clip=grad_clip)
        module(_input()).sum().backward()
        return {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}

    baseline = comm_grads(None)
    generous = comm_grads(1e9)

    assert baseline.keys() == generous.keys()
    for name in baseline:
        assert torch.allclose(baseline[name], generous[name], atol=1e-9), name


@pytest.mark.parametrize("kind", KINDS)
def test_one_transition_is_clipped_the_same_however_many_share_its_batch(kind):
    """Rollout and minibatch forwards use different shapes; the clip must not.

    Transitions do not interact inside these modules -- attention is over agents
    within a transition -- so a fixed transition's gradient must not depend on
    what else was batched alongside it. A tensor-global norm would fail this:
    the scale factor would shrink as the batch grew, so the same configured
    threshold would mean one thing during collection and another during the PPO
    update.
    """

    torch.manual_seed(7)
    subject = torch.randn(1, AGENTS, HIDDEN)
    company = torch.randn(7, AGENTS, HIDDEN) * 50.0

    def gradient_of_subject(batch: torch.Tensor) -> torch.Tensor:
        # residual=False so the input gradient is purely the communication path.
        # With the skip connection present its constant term is ~1e6 and would
        # swamp the clipped contribution this test is about.
        module = _seeded(kind, grad_clip=0.01, residual=False)
        inputs = batch.clone().requires_grad_(True)
        (module(inputs).sum() * 1e6).backward()
        return inputs.grad[0].detach().clone()

    alone = gradient_of_subject(subject)
    crowded = gradient_of_subject(torch.cat((subject, company), dim=0))

    assert torch.allclose(alone, crowded, atol=1e-6)


def test_multi_round_attention_clips_every_round():
    """rounds>=2 is the setting that diverged; each round must be bounded."""

    def comm_grad_norm(rounds, grad_clip):
        module = _seeded("attention", rounds=rounds, grad_clip=grad_clip)
        (module(_input()).sum() * 1e6).backward()
        return module.value_projection.weight.grad.norm()

    for rounds in (1, 2, 3):
        assert comm_grad_norm(rounds, 1.0) < comm_grad_norm(rounds, None)

    # Compounding across rounds is the mechanism; the clip must hold it down as
    # depth grows rather than degrading with it.
    assert comm_grad_norm(3, 1.0) < comm_grad_norm(1, None)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_clipping_reports_how_hard_it_is_working(kind):
    """The threshold has to be calibrated from data, so expose the pre-clip norm."""

    module = _module(kind, grad_clip=1e-4)
    (module(_input()).sum() * 1e6).backward()
    stats = module.communication_stats()

    assert stats["grad_clip_fraction"] == pytest.approx(1.0)
    assert stats["grad_clip_observed_max_norm"] > 1e-4


@pytest.mark.parametrize("kind", KINDS)
def test_no_clipping_diagnostics_when_disabled(kind):
    module = _module(kind)
    module(_input()).sum().backward()
    stats = module.communication_stats()

    assert "grad_clip_fraction" not in stats
    assert "grad_clip_observed_max_norm" not in stats


def test_observed_max_norm_is_a_window_maximum_not_an_average():
    """Averaging would hide the single spike that precedes a divergence."""

    module = _module("attention", grad_clip=1e9)
    for scale in (1.0, 1e4, 1.0):
        module(_input()).sum().mul(scale).backward()

    stats = module.communication_stats()
    small = _module("attention", grad_clip=1e9)
    small(_input()).sum().backward()

    assert stats["grad_clip_observed_max_norm"] > (
        small.communication_stats()["grad_clip_observed_max_norm"] * 100
    )


def test_reset_stats_clears_clipping_diagnostics():
    module = _module("attention", grad_clip=1e-4)
    (module(_input()).sum() * 1e6).backward()
    module.reset_stats()

    assert "grad_clip_fraction" not in module.communication_stats()


# --------------------------------------------------------------------------
# Normalization: the guard that actually bounds the divergence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_normalization_defaults_to_disabled(kind):
    """It changes the forward pass, so the frozen protocol must not inherit it."""

    assert _module(kind).normalize_comm_path is False


def test_normalization_rejects_non_boolean():
    with pytest.raises(TypeError, match="normalize_comm_path must be a bool"):
        _module("attention", normalize_comm_path=1.0)


def _amplification(kind, *, rounds, weight_gain, normalize):
    """Output magnitude when the comm weights have grown by ``weight_gain``.

    Scaling every projection stands in for weights drifting upward over training.
    ``output_projection`` initializes at xavier gain 0.01, so scaling it alone
    barely moves a small test module; the runaway needs the value/message path to
    grow with it, which is what happens in a real run.
    """

    extra = {"rounds": rounds} if kind in ("attention", "graph") else {}
    module = _seeded(kind, normalize_comm_path=normalize, **extra)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("weight") and "gate_network" not in name:
                parameter.mul_(weight_gain)
        return module(_input()).norm(dim=-1).max().item()


@pytest.mark.parametrize("kind", KINDS)
def test_normalization_makes_the_output_invariant_to_weight_scale(kind):
    """The divergence mechanism: growing weights amplify without a bound.

    Unnormalized, a 1000x weight scale drives the activation norm into the
    1e13 range seen in the diverged runs' gradient traces. Normalized, the
    output is capped no matter how large the weights get.
    """

    modest = _amplification(kind, rounds=1, weight_gain=10, normalize=True)
    extreme = _amplification(kind, rounds=1, weight_gain=1000, normalize=True)
    # Exact for attention/graph; broadcast and gated tanh their messages, so a
    # 100x weight change moves the saturation point slightly.
    assert extreme == pytest.approx(modest, rel=0.05)

    unnormalized = _amplification(kind, rounds=1, weight_gain=1000, normalize=False)
    assert unnormalized > 100 * extreme


@pytest.mark.parametrize("kind", ("attention", "graph"))
def test_normalization_stops_amplification_compounding_across_rounds(kind):
    """rounds>=2 diverged because shared-parameter residual rounds multiply."""

    unnormalized = [
        _amplification(kind, rounds=r, weight_gain=100, normalize=False)
        for r in (1, 2, 3)
    ]
    normalized = [
        _amplification(kind, rounds=r, weight_gain=100, normalize=True)
        for r in (1, 2, 3)
    ]

    # Without the guard each extra round multiplies the magnitude.
    assert unnormalized[1] > 10 * unnormalized[0]
    assert unnormalized[2] > 10 * unnormalized[1]

    # With it, depth adds a bounded amount rather than a multiplicative one.
    assert normalized[2] < 4 * normalized[0]
    assert normalized[2] < unnormalized[2] / 1000


@pytest.mark.parametrize("kind", KINDS)
def test_normalization_adds_exactly_one_scalar_parameter(kind):
    """comm_params is a reported study metric, so the cost must be exactly known.

    The LayerNorm is affine-free; the only addition is the scalar gain that
    preserves the near-identity initialization.
    """

    plain = sum(p.numel() for p in _seeded(kind).parameters())
    normalized = sum(
        p.numel() for p in _seeded(kind, normalize_comm_path=True).parameters()
    )

    assert normalized == plain + 1


@pytest.mark.parametrize("kind", KINDS)
def test_normalization_preserves_the_near_identity_initialization(kind):
    """Unit-RMS alone made communication start loud and cost ~800 return points.

    The modules initialize their output projection at xavier gain 0.01 so the
    comm path starts as almost a no-op. Normalizing without the scalar gain
    destroyed that; this pins the fix.
    """

    inputs = _input()
    plain = _seeded(kind)(inputs)
    normalized = _seeded(kind, normalize_comm_path=True)(inputs)

    plain_contribution = (plain - inputs).norm()
    normalized_contribution = (normalized - inputs).norm()

    assert normalized_contribution < 10 * plain_contribution.clamp_min(1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_normalization_keeps_output_finite_and_differentiable(kind):
    module = _seeded(kind, normalize_comm_path=True)
    inputs = _input()
    output = module(inputs)
    output.sum().backward()

    assert torch.isfinite(output).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("kind", KINDS)
def test_normalization_and_clipping_compose(kind):
    module = _seeded(kind, normalize_comm_path=True, grad_clip=1e-4)
    (module(_input()).sum() * 1e6).backward()

    assert module.communication_stats()["grad_clip_fraction"] == pytest.approx(1.0)
