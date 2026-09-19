"""Scientific invariants for the bounded visibility/budget model family."""

from __future__ import annotations

import copy
import csv
import math

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from commstudy.communication import (
    BroadcastComm,
    BudgetedBroadcastComm,
    CommContext,
    IdentityComm,
    LocalCapacityComm,
)
from commstudy.communication.channel import (
    CommChannel,
    channel_intervention,
    channel_phase,
)
from commstudy.models.model import CommPolicyModel


def _actor(module, kwargs):
    observation = Composite(
        adversary=Composite(
            observation=Unbounded(shape=(3, 17)), shape=(3,)
        )
    )
    output = Composite(
        adversary=Composite(logits=Unbounded(shape=(3, 4)), shape=(3,))
    )
    return CommPolicyModel(
        hidden_dim=128,
        num_encoder_layers=2,
        activation_class_path="torch.nn.Tanh",
        comm_class_path=f"{module.__module__}.{module.__name__}",
        comm_kwargs=kwargs,
        comm_context_keys={"sender_mask": "comm_sender_mask"},
        actor_observation_keys=["observation"],
        input_spec=observation,
        output_spec=output,
        agent_group="adversary",
        input_has_agent_dim=True,
        n_agents=3,
        centralised=False,
        share_params=True,
        device="cpu",
        action_spec=Composite(),
        model_index=0,
        is_critic=False,
    )


@pytest.mark.parametrize("shape", [(3, 8), (2, 3, 8), (2, 4, 3, 8)])
@pytest.mark.parametrize("normalize", [False, True])
def test_local_capacity_matches_broadcast_parameterization_and_is_active(shape, normalize):
    local = LocalCapacityComm(8, message_dim=4, normalize_comm_path=normalize)
    broadcast = BroadcastComm(8, message_dim=4, normalize_comm_path=normalize)
    assert {k: v.shape for k, v in local.state_dict().items()} == {
        k: v.shape for k, v in broadcast.state_dict().items()
    }
    local.load_state_dict(broadcast.state_dict(), strict=True)
    h = torch.randn(shape, requires_grad=True)
    result = local(h)
    assert result.shape == h.shape
    assert not torch.equal(result, h)
    result.square().sum().backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == p.numel()
               for p in local.parameters())
    assert torch.isfinite(result).all()


def test_local_capacity_never_consumes_other_agents_or_context():
    module = LocalCapacityComm(8, message_dim=4)
    h = torch.randn(2, 3, 8, requires_grad=True)
    result = module(h)
    altered = h.detach().clone()
    altered[:, 1:] = torch.randn_like(altered[:, 1:]) * 100
    context = CommContext(
        mask=torch.ones(3, 3, dtype=torch.bool),
        class_id=torch.arange(3),
        extras={"sender_mask": torch.zeros(3, dtype=torch.bool), "state": torch.randn(22)},
    )
    assert torch.equal(result[:, 0], module(altered, context)[:, 0])
    assert torch.equal(result, module(h, context))
    result[:, 0].sum().backward()
    assert torch.count_nonzero(h.grad[:, 1:]) == 0
    assert torch.count_nonzero(h.grad[:, 0]) > 0
    assert not any(isinstance(child, CommChannel) for child in module.modules())
    stats = module.communication_stats()
    assert stats["local_bottleneck_dim"] == 4
    assert all(value == 0 for key, value in stats.items() if key != "local_bottleneck_dim")


@pytest.mark.parametrize("budget", [-1, True, 1.5, "3", None])
def test_budget_rejects_non_integer_or_negative_values(budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        BudgetedBroadcastComm(8, sender_budget=budget)


@pytest.mark.parametrize("budget", [1, 2, 4])
def test_budget_rejects_unsupported_partial_or_oversized_scheduling(budget):
    module = BudgetedBroadcastComm(8, sender_budget=budget)
    with pytest.raises(ValueError, match="only sender_budget=0 or"):
        module(torch.randn(3, 8))


@pytest.mark.parametrize("shape", [(3, 8), (5, 3, 8), (2, 4, 3, 8)])
def test_full_budget_is_exact_original_broadcast_forward_and_backward(shape):
    original = BroadcastComm(8, message_dim=4)
    budgeted = BudgetedBroadcastComm(8, sender_budget=3, message_dim=4)
    budgeted.load_state_dict(original.state_dict(), strict=True)
    h = torch.randn(shape, requires_grad=True)
    copied = h.detach().clone().requires_grad_()
    expected, actual = original(h), budgeted(copied)
    assert torch.equal(expected, actual)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert torch.equal(h.grad, copied.grad)
    for first, second in zip(original.parameters(), budgeted.parameters(), strict=True):
        assert torch.equal(first.grad, second.grad)


def test_full_budget_packet_costs_sender_suppression_and_graph_accounting():
    module = BudgetedBroadcastComm(128, sender_budget=3, message_dim=32)
    h = torch.randn(2, 3, 128)
    module(h)
    stats = module.communication_stats()
    expected = {
        "configured_sender_budget": 3,
        "messages_per_step": 3,
        "active_edges_per_step": 6,
        "realized_sender_scalars_per_step": 96,
        "realized_sender_bits_per_step": 3072,
        "realized_scalar_transmissions_per_step": 192,
        "realized_bits_per_step": 6144,
    }
    assert {key: stats[key] for key in expected} == expected
    module.reset_stats()
    with channel_intervention([module.channel], torch.tensor([True, False, True])):
        module(h, CommContext(extras={"sender_mask": torch.ones(3, dtype=torch.bool)}))
    stats = module.communication_stats()
    assert stats["messages_per_step"] == 2
    assert stats["active_edges_per_step"] == 4
    assert stats["realized_sender_bits_per_step"] == 2048
    assert stats["realized_bits_per_step"] == 4096
    module.reset_stats()
    edges = torch.zeros(3, 3, dtype=torch.bool)
    edges[0, 1] = True
    module(h, CommContext(mask=edges))
    stats = module.communication_stats()
    assert stats["messages_per_step"] == stats["active_edges_per_step"] == 1
    assert stats["realized_sender_bits_per_step"] == stats["realized_bits_per_step"] == 1024


@pytest.mark.parametrize("phase", ["collection", "optimization", "evaluation"])
@pytest.mark.parametrize("normalize", [False, True])
def test_zero_budget_is_exact_identity_no_rng_and_authoritative_over_replay(phase, normalize):
    module = BudgetedBroadcastComm(
        8,
        sender_budget=0,
        message_dim=4,
        normalize_comm_path=normalize,
        channel=[
            {"type": "dropout", "p": 0.5, "mode": "always"},
            {"type": "gaussian", "std": 2.0, "mode": "always"},
        ],
    )
    channels = [child for child in module.modules() if isinstance(child, CommChannel)]
    # Initialize private streams, then prove the zero budget advances none.
    for channel in channels:
        channel._generator(torch.device("cpu"))
    private = [[g.get_state().clone() for g in c._rng.values()] for c in channels]
    h = torch.randn(2, 3, 8, requires_grad=True)
    with torch.no_grad():
        h[0, 0, 0] = -0.0
    rng = torch.get_rng_state().clone()
    context = CommContext(extras={"sender_mask": torch.ones(2, 3, dtype=torch.bool)})
    with channel_phase(phase), channel_intervention(channels, True):
        result = module(h, context)
        assert not module.channel.last_sender_mask.any()
    assert result is h
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.signbit(result[0, 0, 0])
    for channel, states in zip(channels, private, strict=True):
        for generator, state in zip(channel._rng.values(), states, strict=True):
            assert torch.equal(generator.get_state(), state)
    result.sum().backward()
    assert torch.equal(h.grad, torch.ones_like(h))
    assert all(p.grad is None for p in module.parameters())
    stats = module.communication_stats()
    for key, value in stats.items():
        if key.endswith("per_step") or key in {"active_sender_fraction", "active_edge_fraction"}:
            assert value == 0, key


def test_zero_budget_actor_publishes_false_mask_and_preserves_identity_logits():
    torch.manual_seed(27)
    identity = _actor(IdentityComm, {})
    torch.manual_seed(27)
    zero = _actor(BudgetedBroadcastComm, {"sender_budget": 0, "message_dim": 32})
    observations = torch.randn(4, 3, 17)
    td = TensorDict({("adversary", "observation"): observations}, [4])
    expected = identity(td.clone())["adversary", "logits"]
    actual = zero(td.clone())
    assert torch.equal(expected, actual["adversary", "logits"])
    assert not actual["adversary", "comm_sender_mask"].any()
    assert actual["adversary", "comm_sender_mask_generated"].all()


def test_actual_pcp_actor_active_capacity_counts_match():
    identity = _actor(IdentityComm, {})
    local = _actor(LocalCapacityComm, {"message_dim": 32})
    full = _actor(BudgetedBroadcastComm, {"sender_budget": 3, "message_dim": 32})
    assert sum(p.numel() for p in identity.parameters()) == 19332
    assert sum(p.numel() for p in local.parameters()) == 27524
    assert sum(p.numel() for p in full.parameters()) == 27524
    for actor in (local, full):
        assert sum(p.numel() for p in actor.comm.parameters()) == 8192
        result = actor(TensorDict({("adversary", "observation"): torch.randn(2, 3, 17)}, [2]))
        result["adversary", "logits"].square().sum().backward()
        assert all(p.grad is not None and torch.count_nonzero(p.grad) == p.numel()
                   for p in actor.parameters())


@pytest.mark.parametrize("module,kwargs", [
    (LocalCapacityComm, {"message_dim": 32}),
    (BudgetedBroadcastComm, {"sender_budget": 0, "message_dim": 32}),
    (BudgetedBroadcastComm, {"sender_budget": 3, "message_dim": 32}),
])
def test_real_pcp_6k_optimization_replay_and_costs(config_root, tmp_path, module, kwargs):
    """Temporary one-update validation only; no managed study row is launched."""
    from commstudy.experiments import build_experiment, load_experiment_spec
    from commstudy.experiments.health import TrainingHealthCallback
    from commstudy.experiments.metrics import ExperimentMetricsCallback

    spec = load_experiment_spec(config_root, [
        "task=vmas_predator_capture_prey", "model=pcp_comm_identity", "critic_model=pcp_critic",
        "seed=3", "experiment.max_n_frames=6000", "experiment.on_policy_n_minibatch_iters=1",
        "experiment.evaluation=false", "experiment.loggers=[]", "experiment.create_json=false",
        "experiment.checkpoint_interval=0", "experiment.checkpoint_at_end=false",
        f"experiment.save_folder={tmp_path / 'benchmarl'}",
    ])
    params = spec.model_config["groups"]["adversary"]["params"]
    params["comm_class_path"] = f"{module.__module__}.{module.__name__}"
    params["comm_kwargs"] = copy.deepcopy(kwargs)
    params["comm_context_keys"] = {"sender_mask": "comm_sender_mask"}
    experiment = build_experiment(
        spec, callbacks=[TrainingHealthCallback(), ExperimentMetricsCallback(tmp_path)]
    )
    actor = next(m for m in experiment.group_policies["adversary"].modules()
                 if isinstance(m, CommPolicyModel))
    before = {k: p.detach().clone() for k, p in actor.comm.named_parameters()}
    try:
        experiment.run()
        assert experiment.total_frames == 6000
        assert experiment.n_iters_performed == 1
        changed = [not torch.equal(before[k], p) for k, p in actor.comm.named_parameters()]
        assert all(changed) if kwargs.get("sender_budget") != 0 else not any(changed)
        with (tmp_path / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert all(math.isfinite(float(row["value"])) for row in rows)
        selected = {r["metric"]: float(r["value"]) for r in rows
                    if r["phase"] == "collection" and r["group"] == "adversary"}
        assert selected["replay_transition_count"] == 6000
        assert selected["replay_accepted_log_prob_max_tolerance_ratio"] <= 1
        assert selected["comm_messages_per_step"] == (3 if kwargs.get("sender_budget") == 3 else 0)
        assert selected["comm_realized_sender_bits_per_step"] == (
            3072 if kwargs.get("sender_budget") == 3 else 0
        )
    finally:
        experiment.close()
