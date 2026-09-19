from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from benchmarl.experiment.callback import Callback

from commstudy.communication.base import CommModule
from commstudy.communication.channel import DropoutChannel, channel_phase
from commstudy.experiments.health import (
    TrainingHealthCallback,
    TrainingHealthError,
    check_behavior_replay,
    guard_optimizer_step,
)
from commstudy.experiments.metrics import (
    CommunicationHealthObserver,
    ExperimentMetricsCallback,
    group_health_metrics,
)


def test_group_health_ignores_other_groups_and_counts_actual_transitions():
    batch = TensorDict(
        {
            "predators": TensorDict(
                {
                    "loc": torch.arange(12, dtype=torch.float).reshape(2, 3, 2),
                    "scale": torch.ones(2, 3, 2),
                    "action": torch.zeros(2, 3, 2),
                },
                [2, 3],
            ),
            "prey": TensorDict({"loc": torch.full((2, 1, 2), 1e20)}, [2, 1]),
        },
        [2],
    )
    metrics = group_health_metrics(batch, "predators")
    assert metrics["health_transition_count"] == 2
    assert metrics["loc_scalar_count"] == 12
    assert metrics["loc_mean"] == 5.5
    assert metrics["action_saturated_fraction"] == 0
    batch.set(("prey", "loc"), torch.full((2, 1, 2), float("nan")))
    assert group_health_metrics(batch, "predators") == metrics


class _ObservedComm(CommModule):
    def forward(self, h, context=None):
        self._record_stats(sample_mean=h.mean())
        return h * 2


def test_comm_health_weights_unequal_batch_sizes_and_preserves_output():
    module = _ObservedComm(hidden_dim=2)
    observer = CommunicationHealthObserver(module, lambda: False)
    small = torch.ones(1, 3, 2)
    large = torch.full((9, 3, 2), 3.0)
    assert torch.equal(module(small), 2 * small)
    assert torch.equal(module(large), 2 * large)
    assert module.communication_stats()["sample_mean"] == pytest.approx(2.8)
    metrics = observer.drain()
    assert metrics["transition_count"] == 10
    assert metrics["agent_transition_count"] == 30
    assert metrics["encoder_norm_mean"] == pytest.approx(2.8 * 2**0.5)
    assert metrics["contribution_to_encoder_norm_ratio"] == pytest.approx(1)
    assert observer.drain()["transition_count"] == 0


@pytest.mark.parametrize("target", ["parameter", "gradient", "optimizer_state"])
def test_optimizer_hook_rejects_poison_before_adam_changes_parameters(target):
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    optimizer.register_step_pre_hook(guard_optimizer_step)
    parameter.square().backward()
    optimizer.step()
    optimizer.zero_grad()
    parameter.square().backward()
    if target == "parameter":
        with torch.no_grad():
            parameter.fill_(float("nan"))
    elif target == "gradient":
        parameter.grad.fill_(float("inf"))
    else:
        optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    before = parameter.detach().clone()
    step_before = optimizer.state[parameter]["step"].clone()
    with pytest.raises(TrainingHealthError, match="optimizer step rejected"):
        optimizer.step()
    torch.testing.assert_close(parameter, before, equal_nan=True)
    assert torch.equal(optimizer.state[parameter]["step"], step_before)


class _BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.square()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


@pytest.mark.parametrize("mode", ["loss", "gradient"])
def test_health_rejects_nonfinite_loss_or_preclipping_gradient(mode):
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    class Loss(torch.nn.Module):
        def forward(self, batch):
            loss = parameter * float("nan") if mode == "loss" else _BadBackward.apply(parameter)
            return TensorDict({"loss_objective": loss}, [])

    loss = Loss()
    optimizer = torch.optim.Adam([parameter])
    callback = TrainingHealthCallback()
    callback.experiment = SimpleNamespace(
        train_group_map={"predators": ["p"]},
        losses={"predators": loss},
        optimizers={"predators": {"loss_objective": optimizer}},
    )
    callback.on_setup()
    with pytest.raises(TrainingHealthError):
        loss(None)["loss_objective"].backward()
    assert parameter.item() == 2
    assert not optimizer.state


@pytest.mark.parametrize("phase", ["collection", "optimization", "evaluation"])
@pytest.mark.parametrize("interaction", ["random", "deterministic"])
def test_explicit_channel_phase_controls_both_exploration_modes(phase, interaction):
    from tensordict.nn.probabilistic import InteractionType, set_interaction_type

    channel = DropoutChannel(p=1, mode="evaluation")
    channel.train()
    mode = InteractionType.RANDOM if interaction == "random" else InteractionType.DETERMINISTIC
    with torch.no_grad(), set_interaction_type(mode), channel_phase(phase):
        result = channel(torch.ones(3, 2))
    assert bool(result.sender_mask.any()) is (phase != "evaluation")


def test_real_dropout_replay_precedes_update_and_diagnostics_separate_phases(config_root, tmp_path):
    from commstudy.experiments import build_experiment, load_experiment_spec

    spec = load_experiment_spec(
        config_root,
        [
            "model=comm_attention",
            "model_config.params.comm_kwargs.channel.type=dropout",
            "model_config.params.comm_kwargs.channel.p=0.5",
            "model_config.params.comm_kwargs.channel.mode=always",
            "experiment.max_n_frames=6000",
            "experiment.on_policy_n_minibatch_iters=1",
            "experiment.evaluation=false",
            "experiment.loggers=[]",
            "experiment.create_json=false",
            "experiment.checkpoint_at_end=false",
            f"experiment.save_folder={tmp_path / 'benchmarl'}",
        ],
    )
    health = TrainingHealthCallback()
    metrics = ExperimentMetricsCallback(tmp_path)
    experiment = build_experiment(spec, callbacks=[health, metrics])
    # Inspect the stored complete behavior batch before any optimizer is called.
    batch = next(iter(experiment.collector))
    policy = experiment.group_policies["agents"]
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    replay = check_behavior_replay(policy, batch, "agents")
    assert replay["replay_log_prob_max_abs_error"] < 1e-5
    assert replay["replay_transition_count"] == 6000
    batch.set(("agents", "log_prob"), batch.get(("agents", "log_prob")) + 0.1)
    with pytest.raises(TrainingHealthError, match="replay mismatch"):
        check_behavior_replay(policy, batch, "agents")
    for name, parameter in policy.named_parameters():
        assert torch.equal(parameter, before[name])
    # A diagnostic collector consumes its frame budget. Use a fresh experiment
    # for the genuine untouched framework optimization.
    experiment.close()
    experiment = build_experiment(
        spec, callbacks=[TrainingHealthCallback(), ExperimentMetricsCallback(tmp_path)]
    )
    experiment.run()
    with (tmp_path / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    def value(phase, metric):
        values = [
            float(row["value"])
            for row in rows
            if row["group"] == "agents" and row["phase"] == phase and row["metric"] == metric
        ]
        assert len(values) == 1, (phase, metric, values)
        return values[0]

    assert value("collection", "health_transition_count") == 6000
    assert value("collection", "comm_transition_count") == 6000
    assert value("training", "optimization_transition_count") == 6000
    assert value("training", "comm_transition_count") == 6000
    assert value("training", "advantage_scalar_count") == 18000
    assert value("collection", "replay_log_prob_max_abs_error") < 1e-5


class _BatchRoundoffPolicy(torch.nn.Module):
    """Inject a batch-only discrepancy to exercise fallback on every backend."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.comm = _ObservedComm(hidden_dim=1)

    def get_dist(self, batch):
        observation = batch["g", "observation"]
        self.comm(observation)
        torch.rand(())  # The guard must restore this draw, including on rejection.
        loc = observation + self.bias + (2e-5 if batch.ndim == 2 else 0.0)
        batch["g", "loc"] = loc
        batch["g", "scale"] = torch.ones_like(loc)
        return SimpleNamespace(log_prob=lambda action: loc.squeeze(-1))


def _roundoff_batch(time_axis=1):
    names = [None, None]
    if time_axis is not None:
        names[time_axis] = "time"
    return TensorDict(
        {
            "g": {
                "observation": torch.zeros(2, 3, 1, 1),
                "action": torch.zeros(2, 3, 1, 1),
                "log_prob": torch.zeros(2, 3, 1),
            }
        },
        [2, 3],
        names=names,
    )


@pytest.mark.parametrize("time_axis", [0, 1])
def test_replay_fallback_preserves_original_tolerance_rng_parameters_and_stats(time_axis):
    policy = _BatchRoundoffPolicy()
    policy.comm(torch.ones(2, 1, 1))
    stats = policy.comm.communication_stats()
    batch = _roundoff_batch(time_axis)
    before = batch.clone()
    rng = torch.get_rng_state().clone()
    details, capture = {}, {}
    result = check_behavior_replay(policy, batch, "g", diagnostics=details, capture=capture)
    assert result["replay_collection_shape_fallback_count"] == 1
    assert result["replay_collection_shape_fallback_slices"] == batch.shape[time_axis]
    assert result["replay_batched_log_prob_max_tolerance_ratio"] > 1
    assert result["replay_log_prob_max_abs_error"] == 0
    assert details["rtol"] == details["atol"] == 1e-5
    assert details["time_axis"] == time_axis
    assert capture["collection_shaped"]["log_prob"].eq(0).all()
    assert torch.equal(torch.get_rng_state(), rng)
    assert policy.comm.communication_stats() == stats
    assert policy.bias.item() == 0 and policy.bias.grad is None
    assert (batch == before).all()


@pytest.mark.parametrize("corruption", ["missing_time", "parameter", "stored_log_prob"])
def test_replay_fallback_rejects_missing_shape_contract_or_genuine_corruption(corruption):
    policy = _BatchRoundoffPolicy()
    batch = _roundoff_batch(None if corruption == "missing_time" else 1)
    if corruption == "parameter":
        with torch.no_grad():
            policy.bias.add_(0.1)
    elif corruption == "stored_log_prob":
        batch["g", "log_prob"].add_(0.1)
    rng = torch.get_rng_state().clone()
    with pytest.raises(TrainingHealthError, match="replay mismatch") as error:
        check_behavior_replay(policy, batch, "g")
    assert error.value.details["batched"]["violation_count"] > 0
    if corruption == "missing_time":
        assert "fallback_unavailable" in error.value.details
    else:
        assert error.value.details["collection_shaped"]["violation_count"] > 0
    assert torch.equal(torch.get_rng_state(), rng)


def test_callback_preserves_first_fallback_and_terminal_failure_snapshots(tmp_path):
    policy = _BatchRoundoffPolicy()
    callback = TrainingHealthCallback()
    callback.experiment = SimpleNamespace(
        train_group_map={"g": ["p"]},
        group_policies={"g": policy},
        folder_name=tmp_path,
        total_frames=6000,
    )
    batch = _roundoff_batch()
    callback.on_batch_collected(batch)
    callback.on_batch_collected(batch)
    files = list((tmp_path / "replay_diagnostics").glob("*.pt"))
    assert len(files) == 1
    saved = torch.load(files[0], weights_only=False)
    assert saved["diagnostic_only"] is True and saved["not_resume"] is True
    assert saved["reason"] == "first_fallback"
    assert saved["details"]["batched"]["violation_count"] > 0
    assert saved["details"]["collection_shaped"]["violation_count"] == 0
    assert (saved["batch"] == batch).all()
    assert len(saved["source_sha256"]) == 64
    assert saved["group_policies"]["g"]["bias"].item() == 0
    with torch.no_grad():
        policy.bias.add_(0.1)
    with pytest.raises(TrainingHealthError):
        callback.on_batch_collected(batch)
    assert len(list((tmp_path / "replay_diagnostics").glob("*failure*.pt"))) == 1
    assert not callback.experiment._checking_policy_replay


def test_snapshot_io_or_warning_failure_never_masks_original_replay_rejection(
    tmp_path, monkeypatch
):
    import warnings

    policy = _BatchRoundoffPolicy()
    with torch.no_grad():
        policy.bias.add_(0.1)
    callback = TrainingHealthCallback()
    callback.experiment = SimpleNamespace(
        train_group_map={"g": ["p"]},
        group_policies={"g": policy},
        folder_name=tmp_path,
        total_frames=6000,
    )

    def broken_save(*args, **kwargs):
        raise OSError("deliberate diagnostic disk failure")

    monkeypatch.setattr(torch, "save", broken_save)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TrainingHealthError, match="persists at collection shape"):
            callback.on_batch_collected(_roundoff_batch())


def test_real_480k_actor_constructed_support_points_survive_batch_roundoff(config_root, tmp_path):
    import platform
    from commstudy.experiments import build_experiment, load_experiment_spec

    fixture = torch.load(
        Path(__file__).parent / "fixtures" / "pcp_replay_roundoff_seed010.pt",
        weights_only=True,
        map_location="cpu",
    )
    spec = load_experiment_spec(
        config_root,
        [
            "task=vmas_predator_capture_prey",
            "model=pcp_comm_identity",
            "critic_model=pcp_critic",
            "experiment.max_n_frames=6000",
            "experiment.evaluation=false",
            "experiment.loggers=[]",
            "experiment.create_json=false",
            "experiment.checkpoint_at_end=false",
            f"experiment.save_folder={tmp_path}",
        ],
    )
    experiment = build_experiment(spec)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy = experiment.group_policies["adversary"]
        policy.load_state_dict(fixture["actor_state"], strict=True)
        batch = TensorDict(
            {"adversary": {key: fixture[key] for key in ("observation", "action", "log_prob")}},
            [10, 600],
            names=[None, "time"],
        )
        result = check_behavior_replay(policy, batch, "adversary")
        assert result["replay_accepted_log_prob_max_tolerance_ratio"] <= 1
        # CPU kernels differ across supported hosts. The original matching
        # macOS/ARM runtime must reproduce the proven false rejection.
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            assert result["replay_collection_shape_fallback_count"] == 1
            assert result["replay_batched_log_prob_max_tolerance_ratio"] > 1
            assert result["replay_log_prob_max_abs_error"] == 0
        with torch.no_grad():
            next(policy.parameters()).add_(0.1)
        with pytest.raises(TrainingHealthError, match="replay mismatch"):
            check_behavior_replay(policy, batch, "adversary")
    finally:
        torch.set_num_threads(prior_threads)
        experiment.close()


class _PCPDropoutReplayProbe(Callback):
    def on_batch_collected(self, batch):
        from commstudy.utils.rng import RNGState

        assert batch.names == [None, "time"]
        assert batch.batch_size == torch.Size([10, 600])
        policy = self.experiment.group_policies["adversary"]
        rng = RNGState.capture()
        original_mask = batch["adversary", "comm_sender_mask"].clone()
        metrics = check_behavior_replay(policy, batch, "adversary")
        assert metrics["replay_collection_shape_fallback_count"] == 1
        assert metrics["replay_collection_shape_fallback_slices"] == 600
        assert metrics["replay_accepted_log_prob_max_tolerance_ratio"] <= 1
        assert torch.equal(rng.torch_cpu, torch.get_rng_state())
        assert torch.equal(original_mask, batch["adversary", "comm_sender_mask"])
        corrupt = batch.clone()
        corrupt["adversary", "comm_sender_mask"].logical_not_()
        with pytest.raises(TrainingHealthError, match="persists at collection shape"):
            check_behavior_replay(policy, corrupt, "adversary")


def test_real_pcp_dropout_fallback_preserves_masks_and_allows_framework_update(
    config_root, tmp_path
):
    from commstudy.experiments import build_experiment, load_experiment_spec

    spec = load_experiment_spec(
        config_root,
        [
            "task=vmas_predator_capture_prey",
            "model=pcp_comm_attention",
            "critic_model=pcp_critic",
            "model_config.groups.adversary.params.comm_kwargs.channel.type=dropout",
            "model_config.groups.adversary.params.comm_kwargs.channel.p=0.5",
            "model_config.groups.adversary.params.comm_kwargs.channel.mode=always",
            "experiment.max_n_frames=6000",
            "experiment.on_policy_n_minibatch_iters=1",
            "experiment.evaluation=false",
            "experiment.loggers=[]",
            "experiment.create_json=false",
            "experiment.checkpoint_at_end=false",
            f"experiment.save_folder={tmp_path / 'benchmarl'}",
        ],
    )
    experiment = build_experiment(
        spec, callbacks=[_PCPDropoutReplayProbe(), ExperimentMetricsCallback(tmp_path)]
    )
    policy = experiment.group_policies["adversary"]
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    original_get_dist = policy.get_dist

    def simulated_batch_roundoff(batch, *args, **kwargs):
        distribution = original_get_dist(batch, *args, **kwargs)
        if batch.ndim == 2:
            return SimpleNamespace(log_prob=lambda action: distribution.log_prob(action) + 2e-4)
        return distribution

    policy.get_dist = simulated_batch_roundoff
    try:
        experiment.run()
        assert experiment.total_frames == 6000
        assert any(
            not torch.equal(old, new) for old, new in zip(before, policy.parameters(), strict=True)
        )
        with (tmp_path / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        fallback = [
            row
            for row in rows
            if row["group"] == "adversary"
            and row["metric"] == "replay_collection_shape_fallback_count"
        ]
        assert len(fallback) == 1 and float(fallback[0]["value"]) == 1
        assert list(
            (Path(experiment.folder_name) / "replay_diagnostics").glob("*first_fallback.pt")
        )
    finally:
        experiment.close()
