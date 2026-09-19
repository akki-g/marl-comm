"""Configuration mistakes must fail before a costly scientific worker starts."""

from copy import deepcopy
from dataclasses import asdict, replace
import warnings

import pytest
import torch
from omegaconf import OmegaConf

from commstudy.experiments.config import (
    canonical_scientific_config, experiment_spec_from_dict, load_experiment_spec,
    resolved_experiment_dict, scientific_config_sha256,
)
from commstudy.experiments.runner import build_model_config
from commstudy.tasks import resolve_task
from commstudy.tasks.vmas import CustomVmasTask
from commstudy.tasks.vmas.scenarios.predator_capture_prey import PredatorCapturePreyScenario


PCP = ["task=vmas_predator_capture_prey", "model=pcp_comm_attention", "critic_model=pcp_critic"]


@pytest.mark.parametrize("override", [
    "task_config.params.num_obstacles=7",
    "task_config.extra=true",
    "algorithm_config.params.entopy_coef=0.1",
    "algorithm_config.entropy_coef=0.1",
    "model_config.params.comm_kwargs.num_heads=8",
    "model_config.params={}",
    "model_config.model_type=communication",
    "model_config.groups.adversary.params.comm_kwargs.num_heeds=8",
    "model_config.groups.adversary.params.hidden_dims=64",
    "model_config.groups.adversary.params.comm_kwargs.channel.typo=true",
    "model_config.groups.agent.params.activation_class=torch.nn.ReLU",
    "model_config.groups.adversary.params.use_role_embedding=true",
    "model_config.groups.adversary.params.comm_kwargs.role_aware=true",
    "experiment.lrr=0.01",
    "sead=3",
    "seed=1.5",
    "seed=true",
    "seed=-1",
    "experiment.gamma=.nan",
    "experiment.lr=.inf",
    "algorithm_config.params.entropy_coef=.nan",
    "algorithm_config.params.clip_epsilon=-0.1",
    "experiment.evaluation_episodes=0",
    "experiment.on_policy_n_envs_per_worker=1.5",
    "experiment.evaluation='false'",
    "model_config.groups.adversary.params.hidden_dim=true",
    "model_config.groups.adversary.params.comm_kwargs.message_dim=2.5",
    "model_config.groups.adversary.params.comm_kwargs.num_heads=3",
    "model_config.groups.adversary.params.comm_kwargs.temperature=.inf",
    "model_config.groups.adversary.params.comm_kwargs.sender_selection=missing",
    "model_config.groups.adversary.params.comm_kwargs.channel={type: dropout, p: .nan}",
    "task_config.return_groups=[adversary,adversary]",
    "task_config.return_groups=[agents]",
    "task_config.return_groups=[]",
    "model_config.groups.misspelled={model_type: benchmarl_mlp}",
])
def test_invalid_cli_configuration_is_rejected(config_root, override):
    with pytest.raises((ValueError, TypeError)):
        load_experiment_spec(config_root, [*PCP, override])


def test_grouped_override_changes_real_constructor(config_root):
    spec = load_experiment_spec(config_root, [
        *PCP, "model_config.groups.adversary.params.comm_kwargs.num_heads=8",
    ])
    model = build_model_config(spec.model_config)
    assert model.model_configs_map["adversary"].comm_kwargs["num_heads"] == 8


@pytest.mark.parametrize("params", [
    {"num_good_agents": 0}, {"num_adversaries": -1}, {"num_landmarks": -1},
    {"num_adversaries": 2.5}, {"num_adversaries": True},
    {"prey_detection_radius": 0}, {"prey_detection_radius": float("inf")},
    {"predator_sensing_radius": -1}, {"predator_sensing_radius": float("nan")},
    {"prey_noise_std": -1}, {"prey_boundary_margin": 0},
    {"prey_boundary_margin": 1.1}, {"prey_min_flee_frac": 1.1},
    {"observe_pos": False}, {"observe_vel": False}, {"num_obstacles": 4},
])
def test_pcp_validation_covers_registry_and_direct_vmas(params):
    with pytest.raises(ValueError):
        resolve_task("vmas_predator_capture_prey", params)
    with pytest.raises(ValueError):
        PredatorCapturePreyScenario().make_world(1, "cpu", **params)


def test_pcp_typed_loader_eliminates_missing_schema_warning():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        task = CustomVmasTask.PREDATOR_CAPTURE_PREY.get_from_yaml()
    assert not any("TaskConfig" in str(item.message) for item in recorded)
    assert task.config["adversaries_share_rew"] is True
    assert task.config["observe_pos"] is True
    with pytest.raises(ValueError, match="num_obstacles"):
        CustomVmasTask.PREDATOR_CAPTURE_PREY.get_task({"num_obstacles": 4})


@pytest.mark.parametrize("radius", [None, 0, 1.0])
def test_pcp_visibility_modes_are_explicit_valid_settings(radius):
    task = resolve_task("vmas_predator_capture_prey", {"predator_sensing_radius": radius})
    assert task.config["predator_sensing_radius"] == radius


def test_saved_snapshot_round_trip_is_independent_of_project_yaml(config_root, monkeypatch):
    spec = load_experiment_spec(config_root, PCP)
    exported = resolved_experiment_dict(spec)
    serialized = OmegaConf.to_yaml(OmegaConf.create(exported), sort_keys=True)
    reconstructed = OmegaConf.to_container(OmegaConf.create(serialized), resolve=True)
    monkeypatch.setattr("commstudy.experiments.config._load_yaml", lambda *args: pytest.fail(
        "Snapshot reconstruction must not load current project YAML"
    ))
    restored = experiment_spec_from_dict(reconstructed)
    assert scientific_config_sha256(restored) == scientific_config_sha256(spec)
    assert "prey_noise_std" in restored.task_config["params"]
    assert "adam_eps" in restored.experiment
    assert restored.model_config["groups"]["adversary"]["params"]["num_roles"] == 2


def test_snapshot_rejects_partial_or_unknown_fields(config_root):
    raw = asdict(load_experiment_spec(config_root))
    partial = deepcopy(raw)
    partial.pop("critic_model")
    with pytest.raises(ValueError, match="missing.*critic_model"):
        experiment_spec_from_dict(partial)
    with pytest.raises(ValueError, match="unknown.*extra"):
        experiment_spec_from_dict({**raw, "extra": True})


def test_scientific_hash_captures_effective_values_and_leaves_rng_unchanged(config_root):
    spec = load_experiment_spec(config_root, PCP)
    before = torch.get_rng_state().clone()
    digest = scientific_config_sha256(spec)
    assert torch.equal(before, torch.get_rng_state())
    runtime = replace(spec, experiment={**spec.experiment, "train_device": "cuda",
                                       "save_folder": "/tmp/new-output", "loggers": []})
    assert scientific_config_sha256(runtime) == digest
    assert scientific_config_sha256(replace(spec, seed=17)) != digest
    assert scientific_config_sha256(replace(spec, seed=17), include_seed=False) == (
        scientific_config_sha256(spec, include_seed=False)
    )
    changed = replace(spec, task_config={**spec.task_config, "params": {
        **spec.task_config["params"], "prey_noise_std": 0.3,
    }})
    assert scientific_config_sha256(changed) != digest
    assert "seed" not in canonical_scientific_config(spec, include_seed=False)


def test_all_shipped_actor_and_critic_documents_validate(config_root):
    for directory in ("models", "critic_models"):
        for path in (config_root / directory).glob("*.yaml"):
            document = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            build_model_config(document)


@pytest.mark.parametrize("params", [
    {"n_agents": True}, {"n_agents": 0}, {"max_steps": 1.5}, {"unknown": 3},
])
def test_stock_task_overrides_are_strict_too(params):
    with pytest.raises(ValueError):
        resolve_task("vmas_simple_spread", params)
