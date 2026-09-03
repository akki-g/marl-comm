from __future__ import annotations

import dataclasses

import pytest
import torch
from benchmarl.models import EnsembleModelConfig, MlpConfig
from benchmarl.models.mlp import Mlp
from omegaconf import OmegaConf

from commstudy.experiments import (
    build_experiment,
    build_model_config,
    load_experiment_spec,
)
from commstudy.experiments.sweeps import expand_suite_config
from commstudy.models import CommPolicyConfig, CommPolicyModel
from commstudy.tasks import resolve_task
from commstudy.utils.imports import import_from_path


PCP_TASK = "vmas_predator_capture_prey"

#: The trained predators. Every communication question on this task is about
#: this group; the ``agent`` group is the scripted prey, whose policy output is
#: discarded by ``PredatorCapturePreyScenario.process_action`` every step.
ADVERSARY_GROUP = "adversary"
PREY_GROUP = "agent"

PCP_COMM_MODELS = (
    "pcp_comm_identity",
    "pcp_comm_broadcast",
    "pcp_comm_gated",
    "pcp_comm_attention",
    "pcp_comm_graph",
)

PCP_ABLATION_SUITES = {
    "pcp_comm_stage2_message_dim.yaml": 48,
    "pcp_comm_stage2_dropout.yaml": 48,
    "pcp_comm_stage3_rounds.yaml": 18,
    "pcp_comm_stage3_heads.yaml": 24,
    "pcp_comm_stage4_graph_topology.yaml": 9,
    "pcp_comm_stage4_sender_budget.yaml": 36,
    "pcp_comm_stage4_self_communication.yaml": 18,
}

PCP_SUITES = {
    "pcp_identity_pilot.yaml": 2,
    "pcp_comm_main.yaml": 30,
    **PCP_ABLATION_SUITES,
}


def _load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def load_component_yaml(config_root, group, name):
    return _load_yaml(config_root / group / f"{name}.yaml")


def test_pcp_task_layer_uses_the_landmark_kwarg_vmas_consumes(config_root):
    """The project's ``params`` layer must reach VMAS under its real name.

    ``simple_tag.make_world`` reads ``num_landmarks``; an unknown kwarg is only
    warned about by ``ScenarioUtils.check_kwargs_consumed``, so a misnamed key
    would be dropped instead of failing the run.
    """
    spec = load_experiment_spec(config_root, [f"task={PCP_TASK}"])
    task = resolve_task(spec.task, spec.task_config["params"])

    assert task.config["num_landmarks"] == 2
    assert "num_obstacles" not in task.config
    assert task.config["num_adversaries"] == 3
    assert task.config["num_good_agents"] == 1


@pytest.mark.parametrize("model", PCP_COMM_MODELS)
def test_pcp_comm_yaml_builds_a_communication_adversary_group(config_root, model):
    """The grouped branch of ``build_model_config`` must keep the groups apart.

    A flat merge would hand the predators an MLP and silently drop the
    communication module the run is named after.
    """
    config = build_model_config(
        load_component_yaml(config_root, "models", model)
    )

    assert isinstance(config, EnsembleModelConfig)
    assert set(config.model_configs_map) == {ADVERSARY_GROUP, PREY_GROUP}

    adversary = config.model_configs_map[ADVERSARY_GROUP]
    assert isinstance(adversary, CommPolicyConfig)
    assert not isinstance(adversary, MlpConfig)
    assert adversary.hidden_dim == 128
    assert adversary.num_encoder_layers == 2
    assert adversary.comm_class_path == (
        load_component_yaml(config_root, "models", model.removeprefix("pcp_"))[
            "params"
        ]["comm_class_path"]
    )
    # Role conditioning is deferred: PCP's predators are homogeneous and the
    # scenario inherits stock simple_tag observations, which carry no class id.
    assert adversary.use_role_embedding is False

    prey = config.model_configs_map[PREY_GROUP]
    assert isinstance(prey, MlpConfig)
    assert prey.num_cells == [8]


def test_pcp_actor_and_critic_are_mlp_only_group_ensembles(config_root):
    actor = build_model_config(
        load_component_yaml(config_root, "models", "pcp_actor")
    )
    critic = build_model_config(
        load_component_yaml(config_root, "critic_models", "pcp_critic")
    )

    for config in (actor, critic):
        assert isinstance(config, EnsembleModelConfig)
        assert set(config.model_configs_map) == {ADVERSARY_GROUP, PREY_GROUP}
        assert all(
            isinstance(group_config, MlpConfig)
            for group_config in config.model_configs_map.values()
        )

    assert actor.model_configs_map[ADVERSARY_GROUP].num_cells == [128, 128]
    assert critic.model_configs_map[ADVERSARY_GROUP].num_cells == [128, 128]


@pytest.mark.parametrize(("filename", "expected_count"), PCP_SUITES.items())
def test_pcp_suites_expand_with_the_grouped_critic_selected(
    filename,
    expected_count,
    config_root,
    tmp_path,
):
    """Every PCP row must pin ``pcp_critic``.

    ``expand_suite_config`` emits only algorithm/task/model/seed/max_n_frames
    plus the ``overrides`` block, so a critic selected as a top-level suite key
    would never reach the run and the row would quietly fall back to
    ``base.yaml``'s flat default.
    """
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps" / filename),
        repo_root=tmp_path,
    )

    assert len(plans) == expected_count
    assert len({plan.run_id for plan in plans}) == expected_count
    for plan in plans:
        assert plan.suite_id == filename.removesuffix(".yaml")
        assert plan.task == PCP_TASK
        assert plan.algorithm == "mappo"
        assert plan.model.startswith("pcp_")
        assert 'critic_model="pcp_critic"' in plan.overrides

    spec = load_experiment_spec(config_root, plans[0].overrides)
    assert set(spec.critic_model) == {"groups"}
    assert set(spec.critic_model["groups"]) == {ADVERSARY_GROUP, PREY_GROUP}
    assert set(spec.model_config) == {"groups"}


def test_pcp_run_ids_are_unique_across_every_suite(config_root, tmp_path):
    run_ids = []
    for filename in PCP_SUITES:
        plans = expand_suite_config(
            _load_yaml(config_root / "sweeps" / filename),
            repo_root=tmp_path,
        )
        run_ids.extend(plan.run_id for plan in plans)

    assert len(run_ids) == sum(PCP_SUITES.values())
    assert len(run_ids) == len(set(run_ids))
    assert all(run_id.startswith("predator_capture_prey__") for run_id in run_ids)


def test_heads_ablation_overrides_the_adversary_group_not_a_stray_key(
    config_root,
    tmp_path,
):
    """Guards the silent no-op: OmegaConf would happily create ``params``.

    On a grouped config, ``model_config.params.comm_kwargs.num_heads`` merges in
    as a brand new top-level key that nothing reads, so the ablation would run
    every row at the default head count without any error.
    """
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/pcp_comm_stage3_heads.yaml"),
        repo_root=tmp_path,
    )

    assert {plan.ablation_value for plan in plans} == {"1", "2", "4", "8"}
    for plan in plans:
        spec = load_experiment_spec(config_root, plan.overrides)
        groups = spec.model_config["groups"]

        assert set(spec.model_config) == {"groups"}
        assert "params" not in spec.model_config

        kwargs = groups[ADVERSARY_GROUP]["params"]["comm_kwargs"]
        assert kwargs["num_heads"] == int(plan.ablation_value)
        assert kwargs["message_dim"] == 32
        assert kwargs["key_dim"] == 32

        # The scripted prey keeps its untouched MLP block.
        assert groups[PREY_GROUP]["model_type"] == "benchmarl_mlp"
        assert "comm_kwargs" not in groups[PREY_GROUP]["params"]


@pytest.mark.parametrize(
    ("filename", "expected_count"), PCP_ABLATION_SUITES.items()
)
def test_pcp_ablation_rows_construct_their_communication_module(
    filename,
    expected_count,
    config_root,
    tmp_path,
):
    """Resolve every row and build its module, as the V1/V2 ablation test does.

    This is what catches a stale or misspelled dot-override key before launch.
    """
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps" / filename),
        repo_root=tmp_path,
    )

    assert len(plans) == expected_count
    assert {plan.seed for plan in plans} == {0, 1, 2}
    assert {plan.max_n_frames for plan in plans} == {60_000}
    if "dropout" not in filename:
        assert all("dropout" not in " ".join(plan.overrides) for plan in plans)

    for plan in plans:
        spec = load_experiment_spec(config_root, plan.overrides)
        params = spec.model_config["groups"][ADVERSARY_GROUP]["params"]
        comm_class = import_from_path(params["comm_class_path"])
        comm_class(hidden_dim=params["hidden_dim"], **params["comm_kwargs"])


def test_mappo_real_iteration_trains_the_pcp_adversary_communication_module(
    config_root,
    tmp_path,
):
    """One real 6,000-frame MAPPO iteration on the two-group PCP environment.

    The reference task's integration tests exercise a real optimization step,
    so PCP gets the same treatment for one representative module: the
    communication actor has to see three adversaries, the scripted prey has to
    keep its plain MLP, and the channel has to receive gradient.
    """
    spec = load_experiment_spec(
        config_root,
        [
            f"task={PCP_TASK}",
            "model=pcp_comm_attention",
            "critic_model=pcp_critic",
            "seed=0",
            "experiment.max_n_frames=6000",
            "experiment.on_policy_n_minibatch_iters=1",
            "experiment.evaluation=false",
            "experiment.loggers=[]",
            "experiment.create_json=false",
            "experiment.checkpoint_at_end=false",
        ],
    )
    spec = dataclasses.replace(
        spec,
        experiment={**spec.experiment, "save_folder": str(tmp_path / "pcp")},
    )
    experiment = build_experiment(spec)

    assert set(experiment.group_map) == {ADVERSARY_GROUP, PREY_GROUP}
    assert len(experiment.group_map[ADVERSARY_GROUP]) == 3
    assert len(experiment.group_map[PREY_GROUP]) == 1

    actors = [
        module
        for module in experiment.group_policies[ADVERSARY_GROUP].modules()
        if isinstance(module, CommPolicyModel)
    ]
    prey_actors = [
        module
        for module in experiment.group_policies[PREY_GROUP].modules()
        if isinstance(module, (Mlp, CommPolicyModel))
    ]
    assert len(actors) == 1
    assert actors[0].n_agents == 3
    assert len(prey_actors) == 1
    assert isinstance(prey_actors[0], Mlp)

    before = {
        name: parameter.detach().clone()
        for name, parameter in actors[0].comm.named_parameters()
    }
    assert before

    experiment.run()

    assert experiment.total_frames == 6000
    assert experiment.n_iters_performed == 1
    after = dict(actors[0].comm.named_parameters())
    assert any(
        not torch.equal(before[name], after[name].detach()) for name in before
    )


def test_pcp_sweep_files_cover_every_communication_model(config_root, tmp_path):
    """The main comparison must include the baseline and all five modules."""
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/pcp_comm_main.yaml"),
        repo_root=tmp_path,
    )

    assert {plan.model for plan in plans} == {"pcp_actor", *PCP_COMM_MODELS}
    assert {plan.seed for plan in plans} == {0, 1, 2, 3, 4}
    assert {plan.ablation for plan in plans} == {"main"}
    assert {plan.max_n_frames for plan in plans} == {60_000}


def test_pilot_shares_the_main_comparisons_optimizer_protocol(config_root, tmp_path):
    """The pilot decides the main suite's budget, so it must differ only in that.

    The Simple Spread pilot this one mirrors predates the frozen protocol, so
    copying its settings verbatim would have measured a configuration no PCP
    suite runs -- gamma above all, which is itself one of the open questions on
    this task.
    """
    pilot = expand_suite_config(
        _load_yaml(config_root / "sweeps/pcp_identity_pilot.yaml"),
        repo_root=tmp_path,
    )
    main = expand_suite_config(
        _load_yaml(config_root / "sweeps/pcp_comm_main.yaml"),
        repo_root=tmp_path,
    )

    main_spec = load_experiment_spec(config_root, main[0].overrides)
    for plan in pilot:
        spec = load_experiment_spec(config_root, plan.overrides)
        assert spec.experiment["gamma"] == main_spec.experiment["gamma"]
        assert (
            spec.algorithm_config["params"]["entropy_coef"]
            == main_spec.algorithm_config["params"]["entropy_coef"]
        )
        assert (
            spec.experiment["on_policy_n_minibatch_iters"]
            == main_spec.experiment["on_policy_n_minibatch_iters"]
        )
        assert spec.experiment["evaluation_interval"] == 12_000

    # The budget is the variable under test.
    assert {plan.max_n_frames for plan in pilot} == {60_000, 120_000}


def _pcp_env(**kwargs):
    """A bare VMAS PCP env, bypassing BenchMARL so the scenario is under test."""
    from vmas import make_env

    from commstudy.tasks.vmas.scenarios.predator_capture_prey import (
        PredatorCapturePreyScenario,
    )

    params = {
        "max_steps": 100,
        "num_good_agents": 1,
        "num_adversaries": 3,
        "num_landmarks": 2,
    }
    params.update(kwargs)
    env = make_env(
        scenario=PredatorCapturePreyScenario(),
        num_envs=64,
        device="cpu",
        seed=0,
        **params,
    )
    env.reset()
    return env


def test_predator_sensing_radius_defaults_to_stock_full_observability():
    """Omitting the radius must not silently change the task.

    The 2026-09-03 pilot ran without it, so its numbers describe this
    observation, and `docs/RESULTS_pcp_pilot.md` compares against it.
    """
    env = _pcp_env()
    adversary = env.scenario.adversaries()[0]
    assert env.scenario.predator_sensing_radius is None
    # vel(2) + pos(2) + 2 landmarks(4) + 2 teammates(4) + prey pos(2) + prey vel(2)
    assert env.scenario.observation(adversary).shape[-1] == 16


def test_predator_sensing_radius_hides_only_the_prey_and_flags_it():
    """The masked block must be the prey's, found by layout not by index.

    This is Gate A of the PCP plan: without it every predator always sees the
    prey and a message can only re-transmit what its receiver already has.
    """
    radius = 1.0
    env = _pcp_env(predator_sensing_radius=radius)
    reference = _pcp_env()
    scenario, stock = env.scenario, reference.scenario
    adversary = scenario.adversaries()[0]
    prey = scenario.good_agents()[0]

    observation = scenario.observation(adversary)
    expected = stock.observation(stock.adversaries()[0])
    assert observation.shape[-1] == 17

    distance = (prey.state.pos - adversary.state.pos).norm(dim=-1)
    visible = observation[:, -1].bool()
    assert torch.equal(visible, distance <= radius)
    # Both regimes have to occur, or the assertions below prove nothing.
    assert bool(visible.any()) and bool((~visible).any())

    # Everything that is not the prey is untouched: self, landmarks, teammates.
    assert torch.equal(observation[:, :12], expected[:, :12])
    # The prey block is the stock one where visible, exactly zero where not.
    assert torch.equal(observation[visible][:, 12:16], expected[visible][:, 12:16])
    assert not observation[~visible][:, 12:16].any()


def test_predator_sensing_radius_layout_generalises_beyond_one_prey():
    """The mask is located by counting prey, not by hardcoding 12:16."""
    env = _pcp_env(
        predator_sensing_radius=1.0, num_good_agents=2, num_adversaries=4, num_landmarks=3
    )
    scenario = env.scenario
    adversary = scenario.adversaries()[0]
    observation = scenario.observation(adversary)

    # vel2 + pos2 + landmarks 3*2 + teammates 3*2 + prey pos 2*2 + prey vel 2*2 + 2 flags
    assert observation.shape[-1] == 26
    for index, prey in enumerate(scenario.good_agents()):
        distance = (prey.state.pos - adversary.state.pos).norm(dim=-1)
        assert torch.equal(observation[:, -2 + index].bool(), distance <= 1.0)


def test_pcp_task_config_ships_the_sensing_radius(config_root):
    """A radius that is set but never reaches the env is the worst outcome:
    the suite would look partially observable and run fully observable."""
    spec = load_experiment_spec(
        config_root,
        [f"task={PCP_TASK}", "model=pcp_comm_broadcast", "critic_model=pcp_critic"],
    )
    assert spec.task_config["params"]["predator_sensing_radius"] == 1.0

    experiment = build_experiment(spec)
    observation_dim = experiment.observation_spec[ADVERSARY_GROUP][
        "observation"
    ].shape[-1]
    assert observation_dim == 17
