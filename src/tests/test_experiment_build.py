import dataclasses
from pathlib import Path

import pytest

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment
from benchmarl.models.mlp import Mlp

from commstudy.experiments import build_experiment, load_experiment_spec
from commstudy.models import CommPolicyModel


GROUP = "agents"

# Smallest configuration that still exercises the full assembly path
# without running a long rollout.
SHORT_RUN_OVERRIDES = [
    "experiment.max_n_frames=6000",
    "experiment.evaluation=false",
]


def spec_for(config_root, model, tmp_path):
    spec = load_experiment_spec(
        config_root,
        [f"model={model}", *SHORT_RUN_OVERRIDES],
    )

    # Keep BenchMARL's run folder out of the working directory.
    # BenchMARL creates `save_folder / run_name` with parents=False,
    # so save_folder itself has to exist first.
    Path(tmp_path).mkdir(parents=True, exist_ok=True)

    return dataclasses.replace(
        spec,
        experiment={**spec.experiment, "save_folder": str(tmp_path)},
    )


def actor_models(experiment):
    return [
        module
        for module in experiment.group_policies[GROUP].modules()
        if isinstance(module, (Mlp, CommPolicyModel))
    ]


def critic_models(experiment):
    return [
        module
        for module in experiment.losses[GROUP].critic_network.modules()
        if isinstance(module, (Mlp, CommPolicyModel))
    ]


@pytest.fixture(params=["benchmarl_mlp", "comm_identity"])
def built(request, config_root, tmp_path):
    experiment = build_experiment(
        spec_for(config_root, request.param, tmp_path)
    )

    yield request.param, experiment

    experiment.collector.shutdown()


def test_experiment_constructs(built):
    _, experiment = built

    assert isinstance(experiment, Experiment)
    assert isinstance(experiment.algorithm_config, MappoConfig)
    assert experiment.task.name == "SIMPLE_SPREAD"

    # Construction must not have started training.
    assert experiment.total_frames == 0
    assert experiment.n_iters_performed == 0


def test_intended_actor_model_is_injected(built):
    model, experiment = built

    expected = Mlp if model == "benchmarl_mlp" else CommPolicyModel

    actors = actor_models(experiment)

    assert len(actors) == 1
    assert isinstance(actors[0], expected)


def test_critic_is_always_the_benchmarl_mlp(built):
    """
    CommPolicyModel is a decentralized actor only. BenchMARL must not
    be left to clone it as the centralized MAPPO critic.
    """
    _, experiment = built

    critics = critic_models(experiment)

    assert len(critics) == 1
    assert isinstance(critics[0], Mlp)
    assert critics[0].centralised
    assert critics[0].is_critic


def test_comm_actor_uses_identity_comm(config_root, tmp_path):
    experiment = build_experiment(
        spec_for(config_root, "comm_identity", tmp_path)
    )

    actor = actor_models(experiment)[0]

    assert isinstance(actor, CommPolicyModel)
    assert actor.comm.__class__.__name__ == "IdentityComm"
    assert actor.comm.message_bits() == 0
    assert not actor.centralised
    assert not actor.is_critic

    experiment.collector.shutdown()


def test_comparison_differs_only_in_the_actor(config_root, tmp_path):
    """
    Everything except the policy model must be identical between the
    two baseline experiments.
    """
    mlp = load_experiment_spec(config_root, ["model=benchmarl_mlp"])
    comm = load_experiment_spec(config_root, ["model=comm_identity"])

    differing = {
        field.name
        for field in dataclasses.fields(mlp)
        if getattr(mlp, field.name) != getattr(comm, field.name)
    }

    assert differing == {"model", "model_config"}


def test_actors_are_parameter_matched(config_root, tmp_path):
    """
    IdentityComm adds no parameters, so the communication actor should
    be the same size as the MLP baseline.
    """
    built = {}

    for model in ("benchmarl_mlp", "comm_identity"):
        experiment = build_experiment(
            spec_for(config_root, model, tmp_path / model)
        )
        actor = actor_models(experiment)[0]
        built[model] = [tuple(p.shape) for p in actor.parameters()]
        experiment.collector.shutdown()

    assert built["benchmarl_mlp"] == built["comm_identity"]
