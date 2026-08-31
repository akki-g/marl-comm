import pytest

from commstudy.experiments import ExperimentSpec, load_experiment_spec


def test_defaults_come_from_base_yaml(config_root):
    spec = load_experiment_spec(config_root)

    assert isinstance(spec, ExperimentSpec)
    assert spec.algorithm == "mappo"
    assert spec.task == "vmas_simple_spread"
    assert spec.model == "benchmarl_mlp"
    assert spec.seed == 0


def test_documented_cli_invocation(config_root):
    spec = load_experiment_spec(
        config_root,
        [
            "algorithm=mappo",
            "task=vmas_simple_spread",
            "model=comm_identity",
            "seed=3",
            "experiment.max_n_frames=12000",
        ],
    )

    assert spec.algorithm == "mappo"
    assert spec.task == "vmas_simple_spread"
    assert spec.model == "comm_identity"
    assert spec.seed == 3

    # Selecting model=comm_identity must load configs/models/comm_identity.yaml
    assert spec.model_config["model_type"] == "communication"
    assert spec.model_config["params"]["hidden_dim"] == 128

    # Selecting algorithm=mappo must load configs/algorithms/mappo.yaml
    assert spec.algorithm_config["params"]["clip_epsilon"] == 0.2

    # Selecting task=vmas_simple_spread must load its yaml
    assert spec.task_config["params"]["n_agents"] == 3

    assert spec.experiment["max_n_frames"] == 12000


def test_cli_overrides_beat_algorithm_yaml(config_root):
    default = load_experiment_spec(config_root)

    assert default.experiment["max_n_frames"] == 120000
    assert default.experiment["lr"] == 0.00005

    spec = load_experiment_spec(
        config_root,
        [
            "experiment.max_n_frames=12000",
            "experiment.lr=0.001",
            "experiment.evaluation=false",
        ],
    )

    assert spec.experiment["max_n_frames"] == 12000
    assert spec.experiment["lr"] == 0.001
    assert spec.experiment["evaluation"] is False


def test_algorithm_yaml_beats_project_defaults(config_root):
    """
    base.yaml sets project-wide experiment defaults; the selected
    algorithm's `experiment` block layers on top of them.
    """
    spec = load_experiment_spec(config_root, ["algorithm=qmix"])

    # From base.yaml, untouched by qmix.yaml
    assert spec.experiment["train_device"] == "cpu"
    assert spec.experiment["loggers"] == ["csv"]

    # From qmix.yaml
    assert spec.experiment["prefer_continuous_actions"] is False
    assert spec.experiment["off_policy_train_batch_size"] == 128


def test_algorithm_config_holds_only_algorithm_params(config_root):
    """
    Training settings belong to BenchMARL's ExperimentConfig, not to
    the AlgorithmConfig.
    """
    spec = load_experiment_spec(config_root)

    assert set(spec.algorithm_config) == {"params"}

    params = spec.algorithm_config["params"]

    for experiment_field in (
        "lr",
        "gamma",
        "max_n_frames",
        "on_policy_collected_frames_per_batch",
        "evaluation_interval",
    ):
        assert experiment_field not in params


def test_critic_config_is_independent_of_model_selection(config_root):
    mlp = load_experiment_spec(config_root, ["model=benchmarl_mlp"])
    comm = load_experiment_spec(config_root, ["model=comm_identity"])

    assert mlp.critic_model == comm.critic_model
    assert mlp.critic_model["model_type"] == "benchmarl_mlp"


def test_malformed_override_raises(config_root):
    with pytest.raises(ValueError):
        load_experiment_spec(config_root, ["not_an_override"])


def test_unknown_component_selection_raises(config_root):
    with pytest.raises(FileNotFoundError):
        load_experiment_spec(config_root, ["model=does_not_exist"])
