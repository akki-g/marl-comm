import pytest

from benchmarl.algorithms import MappoConfig, QmixConfig

from commstudy.algorithms import available_algorithms, resolve_algorithm


def test_mappo_resolves_to_benchmarl_config():
    config = resolve_algorithm("mappo")

    assert isinstance(config, MappoConfig)


def test_qmix_resolves_to_benchmarl_config():
    config = resolve_algorithm("qmix")

    assert isinstance(config, QmixConfig)


def test_unknown_algorithm_raises_clearly():
    with pytest.raises(ValueError) as excinfo:
        resolve_algorithm("not_an_algorithm")

    message = str(excinfo.value)

    assert "not_an_algorithm" in message
    assert "mappo" in message
    assert "qmix" in message


def test_available_algorithms_is_sorted():
    assert available_algorithms() == ("mappo", "qmix")


def test_params_override_benchmarl_defaults():
    default = resolve_algorithm("mappo")

    config = resolve_algorithm(
        "mappo",
        {"clip_epsilon": 0.123},
    )

    assert default.clip_epsilon != 0.123
    assert config.clip_epsilon == 0.123

    # Untouched fields keep BenchMARL's defaults.
    assert config.lmbda == default.lmbda


def test_unknown_algorithm_param_raises():
    with pytest.raises(ValueError) as excinfo:
        resolve_algorithm(
            "mappo",
            {"lr": 0.001},
        )

    # Experiment-level settings must not be smuggled into the
    # algorithm config.
    assert "lr" in str(excinfo.value)


def test_experiment_level_settings_are_not_algorithm_fields():
    config = resolve_algorithm("mappo")

    for experiment_field in (
        "lr",
        "gamma",
        "max_n_frames",
        "on_policy_collected_frames_per_batch",
        "on_policy_n_envs_per_worker",
        "evaluation_interval",
        "train_device",
    ):
        assert not hasattr(config, experiment_field)
