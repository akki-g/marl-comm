"""The study's return must come from the trained groups only.

PCP trains two BenchMARL groups but evaluates one: its ``agent`` group is a
scripted prey whose policy output is discarded every step. ``simple_tag``
rewards are exactly zero-sum between the two groups, so averaging over both
gives a return of exactly zero regardless of what the predators learn. These
tests pin the selection, the reduction formulas, and the single-group behaviour
that keeps Simple Spread numerically unchanged.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from commstudy.experiments.returns import (
    RETURN_GROUPS_KEY,
    group_collection_returns,
    group_rollout_returns,
    mean_over_groups,
    resolve_return_groups,
)


PREDATORS = "adversary"
PREY = "agent"


def _two_group_rollout(predator_reward: float, prey_reward: float, steps: int = 4):
    """One rollout with a per-step reward for each group."""

    return TensorDict(
        {
            "next": TensorDict(
                {
                    PREDATORS: TensorDict(
                        {"reward": torch.full((steps, 3, 1), predator_reward)},
                        batch_size=[steps, 3],
                    ),
                    PREY: TensorDict(
                        {"reward": torch.full((steps, 1, 1), prey_reward)},
                        batch_size=[steps, 1],
                    ),
                },
                batch_size=[steps],
            )
        },
        batch_size=[steps],
    )


def test_no_declaration_measures_every_group():
    """Omitting the key keeps the previous all-groups behaviour."""

    assert resolve_return_groups({"agents": []}) == ("agents",)
    assert resolve_return_groups({PREDATORS: [], PREY: []}) == (PREDATORS, PREY)


def test_declared_groups_are_selected_in_order():
    assert resolve_return_groups({PREDATORS: [], PREY: []}, [PREDATORS]) == (PREDATORS,)


def test_unknown_group_fails_loudly():
    """A typo must not silently produce a study with no headline metric."""

    with pytest.raises(ValueError) as excinfo:
        resolve_return_groups({PREDATORS: [], PREY: []}, ["adversaries"])

    message = str(excinfo.value)
    assert "adversaries" in message
    assert PREDATORS in message


def test_empty_declaration_fails_loudly():
    with pytest.raises(ValueError) as excinfo:
        resolve_return_groups({PREDATORS: []}, [])

    assert RETURN_GROUPS_KEY in str(excinfo.value)


def test_rollout_return_sums_time_and_averages_the_group():
    rollout = _two_group_rollout(predator_reward=0.5, prey_reward=-0.5, steps=4)

    returns = group_rollout_returns(rollout, (PREDATORS, PREY))

    assert returns[PREDATORS] == pytest.approx(2.0)
    assert returns[PREY] == pytest.approx(-2.0)


def test_zero_sum_groups_cancel_when_both_are_measured():
    """The exact failure this selection exists to prevent."""

    rollout = _two_group_rollout(predator_reward=0.5, prey_reward=-0.5)

    both = mean_over_groups(group_rollout_returns(rollout, (PREDATORS, PREY)))
    predators_only = mean_over_groups(group_rollout_returns(rollout, (PREDATORS,)))

    assert both == pytest.approx(0.0)
    assert predators_only == pytest.approx(2.0)


def test_collection_return_reads_episode_reward_at_the_global_done():
    """Mirrors BenchMARL's own per-group collection figure."""

    steps = 3
    done = torch.zeros(steps, 1, dtype=torch.bool)
    done[-1] = True
    batch = TensorDict(
        {
            "next": TensorDict(
                {
                    "done": done,
                    PREDATORS: TensorDict(
                        {
                            "episode_reward": torch.tensor(
                                [[[1.0], [1.0], [1.0]],
                                 [[5.0], [5.0], [5.0]],
                                 [[9.0], [9.0], [9.0]]]
                            )
                        },
                        batch_size=[steps, 3],
                    ),
                },
                batch_size=[steps],
            )
        },
        batch_size=[steps],
    )

    returns = group_collection_returns(batch, (PREDATORS,))

    # Only the finished episode counts, averaged over the group's agents.
    assert returns[PREDATORS] == pytest.approx(9.0)


def test_collection_return_is_absent_when_no_episode_ended():
    """BenchMARL reports NaN here; a NaN must not reach the analysis."""

    steps = 2
    batch = TensorDict(
        {
            "next": TensorDict(
                {
                    "done": torch.zeros(steps, 1, dtype=torch.bool),
                    PREDATORS: TensorDict(
                        {"episode_reward": torch.ones(steps, 3, 1)},
                        batch_size=[steps, 3],
                    ),
                },
                batch_size=[steps],
            )
        },
        batch_size=[steps],
    )

    assert group_collection_returns(batch, (PREDATORS,)) == {}
    assert mean_over_groups({}) is None


@pytest.mark.parametrize(
    ("task", "expected"),
    (
        ("vmas_simple_spread", ["agents"]),
        ("vmas_predator_capture_prey", ["adversary"]),
    ),
)
def test_task_configs_declare_their_measured_groups(config_root, task, expected):
    document = OmegaConf.to_container(
        OmegaConf.load(config_root / "tasks" / f"{task}.yaml"), resolve=True
    )

    assert document[RETURN_GROUPS_KEY] == expected


def _run_short_experiment(config_root, tmp_path, *, task, model, critic_model):
    """One real MAPPO iteration with the metrics callback attached."""

    import dataclasses

    from commstudy.experiments import build_experiment, load_experiment_spec
    from commstudy.experiments.metrics import ExperimentMetricsCallback

    spec = load_experiment_spec(
        config_root,
        [
            f"task={task}",
            f"model={model}",
            f"critic_model={critic_model}",
            "seed=0",
            "experiment.max_n_frames=6000",
            "experiment.on_policy_n_minibatch_iters=1",
            "experiment.evaluation=true",
            "experiment.evaluation_interval=6000",
            "experiment.evaluation_episodes=2",
            "experiment.loggers=[]",
            "experiment.create_json=false",
            "experiment.checkpoint_at_end=false",
        ],
    )
    spec = dataclasses.replace(
        spec,
        experiment={**spec.experiment, "save_folder": str(tmp_path / "benchmarl")},
    )
    callback = ExperimentMetricsCallback(
        tmp_path,
        return_groups=spec.task_config.get(RETURN_GROUPS_KEY),
    )
    experiment = build_experiment(spec, callbacks=[callback])
    experiment.run()
    return experiment


def _return_series(tmp_path, phase, group):
    import csv

    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as file:
        return [
            float(row["value"])
            for row in csv.DictReader(file)
            if row["phase"] == phase
            and row["metric"] == "return_mean"
            and row["group"] == group
            and not row["sample"]
        ]


def test_simple_spread_return_is_unchanged_by_the_group_selection(
    config_root,
    tmp_path,
):
    """The reference study must be bit-identical, not merely similar.

    Simple Spread has one group, so selecting it has to reproduce BenchMARL's
    own all-groups aggregate exactly. Anything else would silently reinterpret
    the completed V2 results.
    """

    experiment = _run_short_experiment(
        config_root,
        tmp_path,
        task="vmas_simple_spread",
        model="benchmarl_mlp",
        critic_model="benchmarl_mlp",
    )

    study = _return_series(tmp_path, "collection", "")
    agents = _return_series(tmp_path, "collection", "agents")

    assert study
    assert study == agents
    assert study[-1] == pytest.approx(experiment.mean_return, rel=1e-6)


def test_pcp_return_is_the_predators_and_not_the_zero_sum_average(
    config_root,
    tmp_path,
):
    """The same code path on the task that exposed the defect.

    simple_tag's two groups are exactly zero-sum, so the previous all-groups
    average was 0.0 whatever the predators did. The study figure must now be the
    predators' own return, while the scripted prey stays visible as its own row.
    """

    _run_short_experiment(
        config_root,
        tmp_path,
        task="vmas_predator_capture_prey",
        model="pcp_actor",
        critic_model="pcp_critic",
    )

    study = _return_series(tmp_path, "collection", "")
    predators = _return_series(tmp_path, "collection", PREDATORS)
    prey = _return_series(tmp_path, "collection", PREY)

    assert study
    assert study == predators
    # The excluded group is still recorded, so a mis-declaration is visible.
    assert len(prey) == len(predators)
    for predator_value, prey_value in zip(predators, prey, strict=True):
        # Zero-sum: this is exactly why averaging the two reported 0.0.
        assert predator_value == pytest.approx(-prey_value)
        assert (predator_value + prey_value) / 2 == pytest.approx(0.0)
