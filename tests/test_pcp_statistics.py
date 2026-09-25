from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from commstudy.tasks.pcp_statistics import (
    domain_episode_metrics,
    domain_transition_series,
    summarize_domain_episodes,
)
from commstudy.tasks.pcp_metrics import pcp_metric_specs


def episode(contacts=(0, 2, 1, 0), *, complete=True):
    steps = len(contacts)
    fields = {name: torch.zeros(steps, 3, 1) for name in pcp_metric_specs(3)}
    pairs = torch.tensor(contacts, dtype=torch.float32)
    fields["contact_pairs"] = pairs[:, None, None].expand(-1, 3, 1).clone()
    fields["any_contact"] = (pairs > 0)[:, None, None].float().expand(-1, 3, 1).clone()
    fields["simultaneous_contact"] = (pairs >= 2)[:, None, None].float().expand(-1, 3, 1)
    fields["contacting_predators"] = fields["contact_pairs"].clone()
    fields["expected_predator_reward"] = fields["contact_pairs"] * 10
    fields["prey_visible_to_1_predators"] = torch.ones(steps, 3, 1)
    fields["sender_visible_receiver_blind_pairs"] = torch.full((steps, 3, 1), 2.0)
    fields["sender_visible_receiver_blind_prey_triples"] = torch.full((steps, 3, 1), 2.0)
    done = torch.zeros(steps, 1, dtype=torch.bool)
    done[-1] = complete
    return TensorDict(
        {
            "next": TensorDict(
                {
                    "adversary": TensorDict(
                        {
                            "reward": pairs[:, None, None].expand(-1, 3, 1) * 10,
                            "info": TensorDict(fields, batch_size=[steps, 3]),
                        },
                        batch_size=[steps, 3],
                    ),
                    "done": done,
                    "truncated": done.clone(),
                    "terminated": torch.zeros_like(done),
                },
                batch_size=[steps],
            )
        },
        batch_size=[steps],
    )


def test_episode_contact_accounting_counts_team_statistic_once():
    result = domain_episode_metrics(episode())
    assert result["return"] == 30
    assert result["contact_pair_steps"] == 3
    assert result["contact_duration_steps"] == 2
    assert result["first_contact_step_or_horizon"] == 2
    assert result["simultaneous_contact_steps"] == 1
    assert result["any_contact"] == 1
    assert result["reward_accounting_max_abs_error"] == 0
    assert result["truncated"] == 1 and result["terminated"] == 0
    assert result["sender_visible_receiver_blind_pairs_mean"] == 2


def test_no_contact_is_censored_and_not_dropped_from_probability():
    contact = domain_episode_metrics(episode())
    no_contact = domain_episode_metrics(episode((0, 0, 0, 0)))
    assert no_contact["first_contact_step_or_horizon"] == 4
    assert no_contact["first_contact_observed"] == 0
    summary = summarize_domain_episodes([{"metrics": contact}, {"metrics": no_contact}])
    assert summary["means"]["any_contact"] == 0.5
    assert summary["contact_episode_count"] == 1
    assert summary["first_contact_step_conditional_mean"] == 2
    assert summary["means"]["first_contact_step_or_horizon"] == 3
    assert summary["transition_count"] == 8


@pytest.mark.parametrize("defect", ["disagree", "missing", "nan", "negative", "fractional"])
def test_domain_corruption_fails_instead_of_averaging_it_away(defect):
    rollout = episode()
    key = ("next", "adversary", "info", "contact_pairs")
    if defect == "missing":
        rollout.del_(key)
    elif defect == "disagree":
        rollout[key][0, 1] = 1
    else:
        rollout[key][0] = {"nan": float("nan"), "negative": -1, "fractional": 0.5}[defect]
    with pytest.raises(ValueError):
        domain_transition_series(rollout)


def test_incomplete_or_post_terminal_episode_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        domain_episode_metrics(episode(complete=False))
    rollout = episode()
    rollout["next", "done"][1] = True
    with pytest.raises(ValueError, match="earlier done"):
        domain_episode_metrics(rollout)


def test_reward_error_remains_visible_and_episode_count_is_explicit():
    rollout = episode()
    rollout["next", "adversary", "reward"][2] += 10
    result = domain_episode_metrics(rollout)
    assert result["reward_accounting_max_abs_error"] == 10
    summary = summarize_domain_episodes([{"metrics": result}])
    assert summary["episode_count"] == summary["complete_episodes"] == 1
    assert summary["reward_accounting_max_abs_error"] == 10
