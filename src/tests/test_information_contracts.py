"""Information boundaries: adversarial extra leaves and controlled PCP state."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from benchmarl.models.mlp import Mlp
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs import check_env_specs

from commstudy.experiments import build_experiment, load_experiment_spec
from commstudy.experiments.bookkeeping import parameter_counts, task_runtime_contract
from commstudy.models import CommPolicyConfig, CommPolicyModel
from commstudy.tasks import resolve_task
from commstudy.tasks.vmas.critic_state import pcp_critic_state


def _model(*, actor_keys=None, extra_spec=None, context=None):
    leaves = {"observation": Unbounded(shape=(3, 5))}
    leaves.update(extra_spec or {})
    return CommPolicyModel(
        hidden_dim=8,
        num_encoder_layers=2,
        activation_class_path="torch.nn.Tanh",
        comm_class_path="commstudy.communication.identity.IdentityComm",
        actor_observation_keys=actor_keys,
        comm_context_keys=context,
        input_spec=Composite({"adversary": Composite(leaves, shape=(3,))}),
        output_spec=Composite({
            "adversary": Composite({"logits": Unbounded(shape=(3, 4))}, shape=(3,))
        }),
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


def test_actor_contract_remains_compatible_with_framework_structured_configs():
    config = OmegaConf.structured(CommPolicyConfig)
    assert config.actor_observation_keys == ["observation"]


@pytest.mark.parametrize("leaf", ["diagnostic", "global_state", "reward"])
def test_undeclared_observation_spec_leaves_fail_closed(leaf):
    with pytest.raises(ValueError, match="Actor input spec violates"):
        _model(extra_spec={leaf: Unbounded(shape=(3, 2))})


@pytest.mark.parametrize("keys", [["state"], [["agent", "observation"]], [["state"]]])
def test_actor_contract_cannot_select_state_or_another_group(keys):
    with pytest.raises(ValueError, match="Privileged|agent group"):
        _model(actor_keys=keys)


def test_actor_contract_rejects_context_overlap_and_missing_inputs():
    with pytest.raises(ValueError, match="disjoint"):
        _model(context={"mask": "observation"})
    with pytest.raises(ValueError, match="missing="):
        _model(actor_keys=["observation", "missing"])


def test_declared_additional_local_input_changes_only_its_identity_receiver():
    model = _model(
        actor_keys=["observation", "local_sensor"],
        extra_spec={"local_sensor": Unbounded(shape=(3, 2))},
    )
    td = TensorDict({"adversary": {
        "observation": torch.randn(2, 3, 5), "local_sensor": torch.randn(2, 3, 2)
    }}, batch_size=[2])
    changed = td.clone()
    changed["adversary", "local_sensor"][:, 1] += 100
    before = model(td)["adversary", "logits"]
    after = model(changed)["adversary", "logits"]
    assert torch.equal(before[:, [0, 2]], after[:, [0, 2]])
    assert not torch.equal(before[:, 1], after[:, 1])


def _pcp_task(config_root, radius):
    spec = load_experiment_spec(config_root, ["task=vmas_predator_capture_prey"])
    params = {**spec.task_config["params"], "predator_sensing_radius": radius}
    return resolve_task(spec.task, params)


def test_pcp_critic_state_is_identical_across_visibility_conditions(config_root):
    tasks = [_pcp_task(config_root, radius) for radius in (0.0, None)]
    envs = [task.get_env_fun(2, True, 31, "cpu")() for task in tasks]
    try:
        resets = []
        for env in envs:
            env.set_seed(31)
            resets.append(env.reset())
            check_env_specs(env)
        # The spec checker advances its environment. Reestablish common state.
        for index, env in enumerate(envs):
            env.set_seed(31)
            resets[index] = env.reset()
        assert torch.equal(resets[0]["state"], resets[1]["state"])
        assert resets[0]["state"].shape == (2, 22)
        assert resets[0]["adversary", "observation"].shape == (2, 3, 17)
        assert resets[1]["adversary", "observation"].shape == (2, 3, 17)
        assert torch.equal(resets[0]["adversary", "observation"][..., -1], torch.zeros(2, 3))
        assert torch.equal(resets[1]["adversary", "observation"][..., -1], torch.ones(2, 3))
        assert not torch.equal(
            resets[0]["adversary", "observation"],
            resets[1]["adversary", "observation"],
        )
        for task, env in zip(tasks, envs, strict=True):
            assert set(task.state_spec(env).keys(True, True)) == {"state"}
            assert set(task.observation_spec(env).keys(True, True)) == {
                ("adversary", "observation"), ("agent", "observation")
            }
            info = task.info_spec(env)
            assert info is None or "state" not in info.keys(True, True)
        # Identical actions and seeded disturbances preserve matched physics,
        # rewards, and critic information after the visibility intervention.
        actions = envs[0].full_action_spec.zero()
        for _ in range(3):
            transitions = [env.step(td.update(actions.clone())) for env, td in
                           zip(envs, resets, strict=True)]
            next_td = [transition["next"] for transition in transitions]
            assert torch.equal(next_td[0]["state"], next_td[1]["state"])
            assert torch.equal(
                next_td[0]["adversary", "reward"], next_td[1]["adversary", "reward"]
            )
            resets = next_td
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("model_name", ["pcp_actor", "pcp_comm_identity"])
def test_real_pcp_actor_ignores_privileged_and_other_agent_inputs(
    config_root, tmp_path, model_name
):
    spec = load_experiment_spec(config_root, [
        "task=vmas_predator_capture_prey", f"model={model_name}",
        "critic_model=pcp_critic", "experiment.loggers=[]",
        "experiment.create_json=false", "experiment.evaluation=false",
        "experiment.checkpoint_at_end=false",
    ])
    spec = dataclasses.replace(
        spec, experiment={**spec.experiment, "save_folder": str(tmp_path / model_name)}
    )
    experiment = build_experiment(spec)
    env = experiment.test_env
    try:
        td = env.reset()
        actor = next(module for module in experiment.group_policies["adversary"].modules()
                     if isinstance(module, (Mlp, CommPolicyModel)))
        before = actor(td.clone())[actor.out_key].clone()
        changed = td.clone()
        changed["state"] = torch.full_like(changed["state"], 1e9)
        changed["agent", "observation"] = torch.full_like(
            changed["agent", "observation"], 1e9
        )
        changed["adversary", "diagnostic"] = torch.full(
            (*changed["adversary", "observation"].shape[:-1], 2), 1e9
        )
        assert torch.equal(before, actor(changed.clone())[actor.out_key])
        changed["adversary", "observation"][:, 1] += 100
        after = actor(changed)[actor.out_key]
        assert torch.equal(before[:, [0, 2]], after[:, [0, 2]])
        assert not torch.equal(before[:, 1], after[:, 1])

        critic = next(module for module in
                      experiment.losses["adversary"].critic_network.modules()
                      if isinstance(module, Mlp))
        assert critic.centralised and not critic.input_has_agent_dim
        assert critic.in_keys == ["state"]
        assert critic.input_spec["state"].shape == (22,)
        counts = parameter_counts(experiment)
        assert counts["group/adversary/actor_total"] > counts["group/agent/actor_total"]
        assert counts["group/adversary/communication_total"] == 0
        assert counts["group/adversary/critic_total"] == 19_585
        assert counts["group/agent/critic_total"] == 193
        for component in ("actor", "communication", "critic"):
            assert counts[f"{component}_total"] == sum(
                counts[f"group/{group}/{component}_total"] for group in ("adversary", "agent")
            )
        contract = task_runtime_contract(experiment)
        assert contract["randomness_protocol"] == "pcp_rng_v2"
        assert contract["observation_protocol"] == "pcp_visibility_v2"
        assert contract["evaluation_randomness_protocol"] == "evaluation_rng_v2"
        assert contract["channel_randomness_protocol"] == "channel_rng_v2"
        assert contract["actor_observation_keys"]["adversary"] == [["adversary", "observation"]]
        state = contract["critic_state"]
        assert state["protocol"] == "pcp_physical_state_v1"
        assert state["shape"] == [22]
        assert sum(field["width"] for field in state["fields"]) == 22
        assert state["fields"][-1]["field"] == "wander_direction"
    finally:
        env.close()


def test_critic_state_reads_true_velocity_and_returns_an_unaliased_tensor(config_root):
    task = _pcp_task(config_root, 0.0)
    env = task.get_env_fun(1, True, 9, "cpu")()
    try:
        td = env.reset()
        scenario = env.base_env.scenario
        before = pcp_critic_state(scenario)
        prey = scenario.good_agents()[0]
        prey.state.vel[:] += 10
        after = pcp_critic_state(scenario)
        assert not torch.equal(before, after)
        # TensorDict's stored state snapshot must survive subsequent simulation
        # changes; collection must never alias mutable simulator tensors.
        assert torch.equal(td["state"], before)
    finally:
        env.close()
