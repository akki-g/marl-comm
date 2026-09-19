"""A visibility-independent PCP state for BenchMARL's centralized MLP critic.

The fixed ordering is all agents' absolute position and velocity (VMAS world
order), all landmarks' absolute position, then all scripted prey wander
directions. These are simulator fields, never reconstructed from masked actor
observations. The state contains no rewards, visibility masks, episode IDs,
future random draws, or diagnostics. Sensing interventions do not alter it.
"""

from __future__ import annotations

import torch
from commstudy.tasks.vmas.domain_metrics import PCP_DOMAIN_PROTOCOL
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs.transforms import Transform


CRITIC_STATE_PROTOCOL = "pcp_physical_state_v1"


def _state_fields(scenario):
    return [
        (agent.name, field, tensor)
        for agent in scenario.world.agents
        for field, tensor in (("position", agent.state.pos), ("velocity", agent.state.vel))
    ] + [
        (landmark.name, "position", landmark.state.pos) for landmark in scenario.world.landmarks
    ] + [(prey.name, "wander_direction", prey.wander_dir) for prey in scenario.good_agents()]


def pcp_critic_state(scenario) -> torch.Tensor:
    """Return a fresh [environments, features] tensor; expose no simulator aliases."""
    return torch.cat([tensor for _, _, tensor in _state_fields(scenario)], dim=-1)


def pcp_runtime_contract(env) -> dict:
    """Describe the actual task input layout for run and evaluation provenance."""
    scenario = env.scenario
    spec = env.full_observation_spec_unbatched
    offset = 0
    fields = []
    for entity, field, tensor in _state_fields(scenario):
        width = tensor.shape[-1]
        fields.append({"entity": entity, "field": field, "offset": offset, "width": width})
        offset += width
    return {
        "task": "predator_capture_prey",
        "randomness_protocol": scenario.randomness_protocol,
        "observation_protocol": scenario.observation_protocol,
        "domain_metrics_protocol": PCP_DOMAIN_PROTOCOL,
        "time_limit_protocol": "pcp_time_limit_v1",
        "predator_sensing_radius": scenario.predator_sensing_radius,
        "actor_observation_keys": {group: [[group, "observation"]] for group in env.group_map},
        "actor_observation_shapes": {
            group: list(spec[group, "observation"].shape) for group in env.group_map
        },
        "critic_state": {
            "protocol": CRITIC_STATE_PROTOCOL,
            "key": "state",
            "shape": list(spec["state"].shape),
            "source": "simulator physical state and scripted-prey wander state",
            "fields": fields,
            "visibility_independent": True,
        },
    }


class PCPStateTransform(Transform):
    """Publish privileged state at a separate root leaf on reset and step.

The task must select only this leaf in ``state_spec`` and only the declared
group observations in ``observation_spec``. This lets stock MAPPO choose its
supported state-based critic path without ever passing state to an actor.
"""

    def __init__(self, scenario):
        super().__init__(in_keys=[], out_keys=["state"])
        self.scenario = scenario

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.set("state", pcp_critic_state(self.scenario))
        return tensordict

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        result = observation_spec.clone()
        state = pcp_critic_state(self.scenario)
        result["state"] = Unbounded(shape=state.shape, dtype=state.dtype, device=state.device)
        return result
