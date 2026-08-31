import inspect
from dataclasses import fields

import pytest
import torch
from torch import nn
from torchrl.data import Composite, Unbounded

from benchmarl.models import MlpConfig
from benchmarl.models.common import ModelConfig

from commstudy.experiments import build_model_config
from commstudy.models import CommPolicyConfig, CommPolicyModel


N_AGENTS = 3
OBS_DIM = 8
OUT_DIM = 4
GROUP = "agents"


def load_model_yaml(config_root, name):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(
        OmegaConf.load(config_root / "models" / f"{name}.yaml"),
        resolve=True,
    )


def test_benchmarl_mlp_yaml_builds_mlp_config(config_root):
    config = build_model_config(
        load_model_yaml(config_root, "benchmarl_mlp")
    )

    assert isinstance(config, MlpConfig)
    assert config.num_cells == [128, 128]

    # Import paths must be resolved to real classes, not left as strings.
    assert config.layer_class is nn.Linear
    assert config.activation_class is nn.Tanh
    assert config.norm_class is None


def test_comm_identity_yaml_builds_comm_policy_config(config_root):
    config = build_model_config(
        load_model_yaml(config_root, "comm_identity")
    )

    assert isinstance(config, CommPolicyConfig)
    assert isinstance(config, ModelConfig)
    assert config.hidden_dim == 128
    assert config.num_encoder_layers == 2
    assert config.comm_class_path.endswith("IdentityComm")


def test_unknown_model_type_raises():
    with pytest.raises(ValueError) as excinfo:
        build_model_config({"model_type": "nonsense"})

    assert "nonsense" in str(excinfo.value)


def test_comm_policy_config_associated_class():
    assert CommPolicyConfig.associated_class() is CommPolicyModel


def test_comm_policy_config_fields_match_model_signature():
    """
    ``ModelConfig.get_model`` forwards ``asdict(self)`` into the model,
    so every dataclass field must be a named model argument.
    """
    config_fields = {field.name for field in fields(CommPolicyConfig)}

    parameters = inspect.signature(CommPolicyModel.__init__).parameters

    model_arguments = {
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }

    assert config_fields == model_arguments


def test_get_model_passes_config_values_through(config_root):
    config = build_model_config(
        load_model_yaml(config_root, "comm_identity")
    )

    model = config.get_model(
        input_spec=Composite(
            {
                GROUP: Composite(
                    {"observation": Unbounded(shape=(N_AGENTS, OBS_DIM))},
                    shape=(N_AGENTS,),
                )
            }
        ),
        output_spec=Composite(
            {
                GROUP: Composite(
                    {"logits": Unbounded(shape=(N_AGENTS, OUT_DIM))},
                    shape=(N_AGENTS,),
                )
            }
        ),
        agent_group=GROUP,
        input_has_agent_dim=True,
        n_agents=N_AGENTS,
        centralised=False,
        share_params=True,
        device="cpu",
        action_spec=Composite(),
    )

    assert isinstance(model, CommPolicyModel)
    assert model.hidden_dim == config.hidden_dim
    assert model.comm.__class__.__name__ == "IdentityComm"
    assert len(model.encoders[0]) == 2 * config.num_encoder_layers
    assert model.output_features == OUT_DIM


def test_comm_policy_config_rejects_critic_usage(config_root):
    """
    BenchMARL sets ``is_critic`` on whatever config is used as a critic.
    Doing that with CommPolicyConfig must fail loudly rather than
    silently build a centralized critic out of the policy model.
    """
    config = build_model_config(
        load_model_yaml(config_root, "comm_identity")
    )
    config.is_critic = True

    with pytest.raises(ValueError):
        config.get_model(
            input_spec=Composite(
                {
                    GROUP: Composite(
                        {"observation": Unbounded(shape=(N_AGENTS, OBS_DIM))},
                        shape=(N_AGENTS,),
                    )
                }
            ),
            output_spec=Composite(
                {
                    GROUP: Composite(
                        {"state_value": Unbounded(shape=(N_AGENTS, 1))},
                        shape=(N_AGENTS,),
                    )
                }
            ),
            agent_group=GROUP,
            input_has_agent_dim=True,
            n_agents=N_AGENTS,
            centralised=True,
            share_params=True,
            device="cpu",
            action_spec=Composite(),
        )


@pytest.mark.parametrize("output_dim", [4, 6, 9])
def test_output_size_follows_output_spec(config_root, output_dim):
    """
    The model must not assume what the output means: MAPPO asks for
    distribution parameters, QMIX will ask for per-agent action values.
    """
    config = build_model_config(
        load_model_yaml(config_root, "comm_identity")
    )

    model = config.get_model(
        input_spec=Composite(
            {
                GROUP: Composite(
                    {"observation": Unbounded(shape=(N_AGENTS, OBS_DIM))},
                    shape=(N_AGENTS,),
                )
            }
        ),
        output_spec=Composite(
            {
                GROUP: Composite(
                    {"logits": Unbounded(shape=(N_AGENTS, output_dim))},
                    shape=(N_AGENTS,),
                )
            }
        ),
        agent_group=GROUP,
        input_has_agent_dim=True,
        n_agents=N_AGENTS,
        centralised=False,
        share_params=True,
        device="cpu",
        action_spec=Composite(),
    )

    from tensordict import TensorDict

    for batch_shape in ([], [5], [7, 5]):
        td = TensorDict({}, batch_size=batch_shape)
        td.set(
            (GROUP, "observation"),
            torch.randn(*batch_shape, N_AGENTS, OBS_DIM),
        )

        out = model(td).get((GROUP, "logits"))

        assert out.shape == (*batch_shape, N_AGENTS, output_dim)
