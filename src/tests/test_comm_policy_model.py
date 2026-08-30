import torch

from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from commstudy.models.model import CommPolicyModel


N_AGENTS = 3
OBS_DIM = 8
HIDDEN_DIM = 16
N_ACTIONS = 5
BATCH_SIZE = 4

GROUP = "agents"


def make_specs():
    """
    Build minimal BenchMARL-style input/output specs.

    Input:
        [N, OBS_DIM]

    Output:
        [N, N_ACTIONS]
    """

    input_spec = Composite(
        {
            GROUP: Composite(
                {
                    "observation": Unbounded(
                        shape=(N_AGENTS, OBS_DIM)
                    )
                },
                shape=(N_AGENTS,),
            )
        }
    )

    output_spec = Composite(
        {
            GROUP: Composite(
                {
                    "logits": Unbounded(
                        shape=(N_AGENTS, N_ACTIONS)
                    )
                },
                shape=(N_AGENTS,),
            )
        }
    )

    return input_spec, output_spec


def make_model(
    *,
    share_params=True,
):
    input_spec, output_spec = make_specs()

    model = CommPolicyModel(
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=2,
        activation_class_path="torch.nn.Tanh",
        comm_class_path=(
            "commstudy.communication.identity.IdentityComm"
        ),
        comm_kwargs={},
        comm_context_keys={},

        # Arguments normally supplied automatically
        # by BenchMARL ModelConfig.get_model().
        input_spec=input_spec,
        output_spec=output_spec,
        agent_group=GROUP,
        input_has_agent_dim=True,
        n_agents=N_AGENTS,
        centralised=False,
        share_params=share_params,
        device="cpu",
        action_spec=Composite(),
        model_index=0,
        is_critic=False,
    )

    return model


def make_batch():
    obs = torch.randn(
        BATCH_SIZE,
        N_AGENTS,
        OBS_DIM,
    )

    td = TensorDict(
        {},
        batch_size=[BATCH_SIZE],
    )

    td.set(
        (GROUP, "observation"),
        obs,
    )

    return td


def test_model_instantiates():
    model = make_model()

    assert isinstance(
        model,
        CommPolicyModel,
    )


def test_model_uses_identity_comm():
    model = make_model()

    assert (
        model.comm.__class__.__name__
        == "IdentityComm"
    )


def test_model_forward_output_shape():
    model = make_model()

    td = make_batch()

    output_td = model(td)

    logits = output_td.get(
        (GROUP, "logits")
    )

    assert logits.shape == (
        BATCH_SIZE,
        N_AGENTS,
        N_ACTIONS,
    )


def test_model_writes_expected_output_key():
    model = make_model()

    td = make_batch()

    output_td = model(td)

    assert (
        GROUP,
        "logits",
    ) in output_td.keys(
        include_nested=True,
        leaves_only=True,
    )


def test_model_preserves_batch_and_agent_dimensions():
    model = make_model()

    td = make_batch()

    output_td = model(td)

    output = output_td.get(
        (GROUP, "logits")
    )

    assert output.shape[0] == BATCH_SIZE
    assert output.shape[1] == N_AGENTS


def test_model_supports_unbatched_input():
    model = make_model()

    obs = torch.randn(
        N_AGENTS,
        OBS_DIM,
    )

    td = TensorDict(
        {},
        batch_size=[],
    )

    td.set(
        (GROUP, "observation"),
        obs,
    )

    output_td = model(td)

    output = output_td.get(
        (GROUP, "logits")
    )

    assert output.shape == (
        N_AGENTS,
        N_ACTIONS,
    )


def test_model_allows_gradients():
    model = make_model()

    td = make_batch()

    output_td = model(td)

    output = output_td.get(
        (GROUP, "logits")
    )

    loss = output.sum()

    loss.backward()

    parameters_with_grad = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert len(parameters_with_grad) > 0


def test_shared_policy_has_one_encoder():
    model = make_model(
        share_params=True
    )

    assert len(model.encoders) == 1
    assert len(model.output_heads) == 1


def test_unshared_policy_has_encoder_per_agent():
    model = make_model(
        share_params=False
    )

    assert len(model.encoders) == N_AGENTS
    assert len(model.output_heads) == N_AGENTS


def test_model_rejects_centralised_usage():
    input_spec, output_spec = make_specs()

    try:
        CommPolicyModel(
            hidden_dim=HIDDEN_DIM,
            num_encoder_layers=2,
            activation_class_path="torch.nn.Tanh",
            comm_class_path=(
                "commstudy.communication.identity.IdentityComm"
            ),
            comm_kwargs={},
            comm_context_keys={},
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=GROUP,
            input_has_agent_dim=True,
            n_agents=N_AGENTS,
            centralised=True,
            share_params=True,
            device="cpu",
            action_spec=Composite(),
            model_index=0,
            is_critic=False,
        )

    except ValueError:
        return

    raise AssertionError(
        "CommPolicyModel should reject centralised=True"
    )


def test_model_rejects_critic_usage():
    input_spec, output_spec = make_specs()

    try:
        CommPolicyModel(
            hidden_dim=HIDDEN_DIM,
            num_encoder_layers=2,
            activation_class_path="torch.nn.Tanh",
            comm_class_path=(
                "commstudy.communication.identity.IdentityComm"
            ),
            comm_kwargs={},
            comm_context_keys={},
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=GROUP,
            input_has_agent_dim=True,
            n_agents=N_AGENTS,
            centralised=False,
            share_params=True,
            device="cpu",
            action_spec=Composite(),
            model_index=0,
            is_critic=True,
        )

    except ValueError:
        return

    raise AssertionError(
        "CommPolicyModel should not be usable as a critic"
    )