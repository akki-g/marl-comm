import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from commstudy.experiments import build_model_config
from commstudy.models import CommPolicyConfig
from commstudy.models.model import CommPolicyModel


N_AGENTS = 3
OBS_DIM = 8
HIDDEN_DIM = 16
OUT_DIM = 5
BATCH_SIZE = 2
GROUP = "agents"


def make_specs(*, invalid_mask=False):
    mask_shape = (N_AGENTS, N_AGENTS + 1) if invalid_mask else (N_AGENTS, N_AGENTS)
    input_spec = Composite(
        {
            GROUP: Composite(
                {
                    "observation": Unbounded(shape=(N_AGENTS, OBS_DIM)),
                    "comm_mask": Unbounded(shape=mask_shape, dtype=torch.bool),
                    "comm_sender_mask": Unbounded(
                        shape=(N_AGENTS,),
                        dtype=torch.bool,
                    ),
                    "agent_type": Unbounded(shape=(N_AGENTS,), dtype=torch.long),
                },
                shape=(N_AGENTS,),
            )
        }
    )
    output_spec = Composite(
        {
            GROUP: Composite(
                {"logits": Unbounded(shape=(N_AGENTS, OUT_DIM))},
                shape=(N_AGENTS,),
            )
        }
    )
    return input_spec, output_spec


def make_model(*, context_keys=None, invalid_mask=False):
    input_spec, output_spec = make_specs(invalid_mask=invalid_mask)
    return CommPolicyModel(
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=2,
        activation_class_path="torch.nn.Tanh",
        comm_class_path="commstudy.communication.identity.IdentityComm",
        comm_kwargs={},
        comm_context_keys=context_keys
        or {
            "mask": "comm_mask",
            "sender_mask": "comm_sender_mask",
            "class_id": "agent_type",
        },
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
        is_critic=False,
    )


def make_tensordict(mask):
    td = TensorDict({}, batch_size=[BATCH_SIZE])
    td.set((GROUP, "observation"), torch.randn(BATCH_SIZE, N_AGENTS, OBS_DIM))
    td.set((GROUP, "comm_mask"), mask)
    td.set(
        (GROUP, "comm_sender_mask"),
        torch.tensor([[True, False, True], [False, True, True]]),
    )
    td.set(
        (GROUP, "agent_type"),
        torch.tensor([[0, 1, 1], [0, 0, 1]]),
    )
    return td


def test_context_leaves_are_partitioned_from_local_encoder_features():
    model = make_model()

    assert model.encoder_in_keys == [(GROUP, "observation")]
    assert model.input_features == OBS_DIM
    assert model.encoders[0][0].in_features == OBS_DIM


def test_context_values_are_forwarded_with_expected_semantics():
    model = make_model()
    mask = torch.ones(BATCH_SIZE, N_AGENTS, N_AGENTS, dtype=torch.bool)
    td = make_tensordict(mask)

    context = model._build_comm_context(td)

    assert context.mask is td.get((GROUP, "comm_mask"))
    assert context.class_id is td.get((GROUP, "agent_type"))
    assert context.extras["sender_mask"] is td.get((GROUP, "comm_sender_mask"))


def test_identity_output_no_longer_leaks_context_mask_into_local_observation():
    torch.manual_seed(0)
    model = make_model()
    zeros = torch.zeros(BATCH_SIZE, N_AGENTS, N_AGENTS, dtype=torch.bool)
    ones = torch.ones(BATCH_SIZE, N_AGENTS, N_AGENTS, dtype=torch.bool)
    td_zeros = make_tensordict(zeros)
    td_ones = td_zeros.clone().set((GROUP, "comm_mask"), ones)

    output_zeros = model(td_zeros).get((GROUP, "logits"))
    output_ones = model(td_ones).get((GROUP, "logits"))

    assert torch.equal(output_zeros, output_ones)


def test_explicit_single_component_key_can_read_top_level_context():
    model = make_model(
        context_keys={
            "mask": ["top_mask"],
            "declared_mask": "comm_mask",
            "sender_mask": "comm_sender_mask",
            "class_id": "agent_type",
        }
    )
    td = make_tensordict(
        torch.zeros(BATCH_SIZE, N_AGENTS, N_AGENTS, dtype=torch.bool)
    )
    top_mask = torch.ones(BATCH_SIZE, N_AGENTS, N_AGENTS, dtype=torch.bool)
    td.set("top_mask", top_mask)

    assert model._build_comm_context(td).mask is top_mask


def test_missing_optional_context_key_is_safe():
    model = make_model(
        context_keys={
            "mask": "not_present",
            "declared_mask": "comm_mask",
            "sender_mask": "comm_sender_mask",
            "class_id": "agent_type",
        }
    )
    td = make_tensordict(
        torch.zeros(BATCH_SIZE, N_AGENTS, N_AGENTS, dtype=torch.bool)
    )

    context = model._build_comm_context(td)

    assert context.mask is None
    assert "sender_mask" in context.extras


def test_sampled_sender_mask_is_written_and_replayed_without_resampling():
    input_spec, output_spec = make_specs()
    model = CommPolicyModel(
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=2,
        activation_class_path="torch.nn.Tanh",
        comm_class_path="commstudy.communication.attention.AttentionComm",
        comm_kwargs={
            "message_dim": 8,
            "key_dim": 8,
            "num_heads": 2,
            "channel": {"type": "dropout", "p": 0.5, "mode": "always"},
        },
        comm_context_keys={
            "mask": "comm_mask",
            "sender_mask": "comm_sender_mask",
            "class_id": "agent_type",
        },
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
        is_critic=False,
    )
    observation = torch.randn(BATCH_SIZE, N_AGENTS, OBS_DIM)

    torch.manual_seed(11)
    rollout_td = TensorDict(
        {(GROUP, "observation"): observation.clone()},
        batch_size=[BATCH_SIZE],
    )
    rollout_output = model(rollout_td)
    realized = rollout_output.get((GROUP, "comm_sender_mask")).clone()
    rollout_logits = rollout_output.get((GROUP, "logits")).clone()

    assert realized.shape == (BATCH_SIZE, N_AGENTS)
    assert realized.dtype is torch.bool
    assert not realized.requires_grad
    generated_marker = rollout_output.get(
        (GROUP, "comm_sender_mask_generated")
    ).clone()
    assert generated_marker.all()

    # A different RNG state would draw a different mask if the model ignored
    # the replay leaf. The explicit realization must instead be authoritative.
    torch.manual_seed(999)
    replay_td = TensorDict(
        {
            (GROUP, "observation"): observation.clone(),
            (GROUP, "comm_sender_mask"): realized.clone(),
            (GROUP, "comm_sender_mask_generated"): generated_marker,
            # Stored PPO/QMIX training data contains the action whose
            # probability/value is being recomputed.
            (GROUP, "action"): torch.zeros(BATCH_SIZE, N_AGENTS, 1),
        },
        batch_size=[BATCH_SIZE],
    )
    replay_output = model(replay_td)

    assert torch.equal(
        replay_output.get((GROUP, "comm_sender_mask")),
        realized,
    )
    assert torch.equal(replay_output.get((GROUP, "logits")), rollout_logits)


def test_generated_sender_mask_is_resampled_for_a_fresh_action_step():
    input_spec, output_spec = make_specs()
    model = CommPolicyModel(
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=2,
        activation_class_path="torch.nn.Tanh",
        comm_class_path="commstudy.communication.broadcast.BroadcastComm",
        comm_kwargs={
            "message_dim": 8,
            "channel": {"type": "dropout", "p": 0.5, "mode": "always"},
        },
        comm_context_keys={
            "mask": "comm_mask",
            "sender_mask": "comm_sender_mask",
            "class_id": "agent_type",
        },
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
        is_critic=False,
    )
    observation = torch.randn(BATCH_SIZE, N_AGENTS, OBS_DIM)
    stale = torch.zeros(BATCH_SIZE, N_AGENTS, dtype=torch.bool)
    fresh_td = TensorDict(
        {
            (GROUP, "observation"): observation,
            (GROUP, "comm_sender_mask"): stale,
            (GROUP, "comm_sender_mask_generated"): torch.ones_like(stale),
        },
        batch_size=[BATCH_SIZE],
    )

    torch.manual_seed(23)
    expected = torch.rand(BATCH_SIZE, N_AGENTS) >= 0.5
    torch.manual_seed(23)
    model(fresh_td)

    assert torch.equal(
        fresh_td.get((GROUP, "comm_sender_mask")),
        expected,
    )


def test_invalid_declared_mask_shape_is_rejected():
    with pytest.raises(ValueError, match="Communication mask"):
        make_model(invalid_mask=True)


def test_duplicate_context_key_mappings_are_rejected():
    with pytest.raises(ValueError, match="unique key"):
        make_model(
            context_keys={
                "mask": "comm_mask",
                "duplicate": "comm_mask",
            }
        )


@pytest.mark.parametrize(
    ("config_name", "class_suffix"),
    [("comm_broadcast", "BroadcastComm"), ("comm_gated", "GatedComm")],
)
def test_new_model_configs_build(config_root, config_name, class_suffix):
    document = OmegaConf.to_container(
        OmegaConf.load(config_root / "models" / f"{config_name}.yaml"),
        resolve=True,
    )

    config = build_model_config(document)

    assert isinstance(config, CommPolicyConfig)
    assert config.comm_class_path.endswith(class_suffix)
    assert config.comm_kwargs["message_dim"] == 32
    assert config.comm_context_keys["mask"] == "comm_mask"
    if config_name == "comm_gated":
        assert config.comm_kwargs["sender_selection"] == "learned"
        assert config.comm_kwargs["sender_selection_seed"] == 0
