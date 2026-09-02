from __future__ import annotations

import torch
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model

from commstudy.communication.base import CommContext, CommModule
from commstudy.utils.imports import import_from_path


class CommPolicyModel(Model):
    """
    BenchMARL decentralized policy model with a communication injection point
    
    Architecture:
        local obs
            -> shared encoder
            -> communication module
            -> output head
    
    BenchMARL decides what the output means

    MAPPO: action logits
    QMIX: per-agent action Q-values
    """

    def __init__(
        self,
        hidden_dim: int,
        num_encoder_layers: int,
        activation_class_path: str,
        comm_class_path: str,
        comm_kwargs: dict | None = None,
        comm_context_keys: dict | None = None,
        use_role_embedding: bool = False,
        num_roles: int = 2,
        **kwargs,
    ):
        agent_group = kwargs.get("agent_group")
        if not isinstance(agent_group, str):
            raise TypeError("CommPolicyModel requires a string agent_group.")

        self.comm_context_keys = {
            str(context_name): self._normalize_context_key(
                agent_group,
                td_key,
            )
            for context_name, td_key in dict(comm_context_keys or {}).items()
        }
        if len(set(self.comm_context_keys.values())) != len(self.comm_context_keys):
            raise ValueError("Each communication context field must map to a unique key.")

        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.encoder_in_keys = self._get_encoder_in_keys()

        self.input_features = sum(
            self.input_spec[key].shape[-1]
            for key in self.encoder_in_keys
        )

        self.output_features = self.output_leaf_spec.shape[-1]

        activation_class = import_from_path(
            activation_class_path
        )

        if (
            not isinstance(activation_class, type)
            or not issubclass(activation_class, nn.Module)
        ):
            raise TypeError(
                f"{activation_class_path} must resolve to an nn.Module class"
            )
        
        self.encoders = self._build_encoders(
            input_dim=self.input_features,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            activation_class = activation_class,
        )

        self.output_heads = self._build_output_heads(
            hidden_dim=hidden_dim,
            output_dim=self.output_features,
        )

        self.num_roles = int(num_roles)
        if use_role_embedding: 
            if "class_id" not in self.comm_context_keys:
                raise ValueError(
                    "use_role_embedding requires a 'class_id' entry in "
                    "comm_context_keys."
                )
            if self.num_roles < 1:
                raise ValueError("num_roles must be <= 1")
        self.role_emb = (
            nn.Embedding(self.num_roles, hidden_dim) if use_role_embedding else None
        )
        if self.role_emb is not None:
            nn.init.zeros_(self.role_emb.weight)
        self._role_range_checked = False

        comm_class = import_from_path(comm_class_path)

        if (
            not isinstance(comm_class, type)
            or not issubclass(comm_class, CommModule)
        ):
            raise TypeError(
                f"{comm_class_path} must resolve to a CommModule subclass"
            )

        self.comm = comm_class(
            hidden_dim=hidden_dim,
            **(comm_kwargs or {}),
        )
        channel = getattr(self.comm, "channel", None)
        validate_replay_safety = getattr(
            channel,
            "validate_policy_replay_safety",
            None,
        )
        if callable(validate_replay_safety):
            validate_replay_safety()
            
        self.to(self.device)

    @staticmethod
    def _normalize_context_key(
        agent_group: str,
        key: str | list[str] | tuple[str, ...],
    ) -> str | tuple[str, ...]:
        """Resolve relative strings and preserve explicit nested TensorDict keys."""

        if isinstance(key, str):
            return (agent_group, key)
        if not isinstance(key, (list, tuple)) or not key:
            raise TypeError(
                "Communication context keys must be relative strings or non-empty "
                "string sequences."
            )
        if not all(isinstance(part, str) for part in key):
            raise TypeError("Every component of a communication context key must be a string.")

        normalized = tuple(key)
        return normalized[0] if len(normalized) == 1 else normalized

    def _get_encoder_in_keys(self) -> list[str | tuple[str, ...]]:
        context_keys = set(self.comm_context_keys.values())
        return [key for key in self.in_keys if key not in context_keys]

    def _perform_checks(self) -> None:
        """
        Called by ``benchmarl.models.common.Model.__init__``.

        Runs BenchMARL's own spec checks and then the usage
        restrictions specific to this model.
        """
        super()._perform_checks()

        self._validate_usage()
        self._validate_specs()

    def _validate_usage(self) -> None:
        if not self.input_has_agent_dim:
            raise ValueError(
                "CommPolicyModel requires per-agent input."
            )

        if self.centralised:
            raise ValueError(
                "CommPolicyModel is only for decentralized models."
            )

        if self.is_critic:
            raise ValueError(
                "CommPolicyModel cannot be used as a critic."
            )

    def _validate_specs(self) -> None:
        """
        Encoder inputs must be [..., n_agents, features] and the output
        must be [..., n_agents, output_features]. Context leaves are
        validated according to their distinct semantic shapes.

        Without this the agent dimension could be silently folded into
        the feature dimension, producing a model that trains on
        misaligned data instead of failing.
        """
        encoder_in_keys = self._get_encoder_in_keys()
        if not encoder_in_keys:
            raise ValueError("CommPolicyModel requires at least one local encoder input.")

        for key in encoder_in_keys:
            spec = self.input_spec[key]
            if len(spec.shape) < 2 or spec.shape[-2] != self.n_agents:
                raise ValueError(
                    f"CommPolicyModel input '{key}' should have shape "
                    f"[..., n_agents, features] with n_agents="
                    f"{self.n_agents}, got {tuple(spec.shape)}."
                )

        input_spec_keys = set(self.in_keys)
        for context_name, key in self.comm_context_keys.items():
            if key not in input_spec_keys:
                # Context is optional and may be injected as an extra TensorDict
                # leaf by a task transform rather than declared as observation.
                continue

            shape = self.input_spec[key].shape
            if context_name == "mask" and tuple(shape[-2:]) != (
                self.n_agents,
                self.n_agents,
            ):
                raise ValueError(
                    f"Communication mask '{key}' must end in "
                    f"({self.n_agents}, {self.n_agents}), got {tuple(shape)}."
                )
            if context_name in {"class_id", "sender_mask"} and (
                len(shape) < 1 or shape[-1] != self.n_agents
            ):
                raise ValueError(
                    f"Communication context '{context_name}' at '{key}' must end "
                    f"in ({self.n_agents},), got {tuple(shape)}."
                )

        output_shape = self.output_leaf_spec.shape

        if len(output_shape) < 2 or output_shape[-2] != self.n_agents:
            raise ValueError(
                f"CommPolicyModel output '{self.out_key}' should have "
                f"shape [..., n_agents, features] with n_agents="
                f"{self.n_agents}, got {tuple(output_shape)}."
            )

    def _build_encoders(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_class: type[nn.Module],
    ) -> nn.ModuleList:

        if num_layers < 1:
            raise ValueError(
                "num_encoder_layers must be >= 1"
            )

        count = 1 if self.share_params else self.n_agents

        return nn.ModuleList(
            [
                self._build_mlp(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    activation_class=activation_class,
                )
                for _ in range(count)
            ]
        )

    def _build_output_heads(
        self,
        hidden_dim: int,
        output_dim: int,
    ) -> nn.ModuleList:

        count = 1 if self.share_params else self.n_agents

        return nn.ModuleList(
            [
                nn.Linear(
                    hidden_dim,
                    output_dim,
                )
                for _ in range(count)
            ]
        )

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_class: type[nn.Module],
    ) -> nn.Sequential:

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(
                nn.Linear(
                    current_dim,
                    hidden_dim,
                )
            )
            layers.append(
                activation_class()
            )

            current_dim = hidden_dim

        return nn.Sequential(*layers)

    def _apply_agent_modules(
        self,
        x: torch.Tensor,
        modules: nn.ModuleList,
    ) -> torch.Tensor:
        """
        Apply either one shared module or one module per agent.
        """

        if self.share_params:
            return modules[0](x)

        return torch.stack(
            [
                modules[i](x[..., i, :])
                for i in range(self.n_agents)
            ],
            dim=-2,
        )

    def _build_comm_context(
        self,
        tensordict: TensorDictBase,
    ) -> CommContext:

        if not self.comm_context_keys:
            # IdentityComm and any other context-free module take this
            # path, so the key scan stays out of the hot forward pass.
            return CommContext()

        available_keys = set(
            tensordict.keys(
                include_nested=True,
                leaves_only=True,
            )
        )

        values = {}
        generated_sender_key = self._generated_sender_mask_key()
        generated_sender_mask = (
            generated_sender_key is not None
            and generated_sender_key in available_keys
        )
        # BenchMARL's collectors and evaluator invoke the actor under
        # ``torch.no_grad()``, whereas PPO's policy-loss recomputation is
        # gradient enabled.  The collector retains the previous action and
        # arbitrary policy-added leaves, so key presence alone cannot
        # distinguish a fresh environment step from replay.  Requiring both
        # the stored action and a gradient-enabled forward gives the generated
        # mask the intended lifecycle: resample for behavior/evaluation,
        # replay exactly for the differentiable policy update.
        replaying_policy_data = torch.is_grad_enabled() and (
            self.agent_group,
            "action",
        ) in available_keys

        for context_name, key in self.comm_context_keys.items():
            if key in available_keys:
                if (
                    context_name == "sender_mask"
                    and generated_sender_mask
                    and not replaying_policy_data
                ):
                    # TorchRL carries policy-added leaves into the next
                    # collector input. An internally generated mask is only
                    # authoritative when recomputing a stored action during a
                    # loss forward; a fresh behavior/evaluation action needs a
                    # fresh per-step realization. Environment-provided masks
                    # have no generated marker and are always authoritative.
                    continue
                values[context_name] = tensordict.get(key)

        mask = values.pop("mask", None)
        class_id = values.pop("class_id", None)

        return CommContext(
            mask=mask,
            class_id=class_id,
            extras=values,
        )

    def _role_features(self, context: CommContext) -> torch.Tensor:
        class_id = context.class_id
        if class_id is None:
            raise ValueError(
                "use_role_embedding is enabled but no class_id leaf was found "
                f"at {self.comm_context_keys['class_id']}."
            )
        class_id = class_id.long()

        if not self._role_range_checked:
            # One-off. A per-step min/max on CUDA forces a device sync.
            lo, hi = int(class_id.min()), int(class_id.max())
            if lo < 0 or hi >= self.num_roles:
                raise ValueError(
                    f"class_id values must lie in [0, {self.num_roles}); "
                    f"observed [{lo}, {hi}]."
                )
            self._role_range_checked = True

        return self.role_emb(class_id)

    def _generated_sender_mask_key(self) -> str | tuple[str, ...] | None:
        key = self.comm_context_keys.get("sender_mask")
        if key is None:
            return None
        if isinstance(key, tuple):
            return (*key[:-1], f"{key[-1]}_generated")
        return f"{key}_generated"

    def _write_realized_sender_mask(
        self,
        tensordict: TensorDictBase,
        context: CommContext,
    ) -> None:
        """Persist a newly sampled channel realization for policy replay.

        PPO recomputes action log-probabilities from rollout TensorDicts. If a
        stochastic communication failure were sampled again during that
        recomputation, the probability ratio would compare policies
        conditioned on different channel states. Learned modules therefore
        expose their channel's last detached sender mask, which is written as a
        policy output when no authoritative mask was already supplied.

        The extra key is deliberately not added to ``out_keys``: BenchMARL's
        action/distribution assembly remains unchanged, while TensorDict and
        collectors still preserve the leaf alongside the action data.
        """

        key = self.comm_context_keys.get("sender_mask")
        if key is None or "sender_mask" in context.extras:
            return

        channel = getattr(self.comm, "channel", None)
        realized = getattr(channel, "last_sender_mask", None)
        if realized is None:
            return

        realized = realized.detach().clone()
        tensordict.set(key, realized)
        marker_key = self._generated_sender_mask_key()
        if marker_key is not None:
            tensordict.set(
                marker_key,
                torch.ones_like(realized, dtype=torch.bool),
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = torch.cat(
            [
                tensordict.get(in_key)
                for in_key in self.encoder_in_keys
            ],
            dim=-1
        )

        # local representation for every agent 
        h = self._apply_agent_modules(
            obs, self.encoders,
        )

        context = self._build_comm_context(tensordict)

        #generic comm injection point
        h = self.comm(
            h,
            context=context
        )

        self._write_realized_sender_mask(
            tensordict,
            context,
        )

        if self.role_emb is not None:
            h = h + self._role_features(context)

        output = self._apply_agent_modules(
            h, 
            self.output_heads,
        )

        tensordict.set(
            self.out_key,
            output,
        )

        return tensordict
