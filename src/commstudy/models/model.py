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
        **kwargs,
    ): 
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.comm_context_keys = dict(comm_context_keys or {})

        self.input_features = sum(
            spec.shape[-1]
            for spec in self.input_spec.values(True,True)
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
            
        self.to(self.device)

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
        Every input leaf must be [..., n_agents, features] and the
        output leaf must be [..., n_agents, output_features].

        Without this the agent dimension could be silently folded into
        the feature dimension, producing a model that trains on
        misaligned data instead of failing.
        """
        for key, spec in self.input_spec.items(True, True):
            if len(spec.shape) < 2 or spec.shape[-2] != self.n_agents:
                raise ValueError(
                    f"CommPolicyModel input '{key}' should have shape "
                    f"[..., n_agents, features] with n_agents="
                    f"{self.n_agents}, got {tuple(spec.shape)}."
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

        for context_name, td_key in self.comm_context_keys.items():
            key = (
                self.agent_group,
                td_key,
            )

            if key in available_keys:
                values[context_name] = tensordict.get(key)

        mask = values.pop("mask", None)
        class_id = values.pop("class_id", None)

        return CommContext(
            mask=mask,
            class_id=class_id,
            extras=values,
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = torch.cat(
            [
                tensordict.get(in_key)
                for in_key in self.in_keys
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

        output = self._apply_agent_modules(
            h, 
            self.output_heads,
        )

        tensordict.set(
            self.out_key,
            output,
        )

        return tensordict
