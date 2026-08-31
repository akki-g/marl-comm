"""Generic differentiable transformations applied to sender messages.

An explicit ``sender_mask`` is an authoritative realization supplied by the
rollout/replay path. This matters for PPO: randomly resampling failures during
the optimizer forward would make stored rollout log-probabilities inconsistent
with the communication realization used to update the policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
import math

import torch
from torch import nn
from tensordict.nn.probabilistic import InteractionType, interaction_type


ChannelMode = Literal["always", "training", "evaluation", "disabled"]
_CHANNEL_MODES = {"always", "training", "evaluation", "disabled"}


@dataclass(frozen=True)
class ChannelOutput:
    """Transformed sender payloads and their realized availability."""

    messages: torch.Tensor
    sender_mask: torch.Tensor


def _validate_messages(messages: torch.Tensor) -> None:
    if not isinstance(messages, torch.Tensor):
        raise TypeError(f"Channel messages must be a Tensor, got {type(messages).__name__}.")
    if messages.dim() < 2:
        raise ValueError(
            f"Channel messages must be shaped [..., N, M], got {tuple(messages.shape)}."
        )
    if messages.shape[-2] < 1 or messages.shape[-1] < 1:
        raise ValueError("Channel messages require at least one sender and one scalar.")
    if not messages.is_floating_point():
        raise TypeError(
            f"Channel messages must use a real floating dtype, got {messages.dtype}."
        )


def _resolve_sender_mask(
    messages: torch.Tensor,
    sender_mask: torch.Tensor | None,
) -> torch.Tensor:
    _validate_messages(messages)
    target_shape = messages.shape[:-1]
    if sender_mask is None:
        return torch.ones(target_shape, dtype=torch.bool, device=messages.device)
    if not isinstance(sender_mask, torch.Tensor):
        raise TypeError(f"sender_mask must be a Tensor, got {type(sender_mask).__name__}.")
    if sender_mask.dtype is not torch.bool:
        raise TypeError(f"sender_mask must be boolean, got {sender_mask.dtype}.")
    if sender_mask.dim() < 1 or sender_mask.shape[-1] != messages.shape[-2]:
        raise ValueError(
            f"sender_mask must end in ({messages.shape[-2]},), "
            f"got {tuple(sender_mask.shape)}."
        )
    try:
        return torch.broadcast_to(
            sender_mask.to(device=messages.device),
            target_shape,
        ).clone()
    except RuntimeError as exc:
        raise ValueError(
            f"sender_mask shape {tuple(sender_mask.shape)} is not broadcastable "
            f"to {target_shape}."
        ) from exc


class CommChannel(nn.Module, ABC):
    """Base class for sender-level channel transformations."""

    requested_dropout_rate: float = 0.0

    def __init__(self, mode: ChannelMode = "always") -> None:
        super().__init__()
        if mode not in _CHANNEL_MODES:
            expected = ", ".join(sorted(_CHANNEL_MODES))
            raise ValueError(f"Unknown channel mode '{mode}'. Expected one of: {expected}.")
        self.mode: ChannelMode = mode
        self._last_sender_mask: torch.Tensor | None = None

    @property
    def last_sender_mask(self) -> torch.Tensor | None:
        """Most recent detached sender realization, if the channel has run."""

        return self._last_sender_mask

    def _should_apply(self) -> bool:
        # BenchMARL 1.5 performs deterministic evaluation under a no-gradient
        # interaction context but does not call ``policy.eval()``. Recognize
        # that lifecycle explicitly while retaining ordinary ``train()/eval()``
        # semantics for standalone use. PPO's differentiable policy replay is
        # also deterministic, hence the no-gradient condition is essential.
        deterministic_interactions = {
            InteractionType.DETERMINISTIC,
            InteractionType.MODE,
            InteractionType.MEAN,
            InteractionType.MEDIAN,
        }
        benchmarl_evaluation = (
            not torch.is_grad_enabled()
            and interaction_type() in deterministic_interactions
        )
        evaluation = not self.training or benchmarl_evaluation
        return (
            self.mode == "always"
            or (self.mode == "training" and not evaluation)
            or (self.mode == "evaluation" and evaluation)
        )

    def validate_policy_replay_safety(self) -> None:
        """Raise when a configured stochastic payload cannot be replayed.

        Sender availability is persisted by ``CommPolicyModel``. Channels
        whose randomness changes the payload itself need a stronger replay
        representation and may override this check.
        """

    def _finish(
        self,
        messages: torch.Tensor,
        sender_mask: torch.Tensor,
    ) -> ChannelOutput:
        _validate_messages(messages)
        masked_messages = messages * sender_mask.unsqueeze(-1).to(messages.dtype)
        self._last_sender_mask = sender_mask.detach().clone()
        return ChannelOutput(messages=masked_messages, sender_mask=sender_mask)

    @abstractmethod
    def forward(
        self,
        messages: torch.Tensor,
        sender_mask: torch.Tensor | None = None,
    ) -> ChannelOutput:
        raise NotImplementedError


class IdentityChannel(CommChannel):
    """No channel distortion; an explicit sender mask is still enforced."""

    def forward(
        self,
        messages: torch.Tensor,
        sender_mask: torch.Tensor | None = None,
    ) -> ChannelOutput:
        available = _resolve_sender_mask(messages, sender_mask)
        return self._finish(messages, available)


class DropoutChannel(CommChannel):
    """Drop complete sender payloads rather than individual scalar elements.

    The default mode is evaluation-only. Training-time stochastic use requires
    the realized mask to be carried through rollout/replay and supplied as the
    explicit ``sender_mask`` on every corresponding optimizer forward.
    """

    def __init__(
        self,
        p: float,
        mode: ChannelMode = "evaluation",
    ) -> None:
        super().__init__(mode=mode)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}.")
        self.p = float(p)
        self.requested_dropout_rate = self.p

    def forward(
        self,
        messages: torch.Tensor,
        sender_mask: torch.Tensor | None = None,
    ) -> ChannelOutput:
        if sender_mask is not None:
            # An explicit replayed realization is authoritative: do not draw a
            # second stochastic failure mask.
            available = _resolve_sender_mask(messages, sender_mask)
        elif self._should_apply() and self.p > 0.0:
            available = torch.rand(
                messages.shape[:-1],
                device=messages.device,
            ) >= self.p
        else:
            available = _resolve_sender_mask(messages, None)

        return self._finish(messages, available)


class GaussianNoiseChannel(CommChannel):
    """Add zero-mean Gaussian noise to available sender payloads."""

    def __init__(
        self,
        std: float,
        mode: ChannelMode = "evaluation",
    ) -> None:
        super().__init__(mode=mode)
        if not math.isfinite(std) or std < 0.0:
            raise ValueError(f"Gaussian std must be finite and >= 0, got {std}.")
        self.std = float(std)

    def forward(
        self,
        messages: torch.Tensor,
        sender_mask: torch.Tensor | None = None,
    ) -> ChannelOutput:
        available = _resolve_sender_mask(messages, sender_mask)
        if self._should_apply() and self.std > 0.0:
            messages = messages + torch.randn_like(messages) * self.std
        return self._finish(messages, available)

    def validate_policy_replay_safety(self) -> None:
        if self.std > 0.0 and self.mode in {"always", "training"}:
            raise ValueError(
                "Train-time Gaussian communication noise is not replay-safe: "
                "the sender mask does not reproduce sampled payload noise. "
                "Use mode='evaluation' or mode='disabled'."
            )


class QuantizedChannel(CommChannel):
    """Uniformly quantize a bounded payload, optionally with STE gradients."""

    def __init__(
        self,
        levels: int,
        clip_value: float = 1.0,
        straight_through: bool = True,
        mode: ChannelMode = "evaluation",
    ) -> None:
        super().__init__(mode=mode)
        if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
            raise ValueError("Quantization levels must be an integer >= 2.")
        if not math.isfinite(clip_value) or clip_value <= 0.0:
            raise ValueError("clip_value must be finite and > 0.")
        self.levels = int(levels)
        self.clip_value = float(clip_value)
        self.straight_through = bool(straight_through)

    def forward(
        self,
        messages: torch.Tensor,
        sender_mask: torch.Tensor | None = None,
    ) -> ChannelOutput:
        available = _resolve_sender_mask(messages, sender_mask)
        if self._should_apply():
            clipped = messages.clamp(-self.clip_value, self.clip_value)
            step = 2.0 * self.clip_value / (self.levels - 1)
            quantized = (
                torch.round((clipped + self.clip_value) / step) * step
                - self.clip_value
            )
            if self.straight_through:
                messages = messages + (quantized - messages).detach()
            else:
                messages = quantized
        return self._finish(messages, available)


class SequentialChannel(CommChannel):
    """Compose channel transformations while preserving sender availability."""

    def __init__(self, channels: Sequence[CommChannel]) -> None:
        super().__init__(mode="always")
        if not channels:
            raise ValueError("SequentialChannel requires at least one channel.")
        self.channels = nn.ModuleList(channels)
        keep_probability = 1.0
        for channel in channels:
            keep_probability *= 1.0 - channel.requested_dropout_rate
        self.requested_dropout_rate = 1.0 - keep_probability

    def forward(
        self,
        messages: torch.Tensor,
        sender_mask: torch.Tensor | None = None,
    ) -> ChannelOutput:
        combined = _resolve_sender_mask(messages, sender_mask)
        transformed = messages

        for channel in self.channels:
            # Pass the original explicit realization to every component. With
            # no explicit mask, independent stochastic components may each draw
            # their own realization and are combined below.
            result = channel(transformed, sender_mask=sender_mask)
            transformed = result.messages
            combined = combined & result.sender_mask

        return self._finish(transformed, combined)

    def validate_policy_replay_safety(self) -> None:
        for channel in self.channels:
            channel.validate_policy_replay_safety()


def build_channel(
    config: CommChannel | Mapping[str, Any] | Sequence[Any] | None,
) -> CommChannel:
    """Construct a channel from a module or serializable configuration."""

    if config is None:
        return IdentityChannel()
    if isinstance(config, CommChannel):
        return config
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        return SequentialChannel([build_channel(item) for item in config])
    if not isinstance(config, Mapping):
        raise TypeError(
            "Channel configuration must be a CommChannel, mapping, sequence, or None."
        )

    values = dict(config)
    channel_type = str(values.pop("type", "identity")).lower()

    if channel_type == "identity":
        return IdentityChannel(**values)
    if channel_type == "dropout":
        return DropoutChannel(**values)
    if channel_type in {"gaussian", "gaussian_noise"}:
        return GaussianNoiseChannel(**values)
    if channel_type in {"quantized", "quantization"}:
        return QuantizedChannel(**values)
    if channel_type in {"sequential", "sequence"}:
        channels = values.pop("channels", None)
        if values:
            unknown = ", ".join(sorted(values))
            raise TypeError(f"Unknown SequentialChannel options: {unknown}.")
        if channels is None:
            raise ValueError("Sequential channel config requires 'channels'.")
        return build_channel(channels)

    raise ValueError(f"Unknown communication channel type '{channel_type}'.")
