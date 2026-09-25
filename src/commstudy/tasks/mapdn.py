"""MAPDN task settings."""

from dataclasses import asdict, dataclass, field
from commstudy.utils.validation import dataclass_values, number


@dataclass
class TaskConfig:
    data_path: str = "data/case33_3min_final"
    manifest_path: str = "data/mapdn_case33_manifest.json"
    normalization_path: str | None = None
    max_steps: int = 239
    history: int = 1
    split: str = "train"
    evaluation_split: str = "validation"
    measurement_noise: bool = True
    reset_action: bool = False
    mode: str = "distributed"
    voltage_barrier_type: str = "l1"
    voltage_weight: float = 1.0
    q_weight: float | None = 0.1
    line_weight: float | None = None
    pv_scale: float = 1.0
    demand_scale: float = 1.0
    v_lower: float = 0.95
    v_upper: float = 1.05
    action_scale: float = 0.8
    action_bias: float = 0.0
    state_space: list[str] = field(
        default_factory=lambda: ["pv", "demand", "reactive", "vm_pu", "va_degree"]
    )
    solver_algorithm: str = "nr"
    solver_max_iteration: int = 20
    solver_tolerance_mva: float = 1e-8

    def __post_init__(self):
        dataclass_values(type(self), asdict(self), "task_config.params")
        for key in ("max_steps", "history", "solver_max_iteration"):
            number(getattr(self, key), f"task_config.params.{key}", minimum=1)
        for key in ("pv_scale", "demand_scale", "solver_tolerance_mva", "action_scale"):
            number(getattr(self, key), f"task_config.params.{key}", strict=True)
        if not self.data_path or not self.manifest_path:
            raise ValueError("MAPDN data_path and manifest_path must be explicit nonempty paths.")
        if self.split != "train":
            raise ValueError(
                "Managed MAPDN collection must use split='train'; offline evaluators "
                "select a held-out split explicitly after task validation."
            )
        if self.evaluation_split not in {"validation", "test"}:
            raise ValueError("MAPDN evaluation must use a held-out split.")
        if self.mode != "distributed":
            raise ValueError("Only one-inverter-per-agent distributed MAPDN is supported.")
        if self.voltage_barrier_type not in {"l1", "l2", "bowl", "courant_beltrami", "bump"}:
            raise ValueError("Unknown voltage barrier.")
        if not 0 < self.v_lower < self.v_upper:
            raise ValueError("Voltage limits must satisfy 0 < lower < upper.")
        if self.action_bias - self.action_scale < -1 or self.action_bias + self.action_scale > 1:
            raise ValueError("Reactive action fractions must remain within [-1,1].")
        for key in ("q_weight", "line_weight", "voltage_weight"):
            value = getattr(self, key)
            if value is not None:
                number(value, f"task_config.params.{key}")
        if (self.q_weight is None) == (self.line_weight is None):
            raise ValueError("Declare exactly one reactive-power or line-loss penalty.")
        allowed = {"pv", "demand", "reactive", "vm_pu", "va_degree"}
        if (
            not self.state_space
            or len(set(self.state_space)) != len(self.state_space)
            or (not set(self.state_space) <= allowed)
        ):
            raise ValueError("state_space must contain distinct supported local observations.")
        if self.solver_algorithm != "nr":
            raise ValueError("The initial MAPDN protocol supports Newton-Raphson only.")
