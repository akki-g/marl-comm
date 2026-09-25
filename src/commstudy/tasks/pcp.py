"""Typed PCP task document, including the inherited VMAS Simple Tag options.

BenchMARL discovers stock TaskConfig classes under its installed namespace.
The custom Task enum invokes this project-owned equivalent directly: no patch
to the installed package or suppression of a missing-schema warning is needed.
"""

from dataclasses import asdict, dataclass

from commstudy.utils.validation import dataclass_values, number


@dataclass
class TaskConfig:
    max_steps: int = 100
    num_good_agents: int = 1
    num_adversaries: int = 3
    num_landmarks: int = 2
    shape_agent_rew: bool = False
    shape_adversary_rew: bool = False
    agents_share_rew: bool = False
    adversaries_share_rew: bool = True
    observe_same_team: bool = True
    observe_pos: bool = True
    observe_vel: bool = True
    bound: float = 1.0
    respawn_at_catch: bool = False
    prey_detection_radius: float = 0.8
    prey_obstacle_margin: float = 0.4
    prey_boundary_margin: float = 0.3
    prey_wander_force: float = 0.15
    prey_noise_std: float = 0.05
    prey_min_flee_frac: float = 0.3
    predator_sensing_radius: float | None = 1.0

    def __post_init__(self):
        values = asdict(self)
        dataclass_values(type(self), values, "task_config.params")
        for key in ("max_steps", "num_good_agents", "num_adversaries"):
            number(values[key], f"task_config.params.{key}", minimum=1)
        number(self.num_landmarks, "task_config.params.num_landmarks")
        for key in ("bound", "prey_detection_radius", "prey_boundary_margin"):
            number(values[key], f"task_config.params.{key}", strict=True)
        for key in ("prey_obstacle_margin", "prey_wander_force", "prey_noise_std"):
            number(values[key], f"task_config.params.{key}")
        number(self.prey_min_flee_frac, "task_config.params.prey_min_flee_frac", maximum=1)
        if self.predator_sensing_radius is not None:
            number(self.predator_sensing_radius, "task_config.params.predator_sensing_radius")
        if self.prey_boundary_margin >= self.bound:
            raise ValueError("prey_boundary_margin must be smaller than bound.")
        if not self.observe_pos or not self.observe_vel:
            raise ValueError(
                "PCP's visibility layout requires observe_pos=true and observe_vel=true."
            )
