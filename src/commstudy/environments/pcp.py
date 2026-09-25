"""VMAS Simple Tag with scripted prey and episode-indexed exogenous noise.

``pcp_rng_v2`` resets wander state and isolates noise from policy/channel draws.
It is a new scientific protocol: old PCP calibration trajectories must retain
their original source/version and must not be labelled corrected-protocol runs.
"""

from vmas.scenarios.mpe.simple_tag import Scenario as SimpleTagScenario
import torch
from commstudy.tasks.pcp import TaskConfig
from commstudy.utils.validation import dataclass_values
from commstudy.tasks.pcp_metrics import PCP_DOMAIN_PROTOCOL, compute_pcp_domain_metrics


class PredatorCapturePreyScenario(SimpleTagScenario):
    randomness_protocol = "pcp_rng_v2"
    observation_protocol = "pcp_visibility_v2"
    domain_metrics_protocol = PCP_DOMAIN_PROTOCOL
    _noise_window = 256

    def make_world(self, batch_dim, device, **kwargs):
        # Direct VMAS users receive the same validation as the managed loader.
        # Do not replace kwargs with defaults: preserve this factory's existing
        # visibility default and the simulator's consumed-argument semantics.
        dataclass_values(TaskConfig, kwargs, "task_config.params")
        TaskConfig(**kwargs)
        # Tunable knobs for the scripted prey. Pop them before calling super()
        # so they don't get passed through to VMAS's own kwarg parsing.
        self.prey_detection_radius = kwargs.pop("prey_detection_radius", 0.8)
        self.prey_obstacle_margin = kwargs.pop("prey_obstacle_margin", 0.4)
        self.prey_boundary_margin = kwargs.pop("prey_boundary_margin", 0.3)
        self.prey_wander_force = kwargs.pop("prey_wander_force", 0.15)
        self.prey_noise_std = kwargs.pop("prey_noise_std", 0.05)
        self.prey_min_flee_frac = kwargs.pop("prey_min_flee_frac", 0.3)

        # None reveals every prey and appends all-true visibility flags, keeping
        # actor width fixed across information conditions in protocol v2.
        self.predator_sensing_radius = kwargs.pop("predator_sensing_radius", None)

        world = super().make_world(batch_dim, device, **kwargs)

        # Cache a persistent per-agent wander direction so idle motion looks
        # like gentle drifting instead of jittering randomly every step.
        for agent in world.agents:
            if not agent.adversary:
                agent.wander_dir = torch.zeros(batch_dim, world.dim_p, device=device)
        prey_count = len([agent for agent in world.agents if not agent.adversary])
        # CPU generators generate a fixed-size block independent of policy,
        # threats, collisions, and the reset history of other vectorized worlds.
        # Blocks are extended on demand, so max_steps is not a hidden bound.
        self.exogenous_episode_ids = torch.zeros(batch_dim, dtype=torch.long)
        self.exogenous_steps = torch.zeros(batch_dim, prey_count, dtype=torch.long)
        self._noise_generators = [
            [torch.Generator() for _ in range(prey_count)] for _ in range(batch_dim)
        ]
        shape = (batch_dim, prey_count, self._noise_window)
        self._wander_resample = torch.empty(*shape, 1, dtype=torch.bool, device=device)
        self._wander_headings = torch.empty(*shape, world.dim_p, device=device)
        self._force_noise = torch.empty_like(self._wander_headings)
        self._transition_domain_metrics = None
        return world

    def reset_world_at(self, env_index=None):
        """Reset physical and scripted state only for the selected worlds."""
        super().reset_world_at(env_index)
        indices = range(self.world.batch_dim) if env_index is None else [env_index]
        seeds = torch.randint(0, 2**63 - 1, (len(indices),), device="cpu")
        for row, episode_seed in zip(indices, seeds.tolist(), strict=True):
            self.exogenous_episode_ids[row] = episode_seed
            self.exogenous_steps[row].zero_()
            for prey_index, prey in enumerate(self.good_agents()):
                prey.wander_dir[row].zero_()
                self._noise_generators[row][prey_index].manual_seed(
                    (episode_seed + 104729 * prey_index) % (2**63 - 1)
                )
                self._fill_noise_window(row, prey_index)
        # A full recomputation also preserves unaffected worlds on partial
        # reset. There is no episode accumulator or RNG state in these metrics.
        self._transition_domain_metrics = None

    def post_step(self):
        super().post_step()
        # Capture before inherited reward() can respawn a caught prey. This is
        # the same post-physics state that supplies this transition's reward.
        self._transition_domain_metrics = compute_pcp_domain_metrics(self)

    def info(self, agent):
        """Team transition metrics repeated per predator, outside policy inputs."""
        if not agent.adversary:
            return {}
        values = self._transition_domain_metrics
        if values is None:
            values = compute_pcp_domain_metrics(self)
        return {key: value.clone() for key, value in values.items()}

    def _fill_noise_window(self, row, prey_index):
        generator = self._noise_generators[row][prey_index]
        steps, dim = self._noise_window, self.world.dim_p
        self._wander_resample[row, prey_index] = (
            torch.rand(steps, 1, generator=generator) < 0.02
        ).to(self.world.device)
        self._wander_headings[row, prey_index] = torch.randn(steps, dim, generator=generator).to(
            self.world.device
        )
        self._force_noise[row, prey_index] = torch.randn(steps, dim, generator=generator).to(
            self.world.device
        )

    def _prey_disturbances(self, prey):
        """Return the noise at each world's (episode ID, prey ID, timestep)."""
        prey_index = self.good_agents().index(prey)
        steps = self.exogenous_steps[:, prey_index]
        rollover = (steps > 0) & (steps % self._noise_window == 0)
        for row in rollover.nonzero(as_tuple=False).flatten().tolist():
            self._fill_noise_window(row, prey_index)
        rows = torch.arange(self.world.batch_dim, device=self.world.device)
        offsets = (steps % self._noise_window).to(self.world.device)
        resample = self._wander_resample[rows, prey_index, offsets]
        headings = self._wander_headings[rows, prey_index, offsets]
        noise = self._force_noise[rows, prey_index, offsets]
        steps.add_(1)
        return resample, headings, noise

    def process_action(self, agent):
        if not agent.adversary:
            agent.action.u = self._scripted_prey_action(agent)
        else:
            super().process_action(agent)

    def observation(self, agent):
        """Stock simple_tag observations, with the prey hidden past a radius.

        When `predator_sensing_radius` is set, a predator's view of the prey is
        zeroed beyond that distance and one visibility flag per prey is appended
        (1.0 = in range). With radius None, every flag is one and prey fields
        remain visible. Predators still see each other's positions and the
        landmarks; private teammate velocities remain potential message content
        even when prey visibility is global.

        The masked block is found by layout rather than by a hardcoded index.
        `SimpleTagScenario.observation` walks `world.agents` in order, and
        `make_world` adds every adversary before every good agent, so for a
        predator the tail is always all prey positions followed by all prey
        velocities. The prey's own observation is untouched -- it is scripted,
        so its policy never runs.
        """
        obs = super().observation(agent)
        if not agent.adversary:
            return obs

        prey = self.good_agents()
        if self.predator_sensing_radius is None:
            return torch.cat([obs, obs.new_ones((*obs.shape[:-1], len(prey)))], dim=-1)
        visible = torch.cat(
            [
                (
                    (other.state.pos - agent.state.pos).norm(dim=-1, keepdim=True)
                    <= self.predator_sensing_radius
                ).to(obs.dtype)
                for other in prey
            ],
            dim=-1,
        )
        mask = visible.repeat_interleave(self.world.dim_p, dim=-1)
        block = len(prey) * self.world.dim_p
        return torch.cat(
            [
                obs[..., : -2 * block],  # self, landmarks, teammates
                obs[..., -2 * block : -block] * mask,  # prey positions
                obs[..., -block:] * mask,  # prey velocities
                visible,
            ],
            dim=-1,
        )

    def _scripted_prey_action(self, prey):
        world = self.world
        device = prey.state.pos.device
        batch_dim = prey.state.pos.shape[0]

        force = torch.zeros(batch_dim, world.dim_p, device=device)
        # Tracks how "threatened" the prey is this step (0 = safe, 1 = predator on top of it)
        threat_level = torch.zeros(batch_dim, 1, device=device)

        # --- 1. Flee from nearby predators (weighted by proximity) ---
        for other in world.agents:
            if other is prey or not other.adversary:
                continue
            delta = prey.state.pos - other.state.pos  # points away from predator
            dist = delta.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            within_range = (dist < self.prey_detection_radius).float()

            # Linear falloff: 1 at distance 0, 0 at detection_radius
            urgency = ((self.prey_detection_radius - dist) / self.prey_detection_radius).clamp(
                min=0.0
            )
            direction = delta / dist

            force += within_range * urgency * direction
            threat_level = torch.maximum(threat_level, within_range * urgency)

        # --- 2. Avoid static obstacles ---
        for landmark in world.landmarks:
            if not landmark.collide:
                continue
            delta = prey.state.pos - landmark.state.pos
            dist = delta.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            margin = self.prey_obstacle_margin + landmark.shape.radius + prey.shape.radius
            within_range = (dist < margin).float()
            urgency = ((margin - dist) / margin).clamp(min=0.0)
            direction = delta / dist
            force += within_range * urgency * direction

        # --- 3. Soft boundary avoidance (VMAS has no hard walls by default) ---
        if world.x_semidim is not None:
            x = prey.state.pos[:, 0:1]
            limit = world.x_semidim - self.prey_boundary_margin
            over = (x.abs() > limit).float()
            push = -torch.sign(x) * ((x.abs() - limit) / self.prey_boundary_margin).clamp(min=0.0)
            force[:, 0:1] += over * push
            threat_level = torch.maximum(threat_level, over * push.abs())
        if world.y_semidim is not None:
            y = prey.state.pos[:, 1:2]
            limit = world.y_semidim - self.prey_boundary_margin
            over = (y.abs() > limit).float()
            push = -torch.sign(y) * ((y.abs() - limit) / self.prey_boundary_margin).clamp(min=0.0)
            force[:, 1:2] += over * push
            threat_level = torch.maximum(threat_level, over * push.abs())

        # --- 4. Idle wander when nothing is threatening it ---
        is_calm = (threat_level < 1e-3).float()
        # Occasionally re-randomize the wander heading (~2% chance/step) so it
        # drifts instead of jittering every frame.
        resample, new_dir, noise = self._prey_disturbances(prey)
        new_dir = new_dir / new_dir.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        prey.wander_dir = torch.where(resample.bool(), new_dir, prey.wander_dir)
        force += is_calm * self.prey_wander_force * prey.wander_dir

        # --- 5. Scale response to threat level instead of always maxing out ---
        # Even when fleeing, don't always demand full force -- keeps the
        # prey catchable instead of teleporting away at u_range every step.
        flee_scale = self.prey_min_flee_frac + (1 - self.prey_min_flee_frac) * threat_level
        force = force * flee_scale

        # --- 6. Small noise so behavior isn't perfectly deterministic ---
        force += self.prey_noise_std * noise

        # --- 7. Normalize direction and clamp magnitude to u_range ---
        norm = force.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = force / norm
        magnitude = norm.clamp(max=prey.u_range)
        action = direction * magnitude

        return action
