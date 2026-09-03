"""
Predator Capture Prey environment 

Ports a simple_tag env from benchmarl.environments.vmas, uses prey scripting to simulate prey actions
Trains adversarial agents to attack prey
"""

from vmas.scenarios.mpe.simple_tag import Scenario as SimpleTagScenario
import torch


class PredatorCapturePreyScenario(SimpleTagScenario):
    def make_world(self, batch_dim, device, **kwargs):
        # Tunable knobs for the scripted prey. Pop them before calling super()
        # so they don't get passed through to VMAS's own kwarg parsing.
        self.prey_detection_radius = kwargs.pop("prey_detection_radius", 0.8)
        self.prey_obstacle_margin = kwargs.pop("prey_obstacle_margin", 0.4)
        self.prey_boundary_margin = kwargs.pop("prey_boundary_margin", 0.3)
        self.prey_wander_force = kwargs.pop("prey_wander_force", 0.15)
        self.prey_noise_std = kwargs.pop("prey_noise_std", 0.05)
        self.prey_min_flee_frac = kwargs.pop("prey_min_flee_frac", 0.3)

        # Distance beyond which a predator stops observing the prey. `None`
        # keeps stock simple_tag observations, where every predator always sees
        # the prey's position and velocity -- fully observable, and therefore
        # unable to pose the selection problem this study measures. See
        # `docs/RESULTS_pcp_pilot.md` for why that matters.
        self.predator_sensing_radius = kwargs.pop("predator_sensing_radius", None)

        world = super().make_world(batch_dim, device, **kwargs)
 
        # Cache a persistent per-agent wander direction so idle motion looks
        # like gentle drifting instead of jittering randomly every step.
        for agent in world.agents:
            if not agent.adversary:
                agent.wander_dir = torch.zeros(
                    batch_dim, world.dim_p, device=device
                )
        return world


    def process_action(self, agent):
        if not agent.adversary:
            agent.action.u = self._scripted_prey_action(agent)
        else:
            super().process_action(agent)


    def observation(self, agent):
        """Stock simple_tag observations, with the prey hidden past a radius.

        When `predator_sensing_radius` is set, a predator's view of the prey is
        zeroed beyond that distance and one visibility flag per prey is appended
        (1.0 = in range). Predators still see each other and the landmarks, so
        the only thing a message can carry that its receiver lacks is where the
        prey is -- which is the point: at any step only some predators can
        answer that, and which ones changes as the chase moves.

        The masked block is found by layout rather than by a hardcoded index.
        `SimpleTagScenario.observation` walks `world.agents` in order, and
        `make_world` adds every adversary before every good agent, so for a
        predator the tail is always all prey positions followed by all prey
        velocities. The prey's own observation is untouched -- it is scripted,
        so its policy never runs.
        """
        obs = super().observation(agent)
        if self.predator_sensing_radius is None or not agent.adversary:
            return obs

        prey = self.good_agents()
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
                obs[..., : -2 * block],          # self, landmarks, teammates
                obs[..., -2 * block : -block] * mask,   # prey positions
                obs[..., -block:] * mask,               # prey velocities
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
            urgency = ((self.prey_detection_radius - dist) / self.prey_detection_radius).clamp(min=0.0)
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
        resample = (torch.rand(batch_dim, 1, device=device) < 0.02).float()
        new_dir = torch.randn(batch_dim, world.dim_p, device=device)
        new_dir = new_dir / new_dir.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        prey.wander_dir = torch.where(resample.bool(), new_dir, prey.wander_dir)
        force += is_calm * self.prey_wander_force * prey.wander_dir
 
        # --- 5. Scale response to threat level instead of always maxing out ---
        # Even when fleeing, don't always demand full force -- keeps the
        # prey catchable instead of teleporting away at u_range every step.
        flee_scale = self.prey_min_flee_frac + (1 - self.prey_min_flee_frac) * threat_level
        force = force * flee_scale
 
        # --- 6. Small noise so behavior isn't perfectly deterministic ---
        force += self.prey_noise_std * torch.randn(batch_dim, world.dim_p, device=device)
 
        # --- 7. Normalize direction and clamp magnitude to u_range ---
        norm = force.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = force / norm
        magnitude = norm.clamp(max=prey.u_range)
        action = direction * magnitude
 
        return action
        


