"""Read-only PCP transition measurements with explicit units and scope.

Values describe the post-physics, pre-reward state of one transition. The
scenario publishes the same team value on every predator's ``info`` leaf;
consumers must select one copy or average copies, never sum over predators.
TorchRL's VMAS wrapper encodes all info values as float32, including indicators
and exact counts. None of these fields belongs in actor or critic input specs.
"""

from __future__ import annotations

import torch


PCP_DOMAIN_PROTOCOL = "pcp_transition_metrics_v1"
BOUNDARY_BAND_FRACTION = 0.9


def pcp_metric_specs(num_predators: int) -> dict[str, dict[str, str]]:
    """Stable metric names, value kinds, units, and interpretation.

    Visibility bins run from zero through the configured predator count, so
    the standard three-predator task has exactly bins 0, 1, 2, and 3. Counts
    measure occupancy on this transition, not newly occurring contact events.
    """
    definitions = {
        "expected_predator_reward": (
            "signed_reward",
            "reward per predator",
            "Expected mean predator reward from this pre-respawn state: contact reward "
            "10 * contact_pairs minus 0.1 times the sum of each predator's nearest-prey "
            "distance when adversary shaping is enabled; divide by predator count "
            "only when adversaries_share_rew is false. Computed independently of reward().",
        ),
        "contact_pairs": (
            "count",
            "predator-prey pairs",
            "Reward-eligible contact pairs: colliding predator enabled and center distance "
            "strictly less than the sum of radii, exactly as inherited adversary_reward.",
        ),
        "any_contact": (
            "indicator",
            "0 or 1",
            "At least one reward-eligible predator-prey contact.",
        ),
        "simultaneous_contact": (
            "indicator",
            "0 or 1",
            "At least two predators contact the same prey this transition.",
        ),
        "contacting_predators": (
            "count",
            "predators",
            "Distinct predators in contact with at least one prey.",
        ),
        "nearest_predator_prey_distance": (
            "distance",
            "VMAS position units",
            "Minimum center distance over every predator-prey pair.",
        ),
        "mean_predator_nearest_prey_distance": (
            "distance",
            "VMAS position units",
            "Mean over predators of their nearest prey center distance.",
        ),
        "predator_boundary_fraction": (
            "fraction",
            "fraction of predators",
            "Fraction whose center has any abs coordinate >= 0.9 * bound; a fixed outer-10% band.",
        ),
        "prey_boundary_fraction": (
            "fraction",
            "fraction of prey",
            "Fraction whose center has any abs coordinate >= 0.9 * bound; a fixed outer-10% band.",
        ),
        "predator_obstacle_collision_pairs": (
            "count",
            "predator-landmark pairs",
            "Strict center-distance/radii overlaps with colliding landmarks; "
            "both collision flags enabled.",
        ),
        "predator_teammate_collision_pairs": (
            "count",
            "unordered predator pairs",
            "Strict center-distance/radii overlaps between enabled predators; "
            "each pair counted once.",
        ),
        "sender_visible_receiver_blind_pairs": (
            "count",
            "directed receiver-sender pairs",
            "Distinct ordered pairs with at least one prey visible to sender and hidden "
            "from receiver; an information opportunity independent of communication "
            "topology or sender selection.",
        ),
        "sender_visible_receiver_blind_prey_triples": (
            "count",
            "directed receiver-sender-prey triples",
            "Ordered pair/prey combinations where sender sees prey and receiver does not; "
            "different prey can contribute to the same pair.",
        ),
    }
    for count in range(num_predators + 1):
        definitions[f"prey_visible_to_{count}_predators"] = (
            "count",
            "prey",
            f"Number of prey visible to exactly {count} predators.",
        )
    return {
        key: {
            "kind": kind,
            "units": units,
            "description": description,
            "scope": "team value repeated per predator",
            "transport_dtype": "float32",
        }
        for key, (kind, units, description) in definitions.items()
    }


@torch.no_grad()
def compute_pcp_domain_metrics(scenario) -> dict[str, torch.Tensor]:
    """Measure the current physical state without advancing state or RNG.

    Every returned tensor is fresh, detached, float32 and shaped [worlds, 1].
    ``expected_predator_reward`` independently accounts for both configured
    shaping and sharing. With unshaped, shared predator rewards, every predator
    receives exactly ``10 * contact_pairs``. This never invokes reward(), which
    can respawn prey and mutate the simulator.
    """
    predators, prey = scenario.adversaries(), scenario.good_agents()
    device = scenario.world.device
    predator_positions = torch.stack([agent.state.pos for agent in predators], dim=1)
    prey_positions = torch.stack([agent.state.pos for agent in prey], dim=1)
    predator_radii = predator_positions.new_tensor([agent.shape.radius for agent in predators])
    prey_radii = prey_positions.new_tensor([agent.shape.radius for agent in prey])
    predator_enabled = torch.tensor([agent.collide for agent in predators], device=device)
    distances = torch.linalg.vector_norm(
        predator_positions[:, :, None] - prey_positions[:, None, :], dim=-1
    )
    contacts = (distances < predator_radii[:, None] + prey_radii[None, :]) & predator_enabled[
        None, :, None
    ]
    contact_pairs = contacts.sum(dim=(1, 2))
    nearest_prey = distances.amin(dim=2)
    individual_reward = contacts.sum(dim=2).to(torch.float32) * 10
    if scenario.shape_adversary_rew:
        individual_reward = individual_reward - 0.1 * nearest_prey
    expected_reward = (
        individual_reward.sum(dim=1)
        if scenario.adversaries_share_rew else individual_reward.mean(dim=1)
    )
    predator_distances = torch.linalg.vector_norm(
        predator_positions[:, :, None] - predator_positions[:, None, :], dim=-1
    )
    teammate_overlaps = (
        (predator_distances < predator_radii[:, None] + predator_radii[None, :])
        & predator_enabled[None, :, None]
        & predator_enabled[None, None, :]
    )
    obstacles = [landmark for landmark in scenario.world.landmarks if landmark.collide]
    obstacle_pairs = torch.zeros(scenario.world.batch_dim, device=device, dtype=torch.float32)
    if obstacles:
        obstacle_positions = torch.stack([landmark.state.pos for landmark in obstacles], dim=1)
        obstacle_radii = predator_positions.new_tensor(
            [landmark.shape.radius for landmark in obstacles]
        )
        obstacle_distances = torch.linalg.vector_norm(
            predator_positions[:, :, None] - obstacle_positions[:, None, :], dim=-1
        )
        obstacle_pairs = (
            (obstacle_distances < predator_radii[:, None] + obstacle_radii[None, :])
            & predator_enabled[None, :, None]
        ).sum(dim=(1, 2))
    visible = (
        torch.ones_like(distances, dtype=torch.bool)
        if scenario.predator_sensing_radius is None
        else distances <= scenario.predator_sensing_radius
    )
    visible_predators_per_prey = visible.sum(dim=1)
    # Receiver row, sender column, prey channel. Self-pairs are always false.
    opportunities = visible[:, None, :, :] & ~visible[:, :, None, :]
    threshold = BOUNDARY_BAND_FRACTION * scenario.bound
    metrics = {
        "expected_predator_reward": expected_reward,
        "contact_pairs": contact_pairs,
        "any_contact": contact_pairs > 0,
        "simultaneous_contact": (contacts.sum(dim=1) >= 2).any(dim=1),
        "contacting_predators": contacts.any(dim=2).sum(dim=1),
        "nearest_predator_prey_distance": distances.amin(dim=(1, 2)),
        "mean_predator_nearest_prey_distance": nearest_prey.mean(dim=1),
        "predator_boundary_fraction": (predator_positions.abs().amax(dim=2) >= threshold)
        .float()
        .mean(dim=1),
        "prey_boundary_fraction": (prey_positions.abs().amax(dim=2) >= threshold)
        .float()
        .mean(dim=1),
        "predator_obstacle_collision_pairs": obstacle_pairs,
        "predator_teammate_collision_pairs": teammate_overlaps.triu(diagonal=1).sum(dim=(1, 2)),
        "sender_visible_receiver_blind_pairs": opportunities.any(dim=3).sum(dim=(1, 2)),
        "sender_visible_receiver_blind_prey_triples": opportunities.sum(dim=(1, 2, 3)),
    }
    metrics.update(
        {
            f"prey_visible_to_{count}_predators": (visible_predators_per_prey == count).sum(dim=1)
            for count in range(len(predators) + 1)
        }
    )
    return {key: value.to(torch.float32).unsqueeze(-1) for key, value in metrics.items()}
