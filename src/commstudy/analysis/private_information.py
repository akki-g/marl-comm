"""Constructed-state, finite-action private-information diagnostics.

Privileged state is used only to set up counterfactual simulator experiments.
No diagnostic state or value enters any learned actor. See the prospective plan
for the strict, deliberately limited interpretation of these quantities.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


VERSION = "private_information_constructed_pairs_v1"
UTILITY_TOLERANCE = 1e-10


def json_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_plan():
    """Declared before any simulator outcomes; no adaptive search parameters."""
    return {
        "version": VERSION,
        "utility_tolerance": UTILITY_TOLERANCE,
        "pcp": {
            "seeds": [910101, 910102], "rotations": [0.0, math.pi / 2],
            "visibility": ["radius1", "global"],
            "interventions": ["prey_position", "teammate_velocity"],
            "horizon": 25, "recipient": 0, "donor": 1,
            "recipient_position": [-0.35, 0.0],
            "donor_position": [0.55, 0.0], "third_position": [-0.65, 0.65],
            "landmarks": [[-0.75, -0.70], [0.20, -0.75]],
            "prey_position_states": [[0.70, -0.35], [0.70, 0.35]],
            "velocity_pair_donor_position": [0.35, 0.0],
            "velocity_pair_prey_position": [0.65, 0.35],
            "donor_velocity_states": [[0.0, -0.8], [0.0, 0.8]],
            "actions": [[0.0, 0.0]] + [
                [math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)]
                for k in range(8)],
        },
        "mapdn": {
            "seeds": [920101, 920102, 920103, 920104],
            "row_selection": "four_evenly_spaced_training_start_indices_inclusive",
            "recipient": 0, "donor_selection": "first_inverter_in_distinct_zone",
            "load_multipliers": [0.8, 1.2],
            "actions": [-0.8, -0.4, 0.0, 0.4, 0.8], "steps": 1,
            "max_steps": 239, "history": 1, "measurement_noise": False,
            "normalizer": None,
        },
    }


def observation_comparison(recipient, donor):
    recipient, donor = np.asarray(recipient), np.asarray(donor)
    if recipient.ndim != 2 or donor.ndim != 2 or len(recipient) != 2 or len(donor) != 2:
        raise ValueError("Observation comparisons require two complete vectors per agent.")
    if not np.isfinite(recipient).all() or not np.isfinite(donor).all():
        raise ValueError("Observation comparisons cannot contain nonfinite values.")
    return {
        "recipient_exact_equal": bool(np.array_equal(recipient[0], recipient[1])),
        "recipient_linf_distance": float(np.abs(recipient[0] - recipient[1]).max()),
        "donor_exact_equal": bool(np.array_equal(donor[0], donor[1])),
        "donor_linf_distance": float(np.abs(donor[0] - donor[1]).max()),
    }


def finite_action_summary(utilities, *, attributable, tolerance=UTILITY_TOLERANCE):
    """Uniform two-state mixture, with ties preserved and no population claim."""
    values = np.asarray(utilities, dtype=float)
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] < 1:
        raise ValueError("Utilities must be a complete two-state by action table.")
    if not np.isfinite(values).all():
        return {"complete": False, "private_information_attribution_valid": False}
    maxima = values.max(axis=1)
    preferences = [np.flatnonzero(row >= maximum - tolerance).tolist()
                   for row, maximum in zip(values, maxima, strict=True)]
    informed = float(maxima.mean())
    uninformed = float(values.mean(axis=0).max())
    difference = informed - uninformed
    # A tiny negative here is floating-point arithmetic, not negative information value.
    if difference < -tolerance:
        raise ArithmeticError("Finite-grid information value violated its nonnegative bound.")
    value = max(0.0, difference)
    return {
        "complete": True, "utility_table": values.tolist(),
        "maximizer_sets": preferences,
        "strict_preference_conflict": not bool(set(preferences[0]) & set(preferences[1])),
        "mean_state_best_utility": informed, "best_common_action_mean_utility": uninformed,
        "conditional_state_information_value": value,
        "private_information_attribution_valid": bool(attributable),
        "private_information_value": value if attributable else None,
        "interpretation": "constructed equally weighted pair; declared finite action plans only",
    }


def _rotation(vector, angle):
    x, y = vector
    return [math.cos(angle) * x - math.sin(angle) * y,
            math.sin(angle) * x + math.cos(angle) * y]


def _pcp_setup(env, spec, intervention, state, angle, seed):
    import torch
    env.set_seed(seed)
    td = env.reset()
    scenario = env.base_env.scenario
    positions = [spec["recipient_position"], spec["donor_position"], spec["third_position"],
                 spec["prey_position_states"][state]]
    velocities = [[0.0, 0.0] for _ in positions]
    if intervention == "teammate_velocity":
        positions[1] = spec["velocity_pair_donor_position"]
        positions[3] = spec["velocity_pair_prey_position"]
        velocities[1] = spec["donor_velocity_states"][state]
    for agent, position, velocity in zip(scenario.world.agents, positions, velocities, strict=True):
        agent.set_pos(
            torch.tensor([_rotation(position, angle)], dtype=torch.float32), batch_index=None)
        agent.set_vel(
            torch.tensor([_rotation(velocity, angle)], dtype=torch.float32), batch_index=None)
    for landmark, position in zip(scenario.world.landmarks, spec["landmarks"], strict=True):
        landmark.set_pos(
            torch.tensor([_rotation(position, angle)], dtype=torch.float32), batch_index=None)
    entities = list(scenario.world.agents) + list(scenario.world.landmarks)
    finite = all(torch.isfinite(entity.state.pos).all().item() for entity in entities)
    in_bounds = all((entity.state.pos.abs() <= scenario.bound).all().item()
                    for entity in entities)
    speeds_valid = all(agent.state.vel.norm().item() <= agent.max_speed + 1e-6
                       for agent in scenario.world.agents)
    overlap = any((left.state.pos - right.state.pos).norm().item() < (
        left.shape.radius + right.shape.radius)
        for index, left in enumerate(entities) for right in entities[index + 1:])
    observations = [scenario.observation(agent)[0].tolist() for agent in scenario.adversaries()]
    physical = {"positions": [entity.state.pos[0].tolist() for entity in entities],
                "velocities": [agent.state.vel[0].tolist() for agent in scenario.world.agents]}
    noise = {
        "episode_ids": scenario.exogenous_episode_ids.tolist(),
        "steps": scenario.exogenous_steps.tolist(),
        "force_noise_sha256": hashlib.sha256(
            scenario._force_noise.detach().cpu().numpy().tobytes()).hexdigest(),
        "wander_headings_sha256": hashlib.sha256(
            scenario._wander_headings.detach().cpu().numpy().tobytes()).hexdigest(),
        "wander_resample_sha256": hashlib.sha256(
            scenario._wander_resample.detach().cpu().numpy().tobytes()).hexdigest(),
    }
    return td, scenario, {
        "native_state_valid": bool(finite and in_bounds and speeds_valid and not overlap),
        "finite": bool(finite), "inside_bounds": bool(in_bounds),
        "within_speed_limits": bool(speeds_valid), "initial_collision": bool(overlap),
        "recipient_observation": observations[spec["recipient"]],
        "donor_observation": observations[spec["donor"]],
        "physical_state": physical, "physical_state_sha256": json_digest(physical),
        "exogenous_schedule": noise,
        "reachability_from_reset_proven": False,
    }


def _pcp_rollout(env, spec, intervention, state, angle, seed, action):
    import torch
    td, scenario, setup = _pcp_setup(env, spec, intervention, state, angle, seed)
    if not setup["native_state_valid"]:
        return {"setup": setup, "valid": False, "attempted_transitions": 0,
                "error": "constructed state failed native physical checks"}
    reward, contact, distance = 0.0, False, 0.0
    for _ in range(spec["horizon"]):
        td = td.clone()
        td.update(env.full_action_spec.zero())
        td["adversary", "action"][0, spec["recipient"]] = torch.tensor(action)
        td = env.step(td)["next"]
        reward += float(td["adversary", "reward"].mean())
        contact |= bool(td["adversary", "info", "any_contact"].any())
        delta = scenario.adversaries()[spec["recipient"]].state.pos - (
            scenario.good_agents()[0].state.pos)
        distance += float(delta.norm())
    return {"setup": setup, "valid": True, "attempted_transitions": spec["horizon"],
            "native_return": reward, "any_contact": float(contact),
            "negative_mean_recipient_distance": -distance / spec["horizon"]}


def run_pcp_diagnostic(plan):
    from commstudy.tasks import resolve_task
    from commstudy.utils.rng import preserve_rng_state
    spec, pairs = plan["pcp"], []
    # Import/initialize VMAS before capturing its class-shared RNG stream.
    resolve_task("vmas_predator_capture_prey", {})
    with preserve_rng_state():
        for visibility in spec["visibility"]:
            task = resolve_task("vmas_predator_capture_prey", {
                "max_steps": 100,
                "predator_sensing_radius": 1.0 if visibility == "radius1" else None})
            env = task.get_env_fun(1, True, spec["seeds"][0], "cpu")()
            try:
                for intervention in spec["interventions"]:
                    for angle in spec["rotations"]:
                        for seed in spec["seeds"]:
                            rows = [[_pcp_rollout(
                                env, spec, intervention, state, angle, seed, action)
                                for action in spec["actions"]] for state in range(2)]
                            setups = [state_rows[0]["setup"] for state_rows in rows]
                            setup_replayed = all(
                                row["setup"] == state_rows[0]["setup"]
                                for state_rows in rows for row in state_rows)
                            exogenous_paired = setups[0]["exogenous_schedule"] == (
                                setups[1]["exogenous_schedule"])
                            comparison = observation_comparison(
                                [s["recipient_observation"] for s in setups],
                                [s["donor_observation"] for s in setups])
                            valid = all(row["valid"] for state_rows in rows for row in state_rows)
                            attribution = valid and setup_replayed and exogenous_paired and (
                                comparison["recipient_exact_equal"] and
                                not comparison["donor_exact_equal"])
                            summaries = {metric: finite_action_summary(
                                [[row.get(metric, float("nan")) for row in state_rows]
                                 for state_rows in rows], attributable=attribution)
                                for metric in ("native_return", "any_contact",
                                               "negative_mean_recipient_distance")}
                            pairs.append({
                                "visibility": visibility, "intervention": intervention,
                                "rotation": angle, "seed": seed, "valid": valid,
                                "initial_state_exactly_replayed_across_actions": setup_replayed,
                                "exogenous_schedule_paired": exogenous_paired,
                                "observation_comparison": comparison,
                                "private_information_attribution_valid": attribution,
                                "states": setups, "utilities": summaries,
                                "rollouts": [[{k: v for k, v in row.items() if k != "setup"}
                                              for row in state_rows] for state_rows in rows],
                            })
            finally:
                env.close()
    return {"version": VERSION, "environment": "pcp", "pairs": pairs,
            "attempted_pairs": len(pairs),
            "valid_pairs": sum(pair["valid"] for pair in pairs),
            "exact_private_information_pairs": sum(
                pair["private_information_attribution_valid"] for pair in pairs),
            "attempted_transitions": sum(row["attempted_transitions"] for pair in pairs
                                         for state_rows in pair["rollouts"] for row in state_rows)}


def prepare_mapdn_inputs(output, data_path):
    from commstudy.tasks.torchrl.mapdn_data import build_split_manifest, save_split_manifest
    from commstudy.tasks.torchrl.power_grids import TaskConfig
    output = Path(output)
    config = asdict(TaskConfig(
        data_path=str(Path(data_path).resolve()),
        manifest_path=str((output / "mapdn_split_manifest.json").resolve()),
        max_steps=239, history=1, measurement_noise=False, reset_action=False))
    manifest = build_split_manifest(
        config["data_path"], config["max_steps"], history=1,
        settings={key: value for key, value in config.items()
                  if key not in {"data_path", "manifest_path", "normalization_path"}})
    save_split_manifest(manifest, config["manifest_path"])
    starts = manifest["splits"]["train"]["starts"]
    indices = np.linspace(0, len(starts) - 1, 4, dtype=int)
    return {"task_config": config, "split": "train", "starts": [starts[i] for i in indices],
            "manifest_sha256": manifest["sha256"],
            "manifest_file_sha256": file_digest(config["manifest_path"])}


def _mapdn_setup(adapter, row, seed, multiplier, recipient, donor, load_indices):
    adapter.reset(seed=seed, options={"start_row": row})
    native = adapter._env
    for column in ("p_mw", "q_mvar"):
        native.powergrid.load.loc[load_indices, column] *= multiplier
    loads = native.powergrid.load[["p_mw", "q_mvar"]].to_numpy()
    loads_valid = bool(np.isfinite(loads).all() and (loads >= 0).all())
    solved = bool(loads_valid and native._solve())
    if not solved:
        return {"native_state_valid": False, "load_values_valid": loads_valid,
                "counterfactual_power_flow_solved": solved}
    raw = np.asarray(native.get_obs())
    observations = adapter._observation_dict(raw)
    state = {"load_p_mw": native.powergrid.load["p_mw"].tolist(),
             "load_q_mvar": native.powergrid.load["q_mvar"].tolist(),
             "sgen_p_mw": native.powergrid.sgen["p_mw"].tolist(),
             "sgen_q_mvar": native.powergrid.sgen["q_mvar"].tolist(),
             "voltage": native.powergrid.res_bus["vm_pu"].tolist(),
             "angle": native.powergrid.res_bus["va_degree"].tolist()}
    return {
        "native_state_valid": True, "load_values_valid": True,
        "counterfactual_power_flow_solved": True, "physical_state_sha256": json_digest(state),
        "physical_state": state, "raw_recipient_observation": raw[recipient].tolist(),
        "raw_donor_observation": raw[donor].tolist(),
        "recipient_observation": observations[f"agent_{recipient}"].tolist(),
        "donor_observation": observations[f"agent_{donor}"].tolist(),
        "exogenous_rng_state_sha256": json_digest(native._rng.bit_generator.state),
        "profile_row": int(native._row), "reachability_from_profile_reset_proven": False,
    }


def run_mapdn_diagnostic(plan):
    from commstudy.tasks.torchrl.power_grids import _make_adapter
    from commstudy.utils.rng import preserve_rng_state
    spec, inputs, pairs = plan["mapdn"], plan["mapdn_inputs"], []
    config = inputs["task_config"]
    if config["history"] != 1 or config.get("normalization_path") is not None:
        raise ValueError("Diagnostic requires history 1 and explicitly raw actor observations.")
    if file_digest(config["manifest_path"]) != inputs["manifest_file_sha256"]:
        raise ValueError("Diagnostic MAPDN manifest changed after preparation.")
    with preserve_rng_state():
        adapter = _make_adapter(config, seed=spec["seeds"][0], split="train")
        try:
            recipient = spec["recipient"]
            topology = adapter.agent_topology
            local_zone = topology[f"agent_{recipient}"]["zone"]
            candidates = [i for i in range(adapter.num_mapdn_agents)
                          if topology[f"agent_{i}"]["zone"] != local_zone]
            if not candidates:
                return {"version": VERSION, "environment": "mapdn", "attempted_pairs": 4,
                        "valid_pairs": 0, "exact_private_information_pairs": 0,
                        "pairs": [], "error": "no donor inverter in a distinct zone"}
            donor = candidates[0]
            donor_zone = topology[f"agent_{donor}"]["zone"]
            network = adapter._env.base_powergrid
            buses = network.bus.index[network.bus["zone"].astype(str) == donor_zone].tolist()
            load_indices = network.load.index[network.load["bus"].isin(buses)].tolist()
            if not load_indices:
                raise ValueError("Predeclared donor zone has no loads to perturb.")
            for row, seed in zip(inputs["starts"], spec["seeds"], strict=True):
                states, outcomes, failures = [], [], []
                for multiplier in spec["load_multipliers"]:
                    setups, state_rows = [], []
                    for action in spec["actions"]:
                        setup = {"native_state_valid": False}
                        transition_attempted = 0
                        try:
                            setup = _mapdn_setup(
                                adapter, row, seed, multiplier, recipient, donor, load_indices)
                            if not setup["native_state_valid"]:
                                setups.append(setup)
                                state_rows.append({"valid": False, "attempted_transitions": 0})
                                continue
                            actions = np.zeros(adapter.num_mapdn_agents)
                            actions[recipient] = action
                            transition_attempted = 1
                            reward, _, info = adapter._env.step(actions)
                            state_rows.append({
                                "valid": True, "attempted_transitions": 1,
                                "native_reward": float(reward),
                                "negative_voltage_violation": (
                                    -float(info["voltage_violation_magnitude"])
                                    if info["voltage_metrics_valid"] else None),
                                "domain": {key: float(value) for key, value in info.items()},
                            })
                        except Exception as exc:
                            failures.append({"multiplier": multiplier, "action": action,
                                             "error_type": type(exc).__name__, "error": str(exc)})
                            state_rows.append({"valid": False,
                                               "attempted_transitions": transition_attempted})
                        setups.append(setup)
                    states.append(setups)
                    outcomes.append(state_rows)
                valid = all(r["valid"] for state_rows in outcomes for r in state_rows)
                replayed = all(s == state_set[0] for state_set in states for s in state_set)
                pair = {"row": row, "seed": seed, "valid": valid,
                        "initial_state_exactly_replayed_across_actions": replayed,
                        "states": [s[0] for s in states], "rollouts": outcomes,
                        "failures": failures}
                if all(state_set[0]["native_state_valid"] for state_set in states):
                    pair["raw_observation_comparison"] = observation_comparison(
                        [s[0]["raw_recipient_observation"] for s in states],
                        [s[0]["raw_donor_observation"] for s in states])
                    comparison = observation_comparison(
                        [s[0]["recipient_observation"] for s in states],
                        [s[0]["donor_observation"] for s in states])
                    pair["observation_comparison"] = comparison
                    exogenous_paired = states[0][0]["exogenous_rng_state_sha256"] == (
                        states[1][0]["exogenous_rng_state_sha256"])
                    pair["exogenous_rng_paired"] = exogenous_paired
                    attributable = valid and replayed and exogenous_paired and (
                        comparison["recipient_exact_equal"] and not comparison["donor_exact_equal"])
                else:
                    attributable = False
                pair["private_information_attribution_valid"] = attributable
                pair["interpretation"] = ("exact native actor-observation pair" if attributable
                                          else "inconclusive for private information")
                pair["utilities"] = {metric: finite_action_summary(
                    [[r.get(metric) if r.get(metric) is not None else float("nan")
                      for r in state_rows] for state_rows in outcomes], attributable=attributable)
                    for metric in ("native_reward", "negative_voltage_violation")}
                pairs.append(pair)
        finally:
            adapter.close()
    return {
        "version": VERSION, "environment": "mapdn", "pairs": pairs,
        "recipient": topology[f"agent_{recipient}"], "donor": topology[f"agent_{donor}"],
        "changed_load_indices": [int(i) for i in load_indices],
        "changed_load_buses": [int(network.load.loc[i, "bus"]) for i in load_indices],
        "attempted_pairs": len(pairs), "valid_pairs": sum(p["valid"] for p in pairs),
        "attempted_action_evaluations": sum(len(s) for p in pairs for s in p["rollouts"]),
        "exact_private_information_pairs": sum(
            p["private_information_attribution_valid"] for p in pairs),
        "attempted_transitions": sum(r["attempted_transitions"] for p in pairs
                                     for s in p["rollouts"] for r in s),
        "action_solver_failures": sum(r.get("domain", {}).get("action_solver_failure", 0)
                                      for p in pairs for s in p["rollouts"] for r in s),
        "advance_solver_failures": sum(r.get("domain", {}).get("advance_solver_failure", 0)
                                       for p in pairs for s in p["rollouts"] for r in s),
    }


def source_inventory(root):
    """Bind package, runtime task defaults, vendor and diagnostic interface."""
    root = Path(root)
    paths = sorted((root / "src/commstudy").rglob("*.py"))
    paths += sorted(path for path in (root / "configs").rglob("*") if path.is_file())
    paths += sorted(path for path in (root / "vendor/mapdn-source").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                    and path.suffix != ".pyc")
    paths += [root / "scripts/diagnose_private_information.py",
              root / "docs/PRIVATE_INFORMATION_DIAGNOSTIC_PLAN.md"]
    return {str(path.relative_to(root)): file_digest(path) for path in paths}


def verify_plan(plan, root):
    material = deepcopy(plan)
    digest = material.pop("sha256", None)
    if json_digest(material) != digest:
        raise ValueError("Diagnostic plan content digest mismatch.")
    declared = default_plan()
    if any(plan.get(key) != value for key, value in declared.items()):
        raise ValueError("Diagnostic design differs from its prospective version.")
    if source_inventory(root) != plan["source_inventory"]:
        raise ValueError("Scientific source changed after diagnostic plan preparation.")
