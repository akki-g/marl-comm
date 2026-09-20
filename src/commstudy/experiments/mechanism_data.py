"""Metadata-only bank selection, training-only normalization and real feeder maps."""

from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path

from commstudy.experiments.bookkeeping import read_json
from commstudy.experiments.mechanism_suite import ROOT, file_sha, write_new


def historical_inventory(doc):
    """Inspect bank declarations/seed metadata, never historical outcome trajectories."""
    roots = [ROOT / "runs", ROOT / "results", *map(Path, doc["historical_roots"])]
    files, unavailable, seeds, windows = {}, [], set(), []

    def visit(value, scope=""):
        if isinstance(value, dict):
            if type(value.get("start_row")) is int:
                horizon = value.get("horizon", 239)
                history = value.get("history", 1)
                windows.append([value["start_row"] - history + 1, value["start_row"] + horizon + 1])
            for key, child in value.items():
                if key in {"seed", "training_seed"} and type(child) is int:
                    seeds.add(child)
                elif key in {"episode_seeds", "seeds", "training_seeds"} and isinstance(
                    child, list
                ):
                    seeds.update(s for s in child if type(s) is int)
                elif isinstance(child, (dict, list)):
                    visit(child, key)
        elif isinstance(value, list):
            for child in value:
                visit(child, scope)

    for root in roots:
        if not root.exists():
            unavailable.append(str(root))
            continue
        candidates = sorted(
            {
                *root.rglob("*bank*.json"),
                *root.rglob("*contract.json"),
                *root.rglob("metadata.json"),
                *root.rglob("split_manifest.json"),
            }
        )
        for path in candidates:
            # Declarations only. Evaluation/replay/result bodies are not bank metadata.
            if any(word in path.name for word in ("evaluation", "replay", "result", "summary")):
                continue
            try:
                value = json.loads(path.read_text())
            except (ValueError, OSError) as exc:
                raise ValueError(f"Historical declaration unreadable: {path}") from exc
            files[str(path.resolve())] = file_sha(path)
            visit(value)
    proposed = set(
        doc["seeds"]
        + doc["qualification_seeds"]
        + [doc["smoke_seed"]]
        + doc["pcp_test_seeds"]
        + doc["pcp_development_seeds"]
        + doc["pcp_random_audit_seeds"]
    )
    proposed.update(range(3200000, 3200016))
    proposed.update(range(3300000, 3300008))
    proposed.update(range(3400000, 3400008))
    proposed.update(range(3500000, 3500016))
    collision = sorted(seeds & proposed)
    if collision:
        raise ValueError(
            f"Seed namespace collision {collision}; replace the entire stage namespace"
        )
    return {
        "files": files,
        "unavailable_roots": unavailable,
        "historical_overlap_verification": (
            "available_bank_metadata_audited"
            if any("bank" in Path(p).name for p in files)
            else "unavailable"
        ),
        "all_historical_artifacts_claimed_available": False,
        "excluded_footprints": sorted({tuple(v) for v in windows}),
        "seed_collisions": collision,
        "test_outcomes_opened": False,
    }


def topology_bank(feeder_graph, agent_buses, seeds):
    import networkx as nx

    n = len(agent_buses)
    if n < 3:
        raise ValueError("Topology supplement degenerate: fewer than three inverter agents")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, bus in enumerate(agent_buses):
        distance = nx.single_source_shortest_path_length(feeder_graph, bus)
        if any(other not in distance for other in agent_buses):
            raise ValueError("Disconnected electrical bus mapping")
        nearest = sorted((distance[other], j) for j, other in enumerate(agent_buses) if j != i)
        graph.add_edges_from((i, j) for _, j in nearest[: min(2, n - 1)])
    if not nx.is_connected(graph):
        raise ValueError("Electrical agent kNN graph disconnected; declare an amendment")

    def mask(g):
        return [[i != j and g.has_edge(i, j) for j in range(n)] for i in range(n)]

    random_masks, stats = {}, {}
    original = {tuple(sorted(e)) for e in graph.edges()}
    for seed in seeds:
        found = None
        for trial in range(32):
            candidate = graph.copy()
            try:
                nx.double_edge_swap(
                    candidate,
                    nswap=max(1, graph.number_of_edges()),
                    max_tries=2000,
                    seed=3700000 + seed * 37 + trial,
                )
            except (nx.NetworkXError, nx.NetworkXAlgorithmError):
                continue
            if (
                nx.is_connected(candidate)
                and {tuple(sorted(e)) for e in candidate.edges()} != original
            ):
                found = candidate
                break
        if found is None:
            raise ValueError(
                "Topology supplement degenerate: no distinct connected degree-matched graph"
            )
        random_masks[str(seed)] = mask(found)
        stats[str(seed)] = {
            "degree": [found.degree(i) for i in range(n)],
            "edges": found.number_of_edges(),
            "connected": True,
            "generation_seed": 3700000 + seed * 37 + trial,
        }
    return {
        "status": "passed",
        "rule": "electrical_hop_knn_k2_symmetric_union_index_ties",
        "agent_buses": agent_buses,
        "graph_electrical": {str(s): mask(graph) for s in seeds},
        "graph_random": random_masks,
        "random_statistics": stats,
        "electrical_statistics": {
            "degree": [graph.degree(i) for i in range(n)],
            "edges": graph.number_of_edges(),
            "connected": True,
        },
    }


def prepare_mapdn(root, packet, history):
    import numpy as np
    from commstudy.experiments.next_phase import select_nonoverlapping_rows
    from commstudy.tasks.torchrl.mapdn_data import (
        TrainingNormalizer,
        build_split_manifest,
        save_split_manifest,
    )
    from commstudy.tasks.torchrl.power_grids import TaskConfig, _make_adapter
    from commstudy.analysis.mapdn_paired import build_mapdn_bank

    # Reuse the tested zero-reactive-action training observation collector.
    from run_mapdn_smoke import _roll_bank

    prep = root / "preparation"
    config = asdict(
        TaskConfig(
            data_path=packet["data_path"],
            manifest_path=str(prep / "split_manifest.json"),
            measurement_noise=False,
            reset_action=False,
        )
    )
    source = ROOT / "vendor/mapdn-source/COMMSTUDY_SOURCE_ORIGIN.json"
    split = build_split_manifest(
        packet["data_path"],
        239,
        history=1,
        source_revision=f"vendor-origin:{file_sha(source)}",
        settings={
            k: v
            for k, v in config.items()
            if k not in {"data_path", "manifest_path", "normalization_path"}
        },
    )
    save_split_manifest(split, prep / "split_manifest.json")
    excluded = {
        name: list(history["excluded_footprints"]) for name in ("train", "validation", "test")
    }
    banks = {}
    for name, block, count, seed in [
        ("normalization", "train", 16, 3200000),
        ("qualification", "validation", 8, 3300000),
        ("validation", "validation", 8, 3400000),
        ("test", "test", 16, 3500000),
    ]:
        starts = select_nonoverlapping_rows(
            split["splits"][block]["starts"],
            count,
            horizon=239,
            history=1,
            excluded=excluded[block],
        )
        banks[name] = [
            {"episode_id": f"mechanisms_v1:{name}:{i}", "start_row": row, "seed": seed + i}
            for i, row in enumerate(starts)
        ]
        excluded[block].extend((r, r + 240) for r in starts)
    write_new(prep / "mapdn_banks.json", banks)
    summary, observations = _roll_bank(
        config, banks["normalization"], split="train", retain_observations=True
    )
    if summary["solver_failure_count"] or any(
        e["transition_count"] != 239 for e in summary["episodes"]
    ):
        raise ValueError("Training-only normalizer collection failed; no replacement windows")
    write_new(prep / "normalization_collection.json", summary)
    normalizer = TrainingNormalizer.fit(
        np.stack(observations), split="train", manifest_sha256=split["sha256"]
    )
    normalizer.save(prep / "normalizer.json")
    config["normalization_path"] = str(prep / "normalizer.json")
    for name in ("qualification", "validation", "test"):
        bank = build_mapdn_bank(
            config,
            banks[name],
            split="test" if name == "test" else "validation",
            purpose=f"mechanisms_v1_{name}",
            source_root=ROOT,
        )
        write_new(prep / f"mapdn_{name}_bank.json", bank)
    env = _make_adapter(config, seed=900, split="train")
    try:
        net = env._env.powergrid
        mapping = [
            {"agent": agent, "inverter_index": int(i), "bus": int(net.sgen.loc[i, "bus"])}
            for agent, i in zip(env.possible_agents, net.sgen.index, strict=True)
        ]
        physical = {
            "agent_to_inverter_bus": mapping,
            "bus_ids": [int(i) for i in net.bus.index],
            "inverter_capacity_source": "training block only",
            "critic_information": "concatenated zone-local observations",
            "tables": {
                name: json.loads(net[name].to_json(orient="split"))
                for name in ("line", "trafo", "switch")
            },
            "data": split["data"],
            "source_origin": read_json(source),
            "license": "vendor/mapdn-source/LICENSE; data upstream MAPDN attribution",
            "download_receipt": read_json(Path(packet["data_path"]) / "download_receipt.json"),
        }
        write_new(prep / "data_inventory.json", physical)
        doc = read_json(prep / "suite.json")
        if "mapdn_topology" in doc["include"]:
            import pandapower.topology as top

            topology = topology_bank(
                top.create_nxgraph(net, respect_switches=True),
                [m["bus"] for m in mapping],
                doc["seeds"],
            )
            write_new(prep / "topologies.json", topology)
    finally:
        env.close()
    return config
