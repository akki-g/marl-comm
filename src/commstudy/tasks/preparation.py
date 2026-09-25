"""One training-only calibration shared by every MAPDN policy in an experiment."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from commstudy.experiments.bookkeeping import atomic_write_json, read_json
from commstudy.tasks.mapdn import TaskConfig
from commstudy.tasks.mapdn_data import (
    TrainingNormalizer,
    build_split_manifest,
    save_split_manifest,
)


def data_manifest(config):
    """Validate all four local input files and the complete chronological windows."""
    params = asdict(TaskConfig(**config["task_params"]))
    settings = {
        k: v
        for k, v in params.items()
        if k not in {"data_path", "manifest_path", "normalization_path"}
    }
    return build_split_manifest(
        params["data_path"], params["max_steps"], params["history"], settings=settings
    )


def normalization_windows(config, manifest):
    starts = manifest["splits"]["train"]["starts"]
    count = config["normalization"]["episodes"]
    chosen = [starts[((2 * i + 1) * len(starts)) // (2 * count)] for i in range(count)]
    width = manifest["max_steps"] + manifest["history"]
    if any(b - a < width for a, b in zip(chosen[:-1], chosen[1:], strict=True)):
        raise ValueError("Training split is too short for nonoverlapping normalization episodes")
    return [
        {"start_row": row, "seed": config["normalization"]["seed"] + i}
        for i, row in enumerate(chosen)
    ]


def prepare_mapdn(config, manifest):
    from commstudy.adapters.mapdn import _make_adapter
    from commstudy.utils.rng import preserve_rng_state

    root = Path(config["output_dir"]) / "data"
    root.mkdir(parents=True, exist_ok=True)
    windows = normalization_windows(config, manifest)
    record = {
        "manifest_sha256": manifest["sha256"],
        "windows": windows,
        "split": "train",
        "policy": "zero_reactive_action",
        "observations": "initial and every subsequent observation, all agents",
    }
    normalizer_path = root / "normalizer.json"
    if normalizer_path.exists():
        if read_json(root / "split.json") != manifest:
            raise ValueError("Existing MAPDN split is missing or incompatible")
        normalizer = TrainingNormalizer.load(normalizer_path)
        stored = read_json(root / "normalization.json", {})
        if (
            normalizer.manifest_sha256 != manifest["sha256"]
            or any(stored.get(k) != v for k, v in record.items())
            or stored.get("normalizer_sha256") != normalizer.sha256
        ):
            raise ValueError("Existing MAPDN preparation is incompatible with this configuration")
        return stored
    split_path = root / "split.json"
    if split_path.exists():
        if read_json(split_path) != manifest:
            raise ValueError("Existing MAPDN split differs from the current data/settings")
    else:
        save_split_manifest(manifest, split_path)
    params = {**config["task_params"], "manifest_path": str(split_path), "normalization_path": None}
    observations, episodes = [], []
    with preserve_rng_state():
        env = _make_adapter(params, seed=windows[0]["seed"])
        try:
            for window in windows:
                obs, _ = env.reset(seed=window["seed"], options={"start_row": window["start_row"]})
                agents = list(env.agents)
                observations.append(np.stack([obs[a] for a in agents]))
                total = 0.0
                for step in range(manifest["max_steps"]):  # noqa: B007 - used after the loop
                    obs, reward, terminated, truncated, _ = env.step(
                        {a: np.zeros(1, dtype=np.float32) for a in agents}
                    )
                    observations.append(np.stack([obs[a] for a in agents]))
                    total += reward[agents[0]]
                    if any(terminated.values()) or env.get_diagnostics().get("destroy", 0):
                        raise ValueError(f"Power flow failed in normalization window {window}")
                    if all(truncated.values()):
                        break
                if step + 1 != manifest["max_steps"] or not all(truncated.values()):
                    raise ValueError("Normalization episode did not complete its declared horizon")
                episodes.append({**window, "transitions": step + 1, "return": total})
        finally:
            env.close()
    normalizer = TrainingNormalizer.fit(
        np.stack(observations), split="train", manifest_sha256=manifest["sha256"]
    )
    record.update(episodes=episodes, normalizer_sha256=normalizer.sha256)
    atomic_write_json(root / "normalization.json", record)
    normalizer.save(normalizer_path)
    return record
