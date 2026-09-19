"""Real case33 engineering checks for the new three-arm phase; no scientific claim."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
from commstudy.analysis.diagnostics import finiteness_report, load_frozen_experiment
from commstudy.analysis.mapdn_paired import (
    build_mapdn_bank,
    evaluate_mapdn_paired,
    verify_native_checkpoint,
)
from commstudy.experiments.bookkeeping import RunContext, parameter_counts
from commstudy.experiments.runner import run_managed_experiment
from commstudy.tasks.torchrl.mapdn_data import (
    TrainingNormalizer,
    build_split_manifest,
    save_split_manifest,
)
from commstudy.tasks.torchrl.power_grids import TaskConfig
from run_main_direction_phase import PhaseEvidence, phase_spec, source_inventory
from run_mapdn_confirmation import _runtime, _write_new_json
from run_mapdn_smoke import _actor_state, _bank, _roll_bank, _sha, _source_provenance, _state_sha


def run(root, output, data):
    before, runtime = source_inventory(), _runtime()
    output.mkdir(parents=True, exist_ok=False)
    config = asdict(
        TaskConfig(
            data_path=str(data),
            manifest_path=str(output / "manifest.json"),
            max_steps=16,
            history=1,
            measurement_noise=False,
            reset_action=False,
        )
    )
    sources = _source_provenance(root)
    manifest = build_split_manifest(
        data,
        16,
        history=1,
        source_revision=f"vendor-tree-sha256:{sources['mapdn_tree_sha256']}",
        settings={
            key: value
            for key, value in config.items()
            if key not in {"data_path", "manifest_path", "normalization_path"}
        },
    )
    save_split_manifest(manifest, output / "manifest.json")
    training_bank = _bank(manifest, "train", 2, 980100)
    validation_bank = _bank(manifest, "validation", 2, 980200)
    _write_new_json(
        output / "precollection.json",
        {
            "purpose": "engineering_only",
            "source_inventory": before,
            "runtime": runtime,
            "training_bank": training_bank,
            "validation_bank": validation_bank,
            "seeds": [980300, 980301, 980302],
            "frames_per_method": 64,
            "scientific_learning_confirmed": False,
        },
    )
    collection, observations = _roll_bank(
        config, training_bank, split="train", retain_observations=True
    )
    _write_new_json(output / "normalization_collection.json", collection)
    if collection["solver_failure_count"]:
        raise ValueError("Engineering normalization has a solver failure")
    normalizer = TrainingNormalizer.fit(
        np.stack(observations), split="train", manifest_sha256=manifest["sha256"]
    )
    normalizer.save(output / "normalizer.json")
    config["normalization_path"] = str(output / "normalizer.json")
    bank = build_mapdn_bank(
        config,
        validation_bank,
        split="validation",
        purpose="engineering_only_paired_replay",
        source_root=root,
    )
    _write_new_json(output / "bank_contract.json", bank)
    rows = []
    for index, method in enumerate(("identity", "local_capacity", "broadcast")):
        evidence = output / method
        evidence.mkdir()
        spec = phase_spec(config, 980300 + index, method, 64, checkpoint_interval=64)
        spec = replace(
            spec, experiment={**spec.experiment, "on_policy_collected_frames_per_batch": 64}
        )
        context = RunContext(
            suite_id="main_direction_engineering",
            run_id=method,
            output_root=output / "runs",
            command=tuple(sys.argv),
        )
        callback = PhaseEvidence(
            config, evidence, _sha(output / "precollection.json"), bank, (0, 64)
        )
        experiment = run_managed_experiment(
            spec, context, repo_root=root, callbacks=[callback], validation_run=True
        )
        try:
            trained_sha = _state_sha(_actor_state(experiment))
            result = evaluate_mapdn_paired(
                experiment, bank, require_exact_local_null=method != "broadcast"
            )
            counts = parameter_counts(experiment)
            _write_new_json(evidence / "paired_trained.json", result)
        finally:
            experiment.close()
        reloaded = load_frozen_experiment(context.run_dir, scratch_root=evidence / "reload_scratch")
        try:
            paths = list(context.run_dir.glob("benchmarl/**/checkpoints/checkpoint_64.pt"))
            if len(paths) != 1:
                raise ValueError("Engineering native checkpoint missing")
            native = verify_native_checkpoint(reloaded, paths[0], expected_frames=64)
            replay = evaluate_mapdn_paired(
                reloaded, bank, require_exact_local_null=method != "broadcast"
            )
            _write_new_json(evidence / "paired_reloaded.json", replay)
            checks = {
                "exact_reloaded_actor": _state_sha(_actor_state(reloaded)) == trained_sha,
                "exact_paired_replay": result == replay,
                "exact_collection": callback.collected_frames == 64,
                "initial_and_final_measurements": [p["frames"] for p in callback.points] == [0, 64],
                "actor_updated": trained_sha != callback.initial_manifest["actor_state_sha256"],
                "finite_training": finiteness_report(context.run_dir)["all_finite"],
                "full_episodes_without_solver_failure": all(
                    value["full_horizon_episodes"] == 2 and value["solver_failure_count"] == 0
                    for value in result["summaries"].values()
                ),
            }
            row = {
                "method": method,
                "checks": checks,
                "passed": all(checks.values()),
                "parameter_counts": counts,
                "native_reload": native,
                "paired_result_sha256": _sha(evidence / "paired_trained.json"),
            }
            _write_new_json(evidence / "report.json", row)
            rows.append(row)
            if not row["passed"]:
                raise ValueError(f"Engineering checks failed for {method}: {checks}")
        finally:
            reloaded.close()
    if before != source_inventory() or runtime != _runtime():
        raise ValueError("Source/runtime changed during engineering checks")
    report = {
        "purpose": "engineering_only",
        "scientific_learning_confirmed": False,
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "source_unchanged": True,
        "source_inventory": before,
        "runtime": runtime,
    }
    _write_new_json(output / "report.json", report)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    args = parser.parse_args()
    return run(
        Path(__file__).resolve().parents[1], args.out_dir.resolve(), args.data_path.resolve()
    )


if __name__ == "__main__":
    raise SystemExit(main())
