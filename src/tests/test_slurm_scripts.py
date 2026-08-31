"""The submitted array ranges must match what the suites actually expand to.

A Slurm array range is written by hand in the sbatch header, while the row
count comes from the suite YAMLs. If someone adds a seed or an ablation value,
a stale range silently under-runs the manifest: the extra rows are simply never
executed, and nothing reports an error -- the suite just quietly finishes
incomplete. These tests make that drift a test failure instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from commstudy.experiments.sweeps import create_combined_manifest, expand_suite_config


REPO_ROOT = Path(__file__).resolve().parents[2]
SLURM = REPO_ROOT / "slurm"

MAIN_SUITE = "simple_spread_comm_v2.yaml"
ABLATION_SUITES = (
    "simple_spread_comm_v2_stage2_message_dim.yaml",
    "simple_spread_comm_v2_stage2_dropout.yaml",
    "simple_spread_comm_v2_stage3_rounds.yaml",
    "simple_spread_comm_v2_stage3_heads.yaml",
    "simple_spread_comm_v2_stage4_graph_topology.yaml",
    "simple_spread_comm_v2_stage4_sender_budget.yaml",
    "simple_spread_comm_v2_stage4_self_communication.yaml",
)

SCRIPTS = (
    "01_setup.sbatch",
    "02_main_comparison.sbatch",
    "03_ablations.sbatch",
    "04_analyze.sbatch",
)


def _plans(filename, config_root, tmp_path):
    document = OmegaConf.to_container(
        OmegaConf.load(config_root / "sweeps" / filename), resolve=True
    )
    return expand_suite_config(document, repo_root=tmp_path)


def _array_upper_bound(script: str) -> int:
    match = re.search(r"#SBATCH --array=0-(\d+)", script)
    assert match, "script has no #SBATCH --array=0-N directive"
    return int(match.group(1))


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_documented_script_exists(name):
    assert (SLURM / name).is_file()


def test_main_array_range_covers_every_planned_row(config_root, tmp_path):
    plans = _plans(MAIN_SUITE, config_root, tmp_path)
    script = (SLURM / "02_main_comparison.sbatch").read_text(encoding="utf-8")

    # Array indices are inclusive, so N rows means 0-(N-1).
    assert _array_upper_bound(script) == len(plans) - 1


def test_ablation_array_range_covers_every_planned_row(config_root, tmp_path):
    total = sum(len(_plans(name, config_root, tmp_path)) for name in ABLATION_SUITES)
    script = (SLURM / "03_ablations.sbatch").read_text(encoding="utf-8")

    assert _array_upper_bound(script) == total - 1


def test_setup_builds_exactly_the_suites_the_arrays_consume():
    setup = (SLURM / "01_setup.sbatch").read_text(encoding="utf-8")

    for name in (MAIN_SUITE, *ABLATION_SUITES):
        assert name in setup, f"01_setup.sbatch never builds a manifest for {name}"


def test_combined_manifest_matches_the_ablation_array(config_root, tmp_path):
    destination = tmp_path / "combined.csv"

    _, plans = create_combined_manifest(
        [config_root / "sweeps" / name for name in ABLATION_SUITES],
        destination,
        repo_root=tmp_path,
    )
    script = (SLURM / "03_ablations.sbatch").read_text(encoding="utf-8")

    assert destination.exists()
    assert _array_upper_bound(script) == len(plans) - 1
    assert len({plan.run_id for plan in plans}) == len(plans)
    # Rows keep their own suite identity, so per-suite analysis still works.
    assert len({plan.suite_id for plan in plans}) == len(ABLATION_SUITES)


def test_combined_manifest_rejects_duplicate_rows(config_root, tmp_path):
    duplicated = [config_root / "sweeps" / MAIN_SUITE] * 2

    with pytest.raises(ValueError, match="Duplicate run_id"):
        create_combined_manifest(duplicated, tmp_path / "dup.csv", repo_root=tmp_path)


@pytest.mark.parametrize("name", ("02_main_comparison.sbatch", "03_ablations.sbatch"))
def test_training_scripts_reclaim_stale_rows_and_pin_the_device(name):
    script = (SLURM / name).read_text(encoding="utf-8")

    # Without reclamation, a preempted row is never retried.
    assert "--reclaim-stale" in script
    # All three device settings must move together or collection and training
    # end up on different devices.
    for key in ("sampling_device", "train_device", "buffer_device"):
        assert f"experiment.{key}=" in script


@pytest.mark.parametrize("name", ("02_main_comparison.sbatch", "03_ablations.sbatch"))
def test_training_scripts_fail_loudly_without_a_manifest(name):
    script = (SLURM / name).read_text(encoding="utf-8")

    assert "01_setup.sbatch" in script
    assert "exit 1" in script


def test_manifest_paths_agree_between_setup_and_array_scripts():
    setup = (SLURM / "01_setup.sbatch").read_text(encoding="utf-8")
    ablations = (SLURM / "03_ablations.sbatch").read_text(encoding="utf-8")

    assert "runs/_manifests/ablations.csv" in setup
    assert "runs/_manifests/ablations.csv" in ablations


def test_scripts_are_sourced_not_executed():
    """The environment file is sourced, so it needs no execute permission."""

    for name in ("02_main_comparison.sbatch", "03_ablations.sbatch", "04_analyze.sbatch"):
        script = (SLURM / name).read_text(encoding="utf-8")
        assert "source slurm/newton_env.sh" in script
