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
    "00_probe.sbatch",
    "01_setup.sbatch",
    "02_main_comparison.sbatch",
    "03_ablations.sbatch",
    "04_analyze.sbatch",
    "pcp_01_setup.sbatch",
    "pcp_02_pilot.sbatch",
    "pcp_02b_protocol.sbatch",
    "pcp_03_main_comparison.sbatch",
    "pcp_04_ablations.sbatch",
    "pcp_05_analyze.sbatch",
)

PCP_PILOT_SUITE = "pcp_identity_pilot.yaml"
PCP_PROTOCOL_SUITE = "pcp_protocol_gate.yaml"
PCP_MAIN_SUITE = "pcp_comm_main.yaml"
PCP_ABLATION_SUITES = (
    "pcp_comm_stage2_message_dim.yaml",
    "pcp_comm_stage2_dropout.yaml",
    "pcp_comm_stage3_rounds.yaml",
    "pcp_comm_stage3_heads.yaml",
    "pcp_comm_stage4_graph_topology.yaml",
    "pcp_comm_stage4_sender_budget.yaml",
    "pcp_comm_stage4_self_communication.yaml",
)

PCP_TRAINING_SCRIPTS = (
    "pcp_02_pilot.sbatch",
    "pcp_02b_protocol.sbatch",
    "pcp_03_main_comparison.sbatch",
    "pcp_04_ablations.sbatch",
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


def test_setup_never_relies_on_a_bare_python_before_activation():
    """Regression: Spack modules provide `python3`, not always `python`.

    Calling `python` before the venv is activated silently falls through to
    /usr/bin/python, which on this cluster has no `venv` module. The setup
    script must resolve an interpreter explicitly and verify it.
    """

    setup = (SLURM / "01_setup.sbatch").read_text(encoding="utf-8")

    # It must create the venv through a resolved, verified interpreter.
    assert '"$PYTHON_BIN" -m venv' in setup
    assert "python -m venv" not in setup

    # Resolution must reject system interpreters and prove venv actually works.
    assert "/usr/bin/*|/bin/*" in setup
    assert "-m venv --help" in setup
    assert "version_info[:2] >= (3,11)" in setup


def test_setup_tries_multiple_module_names_and_points_at_the_probe():
    setup = (SLURM / "01_setup.sbatch").read_text(encoding="utf-8")

    assert "PYTHON_MODULE_CANDIDATES" in setup
    assert "anaconda" in setup, "anaconda is the documented fallback on Newton"
    assert "COMMSTUDY_PYTHON_MODULE" in setup
    assert "00_probe.sbatch" in setup, "failure path must name the diagnostic job"


def test_setup_records_the_resolved_environment_for_training_jobs():
    """Training jobs must reuse the interpreter setup verified, not re-guess."""

    setup = (SLURM / "01_setup.sbatch").read_text(encoding="utf-8")
    env = (SLURM / "newton_env.sh").read_text(encoding="utf-8")

    assert "slurm/.resolved_env" in setup
    assert ".resolved_env" in env


def test_env_error_message_points_at_a_script_that_exists():
    """A guard is only useful if the command it recommends is real."""

    env = (SLURM / "newton_env.sh").read_text(encoding="utf-8")

    assert "01_setup.sbatch" in env
    for name in ("bootstrap_newton.sh", "submit.sh", "run_suite.sbatch"):
        assert name not in env, f"stale reference to deleted {name}"


def test_env_guard_verifies_the_venv_works_not_just_that_it_exists():
    """Regression: a half-built venv passed the old existence-only check.

    When setup dies partway (typically pip failing on a network-isolated
    compute node) the venv directory still exists. Every array task then sailed
    past the guard and died separately inside an import traceback. The guard
    must import the actual stack and fail once, with the remedy.
    """

    env = (SLURM / "newton_env.sh").read_text(encoding="utf-8")

    assert "import torch, benchmarl, torchrl, tensordict, vmas" in env
    assert "incomplete" in env
    # The error must name the login-node remedy, not just report failure.
    assert "LOGIN node" in env
    assert "pip install" in env


def test_setup_checks_network_before_building_anything():
    """Fail fast on a network-isolated node instead of leaving a broken venv."""

    setup = (SLURM / "01_setup.sbatch").read_text(encoding="utf-8")

    assert "pypi.org" in setup
    # The check must precede venv creation, or it cannot prevent the half-built
    # state it exists to avoid.
    assert setup.index("pypi.org") < setup.index("creating virtualenv")
    assert "cannot reach pypi.org" in setup


def test_setup_is_idempotent_when_the_environment_is_already_complete():
    """Resubmitting after a login-node install must skip straight to manifests."""

    setup = (SLURM / "01_setup.sbatch").read_text(encoding="utf-8")

    assert "NEED_INSTALL" in setup
    assert "already complete" in setup
    assert 'if [[ "$NEED_INSTALL" -eq 1 ]]; then' in setup


def test_probe_reports_venv_completeness_and_network():
    probe = (SLURM / "00_probe.sbatch").read_text(encoding="utf-8")

    assert "pypi.org" in probe
    assert "MISSING" in probe
    for package in ("torch", "benchmarl", "torchrl", "tensordict", "vmas"):
        assert package in probe


def test_probe_diagnoses_shared_object_mapping_failures():
    """`failed to map segment` has two causes; the probe must separate them.

    Either the login node caps address space (install is fine), or the library
    was truncated by a quota limit (install is broken). Reporting ulimit, quota,
    and the actual library size distinguishes them.
    """

    probe = (SLURM / "00_probe.sbatch").read_text(encoding="utf-8")

    assert "ulimit -a" in probe
    assert "quota" in probe
    assert "libtorch_cuda.so" in probe


def test_probe_reports_gpu_selection_strings():
    """Pinning V100 vs H100 requires the cluster's real GRES/feature strings."""

    probe = (SLURM / "00_probe.sbatch").read_text(encoding="utf-8")

    assert "--gres=gpu:h100:1" in probe
    assert "--constraint=h100" in probe
    assert "scontrol show nodes" in probe


def test_probe_makes_no_changes():
    """The diagnostic must be safe to run at any time, including mid-study."""

    probe = (SLURM / "00_probe.sbatch").read_text(encoding="utf-8")

    for destructive in ("pip install", "scripts/sweep.py", "rm -", "mkdir "):
        assert destructive not in probe, f"probe must not run: {destructive}"
    # `venv --help` is a capability check; actually building one is not.
    assert "-m venv --help" in probe
    assert '-m venv "' not in probe


def test_scripts_are_sourced_not_executed():
    """The environment file is sourced, so it needs no execute permission."""

    for name in ("02_main_comparison.sbatch", "03_ablations.sbatch", "04_analyze.sbatch"):
        script = (SLURM / name).read_text(encoding="utf-8")
        assert "source slurm/newton_env.sh" in script


def test_pcp_pilot_array_range_covers_every_planned_row(config_root, tmp_path):
    plans = _plans(PCP_PILOT_SUITE, config_root, tmp_path)
    script = (SLURM / "pcp_02_pilot.sbatch").read_text(encoding="utf-8")

    assert _array_upper_bound(script) == len(plans) - 1


def test_pcp_main_array_range_covers_every_planned_row(config_root, tmp_path):
    plans = _plans(PCP_MAIN_SUITE, config_root, tmp_path)
    script = (SLURM / "pcp_03_main_comparison.sbatch").read_text(encoding="utf-8")

    assert _array_upper_bound(script) == len(plans) - 1


def test_pcp_ablation_array_range_covers_every_planned_row(config_root, tmp_path):
    total = sum(
        len(_plans(name, config_root, tmp_path)) for name in PCP_ABLATION_SUITES
    )
    script = (SLURM / "pcp_04_ablations.sbatch").read_text(encoding="utf-8")

    assert _array_upper_bound(script) == total - 1


def test_pcp_setup_builds_exactly_the_suites_the_arrays_consume():
    setup = (SLURM / "pcp_01_setup.sbatch").read_text(encoding="utf-8")

    for name in (PCP_PILOT_SUITE, PCP_MAIN_SUITE, *PCP_ABLATION_SUITES):
        assert name in setup, f"pcp_01_setup.sbatch never builds a manifest for {name}"


def test_pcp_combined_manifest_matches_the_ablation_array(config_root, tmp_path):
    destination = tmp_path / "pcp_combined.csv"

    _, plans = create_combined_manifest(
        [config_root / "sweeps" / name for name in PCP_ABLATION_SUITES],
        destination,
        repo_root=tmp_path,
    )
    script = (SLURM / "pcp_04_ablations.sbatch").read_text(encoding="utf-8")

    assert destination.exists()
    assert _array_upper_bound(script) == len(plans) - 1
    assert len({plan.run_id for plan in plans}) == len(plans)
    assert len({plan.suite_id for plan in plans}) == len(PCP_ABLATION_SUITES)


def test_pcp_manifest_paths_agree_and_do_not_collide_with_simple_spread():
    """A shared combined-manifest path would let one study overwrite the other."""

    setup = (SLURM / "pcp_01_setup.sbatch").read_text(encoding="utf-8")
    ablations = (SLURM / "pcp_04_ablations.sbatch").read_text(encoding="utf-8")
    simple_spread = (SLURM / "03_ablations.sbatch").read_text(encoding="utf-8")

    assert "runs/_manifests/pcp_ablations.csv" in setup
    assert "runs/_manifests/pcp_ablations.csv" in ablations
    assert "runs/_manifests/pcp_ablations.csv" not in simple_spread


@pytest.mark.parametrize("name", PCP_TRAINING_SCRIPTS)
def test_pcp_training_scripts_reclaim_stale_rows_and_pin_the_device(name):
    script = (SLURM / name).read_text(encoding="utf-8")

    assert "--reclaim-stale" in script
    for key in ("sampling_device", "train_device", "buffer_device"):
        assert f"experiment.{key}=" in script


@pytest.mark.parametrize("name", PCP_TRAINING_SCRIPTS)
def test_pcp_training_scripts_fail_loudly_without_a_manifest(name):
    script = (SLURM / name).read_text(encoding="utf-8")

    assert "pcp_01_setup.sbatch" in script
    assert "exit 1" in script


def test_pcp_scripts_reuse_the_shared_environment_and_do_not_rebuild_it():
    """The venv is built in one place. Two copies of that logic would drift."""

    for name in (*PCP_TRAINING_SCRIPTS, "pcp_01_setup.sbatch", "pcp_05_analyze.sbatch"):
        script = (SLURM / name).read_text(encoding="utf-8")
        assert "source slurm/newton_env.sh" in script
        assert "-m venv" not in script
        assert "pip install" not in script


def test_pcp_analysis_covers_every_suite_the_arrays_produce():
    analyze = (SLURM / "pcp_05_analyze.sbatch").read_text(encoding="utf-8")

    for name in (PCP_PILOT_SUITE, PCP_MAIN_SUITE, *PCP_ABLATION_SUITES):
        assert name.removesuffix(".yaml") in analyze


def test_pcp_launch_scripts_state_the_uncalibrated_budget():
    """The horizon is a first pass, not a gate result. Saying so is the guard.

    Nothing in the tooling stops someone submitting 231 rows at a budget no
    diagnostic has justified, which is precisely how V1 was lost on Simple
    Spread, so the scripts have to carry the warning themselves.
    """

    main = (SLURM / "pcp_03_main_comparison.sbatch").read_text(encoding="utf-8")
    pilot = (SLURM / "pcp_02_pilot.sbatch").read_text(encoding="utf-8")

    assert "60,000 frames" in main
    assert "PILOT" in main
    assert "gate" in pilot


def test_protocol_gate_array_range_covers_every_planned_row(config_root, tmp_path):
    plans = _plans(PCP_PROTOCOL_SUITE, config_root, tmp_path)
    script = (SLURM / "pcp_02b_protocol.sbatch").read_text(encoding="utf-8")

    assert len(plans) == 27
    assert _array_upper_bound(script) == len(plans) - 1
    assert "runs/pcp_protocol_gate/manifest.csv" in script


def test_setup_builds_the_protocol_gate_manifest():
    """The gate cannot run from a manifest nobody writes."""
    setup = (SLURM / "pcp_01_setup.sbatch").read_text(encoding="utf-8")
    assert "configs/sweeps/pcp_protocol_gate.yaml" in setup
    assert "pcp_02b_protocol.sbatch" in setup


def test_pcp_comparison_suites_carry_enough_evaluation_episodes(config_root):
    """Five episodes put the standard error above the effect being measured.

    Measured over 300 random episodes: mean 0.833, std 3.789, 94% of episodes
    scoring exactly zero, so SEM at n=5 is 1.69 against effects of 1-3 return
    points. The evaluation env is batched, so raising this widens the batch
    instead of adding rollouts. See docs/RESULTS_pcp_pilot.md.
    """
    suites = (PCP_MAIN_SUITE, PCP_PROTOCOL_SUITE, *PCP_ABLATION_SUITES)
    for name in suites:
        document = OmegaConf.to_container(
            OmegaConf.load(config_root / "sweeps" / name), resolve=True
        )
        episodes = document["overrides"]["experiment.evaluation_episodes"]
        assert episodes >= 128, f"{name} evaluates on only {episodes} episodes"


def test_blocked_pcp_suites_say_so_where_someone_would_look(config_root):
    """The placeholder budget/gamma/entropy must not read as settled.

    V1 on Simple Spread is preserved in this repo as evidence of what launching
    a full grid under an unvalidated protocol costs; these files are the last
    thing between that and 231 PCP rows.
    """
    for name in (PCP_MAIN_SUITE, *PCP_ABLATION_SUITES):
        text = (config_root / "sweeps" / name).read_text(encoding="utf-8")
        assert "BLOCKED" in text, name
        assert "pcp_protocol_gate.yaml" in text, name
