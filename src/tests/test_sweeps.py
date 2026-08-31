from __future__ import annotations

import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from commstudy.experiments import load_experiment_spec
from commstudy.experiments.sweeps import (
    expand_suite_config,
    read_manifest,
    select_plans,
    write_manifest,
)
from commstudy.utils.imports import import_from_path


HISTORICAL_V1_ABLATION_SUITES = {
    "simple_spread_comm_v1_stage2_message_dim.yaml": 48,
    "simple_spread_comm_v1_stage2_dropout.yaml": 48,
    "simple_spread_comm_v1_stage3_rounds.yaml": 18,
    "simple_spread_comm_v1_stage3_heads.yaml": 24,
    "simple_spread_comm_v1_stage4_graph_topology.yaml": 9,
    "simple_spread_comm_v1_stage4_sender_budget.yaml": 36,
    "simple_spread_comm_v1_stage4_self_communication.yaml": 18,
}

HISTORICAL_V1_SUITES = {
    "simple_spread_comm_v1.yaml": 30,
    **HISTORICAL_V1_ABLATION_SUITES,
}

V2_ABLATION_SUITES = {
    filename.replace("_v1_", "_v2_"): count
    for filename, count in HISTORICAL_V1_ABLATION_SUITES.items()
}

V2_SUITES = {
    "simple_spread_comm_v2.yaml": 30,
    **V2_ABLATION_SUITES,
}

# Instrumentation rather than a scientific comparison: excluded from the
# protocol/design-parity checks that apply to the V2 study suites.
V2_INSTRUMENT_SUITES = {
    "simple_spread_comm_v2_wallclock.yaml": 6,
}

V1_TO_V2_SUITES = {
    filename: filename.replace("_v1", "_v2", 1)
    for filename in HISTORICAL_V1_SUITES
}


def _load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def test_suite_expansion_pairs_models_and_seeds(tmp_path):
    plans = expand_suite_config(
        {
            "suite_id": "simple_spread_comm_v1",
            "output_root": str(tmp_path / "runs"),
            "algorithm": "mappo",
            "task": "vmas_simple_spread",
            "models": ["comm_identity", "comm_attention"],
            "seeds": [0, 1, 2],
            "max_n_frames": 600_000,
            "overrides": {"experiment.evaluation_interval": 12_000},
        },
        repo_root=tmp_path,
    )
    assert len(plans) == 6
    assert len({plan.run_id for plan in plans}) == 6
    assert {plan.seed for plan in plans if plan.model == "comm_identity"} == {0, 1, 2}
    assert all("experiment.max_n_frames=600000" in plan.overrides for plan in plans)


def test_optional_run_namespace_is_in_run_id_but_not_ablation_metadata(tmp_path):
    plans = expand_suite_config(
        {
            "suite_id": "simple_spread_comm_v2",
            "run_namespace": "v2",
            "models": ["comm_identity"],
            "seeds": [0],
            "ablation": "main",
        },
        repo_root=tmp_path,
    )

    assert plans[0].run_id == "simple_spread__mappo__identity__v2__seed000"
    assert plans[0].ablation == "main"
    assert all("run_namespace" not in override for override in plans[0].overrides)


def test_identity_stability_suite_has_exact_common_protocol_and_variants(
    config_root,
    tmp_path,
):
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/simple_spread_identity_stability.yaml"),
        repo_root=tmp_path,
    )

    expected_variants = {
        "gamma0.9_entropy0": (0.9, 0.0),
        "gamma0.99_entropy0.01": (0.99, 0.01),
        "gamma0.9_entropy0.01": (0.9, 0.01),
    }
    assert len(plans) == 3
    assert len({plan.run_id for plan in plans}) == 3
    assert {plan.model for plan in plans} == {"comm_identity"}
    assert {plan.seed for plan in plans} == {0}
    assert {plan.max_n_frames for plan in plans} == {240_000}
    assert {plan.ablation for plan in plans} == {"optimizer_stability"}
    assert {plan.ablation_value for plan in plans} == set(expected_variants)

    for plan in plans:
        spec = load_experiment_spec(config_root, plan.overrides)
        gamma, entropy_coef = expected_variants[plan.ablation_value]
        assert spec.experiment["gamma"] == gamma
        assert spec.algorithm_config["params"]["entropy_coef"] == entropy_coef
        assert spec.experiment["max_n_frames"] == 240_000
        assert spec.experiment["on_policy_n_minibatch_iters"] == 10
        assert spec.experiment["evaluation_interval"] == 12_000
        assert spec.experiment["checkpoint_at_end"] is True
        assert spec.experiment["keep_checkpoints_num"] == 2


def test_identity_stable_horizon_suite_retains_exact_historical_protocol(
    config_root,
    tmp_path,
):
    plans = expand_suite_config(
        _load_yaml(
            config_root / "sweeps/simple_spread_identity_stable_horizon.yaml"
        ),
        repo_root=tmp_path,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.run_id == (
        "simple_spread__mappo__identity__stable_horizon__600000__seed000"
    )
    assert plan.suite_id == "simple_spread_identity_stable_horizon"
    assert plan.task == "vmas_simple_spread"
    assert plan.algorithm == "mappo"
    assert plan.model == "comm_identity"
    assert plan.seed == 0
    assert plan.ablation == "stable_horizon"
    assert plan.ablation_value == "600000"
    assert plan.max_n_frames == 600_000

    spec = load_experiment_spec(config_root, plan.overrides)
    assert spec.experiment["gamma"] == 0.9
    assert spec.algorithm_config["params"]["entropy_coef"] == 0.01
    assert spec.experiment["max_n_frames"] == 600_000
    assert spec.experiment["on_policy_n_minibatch_iters"] == 10
    assert spec.experiment["evaluation_interval"] == 12_000
    assert spec.experiment["checkpoint_at_end"] is True
    assert spec.experiment["keep_checkpoints_num"] == 2


def test_mappo_multiseed_stability_suite_has_exact_variants_and_preserves_controls(
    config_root,
    tmp_path,
):
    plans = expand_suite_config(
        _load_yaml(
            config_root / "sweeps/simple_spread_mappo_multiseed_stability.yaml"
        ),
        repo_root=tmp_path,
    )

    expected_variants = {
        "entropy0.1_iters10": (0.1, 10),
        "entropy0.1_iters5": (0.1, 5),
        "entropy0.01_iters5": (0.01, 5),
    }
    expected_run_ids = {
        (
            "simple_spread__mappo__benchmarl_mlp__multiseed_stability__"
            f"{variant}__seed{seed:03d}"
        )
        for variant in expected_variants
        for seed in (2, 3, 4)
    }

    assert len(plans) == 9
    assert {plan.run_id for plan in plans} == expected_run_ids
    assert {plan.model for plan in plans} == {"benchmarl_mlp"}
    assert {plan.seed for plan in plans} == {2, 3, 4}
    assert {plan.max_n_frames for plan in plans} == {360_000}
    assert {plan.ablation for plan in plans} == {"multiseed_stability"}
    assert {plan.ablation_value for plan in plans} == set(expected_variants)

    for plan in plans:
        spec = load_experiment_spec(config_root, plan.overrides)
        entropy_coef, minibatch_iters = expected_variants[plan.ablation_value]
        assert spec.task == "vmas_simple_spread"
        assert spec.algorithm == "mappo"
        assert spec.model == "benchmarl_mlp"
        assert spec.seed == plan.seed
        assert spec.experiment["gamma"] == 0.9
        assert spec.algorithm_config["params"]["entropy_coef"] == entropy_coef
        assert spec.experiment["on_policy_n_minibatch_iters"] == minibatch_iters
        assert spec.experiment["max_n_frames"] == 360_000
        assert spec.experiment["evaluation_interval"] == 12_000
        assert spec.experiment["checkpoint_interval"] == 120_000
        assert spec.experiment["checkpoint_at_end"] is True
        assert spec.experiment["keep_checkpoints_num"] == 2

    # The existing main rows are distinct, longer controls under the
    # provisional entropy=0.01 / 10-iteration protocol. Together they form a
    # clean 2x2 entropy-by-minibatch-iteration design; the diagnostic neither
    # duplicates nor mutates the controls.
    main_plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/simple_spread_comm_v1.yaml"),
        repo_root=tmp_path,
    )
    controls = [
        plan
        for plan in main_plans
        if plan.model == "benchmarl_mlp" and plan.seed in {2, 3, 4}
    ]
    assert {plan.run_id for plan in controls} == {
        f"simple_spread__mappo__benchmarl_mlp__seed{seed:03d}"
        for seed in (2, 3, 4)
    }
    assert expected_run_ids.isdisjoint({plan.run_id for plan in controls})
    for control in controls:
        spec = load_experiment_spec(config_root, control.overrides)
        assert spec.experiment["gamma"] == 0.9
        assert spec.algorithm_config["params"]["entropy_coef"] == 0.01
        assert spec.experiment["on_policy_n_minibatch_iters"] == 10
        assert spec.experiment["max_n_frames"] == 600_000


@pytest.mark.parametrize(
    ("filename", "expected_count"), HISTORICAL_V1_SUITES.items()
)
def test_historical_comm_v1_suites_retain_exact_provisional_protocol(
    filename,
    expected_count,
    config_root,
    tmp_path,
):
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps" / filename),
        repo_root=tmp_path,
    )

    assert len(plans) == expected_count
    assert len({plan.run_id for plan in plans}) == expected_count
    for plan in plans:
        assert plan.task == "vmas_simple_spread"
        assert plan.algorithm == "mappo"
        assert plan.max_n_frames == 600_000

        spec = load_experiment_spec(config_root, plan.overrides)
        assert spec.task == "vmas_simple_spread"
        assert spec.algorithm == "mappo"
        assert spec.model == plan.model
        assert spec.seed == plan.seed
        assert spec.experiment["gamma"] == 0.9
        assert spec.algorithm_config["params"]["entropy_coef"] == 0.01
        assert spec.experiment["on_policy_n_minibatch_iters"] == 10
        assert spec.experiment["max_n_frames"] == 600_000
        assert spec.experiment["evaluation_interval"] == 12_000
        assert spec.experiment["checkpoint_interval"] == 120_000
        assert spec.experiment["checkpoint_at_end"] is True
        assert spec.experiment["keep_checkpoints_num"] == 2


@pytest.mark.parametrize(("filename", "expected_count"), V2_SUITES.items())
def test_comm_v2_suites_have_exact_selected_pending_confirmation_protocol(
    filename,
    expected_count,
    config_root,
    tmp_path,
):
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps" / filename),
        repo_root=tmp_path,
    )

    assert len(plans) == expected_count
    assert len({plan.run_id for plan in plans}) == expected_count
    assert all("__v2__" in plan.run_id for plan in plans)
    for plan in plans:
        assert plan.suite_id == filename.removesuffix(".yaml")
        assert plan.task == "vmas_simple_spread"
        assert plan.algorithm == "mappo"
        assert plan.max_n_frames == 600_000

        spec = load_experiment_spec(config_root, plan.overrides)
        assert spec.task == "vmas_simple_spread"
        assert spec.algorithm == "mappo"
        assert spec.model == plan.model
        assert spec.seed == plan.seed
        assert spec.experiment["gamma"] == 0.9
        assert spec.algorithm_config["params"]["entropy_coef"] == 0.1
        assert spec.experiment["on_policy_n_minibatch_iters"] == 5
        assert spec.experiment["max_n_frames"] == 600_000
        assert spec.experiment["evaluation_interval"] == 12_000
        assert spec.experiment["checkpoint_interval"] == 120_000
        assert spec.experiment["checkpoint_at_end"] is True
        assert spec.experiment["keep_checkpoints_num"] == 2


@pytest.mark.parametrize(("v1_filename", "v2_filename"), V1_TO_V2_SUITES.items())
def test_comm_v2_preserves_v1_scientific_design_except_selected_protocol(
    v1_filename,
    v2_filename,
    config_root,
    tmp_path,
):
    v1_plans = expand_suite_config(
        _load_yaml(config_root / "sweeps" / v1_filename),
        repo_root=tmp_path,
    )
    v2_plans = expand_suite_config(
        _load_yaml(config_root / "sweeps" / v2_filename),
        repo_root=tmp_path,
    )

    def scientific_design(plan):
        selected_protocol_keys = (
            "algorithm_config.params.entropy_coef=",
            "experiment.on_policy_n_minibatch_iters=",
        )
        return (
            plan.task,
            plan.algorithm,
            plan.model,
            plan.seed,
            plan.ablation,
            plan.ablation_value,
            plan.max_n_frames,
            tuple(
                override
                for override in plan.overrides
                if not override.startswith(selected_protocol_keys)
            ),
        )

    assert [scientific_design(plan) for plan in v2_plans] == [
        scientific_design(plan) for plan in v1_plans
    ]


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_comm_main_suite_has_exact_models_and_seeds(version, config_root, tmp_path):
    plans = expand_suite_config(
        _load_yaml(config_root / f"sweeps/simple_spread_comm_{version}.yaml"),
        repo_root=tmp_path,
    )

    assert {plan.model for plan in plans} == {
        "benchmarl_mlp",
        "comm_identity",
        "comm_broadcast",
        "comm_gated",
        "comm_attention",
        "comm_graph",
    }
    assert {plan.seed for plan in plans} == {0, 1, 2, 3, 4}
    assert {plan.ablation for plan in plans} == {"main"}


def test_manifest_roundtrip_and_selection(tmp_path):
    plans = expand_suite_config(
        {
            "suite_id": "suite",
            "models": ["comm_identity"],
            "seeds": [0, 1],
        },
        repo_root=tmp_path,
    )
    path = tmp_path / "manifest.csv"
    write_manifest(path, plans)
    loaded = read_manifest(path)
    assert loaded == plans
    assert select_plans(loaded, index=1) == [loaded[1]]
    assert select_plans(loaded, run_id=loaded[0].run_id) == [loaded[0]]


def test_train_dry_run_does_not_create_output(tmp_path, config_root):
    repo_root = config_root.parent
    output_root = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            "--dry-run",
            "--output-root",
            str(output_root),
            "model=comm_identity",
            "experiment.max_n_frames=6000",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"run_id": "simple_spread__mappo__identity__seed000"' in result.stdout
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("filename", "expected_count"), HISTORICAL_V1_ABLATION_SUITES.items()
)
def test_staged_ablation_suite_expansion(
    filename, expected_count, config_root, tmp_path
):
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps" / filename),
        repo_root=tmp_path,
    )

    assert len(plans) == expected_count
    assert len({plan.run_id for plan in plans}) == expected_count
    assert {plan.seed for plan in plans} == {0, 1, 2}
    assert {plan.max_n_frames for plan in plans} == {600_000}
    if "dropout" not in filename:
        assert all("dropout" not in " ".join(plan.overrides) for plan in plans)

    # Resolve the real component config and construct every communication
    # module. This catches stale/misspelled dot-override keys before launch.
    for plan in plans:
        spec = load_experiment_spec(config_root, plan.overrides)
        params = spec.model_config["params"]
        comm_class = import_from_path(params["comm_class_path"])
        comm_class(hidden_dim=params["hidden_dim"], **params["comm_kwargs"])


def test_all_v1_and_v2_run_ids_are_globally_unique(config_root, tmp_path):
    all_suites = {**HISTORICAL_V1_SUITES, **V2_SUITES, **V2_INSTRUMENT_SUITES}
    run_ids = []
    suite_ids = []
    for filename in all_suites:
        plans = expand_suite_config(
            _load_yaml(config_root / "sweeps" / filename),
            repo_root=tmp_path,
        )
        run_ids.extend(plan.run_id for plan in plans)
        suite_ids.extend({plan.suite_id for plan in plans})
    assert len(run_ids) == 468
    assert len(run_ids) == len(set(run_ids))
    assert len(suite_ids) == len(set(suite_ids)) == 17


def test_wallclock_suite_is_a_short_sequential_timing_instrument(config_root, tmp_path):
    """The timing suite must share the V2 optimizer protocol but stay unmixable.

    It exists only to measure per-method compute overhead without collection
    contention, so it must keep a distinct ablation label, a short horizon, and
    run IDs disjoint from the scientific comparison.
    """

    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/simple_spread_comm_v2_wallclock.yaml"),
        repo_root=tmp_path,
    )
    main_plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/simple_spread_comm_v2.yaml"),
        repo_root=tmp_path,
    )

    assert len(plans) == 6
    assert {plan.model for plan in plans} == {plan.model for plan in main_plans}
    assert {plan.seed for plan in plans} == {0}
    assert {plan.ablation for plan in plans} == {"wallclock_benchmark"}
    assert {plan.ablation_value for plan in plans} == {"60000_frames_sequential"}
    assert not {plan.run_id for plan in plans} & {plan.run_id for plan in main_plans}

    for plan in plans:
        assert plan.max_n_frames == 60_000
        spec = load_experiment_spec(config_root, plan.overrides)
        # Identical optimizer protocol, so the timing reflects the model alone.
        assert spec.experiment["gamma"] == 0.9
        assert spec.algorithm_config["params"]["entropy_coef"] == 0.1
        assert spec.experiment["on_policy_n_minibatch_iters"] == 5
        assert spec.experiment["max_n_frames"] == 60_000
        # Checkpointing is disabled: it is I/O time, not communication compute.
        assert spec.experiment["checkpoint_interval"] == 0
        assert spec.experiment["checkpoint_at_end"] is False


def test_heads_hold_total_message_and_key_dimensions_at_32(config_root, tmp_path):
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/simple_spread_comm_v1_stage3_heads.yaml"),
        repo_root=tmp_path,
    )
    assert {plan.ablation_value for plan in plans} == {"1", "2", "4", "8"}
    for plan in plans:
        spec = load_experiment_spec(config_root, plan.overrides)
        kwargs = spec.model_config["params"]["comm_kwargs"]
        assert kwargs["message_dim"] == 32
        assert kwargs["key_dim"] == 32
        assert kwargs["num_heads"] in {1, 2, 4, 8}


def test_dropout_suite_replays_whole_sender_failures_in_all_policy_forwards(
    config_root,
    tmp_path,
):
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/simple_spread_comm_v1_stage2_dropout.yaml"),
        repo_root=tmp_path,
    )
    assert {plan.ablation_value for plan in plans} == {
        "p0.00",
        "p0.25",
        "p0.50",
        "p0.75",
    }
    for plan in plans:
        spec = load_experiment_spec(config_root, plan.overrides)
        params = spec.model_config["params"]
        channel = params["comm_kwargs"]["channel"]
        assert channel["type"] == "dropout"
        assert channel["mode"] == "always"
        assert channel["p"] in {0.0, 0.25, 0.5, 0.75}
        assert params["comm_context_keys"]["sender_mask"] == "comm_sender_mask"


def test_graph_topology_records_erdos_renyi_probability_and_seed(
    config_root, tmp_path
):
    plans = expand_suite_config(
        _load_yaml(
            config_root / "sweeps/simple_spread_comm_v1_stage4_graph_topology.yaml"
        ),
        repo_root=tmp_path,
    )
    assert {plan.ablation_value for plan in plans} == {
        "full",
        "directed_ring",
        "erdos_renyi_p0.5_seed1_connected",
    }
    erdos = [plan for plan in plans if "erdos_renyi" in str(plan.ablation_value)]
    assert len(erdos) == 3
    assert {plan.ablation_value for plan in erdos} == {
        "erdos_renyi_p0.5_seed1_connected"
    }
    for plan in erdos:
        kwargs = load_experiment_spec(config_root, plan.overrides).model_config["params"][
            "comm_kwargs"
        ]
        assert kwargs["topology"] == "erdos_renyi"
        assert kwargs["erdos_renyi_p"] == 0.5
        assert kwargs["topology_seed"] == 1

    erdos_spec = load_experiment_spec(config_root, erdos[0].overrides)
    params = erdos_spec.model_config["params"]
    graph = import_from_path(params["comm_class_path"])(
        hidden_dim=params["hidden_dim"],
        **params["comm_kwargs"],
    )
    assert graph._fallback_topology(3).tolist() == [
        [False, True, True],
        [True, False, False],
        [True, False, False],
    ]


def test_sender_budget_has_learned_and_seeded_random_controls(config_root, tmp_path):
    plans = expand_suite_config(
        _load_yaml(config_root / "sweeps/simple_spread_comm_v1_stage4_sender_budget.yaml"),
        repo_root=tmp_path,
    )
    assert {plan.model for plan in plans} == {"comm_gated", "comm_attention"}
    assert {plan.ablation_value.split("_k")[0] for plan in plans} == {
        "learned",
        "random",
    }
    assert {int(plan.ablation_value.split("_k")[1].split("_")[0]) for plan in plans} == {
        1,
        2,
        3,
    }
    for plan in plans:
        kwargs = load_experiment_spec(config_root, plan.overrides).model_config["params"][
            "comm_kwargs"
        ]
        if plan.ablation_value.startswith("random"):
            assert kwargs["sender_selection"] == "random"
            assert kwargs["sender_selection_seed"] == 0
        elif plan.model == "comm_gated":
            assert kwargs["sender_selection"] == "learned"
        else:
            assert kwargs["sender_selection"] == "attention"
