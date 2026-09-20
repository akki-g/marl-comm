"""Fixtures only: no scientific test bank is rolled out by this file."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from commstudy.experiments import mechanism_suite as suite
from commstudy.analysis.mechanism_suite import contact_by_deadline, contrasts

CONFIG = suite.ROOT / "configs/experiments/pcp_mapdn_mechanisms_v1.yaml"


@pytest.fixture
def planned(tmp_path):
    root = tmp_path / "study with spaces"
    suite.plan(CONFIG, root)
    return root


def test_exact_counts_and_index_mapping(planned):
    assert suite.verify_plan(planned)["counts"] == {
        "pcp_main": 60,
        "mapdn_main": 30,
        "pcp_smoke": 12,
        "mapdn_smoke": 6,
        "pcp_qualification": 4,
        "mapdn_qualification": 2,
    }
    for task, count in [("pcp", 60), ("mapdn", 30)]:
        rows = suite.manifest(planned, task)
        assert len({(r["condition"], r["method"], r["seed"]) for r in rows}) == count
        assert all(suite.select_row(planned, task, "main", i) == row for i, row in enumerate(rows))
        for index in [-1, count, True]:
            with pytest.raises(ValueError):
                suite.select_row(planned, task, "main", index)
        assert all(r["spec"]["algorithm"] == "mappo" for r in rows)
        assert all(r["spec"]["experiment"]["keep_checkpoints_num"] is None for r in rows)
        assert {json.dumps(r["spec"]["critic_model"], sort_keys=True) for r in rows}.__len__() == 1
        assert {
            json.dumps(r["spec"]["algorithm_config"], sort_keys=True) for r in rows
        }.__len__() == 1


def test_optional_rows_append_without_changing_core(tmp_path, planned):
    other = tmp_path / "supplements"
    suite.plan(CONFIG, other, include=["capacity", "mapdn_topology"])
    assert len(suite.manifest(other, "pcp")) == 70
    assert len(suite.manifest(other, "mapdn")) == 45
    for task, count in [("pcp", 60), ("mapdn", 30)]:
        assert (
            suite.manifest(other, task)[0]["scientific_config_sha256"]
            == suite.manifest(planned, task)[0]["scientific_config_sha256"]
        )
        original = suite.manifest(planned, task)
        extended = suite.manifest(other, task)
        assert [(r["method"], r["condition"], r["seed"]) for r in extended[:count]] == [
            (r["method"], r["condition"], r["seed"]) for r in original
        ]
    with pytest.raises(ValueError, match="Topology supplement blocked"):
        suite.effective_spec(other, suite.manifest(other, "mapdn")[-1])


def test_unknown_schema_and_mutation(tmp_path, planned):
    from omegaconf import OmegaConf

    doc = OmegaConf.load(CONFIG)
    doc["bogus"] = True
    bad = tmp_path / "bad.yaml"
    OmegaConf.save(doc, bad)
    with pytest.raises(ValueError):
        suite.load_suite(bad)
    path = planned / "preparation/pcp_main_manifest.json"
    rows = json.loads(path.read_text())
    rows[0]["spec"]["experiment"]["max_n_frames"] = 6000
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="altered"):
        suite.verify_plan(planned)


def test_all_presets_fresh_interpreter(planned):
    code = """
import json,sys
from commstudy.experiments.config import experiment_spec_from_dict
from commstudy.experiments.runner import build_model_config
from pathlib import Path
for task in ('pcp','mapdn'):
 for row in json.loads((Path(sys.argv[1])/'preparation'/f'{task}_main_manifest.json').read_text()):
  spec=experiment_spec_from_dict(row['spec'])
  build_model_config(spec.model_config)
print('all 90 reconstructed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(planned)], cwd=suite.ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "all 90 reconstructed" in result.stdout


@pytest.mark.parametrize(
    "method,count",
    [
        ("identity", 0),
        ("local_capacity", 8192),
        ("broadcast", 8192),
        ("gated", 8321),
        ("attention", 16384),
        ("graph_full", 16416),
    ],
)
def test_modules_active_capacity_and_null_masks(method, count):
    from commstudy.utils.imports import import_from_path
    from commstudy.communication.base import CommContext

    raw = suite.make_spec(
        "pcp", "radius1", method, 900, "smoke", Path("/tmp/fixture"), Path("/no-data")
    )
    params = raw["model_config"]["groups"]["adversary"]["params"]
    module = import_from_path(params["comm_class_path"])(128, **params["comm_kwargs"])
    assert sum(p.numel() for p in module.parameters()) == count
    h = torch.randn(2, 3, 128, requires_grad=True)
    output = module(h, CommContext(mask=torch.zeros(2, 3, 3, dtype=torch.bool)))
    assert torch.isfinite(output).all()
    if method == "local_capacity":
        assert not torch.equal(h, output)
        output[0, 0].sum().backward()
        assert h.grad[0, 1:].count_nonzero() == 0
        assert all(p.grad is not None and p.grad.count_nonzero() > 0 for p in module.parameters())
    else:
        assert torch.equal(h, output)
    if method == "identity":
        assert output is h


def test_fixed_topology_resists_full_runtime_override():
    from commstudy.communication.graph import GraphComm
    from commstudy.communication.base import CommContext

    mask = [[False, True, False], [True, False, True], [False, True, False]]
    module = GraphComm(8, message_dim=4, key_dim=4, num_heads=2, fixed_mask=mask)
    h = torch.randn(1, 3, 8, requires_grad=True)
    out = module(h, CommContext(mask=torch.ones(1, 3, 3, dtype=torch.bool)))
    out[0, 0].sum().backward()
    assert h.grad[0, 2].count_nonzero() == 0
    assert h.grad[0, 1].count_nonzero() > 0


@pytest.mark.parametrize("step,expected", [(49, 1), (50, 1), (51, 0), (100, 0)])
def test_contact_boundary(step, expected):
    assert (
        contact_by_deadline({"first_contact_observed": 1, "first_contact_step_or_horizon": step})
        == expected
    )
    assert (
        contact_by_deadline({"first_contact_observed": 0, "first_contact_step_or_horizon": step})
        == 0
    )


def test_seed_signs_interaction_and_incomplete():
    rows = []
    for task, conditions in suite.CONDITIONS.items():
        for condition in conditions:
            for method in suite.METHODS:
                for seed in range(5):
                    value = (
                        (0.4 if condition == "radius1" else 0.3) if method == "broadcast" else 0.2
                    )
                    rows.append(
                        dict(
                            task=task,
                            condition=condition,
                            method=method,
                            seed=seed,
                            live=value,
                            severed=0.2,
                            valid=True,
                        )
                    )
    result = contrasts(rows, list(range(5)), 17)
    primary = [r for r in result["effects"] if r["primary"]]
    assert len(primary) == 12
    assert (
        next(r for r in primary if r["task"] == "mapdn" and r["method"] == "broadcast")["estimate"][
            "mean"
        ]
        < 0
    )
    assert next(r for r in result["visibility_interactions"] if r["method"] == "broadcast")[
        "estimate"
    ]["mean"] == pytest.approx(0.1)
    rows.pop()
    incomplete = contrasts(rows, list(range(5)), 17)
    assert any(not r["complete"] for r in incomplete["effects"])


def test_empty_study_report_retains_missing_denominator(planned):
    from commstudy.analysis.mechanism_report import analyze

    report = analyze(planned)
    assert report["scientific_complete"] is False
    assert report["completion"]["planned"] == 90
    assert len(report["completion"]["cells"]) == 90
    assert report["outcomes"] == []
    assert report["statistics"]["supplement_effects"] == []
    assert all(r["estimate"] is None for r in report["statistics"]["effects"])
    assert not list(Path(report["output"]).glob("*.png"))
    assert json.loads((Path(report["output"]) / "report.json").read_text()) == report


def test_optional_contrasts_are_secondary_and_seed_paired():
    rows = [
        dict(
            task="mapdn",
            condition="case33",
            method=method,
            seed=seed,
            live=value,
            severed=value,
            valid=True,
        )
        for seed in (1100, 1101)
        for method, value in (
            ("graph_full", 0.3),
            ("graph_electrical", 0.1),
            ("graph_random", 0.2),
            ("local_capacity64", 0.25),
        )
    ]
    result = contrasts(rows, [1100, 1101, 1102], 17)
    electrical = next(
        r
        for r in result["supplement_effects"]
        if r["method"] == "graph_electrical" and r["baseline"] == "graph_random"
    )
    assert electrical["estimate"]["seed_differences"] == pytest.approx([0.1, 0.1])
    assert electrical["planned_n"] == 3 and electrical["complete"] is False
    assert all(r["primary"] is False for r in result["supplement_effects"])
    assert len([r for r in result["effects"] if r["primary"]]) == 12


def test_new_launch_guards_and_old_guard_unchanged(planned):
    from commstudy.experiments.protocols import validate_protocol_launch

    row = suite.select_row(planned, "pcp", "main", 0)
    spec = suite.effective_spec(planned, row)
    with pytest.raises(ValueError, match="preflight"):
        validate_protocol_launch(
            spec,
            dict(kind=suite.SUITE, run_root=str(planned), task="pcp", stage="main", index=0),
            repo_root=suite.ROOT,
        )
    with pytest.raises(ValueError):
        validate_protocol_launch(spec, {}, repo_root=suite.ROOT)
    with pytest.raises(ValueError):
        suite.freeze(planned)


def test_string_training_closure_to_evaluator_route(tmp_path, monkeypatch):
    from commstudy.experiments.mechanism_execution import reload_closed_policy
    from commstudy.experiments.bookkeeping import atomic_write_json, read_json
    from commstudy.analysis import diagnostics

    path = tmp_path / "training.json"
    atomic_write_json(str(path), {"run_dir": str(tmp_path / "real run")})
    closure = json.loads(json.dumps({"training_summary_path": str(path)}))
    called = []
    monkeypatch.setattr(
        diagnostics,
        "load_frozen_experiment",
        lambda run_dir, **kw: called.append(run_dir) or "loaded",
    )
    assert reload_closed_policy(closure["training_summary_path"], tmp_path / "scratch") == "loaded"
    assert called == [str(tmp_path / "real run")]
    assert read_json(str(path))["run_dir"] == called[0]
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "bad.json", {"bad": float("nan")})


def test_evaluation_before_closure_cannot_open_test_bank(planned, monkeypatch):
    from commstudy.experiments import mechanism_execution as execution

    monkeypatch.setattr(execution, "verify_approval", lambda root: {})
    monkeypatch.setattr(execution, "verify_preflight", lambda root, task: {})
    with pytest.raises(ValueError, match="training_closure"):
        execution.evaluate(planned, "pcp", 0)
    assert not (planned / "evaluation").exists()


def test_topology_matching_and_degeneracy():
    import networkx as nx
    from commstudy.experiments.mechanism_data import topology_bank

    with pytest.raises(ValueError, match="degenerate"):
        topology_bank(nx.path_graph(3), [0, 1, 2], [10])
    result = topology_bank(nx.path_graph(12), list(range(12)), [10, 11])
    for seed in ("10", "11"):
        left, right = result["graph_electrical"][seed], result["graph_random"][seed]
        assert left != right
        assert [sum(row) for row in left] == [sum(row) for row in right]
        assert all(not row[i] for i, row in enumerate(right))


def test_failures_are_not_automatically_retried(tmp_path):
    from commstudy.experiments.mechanism_execution import failure_category
    from commstudy.experiments.health import TrainingHealthError

    assert failure_category(TrainingHealthError("Non-finite gradient")) == "scientific_failure"
    assert failure_category(TrainingHealthError("PPO replay mismatch")) == "integrity_failure"
    assert failure_category(InterruptedError()) == "infrastructure_failure"
    assert failure_category(ValueError("corrupted checkpoint")) == "integrity_failure"


def test_misplaced_receipt_cannot_close_another_policy(planned):
    row = suite.select_row(planned, "pcp", "main", 0)
    other = suite.select_row(planned, "pcp", "main", 1)
    attempt = planned / "runs/pcp" / row["run_id"] / "attempt_000"
    attempt.mkdir(parents=True)
    suite.write_new(attempt / "receipt.json", {"row": other, "status": "completed"})
    with pytest.raises(ValueError, match="different manifest row"):
        suite.latest_attempt(planned, row)
