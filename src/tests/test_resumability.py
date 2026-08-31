"""Resumability semantics for interrupted runs.

A worker can vanish without ever writing a terminal status: Slurm preemption,
a walltime kill, an OOM kill, or a crashed node. The row is then left claiming
``running`` forever. These tests pin the behaviour that lets a suite recover
from that without ever stealing a row from a worker that is still alive.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta

import pytest

from commstudy.experiments.bookkeeping import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    RunContext,
    RunRecorder,
    atomic_write_json,
    current_status,
    is_stale_running,
    scheduler_identity,
    seconds_since_heartbeat,
)
from commstudy.experiments.config import load_experiment_spec
from commstudy.experiments.sweeps import (
    DEFAULT_STALE_AFTER_SECONDS,
    RunPlan,
    execute_plan,
    select_plans,
    sync_manifest_status,
    write_manifest,
)


def _status(run_dir, status, *, heartbeat_age_seconds=None, omit_heartbeat=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    document = {"status": status, "timestamp": datetime.now(UTC).isoformat(), "error": None}
    if not omit_heartbeat:
        stamp = datetime.now(UTC) - timedelta(seconds=heartbeat_age_seconds or 0)
        document["heartbeat"] = stamp.isoformat()
    atomic_write_json(run_dir / "status.json", document)
    return run_dir


def _plan(tmp_path, run_id="run", status="pending"):
    return RunPlan(
        run_id=run_id,
        suite_id="suite",
        task="vmas_simple_spread",
        algorithm="mappo",
        model="comm_identity",
        seed=0,
        ablation="main",
        ablation_value=None,
        max_n_frames=6_000,
        status=status,
        output_root=tmp_path / "runs",
        overrides=("model=comm_identity", "seed=0"),
        command="uv run python scripts/train.py",
    )


def test_fresh_heartbeat_is_not_stale(tmp_path):
    run_dir = _status(tmp_path / "live", STATUS_RUNNING, heartbeat_age_seconds=5)

    assert current_status(run_dir) == STATUS_RUNNING
    assert seconds_since_heartbeat(run_dir) < 60
    assert is_stale_running(run_dir, 3_600) is False


def test_silent_running_row_is_stale(tmp_path):
    run_dir = _status(tmp_path / "dead", STATUS_RUNNING, heartbeat_age_seconds=7_200)

    assert is_stale_running(run_dir, 3_600) is True


def test_running_row_without_any_heartbeat_is_stale(tmp_path):
    """Rows written before heartbeats existed must still be reclaimable."""

    run_dir = tmp_path / "legacy"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"status": STATUS_RUNNING}), encoding="utf-8")

    assert seconds_since_heartbeat(run_dir) is None
    assert is_stale_running(run_dir, 3_600) is True


@pytest.mark.parametrize("status", [STATUS_COMPLETED, STATUS_FAILED])
def test_terminal_rows_are_never_stale(tmp_path, status):
    run_dir = _status(tmp_path / status, status, heartbeat_age_seconds=99_999)

    assert is_stale_running(run_dir, 3_600) is False


def test_execute_plan_leaves_a_live_running_row_alone(tmp_path):
    plan = _plan(tmp_path)
    _status(plan.output_root / plan.suite_id / plan.run_id, STATUS_RUNNING, heartbeat_age_seconds=1)

    result, experiment = execute_plan(
        plan,
        config_root=tmp_path,
        repo_root=tmp_path,
        reclaim_stale_after=DEFAULT_STALE_AFTER_SECONDS,
    )

    assert result.status == STATUS_RUNNING
    assert experiment is None


def test_execute_plan_ignores_stale_rows_without_the_opt_in(tmp_path):
    plan = _plan(tmp_path)
    _status(
        plan.output_root / plan.suite_id / plan.run_id,
        STATUS_RUNNING,
        heartbeat_age_seconds=99_999,
    )

    result, experiment = execute_plan(plan, config_root=tmp_path, repo_root=tmp_path)

    assert result.status == STATUS_RUNNING
    assert experiment is None


def test_execute_plan_retries_a_stale_row_under_a_new_id(tmp_path, config_root):
    plan = _plan(tmp_path)
    run_dir = plan.output_root / plan.suite_id / plan.run_id
    _status(run_dir, STATUS_RUNNING, heartbeat_age_seconds=99_999)

    result, _ = execute_plan(
        plan,
        config_root=config_root,
        repo_root=tmp_path,
        reclaim_stale_after=1.0,
    )

    # The abandoned evidence is preserved; the retry is a separate directory.
    assert run_dir.exists()
    assert result.run_id == f"{plan.run_id}__retry01"
    assert result.retry_of == plan.run_id


def test_completed_rows_are_never_reclaimed(tmp_path):
    plan = _plan(tmp_path)
    _status(
        plan.output_root / plan.suite_id / plan.run_id,
        STATUS_COMPLETED,
        heartbeat_age_seconds=99_999,
    )

    result, experiment = execute_plan(
        plan,
        config_root=tmp_path,
        repo_root=tmp_path,
        reclaim_stale_after=1.0,
    )

    assert result.status == STATUS_COMPLETED
    assert experiment is None


def test_sync_manifest_reports_stale_rows_as_failed(tmp_path):
    live = _plan(tmp_path, run_id="live")
    dead = _plan(tmp_path, run_id="dead")
    _status(live.output_root / live.suite_id / live.run_id, STATUS_RUNNING, heartbeat_age_seconds=2)
    _status(
        dead.output_root / dead.suite_id / dead.run_id,
        STATUS_RUNNING,
        heartbeat_age_seconds=99_999,
    )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [live, dead])

    without = {plan.run_id: plan.status for plan in sync_manifest_status(manifest)}
    assert without == {"live": STATUS_RUNNING, "dead": STATUS_RUNNING}

    with_reclaim = {
        plan.run_id: plan.status for plan in sync_manifest_status(manifest, stale_after=3_600)
    }
    assert with_reclaim == {"live": STATUS_RUNNING, "dead": STATUS_FAILED}


def test_heartbeat_refreshes_only_running_rows(tmp_path, config_root):
    spec = load_experiment_spec(config_root, ["model=comm_identity", "seed=0"])
    context = RunContext(suite_id="suite", run_id="run", output_root=tmp_path / "runs")
    recorder = RunRecorder(context, spec, repo_root=tmp_path)
    recorder.start()

    original = json.loads(context.status_path.read_text())
    assert original["status"] == STATUS_RUNNING
    assert original["owner"]["pid"] == scheduler_identity()["pid"]

    _status(context.run_dir, STATUS_RUNNING, heartbeat_age_seconds=500)
    recorder.heartbeat()
    assert seconds_since_heartbeat(context.run_dir) < 60

    # A terminal row must not be resurrected into "running" by a late callback.
    atomic_write_json(
        context.status_path,
        {"status": STATUS_COMPLETED, "timestamp": datetime.now(UTC).isoformat()},
    )
    recorder.heartbeat()
    assert current_status(context.run_dir) == STATUS_COMPLETED


def test_index_selection_supports_array_chunking(tmp_path):
    plans = [_plan(tmp_path, run_id=f"row{index}") for index in range(5)]

    assert [plan.run_id for plan in select_plans(plans, index=0, count=2)] == ["row0", "row1"]
    assert [plan.run_id for plan in select_plans(plans, index=3, count=2)] == ["row3", "row4"]
    # A trailing chunk must clamp instead of failing the whole array task.
    assert [plan.run_id for plan in select_plans(plans, index=4, count=3)] == ["row4"]
    with pytest.raises(IndexError):
        select_plans(plans, index=5, count=1)


def test_manifest_roundtrip_preserves_retry_lineage(tmp_path):
    plan = dataclasses.replace(_plan(tmp_path), retry_of="base", attempt=2)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [plan])

    from commstudy.experiments.sweeps import read_manifest

    (restored,) = read_manifest(manifest)
    assert restored.retry_of == "base"
    assert restored.attempt == 2
