"""Use trivial child processes to test durable orchestration without RL collection."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/supervise_main_direction_phase.py"
    spec = importlib.util.spec_from_file_location("phase_supervisor_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    driver = tmp_path / "fake_scientific_driver.py"
    driver.write_text(
        "import sys,time\n"
        "print('durable child output',flush=True)\n"
        "if sys.argv[1]=='fail': raise SystemExit(7)\n"
        "if sys.argv[1]=='wait': time.sleep(30)\n"
    )
    monkeypatch.setattr(module, "DRIVER", driver)
    operations = tmp_path / "operations"
    operations.mkdir()
    instance = module.Coordinator(SimpleNamespace(
        python=Path(sys.executable), operations=operations, output=tmp_path / "science", workers=2))
    instance.preparation_sha = "fixture-only"
    return instance, operations


def test_output_and_exit_survive_child_completion_without_tool_pipe(coordinator):
    instance, operations = coordinator
    instance.batch([("row", ["ok"])])
    assert (operations / "row.log").read_text() == "durable child output\n"
    assert json.loads((operations / "row_exit.json").read_text())["exit_code"] == 0
    assert json.loads((operations / "row_launch.json").read_text())["automatic_retry"] is False
    with pytest.raises(ValueError, match="already attempted"):
        instance.launch("row", ["ok"])


def test_failed_stage_does_not_start_later_batch_and_cleanup_stops_owned_child(coordinator):
    instance, operations = coordinator
    try:
        with pytest.raises(RuntimeError, match="failed with exit 7"):
            instance.batch([("bad", ["fail"]), ("sibling", ["wait"]), ("later", ["ok"])])
        siblings = [entry[0] for entry in instance.running.values()]
        assert len(siblings) == 1
        assert not (operations / "later_launch.json").exists()
        instance.terminate_owned()
        assert all(process.poll() is not None for process in siblings)
        assert json.loads((operations / "bad_exit.json").read_text())["exit_code"] == 7
    finally:
        instance.terminate_owned()


def test_changed_driver_is_rejected_before_process_launch(coordinator):
    instance, operations = coordinator
    instance.driver_sha = "different-declared-source"
    with pytest.raises(ValueError, match="changed"):
        instance.launch("not_started", ["ok"])
    assert not (operations / "not_started_launch.json").exists()
