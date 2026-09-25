"""End-to-end acceptance of the two-config experiment workflow."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from omegaconf import OmegaConf
import pandas as pd
import pytest
import torch

from commstudy.experiments.build import build_experiment
from commstudy.experiments.config import (
    METHODS,
    allocation,
    config_from_dict,
    load_config,
    make_spec,
)
from commstudy.experiments.bookkeeping import atomic_write_json, read_json
from commstudy.experiments.runner import (
    compatible_completion,
    experiment_lock,
    file_sha,
    load_checkpoint,
    worker_count,
)
from mapdn_fixture import _build_network

ROOT = Path(__file__).resolve().parents[1]


def small_config(root, task="pcp", *, workers=2):
    config = load_config(ROOT / "configs" / f"{task}.yaml")
    config.update(seeds=[1100], frames=24, output_dir=str(root / "results"))
    config["training"].update(batch_frames=8, num_envs=1, epochs=1, minibatch_size=8)
    config["model"].update(hidden_dim=16, critic_hidden_sizes=[16, 16], message_dim=8, key_dim=8)
    config["task_params"]["max_steps"] = 4
    # First native check at frame 8, scheduled at 16, final check at 24.
    config["evaluation"].update(interval=16, episodes=2)
    config["execution"]["workers"] = workers
    if task == "mapdn":
        data = root / "feeder"
        data.mkdir(parents=True)
        _build_network(data)
        config["task_params"]["data_path"] = str(data)
    return config


def command(config, path):
    path.write_text(OmegaConf.to_yaml(config))
    return [sys.executable, str(ROOT / "scripts/run_experiment.py"), str(path)]


def launch(config, path):
    result = subprocess.run(command(config, path), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("task", ["pcp", "mapdn"])
def test_default_and_single_seed_allocations(task):
    config = load_config(ROOT / "configs" / f"{task}.yaml")
    assert len(allocation(config)) == 25
    assert set(method for method, _ in allocation(config)) == set(METHODS)
    config["seeds"] = [1100]
    assert len(allocation(config)) == 5


@pytest.mark.parametrize(
    "change",
    [
        {"seeds": []},
        {"seeds": [1, 1]},
        {"seeds": list(range(6))},
        {"seeds": [True]},
        {"frames": 600001},
        {"training": {"batch_frames": 6001}},
        {"training": {"minibatch_size": 401}},
        {"evaluation": {"interval": 6001}},
        {"unknown": 1},
        {"execution": {"workers": 0}},
        {"algorithm": "oops"},
        {"task": "simple_spread"},
        {"model": {"message_dim": 0}},
    ],
)
def test_invalid_allocation_rejected_without_outputs(tmp_path, change):
    config = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/pcp.yaml"))
    config.update(change, output_dir=str(tmp_path / "output"))
    with pytest.raises((ValueError, TypeError)):
        config_from_dict(config, base_dir=tmp_path)
    assert not (tmp_path / "output").exists()


def test_mapdn_qmix_rejected_before_data_or_training(tmp_path):
    config = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/mapdn.yaml"))
    config["algorithm"] = "qmix"
    with pytest.raises(ValueError, match="QMIX"):
        config_from_dict(config, base_dir=tmp_path)


def test_missing_mapdn_data_fails_dry_run_without_creating_outputs(tmp_path):
    config = small_config(tmp_path)
    config["task"] = "mapdn"
    config["task_params"] = {"data_path": str(tmp_path / "missing")}
    config["normalization"] = {"episodes": 16, "seed": 3200000}
    result = subprocess.run(
        command(config, tmp_path / "config.yaml") + ["--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "5 policies" in result.stdout and "model.p" in result.stderr
    assert not Path(config["output_dir"]).exists()


@pytest.fixture(scope="module", params=["pcp", "mapdn"])
def completed_experiment(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"workflow_{request.param}")
    config = small_config(root, request.param)
    launch(config, root / "config.yaml")
    return config, root


def test_all_methods_optimize_log_and_reload(completed_experiment):
    config, root = completed_experiment
    output = Path(config["output_dir"])
    provenance = read_json(output / "provenance.json")
    assert provenance["source_sha256"] and provenance["dependencies"]["benchmarl"]
    combined = pd.read_csv(output / "metrics.csv", keep_default_na=False)
    assert set(combined.method) == set(METHODS)
    assert torch.isfinite(torch.tensor(combined.value)).all()
    for method in METHODS:
        run = output / method / "seed_1100"
        assert compatible_completion(run, provenance["binding"])
        data = pd.read_csv(run / "metrics.csv", keep_default_na=False)
        evaluated = data[(data.phase == "evaluation") & (data.metric == "return_mean")]
        assert set(evaluated.frames) == {8, 16, 24}
        with (run / "metrics.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        out_rows = [
            json.loads(line.removeprefix("METRIC "))
            for line in (run / "run.out").read_text().splitlines()
            if line.startswith("METRIC ")
        ]
        assert [{k: str(v) for k, v in row.items()} for row in out_rows] == csv_rows
        if config["task"] == "pcp":
            for phase in ("collection", "evaluation"):
                rows = data[(data.phase == phase) & (data.metric == "return_mean")]
                assert list(rows[rows.group == ""].value) == list(
                    rows[rows.group == "adversary"].value
                )
        checkpoint = torch.load(run / "checkpoint.pt", weights_only=False)
        spec = make_spec(config, method, 1100)
        spec.experiment["save_folder"] = str(root / "initial" / method)
        initial = build_experiment(spec)
        group = "adversary" if config["task"] == "pcp" else "agents"
        try:
            before = initial.group_policies[group].state_dict()
            after = checkpoint["policies"][group]
            assert any(not torch.equal(value, after[name]) for name, value in before.items())
            critic_keys = [name for name in checkpoint["losses"][group] if "critic" in name]
            initial_loss = initial.losses[group].state_dict()
            assert critic_keys
            assert any(
                not torch.equal(initial_loss[k], checkpoint["losses"][group][k])
                for k in critic_keys
            )
        finally:
            initial.close()
        restored = load_checkpoint(run / "checkpoint.pt", save_folder=root / "reload" / method)
        try:
            for name, value in restored.group_policies[group].state_dict().items():
                assert torch.equal(value, checkpoint["policies"][group][name])
            for name, value in restored.losses[group].state_dict().items():
                expected = checkpoint["losses"][group][name]
                assert (
                    torch.equal(value, expected)
                    if isinstance(value, torch.Tensor)
                    else value == expected
                )
        finally:
            restored.close()
    if config["task"] == "mapdn":
        record = read_json(output / "data/normalization.json")
        assert len(record["windows"]) == 16
        assert record["split"] == "train"
        assert [w["seed"] for w in record["windows"]] == list(range(3200000, 3200016))


def test_repeat_skips_and_retry_preserves_attempt(completed_experiment):
    config, root = completed_experiment
    output = Path(config["output_dir"])
    runs = [output / method / "seed_1100" for method in METHODS]
    checksums = {path: file_sha(path / "metrics.csv") for path in runs}
    result = launch(config, root / "config.yaml")
    assert result.stdout.count("SKIP ") == 5
    assert all(file_sha(path / "metrics.csv") == checksums[path] for path in runs)
    failed = runs[0]
    status = read_json(failed / "status.json")
    status.update(state="failed", error="Injected incomplete attempt")
    atomic_write_json(failed / "status.json", status)
    result = launch(config, root / "config.yaml")
    assert result.stdout.count("SKIP ") == 4
    assert file_sha(failed / "attempts/attempt_001/metrics.csv") == checksums[failed]
    assert len(pd.read_csv(output / "runs.csv")) == 5
    current = pd.read_csv(failed / "metrics.csv")
    old = pd.read_csv(failed / "attempts/attempt_001/metrics.csv")
    cols = [c for c in current.columns if c != "timestamp"]
    pd.testing.assert_frame_equal(
        current[current.metric != "wall_time_seconds"][cols].reset_index(drop=True),
        old[old.metric != "wall_time_seconds"][cols].reset_index(drop=True),
    )
    changed = deepcopy(config)
    changed["training"]["lr"] *= 2
    bad = subprocess.run(command(changed, root / "changed.yaml"), capture_output=True, text=True)
    assert bad.returncode == 1 and "Incompatible" in bad.stderr


def test_sequential_parallel_exact_reproducibility(completed_experiment):
    config, root = completed_experiment
    sequential = deepcopy(config)
    sequential["output_dir"] = str(root / "sequential")
    sequential["execution"]["workers"] = 1
    launch(sequential, root / "sequential.yaml")
    for method in METHODS:

        def values(config, method=method):
            run = Path(config["output_dir"]) / method / "seed_1100"
            df = pd.read_csv(run / "metrics.csv").drop(columns="timestamp")
            return df[df.metric != "wall_time_seconds"].reset_index(drop=True)

        pd.testing.assert_frame_equal(values(config), values(sequential))


def test_lock_and_cpu_allocation(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert worker_count(5) <= 2
    with experiment_lock(tmp_path):
        with pytest.raises(ValueError, match="Another launch"):
            with experiment_lock(tmp_path):
                pass


def wait_until(predicate, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.02)
    pytest.fail("Process did not reach the expected state")


def test_interruption_and_concurrent_launch_recover(tmp_path):
    config = small_config(tmp_path)
    config["frames"] = 400
    process = subprocess.Popen(
        command(config, tmp_path / "config.yaml"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    output = Path(config["output_dir"])
    try:
        wait_until(lambda: (output / "identity/seed_1100/status.json").exists())
        # Same directory is protected even before a worker has finished importing.
        with pytest.raises(ValueError, match="Another launch"):
            with experiment_lock(output):
                pass
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 130
        statuses = [read_json(p) for p in output.glob("*/seed_*/status.json")]
        assert any(status["state"] == "interrupted" for status in statuses)
        launch(config, tmp_path / "config.yaml")
        assert set(pd.read_csv(output / "runs.csv").state) == {"completed"}
        assert list(output.glob("*/seed_*/attempts/attempt_001/status.json"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_killed_worker_does_not_cancel_other_methods(tmp_path):
    config = small_config(tmp_path)
    config["frames"] = 400
    process = subprocess.Popen(
        command(config, tmp_path / "config.yaml"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    output = Path(config["output_dir"])
    try:
        status = wait_until(
            lambda: (
                s
                if (s := read_json(output / "identity/seed_1100/status.json", {})).get("pid")
                else None
            )
        )
        os.kill(status["pid"], signal.SIGKILL)
        assert process.wait(timeout=120) == 1
        summary = pd.read_csv(output / "runs.csv")
        assert list(summary[summary.method == "identity"].state) == ["failed"]
        assert set(summary[summary.method != "identity"].state) == {"completed"}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_qmix_optimizes_all_five_discrete_pcp_methods(tmp_path):
    config = small_config(tmp_path)
    config.update(algorithm="qmix", algorithm_params={})
    config["training"].pop("epochs")
    config["training"].pop("minibatch_size")
    config["training"].update(
        optimizer_steps=1, train_batch_size=2, memory_size=24, init_random_frames=0
    )
    launch(config, tmp_path / "qmix.yaml")
    for method in METHODS:
        path = Path(config["output_dir"]) / method / "seed_1100/checkpoint.pt"
        checkpoint = torch.load(path, weights_only=False)
        resolved = config_from_dict(config, base_dir=tmp_path)
        spec = make_spec(resolved, method, 1100)
        spec.experiment["save_folder"] = str(tmp_path / "initial" / method)
        initial = build_experiment(spec)
        try:
            assert any(
                not torch.equal(value, checkpoint["policies"]["adversary"][key])
                for key, value in initial.group_policies["adversary"].state_dict().items()
            )
        finally:
            initial.close()


def test_corrupt_completed_outputs_are_not_silently_reused(tmp_path):
    for name in ("checkpoint.pt", "metrics.csv", "run.out"):
        (tmp_path / name).write_text("original")
    atomic_write_json(
        tmp_path / "status.json",
        {
            "state": "completed",
            "binding": "same",
            "sha256": {
                name: file_sha(tmp_path / name) for name in ("checkpoint.pt", "metrics.csv")
            },
        },
    )
    assert compatible_completion(tmp_path, "same")
    with pytest.raises(ValueError, match="Incompatible"):
        compatible_completion(tmp_path, "different")
    (tmp_path / "checkpoint.pt").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        compatible_completion(tmp_path, "same")
