"""Exercise batch setup without a scheduler, module installation or downloads."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm/run_experiment.sbatch"


@pytest.fixture
def cluster(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # The shim uses real advisory locking, including on macOS without flock(1).
    driver = binaries / "driver"
    driver.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
            import fcntl, json, os, pathlib, shlex, shutil, sys, time
            name = pathlib.Path(sys.argv[0]).name
            def record(event, **values):
                with open(os.environ["TEST_EVENTS"], "a") as out:
                    out.write(json.dumps({"event": event, **values}) + "\\n")
            if name == "flock":
                assert sys.argv[1] == "--exclusive"
                fcntl.flock(int(sys.argv[2]), fcntl.LOCK_EX)
            elif name in ("module", "conda"):
                record(name, args=sys.argv[1:])
                if name == "module" and sys.argv[1] == "load":
                    sys.exit(int(os.environ.get("TEST_MODULE_FAILURE", "0")))
            elif name == "curl":
                record("install")
                if os.environ.get("TEST_DOWNLOAD_FAILURE"):
                    sys.exit(22)
                print('mkdir -p "$UV_UNMANAGED_INSTALL"')
                print('cp "$TEST_DRIVER" "$UV_UNMANAGED_INSTALL/uv"')
            elif name == "uv":
                if sys.argv[1] == "--version":
                    print("uv 0.12.5")
                    sys.exit(0)
                assert sys.argv[1:3] == ["sync", "--locked"]
                assert "--no-python-downloads" in sys.argv
                record("sync_start")
                if os.environ.get("TEST_SYNC_FAILURE"):
                    sys.exit(23)
                time.sleep(0.1)
                env = pathlib.Path(os.environ["UV_PROJECT_ENVIRONMENT"])
                (env / "bin").mkdir(parents=True, exist_ok=True)
                shutil.copy(os.environ["TEST_DRIVER"], env / "bin/python")
                (env / "bin/activate").write_text(
                    "export VIRTUAL_ENV=" + shlex.quote(str(env)) + "\\n"
                    "export PATH=" + shlex.quote(str(env / "bin")) + ':"$PATH"\\n'
                )
                record("sync_end")
            elif name == "python":
                assert sys.argv[1:3] == ["-u", "scripts/run_experiment.py"]
                record("run", config=sys.argv[3], venv=os.environ["VIRTUAL_ENV"])
                sys.exit(int(os.environ.get("TEST_RUN_FAILURE", "0")))
            else:
                raise AssertionError(name)
            """)
    )
    driver.chmod(0o755)
    for name in ("flock", "module", "conda", "curl"):
        (binaries / name).symlink_to(driver)
    (binaries / "python3").symlink_to(sys.executable)
    shell_init = tmp_path / "shell-init"
    shell_init.write_text(
        'module() { "$TEST_BIN/module" "$@"; }\nconda() { "$TEST_BIN/conda" "$@"; }\n'
    )
    env = {
        **os.environ,
        "PATH": f"{binaries}:/usr/bin:/bin",
        "BASH_ENV": str(shell_init),
        "SLURM_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "TEST_BIN": str(binaries),
        "TEST_DRIVER": str(driver),
        "TEST_EVENTS": str(tmp_path / "events.jsonl"),
    }
    return tmp_path, env


def launch(env, *args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=15
    )


def events(root):
    path = root / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_first_setup_reuse_and_both_configs(cluster):
    root, env = cluster
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for index in (0, 1):
        result = launch({**env, "SLURM_ARRAY_TASK_ID": str(index)})
        assert result.returncode == 0, result.stderr
    records = events(root)
    assert sum(r["event"] == "install" for r in records) == 1
    assert [r["config"] for r in records if r["event"] == "run"] == [
        "configs/pcp.yaml",
        "configs/mapdn.yaml",
    ]
    assert all(r["venv"] == str(root / ".venv-newton") for r in records if r["event"] == "run")
    assert {tuple(r["args"]) for r in records if r["event"] == "module"} == {
        ("purge",),
        ("load", "anaconda/anaconda-2024.10"),
        ("list",),
    }


def test_simultaneous_array_tasks_serialize_setup(cluster):
    root, env = cluster
    processes = [
        subprocess.Popen(
            ["bash", str(SCRIPT)],
            env={**env, "SLURM_ARRAY_TASK_ID": str(index)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in (0, 1)
    ]
    try:
        for process in processes:
            _, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    records = events(root)
    assert sum(r["event"] == "install" for r in records) == 1
    assert [r["event"] for r in records if r["event"].startswith("sync_")] == [
        "sync_start",
        "sync_end",
        "sync_start",
        "sync_end",
    ]
    assert len([r for r in records if r["event"] == "run"]) == 2


@pytest.mark.parametrize("failure", ["MODULE", "DOWNLOAD", "SYNC"])
def test_setup_failure_never_launches_training(cluster, failure):
    root, env = cluster
    result = launch({**env, f"TEST_{failure}_FAILURE": "1"})
    assert result.returncode != 0
    assert not any(r["event"] == "run" for r in events(root))


def test_training_failure_reaches_scheduler(cluster):
    _, env = cluster
    assert launch({**env, "TEST_RUN_FAILURE": "17"}).returncode == 17


@pytest.mark.parametrize("index,args", [("2", []), ("0", ["configs/pcp.yaml"])])
def test_invalid_submission_stops_before_installation(cluster, index, args):
    root, env = cluster
    assert launch({**env, "SLURM_ARRAY_TASK_ID": index}, *args).returncode == 2
    assert not events(root)


def test_requires_a_compute_allocation(cluster):
    root, env = cluster
    env.pop("SLURM_JOB_ID")
    assert launch(env).returncode != 0
    assert not events(root)
