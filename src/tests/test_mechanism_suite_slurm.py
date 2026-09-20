"""Mock Slurm only. Never calls a real scheduler."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from commstudy.experiments import mechanism_slurm as slurm
from commstudy.experiments.mechanism_suite import ROOT


@pytest.fixture
def cluster(tmp_path):
    root = tmp_path / "study root with spaces"
    prep = root / "preparation"
    prep.mkdir(parents=True)
    manifests = {}
    for task, main, smokes, qual in [("pcp", 60, 12, 4), ("mapdn", 30, 6, 2)]:
        for stage, count in [("main", main), ("smoke", smokes), ("qualification", qual)]:
            path = prep / f"{task}_{stage}_manifest.json"
            path.write_text(json.dumps([{"index": i} for i in range(count)]))
            manifests[path.name] = slurm.sha(path)
    (prep / "plan.json").write_text(json.dumps({"manifests": manifests}))
    (prep / "suite.json").write_text(json.dumps({"include": []}))
    values = {k: "-" for k in slurm.PROFILE_KEYS}
    values.update(
        SOURCE_ROOT=str(ROOT),
        RUN_ROOT=str(root),
        BOOTSTRAP_PYTHON="/usr/bin/python3",
        PCP_PYTHON="/new pcp/bin/python",
        MAPDN_PYTHON="/new mapdn/bin/python",
        WHEELHOUSE="/staged wheels",
        REQUIREMENTS_LOCK="/linux lock.txt",
        ACCOUNT="test_account",
        PARTITION="cpu",
        DEVICE="cpu",
        GPU_COUNT="0",
        CPUS="2",
        MEMORY="8G",
        PCP_TIME="1:00:00",
        MAPDN_TIME="1:00:00",
        CONTROL_TIME="1:00:00",
        PCP_CONCURRENCY="2",
        MAPDN_CONCURRENCY="3",
        MIN_FREE_GIB="20",
    )
    profile = tmp_path / "cluster profile.env"
    import shlex

    profile.write_text("\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items()) + "\n")
    return root, profile


def test_dry_run_zero_submissions_and_manifest_arrays(cluster, monkeypatch):
    root, profile = cluster
    monkeypatch.setattr(
        slurm.subprocess, "run", lambda *a, **k: pytest.fail("Dry-run invoked scheduler")
    )
    result = slurm.submit(root, profile, phase="main")
    jobs = {r["name"]: r for r in result["jobs"]}
    assert jobs["train_pcp"]["indices"] == "0-59"
    assert jobs["train_mapdn"]["indices"] == "0-29"
    assert "--array" in jobs["train_pcp"]["command"]
    assert "0-59%2" in jobs["train_pcp"]["command"]
    assert all(
        Path(r["command"][r["command"].index("--output") + 1]).is_absolute() for r in jobs.values()
    )
    assert (root / "slurm_logs").exists()
    assert not (root / "submissions.jsonl").exists()
    close = jobs["close_pcp"]["command"]
    assert close[close.index("--dependency") + 1].startswith("afterany:")
    evaluation = jobs["evaluate_pcp"]["command"]
    assert (
        evaluation[evaluation.index("--dependency") + 1]
        == "afterok:<close_pcp_jobid>:<close_mapdn_jobid>"
    )
    analysis = jobs["analyze"]["command"]
    assert analysis[analysis.index("--dependency") + 1].startswith("afterany:")
    for job in jobs.values():
        command = job["command"]
        script_index = next(i for i, v in enumerate(command) if v.endswith(".sbatch"))
        assert command[script_index + 1 : script_index + 3] == [str(profile), str(root)]
        assert "--no-requeue" in command[:script_index]


def test_semicolon_ids_dependencies_duplicate_and_failure_preservation(cluster, monkeypatch):
    root, profile = cluster
    (root / "gates").mkdir()
    (root / "gates/main_approval.json").write_text("{}")
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=f"{100 + len(calls)};newton\n", stderr="")

    monkeypatch.setattr(slurm.subprocess, "run", sbatch)
    slurm.submit(root, profile, phase="main", execute=True)
    ledger = [json.loads(x) for x in (root / "submissions.jsonl").read_text().splitlines()]
    assert len(ledger) == 7
    assert all(r["cluster"] == "newton" for r in ledger)
    assert "afterany:101" in calls[1]
    assert "afterok:102:104" in calls[4]
    with pytest.raises(ValueError, match="Duplicate"):
        slurm.submit(root, profile, phase="main", execute=True)
    assert len(calls) == 7


def test_failed_submission_aborts_remaining_dag(cluster, monkeypatch):
    root, profile = cluster
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0 if len(calls) == 1 else 1,
            stdout="123;newton\n" if len(calls) == 1 else "",
            stderr="mock refusal",
        )

    monkeypatch.setattr(slurm.subprocess, "run", sbatch)
    with pytest.raises(RuntimeError, match="earlier submitted"):
        slurm.submit(root, profile, phase="prepare", execute=True)
    ledger = [json.loads(x) for x in (root / "submissions.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert ledger[0]["job_id"] == "123"
    assert ledger[1]["submission_status"] == "failed"
    assert not (root / ".submission.lock").exists()


def test_literal_profiles_reject_unknown_and_cuda(cluster):
    root, profile = cluster
    profile.write_text(profile.read_text() + 'SURPRISE="$(touch /tmp/not-executed)"\n')
    with pytest.raises(ValueError):
        slurm.load_profile(profile, root)
    profile.write_text(profile.read_text().replace("DEVICE=cpu", "DEVICE=cuda"))
    with pytest.raises(ValueError):
        slurm.load_profile(profile, root)


def test_all_shells_parse_and_real_cli_help():
    directory = ROOT / "slurm/communication_comparison_v1"
    files = list(directory.glob("*.sh")) + list(directory.glob("*.sbatch"))
    assert len(files) == 14
    for path in files:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/communication_suite.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "close-training" in result.stdout


def test_spooled_workers_use_absolute_source_and_real_cli(cluster, monkeypatch, tmp_path):
    import sys

    root, profile = cluster
    profile.write_text(
        profile.read_text()
        .replace("'/new pcp/bin/python'", sys.executable)
        .replace("'/new mapdn/bin/python'", sys.executable)
    )
    bindir = tmp_path / "mockbin"
    bindir.mkdir()
    runner = bindir / "srun"
    runner.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$MOCK_ARGV"\n')
    runner.chmod(0o755)
    import os

    env = {
        **os.environ,
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "SLURM_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": "0",
        "COMM_SUITE_CONTROL_PYTHON": sys.executable,
        "MOCK_ARGV": str(tmp_path / "argv"),
    }
    for name, task, command in [
        ("02_smoke.sbatch", "pcp", "smoke"),
        ("03_qualify.sbatch", "mapdn", "qualify"),
        ("04_train_pcp.sbatch", None, "train-row"),
        ("05_train_mapdn.sbatch", None, "train-row"),
        ("06_close_training.sbatch", "pcp", "close-training"),
        ("07_evaluate_pcp.sbatch", None, "evaluate-row"),
        ("08_evaluate_mapdn.sbatch", None, "evaluate-row"),
        ("09_analyze.sbatch", None, "analyze"),
    ]:
        spool = tmp_path / "spooled_script"
        spool.write_text((ROOT / "slurm/communication_comparison_v1" / name).read_text())
        args = ["bash", str(spool), str(profile), str(root)] + ([task] if task else [])
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        arguments = (tmp_path / "argv").read_text().splitlines()
        assert arguments[1] == sys.executable
        assert arguments[2] == str(ROOT / "scripts/communication_suite.py")
        assert arguments[3] == command
        assert str(root) in arguments
        help_result = subprocess.run(
            [sys.executable, arguments[2], command, "--help"], capture_output=True, text=True
        )
        assert help_result.returncode == 0, help_result.stderr
