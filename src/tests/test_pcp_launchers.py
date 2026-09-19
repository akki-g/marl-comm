"""Execute submission wrappers with fake schedulers; never submit or train."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from commstudy.experiments.sweeps import expand_suite_config, write_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = [
    ('pcp_02b_protocol.sbatch', 'confirmation',
     'runs/pcp_candidate_confirmation_corrected_v1/manifest.csv'),
    ('pcp_03_main_comparison.sbatch', 'comparison',
     'runs/pcp_comm_main_corrected_v1/manifest.csv'),
    ('pcp_04_ablations.sbatch', 'ablation',
     'runs/_manifests/pcp_ablations_corrected_v1.csv'),
]
SCRIPTS = ['pcp_01_setup.sbatch', 'pcp_02_pilot.sbatch', *[item[0] for item in LAUNCHERS]]


@pytest.fixture
def launcher_case(tmp_path):
    root = tmp_path / 'launcher_repository'
    for name in ('configs', 'slurm'):
        shutil.copytree(REPO_ROOT / name, root / name)
    shutil.copytree(REPO_ROOT / 'src' / 'commstudy', root / 'src' / 'commstudy',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'scripts').mkdir()
    shutil.copy2(REPO_ROOT / 'scripts' / 'protocol.py', root / 'scripts' / 'protocol.py')
    fake_bin = tmp_path / 'fake_scheduler'
    fake_bin.mkdir()
    log = tmp_path / 'scheduler_invocations.txt'
    for command in ('sbatch', 'srun'):
        executable = fake_bin / command
        executable.write_text(
            '#!/bin/sh\n'
            f'printf "{command} %s\\n" "$*" >> "$COMMSTUDY_TEST_SCHEDULER_LOG"\n'
            'exit 0\n'
        )
        executable.chmod(0o755)
    env = dict(os.environ)
    # Ensure a developer's surrounding allocation cannot accidentally choose
    # the worker branch when this test specifically exercises login submission.
    for key in list(env):
        if key.startswith('SLURM_'):
            env.pop(key)
    env.update({
        'COMMSTUDY_REPO': str(root), 'COMMSTUDY_PYTHON': sys.executable,
        'COMMSTUDY_VENV': str(REPO_ROOT / '.venv'),
        'COMMSTUDY_TEST_SCHEDULER_LOG': str(log),
        'PYTHONPATH': str(root / 'src'),
        'PATH': str(fake_bin) + os.pathsep + env['PATH'],
    })

    def run(name, *, worker=False, python=None, spooled=False):
        process_env = dict(env)
        script = root / 'slurm' / name
        if worker:
            process_env.update(SLURM_JOB_ID='fixture', SLURM_ARRAY_TASK_ID='0',
                               SLURM_CPUS_PER_TASK='4')
        if spooled:
            # Real Slurm runs its own copied script, so BASH_SOURCE cannot
            # locate the checkout; SLURM_SUBMIT_DIR remains authoritative.
            process_env.pop('COMMSTUDY_REPO')
            process_env['SLURM_SUBMIT_DIR'] = str(root)
            script = tmp_path / 'scheduler spool' / name / 'slurm_script'
            script.parent.mkdir(parents=True)
            shutil.copy2(root / 'slurm' / name, script)
        if python is not None:
            process_env['COMMSTUDY_PYTHON'] = str(python)
        return subprocess.run(['bash', str(script)], env=process_env,
                              cwd=tmp_path, capture_output=True, text=True, timeout=60)

    def manifest(stage, relative_path):
        suite = {
            'suite_id': 'launcher_fixture',
            'protocol': 'configs/protocols/pcp_corrected_v1.yaml',
            'stage': stage,
            'models': ['pcp_comm_identity' if stage == 'confirmation' else 'pcp_comm_attention'],
            'seeds': [10 if stage == 'confirmation' else 20],
            'ablation': 'message_dim' if stage == 'ablation' else 'main',
        }
        if stage == 'ablation':
            suite['ablation_value'] = '8'
        plans = expand_suite_config(suite, repo_root=root)
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(destination, plans)
        return destination

    return SimpleNamespace(root=root, log=log, run=run, manifest=manifest, env=env)


@pytest.mark.parametrize('name', SCRIPTS)
def test_pcp_launchers_have_valid_bash_syntax(name):
    completed = subprocess.run(['bash', '-n', str(REPO_ROOT / 'slurm' / name)],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('name,stage,manifest', LAUNCHERS)
@pytest.mark.parametrize('missing', ['manifest', 'approval'])
def test_login_preflight_rejection_prevents_sbatch(launcher_case, name, stage, manifest, missing):
    case = launcher_case
    if missing == 'approval':
        case.manifest(stage, manifest)
    completed = case.run(name)
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert 'PROTOCOL GATE REJECTED' in output, output
    if missing == 'approval':
        assert 'approval is missing' in output, output
    else:
        assert 'No such file' in output or 'missing' in output, output
    assert not case.log.exists(), 'Rejected preflight reached a scheduler command.'


@pytest.mark.parametrize('name,stage,manifest', LAUNCHERS)
def test_worker_rechecks_missing_approval_before_srun(launcher_case, name, stage, manifest):
    case = launcher_case
    case.manifest(stage, manifest)
    completed = case.run(name, worker=True)
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert 'approval is missing' in output, output
    assert not case.log.exists(), 'Worker executed a training/scheduler command after rejection.'


@pytest.mark.parametrize('name,stage,manifest', LAUNCHERS)
def test_spooled_worker_finds_submitted_repository_before_rechecking_approval(
    launcher_case, name, stage, manifest,
):
    case = launcher_case
    case.manifest(stage, manifest)
    completed = case.run(name, worker=True, spooled=True)
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert 'approval is missing' in output, output
    assert not case.log.exists()


@pytest.mark.parametrize('name,stage,manifest', LAUNCHERS)
def test_successful_preflight_is_required_before_the_fake_scheduler(
    launcher_case, name, stage, manifest,
):
    case = launcher_case
    fake_python = case.root / 'preflight_fixture'
    fake_python.write_text(
        '#!/bin/sh\n'
        'printf "preflight %s\\n" "$*" >> "$COMMSTUDY_TEST_SCHEDULER_LOG"\n'
        'exit 0\n'
    )
    fake_python.chmod(0o755)
    completed = case.run(name, python=fake_python)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = case.log.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == f'preflight scripts/protocol.py check --manifest {manifest} --stage {stage}'
    assert lines[1] == f'sbatch slurm/{name}'


def test_retired_pilot_cannot_submit_or_train(launcher_case):
    case = launcher_case
    completed = case.run('pcp_02_pilot.sbatch')
    assert completed.returncode != 0
    assert 'retired' in completed.stderr
    assert not case.log.exists()


def test_setup_only_plans_and_never_submits_training(launcher_case):
    case = launcher_case
    fake_python = case.root / 'planner_fixture'
    fake_python.write_text(
        '#!/bin/sh\n'
        'printf "plan %s\\n" "$*" >> "$COMMSTUDY_TEST_SCHEDULER_LOG"\n'
        'exit 0\n'
    )
    fake_python.chmod(0o755)
    completed = case.run('pcp_01_setup.sbatch', python=fake_python)
    assert completed.returncode == 0, completed.stderr
    commands = case.log.read_text().splitlines()
    assert len(commands) == 3
    assert all(command.startswith('plan scripts/sweep.py ') for command in commands)
    assert not any('--run' in command.split() for command in commands)
    assert 'pcp_candidate_confirmation.yaml' in commands[0]
    assert 'pcp_comm_main.yaml' in commands[1]
    assert '--combine-out runs/_manifests/pcp_ablations_corrected_v1.csv' in commands[2]
