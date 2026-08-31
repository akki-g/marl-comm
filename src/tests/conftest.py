from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def config_root() -> Path:
    """
    The project's ``configs/`` directory, as passed to
    :func:`commstudy.experiments.load_experiment_spec`.
    """
    root = REPO_ROOT / "configs"

    assert root.is_dir(), f"configs directory not found at {root}"

    return root
