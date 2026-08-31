from __future__ import annotations

import sys
from pathlib import Path

from commstudy.experiments import (
    load_experiment_spec,
    run_experiment,
)


def main() -> None:
    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    config_root = (
        repo_root
        / "configs"
    )

    spec = load_experiment_spec(
        config_root=config_root,
        overrides=sys.argv[1:],
    )

    run_experiment(spec)


if __name__ == "__main__":
    main()