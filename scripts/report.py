from __future__ import annotations

import argparse
from pathlib import Path

from commstudy.analysis.report import write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render an analyzed suite as a single readable markdown report."
    )
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args(argv)

    suite_dir = args.suite_dir.resolve()
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else suite_dir.parents[1] / "results" / suite_dir.name
    )
    print(write_report(suite_dir, results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
