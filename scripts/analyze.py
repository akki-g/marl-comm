from __future__ import annotations

import argparse
from pathlib import Path

from commstudy.analysis import analyze_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate a managed commstudy suite.")
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    outputs = analyze_suite(
        args.suite_dir,
        results_dir=args.results_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        plots=not args.no_plots,
    )
    print(outputs.summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
