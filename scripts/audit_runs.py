from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

from commstudy.analysis import audit_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit finished managed runs for non-finite logged values and, "
            "optionally, frozen-policy action saturation."
        )
    )
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument(
        "--run-id",
        action="append",
        help="Audit only these run IDs. Repeatable. Defaults to every completed run.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Audit only runs whose run ID contains this token. Repeatable.",
    )
    parser.add_argument("--out", type=Path, help="Write the report rows to this CSV.")
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Skip the checkpoint rollout and report metrics health only.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--saturation-threshold", type=float, default=0.999)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    suite_dir = args.suite_dir.resolve()

    run_dirs = sorted(
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "metadata.json").exists()
    )
    if args.run_id:
        wanted = set(args.run_id)
        run_dirs = [path for path in run_dirs if path.name in wanted]
    if args.model:
        run_dirs = [
            path for path in run_dirs if any(token in path.name for token in args.model)
        ]
    if not run_dirs:
        raise SystemExit(f"No matching runs under {suite_dir}")

    reports = []
    with tempfile.TemporaryDirectory(prefix="commstudy-audit-") as scratch:
        for run_dir in run_dirs:
            report = audit_run(
                run_dir,
                config_root=repo_root / "configs",
                scratch_root=None if args.no_policy else Path(scratch) / run_dir.name,
                episodes=args.episodes,
                steps=args.steps,
                seed=args.seed,
                saturation_threshold=args.saturation_threshold,
            )
            reports.append(report)
            print(json.dumps(report, indent=2, sort_keys=True, default=str))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for report in reports for key in report})
        with args.out.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            for report in reports:
                writer.writerow({key: report.get(key) for key in keys})
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
