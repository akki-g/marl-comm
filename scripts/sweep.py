from __future__ import annotations

import argparse
from pathlib import Path

from commstudy.experiments.sweeps import (
    DEFAULT_STALE_AFTER_SECONDS,
    create_combined_manifest,
    create_manifest,
    execute_plan,
    format_plan_table,
    manifest_status_counts,
    read_manifest,
    select_plans,
    sync_manifest_status,
    write_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or execute a commstudy sweep.")
    parser.add_argument("suite", nargs="*", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--combine-out",
        type=Path,
        metavar="PATH",
        help=(
            "Expand every given suite into one combined manifest at PATH so a "
            "multi-suite study is a single Slurm array. Per-suite manifests are "
            "still written."
        ),
    )
    parser.add_argument("--run", action="store_true", help="Execute selected rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print without training.")
    parser.add_argument("--index", type=int, help="Zero-based manifest/Slurm array index.")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="With --index, execute this many consecutive rows (array chunking).",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--reclaim-stale",
        nargs="?",
        type=float,
        const=DEFAULT_STALE_AFTER_SECONDS,
        default=None,
        metavar="SECONDS",
        help=(
            "Retry rows stuck in 'running' whose worker stopped reporting for "
            f"this long (default {DEFAULT_STALE_AFTER_SECONDS:.0f}s). Use after "
            "preemption or a walltime kill."
        ),
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print a status summary of the manifest and exit.",
    )
    parser.epilog = (
        "Trailing KEY=VALUE arguments are appended to every executed row as "
        "machine-level overrides, e.g. experiment.train_device=cuda. They are "
        "recorded in each run's metadata so provenance stays truthful."
    )
    return parser


def _partition_overrides(arguments: list[str]) -> tuple[list[str], list[str]]:
    """Split trailing KEY=VALUE overrides out before argparse sees them.

    The suite path is an optional positional, so argparse would otherwise
    consume the first override as the suite and fail confusingly. A token is an
    override when it contains '=' and does not start with '-'.
    """

    parsed: list[str] = []
    overrides: list[str] = []
    for token in arguments:
        if not token.startswith("-") and "=" in token and not token.endswith((".yaml", ".yml")):
            overrides.append(token)
        else:
            parsed.append(token)
    return parsed, overrides


def main(argv: list[str] | None = None) -> int:
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed_arguments, extra_overrides = _partition_overrides(arguments)
    parser = _parser()
    args = parser.parse_args(parsed_arguments)
    if not args.suite and args.manifest is None:
        parser.error("provide one or more suite YAMLs, or --manifest")
    if args.suite and args.manifest is not None:
        parser.error("provide suite YAMLs or --manifest, not both")
    if len(args.suite) > 1 and args.combine_out is None:
        parser.error("multiple suites require --combine-out")
    if args.run and args.dry_run:
        parser.error("--run and --dry-run are mutually exclusive")
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.count > 1 and args.index is None:
        parser.error("--count requires --index")

    repo_root = Path(__file__).resolve().parents[1]
    if args.suite and args.combine_out is not None:
        manifest_path, plans = create_combined_manifest(
            [path.resolve() for path in args.suite],
            args.combine_out.resolve(),
            repo_root=repo_root,
        )
        print(f"{manifest_path}: {len(plans)} rows from {len(args.suite)} suites")
        print(f"array range: 0-{len(plans) - 1}")
    elif args.suite:
        manifest_path, plans = create_manifest(args.suite[0].resolve(), repo_root=repo_root)
        print(f"{manifest_path}: {len(plans)} rows")
        print(f"array range: 0-{len(plans) - 1}")
    else:
        manifest_path = args.manifest.resolve()
        # Array workers treat the manifest as immutable input and write only
        # their isolated status.json. A later unselected invocation syncs it.
        plans = (
            read_manifest(manifest_path)
            if args.index is not None or args.run_id is not None
            else sync_manifest_status(manifest_path, stale_after=args.reclaim_stale)
        )

    if args.status:
        counts = manifest_status_counts(plans)
        print(f"{manifest_path}: {len(plans)} rows")
        for status, count in counts.items():
            print(f"  {status:<10} {count}")
        return 0

    selected = select_plans(plans, index=args.index, run_id=args.run_id, count=args.count)
    print(format_plan_table(selected))
    if not args.run:
        return 0

    updated_by_id = {plan.run_id: plan for plan in plans}
    exit_code = 0
    for plan in selected:
        result, _ = execute_plan(
            plan,
            config_root=repo_root / "configs",
            repo_root=repo_root,
            retry_failed=args.retry_failed,
            reclaim_stale_after=args.reclaim_stale,
            extra_overrides=extra_overrides,
        )
        updated_by_id[result.run_id] = result
        print(f"[sweep] {result.run_id} -> {result.status}", flush=True)
        if result.status == "failed":
            exit_code = 1
            if args.stop_on_failure:
                break

    if args.index is None and args.run_id is None:
        write_manifest(manifest_path, list(updated_by_id.values()))
        sync_manifest_status(manifest_path, stale_after=args.reclaim_stale)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
