from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from commstudy.experiments import (
    RunAlreadyCompletedError,
    RunContext,
    load_experiment_spec,
    make_run_id,
    retry_context,
    run_managed_experiment,
)
from commstudy.experiments.bookkeeping import to_serializable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one reproducibly managed BenchMARL experiment."
    )
    parser.add_argument("--suite-id", default="adhoc")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--ablation", default="main")
    parser.add_argument("--ablation-value")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args, overrides = parser.parse_known_args(arguments)
    invalid = [override for override in overrides if "=" not in override]
    if invalid:
        parser.error(f"invalid configuration overrides: {invalid}")

    repo_root = Path(__file__).resolve().parents[1]
    config_root = repo_root / "configs"
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    spec = load_experiment_spec(
        config_root=config_root,
        overrides=overrides,
    )
    run_id = args.run_id or make_run_id(
        task=spec.task,
        algorithm=spec.algorithm,
        model=spec.model,
        seed=spec.seed,
        ablation=args.ablation,
        ablation_value=args.ablation_value,
    )
    context = RunContext(
        suite_id=args.suite_id,
        run_id=run_id,
        output_root=output_root.resolve(),
        command=tuple([sys.executable, str(Path(__file__).resolve()), *arguments]),
        ablation=args.ablation,
        ablation_value=args.ablation_value,
    )
    if args.retry and context.run_dir.exists():
        context = retry_context(context)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_dir": str(context.run_dir),
                    "run_id": context.run_id,
                    "suite_id": context.suite_id,
                    "resolved_spec": to_serializable(dataclasses.asdict(spec)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        run_managed_experiment(
            spec,
            context,
            repo_root=repo_root,
            overrides=overrides,
        )
    except RunAlreadyCompletedError:
        print(f"Skipping completed run: {context.run_id}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
