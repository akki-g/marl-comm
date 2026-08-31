from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

from commstudy.analysis.diagnostics import load_frozen_experiment
from commstudy.analysis.saliency import communication_saliency
from commstudy.experiments.bookkeeping import atomic_write_json, utc_now


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure communication saliency for finished runs: the paired "
            "evaluation-return difference between the trained policy and the "
            "same policy with every message suppressed."
        )
    )
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--run-id", action="append", help="Restrict to these run IDs.")
    parser.add_argument("--model", action="append", help="Restrict to run IDs containing this.")
    parser.add_argument("--out", type=Path, help="Write rows to this CSV.")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exploration",
        default="DETERMINISTIC",
        choices=("DETERMINISTIC", "RANDOM"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    suite_dir = args.suite_dir.resolve()

    run_dirs = sorted(
        path
        for path in suite_dir.iterdir()
        if path.is_dir()
        and (path / "metadata.json").exists()
        and (path / "checkpoints" / "policy_state.pt").exists()
    )
    if args.run_id:
        wanted = set(args.run_id)
        run_dirs = [path for path in run_dirs if path.name in wanted]
    if args.model:
        run_dirs = [path for path in run_dirs if any(t in path.name for t in args.model)]
    if not run_dirs:
        raise SystemExit(f"No completed runs with saved actors under {suite_dir}")

    rows = []
    with tempfile.TemporaryDirectory(prefix="commstudy-saliency-") as scratch:
        for run_dir in run_dirs:
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            experiment = load_frozen_experiment(
                run_dir,
                config_root=repo_root / "configs",
                scratch_root=Path(scratch) / run_dir.name,
            )
            try:
                result = communication_saliency(
                    experiment,
                    episodes=args.episodes,
                    steps=args.steps,
                    exploration=args.exploration,
                    seed=args.seed,
                )
            finally:
                close = getattr(experiment.test_env, "close", None)
                if callable(close):
                    close()

            row = {
                "run_id": metadata["run_id"],
                "suite_id": metadata["suite_id"],
                "task": metadata["task"],
                "algorithm": metadata["algorithm"],
                "model": metadata["model"],
                "seed": int(metadata["seed"]),
                "ablation": metadata.get("ablation") or "main",
                "ablation_value": metadata.get("ablation_value") or "",
                **result.as_row(),
            }
            rows.append(row)
            # Keep the measurement beside the run it describes so aggregation
            # picks it up automatically and never depends on a separate file
            # being passed around by hand.
            atomic_write_json(
                run_dir / "saliency.json",
                {
                    "schema_version": 1,
                    "run_id": metadata["run_id"],
                    "measured_at": utc_now(),
                    "settings": {
                        "episodes": args.episodes,
                        "steps": args.steps,
                        "seed": args.seed,
                        "exploration": args.exploration,
                    },
                    "per_episode_returns_with_comm": list(result.per_episode_returns_with),
                    "per_episode_returns_without_comm": list(
                        result.per_episode_returns_without
                    ),
                    **result.as_row(),
                },
            )
            print(
                f"{row['model']:<16} seed {row['seed']} "
                f"with={result.return_with_comm:9.2f} "
                f"without={result.return_without_comm:9.2f} "
                f"saliency={result.return_delta:+8.2f} "
                f"action_shift={result.action_shift_mean}",
                flush=True,
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in rows for key in row})
        with args.out.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
