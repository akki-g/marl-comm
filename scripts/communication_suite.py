#!/usr/bin/env python3
"""Manifest-driven fixed-MAPPO communication suite. Final tests require closure."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser(
        "plan", help="Write a fresh deterministic allocation; no rollout/submission"
    )
    plan.add_argument(
        "--suite", type=Path, default=ROOT / "configs/experiments/pcp_mapdn_mechanisms_v1.yaml"
    )
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--data-path", type=Path)
    plan.add_argument("--include", help="Explicit capacity,mapdn_topology supplements")
    plan.add_argument("--sweep", type=Path, help="Optional strict suite allocation descriptor")
    for name in (
        "preflight",
        "smoke",
        "qualify",
        "freeze",
        "train-row",
        "close-training",
        "evaluate-row",
        "analyze",
        "status",
        "retry-plan",
        "reconcile",
        "submit",
    ):
        q = sub.add_parser(name)
        q.add_argument("--run-root", type=Path, required=True)
        if name in (
            "preflight",
            "smoke",
            "qualify",
            "train-row",
            "close-training",
            "evaluate-row",
            "retry-plan",
            "reconcile",
        ):
            q.add_argument("--task", choices=["pcp", "mapdn"], required=True)
        if name in ("smoke", "qualify", "train-row", "evaluate-row", "reconcile"):
            q.add_argument("--index", type=int, required=True)
        if name in ("smoke", "qualify", "train-row", "evaluate-row", "submit"):
            q.add_argument("--retry-plan", type=Path)
        if name in ("preflight", "submit"):
            q.add_argument("--profile", type=Path, required=True)
        if name == "preflight":
            q.add_argument(
                "--local-check",
                action="store_true",
                help="Local evidence never qualifies a cluster",
            )
        if name in ("retry-plan", "reconcile"):
            q.add_argument("--reason", required=True)
            q.add_argument("--stage", choices=["main", "smoke", "qualification"], default="main")
            q.add_argument("--kind", choices=["runs", "evaluation"], default="runs")
        if name == "status":
            q.add_argument("--scheduler", action="store_true")
        if name == "submit":
            q.add_argument(
                "--phase",
                choices=["prepare", "main", "close", "evaluate", "analyze"],
                required=True,
            )
            q.add_argument("--task", choices=["pcp", "mapdn", "all"], default="all")
            q.add_argument("--include")
            q.add_argument("--execute", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command
    if command == "submit":
        import runpy

        submit = runpy.run_path(str(ROOT / "src/commstudy/experiments/mechanism_slurm.py"))[
            "submit"
        ]

        result = submit(
            args.run_root,
            args.profile,
            phase=args.phase,
            task=args.task,
            execute=args.execute,
            include=args.include.split(",") if args.include else None,
            retry=args.retry_plan,
        )
    else:
        import torch

        torch.set_num_interop_threads(1)
        from commstudy.experiments import mechanism_suite as suite

        if command == "plan":
            include = args.include.split(",") if args.include else None
            if args.sweep:
                from omegaconf import OmegaConf

                descriptor = OmegaConf.to_container(OmegaConf.load(args.sweep), resolve=True)
                if (
                    set(descriptor) != {"schema_version", "suite", "include", "note"}
                    or descriptor["schema_version"] != 1
                ):
                    raise ValueError("Unknown sweep descriptor")
                if include is not None and include != descriptor["include"]:
                    raise ValueError("Conflicting supplement selection")
                if (ROOT / descriptor["suite"]).resolve() != args.suite.resolve():
                    raise ValueError("Sweep descriptor and selected suite disagree")
                include = descriptor["include"]
            result = suite.plan(args.suite, args.out, data_path=args.data_path, include=include)
        elif command == "freeze":
            result = suite.freeze(args.run_root)
        elif command in (
            "preflight",
            "smoke",
            "qualify",
            "train-row",
            "close-training",
            "evaluate-row",
        ):
            from commstudy.experiments import mechanism_execution as run

            if command == "preflight":
                result = run.preflight(
                    args.run_root, args.task, args.profile, local=args.local_check
                )
            elif command == "close-training":
                result = run.close_training(args.run_root, args.task)
            elif command == "evaluate-row":
                result = run.evaluate(args.run_root, args.task, args.index, retry=args.retry_plan)
            else:
                stage = {"smoke": "smoke", "qualify": "qualification", "train-row": "main"}[command]
                result = run.train(
                    args.run_root, args.task, stage, args.index, retry=args.retry_plan
                )
        elif command in ("status", "analyze"):
            from commstudy.analysis.mechanism_report import analyze, status

            result = (
                analyze(args.run_root)
                if command == "analyze"
                else status(args.run_root, scheduler=args.scheduler)
            )
        elif command == "retry-plan":
            from commstudy.experiments.mechanism_slurm import retry_plan

            result = retry_plan(
                args.run_root, args.task, args.reason, stage=args.stage, kind=args.kind
            )
        elif command == "reconcile":
            from commstudy.experiments.mechanism_slurm import reconcile

            result = reconcile(
                args.run_root, args.task, args.stage, args.index, args.kind, args.reason
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
