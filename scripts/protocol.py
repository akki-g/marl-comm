"""Inspect protocols and check evidence before submitting or running a worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.protocols import (
    STAGES,
    content_sha256,
    load_protocol,
    ProtocolGateError,
)
from commstudy.experiments.provenance import source_fingerprint
from commstudy.experiments.sweeps import read_manifest, select_plans, validate_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("protocol", type=Path)
    template = commands.add_parser("approval-template")
    template.add_argument("protocol", type=Path)
    template.add_argument("--stage", required=True, choices=sorted(STAGES))
    template.add_argument("--out", required=True, type=Path)
    check = commands.add_parser("check")
    check.add_argument("--manifest", required=True, type=Path)
    check.add_argument("--stage", required=True, choices=sorted(STAGES))
    check.add_argument("--index", type=int)
    check.add_argument(
        "--runtime",
        action="store_true",
        help="Also verify the current worker's package/device/thread constraints.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if args.command in {"inspect", "approval-template"}:
            protocol = load_protocol(args.protocol)
            digest = content_sha256(protocol)
            if args.command == "inspect":
                print(
                    json.dumps(
                        {
                            "protocol_id": protocol["protocol_id"],
                            "status": protocol["status"],
                            "protocol_sha256": digest,
                            "source_sha256": source_fingerprint(repo_root),
                            "selection": protocol["selection"],
                            "runtime": protocol["runtime"],
                        },
                        indent=2,
                    )
                )
            else:
                if args.out.exists():
                    raise ProtocolGateError(f"Refusing to replace an existing review: {args.out}")
                atomic_write_json(
                    args.out,
                    {
                        "schema_version": 1,
                        "protocol_id": protocol["protocol_id"],
                        "protocol_sha256": digest,
                        "source_sha256": source_fingerprint(repo_root),
                        "stage": args.stage,
                        "decision": "pending",
                        "reviewer": "",
                        "rationale": "",
                        "evidence": [],
                        "validated_factors": [],
                    },
                )
                print(f"Wrote pending review template: {args.out}. This does not authorize launch.")
            return 0
        plans = select_plans(read_manifest(args.manifest), index=args.index)
        if not plans:
            raise ProtocolGateError("Manifest contains no launchable rows.")
        for plan in plans:
            if plan.protocol is None or plan.protocol["stage"] != args.stage:
                raise ProtocolGateError("Manifest has no matching protocol launch-stage binding.")
            validate_plan(
                plan,
                config_root=repo_root / "configs",
                repo_root=repo_root,
                check_runtime=args.runtime,
            )
        print(f"Protocol preflight passed for {len(plans)} {args.stage} rows.")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"PROTOCOL GATE REJECTED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
