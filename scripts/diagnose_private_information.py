#!/usr/bin/env python3
"""Freeze, then execute the finite constructed-state information diagnostic."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

from commstudy.analysis.private_information import (
    default_plan, file_digest, json_digest, prepare_mapdn_inputs,
    run_mapdn_diagnostic, run_pcp_diagnostic, source_inventory, verify_plan,
)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def runtime_identity():
    return {"executable": sys.executable, "python": sys.version,
            "platform": platform.platform(),
            "distributions": sorted([distribution.metadata["Name"], distribution.version]
                                    for distribution in importlib.metadata.distributions())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "execute"])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--environment", choices=["pcp", "mapdn", "both"], default="both")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out_dir.resolve()
    if args.stage == "prepare":
        if args.data_path is None:
            parser.error("prepare requires --data-path for the fresh case33 manifest")
        output.mkdir(parents=True, exist_ok=False)
        plan = default_plan()
        plan["mapdn_inputs"] = prepare_mapdn_inputs(output, args.data_path)
        plan["source_root"] = str(root)
        plan["source_inventory"] = source_inventory(root)
        plan["runtime"] = runtime_identity()
        plan["prepared_at"] = datetime.now(UTC).isoformat()
        plan["sha256"] = json_digest(plan)
        write_new(output / "plan.json", plan)
        (output / "frozen_plan.md").write_bytes(
            (root / "docs/PRIVATE_INFORMATION_DIAGNOSTIC_PLAN.md").read_bytes())
        print(json.dumps({"plan_sha256": plan["sha256"], "output": str(output)}), flush=True)
        return
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text())
    if args.expected_plan_sha256 != plan["sha256"]:
        parser.error("execute requires the exact --expected-plan-sha256 printed by prepare")
    verify_plan(plan, root)
    if runtime_identity() != plan["runtime"]:
        raise ValueError("Diagnostic runtime changed after prospective preparation.")
    execution = output / f"execution_{args.environment}"
    execution.mkdir(exist_ok=False)
    started = time.monotonic()
    write_new(execution / "start.json", {
        "started_at": datetime.now(UTC).isoformat(),
        "plan_sha256": plan["sha256"], "environment": args.environment,
        "runtime": plan["runtime"], "execution_source_root": str(root),
        "original_preparation_source_root": plan["source_root"],
    })
    receipt = {"plan_sha256": plan["sha256"], "artifacts": {}, "complete": False}
    try:
        for name, function in (("pcp", run_pcp_diagnostic), ("mapdn", run_mapdn_diagnostic)):
            if args.environment not in {name, "both"}:
                continue
            result = function(plan)
            path = execution / f"{name}_diagnostic.json"
            write_new(path, result)
            receipt["artifacts"][path.name] = file_digest(path)
        verify_plan(plan, root)
        receipt["complete"] = True
    except BaseException as exc:
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipt["source_inventory_unchanged"] = source_inventory(root) == plan["source_inventory"]
        write_new(execution / "receipt.json", receipt)
    print(json.dumps({"complete": True, "execution": str(execution)}), flush=True)


if __name__ == "__main__":
    main()
