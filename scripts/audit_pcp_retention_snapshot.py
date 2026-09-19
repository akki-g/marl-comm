"""CPUv3 wrapper for the unchanged, hash-bound Identity replay snapshot auditor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.protocols import load_protocol
from commstudy.experiments.provenance import source_fingerprint

SOURCE = "7124976af366d28d40a3ccf0ebce62aa2b4019783d13974b9f98dd1c316af262"
AUDITOR_SHA = "17bcdde41492d1a1efaabd45daab0252b99fb2bb28b98cec4dcff79d95184403"


def original_auditor():
    path = Path(__file__).with_name("audit_pcp_replay_snapshot.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != AUDITOR_SHA:
        raise ValueError("The independently reviewed CPUv2 snapshot auditor changed.")
    spec = importlib.util.spec_from_file_location("pcp_immutable_snapshot_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    for name in ("previous-suite", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(
        args.out.resolve().is_relative_to(p.resolve())
        for p in (args.suite_dir, args.previous_suite)
    ):
        parser.error("Derived outputs must remain outside both audited suites.")
    report = {
        "schema_version": 1,
        "audit": "pcp_retention_snapshot_v3",
        "runs": [],
        "issues": [],
        "approval_decision": "not_issued",
        "underlying_auditor_sha256": AUDITOR_SHA,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        auditor = original_auditor()
        root = Path(__file__).resolve().parents[1]
        if source_fingerprint(root) != SOURCE:
            raise ValueError("Run this audit from the frozen CPUv3 execution copy.")
        protocol = load_protocol(args.protocol)
        if (
            protocol["protocol_id"] != "pcp_corrected_cpu_v3"
            or protocol["base_spec"]["experiment"]["keep_checkpoints_num"] is not None
        ):
            raise ValueError("Expected the CPUv3 retention-only protocol.")
        runs = sorted(path.parent for path in args.suite_dir.glob("*/metadata.json"))
        seeds = [json.loads((run / "metadata.json").read_text()).get("seed") for run in runs]
        if sorted(seeds) != [10, 11] or protocol["selection"]["confirmation_seeds"] != [10, 11]:
            raise ValueError("Require exactly declared seeds 10 and 11.")
        with tempfile.TemporaryDirectory(prefix="pcp-retention-snapshot-") as temporary:
            for run in runs:
                try:
                    report["runs"].append(
                        auditor.audit_run(
                            run,
                            args.previous_suite,
                            protocol,
                            args.protocol,
                            root,
                            Path(temporary) / run.name,
                        )
                    )
                except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
                    report["runs"].append(
                        {"run_id": run.name, "checks_passed": False, "error": str(error)}
                    )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        report["issues"].append(str(error))
    report["checks_passed"] = (
        not report["issues"]
        and bool(report["runs"])
        and all(run["checks_passed"] for run in report["runs"])
    )
    sidecar = args.out.with_suffix(".provenance.json")
    atomic_write_json(
        sidecar,
        {
            run["run_id"]: {"snapshot_sha256": run["snapshot_sha256"], **run["provenance"]}
            for run in report["runs"]
            if run["checks_passed"]
        },
    )
    report["provenance_sidecar"] = {
        "path": str(sidecar.resolve()),
        "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
    atomic_write_json(args.out, report)
    print(json.dumps({"checks_passed": report["checks_passed"], "report": str(args.out)}))
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
