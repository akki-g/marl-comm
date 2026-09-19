"""Check bounded PCP confirmation artifacts and produce a review-required report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commstudy.analysis.confirmation import confirm_pcp_suite, file_sha256
from commstudy.experiments.bookkeeping import atomic_write_json


def _audits(path: Path) -> dict:
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or any(
        not isinstance(name, str) or not isinstance(report, dict)
        for name, report in document.items()
    ):
        raise ValueError(f"{path}: expected a JSON object mapping run IDs to audit objects.")
    return document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol", type=Path, default=root / "configs/protocols/pcp_corrected_v1.yaml"
    )
    parser.add_argument(
        "--action-audits",
        type=Path,
        required=True,
        help="JSON mapping run IDs to audit_run reports; both modes, 32 episodes, seed 100000.",
    )
    parser.add_argument(
        "--domain-audits",
        type=Path,
        required=True,
        help="JSON mapping run IDs to policy_domain_audit reports.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = confirm_pcp_suite(
            args.suite_dir,
            protocol_path=args.protocol,
            repo_root=root,
            action_audits=_audits(args.action_audits),
            domain_audits=_audits(args.domain_audits),
        )
        report["audit_artifacts"] = {
            "action": {
                "path": str(args.action_audits.resolve()),
                "sha256": file_sha256(args.action_audits),
            },
            "domain": {
                "path": str(args.domain_audits.resolve()),
                "sha256": file_sha256(args.domain_audits),
            },
        }
    except (OSError, ValueError, TypeError) as error:
        report = {
            "schema_version": 1,
            "checks_passed": False,
            "review_status": "blocked_incomplete_or_invalid",
            "approval_decision": "not_issued",
            "issues": [{"check": "input_artifact", "detail": str(error)}],
        }
    atomic_write_json(args.out, report)
    print(
        json.dumps(
            {
                "checks_passed": report["checks_passed"],
                "review_status": report["review_status"],
                "report": str(args.out),
            },
            indent=2,
        )
    )
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
