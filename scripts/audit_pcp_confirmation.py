"""Create source-bound action and domain evidence for completed PCP candidates."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from commstudy.analysis.diagnostics import audit_run, load_frozen_experiment
from commstudy.analysis.pcp_domain import policy_domain_audit
from commstudy.experiments.bookkeeping import atomic_write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--run-id", action="append")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    runs = sorted(path.parent for path in args.suite_dir.glob("*/metadata.json"))
    if args.run_id:
        requested = set(args.run_id)
        found = {path.name for path in runs}
        if requested - found:
            parser.error(f"Requested runs do not exist: {sorted(requested - found)}")
        runs = [path for path in runs if path.name in requested]
    if not runs:
        parser.error("No managed candidate runs found.")
    actions, domain, failures = {}, {}, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pcp-confirmation-audit-") as temporary:
        scratch = Path(temporary)
        for run in runs:
            try:
                status = json.loads((run / "status.json").read_text())
                if status.get("status") != "completed":
                    raise ValueError(f"Run status is {status.get('status')!r}, not completed.")
                actions[run.name] = audit_run(
                    run,
                    config_root=root / "configs",
                    scratch_root=scratch / run.name / "actions",
                    group="adversary",
                    episodes=args.episodes,
                    seed=args.seed,
                )
                if not actions[run.name]["policy_audited"]:
                    raise ValueError("Completed candidate has no audited actor checkpoint.")
                experiment = load_frozen_experiment(
                    run,
                    scratch_root=scratch / run.name / "domain",
                )
                try:
                    domain[run.name] = policy_domain_audit(
                        experiment, episodes=args.episodes, seed=args.seed,
                    )
                finally:
                    experiment.close()
                print(f"Audited {run.name}: action and domain evidence recorded.")
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                failures[run.name] = f"{type(error).__name__}: {error}"
                print(f"AUDIT FAILED {run.name}: {error}")
            finally:
                # Preserve completed evidence even when a later run fails.
                atomic_write_json(args.out_dir / "action_audits.json", actions)
                atomic_write_json(args.out_dir / "domain_audits.json", domain)
    atomic_write_json(args.out_dir / "audit_failures.json", failures)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
