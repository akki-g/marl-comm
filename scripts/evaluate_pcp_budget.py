"""Persist strict full-horizon and held-out intervention evidence for one D4 row."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from collections.abc import Mapping

import torch

from commstudy.analysis.confirmation import _validate_run, file_sha256
from commstudy.analysis.diagnostics import audit_run, load_frozen_experiment
from commstudy.analysis.pcp_budget import evaluate_budget_policy
from commstudy.analysis.pcp_domain import policy_domain_audit
from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.protocols import content_sha256, load_protocol
from commstudy.experiments.provenance import source_fingerprint
from commstudy.experiments.sweeps import read_manifest, select_plans, validate_plan


def retained_checkpoint_inventory(run):
    inventory = []
    expected_names = {f"checkpoint_{frame}.pt" for frame in range(120000, 600001, 120000)}
    native_paths = list(run.glob("benchmarl/*/checkpoints/checkpoint_*.pt"))
    if len(native_paths) != 5 or {path.name for path in native_paths} != expected_names:
        raise ValueError("Require exactly the five declared native checkpoint files.")
    signatures = []
    final_native_actors = {}
    for frames in range(120000, 600001, 120000):
        paths = list(run.glob(f"benchmarl/*/checkpoints/checkpoint_{frames}.pt"))
        if len(paths) != 1:
            raise ValueError(f"Require exactly one retained {frames}-frame native checkpoint.")
        path = paths[0]
        before = file_sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state"), Mapping):
            raise ValueError("Native checkpoint must contain a state header.")
        state = payload["state"]
        if (
            type(state.get("total_frames")) is not int
            or state["total_frames"] != frames
            or type(state.get("n_iters_performed")) is not int
            or state["n_iters_performed"] != frames // 6000
            or {key for key in payload if key.startswith("loss_")}
            != {"loss_adversary", "loss_agent"}
        ):
            raise ValueError(f"Native checkpoint header/groups differ at {frames} frames.")
        signature = {}
        for group in ("adversary", "agent"):
            saved = payload[f"loss_{group}"]
            if not isinstance(saved, Mapping):
                raise ValueError("Native checkpoint loss state must be a mapping.")
            tensors = {"actor": 0, "critic": 0}
            for name, value in saved.items():
                if isinstance(value, torch.Tensor):
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("Native checkpoint contains nonfinite loss-state tensors.")
                    for network in tensors:
                        tensors[network] += name.startswith(f"{network}_network_params.")
                    signature[group, name] = ("tensor", str(value.dtype), tuple(value.shape))
                elif name.endswith(".__batch_size") and type(value) is torch.Size:
                    signature[group, name] = ("batch_size", tuple(value))
                elif name.endswith(".__device") and value is None:
                    signature[group, name] = ("device", None)
                else:
                    raise ValueError("Native checkpoint contains unsupported loss-state metadata.")
            if not all(tensors.values()):
                raise ValueError("Native checkpoint omits actor or critic tensors for a group.")
            if frames == 600000:
                final_native_actors[group] = {
                    name.removeprefix("actor_network_params."): value
                    for name, value in saved.items()
                    if name.startswith("actor_network_params.") and isinstance(value, torch.Tensor)
                }
        if file_sha256(path) != before:
            raise ValueError("Native checkpoint changed while its contents were inspected.")
        signatures.append(signature)
        inventory.append(
            {
                "path": str(path.resolve()),
                "sha256": before,
                "frames": frames,
                "iterations": frames // 6000,
            }
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Native checkpoint state keys/dtypes/shapes change across the run.")
    managed_path = run / "checkpoints/policy_state.pt"
    managed_hash = file_sha256(managed_path)
    managed = torch.load(managed_path, map_location="cpu", weights_only=False)
    policies = managed.get("group_policies") if isinstance(managed, Mapping) else None
    if not isinstance(policies, Mapping) or set(policies) != set(final_native_actors):
        raise ValueError("Managed final checkpoint must retain the same native actor groups.")
    for group, native in final_native_actors.items():
        if not isinstance(policies[group], Mapping):
            raise ValueError("Managed final policy state must be a mapping.")
        saved = {name: value for name, value in policies[group].items()
                 if isinstance(value, torch.Tensor)}
        if set(native) != set(saved):
            raise ValueError("Final native and managed actor tensor keys differ.")
        for name, value in native.items():
            other = saved[name]
            if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                other.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            ):
                raise ValueError("Final native and managed actor tensors differ.")
    if file_sha256(managed_path) != managed_hash:
        raise ValueError("Managed final policy changed during native checkpoint inspection.")
    diagnostics = []
    for path in sorted(run.glob("benchmarl/*/replay_diagnostics/*.pt")):
        before = file_sha256(path)
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if (not isinstance(snapshot, Mapping) or snapshot.get("diagnostic_only") is not True
                or snapshot.get("not_resume") is not True):
            raise ValueError("Replay snapshot is missing its diagnostic-only contract.")
        if file_sha256(path) != before:
            raise ValueError("Replay snapshot changed while its contents were inspected.")
        diagnostics.append({"path": str(path.resolve()), "sha256": before})
    return {"schema_version": 1, "run_id": run.name, "checkpoints": inventory,
            "final_native_actor_matches_managed": True,
            "managed_final_policy_sha256": managed_hash,
            "replay_diagnostics": diagnostics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    plans = select_plans(read_manifest(args.manifest), run_id=args.run_id)
    if len(plans) != 1:
        parser.error("Exactly one declared comparison row is required.")
    plan = plans[0]
    validate_plan(plan, config_root=root / "configs", repo_root=root, check_runtime=True)
    if plan.protocol["stage"] != "comparison" or plan.protocol["ablation"] != "visibility":
        parser.error("Only the declared D4 visibility comparison is supported.")
    run = plan.output_root / plan.suite_id / plan.run_id
    if args.out_dir.resolve().is_relative_to(run.resolve()):
        parser.error("Derived evaluation outputs must remain outside the audited run.")
    protocol = load_protocol(root / plan.protocol["path"])
    metadata = json.loads((run / "metadata.json").read_text())
    if args.out_dir.exists():
        parser.error(
            "Refusing to overwrite an existing evaluation; preserve it and use a new path."
        )
    args.out_dir.mkdir(parents=True)
    try:
        checkpoint_inventory = retained_checkpoint_inventory(run)
        input_paths = [
            args.manifest, root / plan.protocol["path"],
            root / "docs/PCP_VISIBILITY_BUDGET_PLAN.md", Path(__file__),
            *(run / name for name in ("metadata.json", "status.json", "resolved_config.yaml",
                                     "metrics.csv", "checkpoints/policy_state.pt")),
            *(Path(row["path"]) for row in checkpoint_inventory["checkpoints"]),
            *(Path(row["path"]) for row in checkpoint_inventory["replay_diagnostics"]),
        ]
        before = {str(path.resolve()): file_sha256(path) for path in input_paths}
        source_before = source_fingerprint(root)
        atomic_write_json(
            args.out_dir / "retained_checkpoints.json",
            checkpoint_inventory,
        )
        with tempfile.TemporaryDirectory(prefix="pcp-budget-audit-") as temporary:
            scratch = Path(temporary)
            action = audit_run(
                run,
                config_root=root / "configs",
                scratch_root=scratch / "actions",
                group="adversary",
                episodes=32,
                seed=100000,
            )
            atomic_write_json(args.out_dir / "action_audit.json", action)
            experiment = load_frozen_experiment(run, scratch_root=scratch / "evaluation")
            try:
                domain = policy_domain_audit(experiment, episodes=32, seed=100000)
                atomic_write_json(args.out_dir / "domain_audit.json", domain)
                validation = _validate_run(
                    run,
                    metadata,
                    protocol,
                    content_sha256(protocol),
                    source_fingerprint(root),
                    action,
                    domain,
                    condition={
                        key: plan.protocol[key]
                        for key in ("stage", "model", "ablation", "ablation_value")
                    },
                )
                atomic_write_json(args.out_dir / "training_validation.json", validation)
                if not validation["validation_passed"]:
                    raise ValueError(
                        "Full-horizon training validation failed; inspect preserved report."
                    )
                report, packets = evaluate_budget_policy(
                    experiment,
                    episode_seeds=list(range(200000, 200256)),
                    deadline=50,
                )
            finally:
                experiment.close()
            report.update(
                {
                    "run_id": run.name,
                    "training_seed": metadata["seed"],
                    "model": metadata["model"],
                    "visibility": metadata["ablation_value"],
                    "protocol_sha256": content_sha256(protocol),
                    "training_validation_sha256": file_sha256(
                        args.out_dir / "training_validation.json"
                    ),
                    "retained_checkpoints_sha256": file_sha256(
                        args.out_dir / "retained_checkpoints.json"
                    ),
                    "plan_sha256": file_sha256(root / "docs/PCP_VISIBILITY_BUDGET_PLAN.md"),
                }
            )
            if packets is not None:
                path = args.out_dir / "live_packet_bank.pt"
                torch.save({"packets": packets, "episode_seeds": report["episode_seeds"]}, path)
                report["packet_file_sha256"] = file_sha256(path)
            if before != {str(path.resolve()): file_sha256(path) for path in input_paths}:
                raise ValueError("Audited run, protocol, plan or evaluation source changed.")
            if source_fingerprint(root) != source_before:
                raise ValueError("Package source changed during held-out evaluation.")
            report["input_artifact_sha256"] = before
            report["inputs_unchanged"] = True
            atomic_write_json(args.out_dir / "heldout_evaluation.json", report)
            atomic_write_json(args.out_dir / "status.json", {"status": "completed"})
        return 0
    except Exception as error:
        atomic_write_json(
            args.out_dir / "status.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
