"""Strict offline replay of CPUv2 diagnostic snapshots, never final-policy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torchrl.modules.distributions import TanhNormal

from commstudy.analysis.confirmation import _read_metrics, file_sha256
from commstudy.analysis.diagnostics import _analysis_spec, _runtime_compatibility, rebuild_spec
from commstudy.communication.channel import channel_phase, preserve_channel_rng
from commstudy.communication.identity import IdentityComm
from commstudy.experiments.bookkeeping import atomic_write_json, task_runtime_contract
from commstudy.experiments.config import resolved_experiment_dict, scientific_config_sha256
from commstudy.experiments.health import _preserve_communication_stats
from commstudy.experiments.protocols import content_sha256, load_protocol, protocol_spec
from commstudy.experiments.provenance import source_fingerprint
from commstudy.experiments.runner import build_experiment
from commstudy.models import CommPolicyModel
from commstudy.utils.rng import RNGState, preserve_rng_state


FRAME = 564000
GROUP = "adversary"
RTOL = ATOL = 1e-5
REPLAY_FIELDS = (
    "replay_log_prob_max_abs_error",
    "replay_log_prob_scalar_count",
    "replay_transition_count",
    "replay_batched_log_prob_max_abs_error",
    "replay_batched_log_prob_max_tolerance_ratio",
    "replay_accepted_log_prob_max_tolerance_ratio",
    "replay_collection_shape_fallback_count",
    "replay_collection_shape_fallback_slices",
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_equal(x, y) for x, y in zip(left, right, strict=True))
        )
    return left == right


def _tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{tensor.dtype}/{list(tensor.shape)}".encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _model_hash(experiment):
    values = {}
    for group, policy in experiment.group_policies.items():
        for category, iterator in (
            ("parameter", policy.named_parameters()),
            ("buffer", policy.named_buffers()),
        ):
            for name, tensor in iterator:
                values[f"{group}/{category}/{name}"] = _tensor_hash(tensor)
    return content_sha256(values)


def _load_snapshot_policies(experiment, policies):
    _need(
        isinstance(policies, Mapping) and set(policies) == set(experiment.group_policies),
        "Snapshot must contain exactly all policy groups.",
    )
    hashes = {}
    for group, state in policies.items():
        policy = experiment.group_policies[group]
        initial = policy.state_dict()
        # TensorDict-backed MLPs serialize their structural metadata alongside
        # tensors. It must exactly match the architecture rebuilt from the spec.
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                saved_hash = _policy_metadata_hash(name, value)
                _need(
                    name in initial and saved_hash == _policy_metadata_hash(name, initial[name]),
                    f"{group}/{name}: snapshot metadata differs from the rebuilt policy.",
                )
        policy.load_state_dict(state, strict=True)
        loaded = policy.state_dict()
        _need(set(state) == set(loaded), f"{group}: loaded state keys differ from snapshot.")
        hashes[group] = {}
        for name in state:
            if isinstance(state[name], torch.Tensor):
                _need(
                    isinstance(loaded[name], torch.Tensor), f"{group}/{name}: state type differs."
                )
                saved_hash, loaded_hash = _tensor_hash(state[name]), _tensor_hash(loaded[name])
            else:
                saved_hash = _policy_metadata_hash(name, state[name])
                loaded_hash = _policy_metadata_hash(name, loaded[name])
            _need(saved_hash == loaded_hash, f"{group}/{name}: loaded state bytes/type differ.")
            hashes[group][name] = saved_hash
    return hashes


def _policy_metadata_hash(name, value):
    if name.endswith(".params.__batch_size") and type(value) is torch.Size:
        document = {"type": "torch.Size", "value": list(value)}
    elif name.endswith(".params.__device") and value is None:
        document = {"type": "NoneType", "value": None}
    else:
        raise ValueError(f"{name}: unsupported TensorDict policy metadata type/value.")
    return content_sha256(document)


def _comparison(actual, expected):
    _need(actual.shape == expected.shape, "Replay tensor shapes differ.")
    _need(
        bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "Non-finite replay evidence.",
    )
    error = (actual - expected).abs()
    ratio = error / (ATOL + RTOL * expected.abs())
    indices = torch.nonzero(ratio > 1, as_tuple=False)[:5]
    return {
        "maximum_absolute_error": float(error.max()),
        "maximum_tolerance_ratio": float(ratio.max()),
        "violation_count": int((ratio > 1).sum()),
        "allclose": bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
        "bitwise_equal": _tensor_hash(actual) == _tensor_hash(expected),
        "examples": [
            {
                "index": index.tolist(),
                "stored_log_prob": float(expected[tuple(index)]),
                "replayed_log_prob": float(actual[tuple(index)]),
                "tolerance_ratio": float(ratio[tuple(index)]),
            }
            for index in indices
        ],
    }


def _replay(policy, batch):
    replay = batch.clone()
    distribution = policy.get_dist(replay)
    return {
        "loc": replay[GROUP, "loc"].detach().cpu(),
        "scale": replay[GROUP, "scale"].detach().cpu(),
        "log_prob": distribution.log_prob(batch[GROUP, "action"]).detach().cpu(),
    }


def replay_snapshot(experiment, payload):
    """Recompute explicit batch shapes; supports only the declared pure Identity actor."""
    policy = experiment.group_policies[GROUP]
    actors = [module for module in policy.modules() if isinstance(module, CommPolicyModel)]
    _need(
        len(actors) == 1 and type(actors[0].comm) is IdentityComm,
        "Only the exact feed-forward Identity candidate actor is supported.",
    )
    _need(
        not any(
            isinstance(
                module, (torch.nn.RNNBase, torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)
            )
            for module in policy.modules()
        ),
        "Stateful or batch-dependent actors are unsupported.",
    )
    batch = payload["batch"]
    _need(
        batch.names == [None, "time"] and list(batch.batch_size) == [10, 600],
        "Snapshot must preserve the actual [10,600] collection batch and named time axis.",
    )
    _need(
        payload["details"].get("rtol") == RTOL and payload["details"].get("atol") == ATOL,
        "Snapshot tolerance differs from the original guard.",
    )
    model_before = _model_hash(experiment)
    rng_before = RNGState.capture()
    with (
        preserve_rng_state(),
        preserve_channel_rng([]),
        _preserve_communication_stats(policy),
        channel_phase("optimization"),
        torch.enable_grad(),
    ):
        whole = _replay(policy, batch)
        pieces = [_replay(policy, step) for step in batch.unbind(1)]
        collection = {
            key: torch.stack([piece[key] for piece in pieces], dim=1) for key in pieces[0]
        }
        stored_distribution = TanhNormal(
            batch[GROUP, "loc"], batch[GROUP, "scale"], event_dims=1
        ).log_prob(batch[GROUP, "action"])
    rng_unchanged = _equal(rng_before.__dict__, RNGState.capture().__dict__)
    model_after = _model_hash(experiment)
    expected = batch[GROUP, "log_prob"]
    result = {
        "batch_shape": list(batch.batch_size),
        "batch_names": list(batch.names),
        "collection_batch_shape": [10],
        "rtol": RTOL,
        "atol": ATOL,
        "whole_batch": _comparison(whole["log_prob"], expected),
        "collection_shape": _comparison(collection["log_prob"], expected),
        "stored_distribution": _comparison(stored_distribution, expected),
        "stored_distribution_parameters": {
            name: {
                "stored_sha256": _tensor_hash(batch[GROUP, name]),
                "collection_sha256": _tensor_hash(collection[name]),
                "bitwise_equal": _tensor_hash(batch[GROUP, name]) == _tensor_hash(collection[name]),
            }
            for name in ("loc", "scale")
        },
        "captured_tensors": {},
        "model_sha256_before": model_before,
        "model_sha256_after": model_after,
        "model_unchanged": model_before == model_after,
        "rng_unchanged": rng_unchanged,
        "input_tensor_sha256": {
            name: _tensor_hash(batch[GROUP, name])
            for name in ("observation", "action", "log_prob", "loc", "scale")
        },
    }
    for label, actual in (("batched", whole), ("collection_shaped", collection)):
        captured = payload["replay_tensors"][label]
        _need(set(captured) == set(actual), f"{label}: captured tensor schema differs.")
        result["captured_tensors"][label] = {
            name: {
                "sha256": _tensor_hash(actual[name]),
                "captured_sha256": _tensor_hash(captured[name]),
                "bitwise_equal": _tensor_hash(actual[name]) == _tensor_hash(captured[name]),
            }
            for name in actual
        }
        _need(
            all(item["bitwise_equal"] for item in result["captured_tensors"][label].values()),
            f"{label}: offline replay differs from captured tensors.",
        )
    for label, summary in (
        ("batched", result["whole_batch"]),
        ("collection_shaped", result["collection_shape"]),
    ):
        for name in ("maximum_absolute_error", "maximum_tolerance_ratio", "violation_count"):
            _need(
                payload["details"][label].get(name) == summary[name],
                f"{label}/{name}: snapshot summary differs from recomputation.",
            )
    _need(result["model_unchanged"] and rng_unchanged, "Replay mutated model weights or RNG.")
    _need(
        not result["whole_batch"]["allclose"], "Original whole-batch rejection did not reproduce."
    )
    _need(result["collection_shape"]["allclose"], "Collection-shape replay still fails.")
    _need(
        all(item["bitwise_equal"] for item in result["stored_distribution_parameters"].values()),
        "Stored behavior loc/scale differs from collection-shape replay.",
    )
    _need(
        result["stored_distribution"]["allclose"],
        "Stored behavior distribution does not reproduce the stored log probabilities.",
    )
    return result


def _metric_binding(observations, groups):
    counts = ("replay_collection_shape_fallback_count", "replay_collection_shape_fallback_slices")
    for group in groups:
        for frame in range(6000, FRAME, 6000):
            for name in counts:
                key = ("collection", group, name, frame, "")
                _need(
                    key in observations and observations[key][0] == 0,
                    f"{group}/{frame}/{name}: missing or nonzero before 564k.",
                )
    selected = {}
    iterations = set()
    for name in REPLAY_FIELDS:
        key = ("collection", GROUP, name, FRAME, "")
        _need(key in observations, f"Missing recorded collection diagnostic {name} at {FRAME}.")
        selected[name], iteration = observations[key]
        iterations.add(iteration)
    _need(
        len(iterations) == 1, "Snapshot metrics disagree about their recorded collection iteration."
    )
    _need(
        selected[counts[0]] == 1 and selected[counts[1]] == 600,
        "The first predator fallback is not exactly the declared 564k/600-slice event.",
    )
    return {"recorded_collection_iteration": next(iter(iterations)), "metrics": selected}


def _previous_run(previous_suite, metadata):
    candidates = []
    for path in previous_suite.glob("*/metadata.json"):
        prior = json.loads(path.read_text())
        if prior.get("seed") == metadata["seed"]:
            candidates.append((path.parent, prior))
    _need(len(candidates) == 1, "Require exactly one original CPUv1 run for this seed.")
    path, prior = candidates[0]
    _need(
        prior.get("protocol", {}).get("protocol_id") == "pcp_corrected_cpu_v1"
        and prior.get("model") == "pcp_comm_identity"
        and prior.get("scientific_config_sha256") == metadata["scientific_config_sha256"],
        "Previous run is not the same scientific CPUv1 Identity configuration.",
    )
    return path, prior


def audit_run(run, previous_suite, protocol, protocol_path, root, scratch):
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    _need(
        metadata.get("status") == "completed" and metadata.get("frames") == 600000,
        "CPUv2 run has not completed its 600k horizon.",
    )
    _need(
        json.loads((run / "status.json").read_text()).get("status") == "completed",
        "Managed run status is not completed.",
    )
    _need(
        metadata.get("model") == "pcp_comm_identity",
        "Only candidate Identity snapshots are supported.",
    )
    source = source_fingerprint(root)
    _need(
        metadata.get("source_sha256") == source, "Saved source differs from current package source."
    )
    _need(
        metadata.get("protocol", {}).get("protocol_id") == protocol["protocol_id"]
        and metadata["protocol"].get("protocol_sha256") == content_sha256(protocol)
        and metadata["protocol"].get("source_sha256") == source
        and metadata["protocol"].get("stage") == "confirmation",
        "Saved protocol binding differs from the supplied CPUv2 protocol.",
    )
    runtime = protocol["runtime"]
    _need(
        metadata.get("versions") == {"python": runtime["python"], **runtime["packages"]}
        and metadata.get("runtime", {}).get("torch_num_threads") == runtime["threads"]
        and all(
            metadata.get("runtime", {}).get(key) == runtime["device"]
            for key in ("sampling_device", "train_device", "buffer_device")
        ),
        "Saved runtime differs from the protocol runtime.",
    )
    spec = rebuild_spec(run)
    expected = protocol_spec(protocol, model="pcp_comm_identity", seed=metadata["seed"])
    actual_dict, expected_dict = resolved_experiment_dict(spec), resolved_experiment_dict(expected)
    for document in (actual_dict, expected_dict):
        document["experiment"].pop("save_folder", None)
    _need(
        actual_dict == expected_dict
        and metadata.get("scientific_config_sha256") == scientific_config_sha256(expected),
        "Saved resolved/scientific configuration differs.",
    )
    mismatches, versions, current_source = _runtime_compatibility(metadata, spec)
    _need(not mismatches, f"Saved runtime is incompatible: {mismatches}")
    _need(current_source == source, "Source changed while preparing replay.")
    snapshots = list(run.glob("benchmarl/**/replay_diagnostics/*_adversary_first_fallback*.pt"))
    _need(len(snapshots) == 1, "Require exactly one preserved first predator fallback snapshot.")
    snapshot = snapshots[0]
    previous_run, previous_metadata = _previous_run(previous_suite, metadata)
    previous_status_path = previous_run / "status.json"
    paths = [
        metadata_path,
        run / "status.json",
        run / "resolved_config.yaml",
        run / "metrics.csv",
        snapshot,
        protocol_path,
        previous_status_path,
        previous_run / "metadata.json",
    ]
    before_hashes = {str(path.resolve()): file_sha256(path) for path in paths}
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    _need(
        payload.get("diagnostic_only") is True
        and payload.get("not_resume") is True
        and payload.get("reason") == "first_fallback"
        and payload.get("group") == GROUP
        and payload.get("frames") == FRAME
        and payload.get("source_sha256") == source,
        "Snapshot identity/source/frame or diagnostic-only labels differ.",
    )
    issues = []
    observations, _ = _read_metrics(run / "metrics.csv", issues)
    _need(not issues, f"Metric stream is invalid: {issues[:5]}")
    binding = _metric_binding(observations, protocol["contracts"]["training_groups"])
    with preserve_rng_state():
        analysis_spec, overrides = _analysis_spec(spec, scratch, None)
        experiment = build_experiment(analysis_spec)
        try:
            _need(
                task_runtime_contract(experiment) == metadata["task_runtime_contract"],
                "Rebuilt task semantics differ from the saved contract.",
            )
            policies = payload.get("group_policies")
            loaded_policy_hashes = _load_snapshot_policies(experiment, policies)
            replay = replay_snapshot(experiment, payload)
        finally:
            experiment.close()
    metric_map = binding["metrics"]
    _need(
        metric_map["replay_transition_count"] == payload["batch"].numel()
        and metric_map["replay_log_prob_scalar_count"]
        == payload["batch"][GROUP, "log_prob"].numel(),
        "CSV replay denominators differ from the snapshot.",
    )
    for metric, section in (
        ("replay_batched_log_prob_max_tolerance_ratio", "whole_batch"),
        ("replay_accepted_log_prob_max_tolerance_ratio", "collection_shape"),
    ):
        _need(
            metric_map[metric] == replay[section]["maximum_tolerance_ratio"],
            f"CSV {metric} differs from replay.",
        )
    _need(
        metric_map["replay_batched_log_prob_max_abs_error"]
        == replay["whole_batch"]["maximum_absolute_error"],
        "CSV whole-batch error differs from replay.",
    )
    _need(
        metric_map["replay_log_prob_max_abs_error"]
        == replay["collection_shape"]["maximum_absolute_error"],
        "CSV accepted error differs from replay.",
    )
    previous_status = json.loads(previous_status_path.read_text())
    _need(previous_status.get("status") == "failed", "Original CPUv1 status must remain failed.")
    message = previous_status.get("error", {}).get("message", "")
    match = re.search(r"maximum absolute error ([0-9.eE+\-]+)", message)
    _need(match is not None, "Original failure lacks a recorded rounded maximum replay error.")
    after_hashes = {str(path.resolve()): file_sha256(path) for path in paths}
    _need(
        before_hashes == after_hashes,
        "Original snapshot/run/protocol artifacts changed during replay.",
    )
    return {
        "run_id": run.name,
        "seed": metadata["seed"],
        "checks_passed": True,
        "measured_weights": "recorded 564k collection snapshot; not final 600k policy checkpoint",
        "snapshot": str(snapshot.resolve()),
        "snapshot_sha256": file_sha256(snapshot),
        "snapshot_frames": payload["frames"],
        **binding,
        "replay": replay,
        "provenance": {
            "snapshot_frames": payload["frames"],
            "snapshot_group": payload["group"],
            "recorded_collection_iteration": binding["recorded_collection_iteration"],
            "metrics_sha256": file_sha256(run / "metrics.csv"),
            "metadata_sha256": file_sha256(metadata_path),
            "resolved_config_sha256": file_sha256(run / "resolved_config.yaml"),
            "protocol_sha256": content_sha256(protocol),
            "protocol_file_sha256": file_sha256(protocol_path),
            "source_sha256": source,
            "scientific_config_sha256": scientific_config_sha256(spec),
            "versions": versions,
            "runtime": metadata["runtime"],
            "analysis_overrides": overrides,
            "all_groups_strictly_loaded": sorted(policies),
            "loaded_policy_state_sha256": loaded_policy_hashes,
        },
        "original_failure": {
            "comparison_limit": "Rounded scalar summary only: original CPUv1 564k tensors were "
            "not preserved, so equality does not prove historical exact-batch reproduction.",
            "status_path": str(previous_status_path.resolve()),
            "status_sha256": file_sha256(previous_status_path),
            "source_sha256": previous_metadata.get("source_sha256"),
            "protocol": previous_metadata.get("protocol"),
            "scientific_config_sha256": previous_metadata.get("scientific_config_sha256"),
            "status": "failed",
            "message": message,
            "rounded_maximum_error": match.group(1),
            "rounded_maximum_reproduced": format(
                replay["whole_batch"]["maximum_absolute_error"], ".8g"
            )
            == match.group(1),
        },
        "artifact_hashes_before": before_hashes,
        "artifact_hashes_after": after_hashes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--previous-suite", type=Path, required=True)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol", type=Path, default=root / "configs/protocols/pcp_corrected_cpu_v2.yaml"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _need(
        not any(
            parent == args.out.resolve() or parent in args.out.resolve().parents
            for parent in (args.suite_dir.resolve(), args.previous_suite.resolve())
        ),
        "Derived output must be outside both audited suites.",
    )
    report = {
        "schema_version": 1,
        "audit": "pcp_identity_replay_snapshot_v1",
        "runs": [],
        "issues": [],
        "approval_decision": "not_issued",
        "limits": "Pure Identity candidate only; stateful or batch-dependent actors unsupported. "
        "Original failed rows remain failed; snapshot replay is not final-policy evaluation.",
    }
    try:
        protocol = load_protocol(args.protocol)
        _need(
            protocol["protocol_id"] == "pcp_corrected_cpu_v2"
            and protocol["runtime"]["device"] == "cpu",
            "This audit is restricted to the declared CPUv2 candidate protocol.",
        )
        runs = sorted(path.parent for path in args.suite_dir.glob("*/metadata.json"))
        seeds = [json.loads((run / "metadata.json").read_text()).get("seed") for run in runs]
        _need(
            sorted(seeds) == sorted(protocol["selection"]["confirmation_seeds"]),
            "Suite does not contain exactly the declared confirmation seeds.",
        )
        with tempfile.TemporaryDirectory(prefix="pcp-replay-snapshot-audit-") as temporary:
            for run in runs:
                try:
                    report["runs"].append(
                        audit_run(
                            run,
                            args.previous_suite,
                            protocol,
                            args.protocol,
                            root,
                            Path(temporary) / run.name,
                        )
                    )
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                    report["runs"].append(
                        {
                            "run_id": run.name,
                            "checks_passed": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        report["issues"].append(f"{type(error).__name__}: {error}")
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
            if run.get("checks_passed")
        },
    )
    report["provenance_sidecar"] = {"path": str(sidecar.resolve()), "sha256": file_sha256(sidecar)}
    atomic_write_json(args.out, report)
    print(json.dumps({"checks_passed": report["checks_passed"], "report": str(args.out)}, indent=2))
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
