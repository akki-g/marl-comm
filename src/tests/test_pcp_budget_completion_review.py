"""Independent-audit guard and corruption tests using small synthetic stdlib fixtures.

No real cohort, NumPy bootstrap, Torch tensor or policy rollout is evaluated here.
"""

import copy
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


HEAVY_BEFORE = {name for name in ("numpy", "torch") if name in sys.modules}
SPEC = importlib.util.spec_from_file_location(
    "completion_review",
    Path(__file__).resolve().parents[2] / "scripts/review_pcp_budget_completion.py",
)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def closed_report():
    rows = [
        {
            "model": m,
            "visibility": v,
            "seed": s,
            "run_id": review.expected_run_id(m, v, s),
            "status": "valid",
            "validation_passed": True,
            "issues": [],
        }
        for m, v, s in itertools.product(review.MODELS, review.VISIBILITIES, review.SEEDS)
    ]
    report = {
        "schema_version": 1,
        "analysis": "pcp_visibility_budget_seed_analysis_v1",
        "checks_passed": True,
        "issues": [],
        "status": "complete_for_review",
        "expected_policies": 40,
        "valid_policies": 40,
        "approval_decision": "not_issued",
        "training_seeds": list(review.SEEDS),
        "heldout_episode_seeds": review.EPISODES,
        "deadline": 50,
        "target": 0.8,
        "protocol_sha256": review.PROTOCOL,
        "plan_sha256": review.PLAN,
        "training_source_sha256": review.SOURCE,
        "analysis_source_sha256": review.SOURCE,
        "primary_contrast": review.PRIMARY,
        "rows": rows,
    }
    queue = {
        "status": "completed",
        "phase": "final_analysis",
        "rows": {
            r["run_id"]: {
                "status": "completed",
                "stage": "complete",
                "training_completed": True,
                "evaluation_checks_passed": True,
                "training_reused": review.original(r),
                "archive": {"checks_passed": True},
            }
            for r in rows
        },
    }
    return report, queue


def write_tar(path, sources, inventory, extra=None):
    payload = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, source in sources.items():
            data = source.read_bytes()
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(data), 0o644
            archive.addfile(entry, io.BytesIO(data))
        entry = tarfile.TarInfo("ARCHIVE_MANIFEST.json")
        entry.size, entry.mode = len(payload), 0o644
        archive.addfile(entry, io.BytesIO(payload))
        if extra is not None:
            entry = tarfile.TarInfo(extra)
            entry.size = 0
            if extra == "symlink":
                entry.type, entry.linkname = tarfile.SYMTYPE, "elsewhere"
            archive.addfile(entry, io.BytesIO(b""))
    return hashlib.sha256(payload).hexdigest()


def archive_fixture(root, historical=False):
    root = root.resolve()
    model = "pcp_comm_identity" if historical else "pcp_broadcast_k0"
    seed = 20 if historical else 21
    run_id = review.expected_run_id(model, "radius1", seed)
    row = {
        "run_id": run_id,
        "model": model,
        "visibility": "radius1",
        "seed": seed,
        "parameters": {"total": 100},
    }
    run, queue_dir = root / "runs" / review.SUITE / run_id, root / "execution"
    destination = queue_dir / "run_archives" / run_id
    destination.mkdir(parents=True)
    run.mkdir(parents=True)
    row["run_directory"] = str(run)
    manifest = root / "manifest.csv"
    manifest.write_text("synthetic manifest\n")
    approval = root / "approval.json"
    write_json(approval, {"synthetic_original_approval": True})
    metadata = {
        "status": "completed",
        "run_id": run_id,
        "suite_id": review.SUITE,
        "model": model,
        "seed": seed,
        "ablation": "visibility",
        "ablation_value": "radius1",
        "frames": 600000,
        "iterations": 100,
        "source_sha256": review.SOURCE,
        "scientific_config_sha256": "synthetic-config",
        "parameters": row["parameters"],
        "protocol": {"approval_sha256": review.canonical(json.loads(approval.read_text()))},
    }
    write_json(run / "metadata.json", metadata)
    policy = run / "checkpoints/policy_state.pt"
    policy.parent.mkdir()
    policy.write_bytes(b"synthetic checkpoint; never deserialize")
    worker = queue_dir / f"{run_id}.log"
    worker.write_text(f"[sweep] {run_id} -> completed\n")
    source_names = {
        "source/pyproject.toml": "pyproject.toml",
        "source/uv.lock": "uv.lock",
        "source/src/commstudy/example.py": "src/commstudy/example.py",
        "provenance/protocol.yaml": "configs/protocols/frozen.yaml",
        "provenance/plan.md": "docs/PCP_VISIBILITY_BUDGET_PLAN.md",
        "provenance/archive_helper.py": "scripts/archive_pcp_budget_run.py",
        "provenance/inventory_helper.py": "scripts/evaluate_pcp_budget.py",
        "provenance/source_freeze.json": f"results/{review.SUITE}/package_source_freeze.json",
    }
    sources = {
        "run/metadata.json": run / "metadata.json",
        "run/checkpoints/policy_state.pt": policy,
        "worker/stdout.log": worker,
        "provenance/worker_manifest.csv": manifest,
        "provenance/launch_approval.json": approval,
    }
    for name, relative in source_names.items():
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"synthetic {relative}\n")
        sources[name] = source

    def source_path(path):
        return str(review.OLD_ROOT / path.relative_to(root)) if historical else str(path)

    inventory = {
        "schema_version": 1,
        "archive": "pcp_budget_completed_run_v1",
        "run_id": run_id,
        "suite_id": review.SUITE,
        "source_sha256": review.SOURCE,
        "protocol_sha256": review.PROTOCOL,
        "model": model,
        "seed": seed,
        "visibility": "radius1",
        "frames": 600000,
        "iterations": 100,
        "run_file_count": 2,
        "native_inventory": {
            "schema_version": 1,
            "run_id": run_id,
            "checkpoints": [],
            "replay_diagnostics": [],
            "final_native_actor_matches_managed": True,
            "managed_final_policy_sha256": review.digest(policy),
        },
        "files": {
            name: {
                "sha256": review.digest(path),
                "bytes": path.stat().st_size,
                "mode": 0o644,
                "source_path": source_path(path),
            }
            for name, path in sources.items()
        },
    }
    archive = destination / "run.tar.gz"
    embedded = write_tar(archive, sources, inventory)
    receipt = {
        "schema_version": 1,
        "checks_passed": True,
        "run_id": run_id,
        "archive_file": "run.tar.gz",
        "archive_sha256": review.digest(archive),
        "archive_bytes": archive.stat().st_size,
        "source_before_after_equal": True,
        "all_archive_files_rehashed": True,
        "original_files_removed": False,
        "embedded_manifest_sha256": embedded,
        "archived_files": len(sources),
        "run_file_count": 2,
        "uncompressed_file_bytes": sum(p.stat().st_size for p in sources.values()),
    }
    receipt_path = destination / "archive_manifest.json"
    write_json(receipt_path, receipt)
    archived = {
        "destination": str(destination),
        "archive_sha256": review.digest(archive),
        "receipt_sha256": review.digest(receipt_path),
        "archive_bytes": archive.stat().st_size,
        "checks_passed": True,
    }
    anchors = dict(review.HISTORICAL)
    if historical:
        anchors[model] = archived["archive_sha256"], archived["receipt_sha256"]
        origin = root / "results" / review.SUITE / "interim_hold_archives" / run_id
        origin.mkdir(parents=True)
        (origin / "run.tar.gz").write_bytes(archive.read_bytes())
        (origin / "archive_manifest.json").write_bytes(receipt_path.read_bytes())
        old_log = root / "results" / review.SUITE / "training_queue" / f"{run_id}.log"
        old_log.parent.mkdir(parents=True)
        old_log.write_bytes(worker.read_bytes())
        source_hashes = {str(path): review.digest(path) for path in sources.values()}
        source_hashes.update(
            {
                str(path): review.digest(path)
                for path in (origin / "run.tar.gz", origin / "archive_manifest.json", old_log)
            }
        )
        source = {
            "source": str(origin),
            "run_id": run_id,
            "checks_passed": True,
            "original_approval_matches_metadata": True,
            "all_archive_files_rehashed": True,
            "source_before_after_equal": True,
            "archive_sha256": archived["archive_sha256"],
            "receipt_sha256": archived["receipt_sha256"],
            "embedded_manifest_sha256": embedded,
            "archived_files": len(sources),
            "run_file_count": 2,
            "archive_bytes": archived["archive_bytes"],
            "input_artifact_sha256": source_hashes,
        }
        relocation = root / "relocation_receipt.json"
        write_json(
            relocation,
            {
                "checks_passed": True,
                "source_sha256": review.SOURCE,
                "generated_file_sha256": {manifest.name: review.digest(manifest)},
            },
        )
        reuse = {
            "schema_version": 1,
            "checks_passed": True,
            "run_id": run_id,
            "destination": str(destination),
            "original_files_removed": False,
            "historical_files_modified": False,
            "all_copied_files_rehashed": True,
            "copied_archive_readback_verified": True,
            "source_verification": source,
            "operational_binding_sha256": {
                str(path): review.digest(path) for path in (manifest, relocation)
            },
        }
        write_json(destination / "archive_reuse_receipt.json", reuse)
        archived.update(
            reused_original_archive=True,
            reuse_receipt_sha256=review.digest(destination / "archive_reuse_receipt.json"),
        )
    plan = {
        "output_root": str(root / "runs"),
        "scientific_config_sha256": "synthetic-config",
        "protocol": {"path": "configs/protocols/frozen.yaml", "approval": "approval.json"},
    }
    inputs = review.Inputs()
    inputs.pin(manifest)
    arguments = (
        root,
        row,
        {"archive": archived},
        plan,
        manifest,
        queue_dir,
        {"src/commstudy/example.py"},
        inputs,
    )
    return arguments, sources, inventory, anchors


def episode(seed, model="pcp_comm_identity", arm="live", contact=True):
    metrics = {key: 0.0 for key in review.DOMAIN}
    metrics.update(
        episode_steps=100,
        complete=1,
        truncated=1,
        terminated=0,
        first_contact_step_or_horizon=50 if contact else 100,
        contact_duration_steps=int(contact),
        contact_pair_steps=int(contact),
        contacting_predators_max=int(contact),
        contacting_predators_mean=int(contact) / 100,
        return_value=0,
    )
    metrics.pop("return_value")
    for key in ("any_contact", "first_contact_observed", "contact_by_deadline"):
        metrics[key] = int(contact)
    metrics["return"] = 10 * int(contact)
    metrics["prey_visible_to_0_predators_mean"] = 1.0
    senders = (
        (0 if arm == "severed" else 2 if arm.startswith("suppress_") else 3)
        if model == "pcp_broadcast_k3"
        else 0
    )
    width = 32 if model in ("pcp_broadcast_k0", "pcp_broadcast_k3") else 0
    return {
        "episode_id": f"pcp-heldout:{seed}",
        "seed": seed,
        "initial_physical_state_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        "exogenous_initial": {"exogenous_episode_ids": [seed], "exogenous_steps": [0]},
        "metrics": metrics,
        "action_health": {
            "all_finite": True,
            "action_scalars": 600,
            "saturation_fraction": 0.25,
            "max_abs_action": 1.0,
            "max_abs_loc": 2.0,
            "min_scale": 0.1,
            "max_scale": 1.0,
        },
        "costs": {
            "sender_emissions": 100 * senders,
            "edge_deliveries": 200 * senders,
            "sender_payload_bits": 100 * senders * width * 32,
            "edge_delivery_payload_bits": 200 * senders * width * 32,
            "transitions": 100,
            "packet_scalars": width,
            "bits_per_scalar": 32,
            "cost_model": "one_float32_broadcast_packet_per_available_sender; two_receivers",
            "network_headers_counted": False,
            "measured_network_traffic": False,
        },
        "module_stats": {
            "messages_per_step": senders,
            "realized_sender_bits_per_step": senders * width * 32,
            "realized_bits_per_step": 2 * senders * width * 32,
        },
    }


def fake_interval(values):
    """Unit-test stand-in only; never claim this tests the deferred NumPy calculation."""
    mean = sum(values) / 5
    return {
        "mean": mean,
        "ci95_low": mean,
        "ci95_high": mean,
        "training_seeds": 5,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 73191,
        "unit": "training_seed",
        "method": "paired_seed_percentile_bootstrap",
    }


def statistics_fixture():
    policies = {}
    for model, visibility, seed in itertools.product(
        review.MODELS, review.VISIBILITIES, review.SEEDS
    ):
        methods = {
            "pcp_comm_identity": 0.5,
            "pcp_local_capacity": 0.6,
            "pcp_broadcast_k0": 0.85,
            "pcp_broadcast_k3": 0.9,
        }
        contact = methods[model] - (0.05 if visibility == "global" else 0)
        arms = review.ARMS if model == "pcp_broadcast_k3" else review.ARMS[:2]
        policies[model, visibility, seed] = {
            arm: {
                "contact_by_deadline": contact - (0.1 if arm != "live" and len(arms) == 6 else 0),
                "return": 100 + (10 if model == "pcp_broadcast_k3" and arm == "live" else 0),
                "contact_duration_steps": 10,
                "simultaneous_contact_steps": 0,
                "predator_boundary_fraction": 0.1,
                "prey_boundary_fraction": 0.1,
                "action_saturation_fraction": 0.25,
            }
            for arm in arms
        }
    differences, budgets = review.estimands(policies)
    contrasts = []
    for (name, metric), values in differences.items():
        terms = [
            {"coefficient": c, "model": m, "visibility": v, "arm": a}
            for c, m, v, a in review.contrast_terms()[name, metric]
        ]
        contrasts.append(
            {
                "contrast": name,
                "metric": metric,
                "status": "complete",
                "terms": terms,
                "missing_or_invalid_seeds": [],
                "interval": fake_interval(values),
                "per_seed": [
                    {"seed": s, "value": v} for s, v in zip(review.SEEDS, values, strict=True)
                ],
            }
        )
    frontiers = []
    for visibility in review.VISIBILITIES:
        rows = [
            {
                "sender_budget": k,
                "interval": fake_interval(budgets[visibility, k]),
                "qualifies": fake_interval(budgets[visibility, k])["ci95_low"] >= 0.8,
                "per_seed": [
                    {"seed": s, "value": v}
                    for s, v in zip(review.SEEDS, budgets[visibility, k], strict=True)
                ],
            }
            for k in (0, 3)
        ]
        qualified = [r["sender_budget"] for r in rows if r["qualifies"]]
        frontiers.append(
            {
                "visibility": visibility,
                "target": 0.8,
                "deadline": 50,
                "budgets": rows,
                "smallest_tested_budget": min(qualified) if qualified else "not reached",
            }
        )
    conditions = []
    for model, visibility in itertools.product(review.MODELS, review.VISIBILITIES):
        sample = policies[model, visibility, 20]
        conditions.append(
            {
                "model": model,
                "visibility": visibility,
                "status": "complete",
                "valid_seeds": list(review.SEEDS),
                "means": sample["live"],
                "interventions": sample,
                "intervals": {
                    metric: fake_interval([sample["live"][metric]] * 5)
                    for metric in review.CONDITION_CIS
                },
            }
        )
    return {
        "contrasts": contrasts,
        "budget_frontiers": frontiers,
        "conditions": conditions,
    }, policies


class CompletionReviewTests(unittest.TestCase):
    def test_module_and_incomplete_gate_do_not_import_numpy_torch_or_write_receipt(self):
        self.assertEqual({n for n in ("numpy", "torch") if n in sys.modules}, HEAVY_BEFORE)
        report, queue = closed_report()
        review.closed_cohort(report, queue)
        for fault in ("report", "queue", "row", "grid", "reuse"):
            altered_report, altered_queue = copy.deepcopy(report), copy.deepcopy(queue)
            if fault == "report":
                altered_report["valid_policies"] = 39
            elif fault == "queue":
                altered_queue["status"] = "running"
            elif fault == "row":
                next(iter(altered_queue["rows"].values()))["evaluation_checks_passed"] = False
            elif fault == "grid":
                altered_report["rows"][-1] = copy.deepcopy(altered_report["rows"][0])
            else:
                next(iter(altered_queue["rows"].values()))["training_reused"] = False
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                write_json(root / "analysis/analysis_report.json", altered_report)
                write_json(root / "queue.json", altered_queue)
                with patch.object(
                    review, "package_snapshot", side_effect=AssertionError("too early")
                ):
                    with self.assertRaises(ValueError):
                        review.audit(
                            root, root / "analysis", root / "queue.json", root / "review.json"
                        )
                self.assertFalse((root / "review.json").exists())
        self.assertEqual({n for n in ("numpy", "torch") if n in sys.modules}, HEAVY_BEFORE)

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        for payload in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                review.unique_json(payload)

    def test_archive_readback_checks_every_member_and_gzip_footer(self):
        with tempfile.TemporaryDirectory() as temp:
            args, sources, inventory, _ = archive_fixture(Path(temp))
            path = Path(args[2]["archive"]["destination"]) / "run.tar.gz"
            found, _ = review.archive_inventory(path)
            self.assertEqual(found, inventory)
            for bad_name in (
                "../outside",
                "/absolute",
                "symlink",
                "ARCHIVE_MANIFEST.json",
                "run/metadata.json",
            ):
                with self.subTest(name=bad_name):
                    bad_path = Path(temp).resolve() / "bad.tar.gz"
                    write_tar(bad_path, sources, inventory, extra=bad_name)
                    with self.assertRaises(ValueError):
                        review.archive_inventory(bad_path)
            broken = Path(temp).resolve() / "truncated.tar.gz"
            broken.write_bytes(path.read_bytes()[:-8])
            with self.assertRaises((EOFError, OSError)):
                review.archive_inventory(broken)

    def test_wrong_member_size_hash_and_missing_inventory_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, sources, inventory, _ = archive_fixture(Path(temp))
            for fault in ("hash", "size", "mode", "missing"):
                changed = copy.deepcopy(inventory)
                name = next(iter(changed["files"]))
                if fault == "missing":
                    changed["files"].pop(name)
                else:
                    changed["files"][name][
                        {"hash": "sha256", "size": "bytes", "mode": "mode"}[fault]
                    ] = "wrong"
                path = Path(temp).resolve() / "bad.tar.gz"
                write_tar(path, sources, changed)
                with self.subTest(fault=fault), self.assertRaises(ValueError):
                    review.archive_inventory(path)

    def test_new_and_historical_archives_pass_only_with_exact_sources_and_bindings(self):
        for historical in (False, True):
            with self.subTest(historical=historical), tempfile.TemporaryDirectory() as temp:
                args, _, _, anchors = archive_fixture(Path(temp), historical=historical)
                with patch.object(review, "HISTORICAL", anchors):
                    metadata, _, receipt = review.verify_run_archive(*args)
                self.assertEqual(metadata["run_id"], args[1]["run_id"])
                self.assertEqual(receipt["raw_run_files"], 2)
                self.assertIs(receipt["historical_archive_reused"], historical)
                args[-1].finish()

    def test_archive_source_drift_extra_raw_file_and_swapped_row_are_rejected(self):
        for fault in ("drift", "extra", "swapped"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                args, sources, _, _ = archive_fixture(Path(temp))
                if fault == "drift":
                    sources["run/metadata.json"].write_text("changed raw source")
                elif fault == "extra":
                    (Path(args[1]["run_directory"]) / "unexpected.txt").write_text("extra")
                else:
                    args[1]["run_id"] = "another-policy"
                with self.assertRaises(ValueError):
                    review.verify_run_archive(*args)

    def test_rehashed_historical_receipt_cannot_drop_original_log_or_manifest(self):
        for fault in ("original_log", "manifest", "approval_flag", "other_relocation"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                args, _, _, anchors = archive_fixture(Path(temp), historical=True)
                archived = args[2]["archive"]
                path = Path(archived["destination"]) / "archive_reuse_receipt.json"
                reuse = json.loads(path.read_text())
                if fault == "original_log":
                    hashes = reuse["source_verification"]["input_artifact_sha256"]
                    hashes.pop(next(name for name in hashes if "/training_queue/" in name))
                elif fault == "manifest":
                    reuse["operational_binding_sha256"].pop(str(args[4]))
                elif fault == "approval_flag":
                    reuse["source_verification"]["original_approval_matches_metadata"] = False
                else:
                    binding = reuse["operational_binding_sha256"]
                    original_path = next(Path(p) for p in binding if p != str(args[4]))
                    other = args[0] / "other/relocation_receipt.json"
                    other.parent.mkdir()
                    other.write_bytes(original_path.read_bytes())
                    binding[str(other)] = binding.pop(str(original_path))
                write_json(path, reuse)
                archived["reuse_receipt_sha256"] = review.digest(path)
                with patch.object(review, "HISTORICAL", anchors), self.assertRaises(ValueError):
                    review.verify_run_archive(*args)

    def test_rehashed_outer_receipt_rejects_boolean_schema_version(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, _ = archive_fixture(Path(temp))
            archived = args[2]["archive"]
            path = Path(archived["destination"]) / "archive_manifest.json"
            receipt = json.loads(path.read_text())
            receipt["schema_version"] = True
            write_json(path, receipt)
            archived["receipt_sha256"] = review.digest(path)
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                review.verify_run_archive(*args)

    def test_historical_native_paths_map_only_the_declared_prefix(self):
        root = Path("/synthetic/current")
        old = review.OLD_ROOT / "runs/example/checkpoint_600000.pt"
        self.assertEqual(
            review.mapped_source(old, root, True), root / "runs/example/checkpoint_600000.pt"
        )
        native = {
            "checkpoints": [{"path": str(old), "sha256": "a"}],
            "replay_diagnostics": [{"path": str(old.parent / "replay.pt"), "sha256": "b"}],
        }
        mapped = review.normalize_native(native, root, True)
        self.assertEqual(
            mapped["checkpoints"][0]["path"], str(root / "runs/example/checkpoint_600000.pt")
        )
        self.assertEqual(native["checkpoints"][0]["path"], str(old))
        for path, historical in (
            (Path("/other/root/file"), True),
            (old, False),
            (review.OLD_ROOT / "../outside", True),
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                review.mapped_source(path, root, historical)

    def test_tracker_rejects_conflicting_hashes_and_later_inventory_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            path = root / "source"
            path.write_text("before")
            inputs = review.Inputs()
            inputs.tree(root)
            inputs.pin(path)
            path.write_text("after")
            with self.assertRaisesRegex(ValueError, "first binding"):
                inputs.pin(path)
            path.write_text("before")
            (root / "extra").write_text("new")
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                inputs.finish()

    def test_episode_deadline_null_payload_and_physical_pairing(self):
        row = {"model": "pcp_comm_identity", "run_id": "synthetic"}
        live = [episode(seed, contact=seed % 2 == 0) for seed in review.EPISODES]
        bank = {
            "episode_seeds": review.EPISODES,
            "interventions": {"live": live, "severed": copy.deepcopy(live)},
        }
        row["arms"] = {
            arm: review.aggregate_bank(values) for arm, values in bank["interventions"].items()
        }
        values, reference = review.verify_bank(bank, row)
        self.assertEqual(values["live"]["contact_by_deadline"], 0.5)
        self.assertEqual(values["live"]["return"], 5)
        self.assertEqual(values["live"]["first_contact_conditional_mean"], 50)
        self.assertEqual(values["live"]["transitions"], 25600)
        for fault in ("missing", "null", "deadline", "pairing", "payload", "nonfinite"):
            modified = copy.deepcopy(bank)
            first = modified["interventions"]["severed"][0]
            if fault == "missing":
                modified["interventions"]["severed"].pop()
            elif fault == "null":
                first["action_health"]["saturation_fraction"] = 0.125
            elif fault == "deadline":
                first["metrics"]["contact_by_deadline"] = 0
            elif fault == "pairing":
                first["exogenous_initial"]["exogenous_episode_ids"] = [999]
            elif fault == "payload":
                first["costs"]["sender_payload_bits"] = 1
            else:
                first["metrics"]["return"] = float("nan")
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                review.verify_bank(modified, row)
        reference[0]["initial_physical_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Cross-policy"):
            review.verify_bank(bank, row, reference)

    def test_all_k3_intervention_costs_match_masks_without_tensor_imports(self):
        for arm in review.ARMS:
            record = episode(200000, "pcp_broadcast_k3", arm)
            review.verify_episode(record, 200000, "pcp_broadcast_k3", arm)
            expected = 0 if arm == "severed" else 204800 if arm.startswith("suppress_") else 307200
            self.assertEqual(record["costs"]["sender_payload_bits"], expected)
        self.assertEqual({n for n in ("numpy", "torch") if n in sys.modules}, HEAVY_BEFORE)

    def test_estimands_keep_metrics_distinct_and_reject_reduced_seed_cohort(self):
        report, policies = statistics_fixture()
        effects, _ = review.estimands(policies)
        name = "broadcast_k3_live_minus_severed/radius1"
        self.assertAlmostEqual(effects[name, "contact_by_deadline"][0], 0.1)
        self.assertEqual(effects[name, "return"], [10] * 5)
        self.assertAlmostEqual(effects[review.PRIMARY, "contact_by_deadline"][0], 0.3)
        with patch.object(review, "bootstrap_interval", side_effect=fake_interval):
            checked = review.review_statistics(report, policies)
        self.assertEqual(checked["paired_contrasts_verified"], 27)
        self.assertEqual(checked["smallest_tested_budgets"]["radius1"], 0)
        policies.pop(next(iter(policies)))
        with self.assertRaisesRegex(ValueError, "reduced cohort"):
            review.estimands(policies)

    def test_statistics_reject_sign_collisions_frontier_relabel_and_wrong_interval(self):
        for fault in ("sign", "collision", "frontier", "interval"):
            report, policies = statistics_fixture()
            if fault == "sign":
                report["contrasts"][0]["terms"][0]["coefficient"] = -1
            elif fault == "collision":
                report["contrasts"][-1] = copy.deepcopy(report["contrasts"][0])
            elif fault == "frontier":
                report["budget_frontiers"][0]["smallest_tested_budget"] = 3
            else:
                report["contrasts"][0]["interval"]["ci95_low"] -= 0.01
            with (
                self.subTest(fault=fault),
                patch.object(review, "bootstrap_interval", side_effect=fake_interval),
            ):
                with self.assertRaises(ValueError):
                    review.review_statistics(report, policies)

    def test_bootstrap_invalid_values_fail_before_numpy_import(self):
        for values in ([1] * 4, [1, 1, 1, 1, float("nan")], [True] * 5):
            with self.subTest(values=values), self.assertRaises(ValueError):
                review.bootstrap_interval(values)
        self.assertEqual({n for n in ("numpy", "torch") if n in sys.modules}, HEAVY_BEFORE)

    def test_frontier_uses_lower_bound_not_point_estimate_and_can_be_not_reached(self):
        report, policies = statistics_fixture()

        def wider_interval(values):
            result = fake_interval(values)
            result["ci95_low"] -= 0.06
            result["ci95_high"] += 0.06
            return result

        for contrast in report["contrasts"]:
            contrast["interval"] = wider_interval([r["value"] for r in contrast["per_seed"]])
        for frontier in report["budget_frontiers"]:
            qualified = []
            for budget in frontier["budgets"]:
                budget["interval"] = wider_interval([r["value"] for r in budget["per_seed"]])
                budget["qualifies"] = budget["interval"]["ci95_low"] >= 0.8
                if budget["qualifies"]:
                    qualified.append(budget["sender_budget"])
            frontier["smallest_tested_budget"] = min(qualified) if qualified else "not reached"
        for condition in report["conditions"]:
            for metric in review.CONDITION_CIS:
                condition["intervals"][metric] = wider_interval([
                    policies[condition["model"], condition["visibility"], s]["live"][metric]
                    for s in review.SEEDS])
        with patch.object(review, "bootstrap_interval", side_effect=wider_interval):
            checked = review.review_statistics(report, policies)
        self.assertEqual(checked["smallest_tested_budgets"],
                         {"radius1": 3, "global": "not reached"})


if __name__ == "__main__":
    unittest.main()
