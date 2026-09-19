"""Relocation must preserve science and reject damaged or ambiguous evidence."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/relocate_pcp_budget.py"
SPEC = importlib.util.spec_from_file_location("d4_relocation", MODULE_PATH)
relocation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relocation)


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.thesis = Path(self.temporary.name).resolve() / "arbitrary relocated thesis"
        self.root = self.thesis / "renamed repository"
        self.destination = self.root / "results/continuation/relocation"
        self.baseline = self.thesis / ".codex-worktrees/pcp-retention-confirmation"
        self.write(self.root / "src/commstudy/__init__.py", b"# frozen comparison\n")
        self.write(self.baseline / "src/commstudy/__init__.py", b"# frozen baseline\n")
        self.source = relocation.source_fingerprint(self.root)
        self.baseline_source = relocation.source_fingerprint(self.baseline)
        self.leaf = self.root / "results/retained.json"
        self.write(self.leaf, b'{"historical_verdict": "CONFIRMED"}\n')
        leaf_ref = self.ref(self.leaf)
        baseline_leaf = self.baseline / "runs/smoke/metadata.json"
        self.write(baseline_leaf, b'{"saved_seed": 10}\n')
        identity = {
            "checks_passed": True,
            "compared_scalar_count": 4112,
            "previous_metadata": self.ref(baseline_leaf),
            "previous_metrics": self.ref(baseline_leaf),
            "current_metadata": leaf_ref,
            "current_metrics": leaf_ref,
        }
        self.document("identity_equivalence", identity)
        self.document(
            "source_bridge",
            {
                "approved": True,
                "rationale": "Historical scientific decision.",
                "identity_equivalence": self.ref(
                    self.root / relocation.DOCUMENTS["identity_equivalence"]
                ),
            },
        )
        self.document(
            "scientific_review",
            {
                "overall_verdict": "CONFIRMED",
                "evidence": {"retained/path~with_special_chars": leaf_ref},
                "confirmation_report_sha256": relocation.digest(self.leaf.read_bytes()),
            },
        )
        self.document(
            "promotion_certificate",
            {
                "confirmed_protocol": leaf_ref,
                "report": leaf_ref,
                "review": self.ref(self.root / relocation.DOCUMENTS["scientific_review"]),
                "source_bridge": self.ref(self.root / relocation.DOCUMENTS["source_bridge"]),
            },
        )
        self.document(
            "comparison_approval",
            {
                "decision": "approved",
                "reviewer": "Historical reviewer",
                "evidence": [
                    {
                        **self.ref(self.root / relocation.DOCUMENTS["promotion_certificate"]),
                        "kind": "corrected_protocol_confirmation",
                    },
                    {**leaf_ref, "kind": "prespecified_plan"},
                ],
            },
        )
        self.rows = []
        for seed in range(20, 25):
            for visibility in ("radius1", "global"):
                for model in relocation.MODELS:
                    binding = {
                        "source_sha256": self.source,
                        "sha256": relocation.PROTOCOL_SHA,
                        "stage": "comparison",
                        "approval": str(relocation.DOCUMENTS["comparison_approval"]),
                    }
                    self.rows.append(
                        {
                            "run_id": f"{model}__{visibility}__seed{seed}",
                            "suite_id": "pcp_visibility_budget_cpu_v1",
                            "model": model,
                            "seed": str(seed),
                            "ablation": "visibility",
                            "ablation_value": visibility,
                            "max_n_frames": "600000",
                            "attempt": "0",
                            "retry_of": "",
                            "output_root": str(relocation.OLD_REPO / "runs"),
                            "protocol_json": json.dumps(binding),
                            "resolved_spec_json": '{"scientific_field": "untouched"}',
                            "scientific_config_sha256": "a" * 64,
                            "command": "historical explanatory command",
                            "status": "pending",
                        }
                    )
        self.write_manifests()
        self.anchors = {
            name: relocation.digest((self.root / path).read_bytes())
            for name, path in relocation.DOCUMENTS.items()
        }
        self.patches = patch.multiple(
            relocation,
            SOURCE=self.source,
            BASELINE_SOURCE=self.baseline_source,
            MANIFEST_SHA=relocation.digest((self.root / relocation.MANIFEST).read_bytes()),
            ANCHORS=self.anchors,
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def ref(self, path):
        if path.is_relative_to(self.root):
            original = relocation.OLD_REPO / path.relative_to(self.root)
        else:
            original = relocation.OLD_THESIS / path.relative_to(self.thesis)
        return {"path": str(original), "sha256": relocation.digest(path.read_bytes())}

    def document(self, name, document):
        self.write(self.root / relocation.DOCUMENTS[name], relocation.json_bytes(document))

    def write_manifests(self):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
        writer.writeheader()
        writer.writerows(self.rows)
        payload = stream.getvalue().encode()
        self.write(self.root / relocation.MANIFEST, payload)
        self.write(self.root / relocation.EVIDENCE / "training_queue/manifest.csv", payload)

    def run_relocation(self):
        return relocation.Relocation(self.root, self.destination).publish()

    def test_full_relocation_preserves_originals_and_all_scientific_cells(self):
        original_files = {
            str(path): path.read_bytes() for path in self.thesis.rglob("*") if path.is_file()
        }
        receipt = self.run_relocation()
        self.assertTrue(receipt["checks_passed"])
        self.assertFalse(receipt["runtime_validated"])
        for name, payload in original_files.items():
            self.assertEqual(Path(name).read_bytes(), payload)
        with (self.destination / "manifest.csv").open(newline="") as stream:
            derived = list(csv.DictReader(stream))
        for before, after in zip(self.rows, derived, strict=True):
            self.assertEqual(after.pop("output_root"), str(self.root / "runs"))
            binding = json.loads(after.pop("protocol_json"))
            self.assertEqual(
                binding.pop("approval"), str(self.destination / "comparison_approval.json")
            )
            self.assertEqual(
                binding,
                {
                    key: value
                    for key, value in json.loads(before["protocol_json"]).items()
                    if key != "approval"
                },
            )
            self.assertEqual(
                after,
                {
                    key: value
                    for key, value in before.items()
                    if key not in {"output_root", "protocol_json"}
                },
            )
        for name, expected in receipt["generated_file_sha256"].items():
            self.assertEqual(relocation.digest((self.destination / name).read_bytes()), expected)
        # Each operational chain reference must resolve and bind the actual
        # derived child, while scientific decision text remains historical.
        for item in receipt["documents"]:
            before = json.loads(Path(item["original_path"]).read_text())
            after = json.loads(Path(item["derived_path"]).read_text())
            for change in item["reference_changes"]:
                reference = change["derived"]
                self.assertEqual(
                    relocation.digest(Path(reference["path"]).read_bytes()), reference["sha256"]
                )
                target = after
                components = [
                    part.replace("~1", "/").replace("~0", "~")
                    for part in change["pointer"].strip("/").split("/")
                ]
                for component in components[:-1]:
                    target = (
                        target[int(component)] if isinstance(target, list) else target[component]
                    )
                key = int(components[-1]) if isinstance(target, list) else components[-1]
                target[key] = change["original"]
            self.assertEqual(after, before)
        self.assertEqual(relocation.verify_existing(self.root, self.destination), receipt)

    def test_verifier_rejects_tampered_output_even_with_matching_receipt_hash(self):
        receipt = self.run_relocation()
        path = self.destination / "manifest.csv"
        payload = path.read_bytes().replace(b"600000", b"120000")
        path.write_bytes(payload)
        receipt["generated_file_sha256"]["manifest.csv"] = relocation.digest(payload)
        (self.destination / "relocation_receipt.json").write_bytes(relocation.json_bytes(receipt))
        with self.assertRaisesRegex(ValueError, "receipt differs from anchored"):
            relocation.verify_existing(self.root, self.destination)

    def test_verifier_rejects_every_changed_derived_file(self):
        receipt = self.run_relocation()
        for name in receipt["generated_file_sha256"]:
            with self.subTest(name=name):
                path = self.destination / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "Derived file differs from anchored"):
                    relocation.verify_existing(self.root, self.destination)
                path.write_bytes(original)

    def test_verifier_rechecks_original_evidence_after_publication(self):
        self.run_relocation()
        self.leaf.write_bytes(b"changed after publication")
        with self.assertRaisesRegex(ValueError, "Evidence hash mismatch"):
            relocation.verify_existing(self.root, self.destination)

    def test_verifier_rejects_missing_success_receipt(self):
        self.run_relocation()
        (self.destination / "relocation_receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "relocation receipt"):
            relocation.verify_existing(self.root, self.destination)

    def test_leaf_damage_is_rejected_without_creating_output(self):
        self.leaf.write_bytes(b"damaged retained evidence")
        with self.assertRaisesRegex(ValueError, "Evidence hash mismatch"):
            self.run_relocation()
        self.assertFalse(self.destination.exists())

    def test_missing_baseline_evidence_is_rejected(self):
        (self.baseline / "runs/smoke/metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or nonordinary evidence"):
            self.run_relocation()
        self.assertFalse(self.destination.exists())

    def test_source_drift_is_rejected_for_both_trees(self):
        for root in (self.root, self.baseline):
            with self.subTest(root=root):
                extra = root / "src/commstudy/new_module.py"
                extra.write_text("# unintended new scientific source\n")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    self.run_relocation()
                self.assertFalse(self.destination.exists())
                extra.unlink()

    def test_missing_row_is_rejected_even_if_manifest_bytes_are_newly_bound(self):
        self.rows.pop()
        self.write_manifests()
        with patch.object(
            relocation,
            "MANIFEST_SHA",
            relocation.digest((self.root / relocation.MANIFEST).read_bytes()),
        ):
            with self.assertRaisesRegex(ValueError, "complete forty-row"):
                self.run_relocation()
        self.assertFalse(self.destination.exists())

    def test_pinned_manifest_difference_is_rejected(self):
        (self.root / relocation.EVIDENCE / "training_queue/manifest.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Evidence hash mismatch"):
            self.run_relocation()
        self.assertFalse(self.destination.exists())

    def test_attempt_replacement_is_rejected(self):
        self.rows[0]["attempt"] = "1"
        self.write_manifests()
        with patch.object(
            relocation,
            "MANIFEST_SHA",
            relocation.digest((self.root / relocation.MANIFEST).read_bytes()),
        ):
            with self.assertRaisesRegex(ValueError, "Original row differs"):
                self.run_relocation()
        self.assertFalse(self.destination.exists())

    def test_path_mapping_rejects_prefix_collisions_and_traversal(self):
        context = relocation.Relocation(self.root, self.destination)
        for path in (
            str(relocation.OLD_REPO) + "-other/file.json",
            str(relocation.OLD_REPO / "../secret.json"),
            "/etc/passwd",
            "relative.json",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                context.mapped_path(path)

    def test_existing_partial_or_successful_destination_is_never_overwritten(self):
        self.destination.mkdir(parents=True)
        marker = self.destination / "partial.json"
        marker.write_bytes(b"preserve interrupted attempt")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.run_relocation()
        self.assertEqual(marker.read_bytes(), b"preserve interrupted attempt")

    def test_output_outside_results_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be under results"):
            relocation.Relocation(self.root, self.root / "src/new_output").publish()


if __name__ == "__main__":
    unittest.main()
