"""Original-archive reuse preserves historical provenance and fails closed on drift."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/reuse_pcp_initial_archive.py"
SPEC = importlib.util.spec_from_file_location("initial_archive_reuse", SCRIPT)
reuse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reuse)


class InitialArchiveReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        run_id = (f"predator_capture_prey__mappo__pcp_comm_identity__{reuse.SUITE}"
                  "__visibility__radius1__seed020")
        self.plan = SimpleNamespace(
            model="pcp_comm_identity", run_id=run_id, seed=20, ablation_value="radius1",
            suite_id=reuse.SUITE, max_n_frames=600000, attempt=0, retry_of=None,
            scientific_hash="fixed scientific configuration", output_root=self.root / "runs",
        )
        self.run = self.plan.output_root / reuse.SUITE / run_id
        self.run.mkdir(parents=True)
        self.old_queue = self.root / "results/original_queue"
        self.old_queue.mkdir(parents=True)
        self.approval = self.root / "configs/original_approval.json"
        self.approval.parent.mkdir()
        self.approval.write_text(json.dumps({"original": "approval before relocation"}))
        metadata = {"scientific_config_sha256": self.plan.scientific_hash,
                    "protocol": {"approval_sha256": reuse.content_sha(
                        json.loads(self.approval.read_text()))}}
        (self.run / "metadata.json").write_text(json.dumps(metadata))
        (self.run / "actor.pt").write_bytes(b"unchanged original actor")
        log = self.old_queue / f"{run_id}.log"
        log.write_text(f"[sweep] {run_id} -> completed\n")
        manifest = self.old_queue / "manifest.csv"
        manifest.write_text("original historical manifest\n")
        files = {"run/metadata.json": self.run / "metadata.json",
                 "run/actor.pt": self.run / "actor.pt", "worker/stdout.log": log,
                 "provenance/launch_approval.json": self.approval,
                 "provenance/worker_manifest.csv": manifest}
        records = {
            name: {"source_path": str(reuse.OLD_ROOT / path.relative_to(self.root)),
                   "sha256": reuse.sha(path), "bytes": path.stat().st_size, "mode": 0o644}
            for name, path in files.items()
        }
        self.document = {
            "schema_version": 1, "run_id": run_id, "suite_id": reuse.SUITE,
            "source_sha256": reuse.SOURCE, "protocol_sha256": reuse.PROTOCOL,
            "model": self.plan.model, "seed": 20, "visibility": "radius1",
            "frames": 600000, "iterations": 100, "run_file_count": 2, "files": records,
        }
        self.source = (self.root / "results" / reuse.SUITE / "interim_hold_archives" / run_id)
        self.source.mkdir(parents=True)
        self.archive = self.source / "run.tar.gz"
        payload = json.dumps(self.document).encode()
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, path in files.items():
                member = tarfile.TarInfo(name)
                member.size = path.stat().st_size
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(path.read_bytes()))
            member = tarfile.TarInfo("ARCHIVE_MANIFEST.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        _, embedded_sha = reuse.verify_archive(self.archive)
        self.receipt = self.source / "archive_manifest.json"
        self.receipt.write_text(json.dumps({
            "schema_version": 1, "checks_passed": True, "run_id": run_id,
            "archive_file": "run.tar.gz", "archive_sha256": reuse.sha(self.archive),
            "archive_bytes": self.archive.stat().st_size, "source_before_after_equal": True,
            "all_archive_files_rehashed": True, "original_files_removed": False,
            "embedded_manifest_sha256": embedded_sha, "archived_files": len(files),
            "run_file_count": 2,
            "uncompressed_file_bytes": sum(v["bytes"] for v in records.values()),
        }))
        anchors = {self.plan.model: {"archive": reuse.sha(self.archive),
                                    "receipt": reuse.sha(self.receipt)}}
        override = patch.object(reuse, "ANCHORS", anchors)
        override.start()
        self.addCleanup(override.stop)
        self.manifest = self.root / "results/relocation/manifest.csv"
        self.manifest.parent.mkdir()
        self.manifest.write_text("reviewed relocated manifest\n")
        self.relocation_receipt = self.manifest.parent / "relocation_receipt.json"
        self.relocation_receipt.write_text(json.dumps({
            "checks_passed": True, "source_sha256": reuse.SOURCE,
            "generated_file_sha256": {"manifest.csv": reuse.sha(self.manifest)},
        }))
        self.destination = self.root / "results/new_execution/run_archives" / run_id

    def verify(self):
        return reuse.verify_original(self.root, self.plan, self.old_queue)

    def copy(self, expected=None):
        return reuse.copy_verified(
            self.root, self.plan, self.old_queue, self.manifest, self.relocation_receipt,
            self.destination, self.verify() if expected is None else expected,
        )

    def test_copy_preserves_archive_receipt_and_original_metadata_approval(self):
        original_metadata = (self.run / "metadata.json").read_bytes()
        verification = self.verify()
        copied = self.copy(verification)
        for name in ("run.tar.gz", "archive_manifest.json"):
            self.assertEqual((self.source / name).read_bytes(),
                             (self.destination / name).read_bytes())
        receipt = json.loads((self.destination / "archive_reuse_receipt.json").read_text())
        self.assertTrue(copied["checks_passed"] and copied["reused_original_archive"])
        self.assertEqual(receipt["source_verification"], verification)
        self.assertEqual(receipt["operational_binding_sha256"][str(self.manifest)],
                         reuse.sha(self.manifest))
        self.assertEqual((self.run / "metadata.json").read_bytes(), original_metadata)
        self.assertFalse(receipt["historical_files_modified"])
        self.assertEqual(reuse.verify_archive(self.destination / "run.tar.gz")[0], self.document)

    def test_changed_raw_policy_or_historical_approval_is_rejected(self):
        for path in (self.run / "actor.pt", self.approval, self.old_queue / "manifest.csv"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b"changed historical input")
                with self.assertRaisesRegex(ValueError, "Historical archive input changed"):
                    self.verify()
                path.write_bytes(original)

    def test_new_unarchived_run_file_and_symlink_are_rejected(self):
        extra = self.run / "new.pt"
        extra.write_bytes(b"not originally preserved")
        with self.assertRaisesRegex(ValueError, "raw run inventory differs"):
            self.verify()
        extra.unlink()
        extra.symlink_to(self.run / "actor.pt")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.verify()

    def test_archive_and_receipt_tamper_rejected_before_readback(self):
        for path in (self.archive, self.receipt):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "reviewed immutable bytes"):
                    self.verify()
                path.write_bytes(original)

    def test_noninitial_or_retry_row_cannot_reuse_preservation(self):
        for field, value in (("seed", 21), ("ablation_value", "global"), ("attempt", 1),
                             ("model", "pcp_broadcast_k0"), ("run_id", "another_run")):
            original = getattr(self.plan, field)
            setattr(self.plan, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "two original"):
                self.verify()
            setattr(self.plan, field, original)

    def test_existing_overlapping_and_symlink_destinations_are_preserved(self):
        expected = self.verify()
        self.destination.mkdir(parents=True)
        marker = self.destination / "partial_evidence"
        marker.write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "no retry or overwrite"):
            self.copy(expected)
        self.assertEqual(marker.read_bytes(), b"preserve")
        self.destination = self.run / "nested_archive"
        with self.assertRaisesRegex(ValueError, "overlaps protected inputs"):
            self.copy(expected)
        self.destination = self.root / "results/symlink_destination"
        self.destination.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.copy(expected)

    def test_derived_manifest_change_is_rejected_before_output_creation(self):
        expected = self.verify()
        self.manifest.write_text("unreviewed manifest")
        with self.assertRaisesRegex(ValueError, "Derived manifest"):
            self.copy(expected)
        self.assertFalse(self.destination.exists())

    def test_copy_corruption_or_concurrent_input_change_cannot_publish_success(self):
        expected = self.verify()
        original_copy = reuse.shutil.copyfileobj

        def mutate_after_copy(source, destination, **kwargs):
            original_copy(source, destination, **kwargs)
            self.approval.write_text("changed during copy")

        with patch.object(reuse.shutil, "copyfileobj", side_effect=mutate_after_copy):
            with self.assertRaisesRegex(ValueError, "Historical archive input changed"):
                self.copy(expected)
        self.assertTrue((self.destination / "run.tar.gz").is_file())
        self.assertFalse((self.destination / "archive_reuse_receipt.json").exists())

    def test_truncated_archive_footer_is_rejected(self):
        truncated = self.root / "truncated.tar.gz"
        truncated.write_bytes(self.archive.read_bytes()[:-5])
        with self.assertRaises((EOFError, OSError, tarfile.TarError)):
            reuse.verify_archive(truncated)
