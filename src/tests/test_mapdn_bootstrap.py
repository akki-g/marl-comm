"""Lightweight bootstrap guards; no scientific package import or environment rollout."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/bootstrap_mapdn_local.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_mapdn_local", SCRIPT)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_local_wheel_hash_is_explicit_and_registry_hash_is_preserved(self):
        registry_hash, local_hash = "a" * 64, "b" * 64
        raw = f"local==1.0\nregistry==2.0 \\\n    --hash=sha256:{registry_hash}\n"
        result = bootstrap.complete_hash_lock(raw, {
            "local": {"version": "1.0"}, "registry": {"version": "2.0"},
        }, [{"package": "local", "sha256": local_hash}])
        self.assertIn(f"--hash=sha256:{local_hash}", result)
        self.assertIn(f"--hash=sha256:{registry_hash}", result)

    def test_missing_hash_version_change_and_extra_dependency_fail(self):
        expected = {"one": {"version": "1.0"}}
        for raw in ("one==1.0\n", f"one==2.0 --hash=sha256:{'a' * 64}\n",
                    f"one==1.0 --hash=sha256:{'a' * 64}\ntwo==1.0\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                bootstrap.complete_hash_lock(raw, expected, [])

    def test_local_build_copy_leaves_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "vendor/mapdn-source"
            (source / "mapdn").mkdir(parents=True)
            original = source / "mapdn/__init__.py"
            original.write_text("VERSION = 'original'\n")
            (source / "pyproject.toml").write_text("[project]\nname = 'mapdn'\n")
            destination = root / "verified-copy"
            inventory = bootstrap.copy_project(root, destination, vendor=True)
            self.assertEqual(inventory["mapdn/__init__.py"], bootstrap.sha(original))
            (destination / "mapdn/__init__.py").write_text("changed only in copy\n")
            (destination / "build").mkdir()
            self.assertEqual(original.read_text(), "VERSION = 'original'\n")
            self.assertFalse((source / "build").exists())

    def test_failed_command_is_logged_before_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RuntimeError):
                bootstrap.command(root, root, [sys.executable, "-c",
                    "import sys; print('retained failure', file=sys.stderr); sys.exit(7)"])
            receipt = json.loads((root / "commands/001.json").read_text())
            self.assertEqual(receipt["returncode"], 7)
            stderr = root / "commands/001.stderr.log"
            self.assertIn("retained failure", stderr.read_text())
            self.assertEqual(receipt["stderr_sha256"], bootstrap.sha(stderr))


if __name__ == "__main__":
    unittest.main()
