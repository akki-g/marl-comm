"""Create verified operational D4 path bindings after moving the Thesis directory.

This stdlib-only utility never edits an original artifact or starts computation.
The five derived JSON documents retain the historical scientific decisions;
their receipt records only path relocation, not a new scientific approval.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path


OLD_THESIS = Path("/Users/akshatguduru/Desktop/Thesis")
OLD_REPO = OLD_THESIS / "marl-comm"
SOURCE = "69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e"
BASELINE_SOURCE = "7124976af366d28d40a3ccf0ebce62aa2b4019783d13974b9f98dd1c316af262"
PROTOCOL_SHA = "ec7c109413625f270b79bffc1753adf5b7691d008eb90a4297d844b020898b0a"
MANIFEST = Path("runs/pcp_visibility_budget_cpu_v1/manifest.csv")
EVIDENCE = Path("results/pcp_visibility_budget_cpu_v1")
MANIFEST_SHA = "3af81b774a7801cff6db3aa8392ef2b498540d6ab885a737be53099c06cf5660"
DOCUMENTS = {
    "identity_equivalence": EVIDENCE / "identity_equivalence.json",
    "source_bridge": EVIDENCE / "source_bridge.json",
    "scientific_review": Path("results/pcp_candidate_confirmation_cpu_v3/scientific_review.json"),
    "promotion_certificate": EVIDENCE / "promotion_certificate.json",
    "comparison_approval": Path(
        "configs/protocols/approvals/pcp_visibility_budget_cpu_v1_comparison.json"
    ),
}
ANCHORS = {
    "identity_equivalence": "34be89860e4e9d3ab46741e517051ccb7f619e4de2c89d55b9a25b41d7099f92",
    "source_bridge": "73f0071ffcf44a1a2faab9277a8c54cd513c0634185c7c0345063b48a1466a4f",
    "scientific_review": "0273b33c21fadbfa7b5bdcf543ec034a7cc709f1cf3414638317abea4323c503",
    "promotion_certificate": "52984334ef327572eee1ad84b6be7d94fbe1dc06de730ebfb03a381b862e7bc2",
    "comparison_approval": "325a477278550eab6615284bc734348725c60e2524b09edc656c9406b82feed8",
}
MODELS = ("pcp_comm_identity", "pcp_local_capacity", "pcp_broadcast_k0", "pcp_broadcast_k3")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(document) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def source_fingerprint(root: Path) -> str:
    source = root / "src/commstudy"
    need(source.is_dir(), f"Missing source tree: {source}")
    result = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        need(path.is_file() and not path.is_symlink(), f"Nonordinary source: {path}")
        result.update(path.relative_to(source).as_posix().encode() + b"\0")
        result.update(path.read_bytes() + b"\0")
    return result.hexdigest()


class Relocation:
    def __init__(self, root: Path, destination: Path):
        self.root = root.resolve()
        self.destination = destination.resolve()
        self.inputs = {}
        self.outputs = {}
        self.documents = []

    def mapped_path(self, original: str) -> Path:
        path = Path(original)
        need(path.is_absolute() and ".." not in path.parts, "Unmapped or traversing evidence path")
        if path.is_relative_to(OLD_REPO):
            return self.root / path.relative_to(OLD_REPO)
        baseline = OLD_THESIS / ".codex-worktrees/pcp-retention-confirmation"
        if path.is_relative_to(baseline):
            return (
                self.root.parent
                / ".codex-worktrees/pcp-retention-confirmation"
                / path.relative_to(baseline)
            )
        raise ValueError(f"Evidence path is outside the two declared relocated trees: {path}")

    def read_verified(self, path: Path, expected: str | None = None) -> bytes:
        need(path.is_file() and not path.is_symlink(), f"Missing or nonordinary evidence: {path}")
        payload = path.read_bytes()
        actual = digest(payload)
        need(expected is None or actual == expected, f"Evidence hash mismatch: {path}")
        previous = self.inputs.setdefault(str(path.resolve()), actual)
        need(previous == actual, f"Evidence changed during preflight: {path}")
        return payload

    def bound_ref(self, reference, replacement: str | None = None):
        need(
            isinstance(reference, dict)
            and set(reference) in ({"path", "sha256"}, {"path", "sha256", "kind"}),
            "Unexpected bound-reference fields",
        )
        path = self.mapped_path(reference["path"])
        self.read_verified(path, reference["sha256"])
        result = dict(reference)
        result["path"] = str(path)
        if replacement is not None:
            need(path == self.root / DOCUMENTS[replacement], "Wrong replacement reference target")
            filename = f"{replacement}.json"
            result.update(
                path=str(self.destination / filename), sha256=digest(self.outputs[filename])
            )
        return result

    def clone(self, name, transform):
        original_path = self.root / DOCUMENTS[name]
        original = json.loads(self.read_verified(original_path, ANCHORS[name]))
        derived = copy.deepcopy(original)
        # Only the explicitly enumerated references are transformed. Every
        # original target is verified even when a derived child replaces it.
        changes = []

        def change(container, key, pointer, replacement=None):
            before = copy.deepcopy(container[key])
            after = self.bound_ref(before, replacement)
            container[key] = after
            changes.append({"pointer": pointer, "original": before, "derived": after})

        transform(derived, change)
        filename = f"{name}.json"
        self.outputs[filename] = json_bytes(derived)
        self.documents.append(
            {
                "original_path": str(original_path),
                "original_sha256": ANCHORS[name],
                "derived_path": str(self.destination / filename),
                "derived_sha256": digest(self.outputs[filename]),
                "reference_changes": changes,
            }
        )

    def prepare_documents(self):
        def identity(doc, change):
            for key in (
                "previous_metadata",
                "previous_metrics",
                "current_metadata",
                "current_metrics",
            ):
                change(doc, key, f"/{key}")

        self.clone("identity_equivalence", identity)
        self.clone(
            "source_bridge",
            lambda doc, change: change(
                doc, "identity_equivalence", "/identity_equivalence", "identity_equivalence"
            ),
        )

        def review(doc, change):
            for key in doc["evidence"]:
                escaped_key = key.replace("~", "~0").replace("/", "~1")
                change(doc["evidence"], key, f"/evidence/{escaped_key}")

        self.clone("scientific_review", review)

        def promotion(doc, change):
            for key in ("confirmed_protocol", "report"):
                change(doc, key, f"/{key}")
            change(doc, "review", "/review", "scientific_review")
            change(doc, "source_bridge", "/source_bridge", "source_bridge")

        self.clone("promotion_certificate", promotion)

        def approval(doc, change):
            certificates = 0
            for index, reference in enumerate(doc["evidence"]):
                replacement = None
                if reference["kind"] == "corrected_protocol_confirmation":
                    replacement = "promotion_certificate"
                    certificates += 1
                change(doc["evidence"], index, f"/evidence/{index}", replacement)
            need(certificates == 1, "Expected exactly one original promotion certificate")

        self.clone("comparison_approval", approval)

    def prepare_manifest(self):
        original = self.read_verified(self.root / MANIFEST, MANIFEST_SHA)
        pinned = self.read_verified(
            self.root / EVIDENCE / "training_queue/manifest.csv", MANIFEST_SHA
        )
        need(original == pinned, "Original allocation and pinned queue manifest differ")
        reader = csv.DictReader(io.StringIO(original.decode(), newline=""))
        fieldnames = reader.fieldnames
        rows = list(reader)
        expected = {
            (seed, visibility, model)
            for seed in range(20, 25)
            for visibility in ("radius1", "global")
            for model in MODELS
        }
        need(len(rows) == 40, "Expected the complete forty-row allocation")
        need(
            {(int(row["seed"]), row["ablation_value"], row["model"]) for row in rows} == expected,
            "Manifest condition set differs from the original allocation",
        )
        need(len({row["run_id"] for row in rows}) == 40, "Repeated original run ID")
        changed = []
        for row in rows:
            need(
                None not in row and all(value is not None for value in row.values()),
                "Malformed CSV row",
            )
            binding = json.loads(row["protocol_json"])
            need(
                row["output_root"] == str(OLD_REPO / "runs")
                and row["suite_id"] == "pcp_visibility_budget_cpu_v1"
                and row["ablation"] == "visibility"
                and row["max_n_frames"] == "600000"
                and row["attempt"] == "0"
                and not row["retry_of"]
                and binding["source_sha256"] == SOURCE
                and binding["sha256"] == PROTOCOL_SHA
                and binding["stage"] == "comparison"
                and binding["approval"] == str(DOCUMENTS["comparison_approval"]),
                "Original row differs from frozen source, protocol, placement or attempt",
            )
            for name in ("run_id", "suite_id"):
                need(
                    Path(row[name]).name == row[name] and row[name] not in {"", ".", ".."},
                    "Unsafe original run label",
                )
            after = dict(row)
            after["output_root"] = str(self.root / "runs")
            after["protocol_json"] = json.dumps(
                {**binding, "approval": str(self.destination / "comparison_approval.json")},
                sort_keys=True,
            )
            # Everything else, including authoritative resolved settings and
            # explanatory original commands, is preserved verbatim per CSV cell.
            changed.append(after)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changed)
        self.outputs["manifest.csv"] = output.getvalue().encode()
        return {
            "rows": 40,
            "run_ids": [row["run_id"] for row in rows],
            "original_path": str(self.root / MANIFEST),
            "original_sha256": MANIFEST_SHA,
            "derived_path": str(self.destination / "manifest.csv"),
            "derived_sha256": digest(self.outputs["manifest.csv"]),
            "allowed_changes": ["output_root", "protocol_json.approval"],
            "original_output_root": str(OLD_REPO / "runs"),
            "derived_output_root": str(self.root / "runs"),
            "all_other_csv_cells_equal": True,
            "scientific_settings_conditions_seeds_order_and_attempts_equal": True,
            "command_column": (
                "Historical explanatory command retained; execute using the derived manifest path."
            ),
        }

    def recheck_inputs(self):
        for name, expected in list(self.inputs.items()):
            self.read_verified(Path(name), expected)
        need(source_fingerprint(self.root) == SOURCE, "D4 package source changed")
        baseline = self.root.parent / ".codex-worktrees/pcp-retention-confirmation"
        need(source_fingerprint(baseline) == BASELINE_SOURCE, "Confirmed baseline source changed")

    def build_receipt(self, manifest):
        return {
            "schema_version": 1,
            "checks_passed": True,
            "scope": (
                "Operational path relocation only; historical scientific decisions are unchanged."
            ),
            "originals_modified": False,
            "computation_launched": False,
            "runtime_validated": False,
            "requires_before_computation": (
                "Unmodified protocol runtime and launch checks, plus continuation review."
            ),
            "original_repository": str(OLD_REPO),
            "relocated_repository": str(self.root),
            "source_sha256": SOURCE,
            "confirmed_baseline_source_sha256": BASELINE_SOURCE,
            "protocol_sha256": PROTOCOL_SHA,
            "documents": self.documents,
            "manifest": manifest,
            "input_artifact_sha256": dict(self.inputs),
            "generated_file_sha256": {
                name: digest(payload) for name, payload in self.outputs.items()
            },
            "utility_sha256": digest(Path(__file__).read_bytes()),
        }

    def publish(self):
        need(
            self.destination.is_relative_to(self.root / "results"),
            "Derived output must be under results",
        )
        need(not self.destination.exists(), "Refusing to overwrite existing relocation evidence")
        # Do all source/reference validation before even creating the output directory.
        self.recheck_inputs()
        self.prepare_documents()
        manifest = self.prepare_manifest()
        self.recheck_inputs()
        receipt = self.build_receipt(manifest)
        self.destination.mkdir(parents=True, exist_ok=False)
        for name, payload in self.outputs.items():
            with (self.destination / name).open("xb") as stream:
                stream.write(payload)
            need(
                (self.destination / name).read_bytes() == payload,
                "Relocation output read-back mismatch",
            )
        self.recheck_inputs()
        # Publication of a success receipt is last; interrupted/failed outputs
        # remain visible and are never silently reused by a later invocation.
        with (self.destination / "relocation_receipt.json").open("xb") as stream:
            stream.write(json_bytes(receipt))
        return receipt


def verify_existing(root: Path, destination: Path) -> dict:
    """Read-only verification against independently rebuilt, anchored inputs.

    A self-consistent but modified receipt or operational manifest cannot pass:
    every output is rebuilt from the anchored originals and compared by bytes.
    This verifies relocation; callers must still run the frozen launch/runtime
    checks and enforce the continuation's preservation and resource gates.
    """
    context = Relocation(root, destination)
    need(
        context.destination.is_relative_to(context.root / "results"),
        "Derived output must be under results",
    )
    context.recheck_inputs()
    context.prepare_documents()
    manifest = context.prepare_manifest()
    context.recheck_inputs()
    expected = context.build_receipt(manifest)
    receipt_path = context.destination / "relocation_receipt.json"
    need(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        "Missing or nonordinary relocation receipt",
    )
    receipt_bytes = receipt_path.read_bytes()
    need(
        receipt_bytes == json_bytes(expected),
        "Relocation receipt differs from anchored reconstruction",
    )
    for name, payload in context.outputs.items():
        path = context.destination / name
        need(
            path.is_file() and not path.is_symlink(), f"Missing or nonordinary derived file: {path}"
        )
        need(
            path.read_bytes() == payload,
            f"Derived file differs from anchored reconstruction: {path}",
        )
    context.recheck_inputs()
    need(
        receipt_path.read_bytes() == receipt_bytes, "Relocation receipt changed during verification"
    )
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--verify", action="store_true", help="Verify an existing packet without writes"
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    destination = args.out_dir or root / "results/continuation_20260909/relocation_v2"
    receipt = (
        verify_existing(root, destination)
        if args.verify
        else Relocation(root, destination).publish()
    )
    print(
        json.dumps(
            {
                "checks_passed": True,
                "manifest": receipt["manifest"]["derived_path"],
                "receipt": str(destination / "relocation_receipt.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
