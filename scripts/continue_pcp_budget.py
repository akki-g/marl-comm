"""Recover the interrupted D4 allocation without retrying or replacing policies.

Uses a verified relocation packet and the original frozen workers. Both completed
seed-20 local controls finish fresh full evaluation banks and verified archival
before any unstarted row is submitted. Each later row is trained, evaluated,
independently validated and archived before its worker slot is reused.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, UTC
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from threading import Event, Lock

from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.protocols import content_sha256, load_protocol
from commstudy.experiments.provenance import source_fingerprint
from commstudy.experiments.sweeps import read_manifest, validate_plan


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "69f202bf4fbe998ca3baa7d9b923e5d5bccb3976b51ee0176a20dd30cb73c07e"
MODELS = ("pcp_comm_identity", "pcp_local_capacity", "pcp_broadcast_k0", "pcp_broadcast_k3")
FROZEN_SCRIPTS = (
    "sweep.py",
    "run_pcp_budget_queue.py",
    "archive_pcp_budget_run.py",
    "evaluate_pcp_budget.py",
    "analyze_pcp_budget.py",
)
INITIAL_ARCHIVE_SHA256 = {
    "pcp_comm_identity": "9e0120d070db3a37723fc440a8e9ef5dbd90c7bb23288147ff603ba3c10593b1",
    "pcp_local_capacity": "fbed25c0522a9b67fcd4f701b20603bbcac561206d5eed7a82701ed5167e9a8e",
}
INITIAL_ARCHIVE_HELPER_SHA256 = (
    "a2bcf88ce6029c4545d53c15d22aa88e936457a11af236f6a2d406d75918309c"
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def helper(name):
    spec = importlib.util.spec_from_file_location(f"continuation_{name}", ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(UTC).isoformat()


def verify_frozen_workers(receipt):
    """Bind workers to the reviewed historical snapshot before importing them."""
    snapshot = ROOT / "results/pcp_visibility_budget_cpu_v1/source_snapshot_v2_manifest.json"
    expected = receipt["input_artifact_sha256"].get(str(snapshot.resolve()))
    require(
        expected is not None and sha256(snapshot) == expected,
        "Historical execution-source snapshot is missing or changed.",
    )
    document = json.loads(snapshot.read_text())
    hashes = {}
    for name in FROZEN_SCRIPTS:
        path = ROOT / "scripts" / name
        saved = document["files"][f"repository/scripts/{name}"]["sha256"]
        require(
            path.is_file() and not path.is_symlink() and sha256(path) == saved,
            f"Frozen execution script changed before preflight: {path}",
        )
        hashes[str(path.resolve())] = saved
    return hashes


def verify_initial_archives(initial, old_queue):
    """Require the two reused runs to match their already preserved complete bytes."""
    hashes = {}
    old_root = Path("/Users/akshatguduru/Desktop/Thesis/marl-comm")
    for plan in initial:
        archive = (
            ROOT
            / "results/pcp_visibility_budget_cpu_v1/interim_hold_archives"
            / plan.run_id
            / "run.tar.gz"
        )
        require(
            archive.is_file()
            and not archive.is_symlink()
            and sha256(archive) == INITIAL_ARCHIVE_SHA256[plan.model],
            "Initial run archive differs from the handoff's verified bytes.",
        )
        with tarfile.open(archive, mode="r:gz") as preserved:
            inventory = json.load(preserved.extractfile("ARCHIVE_MANIFEST.json"))
        require(
            inventory["run_id"] == plan.run_id and inventory["source_sha256"] == SOURCE,
            "Initial archive belongs to another run or scientific source.",
        )
        run = plan.output_root / plan.suite_id / plan.run_id
        expected = {}
        for name, entry in inventory["files"].items():
            if not name.startswith("run/"):
                continue
            path = Path(entry["source_path"])
            require(
                path.is_relative_to(old_root) and ".." not in path.parts,
                "Unexpected original archive source path.",
            )
            relocated = ROOT / path.relative_to(old_root)
            require(relocated.is_relative_to(run), "Archive run input lies outside its run.")
            require(
                relocated.is_file()
                and not relocated.is_symlink()
                and sha256(relocated) == entry["sha256"],
                f"Initial run changed since verified preservation: {relocated}",
            )
            expected[str(relocated.resolve())] = entry["sha256"]
        contents = list(run.rglob("*"))
        require(
            all(not path.is_symlink() and (path.is_dir() or path.is_file()) for path in contents),
            "Initial run contains nonordinary evidence.",
        )
        actual = {str(path.resolve()) for path in contents if path.is_file()}
        require(
            len(expected) == inventory["run_file_count"] and set(expected) == actual,
            "Initial run file inventory changed since verified preservation.",
        )
        log = old_queue / f"{plan.run_id}.log"
        require(
            sha256(log) == inventory["files"]["worker/stdout.log"]["sha256"],
            "Original closed worker log differs from its preserved copy.",
        )
        hashes.update(expected)
        hashes[str(archive.resolve())] = INITIAL_ARCHIVE_SHA256[plan.model]
        hashes[str(log.resolve())] = sha256(log)
    return hashes


def partition(plans, old_queue):
    """Only the original two completed controls may have existing run paths."""
    expected = {
        (seed, visibility, model)
        for seed in range(20, 25)
        for visibility in ("radius1", "global")
        for model in MODELS
    }
    require(
        len(plans) == 40 and {(p.seed, p.ablation_value, p.model) for p in plans} == expected,
        "Require the unchanged forty-row allocation.",
    )
    require(len({p.run_id for p in plans}) == 40, "Duplicate run IDs.")
    require(
        len({(p.output_root.resolve(), p.suite_id) for p in plans}) == 1,
        "The allocation must use one managed suite.",
    )
    completed, pending = [], []
    queue = helper("run_pcp_budget_queue.py")
    for plan in sorted(
        plans, key=lambda p: (p.seed, p.ablation_value != "radius1", MODELS.index(p.model))
    ):
        require(
            plan.attempt == 0 and plan.retry_of is None and plan.max_n_frames == 600000,
            "Continuation cannot retry or alter a training budget.",
        )
        require(
            all(
                Path(label).name == label and label not in ("", ".", "..")
                for label in (plan.run_id, plan.suite_id)
            ),
            "Unsafe run identity.",
        )
        run = plan.output_root / plan.suite_id / plan.run_id
        initial = plan.seed == 20 and plan.ablation_value == "radius1" and plan.model in MODELS[:2]
        if initial:
            require(run.is_dir() and not run.is_symlink(), "Initial completed control is missing.")
            status = json.loads((run / "status.json").read_text())
            require(status.get("status") == "completed", "Initial control is not completed.")
            queue.completed_metadata(plan)
            log = old_queue / f"{plan.run_id}.log"
            require(
                log.is_file() and not log.is_symlink(), "Original closed worker log is missing."
            )
            require(
                f"[sweep] {plan.run_id} -> completed" in log.read_text(),
                "Original worker log does not confirm successful closure.",
            )
            completed.append(plan)
        else:
            require(
                not run.exists() and not run.is_symlink(),
                f"Preserve and investigate the existing attempt; do not retry: {run}",
            )
            pending.append(plan)
    return completed, pending


def verify_evaluation_inputs(plan, evaluation, manifest):
    """Bind a completed bank to this exact manifest and the frozen evaluator inputs."""
    report = json.loads((evaluation / "heldout_evaluation.json").read_text())
    retained = json.loads((evaluation / "retained_checkpoints.json").read_text())
    run = plan.output_root / plan.suite_id / plan.run_id
    paths = [
        manifest,
        ROOT / plan.protocol["path"],
        ROOT / "docs/PCP_VISIBILITY_BUDGET_PLAN.md",
        ROOT / "scripts/evaluate_pcp_budget.py",
        *(
            run / name
            for name in (
                "metadata.json",
                "status.json",
                "resolved_config.yaml",
                "metrics.csv",
                "checkpoints/policy_state.pt",
            )
        ),
        *(Path(row["path"]) for row in retained["checkpoints"]),
        *(Path(row["path"]) for row in retained["replay_diagnostics"]),
    ]
    hashes = {str(path.resolve()): sha256(path) for path in paths}
    require(
        report.get("inputs_unchanged") is True and report.get("input_artifact_sha256") == hashes,
        "Evaluation must bind the identical manifest, frozen evaluator and complete run inputs.",
    )
    return hashes


def review_evaluation(plan, evaluation, protocol, manifest):
    analyzer = helper("analyze_pcp_budget.py")
    hashes = {}
    analyzer.validate_row(
        plan,
        evaluation,
        protocol=protocol,
        protocol_hash=content_sha256(protocol),
        plan_hash=sha256(ROOT / "docs/PCP_VISIBILITY_BUDGET_PLAN.md"),
        hashes=hashes,
    )
    hashes.update(verify_evaluation_inputs(plan, evaluation, manifest))
    require(
        all(sha256(path) == digest for path, digest in hashes.items()),
        "Evaluation inputs changed during independent validation.",
    )
    return {"checks_passed": True, "validated_utc": now(), "input_artifact_sha256": hashes}


def evaluation_files(directory):
    """Hash an ordinary completed evaluation tree without following links."""
    require(
        directory.is_dir() and not directory.is_symlink(),
        "Existing initial evaluation must be an ordinary directory.",
    )
    files = {}
    for path in sorted(directory.rglob("*")):
        require(
            not path.is_symlink() and (path.is_file() or path.is_dir()),
            "Initial evaluation contains nonordinary evidence.",
        )
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    require(bool(files), "Existing initial evaluation is empty or incomplete.")
    status = directory / "status.json"
    require(
        status.is_file() and json.loads(status.read_text()).get("status") == "completed",
        "Preserve the partial initial evaluation; wait for its full bank to complete.",
    )
    return files


def copy_initial_evaluation(source, destination, expected):
    """Preserve every source file and verify an exclusive complete-bank copy."""
    require(evaluation_files(source) == expected, "Initial evaluation changed before copying.")
    require(
        not destination.exists() and not destination.is_symlink(),
        "Refusing to overwrite an evaluation destination.",
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name, entry in expected.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with (source / name).open("rb") as input_stream, target.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
        require(
            sha256(target) == entry["sha256"] and target.stat().st_size == entry["bytes"],
            f"Initial evaluation copy failed read-back verification: {target}",
        )
    require(
        evaluation_files(source) == expected and evaluation_files(destination) == expected,
        "Initial evaluation inputs or copied inventory changed during preservation.",
    )
    return {
        "checks_passed": True,
        "source": str(source),
        "destination": str(destination),
        "files": expected,
        "original_files_removed": False,
        "source_before_after_equal": True,
        "all_copied_files_rehashed": True,
    }


class Continuation:
    def __init__(self, relocation, out_dir, old_queue, jobs, initial_evaluations=None):
        require(type(jobs) is int and 1 <= jobs <= 2, "At most two CPU worker slots are allowed.")
        self.relocation = relocation.resolve()
        self.out_dir = out_dir.resolve()
        self.old_queue = old_queue.resolve()
        self.jobs = jobs
        self.initial_evaluations = (
            initial_evaluations.resolve() if initial_evaluations is not None else None
        )
        self.initial_reuse = {}
        self.initial_archive_reuse = {}
        self.manifest = self.relocation / "manifest.csv"
        self.relocator = helper("relocate_pcp_budget.py")
        self.archive_dir = self.out_dir / "run_archives"
        self.evaluations = self.out_dir / "evaluations"
        self.destination = self.out_dir / "queue_status.json"
        self.frozen_hashes = {}
        self.stop_expansion = Event()
        self.expansion_lock = Lock()

    def stop_new_training(self):
        # Submission and failure publication share one ordering point. The
        # lock is never held while a worker performs computation or waits.
        with self.expansion_lock:
            self.stop_expansion.set()

    def preflight(self):
        self.initial_reuse = {}
        self.initial_archive_reuse = {}
        receipt = self.relocator.verify_existing(ROOT, self.relocation)
        frozen_workers = verify_frozen_workers(receipt)
        self.queue = helper("run_pcp_budget_queue.py")
        require(source_fingerprint(ROOT) == SOURCE, "Frozen scientific package changed.")
        require(
            not self.out_dir.exists(), "Preserve existing continuation evidence; choose a new path."
        )
        require(
            self.out_dir.is_relative_to(ROOT / "results"),
            "Continuation outputs must be under the repository results directory.",
        )
        require(
            not self.out_dir.is_relative_to(self.relocation)
            and not self.out_dir.is_relative_to(self.old_queue),
            "Continuation outputs must be outside relocation and original queue evidence.",
        )
        plans = read_manifest(self.manifest)
        for plan in plans:
            validate_plan(plan, config_root=ROOT / "configs", repo_root=ROOT, check_runtime=True)
            run = (plan.output_root / plan.suite_id / plan.run_id).resolve()
            require(not self.out_dir.is_relative_to(run), "Derived outputs must be outside runs.")
        initial, pending = partition(plans, self.old_queue)
        initial_inputs = verify_initial_archives(initial, self.old_queue)
        self.protocol = load_protocol(ROOT / plans[0].protocol["path"])
        if self.initial_evaluations is not None:
            require(
                not self.out_dir.is_relative_to(self.initial_evaluations),
                "Continuation output must be outside preserved initial evaluation inputs.",
            )
            if self.initial_evaluations.exists() or self.initial_evaluations.is_symlink():
                require(
                    self.initial_evaluations.is_dir() and not self.initial_evaluations.is_symlink(),
                    "Initial evaluation root must be an ordinary directory.",
                )
            for plan in initial:
                candidate = self.initial_evaluations / plan.run_id
                if candidate.exists() or candidate.is_symlink():
                    inventory = evaluation_files(candidate)
                    validation = review_evaluation(plan, candidate, self.protocol, self.manifest)
                    require(
                        evaluation_files(candidate) == inventory,
                        "Initial evaluation changed during preflight review.",
                    )
                    self.initial_reuse[plan.run_id] = {
                        "source": str(candidate),
                        "files": inventory,
                        "validation": validation,
                    }
                    initial_inputs.update(
                        {
                            str((candidate / name).resolve()): entry["sha256"]
                            for name, entry in inventory.items()
                        }
                    )
        archive_helper_path = ROOT / "scripts/reuse_pcp_initial_archive.py"
        require(
            archive_helper_path.is_file() and not archive_helper_path.is_symlink()
            and sha256(archive_helper_path) == INITIAL_ARCHIVE_HELPER_SHA256,
            "Initial archive reuse helper differs from the reviewed operational version.",
        )
        self.archive_reuse_helper = helper("reuse_pcp_initial_archive.py")
        for plan in initial:
            preserved = self.archive_reuse_helper.verify_original(ROOT, plan, self.old_queue)
            self.initial_archive_reuse[plan.run_id] = preserved
            initial_inputs.update(preserved["input_artifact_sha256"])
        storage = self.queue.storage_preflight([plans[0].output_root, self.out_dir])
        require(storage["checks_passed"], "Restore the required eight GiB before continuation.")
        files = [
            Path(__file__),
            ROOT / "scripts/relocate_pcp_budget.py",
            archive_helper_path,
            ROOT / "scripts/sweep.py",
            ROOT / "scripts/run_pcp_budget_queue.py",
            ROOT / "scripts/archive_pcp_budget_run.py",
            ROOT / "scripts/evaluate_pcp_budget.py",
            ROOT / "scripts/analyze_pcp_budget.py",
            ROOT / "docs/PCP_VISIBILITY_BUDGET_PLAN.md",
            *(self.relocation / name for name in receipt["generated_file_sha256"]),
            self.relocation / "relocation_receipt.json",
        ]
        files.extend(self.old_queue / f"{plan.run_id}.log" for plan in initial)
        files.extend(ROOT / path for path in {plan.protocol["path"] for plan in plans})
        self.frozen_hashes = {str(path.resolve()): sha256(path) for path in files}
        self.frozen_hashes.update(initial_inputs)
        require(
            all(self.frozen_hashes[path] == digest for path, digest in frozen_workers.items()),
            "Frozen workers changed during preflight.",
        )
        self.initial, self.pending, self.plans = initial, pending, plans
        return {
            "checks_passed": True,
            "recorded_utc": now(),
            "workers": self.jobs,
            "completed_training_to_evaluate": [p.run_id for p in initial],
            "unstarted_training": [p.run_id for p in pending],
            "initial_evaluations_reused": self.initial_reuse,
            "initial_archives_reused": self.initial_archive_reuse,
            "operational_recovery_version": 4,
            "storage": storage,
            "source_sha256": SOURCE,
            "operational_input_sha256": self.frozen_hashes,
            "original_evidence_preserved": True,
            "scientific_rows_unchanged": True,
        }

    def recheck(self):
        require(
            all(sha256(path) == digest for path, digest in self.frozen_hashes.items()),
            "A bound operational input changed after preflight.",
        )
        require(source_fingerprint(ROOT) == SOURCE, "Frozen scientific package changed.")
        self.relocator.verify_existing(ROOT, self.relocation)
        for retained in self.initial_reuse.values():
            require(
                evaluation_files(Path(retained["source"])) == retained["files"],
                "Preserved initial evaluation file set or bytes changed.",
            )
        storage = self.queue.storage_preflight([self.plans[0].output_root, self.out_dir])
        require(storage["checks_passed"], "Storage fell below eight GiB; expansion stopped.")

    def command(self, command, log, *, new_training=False):
        self.recheck()
        if new_training:
            require(
                not self.stop_expansion.is_set(),
                "Expansion stopped before this training process launched.",
            )
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        require(result.returncode == 0, f"Worker failed ({result.returncode}); inspect {log}.")
        self.recheck()

    def worker(self, plan, *, train):
        result = {"training_reused": not train, "started_utc": now()}
        try:
            if train:
                run = plan.output_root / plan.suite_id / plan.run_id
                require(
                    not run.exists() and not run.is_symlink(),
                    "Attempt appeared after preflight; refusing retry.",
                )
                worker_log = self.out_dir / f"{plan.run_id}.log"
                result["stage"] = "training"
                self.command(
                    [
                        sys.executable,
                        str(ROOT / "scripts/sweep.py"),
                        "--manifest",
                        str(self.manifest),
                        "--run-id",
                        plan.run_id,
                        "--run",
                    ],
                    worker_log,
                    new_training=True,
                )
            else:
                worker_log = self.old_queue / f"{plan.run_id}.log"
            self.queue.completed_metadata(plan)
            result["training_completed"] = True
            result["stage"] = "evaluation"
            evaluation = self.evaluations / plan.run_id
            retained = self.initial_reuse.get(plan.run_id) if not train else None
            result["evaluation_reused"] = retained is not None
            if retained is not None:
                self.recheck()
                result["stage"] = "evaluation_preservation"
                result["evaluation_copy"] = copy_initial_evaluation(
                    Path(retained["source"]), evaluation, retained["files"]
                )
                self.recheck()
            else:
                self.command(
                    [
                        sys.executable,
                        str(ROOT / "scripts/evaluate_pcp_budget.py"),
                        "--manifest",
                        str(self.manifest),
                        "--run-id",
                        plan.run_id,
                        "--out-dir",
                        str(evaluation),
                    ],
                    self.out_dir / f"{plan.run_id}.evaluation.log",
                )
            result["stage"] = "evaluation_review"
            self.recheck()
            validation = review_evaluation(plan, evaluation, self.protocol, self.manifest)
            review_path = self.out_dir / f"{plan.run_id}.review.json"
            with review_path.open("x") as stream:
                json.dump(validation, stream, indent=2, sort_keys=True)
                stream.write("\n")
            result["evaluation_checks_passed"] = True
            result["stage"] = "artifact_preservation"
            self.recheck()
            if train:
                result["archive"] = self.queue.archive_completed(
                    plan, self.manifest, self.archive_dir, worker_log, self.out_dir,
                )
            else:
                require(
                    plan.run_id in self.initial_archive_reuse,
                    "Initial archive reuse was not verified in preflight.",
                )
                result["archive"] = self.archive_reuse_helper.copy_verified(
                    ROOT, plan, self.old_queue, self.manifest,
                    self.relocation / "relocation_receipt.json",
                    self.archive_dir / plan.run_id,
                    self.initial_archive_reuse[plan.run_id],
                )
            self.recheck()
            result.update({"status": "completed", "stage": "complete"})
        except Exception as error:
            self.stop_new_training()
            result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        result["finished_utc"] = now()
        return result

    def phase(self, plans, *, train):
        pending = iter(plans)
        stopped = False
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            active = {}

            def submit():
                nonlocal stopped
                if stopped or self.stop_expansion.is_set():
                    stopped = True
                    return
                try:
                    self.recheck()
                except Exception as error:
                    stopped = True
                    self.stop_new_training()
                    self.record["stop_reason"] = f"{type(error).__name__}: {error}"
                    return
                plan = next(pending, None)
                if plan is not None:
                    with self.expansion_lock:
                        if self.stop_expansion.is_set():
                            stopped = True
                            return
                        self.record["rows"][plan.run_id] = {
                            "status": "running",
                            "started_utc": now(),
                            "training_reused": not train,
                        }
                        active[pool.submit(self.worker, plan, train=train)] = plan
                    atomic_write_json(self.destination, self.record)
                    print(
                        f"START {'train' if train else 'recover_evaluation'} {plan.run_id}",
                        flush=True,
                    )

            for _ in range(self.jobs):
                submit()
            while active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    plan = active.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as error:
                        self.stop_new_training()
                        outcome = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    self.record["rows"][plan.run_id] = outcome
                    stopped |= outcome["status"] != "completed"
                    if stopped:
                        self.stop_new_training()
                    print(f"{outcome['status'].upper()} {plan.run_id}", flush=True)
                atomic_write_json(self.destination, self.record)
                if not stopped:
                    for _ in finished:
                        submit()
        return not stopped

    def run(self):
        preflight = self.preflight()
        self.out_dir.mkdir(parents=True)
        atomic_write_json(self.out_dir / "preflight.json", preflight)
        self.record = {
            "schema_version": 1,
            "operational_recovery_version": 4,
            "started_utc": now(),
            "status": "running",
            "phase": "recover_initial_evaluations",
            "workers": self.jobs,
            "manifest": str(self.manifest),
            "manifest_sha256": sha256(self.manifest),
            "no_training_retries": True,
            "stop_on_failure": True,
            "evaluation_and_archive_required_before_next_submission": True,
            "rows": {p.run_id: {"status": "pending"} for p in self.plans},
        }
        atomic_write_json(self.destination, self.record)
        passed = self.phase(self.initial, train=False)
        if passed:
            self.record["phase"] = "remaining_training_evaluation_archival"
            atomic_write_json(self.destination, self.record)
            passed = self.phase(self.pending, train=True)
        if passed:
            self.record["phase"] = "final_analysis"
            atomic_write_json(self.destination, self.record)
            try:
                self.command(
                    [
                        sys.executable,
                        str(ROOT / "scripts/analyze_pcp_budget.py"),
                        "--manifest",
                        str(self.manifest),
                        "--protocol",
                        str(ROOT / "configs/protocols/pcp_visibility_budget_cpu_v1_frozen.yaml"),
                        "--evaluations-dir",
                        str(self.evaluations),
                        "--out-dir",
                        str(self.out_dir / "final_analysis"),
                    ],
                    self.out_dir / "analysis.log",
                )
            except Exception as error:
                passed = False
                self.record["stop_reason"] = f"{type(error).__name__}: {error}"
        self.record.update(
            {"status": "completed" if passed else "stopped_after_failure", "finished_utc": now()}
        )
        atomic_write_json(self.destination, self.record)
        return 0 if passed else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relocation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--original-queue", type=Path, required=True)
    parser.add_argument(
        "--initial-evaluations",
        type=Path,
        help="Preserved complete banks eligible for the two initial controls only.",
    )
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--execute", action="store_true", help="Execute after the strict preflight."
    )
    args = parser.parse_args(argv)
    continuation = Continuation(
        args.relocation, args.out_dir, args.original_queue, args.jobs, args.initial_evaluations
    )
    if args.execute:
        return continuation.run()
    print(json.dumps(continuation.preflight(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
