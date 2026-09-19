"""Exercise interruption recovery without launching training or policy rollouts."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
from pathlib import Path
from threading import Event
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/continue_pcp_budget.py"
SPEC = importlib.util.spec_from_file_location("d4_continuation", MODULE_PATH)
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        root_patch = patch.object(continuation, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.calls = []
        self.fail_stage = None
        self.fail_id = None
        self.queue = SimpleNamespace(
            completed_metadata=Mock(return_value={}),
            storage_preflight=Mock(return_value={"checks_passed": True}),
            archive_completed=Mock(side_effect=self.archive),
        )
        self.relocator = SimpleNamespace(verify_existing=Mock(return_value={}))
        self.archive_reuse = SimpleNamespace(
            copy_verified=Mock(side_effect=lambda _root, plan, *_args: self.archive(plan)),
        )
        helper_patch = patch.object(
            continuation,
            "helper",
            side_effect=lambda name: (
                self.relocator if name == "relocate_pcp_budget.py" else self.queue
            ),
        )
        helper_patch.start()
        self.addCleanup(helper_patch.stop)
        self.popen_patch = patch.object(continuation.subprocess, "run", side_effect=self.subprocess)
        self.popen_patch.start()
        self.addCleanup(self.popen_patch.stop)
        review_patch = patch.object(continuation, "review_evaluation", side_effect=self.review)
        review_patch.start()
        self.addCleanup(review_patch.stop)

    def plan(self, run_id):
        return SimpleNamespace(run_id=run_id, output_root=self.root / "runs", suite_id="suite")

    def harness(self, pending=4, jobs=2):
        item = continuation.Continuation(
            self.root / "results/relocation",
            self.root / "results/continuation",
            self.root / "results/original_queue",
            jobs,
        )
        item.queue = self.queue
        item.archive_reuse_helper = self.archive_reuse
        item.initial = [self.plan("identity"), self.plan("capacity")]
        item.pending = [self.plan(f"pending_{index}") for index in range(pending)]
        item.plans = item.initial + item.pending
        item.initial_archive_reuse = {plan.run_id: {"checks_passed": True}
                                      for plan in item.initial}
        item.protocol = {"status": "frozen"}
        item.old_queue.mkdir(parents=True)
        item.relocation.mkdir(parents=True)
        item.manifest.write_text("fixed manifest\n")
        for plan in item.initial:
            (plan.output_root / plan.suite_id / plan.run_id).mkdir(parents=True)
            (item.old_queue / f"{plan.run_id}.log").write_text("original closed worker log\n")
        item.preflight = Mock(return_value={"checks_passed": True})
        item.recheck = Mock()
        return item

    def subprocess(self, command, **kwargs):
        script = Path(command[1]).name
        stage = {
            "sweep.py": "training",
            "evaluate_pcp_budget.py": "evaluation",
            "analyze_pcp_budget.py": "analysis",
        }[script]
        run_id = command[command.index("--run-id") + 1] if "--run-id" in command else "cohort"
        self.calls.append((stage, run_id, command))
        if (stage, run_id) == (self.fail_stage, self.fail_id):
            return SimpleNamespace(returncode=2)
        if stage == "training":
            (self.root / "runs/suite" / run_id).mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    def archive(self, plan, *_args):
        self.calls.append(("archive", plan.run_id, None))
        if ("archive", plan.run_id) == (self.fail_stage, self.fail_id):
            raise ValueError("preservation failure")
        return {"checks_passed": True}

    def review(self, plan, *_args):
        self.calls.append(("review", plan.run_id, None))
        if ("review", plan.run_id) == (self.fail_stage, self.fail_id):
            raise ValueError("held-out scientific audit failed")
        return {"checks_passed": True}

    def test_initial_pair_barrier_and_no_retraining(self):
        item = self.harness(pending=3)
        capacity_started, release_capacity, identity_archived = Event(), Event(), Event()
        original_subprocess, original_archive = self.subprocess, self.archive

        def subprocess(command, **kwargs):
            if Path(command[1]).name == "evaluate_pcp_budget.py" and "capacity" in command:
                capacity_started.set()
                self.assertTrue(release_capacity.wait(5), "Test did not release paired evaluator")
            return original_subprocess(command, **kwargs)

        def archive(plan, *args):
            result = original_archive(plan, *args)
            if plan.run_id == "identity":
                identity_archived.set()
            return result

        self.queue.archive_completed.side_effect = archive
        self.archive_reuse.copy_verified.side_effect = (
            lambda _root, plan, *args: archive(plan, *args)
        )
        with patch.object(continuation.subprocess, "run", side_effect=subprocess):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(item.run)
                try:
                    self.assertTrue(capacity_started.wait(5))
                    self.assertTrue(identity_archived.wait(5))
                    self.assertFalse(any(stage == "training" for stage, _, _ in self.calls))
                finally:
                    release_capacity.set()
                self.assertEqual(future.result(timeout=5), 0)
        training = [run_id for stage, run_id, _ in self.calls if stage == "training"]
        self.assertEqual(training, [plan.run_id for plan in item.pending])
        self.assertEqual(
            {call.args[1].run_id for call in self.archive_reuse.copy_verified.call_args_list},
            {"identity", "capacity"},
        )
        self.assertCountEqual(
            [call.args[0].run_id for call in self.queue.archive_completed.call_args_list],
            training,
        )
        for run_id in ("identity", "capacity", *training):
            stages = [stage for stage, name, _ in self.calls if name == run_id]
            self.assertEqual(
                stages,
                ([] if run_id in {"identity", "capacity"} else ["training"])
                + ["evaluation", "review", "archive"],
            )
        analysis = next(command for stage, _, command in self.calls if stage == "analysis")
        self.assertEqual(
            analysis[analysis.index("--protocol") + 1],
            str(self.root / "configs/protocols/pcp_visibility_budget_cpu_v1_frozen.yaml"),
        )

    def test_initial_archive_without_preflight_verification_stops_expansion(self):
        item = self.harness(jobs=1)
        item.initial_archive_reuse.pop("identity")
        self.assertEqual(item.run(), 2)
        self.assertIn("not verified in preflight", item.record["rows"]["identity"]["error"])
        self.assertFalse(any(stage == "training" for stage, _, _ in self.calls))
        self.archive_reuse.copy_verified.assert_not_called()
        self.queue.archive_completed.assert_not_called()

    def test_new_training_cannot_use_an_injected_initial_archive_candidate(self):
        item = self.harness(pending=1, jobs=1)
        plan = item.pending[0]
        item.initial_archive_reuse[plan.run_id] = {"checks_passed": True}
        item.out_dir.mkdir()
        result = item.worker(plan, train=True)
        self.assertEqual(result["status"], "completed")
        self.archive_reuse.copy_verified.assert_not_called()
        self.queue.archive_completed.assert_called_once()
        self.assertEqual(self.queue.archive_completed.call_args.args[0], plan)

    def test_each_initial_failure_prevents_all_new_training(self):
        for stage in ("evaluation", "review", "archive"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                # Each case has a new output path and preserved independent attempt.
                self.root = Path(temporary).resolve()
                with patch.object(continuation, "ROOT", self.root):
                    item = self.harness()
                    self.calls.clear()
                    self.fail_stage, self.fail_id = stage, "identity"
                    self.assertEqual(item.run(), 2)
                    self.assertFalse(any(kind == "training" for kind, _, _ in self.calls))
                    self.assertEqual(item.record["rows"]["identity"]["status"], "failed")

    def test_any_later_stage_failure_stops_slot_reuse(self):
        for stage in ("training", "evaluation", "review", "archive"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                self.root = Path(temporary).resolve()
                with patch.object(continuation, "ROOT", self.root):
                    item = self.harness(jobs=1)
                    self.calls.clear()
                    self.fail_stage, self.fail_id = stage, "pending_0"
                    self.assertEqual(item.run(), 2)
                    training = [name for kind, name, _ in self.calls if kind == "training"]
                    self.assertEqual(training, ["pending_0"])
                    self.assertTrue(item.stop_expansion.is_set())
                    self.assertEqual(item.record["rows"]["pending_1"]["status"], "pending")

    def test_worker_sets_shared_stop_before_its_future_is_observed(self):
        item = self.harness()
        item.out_dir.mkdir(parents=True)
        self.fail_stage, self.fail_id = "evaluation", "identity"
        result = item.worker(item.initial[0], train=False)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(item.stop_expansion.is_set())
        item.record = {"rows": {p.run_id: {"status": "pending"} for p in item.pending}}
        item.worker = Mock()
        self.assertFalse(item.phase(item.pending, train=True))
        item.worker.assert_not_called()

    def test_failure_during_slow_slot_recheck_prevents_next_submission(self):
        item = self.harness(jobs=1)
        item.out_dir.mkdir(parents=True)
        item.record = {"rows": {p.run_id: {"status": "pending"} for p in item.pending}}
        item.worker = Mock(return_value={"status": "completed"})
        rechecks = 0

        def recheck():
            nonlocal rechecks
            rechecks += 1
            if rechecks == 2:
                # A sibling reports a failure after submit's initial check,
                # while expensive input hashing is still in progress.
                item.stop_new_training()

        item.recheck.side_effect = recheck
        self.assertFalse(item.phase(item.pending, train=True))
        item.worker.assert_called_once_with(item.pending[0], train=True)
        self.assertEqual(item.record["rows"]["pending_1"]["status"], "pending")

    def test_submitted_worker_still_skips_training_if_failure_arrives_before_launch(self):
        item = self.harness()
        item.out_dir.mkdir(parents=True)
        item.recheck.side_effect = item.stop_new_training
        result = item.worker(item.pending[0], train=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("stopped before this training process launched", result["error"])
        self.assertFalse(self.calls)

    def test_new_attempt_or_broken_symlink_is_never_retried(self):
        item = self.harness()
        item.out_dir.mkdir(parents=True)
        plan = item.pending[0]
        run = plan.output_root / plan.suite_id / plan.run_id
        run.symlink_to(self.root / "missing preserved attempt")
        result = item.worker(plan, train=True)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(run.is_symlink())
        self.assertFalse(self.calls)

    def test_existing_review_is_preserved_and_stops_archival(self):
        item = self.harness()
        item.out_dir.mkdir(parents=True)
        report = item.out_dir / "identity.review.json"
        report.write_text("preserved prior review")
        result = item.worker(item.initial[0], train=False)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(report.read_text(), "preserved prior review")
        self.assertFalse(any(stage == "archive" for stage, _, _ in self.calls))

    def completed_bank(self, item, run_id):
        source = self.root / "results/preserved_initial_evaluations" / run_id
        source.mkdir(parents=True)
        (source / "status.json").write_text('{"status":"completed"}')
        (source / "heldout_evaluation.json").write_text('{"retained_bank":"full"}')
        item.initial_reuse[run_id] = {
            "source": str(source),
            "files": continuation.evaluation_files(source),
        }
        return source

    def test_completed_initial_bank_is_verified_copied_reviewed_and_never_rerun(self):
        item = self.harness()
        item.out_dir.mkdir(parents=True)
        source = self.completed_bank(item, "identity")
        original = {p.name: p.read_bytes() for p in source.iterdir()}
        result = item.worker(item.initial[0], train=False)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["evaluation_reused"])
        self.assertTrue(result["evaluation_copy"]["all_copied_files_rehashed"])
        self.assertEqual([stage for stage, _, _ in self.calls], ["review", "archive"])
        copied = item.evaluations / "identity"
        self.assertEqual({p.name: p.read_bytes() for p in copied.iterdir()}, original)
        self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, original)

    def test_reuse_does_not_apply_to_new_training_rows(self):
        item = self.harness()
        item.out_dir.mkdir(parents=True)
        self.completed_bank(item, "pending_0")
        result = item.worker(item.pending[0], train=True)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["evaluation_reused"])
        self.assertEqual(
            [stage for stage, _, _ in self.calls], ["training", "evaluation", "review", "archive"]
        )

    def test_partial_candidate_bank_is_rejected_and_unchanged(self):
        item = self.harness()
        source = self.completed_bank(item, "identity")
        status = source / "status.json"
        status.unlink()
        with self.assertRaisesRegex(ValueError, "partial initial evaluation"):
            continuation.evaluation_files(source)
        self.assertFalse(status.exists())

    def test_copy_failure_stops_expansion_and_preserves_source(self):
        item = self.harness()
        item.out_dir.mkdir(parents=True)
        source = self.completed_bank(item, "identity")
        original = (source / "heldout_evaluation.json").read_bytes()
        with patch.object(
            continuation.shutil, "copyfileobj", side_effect=OSError("storage failure")
        ):
            result = item.worker(item.initial[0], train=False)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(item.stop_expansion.is_set())
        self.assertEqual((source / "heldout_evaluation.json").read_bytes(), original)
        self.assertFalse(any(stage == "archive" for stage, _, _ in self.calls))

    def test_copy_rejects_source_mutation_or_existing_destination(self):
        item = self.harness()
        source = self.completed_bank(item, "identity")
        expected = item.initial_reuse["identity"]["files"]
        target = item.evaluations / "identity"
        target.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "overwrite an evaluation destination"):
            continuation.copy_initial_evaluation(source, target, expected)
        (source / "extra.json").write_text("unbound new evidence")
        with self.assertRaisesRegex(ValueError, "changed before copying"):
            continuation.copy_initial_evaluation(source, item.evaluations / "other", expected)

    def test_preflight_rejects_partial_bank_before_any_outputs_or_workers(self):
        item = self.harness()
        del item.preflight
        source = self.completed_bank(item, "identity")
        (source / "status.json").unlink()
        item.initial_evaluations = source.parent
        for plan in item.plans:
            plan.protocol = {"path": "configs/frozen.yaml"}
        with (
            patch.object(continuation, "verify_frozen_workers", return_value={}),
            patch.object(continuation, "source_fingerprint", return_value=continuation.SOURCE),
            patch.object(continuation, "read_manifest", return_value=item.plans),
            patch.object(continuation, "validate_plan"),
            patch.object(continuation, "partition", return_value=(item.initial, item.pending)),
            patch.object(continuation, "verify_initial_archives", return_value={}),
            patch.object(continuation, "load_protocol", return_value=item.protocol),
        ):
            with self.assertRaisesRegex(ValueError, "partial initial evaluation"):
                item.preflight()
        self.assertFalse(item.out_dir.exists())
        self.assertFalse(self.calls)

    def test_operation_or_storage_failure_prevents_subprocess_and_expansion(self):
        for error in ("A bound operational input changed", "Storage fell below eight GiB"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                self.root = Path(temporary).resolve()
                with patch.object(continuation, "ROOT", self.root):
                    item = self.harness(jobs=1)
                    item.recheck.side_effect = ValueError(error)
                    self.calls.clear()
                    self.assertEqual(item.run(), 2)
                    self.assertFalse(self.calls)
                    self.assertIn(error, item.record["stop_reason"])

    def test_recheck_detects_operational_tamper_storage_and_relocation_failure(self):
        item = self.harness()
        del item.recheck  # Exercise the actual guard after configuring controlled evidence.
        path = item.root_marker = self.root / "frozen_worker.py"
        path.write_text("original")
        item.frozen_hashes = {str(path): continuation.sha256(path)}
        with patch.object(continuation, "source_fingerprint", return_value=continuation.SOURCE):
            item.recheck()
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "bound operational input changed"):
                item.recheck()
            path.write_text("original")
            self.queue.storage_preflight.return_value = {"checks_passed": False}
            with self.assertRaisesRegex(ValueError, "Storage fell below"):
                item.recheck()
            self.queue.storage_preflight.return_value = {"checks_passed": True}
            self.relocator.verify_existing.side_effect = ValueError("relocation digest mismatch")
            with self.assertRaisesRegex(ValueError, "relocation digest mismatch"):
                item.recheck()

    def test_frozen_workers_are_verified_before_launch_pins_are_accepted(self):
        snapshot = (
            self.root / "results/pcp_visibility_budget_cpu_v1/source_snapshot_v2_manifest.json"
        )
        snapshot.parent.mkdir(parents=True)
        files = {}
        for name in continuation.FROZEN_SCRIPTS:
            path = self.root / "scripts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# reviewed {name}\n")
            files[f"repository/scripts/{name}"] = {"sha256": continuation.sha256(path)}
        snapshot.write_text(json.dumps({"files": files}))
        receipt = {"input_artifact_sha256": {str(snapshot): continuation.sha256(snapshot)}}
        self.assertEqual(len(continuation.verify_frozen_workers(receipt)), 5)
        (self.root / "scripts/sweep.py").write_text("# modified before preflight\n")
        with self.assertRaisesRegex(ValueError, "Frozen execution script changed before preflight"):
            continuation.verify_frozen_workers(receipt)

    def test_worker_slots_cannot_exceed_two(self):
        for jobs in (0, 3, True):
            with self.subTest(jobs=jobs), self.assertRaisesRegex(ValueError, "At most two"):
                self.harness(jobs=jobs)

    def test_preflight_rejects_outputs_inside_relocation_and_checks_receipt_first(self):
        item = self.harness()
        del item.preflight
        item.out_dir = item.relocation / "new_output"
        with (
            patch.object(continuation, "verify_frozen_workers", return_value={}),
            patch.object(continuation, "source_fingerprint", return_value=continuation.SOURCE),
        ):
            with self.assertRaisesRegex(ValueError, "outside relocation"):
                item.preflight()
        self.relocator.verify_existing.assert_called_once_with(self.root, item.relocation)


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.old_queue = self.root / "original_queue"
        self.old_queue.mkdir()
        self.plans = []
        for model in continuation.MODELS:
            for visibility in ("radius1", "global"):
                for seed in range(20, 25):
                    plan = SimpleNamespace(
                        seed=seed,
                        model=model,
                        ablation_value=visibility,
                        attempt=0,
                        retry_of=None,
                        max_n_frames=600000,
                        output_root=self.root / "runs",
                        suite_id="suite",
                        run_id=f"{model}__{visibility}__{seed}",
                    )
                    self.plans.append(plan)
                    if seed == 20 and visibility == "radius1" and model in continuation.MODELS[:2]:
                        run = plan.output_root / plan.suite_id / plan.run_id
                        run.mkdir(parents=True)
                        (run / "status.json").write_text('{"status":"completed"}')
                        (self.old_queue / f"{plan.run_id}.log").write_text(
                            f"[sweep] {plan.run_id} -> completed\n"
                        )
        queue = SimpleNamespace(completed_metadata=Mock())
        helper_patch = patch.object(continuation, "helper", return_value=queue)
        helper_patch.start()
        self.addCleanup(helper_patch.stop)

    def test_partition_keeps_original_order_and_exact_two_controls(self):
        initial, pending = continuation.partition(self.plans, self.old_queue)
        self.assertEqual([p.model for p in initial], list(continuation.MODELS[:2]))
        self.assertEqual(len(pending), 38)
        self.assertEqual([p.model for p in pending[:2]], list(continuation.MODELS[2:]))
        self.assertEqual(
            [(p.seed, p.ablation_value, p.model) for p in initial + pending],
            sorted(
                [(p.seed, p.ablation_value, p.model) for p in self.plans],
                key=lambda values: (
                    values[0],
                    values[1] != "radius1",
                    continuation.MODELS.index(values[2]),
                ),
            ),
        )

    def test_existing_noninitial_attempt_is_preserved_even_if_completed(self):
        plan = next(p for p in self.plans if p.seed == 21)
        run = plan.output_root / plan.suite_id / plan.run_id
        run.mkdir(parents=True)
        (run / "status.json").write_text('{"status":"completed"}')
        with self.assertRaisesRegex(ValueError, "existing attempt; do not retry"):
            continuation.partition(self.plans, self.old_queue)

    def test_initial_incomplete_control_is_not_retrained(self):
        plan = self.plans[0]
        status = plan.output_root / plan.suite_id / plan.run_id / "status.json"
        status.write_text('{"status":"running"}')
        with self.assertRaisesRegex(ValueError, "Initial control is not completed"):
            continuation.partition(self.plans, self.old_queue)

    def test_initial_log_must_record_completed_worker_closure(self):
        next(self.old_queue.glob("*.log")).write_text("interrupted worker")
        with self.assertRaisesRegex(ValueError, "successful closure"):
            continuation.partition(self.plans, self.old_queue)


class ArchivePreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan = SimpleNamespace(
            run_id="identity",
            model="pcp_comm_identity",
            output_root=self.root / "runs",
            suite_id="suite",
        )
        self.run = self.plan.output_root / self.plan.suite_id / self.plan.run_id
        self.actor = self.run / "checkpoints/policy_state.pt"
        self.actor.parent.mkdir(parents=True)
        self.actor.write_bytes(b"original final actor bytes")
        self.old_queue = self.root / "results/original_queue"
        self.old_queue.mkdir(parents=True)
        log = self.old_queue / "identity.log"
        log.write_text("original closed log")
        original_root = Path("/Users/akshatguduru/Desktop/Thesis/marl-comm")
        inventory = {
            "run_id": "identity",
            "source_sha256": continuation.SOURCE,
            "run_file_count": 1,
            "files": {
                "run/checkpoints/policy_state.pt": {
                    "source_path": str(original_root / self.actor.relative_to(self.root)),
                    "sha256": continuation.sha256(self.actor),
                },
                "worker/stdout.log": {"sha256": continuation.sha256(log)},
            },
        }
        self.archive = (
            self.root
            / "results/pcp_visibility_budget_cpu_v1/interim_hold_archives"
            / self.plan.run_id
            / "run.tar.gz"
        )
        self.archive.parent.mkdir(parents=True)
        with tarfile.open(self.archive, "w:gz") as stream:
            payload = json.dumps(inventory).encode()
            info = tarfile.TarInfo("ARCHIVE_MANIFEST.json")
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))
        patches = patch.multiple(
            continuation,
            ROOT=self.root,
            INITIAL_ARCHIVE_SHA256={"pcp_comm_identity": continuation.sha256(self.archive)},
        )
        patches.start()
        self.addCleanup(patches.stop)

    def test_reused_actor_is_bound_to_the_original_archive(self):
        hashes = continuation.verify_initial_archives([self.plan], self.old_queue)
        self.assertEqual(hashes[str(self.actor)], continuation.sha256(self.actor))
        self.actor.write_bytes(b"different but internally valid actor")
        with self.assertRaisesRegex(ValueError, "changed since verified preservation"):
            continuation.verify_initial_archives([self.plan], self.old_queue)

    def test_unpreserved_run_files_cannot_be_silently_adopted(self):
        (self.run / "extra_checkpoint.pt").write_bytes(b"unpreserved checkpoint")
        with self.assertRaisesRegex(ValueError, "file inventory changed"):
            continuation.verify_initial_archives([self.plan], self.old_queue)

    def test_archive_hash_mismatch_is_rejected_before_reading_inventory(self):
        self.archive.write_bytes(b"changed archive")
        with self.assertRaisesRegex(ValueError, "archive differs from the handoff"):
            continuation.verify_initial_archives([self.plan], self.old_queue)


class EvaluationInputBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        root_patch = patch.object(continuation, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.plan = SimpleNamespace(
            output_root=self.root / "runs",
            suite_id="suite",
            run_id="identity",
            protocol={"path": "configs/frozen.yaml"},
        )
        run = self.plan.output_root / self.plan.suite_id / self.plan.run_id
        self.manifest = self.root / "results/relocation/manifest.csv"
        checkpoint = run / "benchmarl/native/checkpoints/checkpoint_600000.pt"
        diagnostic = run / "diagnostics/replay.pt"
        self.inputs = [
            self.manifest,
            self.root / self.plan.protocol["path"],
            self.root / "docs/PCP_VISIBILITY_BUDGET_PLAN.md",
            self.root / "scripts/evaluate_pcp_budget.py",
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
            checkpoint,
            diagnostic,
        ]
        for path in self.inputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
        self.evaluation = self.root / "results/preserved_evaluation"
        self.evaluation.mkdir(parents=True)
        (self.evaluation / "retained_checkpoints.json").write_text(
            json.dumps(
                {
                    "checkpoints": [{"path": str(checkpoint)}],
                    "replay_diagnostics": [{"path": str(diagnostic)}],
                }
            )
        )
        self.hashes = {str(path): continuation.sha256(path) for path in self.inputs}
        self.report = {"inputs_unchanged": True, "input_artifact_sha256": self.hashes}
        (self.evaluation / "heldout_evaluation.json").write_text(json.dumps(self.report))
        (self.evaluation / "status.json").write_text('{"status":"completed"}')

    def test_bytecopied_bank_retains_identical_frozen_input_provenance(self):
        destination = self.root / "results/coordinator/evaluations/identity"
        continuation.copy_initial_evaluation(
            self.evaluation, destination, continuation.evaluation_files(self.evaluation)
        )
        self.assertEqual(
            continuation.verify_evaluation_inputs(self.plan, destination, self.manifest),
            self.hashes,
        )

    def test_different_manifest_source_is_rejected_even_with_identical_manifest_bytes(self):
        other = self.manifest.with_name("other_manifest.csv")
        other.write_bytes(self.manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "identical manifest"):
            continuation.verify_evaluation_inputs(self.plan, self.evaluation, other)

    def test_missing_or_changed_frozen_evaluator_input_binding_is_rejected(self):
        for key in ("inputs_unchanged", "input_artifact_sha256"):
            with self.subTest(key=key):
                report = dict(self.report)
                report.pop(key)
                (self.evaluation / "heldout_evaluation.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "complete run inputs"):
                    continuation.verify_evaluation_inputs(self.plan, self.evaluation, self.manifest)
        (self.evaluation / "heldout_evaluation.json").write_text(json.dumps(self.report))
        self.inputs[-1].write_bytes(b"changed replay diagnostic")
        with self.assertRaisesRegex(ValueError, "complete run inputs"):
            continuation.verify_evaluation_inputs(self.plan, self.evaluation, self.manifest)


if __name__ == "__main__":
    unittest.main()
