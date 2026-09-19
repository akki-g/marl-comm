"""Durable bounded coordinator for the separately frozen MAPDN diagnostic phase.

Scientific children write to exclusive files, never to the tool transport. The
coordinator stops its own children on integrity failure and never retries a row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/run_main_direction_phase.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def status(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def child_environment():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "UV_", "CONDA"))
        and key not in {"VIRTUAL_ENV", "COMMSTUDY_VENV"}
    }
    env.update(
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "vendor/mapdn-source")]),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        COMMSTUDY_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    return env


class Coordinator:
    def __init__(self, args):
        self.args = args
        self.running = {}
        self.completed = []
        self.driver_sha = sha(DRIVER)
        self.environment = child_environment()
        self.preparation_sha = None

    def launch(self, name, arguments):
        if sha(DRIVER) != self.driver_sha:
            raise ValueError("Scientific stage driver changed during coordination")
        if name in self.running or (self.args.operations / f"{name}_launch.json").exists():
            raise ValueError(
                "A declared row was already attempted; automatic retries are forbidden"
            )
        command = [
            str(self.args.python),
            str(DRIVER),
            *arguments,
            "--out-dir",
            str(self.args.output),
        ]
        if arguments[0] != "prepare":
            command += ["--preparation-sha256", self.preparation_sha]
        log_path = self.args.operations / f"{name}.log"
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.running[name] = (process, time.monotonic(), command)
        write_new(
            self.args.operations / f"{name}_launch.json",
            {
                "pid": process.pid,
                "command": command,
                "launched_unix": time.time(),
                "log_path": str(log_path),
                "driver_sha256": self.driver_sha,
                "purpose": "prospective scientific stage",
                "automatic_retry": False,
            },
        )
        self.record()

    def record(self):
        status(
            self.args.operations / "status.json",
            {
                "updated_unix": time.time(),
                "status": "running",
                "workers": self.args.workers,
                "running": {name: process.pid for name, (process, _, _) in self.running.items()},
                "completed": self.completed,
                "preparation_sha256": self.preparation_sha,
            },
        )

    def collect(self):
        for name, (process, started, command) in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            write_new(
                self.args.operations / f"{name}_exit.json",
                {
                    "pid": process.pid,
                    "exit_code": code,
                    "elapsed_seconds": time.monotonic() - started,
                    "finished_unix": time.time(),
                    "command": command,
                    "log_sha256": sha(self.args.operations / f"{name}.log"),
                },
            )
            del self.running[name]
            if code:
                raise RuntimeError(f"Scientific stage {name} failed with exit {code}; no retry")
            self.completed.append(name)
            self.record()

    def batch(self, jobs):
        pending = list(jobs)
        while pending or self.running:
            self.collect()
            while pending and len(self.running) < self.args.workers:
                name, arguments = pending.pop(0)
                self.launch(name, arguments)
            if self.running:
                time.sleep(1)

    def terminate_owned(self):
        for process, _, _ in self.running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, _, _ in self.running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)

    def run(self):
        if self.args.resume_after_calibration:
            packet = json.loads((self.args.output / "preparation.json").read_text())
            self.preparation_sha = packet["sha256"]
            self.batch([("verify_prepared_calibration", ["verify"])])
            return self.run_comparison()
        self.batch(
            [
                (
                    "prepare",
                    [
                        "prepare",
                        "--data-path",
                        str(self.args.data_path),
                        "--previous-banks",
                        str(self.args.previous_banks),
                    ],
                )
            ]
        )
        packet = json.loads((self.args.output / "preparation.json").read_text())
        self.preparation_sha = packet["sha256"]
        self.batch(
            [
                (f"calibration_identity_{seed}", ["calibrate", "--seed", str(seed)])
                for seed in (40, 41)
            ]
        )
        self.batch([("select_budget", ["select-budget"])])
        if self.args.calibration_only:
            return "calibration_complete"
        return self.run_comparison()

    def run_comparison(self):
        self.batch(
            [
                (f"comparison_{method}_{seed}", ["train", "--seed", str(seed), "--method", method])
                for seed in (50, 51, 52, 53, 54)
                for method in ("identity", "local_capacity", "broadcast")
            ]
        )
        self.batch([("close_training", ["close-training"])])
        self.batch(
            [
                (f"test_{method}_{seed}", ["evaluate", "--seed", str(seed), "--method", method])
                for seed in (50, 51, 52, 53, 54)
                for method in ("identity", "local_capacity", "broadcast")
            ]
        )
        self.batch([("final_verify", ["verify"])])
        return "all_declared_training_and_evaluation_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--previous-banks", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--resume-after-calibration", action="store_true")
    args = parser.parse_args()
    for name in ("python", "output", "operations", "data_path", "previous_banks"):
        setattr(args, name, getattr(args, name).absolute())
    if args.operations.exists():
        parser.error("Operations path must be new; previous attempts are preserved")
    if args.resume_after_calibration:
        if args.calibration_only or not (args.output / "budget_decision.json").is_file():
            parser.error("Continuation requires a completed calibration/budget decision")
        if (args.output / "evidence/comparison").exists():
            parser.error("Comparison was already attempted; this is not a retry command")
    elif args.output.exists():
        parser.error("Scientific output path must be new for preparation")
    if (
        not args.python.is_file()
        or not args.data_path.is_dir()
        or not args.previous_banks.is_file()
    ):
        parser.error("Missing reviewed runtime/data/previous bank")
    args.operations.mkdir(parents=True)
    coordinator = Coordinator(args)
    write_new(
        args.operations / "coordinator_launch.json",
        {
            "pid": os.getpid(),
            "source_root": str(ROOT),
            "supervisor_sha256": sha(__file__),
            "driver_sha256": sha(DRIVER),
            "python": str(args.python),
            "workers": args.workers,
            "started_unix": time.time(),
            "output": str(args.output),
            "calibration_only": args.calibration_only,
            "resume_after_calibration": args.resume_after_calibration,
            "transport": "Every child stdout/stderr is an exclusive durable file",
        },
    )
    code = 0
    try:
        terminal = coordinator.run()
        status(
            args.operations / "status.json",
            {
                "status": terminal,
                "completed": coordinator.completed,
                "finished_unix": time.time(),
                "running": {},
            },
        )
    except BaseException as error:
        code = 1
        coordinator.terminate_owned()
        write_new(
            args.operations / "failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "finished_unix": time.time(),
                "completed": coordinator.completed,
                "owned_children_stopped": True,
            },
        )
        status(
            args.operations / "status.json",
            {"status": "halted_after_failure", "completed": coordinator.completed, "running": {}},
        )
    write_new(
        args.operations / "coordinator_exit.json", {"exit_code": code, "finished_unix": time.time()}
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
