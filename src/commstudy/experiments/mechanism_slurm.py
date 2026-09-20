"""Slurm submission and accounting. Scheduler state is never scientific success."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid

PROFILE_KEYS = {
    "SOURCE_ROOT",
    "RUN_ROOT",
    "BOOTSTRAP_PYTHON",
    "PCP_PYTHON",
    "MAPDN_PYTHON",
    "WHEELHOUSE",
    "REQUIREMENTS_LOCK",
    "ACCOUNT",
    "PARTITION",
    "CPUS",
    "MEMORY",
    "PCP_TIME",
    "MAPDN_TIME",
    "CONTROL_TIME",
    "PCP_CONCURRENCY",
    "MAPDN_CONCURRENCY",
    "MIN_FREE_GIB",
    "MODULES",
    "DEVICE",
    "GPU_COUNT",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile(path, root):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        words = shlex.split(value, comments=True)
        if not sep or key not in PROFILE_KEYS or key in values or len(words) != 1:
            raise ValueError(f"Invalid literal profile entry: {key}")
        values[key] = words[0]
    if set(values) != PROFILE_KEYS:
        raise ValueError(f"Missing profile keys: {sorted(PROFILE_KEYS - set(values))}")
    for key in (
        "SOURCE_ROOT",
        "RUN_ROOT",
        "BOOTSTRAP_PYTHON",
        "PCP_PYTHON",
        "MAPDN_PYTHON",
        "WHEELHOUSE",
        "REQUIREMENTS_LOCK",
    ):
        if not Path(values[key]).is_absolute() or "\n" in values[key]:
            raise ValueError(f"{key} requires an absolute path")
    if Path(values["RUN_ROOT"]).resolve() != Path(root).resolve():
        raise ValueError("Profile RUN_ROOT differs from requested run root")
    if values["DEVICE"] != "cpu" or values["GPU_COUNT"] != "0":
        raise ValueError(
            "Only explicit CPU/zero-GPU profile is implemented; CUDA remains unqualified"
        )
    for key in ("CPUS", "PCP_CONCURRENCY", "MAPDN_CONCURRENCY", "MIN_FREE_GIB"):
        if not values[key].isdigit() or int(values[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("ACCOUNT", "PARTITION"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", values[key]) or "REPLACE" in values[key]:
            raise ValueError(f"{key} must be the actual authorized cluster value")
    return values


def scheduler_status(job):
    if not re.fullmatch(r"\d+(?:_\d+)?", str(job)):
        raise ValueError("Invalid scheduler job identity")
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", str(job), "--format=JobIDRaw,State,ExitCode"],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        return {"available": False, "error": result.stderr.strip()}
    records = [line.split("|")[:3] for line in result.stdout.splitlines() if line.strip()]
    return {"available": True, "records": records}


def append_ledger(path, record):
    # Submission lock serializes writers; atomic replacement avoids a torn last line.
    previous = path.read_text() if path.exists() else ""
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        stream.write(previous + json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def submission_lock(root):
    lock = root / ".submission.lock"
    lock.mkdir()
    try:
        yield
    finally:
        lock.rmdir()


def submit(root, profile, *, phase, task="all", execute=False, include=None, retry=None):
    root, profile = Path(root).resolve(), Path(profile).resolve()
    p = load_profile(profile, root)
    prep = root / "preparation"
    packet = json.loads((prep / "plan.json").read_text())
    suite = json.loads((prep / "suite.json").read_text())
    if include is not None and sorted(include) != sorted(suite["include"]):
        raise ValueError(
            "Supplements must match the already generated plan; regenerate before collection"
        )
    for name, expected in packet["manifests"].items():
        if sha(prep / name) != expected:
            raise ValueError(f"Manifest changed: {name}")
    selected = ["pcp", "mapdn"] if task == "all" else [task]
    if phase == "main" and execute and not (root / "gates/main_approval.json").exists():
        raise ValueError("Main approval missing; complete qualification and freeze first")
    ledger_path = root / "submissions.jsonl"
    logdir = root / "slurm_logs"
    logdir.mkdir(exist_ok=True)
    scripts = Path(p["SOURCE_ROOT"]) / "slurm/communication_comparison_v1"
    dag, ids = [], {}
    scope = f"{phase}:{task}" + (f":retry:{sha(retry)}" if retry else "")
    retry_doc = json.loads(Path(retry).read_text()) if retry else None
    if retry_doc and (retry_doc["task"] not in selected or not retry_doc["indices"]):
        raise ValueError("Retry task/indices do not match submission")

    def add(name, script, which, stage=None, deps=(), dep_kind="afterok", extra=()):
        count = None
        indices = None
        if stage:
            rows = json.loads((prep / f"{which}_{stage}_manifest.json").read_text())
            count = len(rows)
            if not count:
                raise ValueError("Empty array")
            indices = ",".join(map(str, retry_doc["indices"])) if retry_doc else f"0-{count - 1}"
        time_key = f"{which.upper()}_TIME" if which in {"pcp", "mapdn"} else "CONTROL_TIME"
        command = [
            "sbatch",
            "--parsable",
            "--no-requeue",
            "--kill-on-invalid-dep=yes",
            "--account",
            p["ACCOUNT"],
            "--partition",
            p["PARTITION"],
            "--cpus-per-task",
            p["CPUS"],
            "--mem",
            p["MEMORY"],
            "--time",
            p[time_key],
            "--chdir",
            p["SOURCE_ROOT"],
            "--job-name",
            f"comm_v1_{name}",
            "--output",
            str(logdir / (name + ("_%A_%a.out" if stage else "_%j.out"))),
            "--error",
            str(logdir / (name + ("_%A_%a.err" if stage else "_%j.err"))),
        ]
        if stage:
            command += ["--array", f"{indices}%{p[which.upper() + '_CONCURRENCY']}"]
        if deps:
            command += ["--dependency", dep_kind + ":" + ":".join(ids[d] for d in deps)]
        command += [str(scripts / script), str(profile), str(root), *extra]
        if retry:
            command += ["--retry-plan", str(Path(retry).resolve())]
        print(shlex.join(command), flush=True)
        record = {
            "scope": scope,
            "phase": phase,
            "task": which,
            "stage": stage,
            "name": name,
            "manifest_sha256": packet["manifests"].get(f"{which}_{stage}_manifest.json"),
            "row_count": count,
            "indices": indices,
            "command": command,
            "profile_sha256": sha(profile),
            "estimated_cost": "unmeasured; profile walltime/memory are requests, not measurements",
        }
        if execute:
            response = subprocess.run(command, text=True, capture_output=True)
            if response.returncode:
                append_ledger(
                    ledger_path,
                    {
                        **record,
                        "submission_status": "failed",
                        "stderr": response.stderr,
                        "returncode": response.returncode,
                    },
                )
                raise RuntimeError(
                    "sbatch failed; earlier submitted jobs remain in submissions.jsonl"
                )
            raw = response.stdout.strip()
            if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", raw):
                append_ledger(ledger_path, {**record, "submission_status": "unparsed", "raw": raw})
                raise ValueError("Unparseable sbatch response; reconcile scheduler before retry")
            ids[name] = raw.split(";", 1)[0]
            record.update(
                job_id=ids[name],
                cluster=raw.partition(";")[2] or None,
                submission_status="submitted",
            )
            append_ledger(ledger_path, record)
        else:
            ids[name] = f"<{name}_jobid>"
        dag.append(record)

    def build():
        if retry_doc:
            which = retry_doc["task"]
            stage = retry_doc["stage"]
            if retry_doc["kind"] == "evaluation":
                script = "07_evaluate_pcp.sbatch" if which == "pcp" else "08_evaluate_mapdn.sbatch"
            elif stage == "main":
                script = "04_train_pcp.sbatch" if which == "pcp" else "05_train_mapdn.sbatch"
            else:
                script = "02_smoke.sbatch" if stage == "smoke" else "03_qualify.sbatch"
            add("retry_" + which, script, which, stage, extra=([which] if stage != "main" else []))
            return
        if phase == "prepare":
            add("probe", "00_probe.sbatch", "control")
            add("bootstrap", "01_bootstrap.sbatch", "control", deps=["probe"])
            for which in selected:
                add(
                    "smoke_" + which,
                    "02_smoke.sbatch",
                    which,
                    "smoke",
                    deps=["bootstrap"],
                    extra=[which],
                )
                add(
                    "qualify_" + which,
                    "03_qualify.sbatch",
                    which,
                    "qualification",
                    deps=["smoke_" + which],
                    extra=[which],
                )
        elif phase == "close":
            for which in selected:
                add("close_" + which, "06_close_training.sbatch", which, extra=[which])
        elif phase == "evaluate":
            for which in selected:
                add(
                    "evaluate_" + which,
                    "07_evaluate_pcp.sbatch" if which == "pcp" else "08_evaluate_mapdn.sbatch",
                    which,
                    "main",
                )
            add(
                "analyze",
                "09_analyze.sbatch",
                "control",
                deps=["evaluate_" + t for t in selected],
                dep_kind="afterany",
            )
        elif phase == "analyze":
            add("analyze", "09_analyze.sbatch", "control")
        else:
            for which in selected:
                add(
                    "train_" + which,
                    "04_train_pcp.sbatch" if which == "pcp" else "05_train_mapdn.sbatch",
                    which,
                    "main",
                )
                add(
                    "close_" + which,
                    "06_close_training.sbatch",
                    which,
                    deps=["train_" + which],
                    dep_kind="afterany",
                    extra=[which],
                )
            if task != "all":
                # Split-task main submissions never open tests before the other
                # task closes. An explicit evaluate phase follows both closures.
                add(
                    "analyze",
                    "09_analyze.sbatch",
                    "control",
                    deps=["close_" + t for t in selected],
                    dep_kind="afterany",
                )
                return
            for which in selected:
                add(
                    "evaluate_" + which,
                    "07_evaluate_pcp.sbatch" if which == "pcp" else "08_evaluate_mapdn.sbatch",
                    which,
                    "main",
                    deps=["close_" + t for t in selected],
                )
            add(
                "analyze",
                "09_analyze.sbatch",
                "control",
                deps=["evaluate_" + t for t in selected],
                dep_kind="afterany",
            )

    if execute:
        with submission_lock(root):
            prior = (
                [json.loads(line) for line in ledger_path.read_text().splitlines()]
                if ledger_path.exists()
                else []
            )
            scopes = {
                r["scope"] for r in prior if r.get("submission_status") in {"submitted", "unparsed"}
            }
            if scope in scopes or (
                not retry and any(f"{phase}:{t}" in scopes for t in [*selected, "all"])
            ):
                raise ValueError(
                    "Duplicate/overlapping submission refused; use explicit retry plans"
                )
            build()
    else:
        build()
    return {"dry_run": not execute, "jobs": dag, "scientific_completion": "artifact-gated"}


def retry_plan(root, task, reason, *, stage="main", kind="runs"):
    from commstudy.experiments.mechanism_suite import (
        manifest,
        latest_attempt,
        verify_plan,
        write_new,
    )

    root = Path(root)
    if not reason.strip():
        raise ValueError("Explicit retry reason required")
    if (
        stage == "main"
        and kind == "runs"
        and (root / f"gates/training_closure_{task}.json").exists()
    ):
        raise ValueError("Cannot retry sealed training")
    indices, prior = [], {}
    for row in manifest(root, task, stage):
        result = latest_attempt(root, row, kind)
        if result and result["status"] == "infrastructure_failure":
            indices.append(row["index"])
            path = sorted((root / kind / task / row["run_id"]).glob("attempt_*/receipt.json"))[-1]
            prior[str(row["index"])] = sha(path)
    output = root / "retry_plans" / f"{uuid.uuid4().hex}.json"
    record = {
        "task": task,
        "stage": stage,
        "kind": kind,
        "indices": indices,
        "reason": reason,
        "prior_receipts": prior,
        "plan_sha256": verify_plan(root)["sha256"],
        "restart": "from initialization; no exact checkpoint-only resume",
    }
    write_new(output, record)
    return {"path": str(output), **record}


def reconcile(root, task, stage, index, kind, reason):
    """Explicit scheduler-backed recovery of a node death/SIGKILL without receipt."""
    from commstudy.experiments.mechanism_suite import (
        select_row,
        latest_attempt,
        write_new,
        verify_plan,
    )

    root = Path(root)
    row = select_row(root, task, stage, index)
    result = latest_attempt(root, row, kind)
    if not result or result["status"] != "interrupted_or_running" or not reason.strip():
        raise ValueError("Reconciliation requires an unresolved attempt and explicit reason")
    attempt = Path(result["path"])
    start = json.loads((attempt / "start.json").read_text())
    identity = start["scheduler"]
    job = identity.get("slurm_job_id")
    array_id = identity.get("slurm_array_job_id")
    array_index = identity.get("slurm_array_task_id")
    if array_id and array_index is not None:
        job = f"{array_id}_{array_index}"
    if not job:
        raise ValueError("No recorded Slurm owner; local process recovery needs manual audit")
    state = scheduler_status(job)
    records = [r for r in state.get("records", []) if r[0] == str(job)]
    allowed = {"TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "CANCELLED", "BOOT_FAIL"}
    if not records or records[0][1].split()[0].rstrip("+") not in allowed:
        raise ValueError(
            "No terminal infrastructure state for exact scheduler owner; refusing reclaim"
        )
    receipt = {
        "status": "infrastructure_failure",
        "row": row,
        "reason": reason,
        "scheduler_evidence": state,
        "plan_sha256": verify_plan(root)["sha256"],
    }
    write_new(attempt / "receipt.json", receipt)
    lock = (
        root / "locks" / task / (f"eval_{row['run_id']}" if kind == "evaluation" else row["run_id"])
    )
    if lock.exists():
        # Preserve ownership evidence; never delete/reclaim solely on age.
        lock.rename(attempt / "interrupted_ownership")
    return receipt
