"""Append observed D4 queue milestones to an existing log; never judge scientific results.

The coordinator publishes atomic snapshots, so polling can miss intermediate states.
A pending append receipt permits restart without duplicating a fully written observation.
Partial log writes fail closed and require inspection; existing log bytes are never repaired.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time


TERMINAL = {"completed", "stopped_after_failure"}
ROW_FIELDS = (
    "status", "stage", "training_reused", "training_completed", "evaluation_reused",
    "evaluation_checks_passed", "archive", "started_utc", "finished_utc", "error",
)
QUEUE_FIELDS = ("status", "phase", "started_utc", "finished_utc", "stop_reason")


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(value):
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Parent traversal is not allowed in observer paths.")
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        if part.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {part}")
    if not path.parent.is_dir():
        raise ValueError(f"An existing ordinary parent directory is required: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Expected an ordinary file: {path}")
    return path


def safe_read(path):
    checked_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Expected an ordinary file: {path}")
        return stream.read()


def atomic_state(path, value):
    checked_path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write((encode(value) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        checked_path(path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def project(queue):
    if (queue.get("schema_version") != 1 or queue.get("status") not in TERMINAL | {"running"}
            or not isinstance(queue.get("phase"), str)
            or not isinstance(queue.get("rows"), dict) or not queue["rows"]):
        raise ValueError("Unsupported or malformed continuation queue snapshot.")
    for run_id, row in queue["rows"].items():
        if (not isinstance(run_id, str) or not run_id or not isinstance(row, dict)
                or row.get("status") not in {"pending", "running", "completed", "failed"}):
            raise ValueError("Malformed queue row identity/status.")
    return {"queue": {key: queue.get(key) for key in QUEUE_FIELDS},
            "rows": {run_id: {key: row.get(key) for key in ROW_FIELDS}
                     for run_id, row in sorted(queue["rows"].items())}}


class Observer:
    def __init__(self, queue, log, state):
        self.queue, self.log, self.state_path = map(checked_path, (queue, log, state))
        self.lock_path = checked_path(str(self.state_path) + ".lock")
        self.paths = {"queue": str(self.queue), "log": str(self.log),
                      "state": str(self.state_path), "lock": str(self.lock_path)}
        self.lock_fd = None
        self.validate_paths()
        if not self.queue.is_file() or not self.log.is_file():
            raise ValueError("Start only after the queue snapshot and existing log are present.")

    def validate_paths(self):
        paths = [checked_path(path) for path in self.paths.values()]
        for index, first in enumerate(paths):
            for second in paths[index + 1:]:
                if (first == second or first in second.parents or second in first.parents
                        or (first.exists() and second.exists() and first.samefile(second))):
                    raise ValueError("Observer input, state, lock and log paths must not overlap.")

    def __enter__(self):
        self.validate_paths()
        self.lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.validate_paths()
            self.state = (json.loads(safe_read(self.state_path)) if self.state_path.exists()
                          else {"schema_version": 1, "paths": self.paths, "sequence": 0,
                                "observer_script_sha256": digest(safe_read(Path(__file__))),
                                "queue_identity": None, "projection": None, "pending": None})
            if self.state.get("schema_version") != 1 or self.state.get("paths") != self.paths:
                raise ValueError("Existing observer receipt belongs to different paths/schema.")
            if self.state.get("pending") is not None:
                self.finish_pending()
            return self
        except BaseException:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise

    def __exit__(self, *_):
        os.close(self.lock_fd)
        self.lock_fd = None

    def finish_pending(self):
        self.validate_paths()
        pending = self.state["pending"]
        block = pending["block"].encode()
        fd = os.open(self.log, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(fd, "r+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            existing = stream.read()
            if "append_offset" not in pending:
                pending.update(append_offset=len(existing), prior_log_sha256=digest(existing))
                atomic_state(self.state_path, self.state)
            offset = pending["append_offset"]
            if (len(existing) < offset
                    or digest(existing[:offset]) != pending["prior_log_sha256"]):
                raise ValueError("Log prefix changed during pending append; inspect the receipt.")
            tail = existing[offset:]
            if tail and block not in tail:
                raise ValueError("Partial or intervening append; inspect without rewriting.")
            if block not in tail:
                # O_APPEND and one write preserve concurrent ordinary log appenders.
                if os.write(stream.fileno(), block) != len(block):
                    raise OSError("Partial append; pending receipt preserved for review.")
            os.fsync(stream.fileno())
        self.state.update(pending["committed"])
        self.state["pending"] = None
        atomic_state(self.state_path, self.state)

    def observe_once(self):
        self.validate_paths()
        raw = safe_read(self.queue)
        queue = json.loads(raw)
        current = project(queue)
        identity = {key: queue.get(key) for key in ("started_utc", "manifest", "manifest_sha256")}
        identity["run_ids"] = sorted(queue["rows"])
        if self.state["queue_identity"] not in (None, identity):
            raise ValueError("Queue identity or allocated run IDs changed; preserve this receipt.")
        previous = self.state["projection"]
        observed = datetime.now(UTC).isoformat()
        committed = {"queue_identity": identity, "projection": current,
                     "last_observed_utc": observed, "queue_sha256": digest(raw),
                     "terminal_observed": queue["status"] in TERMINAL}
        if current != previous:
            sequence = self.state["sequence"] + 1
            changed = {run_id: {"previous_status": (
                previous["rows"][run_id]["status"] if previous else None), "observed": row}
                for run_id, row in current["rows"].items()
                if previous is None or row != previous["rows"][run_id]}
            event = {"sequence": sequence, "observed_utc": observed,
                     "queue_path": str(self.queue), "queue_sha256": digest(raw),
                     "previous_queue": previous["queue"] if previous else None,
                     "observed_queue": current["queue"], "row_observations": changed}
            event_id = digest(encode(event).encode())
            marker = f"<!-- pcp-queue-observation:{event_id} -->"
            block = (f"\n\n### D4 queue observation — {observed}\n\n{marker}\n\n"
                     "Observed coordinator snapshot only; intermediate states may occur between "
                     "polls. Null fields mean unreported. Review/archive fields are coordinator "
                     "reports, not independent verification or scientific conclusions.\n\n"
                     f"```json\n{encode(event)}\n```\n")
            committed.update(sequence=sequence, last_event_id=event_id)
            self.state["pending"] = {"marker": marker, "block": block, "committed": committed}
            atomic_state(self.state_path, self.state)
            self.finish_pending()
            print(encode({"observation": event_id, "sequence": sequence,
                          "queue_status": queue["status"], "changed_rows": sorted(changed)}),
                  flush=True)
        else:
            self.state.update(committed)
            atomic_state(self.state_path, self.state)
        return queue["status"] in TERMINAL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-status", required=True, type=Path)
    parser.add_argument("--agents-log", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 30:
        parser.error("--poll-seconds must be greater than zero and at most 30.")
    try:
        with Observer(args.queue_status, args.agents_log, args.state) as observer:
            while not observer.observe_once():
                time.sleep(args.poll_seconds)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Queue observer stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
