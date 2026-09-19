"""Strict offline final-checkpoint MAPDN held-out intervention evaluation.

The bank is prepared prospectively by the experiment protocol/runner. This CLI
does not select a bank or overwrite an earlier attempt. Its scratch experiment
is outside the read-only source run. Successful output binds the native file,
managed final actor, full held-out bank, and exact repeat of all declared arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commstudy.analysis.diagnostics import load_frozen_experiment
from commstudy.analysis.mapdn_paired import (
    canonical_sha, evaluate_mapdn_paired, file_sha, verify_native_checkpoint,
)


def _write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--expected-bank-sha256", required=True)
    parser.add_argument("--native-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-frames", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", default=["live", "severed"])
    parser.add_argument("--require-exact-local-null", action="store_true")
    args = parser.parse_args(argv)
    run, output = args.run_dir.resolve(), args.out_dir.resolve()
    if output == run or run in output.parents:
        raise ValueError("Evaluation output must be outside the preserved training run")
    bank = json.loads(args.bank.read_text())
    if bank["sha256"] != args.expected_bank_sha256:
        raise ValueError("Bank does not match the prospectively pinned digest")
    native = args.native_checkpoint.resolve()
    if run not in native.parents:
        raise ValueError("Native checkpoint must belong to the declared run")
    source_files = {str(p): file_sha(p) for p in [
        args.bank.resolve(), native, run / "metadata.json", run / "resolved_config.yaml",
        run / "checkpoints/policy_state.pt"]}
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "start.json", {"run_dir": str(run), "source_files": source_files,
                                      "bank_sha256": bank["sha256"], "arms": args.arms})
    experiment = None
    try:
        experiment = load_frozen_experiment(run, scratch_root=output / "scratch")
        native_proof = verify_native_checkpoint(
            experiment, native, expected_frames=args.expected_frames)
        result = evaluate_mapdn_paired(experiment, bank, arms=tuple(args.arms),
                                      require_exact_local_null=args.require_exact_local_null)
        _write_new(output / "evaluation.json", result)
        replay = evaluate_mapdn_paired(experiment, bank, arms=tuple(args.arms),
                                      require_exact_local_null=args.require_exact_local_null)
        _write_new(output / "replay.json", replay)
        if result != replay:
            raise ValueError("Deterministic complete MAPDN intervention replay differs")
        if any(file_sha(path) != sha for path, sha in source_files.items()):
            raise ValueError("Read-only evaluation input changed")
        _write_new(output / "report.json", {
            "status": "passed", "native_checkpoint": native_proof,
            "strict_load_provenance": experiment.analysis_provenance,
            "evaluation_sha256": file_sha(output / "evaluation.json"),
            "replay_sha256": file_sha(output / "replay.json"),
            "canonical_result_sha256": canonical_sha(result), "exact_replay": True,
            "source_files_unchanged": True, "bank_sha256": bank["sha256"]})
    except BaseException as exc:
        _write_new(output / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                              "error": str(exc),
                                              "inputs_unchanged": all(file_sha(p) == sha
                                                  for p, sha in source_files.items())})
        raise
    finally:
        if experiment is not None:
            experiment.close()


if __name__ == "__main__":
    main()
