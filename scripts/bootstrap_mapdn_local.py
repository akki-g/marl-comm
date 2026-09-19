"""Reproduce the recorded local macOS arm64 runtime without modifying research sources.

Run capture, lock, install, verify in order. Every subprocess gets a retained command
receipt, including failures. Source builds use verified copies, never the preserved tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib


THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS")
BUILT_PACKAGES = ("antlr4-python3-runtime", "benchmarl", "gym", "vmas")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def source_inventory(root):
    paths = list((root / "src" / "commstudy").rglob("*.py"))
    paths += list((root / "vendor" / "mapdn-source").rglob("*.py"))
    paths += [root / item for item in (
        "pyproject.toml", "vendor/mapdn-source/pyproject.toml",
        "vendor/mapdn-source/COMMSTUDY_SOURCE_ORIGIN.json",
        "scripts/run_mapdn_smoke.py", "scripts/run_mapdn_confirmation.py",
        "docs/MAPDN_CONFIRMATION_PLAN.md")]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Source inventory contains a symlink or non-regular file")
    return {str(path.relative_to(root)): sha(path) for path in sorted(paths)}


def check_sources(root, captured):
    if source_inventory(root) != captured["source_files"]:
        raise ValueError("Preserved source inventory changed")


def clean_environment(root):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "UV_"))
           and key not in {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"}}
    env.update(PYTHONPATH=f"{root / 'src'}:{root / 'vendor' / 'mapdn-source'}",
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    env.update({name: "1" for name in THREADS})
    return env


def command(out, root, argv, env=None):
    logs = out / "commands"
    logs.mkdir(exist_ok=True)
    index = len(list(logs.glob("*.json"))) + 1
    stem = logs / f"{index:03d}"
    selected_env = env or clean_environment(root)
    receipt = {"argv": list(map(str, argv)), "cwd": str(root),
               "started_unix": time.time(), "bootstrap_script_sha256": sha(__file__),
               "environment": {key: value for key, value in selected_env.items()
                               if key.startswith(("PYTHON", "UV_")) or key in THREADS}}
    print("RUN", index, json.dumps(receipt["argv"]), flush=True)
    with (
        stem.with_suffix(".stdout.log").open("x") as stdout,
        stem.with_suffix(".stderr.log").open("x") as stderr,
    ):
        process = subprocess.run(receipt["argv"], cwd=root, env=selected_env,
                                 stdout=stdout, stderr=stderr, check=False)
    receipt.update(returncode=process.returncode, finished_unix=time.time(),
                   stdout_sha256=sha(stem.with_suffix(".stdout.log")),
                   stderr_sha256=sha(stem.with_suffix(".stderr.log")))
    write_new(stem.with_suffix(".json"), receipt)
    print("EXIT", index, process.returncode, flush=True)
    if process.returncode:
        raise RuntimeError(f"Command {index} failed; retained logs: {stem}")
    return stem.with_suffix(".stdout.log")


def capture(root, out, env_path, evidence, uv):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This bootstrap proof is scoped to local macOS arm64")
    if sys.version_info[:3] != (3, 12, 13):
        raise ValueError("Capture requires the scientific CPython 3.12.13 interpreter")
    out.mkdir(parents=True, exist_ok=False)
    project = tomllib.loads((root / "pyproject.toml").read_text())
    vendor = tomllib.loads((root / "vendor/mapdn-source/pyproject.toml").read_text())
    roots = project["project"]["dependencies"] + vendor["project"]["dependencies"]
    roots += project["project"]["optional-dependencies"]["analysis"]
    roots += ["pytest", "ruff", "setuptools>=75", "wheel"]
    pending = [Requirement(value) for value in roots]
    selected, extras, requirements = {}, {}, {}
    while pending:
        req = pending.pop()
        name = canonicalize_name(req.name)
        dist = metadata.distribution(name)
        if dist.version not in req.specifier:
            raise ValueError(f"Installed version does not satisfy {req}")
        old = extras.get(name)
        new = set(req.extras) | (old or set())
        if old is not None and new == old:
            continue
        extras[name] = new
        selected[name] = dist
        requirements[name] = dist.requires or []
        for value in requirements[name]:
            child = Requirement(value)
            if child.marker is None or any(child.marker.evaluate({"extra": extra})
                                           for extra in {"", *new}):
                pending.append(child)
    if len(selected) != 67:
        raise ValueError(f"Expected the audited 67-distribution closure, got {len(selected)}")
    distributions = {}
    for name, dist in sorted(selected.items()):
        wheel = dist.read_text("WHEEL") or ""
        distributions[name] = {"version": dist.version,
            "wheel_tags": [line[5:] for line in wheel.splitlines() if line.startswith("Tag: ")],
            "reference_metadata_path": str(dist._path),
            "requirements": requirements[name], "selected_extras": sorted(extras[name])}
    interpreter = Path(sys.executable).resolve()
    report = read(evidence / "report.json")
    if report["verdict"] != "CONFIRMED":
        raise ValueError("Expected the closed scientific evidence for this bootstrap proof")
    captured = {"schema_version": 1, "scope": "local macOS arm64 CPython 3.12.13 bootstrap",
        "root": str(root), "environment": str(env_path), "evidence": str(evidence),
        "interpreter": str(interpreter), "interpreter_sha256": sha(interpreter),
        "python_version": sys.version, "platform": platform.platform(), "uv": str(uv),
        "uv_sha256": sha(uv), "source_files": source_inventory(root),
        "distributions": distributions, "roots": roots,
        "scientific_report_sha256": sha(evidence / "report.json"),
        "preparation_sha256": sha(evidence / "preparation.json"),
        "capture_script_sha256": sha(__file__)}
    write_new(out / "capture.json", captured)
    with (out / "exact67.in").open("x") as stream:
        for name, dist in distributions.items():
            stream.write(f"{name}=={dist['version']}\n")
    cache = Path.home() / ".cache/uv/sdists-v9/pypi"
    wheels = out / "local-built-wheels"
    wheels.mkdir()
    receipts = []
    for name in BUILT_PACKAGES:
        candidates = sorted((cache / name / distributions[name]["version"]).glob("*/*.whl"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one cached local wheel for {name}, got {candidates}")
        path = candidates[0]
        target = wheels / path.name
        shutil.copyfile(path, target)
        receipts.append({"package": name, "version": distributions[name]["version"],
                         "source": str(path), "filename": target.name, "sha256": sha(target),
                         "provenance": "existing local uv-built wheel; not an upstream wheel"})
    write_new(out / "local-built-wheel-receipts.json", receipts)
    check_sources(root, captured)


def verify_inputs(root, out):
    captured = read(out / "capture.json")
    if str(root) != captured["root"]:
        raise ValueError("Bootstrap source root differs from captured root")
    check_sources(root, captured)
    for path, digest in ((captured["interpreter"], captured["interpreter_sha256"]),
                         (captured["uv"], captured["uv_sha256"])):
        if sha(path) != digest:
            raise ValueError(f"Captured executable changed: {path}")
    for item in read(out / "local-built-wheel-receipts.json"):
        if sha(out / "local-built-wheels" / item["filename"]) != item["sha256"]:
            raise ValueError("Captured local wheel changed")
    evidence = Path(captured["evidence"])
    for filename, key in (("report.json", "scientific_report_sha256"),
                          ("preparation.json", "preparation_sha256")):
        if sha(evidence / filename) != captured[key]:
            raise ValueError(f"Bound scientific evidence changed: {filename}")
    return captured


def complete_hash_lock(raw, distributions, local_wheels):
    """uv omits hashes for some local find-links wheels; bind their actual saved bytes."""
    local = {item["package"]: item["sha256"] for item in local_wheels}
    blocks = {}
    current = None
    for line in raw.splitlines():
        match = re.match(r"^([a-zA-Z0-9_.-]+)==([^\s\\]+)", line)
        if match:
            current = match[1].lower().replace("_", "-")
            if current in blocks:
                raise ValueError(f"Duplicate resolved distribution: {current}")
            blocks[current] = {"version": match[2], "hashes": []}
        if current and "--hash=sha256:" in line:
            blocks[current]["hashes"] += re.findall(r"--hash=sha256:([a-f0-9]{64})", line)
    if set(blocks) != set(distributions):
        raise ValueError("Resolved lock does not contain the exact captured package closure")
    result = ["# Exact local CPython 3.12.13 macOS arm64 package versions.\n",
              "# Registry hashes: preserved uv-generated.lock; local wheels: SHA receipts.\n"]
    for name, expected in sorted(distributions.items()):
        block = blocks[name]
        hashes = [local[name]] if name in local else block["hashes"]
        if block["version"] != expected["version"] or not hashes:
            raise ValueError(f"Version mismatch or missing artifact hash for {name}")
        result.append(f"{name}=={expected['version']} \\\n")
        result.append(" \\\n".join(f"    --hash=sha256:{value}" for value in sorted(set(hashes))))
        result.append("\n")
    return "".join(result)


def lock(root, out):
    captured = verify_inputs(root, out)
    target = out / "requirements-macos-arm64-cp312.lock"
    if target.exists():
        raise FileExistsError(target)
    raw = out / "uv-generated.lock"
    command(out, root, [captured["uv"], "pip", "compile", out / "exact67.in", "--offline",
        "--python", captured["interpreter"], "--generate-hashes", "--no-deps",
        "--find-links", out / "local-built-wheels", "--output-file", raw])
    content = complete_hash_lock(raw.read_text(), captured["distributions"],
                                 read(out / "local-built-wheel-receipts.json"))
    with target.open("x") as stream:
        stream.write(content)
    write_new(out / "lock-receipt.json", {"requirements_sha256": sha(target),
              "uv_generated_sha256": sha(raw),
              "exact67_input_sha256": sha(out / "exact67.in")})
    check_sources(root, captured)


def verify_lock(out):
    receipt = read(out / "lock-receipt.json")
    if sha(out / "requirements-macos-arm64-cp312.lock") != receipt["requirements_sha256"]:
        raise ValueError("Requirements lock changed")
    if sha(out / "exact67.in") != receipt["exact67_input_sha256"]:
        raise ValueError("Exact version input changed")


def copy_project(root, destination, vendor=False):
    """Copy only package/build inputs; verify copied bytes before building in the copy."""
    destination.mkdir()
    source = root / "vendor/mapdn-source" if vendor else root
    directory = "mapdn" if vendor else "src/commstudy"
    target = destination / directory
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / directory, target, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md", "LICENSE", "COMMSTUDY_SOURCE_ORIGIN.json"):
        path = source / name
        if path.is_file():
            shutil.copyfile(path, destination / name)
    files = {}
    for path in destination.rglob("*"):
        if path.is_file():
            relative = path.relative_to(destination)
            if path.is_symlink() or sha(path) != sha(source / relative):
                raise ValueError(f"Build copy differs: {relative}")
            files[str(relative)] = sha(path)
    return files


def install(root, out):
    captured = verify_inputs(root, out)
    verify_lock(out)
    env_path = Path(captured["environment"])
    if env_path.exists() or env_path.is_symlink():
        raise FileExistsError(f"Refusing existing environment: {env_path}")
    uv, python = captured["uv"], env_path / "bin/python"
    command(out, root, [uv, "venv", "--no-project", "--offline", "--python",
                       captured["interpreter"], env_path])
    command(out, root, [uv, "pip", "sync", "--offline", "--python", python,
        "--require-hashes", "--no-build", "--link-mode", "copy", "--find-links",
        out / "local-built-wheels", out / "requirements-macos-arm64-cp312.lock"])
    with tempfile.TemporaryDirectory(prefix="mapdn-bootstrap-build-") as temporary:
        build_root = Path(temporary)
        copied = {name: copy_project(root, build_root / name, vendor=name == "mapdn")
                  for name in ("commstudy", "mapdn")}
        write_new(out / "build-copy-inputs.json", copied)
        command(out, root, [uv, "pip", "install", "--offline", "--python", python,
            "--no-deps", "--no-build-isolation", "--link-mode", "copy",
            build_root / "commstudy", build_root / "mapdn"])
    check_sources(root, captured)
    write_new(out / "installation.json", {"environment": str(env_path),
              "source_files_unchanged": True, "lock_sha256": sha(
                  out / "requirements-macos-arm64-cp312.lock")})


def proof(root, out):
    """Run in the new interpreter. Imports and a pure metadata contract; no environment."""
    from types import SimpleNamespace

    captured = verify_inputs(root, out)
    verify_lock(out)
    env_path = Path(captured["environment"])
    if Path(sys.prefix).resolve() != env_path.resolve():
        raise ValueError("Proof is not running in the dedicated environment")
    if sys.version != captured["python_version"]:
        raise ValueError("Python runtime version changed")
    if any(".deps" in str(path) or "/marl-comm/.venv" in str(path) for path in sys.path):
        raise ValueError("Original runtime or .deps leaked into proof sys.path")
    actual = {}
    for name, expected in captured["distributions"].items():
        dist = metadata.distribution(name)
        tags = [line[5:] for line in (dist.read_text("WHEEL") or "").splitlines()
                if line.startswith("Tag: ")]
        if dist.version != expected["version"] or tags != expected["wheel_tags"]:
            raise ValueError(f"Package version/wheel tag differs: {name}")
        if not Path(dist._path).resolve().is_relative_to(env_path):
            raise ValueError(f"Package metadata escaped the dedicated environment: {name}")
        actual[name] = {"version": dist.version, "wheel_tags": tags,
                        "metadata_path": str(dist._path)}
    import commstudy
    import mapdn
    import torch
    from mapdn.environments.pettingzoo.research_voltage_control import ResearchVoltageControl
    from commstudy.tasks.torchrl.power_grids import PowerGridTaskClass

    if Path(commstudy.__file__).resolve().parent != root / "src/commstudy":
        raise ValueError("commstudy import did not select preserved local source")
    if Path(mapdn.__file__).resolve().parent != root / "vendor/mapdn-source/mapdn":
        raise ValueError("MAPDN import did not select preserved local vendor")
    if torch.get_num_threads() != 1 or any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError("Proof requires one CPU thread")
    evidence = Path(captured["evidence"])
    report = read(evidence / "report.json")
    # Locate the retained training runtime contract without constructing any environment.
    def contracts(value):
        if isinstance(value, dict):
            if value.get("task") == "mapdn_voltage_control" and "mapdn_source_sha256" in value:
                yield value
            for item in value.values():
                yield from contracts(item)
        elif isinstance(value, list):
            for item in value:
                yield from contracts(item)
    expected_contracts = list(contracts(report))
    config_file = evidence / "spec_seed30.json"
    if not config_file.exists():
        candidates = [path for path in evidence.glob("*.json")
                      if "spec" in path.name and "30" in path.name]
        if len(candidates) != 1:
            raise ValueError("Cannot identify the frozen seed30 specification")
        config_file = candidates[0]
    config = read(config_file)["task_config"]["params"]
    contract = PowerGridTaskClass.runtime_contract(SimpleNamespace(config=config), None)
    if not expected_contracts or not all(
        all(expected.get(key) == value for key, value in contract.items())
        for expected in expected_contracts
    ):
        raise ValueError("New environment MAPDN runtime contract differs from retained evidence")
    tensor_result = (torch.tensor([1.0, 2.0]) @ torch.tensor([3.0, 4.0])).item()
    if tensor_result != 11.0 or ResearchVoltageControl.VERSION != contract["dynamics"]:
        raise ValueError("Torch arithmetic or research adapter semantics check failed")
    check_sources(root, captured)
    result = {"status": "PASS", "scope": captured["scope"],
        "environment": str(env_path), "python_executable": sys.executable,
        "python_version": sys.version, "sys_path": sys.path,
        "distributions": actual, "commstudy_origin": commstudy.__file__,
        "mapdn_origin": mapdn.__file__, "torch_origin": torch.__file__,
        "torch_threads": torch.get_num_threads(), "runtime_contract": contract,
        "source_files_unchanged": True, "environment_rollouts": 0, "training_frames": 0,
        "inputs": {name: sha(out / name) for name in (
            "capture.json", "lock-receipt.json", "requirements-macos-arm64-cp312.lock",
            "local-built-wheel-receipts.json", "build-copy-inputs.json", "installation.json")},
        "bootstrap_script_sha256": sha(__file__),
        "scientific_report_sha256": sha(evidence / "report.json"),
        "frozen_spec_sha256": sha(config_file)}
    write_new(out / "proof.json", result)


def verify(root, out):
    captured = verify_inputs(root, out)
    python = Path(captured["environment"]) / "bin/python"
    command(out, root, [captured["uv"], "pip", "check", "--python", python, "--offline"])
    command(out, root, [python, Path(__file__).resolve(), "proof", "--out-dir", out])
    check_sources(root, captured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "lock", "install", "verify", "proof"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out_dir.resolve()
    if args.stage == "capture":
        if args.env_dir is None or args.evidence_dir is None:
            parser.error("capture requires --env-dir and --evidence-dir")
        uv = shutil.which("uv")
        if uv is None:
            raise FileNotFoundError("uv")
        capture(root, out, args.env_dir.resolve(), args.evidence_dir.resolve(), Path(uv).resolve())
    else:
        {"lock": lock, "install": install, "verify": verify, "proof": proof}[args.stage](root, out)


if __name__ == "__main__":
    main()
