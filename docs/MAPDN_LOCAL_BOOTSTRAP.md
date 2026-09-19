# Verified local MAPDN bootstrap

The isolated environment passed on 2026-09-09 on this Mac: macOS 26.5 arm64,
CPython 3.12.13, and uv 0.12.5. All 67 third-party versions and their installed
wheel tags match the runtime used for the closed case33 confirmation. This is
a local bootstrap proof; it does not establish portability to another machine,
operating system, Python version, or an empty package cache.

The successful environment is
`.bootstrap-envs/mapdn-cp312-20260909-attempt02`. Its evidence is in
`results/mapdn_bootstrap_macos_arm64_cp312_20260909_attempt02/`:

- `capture.json` records the exact interpreter, dependency closure, reference
  wheel tags, source inventory, tool hash, and scientific evidence hashes.
- `exact67.in`, `uv-generated.lock`, and
  `requirements-macos-arm64-cp312.lock` preserve the version input, raw offline
  resolver output, and complete hash requirements used for installation.
- `local-built-wheel-receipts.json` and `local-built-wheels/` retain the cached
  ANTLR, BenchMARL, gym, and VMAS wheels with their actual SHA-256 values. These
  are locally built artifacts, not claimed upstream wheel hashes.
- `build-copy-inputs.json` identifies the verified temporary package copies.
  The temporary copies were deleted after packaging.
- `commands/` contains exact subprocess arguments, selected environment
  variables, return codes, timings, and hashed stdout/stderr logs.
- `proof.json` records PASS, all package versions/tags/import paths, the MAPDN
  runtime contract, and unchanged source inventory. The contract matches every
  retained contract in the closed scientific report. It records zero environment
  rollouts and zero training frames.

The offline install, local package builds, dependency check, and import proof
took approximately 6.6, 0.4, 0.04, and 30.2 seconds respectively. The import proof
used one Torch thread and performed a small tensor arithmetic check. It neither
constructed a power-grid environment nor reconstructed a policy checkpoint.

The first attempt remains at
`results/mapdn_bootstrap_macos_arm64_cp312_20260909/` with its separate empty
environment `.bootstrap-envs/mapdn-cp312-20260909`. uv's offline compiler omitted
hashes for local `--find-links` wheels. The first hash-enforced sync rejected the
ANTLR requirement before installing dependencies; `commands/003.*` preserves
the failure. The script now supplements those entries with the actual wheel
hashes from the retained receipts and rejects any unresolved hash/version gap.
The successful attempt used a fresh directory and environment.

## Source and runtime isolation

The existing project `uv.lock` describes a different MAPDN remote revision and
Farama Notifications 0.0.6; the scientific runtime used the preserved local
MAPDN source and Farama Notifications 0.0.4. Therefore this bootstrap uses its
own exact requirements artifact and does not synchronize the project lock or
request the remote `powergrid` extra.

Third-party packages are copied into the dedicated environment. Local
commstudy and MAPDN package metadata/wheels are built only from temporary,
byte-verified copies. No build runs in the preserved project or vendor tree.
Research imports explicitly select the original isolated `src` and
`vendor/mapdn-source` through `PYTHONPATH`. The proof rejects `.deps` or the
main environment's site-packages in `sys.path`, checks package metadata belongs
to the new environment, and checks the local commstudy/MAPDN import origins.

Source hashes are checked before and after every installation/verification
stage. The original main environment, vendor source, task source, scientific
scripts, protocol plan, preparation, data, and scientific evidence were not
modified. The new interpreter path is deliberately different from the runtime
recorded in the frozen preparation; this proof does not authorize resuming or
rewriting that completed scientific attempt.

## Commands

Run from the isolated project root. `capture` needs the reference runtime's
metadata, including its recorded `.deps` overlay; subsequent commands use the
new environment without that overlay. Use new output and environment names
when reproducing the bootstrap. Existing environments are rejected.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:vendor/mapdn-source:.deps \
  /Users/akki/Desktop/Thesis/marl-comm/.venv/bin/python \
  scripts/bootstrap_mapdn_local.py capture \
  --out-dir results/mapdn_bootstrap_macos_arm64_cp312_20260909_attempt02 \
  --env-dir .bootstrap-envs/mapdn-cp312-20260909-attempt02 \
  --evidence-dir results/mapdn_case33_bounded_confirmation_v1_20260909

PYTHONDONTWRITEBYTECODE=1 /Users/akki/Desktop/Thesis/marl-comm/.venv/bin/python \
  scripts/bootstrap_mapdn_local.py lock \
  --out-dir results/mapdn_bootstrap_macos_arm64_cp312_20260909_attempt02

PYTHONDONTWRITEBYTECODE=1 /Users/akki/Desktop/Thesis/marl-comm/.venv/bin/python \
  scripts/bootstrap_mapdn_local.py install \
  --out-dir results/mapdn_bootstrap_macos_arm64_cp312_20260909_attempt02

PYTHONDONTWRITEBYTECODE=1 /Users/akki/Desktop/Thesis/marl-comm/.venv/bin/python \
  scripts/bootstrap_mapdn_local.py verify \
  --out-dir results/mapdn_bootstrap_macos_arm64_cp312_20260909_attempt02
```

The script resolves CPython to the recorded absolute 3.12.13 interpreter and
checks its SHA-256 before each stage. A generic uv `3.12` selector points to a
different managed Python version on this machine. Every installation command
uses `--offline`; dependency synchronization uses `--require-hashes`,
`--no-build`, and `--link-mode copy`. Local-copy packaging uses `--no-deps` and
`--no-build-isolation`, with setuptools and wheel already in the exact lock.

For subsequent explicitly authorized work in the new environment, use:

```sh
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:vendor/mapdn-source \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .bootstrap-envs/mapdn-cp312-20260909-attempt02/bin/python YOUR_SCRIPT.py
```

Focused bootstrap tests cover missing/version-mismatched dependency hashes,
locally built wheel hashes, source-copy isolation, and retained failure logs.
They import no scientific environment or training modules.
