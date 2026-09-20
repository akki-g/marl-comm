#!/usr/bin/env bash
set -euo pipefail
# sbatch --chdir sets the immutable source root, even from Slurm spool copies.
source "$PWD/slurm/communication_comparison_v1/common_env.sh"
suite_profile "$1" "$2"
profile="$1"
root="$2"
[[ -n "${SLURM_JOB_ID:-}" ]]
[[ -d "$WHEELHOUSE" && -f "$REQUIREMENTS_LOCK" ]] || {
    echo 'Stage a Linux-compatible wheelhouse and complete pip --require-hashes lock first; no network fallback.' >&2
    exit 1
}
for interpreter in "$PCP_PYTHON" "$MAPDN_PYTHON"; do
    [[ "$interpreter" == */bin/python ]] || { echo 'Expected new venv/bin/python path' >&2; exit 1; }
    suite_venv="${interpreter%/bin/python}"
    if [[ -e "$suite_venv" ]]; then
        # Never modify a preexisting environment. A prior successful suite setup
        # may be reused read-only only with identical wheel/source inputs.
        [[ -f "$suite_venv/communication_v1_bootstrap.json" ]] || {
            echo "Refusing existing unowned environment: $suite_venv" >&2; exit 1;
        }
        "$BOOTSTRAP_PYTHON" - "$suite_venv/communication_v1_bootstrap.json" "$REQUIREMENTS_LOCK" "$SOURCE_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['lock_sha256']==hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert r['source_root']==sys.argv[3]
PY
    else
        srun --kill-on-bad-exit=1 "$BOOTSTRAP_PYTHON" -m venv "$suite_venv"
        srun --kill-on-bad-exit=1 "$interpreter" -m pip install --no-index \
            --find-links "$WHEELHOUSE" --require-hashes -r "$REQUIREMENTS_LOCK"
        # Build from a private copy: pip/setuptools must not write build/egg-info
        # into the hash-bound source snapshot imported by scientific workers.
        suite_build="$suite_venv/source-build"
        "$BOOTSTRAP_PYTHON" - "$SOURCE_ROOT" "$suite_build" <<'PYBUILD'
import shutil, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
target.mkdir()
shutil.copy2(source / 'pyproject.toml', target / 'pyproject.toml')
shutil.copytree(source / 'src/commstudy', target / 'src/commstudy',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copytree(source / 'vendor/mapdn-source', target / 'vendor/mapdn-source',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info', 'build'))
PYBUILD
        srun --kill-on-bad-exit=1 "$interpreter" -m pip install --no-index --no-deps \
            --no-build-isolation "$suite_build" "$suite_build/vendor/mapdn-source"
        "$interpreter" -m pip check
        "$interpreter" -m pip freeze > "$suite_venv/communication_v1_freeze.txt"
        "$interpreter" - "$suite_venv/communication_v1_bootstrap.json" "$REQUIREMENTS_LOCK" "$SOURCE_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
with Path(sys.argv[1]).open('x') as stream:
    json.dump({'lock_sha256':hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
               'source_root':sys.argv[3]},stream)
PY
    fi
done
for task in pcp mapdn; do
    interpreter="$(suite_python "$task")"
    srun --kill-on-bad-exit=1 "$interpreter" "$SOURCE_ROOT/scripts/communication_suite.py" \
        preflight --run-root "$root" --profile "$profile" --task "$task"
done
