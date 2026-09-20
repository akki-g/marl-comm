#!/usr/bin/env bash
# Literal profile parsing: never source/eval user configuration.
set -euo pipefail
COMM_SUITE_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PYTHONPATH="$COMM_SUITE_SCRIPT_ROOT/src:$COMM_SUITE_SCRIPT_ROOT/scripts:$COMM_SUITE_SCRIPT_ROOT/vendor/mapdn-source"
COMM_SUITE_CONTROL_PYTHON="${COMM_SUITE_CONTROL_PYTHON:-python3}"

suite_profile() {
    local profile="$1" root="$2" parsed key value
    parsed="$("$COMM_SUITE_CONTROL_PYTHON" - "$profile" "$root" "$COMM_SUITE_SCRIPT_ROOT" <<'PY'
import sys, runpy
from pathlib import Path
load_profile = runpy.run_path(str(Path(sys.argv[3]) / "src/commstudy/experiments/mechanism_slurm.py"))["load_profile"]
for key, value in load_profile(sys.argv[1], sys.argv[2]).items():
    if '\n' in value or '\t' in value:
        raise ValueError('Profile values cannot contain tabs/newlines')
    print(key + '\t' + value)
PY
)"
    while IFS=$'\t' read -r key value; do
        export "$key=$value"
    done <<< "$parsed"
    [[ "$SOURCE_ROOT" == "$COMM_SUITE_SCRIPT_ROOT" ]] || { echo 'Wrong source snapshot' >&2; return 1; }
    if [[ "$MODULES" != "-" ]]; then
        type module >/dev/null 2>&1 || { echo 'Requested module command unavailable' >&2; return 1; }
        local modules=()
        read -r -a modules <<< "$MODULES"
        module load "${modules[@]}"
    fi
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export MPLBACKEND=Agg PYTHONNOUSERSITE=1
    cd -- "$SOURCE_ROOT"
}

suite_python() {
    case "$1" in
        pcp) printf '%s\n' "$PCP_PYTHON" ;;
        mapdn) printf '%s\n' "$MAPDN_PYTHON" ;;
        *) echo 'Unknown task interpreter' >&2; return 1 ;;
    esac
}

suite_worker() {
    local profile="$1" root="$2" task="$3" command="$4"
    shift 4
    suite_profile "$profile" "$root"
    local interpreter
    interpreter="$(suite_python "$task")"
    [[ -x "$interpreter" ]] || { echo "Missing isolated interpreter: $interpreter; run bootstrap" >&2; return 1; }
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Workers require a Slurm allocation' >&2; return 1; }
    srun --kill-on-bad-exit=1 "$interpreter" "$SOURCE_ROOT/scripts/communication_suite.py" \
        "$command" --run-root "$root" --task "$task" "$@"
}
