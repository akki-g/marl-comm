#!/usr/bin/env bash
# Supply --retry-plan from the nonexecuting retry-plan command. Dry-run default.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common_env.sh"
found=false
for argument in "$@"; do
    [[ "$argument" != "--retry-plan" ]] || found=true
done
[[ "$found" == true ]] || { echo 'Pass an explicit --retry-plan PATH; no automatic retries.' >&2; exit 2; }
exec "$COMM_SUITE_CONTROL_PYTHON" "$COMM_SUITE_SCRIPT_ROOT/scripts/communication_suite.py" submit --phase main "$@"
