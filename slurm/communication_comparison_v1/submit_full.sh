#!/usr/bin/env bash
# No submission without explicit --execute; all resource options precede scripts.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common_env.sh"
exec "$COMM_SUITE_CONTROL_PYTHON" "$COMM_SUITE_SCRIPT_ROOT/scripts/communication_suite.py" submit "$@"
