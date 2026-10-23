#!/bin/bash
set -euo pipefail
S11_TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export S11_ROOT=$(cd -- "$S11_TOOLS/../../../.." && pwd)
exec python3 -B "$S11_TOOLS/early_submit.py" "$@"
