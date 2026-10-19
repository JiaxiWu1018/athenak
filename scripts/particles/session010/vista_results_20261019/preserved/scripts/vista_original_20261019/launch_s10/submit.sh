#!/bin/bash
# Run manually on a Vista login node. This is the only submission entry point.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/s10.py" submit "$@"
