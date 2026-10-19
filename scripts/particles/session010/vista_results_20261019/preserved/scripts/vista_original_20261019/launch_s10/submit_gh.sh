#!/bin/bash
# User runs this on Vista login. Exactly three GH jobs; no build or gh-dev job.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/s10.py" submit-gh "$@"
