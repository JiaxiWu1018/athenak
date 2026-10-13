#!/bin/bash
set -euo pipefail
CONFIG=${1:?config required}
SEGMENT=${2:?segment required}
PARENT=${3:?parent job id required}
HANDOFF=$(python3 -c 'import json, os, sys; print(os.path.expanduser(os.path.expandvars(json.load(open(sys.argv[1]))["handoff_dir"])))' "$CONFIG")
SCRIPT_DIR=$HANDOFF/scripts
test -f "$SCRIPT_DIR/workflow.py"
python3 "$SCRIPT_DIR/workflow.py" --config "$CONFIG" inspect "$SEGMENT" "$PARENT"
