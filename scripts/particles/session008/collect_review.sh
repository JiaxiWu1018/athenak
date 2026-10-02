#!/bin/bash
set -euo pipefail
ROOT=/data/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002
SRC=jiaxiwu@anta.caltech.edu:/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/
mkdir -p "$ROOT/review_from_anta"
rsync -a --exclude='*_frames/' --include='*/' --include='*.md' --include='*.png' --include='*.pdf' --include='*.mp4' --exclude='*' "$SRC" "$ROOT/review_from_anta/"
ssh hpcfund.amd.com 'cat /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/control/state.json' > "$ROOT/run_records/latest_amd_state.json"
