#!/bin/bash
set -euo pipefail
CONFIG=${1:?usage: vista_verify.sbatch SITE_CONFIG.json}
HANDOFF=$(python3 -c 'import json, os, sys; print(os.path.expanduser(os.path.expandvars(json.load(open(sys.argv[1]))["handoff_dir"])))' "$CONFIG")
SCRIPT_DIR=$HANDOFF/scripts
WF=$SCRIPT_DIR/workflow.py
test -f "$WF"
test "$(python3 "$WF" --config "$CONFIG" config-get handoff_dir)" = "$HANDOFF"
INPUT_REL=$(python3 "$WF" --config "$CONFIG" config-get input_relpath)
EXPECT_INPUT=$(python3 "$WF" --config "$CONFIG" config-get input_sha256)
CONTROL_REL=$(python3 "$WF" --config "$CONFIG" config-get control_input_relpath)
EXPECT_CONTROL=$(python3 "$WF" --config "$CONFIG" config-get control_input_sha256)
readarray -t RESTART < <(python3 - "$CONFIG" <<'PY'
import json, os, sys
cfg=json.load(open(sys.argv[1]))
for key in ("initial_checkpoint", "initial_checkpoint_sha256", "initial_checkpoint_bytes"):
    value=cfg.get(key, "")
    print(os.path.expanduser(os.path.expandvars(value)) if isinstance(value, str) else value)
PY
)
INITIAL_CHECKPOINT=${RESTART[0]}
INITIAL_CHECKPOINT_SHA256=${RESTART[1]}
INITIAL_CHECKPOINT_BYTES=${RESTART[2]}
readarray -t SOURCE_BUNDLE_CONFIG < <(python3 - "$CONFIG" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1]))
print(cfg.get("source_bundle_relpath", "source/athenak.bundle"))
print(cfg.get("source_bundle_sha256", ""))
print(cfg.get("initial_checkpoint_parameter_end_offset", ""))
PY
)
SOURCE_BUNDLE=$HANDOFF/${SOURCE_BUNDLE_CONFIG[0]}
EXPECTED_SOURCE_BUNDLE_SHA256=${SOURCE_BUNDLE_CONFIG[1]}
EXPECTED_PARAMETER_END_OFFSET=${SOURCE_BUNDLE_CONFIG[2]}
DETAILS=$(python3 "$WF" --config "$CONFIG" config-get work_root)/control/verify_details.${SLURM_JOB_ID:?}.txt
mkdir -p "$(dirname "$DETAILS")"
VERIFY_REPO=$(dirname "$DETAILS")/bundle_verify_${SLURM_JOB_ID}.git
test ! -e "$VERIFY_REPO"
git init --bare "$VERIFY_REPO" >/dev/null
{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "partition=${SLURM_JOB_PARTITION:-none}"
  echo "started=$(date -Is)"
  test -d "$HANDOFF"
  test -f "$HANDOFF/PACKAGE_SHA256SUMS.txt"
  cd "$HANDOFF"
  sha256sum -c PACKAGE_SHA256SUMS.txt
  test -z "$(find . -type l -print -quit)"
  ACTUAL_INPUT=$(sha256sum "$HANDOFF/$INPUT_REL" | awk '{print $1}')
  test "$ACTUAL_INPUT" = "$EXPECT_INPUT"
  ACTUAL_CONTROL=$(sha256sum "$HANDOFF/$CONTROL_REL" | awk '{print $1}')
  test "$ACTUAL_CONTROL" = "$EXPECT_CONTROL"
  if [ -n "$INITIAL_CHECKPOINT" ]; then
    test -f "$INITIAL_CHECKPOINT"
    test "$(stat -c %s "$INITIAL_CHECKPOINT")" = "$INITIAL_CHECKPOINT_BYTES"
    ACTUAL_CHECKPOINT=$(sha256sum "$INITIAL_CHECKPOINT" | awk '{print $1}')
    test "$ACTUAL_CHECKPOINT" = "$INITIAL_CHECKPOINT_SHA256"
    echo "initial_checkpoint=$INITIAL_CHECKPOINT"
    echo "initial_checkpoint_bytes=$INITIAL_CHECKPOINT_BYTES"
    echo "initial_checkpoint_sha256=$ACTUAL_CHECKPOINT"
    PARAMETER_END_OFFSET=$(python3 - "$INITIAL_CHECKPOINT" <<'PY'
import sys
with open(sys.argv[1], "rb") as stream:
    data=stream.read(1024*1024 + 1)
offset=data.find(b"<par_end>")
if offset < 0:
    raise SystemExit("FATAL: <par_end> is absent from the first 1 MiB")
print(offset)
PY
    )
    test -n "$EXPECTED_PARAMETER_END_OFFSET"
    test "$PARAMETER_END_OFFSET" = "$EXPECTED_PARAMETER_END_OFFSET"
    echo "initial_checkpoint_parameter_end_offset=$PARAMETER_END_OFFSET"
  fi
  test -f "$SOURCE_BUNDLE"
  if [ -n "$EXPECTED_SOURCE_BUNDLE_SHA256" ]; then
    test "$(sha256sum "$SOURCE_BUNDLE" | awk '{print $1}')" = \
      "$EXPECTED_SOURCE_BUNDLE_SHA256"
  fi
  git -C "$VERIFY_REPO" bundle verify "$SOURCE_BUNDLE"
  git -C "$VERIFY_REPO" bundle verify "$HANDOFF/source/kokkos.bundle"
  echo "source_bundle=$SOURCE_BUNDLE"
  sha256sum "$SOURCE_BUNDLE"
  echo "input_sha256=$ACTUAL_INPUT"
  echo "control_input_sha256=$ACTUAL_CONTROL"
  echo "completed=$(date -Is)"
} > "$DETAILS" 2>&1
python3 "$WF" --config "$CONFIG" mark verify pass --details "$DETAILS"
