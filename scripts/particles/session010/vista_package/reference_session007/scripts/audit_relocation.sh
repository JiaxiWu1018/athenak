#!/bin/bash
# Reconstruct and inspect an unpacked handoff without building or submitting anything.
set -euo pipefail
HANDOFF=${1:-$(cd "$(dirname "$0")/.." && pwd)}
HANDOFF=$(cd "$HANDOFF" && pwd)
TMP=$(mktemp -d /tmp/s7_relocation_audit.XXXXXX)
export PYTHONPYCACHEPREFIX="$TMP/pycache"
test -f "$HANDOFF/PACKAGE_SHA256SUMS.txt" && \
  (cd "$HANDOFF" && sha256sum -c PACKAGE_SHA256SUMS.txt)
test -z "$(find "$HANDOFF" -type l -print -quit)"
git init --bare "$TMP/verify.git" >/dev/null
git -C "$TMP/verify.git" bundle verify "$HANDOFF/source/athenak.bundle"
git -C "$TMP/verify.git" bundle verify "$HANDOFF/source/kokkos.bundle"
git clone "$HANDOFF/source/athenak.bundle" "$TMP/athenak"
git -C "$TMP/athenak" checkout --detach 36b64e23867358e52f92ba96f9bcb4eb5e6839e6
git clone "$HANDOFF/source/kokkos.bundle" "$TMP/athenak/kokkos"
git -C "$TMP/athenak/kokkos" checkout --detach \
  6739bc623081648af9e752b616d9671527922cbf
test -z "$(git -C "$TMP/athenak" status --short)"
find "$HANDOFF/scripts" -type f \( -name '*.sh' -o -name '*.sbatch' \) \
  -print0 | xargs -0 -n1 bash -n
python3 -m py_compile "$HANDOFF"/scripts/*.py \
  "$HANDOFF"/analysis_support/inventory_session007.py
S7_SOURCE_TREE="$TMP/athenak" python3 "$HANDOFF/tests/test_workflow.py"
python3 - "$HANDOFF" "$TMP/config.json" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
cfg=json.loads((root/'workflow_config.example.json').read_text())
cfg.update(account='relocation-test',modules='gcc cuda openmpi cmake',
           handoff_dir=str(root),work_root=str(out.parent/'work'),
           scratch_root=str(out.parent/'scratch'),production_authorized=True)
out.write_text(json.dumps(cfg,indent=2)+'\n')
PY
python3 "$HANDOFF/scripts/workflow.py" --config "$TMP/config.json" dry-run --full
python3 "$HANDOFF/initial_data/calculate_orbit.py" \
  "$HANDOFF/initial_data/REFERENCE_gi_profile_M076_two_clump_s5.txt" \
  "$TMP/orbit.json"
test "$(sha256sum "$HANDOFF/inputs/gi_cluster_s7_two_clump_ccw_production.athinput" \
  | awk '{print $1}')" = 3cab60a52d9b9ac9aa08b3bc006200381f67e6b4ba1a06d5b4ebaa64e679b95c
if rg -n --glob '!audit_relocation.sh' \
    '/Users/|/work1/|/data/jiaxiwu/|/home/jiaxiwu/' \
    "$HANDOFF/scripts" "$HANDOFF/inputs" "$HANDOFF/workflow_config.example.json"; then
  echo 'FATAL: hidden site-specific runtime path' >&2
  exit 3
fi
if rg -n --glob '!audit_relocation.sh' 'version https://git-lfs.github.com/spec/v1' "$HANDOFF"; then
  echo 'FATAL: Git LFS pointer in portable tree' >&2
  exit 3
fi
echo "SESSION007 RELOCATION AUDIT OK temp=$TMP"
