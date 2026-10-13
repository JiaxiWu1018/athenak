#!/bin/bash
# Allocate compute before hashing/reconstructing the portable handoff. No build/submission.
set -euo pipefail
: "${SLURM_JOB_ID:?relocation audit requires allocated compute}"
HANDOFF=${1:?absolute extracted handoff directory}
HANDOFF=$(cd "$HANDOFF" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/s10_relocation_audit.XXXXXX")
export PYTHONDONTWRITEBYTECODE=1
(cd "$HANDOFF" && sha256sum -c PACKAGE_SHA256SUMS.txt)
test -z "$(find "$HANDOFF" -type l -print -quit)"
test -z "$(find "$HANDOFF" -type f -name '*.rst' -print -quit)"
readarray -t PINS < <(python3 - "$HANDOFF/PROVENANCE.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));print(p['source_commit']);print(p['kokkos_commit'])
PY
)
git init --bare "$TMP/verify.git" >/dev/null
git -C "$TMP/verify.git" bundle verify "$HANDOFF/source/athenak.bundle"
git -C "$TMP/verify.git" bundle verify "$HANDOFF/source/kokkos.bundle"
git clone "$HANDOFF/source/athenak.bundle" "$TMP/athenak"
git -C "$TMP/athenak" checkout --detach "${PINS[0]}"
git clone "$HANDOFF/source/kokkos.bundle" "$TMP/athenak/kokkos"
git -C "$TMP/athenak/kokkos" checkout --detach "${PINS[1]}"
test -z "$(git -C "$TMP/athenak" status --short)"
python3 - "$HANDOFF" <<'PY'
import ast,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);d=json.loads((p/'DECISIONS.json').read_text())
assert d['production_authorized'] is False and d['session010_review_complete'] is False
assert d['fresh_t0'] is True and d['include_session009_checkpoint'] is False
for f in p.rglob('*.py'):ast.parse(f.read_text(),filename=str(f))
PY
find "$HANDOFF/scripts" -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0 | xargs -0 -n1 bash -n
S7_SOURCE_TREE="$TMP/athenak" python3 -m unittest discover -s "$HANDOFF/tests" -p 'test_*.py'
python3 - "$HANDOFF" <<'PYSCAN'
import pathlib,re,sys
root=pathlib.Path(sys.argv[1]);pattern=re.compile(r'/Users/|/work1/|/data/jiaxiwu/|/home/jiaxiwu/')
for directory in ['scripts','inputs','analysis_support']:
 for path in (root/directory).rglob('*'):
  if path.is_file() and path.name!='audit_relocation.sh' and pattern.search(path.read_text()):
   raise SystemExit('FATAL: site-specific runtime path: '+str(path))
for path in root.rglob('*'):
 if path.is_file() and path.suffix not in ['.bundle','.png','.pdf'] and path.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1\n'):
  raise SystemExit('FATAL: unresolved LFS pointer: '+str(path))
PYSCAN
if git -C "$TMP/athenak" grep -n '^version https://git-lfs.github.com/spec/v1$' HEAD -- .; then
 echo 'FATAL: unresolved Git LFS pointer in reconstructed source' >&2;exit 3
fi
test -z "$(git -C "$TMP/athenak" ls-files '*.rst')"
echo "SESSION010 RELOCATION AUDIT OK temp=$TMP source=${PINS[0]}"
