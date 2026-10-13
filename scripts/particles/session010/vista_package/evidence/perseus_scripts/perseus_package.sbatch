#!/bin/bash
#SBATCH -J jeans10-package
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
REPO=$SESSION/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
HEAD_PIN=$(git -C "$REPO" rev-parse HEAD)
KOKKOS_PIN=$(git -C "$REPO/kokkos" rev-parse HEAD)
test "$(git -C "$REPO/kokkos" rev-parse --is-shallow-repository)" = false
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/kokkos" status --short)"
test -z "$(git -C "$REPO" diff 80ef7425 HEAD -- src CMakeLists.txt kokkos)"
test "$(cat "$SESSION/evidence/PUSHED_HEAD.txt")" = "$HEAD_PIN"
PACKAGE=$SESSION/handoff/session010_vista_handoff
ARCHIVE=$SESSION/handoff/session010_vista_handoff.tar.gz
test ! -e "$PACKAGE" && test ! -e "$ARCHIVE"
cp -a "$REPO/scripts/particles/session010/vista_package" "$PACKAGE"
mkdir "$PACKAGE/source"
git -C "$REPO" bundle create "$PACKAGE/source/athenak.bundle" refs/heads/project/GI-in-cluster
git -C "$REPO/kokkos" bundle create "$PACKAGE/source/kokkos.bundle" HEAD
python3 - "$SESSION" "$PACKAGE" "$HEAD_PIN" "$KOKKOS_PIN" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
s,p=map(Path,sys.argv[1:3]);head,kokkos=sys.argv[3:]
compiled=json.loads((p/'evidence/COMPILED_PROVENANCE.json').read_text())
prov=dict(source_commit=head,kokkos_commit=kokkos,source_branch='project/GI-in-cluster',origin='git@github.com:JiaxiWu1018/athenak.git',source_cpp_equivalent_to_compiled=compiled['patched_cpp_commit'],compiled=compiled,input_sha256=hashlib.sha256((p/'inputs/gi_cluster_s10.athinput').read_bytes()).hexdigest(),fresh_t0=True,checkpoints_in_package=0,production_authorized=False,email='jiaxiwu@caltech.edu',preparation_host=os.uname().nodename,packaging_job=os.environ['SLURM_JOB_ID'])
(p/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
(p/'evidence/PUSH_STATUS.txt').write_text((s/'evidence/PUSH_STATUS.txt').read_text())
for f in p.rglob('*'):
 if f.is_symlink():raise RuntimeError('Symlink in package')
 if f.is_file() and f.suffix in ['.rst','.vtk','.bin','.h5','.pyc','.o','.a']:raise RuntimeError(('Raw/built payload in handoff',str(f)))
manifest=[]
for f in sorted(p.rglob('*')):
 if not f.is_file():continue
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 manifest.append(h.hexdigest()+'  '+str(f.relative_to(p)))
(p/'PACKAGE_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')
PY
AUDIT=$SESSION/evidence/packaging_${SLURM_JOB_ID}
mkdir "$AUDIT"
scontrol show job "$SLURM_JOB_ID" > "$AUDIT/slurm.txt"
bash "$PACKAGE/scripts/audit_relocation.sh" "$PACKAGE" > "$AUDIT/source_relocation.log" 2>&1
tar -czf "$ARCHIVE" -C "$SESSION/handoff" session010_vista_handoff
(cd "$SESSION/handoff" && sha256sum session010_vista_handoff.tar.gz > session010_vista_handoff.tar.gz.sha256)
mkdir "$AUDIT/unpack"
tar -xzf "$ARCHIVE" -C "$AUDIT/unpack"
(cd "$AUDIT/unpack/session010_vista_handoff" && sha256sum -c PACKAGE_SHA256SUMS.txt) > "$AUDIT/unpack_verify.log"
python3 - "$SESSION" "$PACKAGE" "$ARCHIVE" "$AUDIT" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
s,p,a,audit=map(Path,sys.argv[1:]);provenance=json.loads((p/'PROVENANCE.json').read_text())
receipt=dict(passed=True,job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,source_commit=provenance['source_commit'],kokkos_commit=provenance['kokkos_commit'],archive=str(a),bytes=a.stat().st_size,sha256=hashlib.sha256(a.read_bytes()).hexdigest(),internal_manifest_files=len((p/'PACKAGE_SHA256SUMS.txt').read_text().splitlines()),checkpoint_files=0,production_launched=False,Vista_transfer_performed=False,relocation_log=str(audit/'source_relocation.log'),unpack_log=str(audit/'unpack_verify.log'),finished_epoch=time.time())
(s/'evidence/PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY
