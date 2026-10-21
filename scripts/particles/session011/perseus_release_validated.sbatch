#!/bin/bash
#SBATCH -J jeans11-release
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021
CUDA_JOB=18291
CUDA_ROOT=$SESSION/evidence/frozen_$CUDA_JOB
AMD_ROOT=/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021
python3 - "$CUDA_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'RESULT.json').read_text())
assert r['stage']=='fresh_frozen_pass' and r['source']=='21eb90937a488723b028250837e676d1de819a3d'
assert 'FROZEN_AH_CHECKS_PASS' in (p/'fresh.log').read_text()
PY
sha256sum --check "$CUDA_ROOT/SHA256SUMS"
git -C "$SESSION/code" diff --exit-code df6ab41d5cdd95c7e33a8b70726891d52691c0cf 21eb90937a488723b028250837e676d1de819a3d -- src CMakeLists.txt cmake kokkos
ssh -o BatchMode=yes hpcfund.amd.com "mkdir -p $AMD_ROOT/evidence/perseus_cuda_$CUDA_JOB"
scp "$CUDA_ROOT/RESULT.json" "$CUDA_ROOT/SHA256SUMS" "$CUDA_ROOT/readback.json" "$CUDA_ROOT/provenance.txt" hpcfund.amd.com:"$AMD_ROOT/evidence/perseus_cuda_$CUDA_JOB/"
ssh -o BatchMode=yes hpcfund.amd.com python3 - <<'PY'
import json,subprocess
from pathlib import Path
root=Path('/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021');control=root/'control'
submission=json.loads((control/'submission.json').read_text())
assert submission['job_id']=='473069' and submission['segment']==0
assert not (root/'runs/segment_0000').exists()
state=subprocess.check_output(['scontrol','show','job','473069','-o'],text=True)
assert 'JobState=PENDING' in state and 'Reason=JobHeldUser' in state
path=control/'VALIDATION_PASS.json';gate=json.loads(path.read_text())
assert gate['source']=='df6ab41d5cdd95c7e33a8b70726891d52691c0cf'
assert gate['tooling_source']=='3272c7bc4f6bd7e7d378ab6893e11033f0229ac9'
assert gate['preflight']['job_id']=='473106'
receipt=json.loads((root/'evidence/perseus_cuda_18291/RESULT.json').read_text())
assert receipt['stage']=='fresh_frozen_pass'
gate['cuda_validation']=dict(receipt=receipt,path=str(root/'evidence/perseus_cuda_18291'),engine_source_identical=True)
tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(gate,indent=2)+'\n');tmp.replace(path)
subprocess.run(['scontrol','release','473069'],check=True)
(control/'CUDA_RELEASE.json').write_text(json.dumps(dict(production_job='473069',cuda_job='18291',preflight_job='473106',status='released'),indent=2)+'\n')
print('Released the one approved production job473069 after AMD and CUDA validation.')
PY
