"""Small metadata/patch staging only; compilation and patch application are allocated."""
import sys
sys.path.insert(0,'/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/src/athenak/scripts/reconstruction_s08')
from common import *
import shlex,re
repo=LOCAL/'src/athenak';e=LOCAL/'evidence'
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
readpin="import json,pathlib;r=pathlib.Path("+repr(str(AMD))+");p=r/'evidence/WORKFLOW_PIN.json';print(json.loads((p if p.exists() else r/'evidence/SOURCE_PIN.json').read_text())['head'])"
old=subprocess.check_output(['ssh','hpcfund.amd.com',shlex.join(['python3','-c',readpin])],text=True,timeout=60).strip()
patch=e/f'{old[:8]}_{head[:8]}.patch'
patch.write_bytes(subprocess.check_output(['git','-C',str(repo),'diff','--binary',old,head]))
pin=load(e/'SOURCE_PIN.json');pin.update(head=head,status='unvalidated corrected candidate',previous=old)
pinfile=e/f'SOURCE_PIN_{head[:8]}.json';atomic(pinfile,pin)
script=e/f'rebuild_{head[:8]}.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH -J st_s08_rebuild
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 01:00:00
#SBATCH -o {AMD}/logs/%x.%j.log
set -euo pipefail
cd {AMD}/src/athenak
git apply --check {AMD}/evidence/{patch.name}
git apply {AMD}/evidence/{patch.name}
cp {AMD}/evidence/{pinfile.name} {AMD}/evidence/SOURCE_PIN.json
cp {AMD}/evidence/{pinfile.name} {AMD}/evidence/WORKFLOW_PIN.json
bash {AMD}/scripts/amd.sh build
''')
cmd(['rsync','-a',patch,pinfile,script,'hpcfund.amd.com:'+str(AMD/'evidence')+'/'])
reply=subprocess.check_output(['ssh','hpcfund.amd.com',shlex.join(['sbatch','--parsable','--dependency=afterany:472988',str(AMD/'evidence'/script.name)])],text=True,stderr=subprocess.STDOUT)
ids=re.findall(r'^([0-9]+)(?:;[^\n]*)?$',reply,re.M);assert len(ids)==1,reply
record=dict(id=ids[0],host='amd',phase='corrected_build_and_frozen_AH',nodes=1,wall_hours=1,utc=now(),head=head,reply=reply)
atomic(e/f'REBUILD_{head[:8]}_SUBMITTED.json',record)
state=load(e/'state.json');state['jobs'].append(record);atomic(e/'state.json',state)
atomic(e/'jobs.json',[j for j in state['jobs'] if j['host']=='amd'])
cmd(['rsync','-a',e/'jobs.json','hpcfund.amd.com:'+str(AMD/'evidence')+'/'])
atomic(e/'SOURCE_PIN.json',pin)
print(json.dumps(record))
