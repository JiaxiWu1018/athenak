"""Stage immutable source archives in a compute allocation, then submit AMD build."""
import sys
sys.path.insert(0,'/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/src/athenak/scripts/reconstruction_s08')
from common import *
import re
import shlex

def remote(args):return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hpcfund.amd.com',shlex.join(list(map(str,args)))],text=True,stderr=subprocess.STDOUT,timeout=120)
def main():
    job();repo=LOCAL/'src/athenak';pin=load(LOCAL/'evidence/SOURCE_PIN.json')
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==pin['head']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    for name,source,commit in [('source',repo,pin['head']),('kokkos',repo/'kokkos',pin['kokkos'])]:
        cmd(['git','-C',source,'archive','--format=tar','--output',LOCAL/f'{name}.tar',commit])
    # A preempted staging allocation can leave complete archives but no submitted job.
    assert not (LOCAL/'evidence/BUILD_SUBMITTED.json').exists()
    code='from pathlib import Path;r=Path('+repr(str(AMD))+');assert not (r/"bootstrap.sbatch").exists();r.mkdir(exist_ok=True);[(r/n).mkdir(parents=True,exist_ok=True) for n in ["src/athenak/kokkos","logs","evidence/jobs","inputs","samples"]]'
    remote(['python3','-c',code])
    cmd(['rsync','-a',LOCAL/'source.tar',LOCAL/'kokkos.tar','hpcfund.amd.com:'+str(AMD)+'/'])
    cmd(['rsync','-a',LOCAL/'evidence/SOURCE_PIN.json','hpcfund.amd.com:'+str(AMD/'evidence')+'/'])
    bootstrap='''#!/bin/bash
#SBATCH -J st_s08_build
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 01:30:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021/logs/%x.%j.log
set -euo pipefail
export ROOT=/work1/eliasmost/jiaxiwu/st_reconstruction_s08_production_20261021
tar -xf "$ROOT/source.tar" -C "$ROOT/src/athenak"
tar -xf "$ROOT/kokkos.tar" -C "$ROOT/src/athenak/kokkos"
ln -s "$ROOT/src/athenak/scripts/reconstruction_s08" "$ROOT/scripts"
bash "$ROOT/scripts/amd.sh" build
'''
    path=LOCAL/'evidence/bootstrap.sbatch';path.write_text(bootstrap)
    cmd(['rsync','-a',path,'hpcfund.amd.com:'+str(AMD)+'/'])
    queue=remote(['squeue','-u','jiaxiwu','-o','%.18i %.30j %.8T %.6D %R'])
    (LOCAL/'evidence/AMD_QUEUE_BEFORE_BUILD.txt').write_text(queue)
    answer=remote(['sbatch','--parsable',AMD/'bootstrap.sbatch'])
    ids=re.findall(r'^([0-9]+)(?:;[^\n]*)?$',answer,re.M);assert len(ids)==1,answer
    record=dict(id=ids[0],host='amd',phase='build_and_frozen_AH',nodes=1,wall_hours=1.5,utc=now(),reply=answer)
    atomic(LOCAL/'evidence/BUILD_SUBMITTED.json',record)
    state=load(LOCAL/'evidence/state.json');state['jobs'].append(record);atomic(LOCAL/'evidence/state.json',state)
    path=LOCAL/'evidence/jobs.json';atomic(path,[record]);cmd(['rsync','-a',path,'hpcfund.amd.com:'+str(AMD/'evidence')+'/'])
    atomic(LOCAL/'evidence/STAGE_OK.json',dict(utc=now(),job=job(),archives=[receipt(LOCAL/'source.tar'),receipt(LOCAL/'kokkos.tar')]))
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
