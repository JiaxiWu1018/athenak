"""Metadata-only serial AMD submission with cumulative allocated-hour admission."""
import sys,argparse,re,shlex
sys.path.insert(0,'/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/src/athenak/scripts/reconstruction_s08')
from common import *
p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('--nodes',type=int,default=1)
p.add_argument('--hours',type=float,default=1);p.add_argument('--dependency')
p.add_argument('--case',choices=[c['id'] for c in CASES]);p.add_argument('--target',type=float,default=300)
p.add_argument('--wall-seconds',type=int,default=12600)
a=p.parse_args();assert a.nodes in [1,2] and 0<a.hours<=6
assert a.phase in ['checks','initialize','run','postprocess']
a.args=[]
if a.phase=='run':
    assert a.case and 0<a.target<=300 and 0<a.wall_seconds<=a.hours*3600-180
    a.args=[a.case,'--target',str(a.target),'--wall-seconds',str(a.wall_seconds)]
partition='mi2101x' if a.phase=='postprocess' else 'mi2104x'
ranks=1 if a.phase=='postprocess' else 4*a.nodes
def remote(args):return subprocess.check_output(['ssh','-o','BatchMode=yes','hpcfund.amd.com',shlex.join(list(map(str,args)))],text=True,stderr=subprocess.STDOUT,timeout=60)
e=LOCAL/'evidence';state=load(e/'state.json');ids=[str(j['id']) for j in state['jobs'] if j['host']=='amd']
accounting=remote(['sacct','-X','-j',','.join(ids),'-n','-P','--format=JobIDRaw,State,ElapsedRaw,AllocNodes'])
rows=[r.split('|') for r in accounting.splitlines() if r.strip()]
used=sum(int(r[2] or 0)*int(r[3] or 0) for r in rows)/3600
active=remote(['squeue','-h','-u','jiaxiwu','-o','%i|%j|%T|%D'])
st=[r.split('|') for r in active.splitlines() if '|st_s08_' in r]
assert not st or (a.dependency and all(r[0]==a.dependency for r in st)),st
assert used+a.hours*a.nodes<99,(used,a.hours,a.nodes)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
script=e/f'{a.phase}_{stamp}.sbatch';seconds=round(a.hours*3600)
command=['bash',str(AMD/'scripts/amd.sh'),a.phase]+a.args
repo=LOCAL/'src/athenak'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
readpin="import json,pathlib;r=pathlib.Path("+repr(str(AMD))+ ");p=r/'evidence/WORKFLOW_PIN.json';print(json.loads((p if p.exists() else r/'evidence/SOURCE_PIN.json').read_text())['head'])"
old=remote(['python3','-c',readpin]).strip();prelude=''
if old!=head:
    names=subprocess.check_output(['git','-C',str(repo),'diff','--name-only',old,head],text=True).splitlines()
    assert all(n.startswith('scripts/reconstruction_s08/') for n in names),names
    patch=e/f'workflow_{old[:8]}_{head[:8]}.patch'
    patch.write_bytes(subprocess.check_output(['git','-C',str(repo),'diff','--binary',old,head]))
    pin=e/f'WORKFLOW_PIN_{head[:8]}.json';atomic(pin,dict(head=head,previous=old,patch=receipt(patch)))
    cmd(['rsync','-a',patch,pin,'hpcfund.amd.com:'+str(AMD/'evidence')+'/'])
    prelude=f'''cd {AMD}/src/athenak
git apply --check {AMD}/evidence/{patch.name}
git apply {AMD}/evidence/{patch.name}
cp {AMD}/evidence/{pin.name} {AMD}/evidence/WORKFLOW_PIN.json
'''
script.write_text(f'''#!/bin/bash
#SBATCH -J st_s08_{a.phase}
#SBATCH -p {partition}
#SBATCH -A eliasmost
#SBATCH -N {a.nodes}
#SBATCH -n {ranks}
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}
#SBATCH -o {AMD}/logs/%x.%j.log
set -euo pipefail
{prelude}
{shlex.join(command)}
''')
cmd(['rsync','-a',script,'hpcfund.amd.com:'+str(AMD/'evidence')+'/'])
submit=['sbatch','--parsable']
if a.dependency:submit.append('--dependency=afterok:'+a.dependency)
reply=remote(submit+[AMD/'evidence'/script.name]);found=re.findall(r'^([0-9]+)(?:;[^\n]*)?$',reply,re.M);assert len(found)==1,reply
record=dict(id=found[0],host='amd',phase=a.phase,args=a.args,nodes=a.nodes,wall_hours=a.hours,utc=now(),used_before=used,script=str(script),reply=reply)
state['jobs'].append(record);atomic(e/'state.json',state)
atomic(e/f'SUBMITTED_{found[0]}.json',record)
atomic(e/'jobs.json',[j for j in state['jobs'] if j['host']=='amd'])
cmd(['rsync','-a',e/'jobs.json','hpcfund.amd.com:'+str(AMD/'evidence')+'/'])
print(json.dumps(record))
