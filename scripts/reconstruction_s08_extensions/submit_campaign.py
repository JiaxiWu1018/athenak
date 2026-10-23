"""Metadata-only submission; register jobs before any further submission."""
import json,re,shlex,subprocess,sys,datetime
from pathlib import Path
r=next(p for p in Path(__file__).resolve().parents if (p/'APPROVED_PLAN.md').is_file() and (p/'evidence/state.json').is_file());a='/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023'
mode=sys.argv[1];assert mode in ['long','inward0020','inward0050','reference','review','render_check']
hours=int(sys.argv[2]) if len(sys.argv)>2 else (6 if mode=='long' else 8)
partition=sys.argv[3] if len(sys.argv)>3 else 'mi2101x';assert partition in ['mi2101x','mi2104x','mi2508x']
nodes=int(sys.argv[4]) if len(sys.argv)>4 else 1;assert nodes in [1,2]
assert partition!='mi2101x' or nodes==1, 'AMD site policy limits mi2101x to one node'
ranks=nodes*({'mi2104x':4,'mi2508x':8,'mi2101x':1}[partition]);assert nodes in [1,2]
statepath=r/'evidence/state.json';state=json.loads(statepath.read_text());records=state['jobs']
def remote(args):return subprocess.check_output(['ssh','-o','BatchMode=yes','hpcfund.amd.com',shlex.join(list(map(str,args)))],text=True,stderr=subprocess.STDOUT,timeout=90)
q=remote(['squeue','-u','jiaxiwu','-h','-o','%i|%j|%D|%T'])
active=[line.split('|') for line in q.splitlines() if '|st8x_' in line]
registered={str(v['id']):v for v in records}
assert all(v[0] in registered for v in active),active
candidate='--candidate' in sys.argv[5:]
same=[registered[v[0]] for v in active if registered[v[0]]['mode']==mode]
assert not same or (candidate and all(v['partition']!=partition for v in same)),('duplicate candidate partition',active)
assert sum(int(v[2]) for v in active)+nodes<=4,active
used=0
if records:
 out=remote(['sacct','-X','-j',','.join(registered),'-n','-P','-o','JobIDRaw,ElapsedRaw,AllocNodes'])
 used=sum(int(f[1] or 0)*int(f[2] or 0)/3600 for f in (line.split('|') for line in out.splitlines() if line.strip()))
reserved=sum(registered[v[0]]['hours']*registered[v[0]]['nodes'] for v in active)
assert used+reserved+hours*nodes<39,(used,reserved,hours,nodes)
cpus=16 if mode!='long' and partition=='mi2101x' else 2
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
f=r/'scripts'/f'{mode}_{partition}_{stamp}.sbatch'
f.write_text(f'''#!/bin/bash
#SBATCH -J st8x_{mode}
#SBATCH -A eliasmost
#SBATCH -p {partition}
#SBATCH -N {nodes}
#SBATCH -n {ranks}
#SBATCH -c {cpus}
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t {hours:02d}:00:00
#SBATCH -o {a}/logs/%x.%j.log
export ST8X_WALL_SECONDS={hours*3600}
bash {a}/scripts/launch.sh {mode}
''')
subprocess.run(['rsync','-a',str(f),'hpcfund.amd.com:'+a+'/scripts/'],check=True)
reply=remote(['sbatch','--parsable',a+'/scripts/'+f.name]);ids=re.findall(r'^([0-9]+)(?:;[^\n]*)?$',reply,re.M);assert len(ids)==1,reply
record=dict(id=ids[0],host='amd',mode=mode,hours=hours,nodes=nodes,ranks=ranks,partition=partition,script=str(f),reply=reply,used_before=used,reserved_before=reserved,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
records.append(record);state['status']='RUNNING_OR_QUEUED';statepath.write_text(json.dumps(state,indent=2)+'\n')
(r/'evidence/jobs.json').write_text(json.dumps(records,indent=2)+'\n')
(r/'evidence'/('SUBMITTED_'+ids[0]+'.json')).write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(['rsync','-a',str(r/'evidence/jobs.json'),'hpcfund.amd.com:'+a+'/evidence/'],check=True)
print(json.dumps(record))
