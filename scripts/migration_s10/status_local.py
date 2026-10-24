"""Small remote metadata snapshot; no login-node science or bulk scans."""
import json
import subprocess
from submit_local import LOCAL, REMOTE, atomic

def refresh():
    program = r'''
import json,datetime,subprocess
from pathlib import Path
r=Path("/work1/eliasmost/jiaxiwu/st_migration_s10_20261024")
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
rows=read(r/'evidence/job_registry.json') or []
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=[],cases={})
for row in rows:
 d=dict(row);p=r/'evidence/allocations'/row['id']
 for name in ['STARTED','WORKER_OK','FAILURE','ALLOCATION_END']:
  d[name]=read(p/(name+'.json'))
 q=subprocess.run(['scontrol','show','job','-o',row['id']],capture_output=True,text=True)
 d['scheduler']=q.stdout.strip() if q.returncode==0 else q.stderr.strip()
 out['jobs'].append(d)
for key in ['yc0500_outward','yc0500_control']:
 p=r/key;s=read(p/'runs/yc0500/state.json')
 out['cases'][key]=dict(state=None if s is None else dict(latest=s['latest'],complete=s.get('complete'),segments=len(s['segments'])),resource_stop=read(p/'evidence/RESOURCE_STOP.json'))
 for name in ['INITIAL_CONSTRAINT_ADMISSION','OUTWARD_20M_OK','INHERITED_700M']:
  q=p/'evidence'/(name+'.json');out['cases'][key][name]=read(q)
out['reference_ready']=(r/'evidence/INDEPENDENT_CONSTRAINT_REFERENCE_OK.json').exists()
print(json.dumps(out))
'''
    q = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        'hpcfund.amd.com', 'python3 -'], input=program, capture_output=True, text=True, timeout=60)
    assert q.returncode == 0, q.stderr
    value = json.loads(q.stdout)
    receipts=LOCAL/'evidence/scheduler_receipts'
    receipts.mkdir(exist_ok=True)
    for row in value['jobs']:
        path=receipts/(row['id']+'.json')
        if any('JobState='+state in row['scheduler'] for state in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY']):
            atomic(path,row)
        elif path.exists():
            row['last_terminal_scheduler_receipt']=json.loads(path.read_text())
    atomic(LOCAL/'evidence/remote_status.json', value)
    return value

if __name__ == '__main__':
    s = refresh()
    print(json.dumps(s, indent=2))
