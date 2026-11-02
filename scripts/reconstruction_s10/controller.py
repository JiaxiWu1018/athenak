"""Metadata-only same-thread continuation; never launches or approves science jobs."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/10_rk3_predictor_20261102')
THREAD='01a19d76-f270-7bb0-ae41-b05addadc05c'
STATE=ROOT/'evidence/controller.json'

def save(obj):
    tmp=STATE.with_suffix('.writing');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(STATE)

def notify(state,key,description):
    events=ROOT/'evidence/events';events.mkdir(exist_ok=True)
    if (events/(key+'.ack')).exists():return
    old=state['events'].get(key,{})
    if time.time()-old.get('sent',0)<7200:return
    message=(f'ST Session10 continuation event {key}: {description}. '
             f'Read {ROOT}/HANDOFF.md and evidence/job_registry.json. '
             'Continue the approved RK3 predictor implementation, native validation, '
             'short timestep controls, four parallel-lane500M runs, constraint/order '
             'analysis and reviewed plots/report. Max2 simulation nodes+1support; '
             '40nodeh/300GiB review thresholds. No four legacy500M reruns. '
             'Preserve earlier campaigns. This continues the existing user task; '
             f'acknowledge in evidence/events/{key}.ack.')
    result=subprocess.run(['/home/jiaxiwu/.local/bin/codex','queue','--resume',
        '--thread',THREAD,'--message',message],text=True,capture_output=True,timeout=30)
    state['events'][key]=dict(sent=time.time(),returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr)

lock=(ROOT/'evidence/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads(STATE.read_text()) if STATE.exists() else dict(events={},observed={})
state.update(pid=os.getpid(),thread=THREAD,poll_seconds=120)
# Historical attempts are already reviewed; only newly observed terminal events notify.
for job in json.loads((ROOT/'evidence/job_registry.json').read_text())['jobs']:
    if str(job.get('status','')).startswith(('FAILED','COMPLETED')):
        state['observed'][job['job_id']]='historical'
last_heartbeat=time.time()
notify(state,'continuation_test','Verify delivery to this same thread; continue current gates')
save(state)
while not (ROOT/'evidence/COMPLETE').exists() and not (ROOT/'evidence/CONTROLLER_STOP').exists():
    try:
        registry=json.loads((ROOT/'evidence/job_registry.json').read_text())
        ids=[j['job_id'] for j in registry['jobs'] if j['host']=='AMD']
        if ids:
            cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hpcfund.amd.com',
                 'sacct -X -n -P -j '+','.join(ids)+' --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes']
            reply=subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT,timeout=45)
            rows=[line.split('|') for line in reply.splitlines() if line.strip()]
            state['accounting']=dict(rows=rows,node_hours=sum(int(r[3])*int(r[4]) for r in rows)/3600)
            for row in rows:
                key=row[0]
                terminal=any(word in row[1] for word in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL'))
                signature='|'.join(row[:3])
                if terminal and state['observed'].get(key) not in ('historical',signature):
                    notify(state,'job_'+key,signature)
                state['observed'][key]=signature if state['observed'].get(key)!='historical' else 'historical'
        if time.time()-last_heartbeat>=7200:
            notify(state,'heartbeat_'+str(int(time.time()/7200)),'Review jobs, queue, budget/storage and next actions')
            last_heartbeat=time.time()
        state.pop('error',None)
    except Exception as error:
        state['error']=repr(error)
    state['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(state)
    time.sleep(120)
state['closed']=True;save(state)
