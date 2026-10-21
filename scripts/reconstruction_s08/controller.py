"""Metadata-only automatic continuation; never submits production on its own."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021')
THREAD='01a15d4b-acc1-7623-b3e7-209e4daa341c'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
    out=ROOT/'evidence/controller.json'
    state=json.loads(out.read_text()) if out.exists() else dict(events={},observed={},heartbeat=0)
    state.update(pid=os.getpid(),thread=THREAD,started=now())
    def event(key,message):
        entry=state['events'].setdefault(key,dict(last_delivery=0))
        if (ROOT/'evidence/events'/f'{key}.ack').exists() or time.time()-entry['last_delivery']<7200:return
        text=(f'ST session 8 event {key}: {message} Continue the approved implementation and five production runs. '
              f'Read {ROOT}/HANDOFF.md, APPROVED_PLAN.md, and evidence/state.json. '
              'Preserve Jeans and earlier sessions; one ST trajectory at a time, at most two nodes, '
              '60 node-hour review/100 hard cap, 350 GiB ST allowance. Complete actual scientific and '
              'visual validation. This is continuation of the existing task, not a new scope. '
              f'Acknowledge in {ROOT}/evidence/events/{key}.ack. Do not stop merely because a job is queued.')
        entry['receipt']=subprocess.check_output(['/home/jiaxiwu/.local/bin/codex','queue','--thread',THREAD,'--message',text],text=True,stderr=subprocess.STDOUT,timeout=30)
        entry['last_delivery']=time.time()
    while not (ROOT/'evidence/COMPLETE').exists() and not (ROOT/'evidence/STOP_CONTROLLER').exists():
        try:
            task=json.loads((ROOT/'evidence/state.json').read_text())
            jobs=[str(j['id']) for j in task['jobs'] if j['host']=='amd']
            if jobs:
                command='TZ=UTC sacct -X -j '+','.join(jobs)+' -n -P --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes'
                data=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hpcfund.amd.com',command],text=True,stderr=subprocess.STDOUT,timeout=60)
                rows=[l.split('|') for l in data.splitlines() if l.strip()]
                atomic(ROOT/'evidence/AMD_ACCOUNTING.json',dict(utc=now(),rows=rows,node_hours=sum(int(r[3] or 0)*int(r[4] or 0) for r in rows)/3600))
                for row in rows:
                    if any(s in row[1] for s in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL']):
                        event('job_'+row[0]+'_'+row[1].split()[0],'|'.join(row[:3]))
            if not state['events']:event('continuation_test','Verify same-thread automatic resumption and continue the active work.')
            if time.time()-state['heartbeat']>7200:
                event('heartbeat_'+str(int(time.time()/7200)),'Inspect job progress, resource forecasts and next required actions.')
                state['heartbeat']=time.time()
            state.pop('last_error',None)
        except Exception as e:state['last_error']=repr(e)
        state['utc']=now();atomic(out,state);time.sleep(120)
    state.update(ended=now());atomic(out,state)
if __name__=='__main__':main()
