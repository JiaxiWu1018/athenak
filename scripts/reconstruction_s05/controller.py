"""Lightweight, durable same-chat continuation; no scientific work on login nodes."""
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from common import ROOT, atomic

HEARTBEAT_INTERVAL_SECONDS = 2 * 60 * 60

def main():
    thread=sys.argv[1]
    lock=open(ROOT/'evidence/controller.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'evidence/controller.json'
    state=json.loads(path.read_text()) if path.exists() else {'events':{},'observed':{},'last_heartbeat':time.time()}
    state.update(pid=os.getpid(),thread=thread,heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS)
    def event(key,message):
        state['events'].setdefault(key,{'message':message,'last_delivery':0})
    event('continuation_test','Verify that this event arrived in the same chat; continue the already-authorized implementation and acknowledge it.')
    while not (ROOT/'evidence/CAMPAIGN_COMPLETE').exists():
        try:
            work=json.loads((ROOT/'evidence/state.json').read_text())
            jobs=[j for j in work.get('jobs',[]) if j.get('host')=='amd']
            # Query allocations together, including all array tasks but excluding steps.
            # Perseus disables sacct; its QA has separate completion receipts.
            cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hpcfund.amd.com',
                 'sacct','--allocations','-j',','.join(str(j['id']) for j in jobs),
                 '-n','-P','--format=JobID,State,ElapsedRaw,AllocNodes']
            reply=subprocess.check_output(cmd,text=True,timeout=45).strip() if jobs else ''
            rows=[line.split('|') for line in reply.splitlines() if line]
            used=sum(int(r[2])*int(r[3]) for r in rows)/3600
            atomic(ROOT/'evidence/AMD_ACCOUNTING.json',{'allocated_node_hours':used,'rows':rows,
                   'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'definition':'allocation elapsed seconds times allocated nodes; includes failed attempts; excludes steps'})
            state.setdefault('terminal',{})
            for job in jobs:
                own=[r for r in rows if r[0].split('_')[0]==str(job['id'])]
                signature='\n'.join('|'.join(r[:2]) for r in own)
                previous=state['observed'].get(str(job['id']),'')
                if signature!=previous:
                    state['observed'][str(job['id'])]=signature
                    terminal='\n'.join('|'.join(r[:2]) for r in own if any(word in r[1] for word in
                                      ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL')))
                    if terminal and terminal!=state['terminal'].get(str(job['id']),''):
                        state['terminal'][str(job['id'])]=terminal
                        event('job_'+str(job['id'])+'_'+str(sum(1 for k in state['events'] if k.startswith('job_'+str(job['id'])))),
                              'Inspect '+job['role']+' job '+str(job['id'])+' status: '+terminal[:1500])
            # Enforce the authorized raw allocation cap; leave 2 hours for closure.
            if used>=98:
                atomic(ROOT/'evidence/BUDGET_STOP',{'allocated_node_hours':used,'ceiling':100})
                active={r[0].split('_')[0] for r in rows if r[1] in ('RUNNING','PENDING','CONFIGURING','COMPLETING')}
                cancel=[str(j['id']) for j in jobs if j.get('role')=='production' and str(j['id']) in active]
                if used>=100:cancel=[str(j['id']) for j in jobs if str(j['id']) in active]
                if cancel:
                    subprocess.run(['ssh','hpcfund.amd.com','scancel',*cancel],check=True,timeout=30)
                event('budget_stop','Allocated AMD cost reached '+str(used)+' node-hours. Production was stopped to preserve the 100-node-hour ceiling; inspect the retained evidence.')
            if time.time()-state['last_heartbeat']>=HEARTBEAT_INTERVAL_SECONDS:
                event('heartbeat_'+str(int(time.time()//HEARTBEAT_INTERVAL_SECONDS)), 'Review campaign progress, budget and storage; continue toward all 24 runs and final reports.')
                state['last_heartbeat']=time.time()
            for key,ev in state['events'].items():
                if (ROOT/'evidence/events'/(key+'.ack')).exists():continue
                if time.time()-ev['last_delivery']<7200:continue
                message=f'ST Reconstruction Session5 event {key}. {ev["message"]} Read {ROOT}/HANDOFF.md and evidence/state.json. Work remains authorized by the approved plan. Acknowledge by creating evidence/events/{key}.ack; do not treat this as a replacement task.'
                receipt=subprocess.check_output(['/home/jiaxiwu/.local/bin/codex','queue','--thread',thread,'--message',message],text=True,stderr=subprocess.STDOUT,timeout=30)
                ev.update(last_delivery=time.time(),receipt=receipt)
            state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            state.pop('last_error',None)
            atomic(path,state)
        except Exception as ex:
            state['last_error']=repr(ex);atomic(path,state)
        time.sleep(60)
    state['closed']=True;atomic(path,state)

if __name__=='__main__':main()
