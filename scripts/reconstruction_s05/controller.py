"""Lightweight, durable same-chat continuation; no scientific work on login nodes."""
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from common import ROOT, atomic

def main():
    thread=sys.argv[1]
    lock=open(ROOT/'evidence/controller.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'evidence/controller.json'
    state=json.loads(path.read_text()) if path.exists() else {'events':{},'observed':{},'last_heartbeat':time.time()}
    state.update(pid=os.getpid(),thread=thread)
    def event(key,message):
        state['events'].setdefault(key,{'message':message,'last_delivery':0})
    event('continuation_test','Verify that this event arrived in the same chat; continue the already-authorized implementation and acknowledge it.')
    while not (ROOT/'evidence/CAMPAIGN_COMPLETE').exists():
        try:
            work=json.loads((ROOT/'evidence/state.json').read_text())
            for job in work.get('jobs',[]):
                if job.get('host')!='amd':
                    # Perseus disables sacct; local QA is recorded by its own receipt.
                    continue
                cmd=['sacct','-j',str(job['id']),'-n','-P','--format=JobIDRaw,State,ElapsedRaw']
                if job.get('host')=='amd':cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hpcfund.amd.com',*cmd]
                reply=subprocess.check_output(cmd,text=True,timeout=45).strip()
                signature='\n'.join('|'.join(line.split('|')[:2]) for line in reply.splitlines())
                previous=state['observed'].get(str(job['id']),'')
                if signature!=previous:
                    state['observed'][str(job['id'])]=signature
                    if any(word in reply for word in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY')):
                        event('job_'+str(job['id'])+'_'+str(sum(1 for k in state['events'] if k.startswith('job_'+str(job['id'])))),
                              'Inspect '+job['role']+' job '+str(job['id'])+' status: '+reply[:1500])
            if time.time()-state['last_heartbeat']>=1800:
                event('heartbeat_'+str(int(time.time()//1800)), 'Review campaign progress, budget and storage; continue toward all 24 runs and final reports.')
                state['last_heartbeat']=time.time()
            for key,ev in state['events'].items():
                if (ROOT/'evidence/events'/(key+'.ack')).exists():continue
                if time.time()-ev['last_delivery']<7200:continue
                message=f'ST Reconstruction Session5 event {key}. {ev["message"]} Read {ROOT}/HANDOFF.md and evidence/state.json. Work remains authorized by the approved plan. Acknowledge by creating evidence/events/{key}.ack; do not treat this as a replacement task.'
                receipt=subprocess.check_output(['/home/jiaxiwu/.local/bin/codex','queue','--thread',thread,'--message',message],text=True,stderr=subprocess.STDOUT,timeout=30)
                ev.update(last_delivery=time.time(),receipt=receipt)
            state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            atomic(path,state)
        except Exception as ex:
            state['last_error']=repr(ex);atomic(path,state)
        time.sleep(60)
    state['closed']=True;atomic(path,state)

if __name__=='__main__':main()
