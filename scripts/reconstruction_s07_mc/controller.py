"""Metadata-only same-chat continuation; no raw transfer and no scientific auto-approval."""
import fcntl
import hashlib
import shlex
import subprocess
import time
from cfg import *

def remote(args):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hpcfund.amd.com',shlex.join(args)],text=True,stderr=subprocess.STDOUT,timeout=90)
def main():
    lock=(ROOT/'evidence/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=ROOT/'evidence/controller.json';state=load(p)if p.exists()else dict(events={},observed={},heartbeat_at=time.time())
    state.update(pid=os.getpid(),thread=THREAD,started_utc=now(),poll_seconds=120)
    def event(key,description):
        item=state['events'].setdefault(key,dict(last_delivery=0,description=description))
        if (ROOT/'evidence/events'/(key+'.ack')).exists()or time.time()-item['last_delivery']<7200:return
        message=(f'ST session7 event {key}: {description} Read {ROOT}/HANDOFF.md and evidence/state.json. '
            'Continue the approved four fresh nested MC seed1 runs (1/2/4/8 million) to1000M, then reductions, '
            'stability/convergence assessment, figures, five movies and final report. All raw stays AMD; never copy '
            'runs/, evolved particles/fields/checkpoints or movie caches/frames to Perseus. Acknowledge in '
            f'evidence/events/{key}.ack. This event continues the existing task; do not restart closed session6 jobs.')
        item['receipt']=subprocess.check_output(['/home/jiaxiwu/.local/bin/codex','queue','--thread',THREAD,'--message',message],text=True,stderr=subprocess.STDOUT,timeout=30)
        item['last_delivery']=time.time()
    while not (ROOT/'evidence/COMPLETE').exists():
        try:
            current=load(ROOT/'evidence/state.json')
            submitted=ROOT/'evidence/PREPARE_SUBMITTED.json'
            if submitted.exists():
                job=load(submitted)
                if not any(j['host']=='amd'and j['id']==job['id']for j in current['jobs']):
                    current['jobs'].append(job);atomic(ROOT/'evidence/state.json',current)
            jobs=[str(j['id'])for j in current['jobs']if j['host']=='amd']
            if jobs:
                reply=remote(['sacct','-X','-j',','.join(jobs),'-n','-P','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes'])
                rows=[line.split('|')for line in reply.splitlines()if line.strip()]
                atomic(ROOT/'evidence/AMD_ACCOUNTING.json',dict(utc=now(),rows=rows,node_hours=sum(int(r[3])*int(r[4])for r in rows)/3600))
                for row in rows:
                    signature='|'.join(row[:3])
                    if state['observed'].get(row[0])!=signature:
                        state['observed'][row[0]]=signature
                        if any(x in row[1]for x in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL']):event('job_'+row[0]+'_'+row[1].split()[0],signature)
            if (ROOT/'evidence/STAGE_OK.json').exists():
                mirror=ROOT/'evidence/amd';mirror.mkdir(exist_ok=True)
                subprocess.run(['rsync','-a','--include=/*.json','--include=/preflight/','--include=/preflight/*.json','--include=/samples/','--include=/samples/*.json','--exclude=*',f'hpcfund.amd.com:{AMD}/evidence/',str(mirror)+'/'],check=True,timeout=90)
                if (mirror/'PREFLIGHT_HEALTH_QA.json').exists()and not(ROOT/'evidence/PRODUCTION_REVIEW.json').exists():event('preflight_review','All native preflights/health ready: inspect actual results and timing/memory before approving production.')
                if (mirror/'ANALYSIS_RENDER_COMPLETE.json').exists():event('final_review','Actual result tables, PNG/PDF previews and five encoded-movie reviews are ready on AMD. Pull thin review products and complete scientific/visual/integrity closeout.')
            if not state['events']:event('continuation_test','Verify same-chat automatic resumption and continue preparation; inspect stage status.')
            state.pop('last_error',None)
        except Exception as error:
            state['last_error']=repr(error)
            try:event('controller_error_'+hashlib.sha256(repr(error).encode()).hexdigest()[:12],repr(error))
            except Exception as delivery:state['notification_error']=repr(delivery)
        if time.time()-state['heartbeat_at']>=7200:
            try:event('heartbeat_'+str(int(time.time()/7200)),'Review progress, queue, storage and next authorized actions.');state['heartbeat_at']=time.time()
            except Exception as delivery:state['notification_error']=repr(delivery)
        state['utc']=now();atomic(p,state);time.sleep(120)
    state.update(closed=True,utc=now());atomic(p,state)
if __name__=='__main__':main()
