"""Lightweight login-node continuation service. All numerical work is submitted.

No cleanup, no unrelated-job operations, no automatic gate overrides. A resource
failure receives at most one retry from a checksum-verified healthy checkpoint.
The service can be restarted: Slurm accounting and durable state prevent duplicate
launches. Only the root's registered jobs are considered.
"""
import argparse,fcntl,json,math,subprocess,time,traceback
from pathlib import Path
from budget_guard import accounting,require_budget,register,ACTIVE
PREF=461.99570043659014
def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default
def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
class Controller:
    def __init__(self,root):
        self.root=root;self.state=read(root/'state/controller.json',dict(stage='preflight',jobs={},milestones={},retries={}))
    def save(self):write(self.root/'state/controller.json',self.state)
    def halt(self,reason):
        self.state.update(stage='stopped',reason=reason);self.save();print(reason,flush=True)
    def submit(self,key,kind,label,hours,script,args=(),partition=None,extra=()):
        if key in self.state['jobs']:return
        require_budget(self.root,hours,12 if kind=='production' else 0)
        duration=int(math.ceil(hours*60))
        command=['sbatch','--parsable','--account=eliasmost',f'--time={duration}',f'--job-name=pl_s3_{label}']
        if partition:command+=['--partition='+partition]
        command+=list(extra)+[str(self.root/'scripts'/script)]+list(args)
        result=subprocess.run(command,check=True,capture_output=True,text=True)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('unexpected submission output '+result.stdout)
        register(self.root,job,kind,label,hours)
        self.state['jobs'][key]=job;self.save();print('submitted',key,job,flush=True)
    def complete(self,key,records):
        job=self.state['jobs'].get(key)
        if job is None:return False
        r=records.get(job)
        if r is None or r['state'] in ACTIVE:return False
        if r['state']!='COMPLETED':raise RuntimeError(f'{key} job {job} ended {r["state"]}')
        return True
    def plan(self,label,endpoint):
        plans=read(self.root/'state/run_plan.json',{})
        plans[label]=dict(endpoint=endpoint);write(self.root/'state/run_plan.json',plans)
    def run(self,label,endpoint,hours,records,kind='pilot',overrides=()):
        key=f'run:{label}:{endpoint:.12g}'
        previous=read(self.root/'runs'/label/'segment_state.json')
        if previous and previous['health']['last_healthy_time'] is not None:
            if not previous['health']['valid']:raise RuntimeError(label+': '+str(previous['health']['reason']))
            if previous['health']['last_healthy_time']>=endpoint-1.e-9 and previous.get('checkpoint'):
                return True
        jobs=[(k,j) for k,j in self.state['jobs'].items() if k.startswith(key)]
        if jobs:
            latest_key,job=jobs[-1];record=records.get(job)
            if record is None or record['state'] in ACTIVE:return False
            # A batch killed by Slurm can skip its final callback. Submit an
            # allocated audit to verify payload hashes; never hash on the login.
            if previous is None or previous['job_id']!=job:
                audit='audit:'+job
                self.submit(audit,'audit',label+'_audit',.5,'amd_analysis.sbatch',
                    ('audit',label,job,'1',str(record['elapsed_seconds']),record['state']))
                if not self.complete(audit,records):return False
                previous=read(self.root/'runs'/label/'segment_state.json')
            if previous is None:raise RuntimeError('missing audited segment '+label)
            if not previous['health']['valid']:raise RuntimeError(label+': '+str(previous['health']['reason']))
            if previous['returncode'] or record['state']!='COMPLETED':
                if not previous.get('resource_failure') or not previous.get('checkpoint'):
                    raise RuntimeError(label+': failure without verified resource-retry checkpoint')
                n=self.state['retries'].get(key,0)
                if n>=1:raise RuntimeError(label+': repeated resource failure')
                self.state['retries'][key]=n+1
            if not previous.get('checkpoint'):raise RuntimeError(label+': no verified checkpoint')
            key+=':segment'+str(len(jobs))
        self.plan(label,endpoint)
        self.submit(key,kind,label,hours,'amd_run.sbatch',(label,)+tuple(overrides))
        return False
    def tick(self):
        records=accounting(self.root)['records'];stage=self.state['stage']
        if stage in ('stopped','complete'):return
        if (self.root/'state/storage_stop.txt').exists():return self.halt('storage watchdog stopped campaign growth')
        if stage=='preflight':
            for arm,job in [('frozen','452594'),('live','452595')]:
                if job not in records or records[job]['state'] in ACTIVE:return
                s=read(self.root/'runs'/('preflight_'+arm)/'segment_state.json')
                if records[job]['state']!='COMPLETED' or not s or not s['health']['valid']:
                    return self.halt('preflight '+arm+' failed; diagnosis required')
            self.submit('budget','analysis','budget',.5,'amd_analysis.sbatch',('budget',))
            if not self.complete('budget',records):return
            budget=read(self.root/'evidence/budget_preflight.json')
            if not budget or not budget['passed']:return self.halt('measured matrix exceeds the 200-node-hour budget; revised scope required')
            self.state['stage']='pilot';self.save();return
        if stage=='pilot':
            validated=read(self.root/'state/validated_build.json')
            if not validated or not validated.get('passed'):return  # Agent must validate the new binary first.
            measured=read(self.root/'evidence/budget_preflight.json')['measured']
            finished=[]
            for arm in ('frozen','live'):
                for prefix,periods in [('pilot',.30),('split',.30),('short',.025),('halfdt',.025)]:
                    label=prefix+'_'+arm
                    seconds=periods*PREF/measured[arm]['dt_min']*measured[arm]['seconds_per_step']*(2 if prefix=='halfdt' else 1)
                    hours=min(4,max(.5,math.ceil((1.5*seconds+600)/1800)*.5))
                    state=read(self.root/'runs'/label/'segment_state.json')
                    # split's first segment terminates on an exact cycle count;
                    # later segments clear nlim and extend the same healthy record.
                    overrides=('time/nlim=100000000',) if prefix=='split' and state else ()
                    finished.append(self.run(label,periods*PREF,hours,records,overrides=overrides))
            if not all(finished):return
            self.submit('pilot_validation','analysis','pilot_validation',.5,'amd_analysis.sbatch',('pilot',))
            if not self.complete('pilot_validation',records):return
            gate=read(self.root/'state/production_gate.json')
            if not gate or not gate.get('passed'):return self.halt('pilot/restart/timestep gates failed')
            self.state['stage']='production';self.save();return
        if stage=='production':
            gate=read(self.root/'state/production_gate.json')
            if not gate or not gate.get('passed'):return self.halt('production gate missing or revoked')
            complete=[]
            for case in read(self.root/'state/matrix.json'):
                label=case['name'];period=self.state['milestones'].get(label,0)+1
                if period>case['periods']:complete.append(True);continue
                overrides=('problem/plummer_constraint_reference='+str(gate['constraint_reference']),)
                if self.run(label,period*PREF,4,records,'production',overrides):
                    key='reduce:'+label+':'+str(period)
                    self.submit(key,'analysis',label+'_reduce',.5,'amd_analysis.sbatch',('reduce',label))
                    if self.complete(key,records):
                        self.state['milestones'][label]=period;self.save()
                complete.append(False)
            if all(complete):self.state['stage']='complete';self.save()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--watch',action='store_true');a=parser.parse_args()
    with (a.root/'state/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller(a.root)
        while True:
            try:controller.tick()
            except Exception as error:
                traceback.print_exc();controller.halt(str(error));break
            if not a.watch or controller.state['stage'] in ('stopped','complete'):break
            time.sleep(60)
if __name__=='__main__':main()
