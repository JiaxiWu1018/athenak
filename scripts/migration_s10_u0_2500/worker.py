"""Allocated continuation stages; no fresh initialization or solver compilation."""
import socket,traceback
from common import *
def main():
    job();role=os.environ['S10_ROLE'];out=EXT/'evidence/allocations'/job();commit=subprocess.check_output(['git','-C',str(EXT/'src/athenak'),'rev-parse','HEAD'],text=True).strip();assert commit==load(EXT/'evidence/SOURCE_PIN.json')['commit']
    atomic(out/'STARTED.json',dict(utc=now(),job=job(),role=role,case=CASE,host=socket.gethostname(),source_commit=commit))
    ticket=next(x for x in load(EXT/'evidence/job_registry.json') if x['id']==job())
    if ticket.get('after'):
        previous=ticket['after'];q=subprocess.run(['scontrol','show','job',previous,'-o'],capture_output=True,text=True)
        if 'JobState=COMPLETED' in q.stdout and 'ExitCode=0:0' in q.stdout:atomic(EXT/'evidence/allocations'/previous/'scheduler_terminal.json',dict(utc=now(),scheduler=q.stdout.strip(),captured_by_successful_afterok_successor=job()))
    try:
        if role=='preflight':
            from continuation import prepare
            prepare()
        elif role=='production':
            from work_policy_handoff import review as review_handoff
            review_handoff(ticket)
            from continuation import main as evolve
            evolve()
        elif role=='analysis':
            from analyze_extension import main as analyze
            analyze()
        else:raise ValueError(role)
        atomic(out/'WORKER_OK.json',dict(utc=now(),job=job(),role=role,case=CASE))
    except BaseException:
        atomic(out/'FAILURE.json',dict(utc=now(),job=job(),role=role,traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
