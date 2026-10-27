"""Record an explicitly requested clean handoff from a frozen older controller."""
from common import *


def review(ticket):
    previous = ticket.get('handoff_from')
    if not previous:
        return
    transition = load(EXT/'evidence/WORK_POLICY_TRANSITION.json')
    assert transition['old_job'] == previous and resource_policy()['campaign_storage_GiB'] is None
    state = load(ROOT/'runs/yc0500/state.json')
    last = state['segments'][-1]
    assert last['job'] == previous and last['returncode'] == 0
    assert last['stop_reason'] == transition['reason']
    a = EXT/'evidence/allocations'/previous
    assert (a/'WORKER_OK.json').exists() and not (a/'FAILURE.json').exists()
    end = load(a/'ALLOCATION_END.json')
    assert end['exit_status'] == 0
    query = subprocess.run(['scontrol','show','job',previous,'-o'],capture_output=True,text=True)
    raw = query.stdout if query.returncode==0 else load(a/'scheduler_terminal.json')['scheduler']
    assert ('JobState=FAILED' in raw and 'ExitCode=1:0' in raw) or ('JobState=COMPLETED' in raw and 'ExitCode=0:0' in raw)
    atomic(a/'scheduler_terminal.json',dict(utc=now(),scheduler=raw.strip(),captured_by=job()))
    record = dict(utc=now(),job=previous,may_retry=False,failure=None,allocation_end=end,
        scheduler=raw.strip(),checkpoint=state['latest'],
        reason='Requested clean checkpoint handoff to directly approved WORK policy; native exit0 and numerical checkpoint accepted')
    acknowledgments = load(EXT/'evidence/REVIEWED_FAILURES.json')
    existing = [v for v in acknowledgments if v['job']==previous]
    if not existing:
        acknowledgments.append(record)
        atomic(EXT/'evidence/REVIEWED_FAILURES.json',acknowledgments)
    pending = EXT/'evidence/DISPATCH_FAILURE.json'
    if pending.exists():
        failure = load(pending)
        assert failure['job']==previous and failure['error']==repr(AssertionError(transition['reason']))
        archive = EXT/'evidence/metadata_history'/('work_policy_handoff_'+previous)
        archive.mkdir(parents=True,exist_ok=True)
        assert not (archive/pending.name).exists()
        pending.rename(archive/pending.name)
    atomic(EXT/'evidence/allocations'/job()/'WORK_POLICY_HANDOFF_ACCEPTED.json',record)
