"""One case per allocation, with short clean checkpointed native segments."""
import sys
import shutil
import time
from common import *
from setup_case import setup
from checkpoint import validate
from restart_identity import gate
from run_segment import run
from extension_resources import admission

def inherited_start(c,work):
    statefile=work/'state.json'
    if statefile.exists():return
    source=EXT/'anchors/yc0500_500.rst' if MACHINE=='perseus' else Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023/long/runs/yc0500/rst/yc0500.00013.rst')
    before=validate(source);assert before['sha256']=='7309c1b16b29d5c3de300dadce3ab37be2ba73796ae92b78ed45681134ff2e64' and before['time']==500
    (work/'rst').mkdir(exist_ok=True);destination=work/'rst'/f"{c['id']}.00013.rst"
    shutil.copy2(source,destination);after=validate(destination);assert after['sha256']==before['sha256'];after.pop('parameters')
    atomic(statefile,dict(segments=[],latest=after,complete=False,inherited_start_M=500))
    atomic(work/'INITIAL_OK.json',dict(inherited_original=True,source=receipt(source),utc=now(),job=job(),initial_count=2000000))

def main():
    job();c=CASE;admission();work=setup(c)
    if c.get('inherited_time'):inherited_start(c,work)
    else:
        from fresh_initial import admit
        admit(c,work)
    state=load(work/'state.json')
    from recover import recover
    state=recover(c,work)
    if state.get('complete'):return
    prior=ROOT/'evidence/PRODUCTION_ADMISSION.json'
    compatible=False
    if prior.exists():
        p=load(prior);compatible=p.get('ranks')==int(os.environ['SLURM_NTASKS']) and p.get('max_nmb_per_rank',max(128,512//int(os.environ['SLURM_NTASKS'])))==rank_capacity() and p.get('transport_profile')==transport_profile() and p.get('machine')==MACHINE and p.get('executable_sha256')==sha(ROOT/'protected/athena')
    if not compatible:
        record=gate(state['latest']['path']);record.update(machine=MACHINE,executable_sha256=sha(ROOT/'protected/athena'));atomic(prior,record)
    start=float(os.environ['ST8X_START_EPOCH']);limit=int(os.environ['ST8X_WALL_SECONDS'])
    assert not (ROOT/'evidence/RESOURCE_STOP.json').exists(), 'Resolve resource guard stop before evolution'
    while not state.get('complete'):
        remaining=limit-(time.time()-start)-480
        if remaining<150:break
        state=run(c,target=c['target'],wall_seconds=int(min(300,remaining)),suppress_segment_3d=True)
        if state['segments'][-1].get('stop_reason') not in [None,'ah_tail_complete']:break
        if (ROOT/'evidence/RESOURCE_STOP.json').exists():break
    from reduce import reduce_case
    summary=reduce_case(c,require_complete=state.get('complete',False))
    if state.get('complete'):
        from final_fields import export
        export(c['id'])
        from case_review import review
        review(c)
        from review_packet import export as packet
        packet(c)
        if MACHINE=='amd':
            from legacy_packet import main as legacy
            legacy()
    atomic(ROOT/'evidence/jobs'/job()/'CASE_RESULT.json',dict(utc=now(),job=job(),case=c,state=state['latest'],complete=state.get('complete',False),reduction=summary))

if __name__=='__main__':main()
