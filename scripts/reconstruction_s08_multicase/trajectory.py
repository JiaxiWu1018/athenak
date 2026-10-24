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

def already_accepted(c):
    """A queued successor must preserve a published immutable review packet."""
    work=ROOT/'runs'/c['id'];statefile=work/'state.json';marker=ROOT/'review_packet/MANIFEST.json'
    if not statefile.exists() or not marker.exists():return False
    state=load(statefile);assert state.get('complete'),'Published packet requires a complete state'
    manifest=load(marker);accept=load(ROOT/'evidence/CASE_NUMERICAL_ACCEPTANCE.json')
    final=load(work/'FINAL_FIELDS_OK.json')
    assert manifest['case']==c and accept['numerical_checks_passed'] and final['no_evolution']
    latest=validate(state['latest']['path']);digest=latest['sha256']
    assert digest==state['latest']['sha256']==manifest['checkpoint']['sha256']==accept['checkpoint']['sha256']==final['checkpoint']['sha256']
    expected=min(c['target'],latest['ah_onset']+20) if latest['ah_onset']>=0 else c['target']
    assert expected-1e-7<=latest['time']<=min(c['target']+1e-7,expected+latest['dt']+1e-7)
    for entry in manifest['files']:
        path=marker.parent/entry['relative']
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],str(path)
    atomic(ROOT/'evidence/jobs'/job()/'CASE_RESULT.json',dict(utc=now(),job=job(),case=c,
      state=state['latest'],complete=True,already_accepted=True,immutable_packet=receipt(marker),
      no_native_evolution_or_republication=True))
    return True

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
    job();c=CASE;admission()
    if already_accepted(c):
        from endpoint_manifest import finish
        finish(c)
        return
    if os.environ.get('ST8_IMPORT_SNAPSHOT'):
        from import_snapshot import materialize
        materialize(os.environ['ST8_IMPORT_SNAPSHOT'])
    work=setup(c)
    if c.get('inherited_time'):inherited_start(c,work)
    else:
        from fresh_initial import admit
        admit(c,work)
    state=load(work/'state.json')
    from recover import recover
    state=recover(c,work)
    if os.environ.get('ST8_RECOVER_ONLY')=='1':
        atomic(ROOT/'evidence/jobs'/job()/'PREFIX_REPAIRED.json',dict(utc=now(),job=job(),checkpoint=state['latest'],no_native_evolution=True,original_interrupted_bytes_preserved=True))
        return
    prior=ROOT/'evidence/PRODUCTION_ADMISSION.json'
    compatible=False
    if prior.exists():
        p=load(prior);compatible=p.get('ranks')==int(os.environ['SLURM_NTASKS']) and p.get('max_nmb_per_rank',max(128,512//int(os.environ['SLURM_NTASKS'])))==rank_capacity() and p.get('transport_profile')==transport_profile() and p.get('machine')==MACHINE and p.get('executable_sha256')==sha(ROOT/'protected/athena')
    if not compatible:
        record=gate(state['latest']['path']);record.update(machine=MACHINE,executable_sha256=sha(ROOT/'protected/athena'));atomic(prior,record)
    start=float(os.environ['ST8X_START_EPOCH']);limit=int(os.environ['ST8X_WALL_SECONDS'])
    assert not (ROOT/'evidence/RESOURCE_STOP.json').exists(), 'Resolve resource guard stop before evolution'
    while not state.get('complete'):
        remaining=limit-(time.time()-start)-120
        if remaining<150:break
        state=run(c,target=c['target'],wall_seconds=int(min(300,remaining)),suppress_segment_3d=True)
        if state['segments'][-1].get('stop_reason') not in [None,'ah_tail_complete']:break
        if (ROOT/'evidence/RESOURCE_STOP.json').exists():break
    summary=None
    if state.get('complete') or os.environ.get('ST8_REDUCE_PARTIAL')=='1':
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
            from endpoint_manifest import finish
            finish(c)
    atomic(ROOT/'evidence/jobs'/job()/'CASE_RESULT.json',dict(utc=now(),job=job(),case=c,state=state['latest'],complete=state.get('complete',False),reduction=summary))

if __name__=='__main__':main()
