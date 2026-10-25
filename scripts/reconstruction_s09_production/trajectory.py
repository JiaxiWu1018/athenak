"""Session 9 actual-data admission, bounded segments and independent closeout."""
import time
from common import *
from setup_case import setup
from restart_identity import gate
from run_segment import run
from extension_resources import admission

def main():
    job();c=CASE;admission()
    assert not c.get('reused_session8'),'Reference must never be evolved'
    assert load(EXT/'evidence/PREPARATION_OK.json')['native_source_unchanged']
    provenance=load(EXT/'evidence/NATIVE_BUILD_PROVENANCE.json')
    assert provenance['pgen_source_matches'] and provenance['executable']['sha256']==sha(EXT/'protected/athena')
    assert load(EXT/'evidence/INDEPENDENT_CONSTRAINT_REFERENCE_OK.json')['results']
    work=setup(c)
    if not (work/'state.json').exists() or not load(work/'state.json').get('complete'):
        from fresh_initial import admit
        admit(c,work)
        from recover import recover
        state=recover(c,work)
        prior=ROOT/'evidence/PRODUCTION_ADMISSION.json';p=load(prior) if prior.exists() else {}
        compatible=p.get('ranks')==int(os.environ['SLURM_NTASKS']) and p.get('max_nmb_per_rank')==rank_capacity() and p.get('transport_profile')==transport_profile() and p.get('machine')==MACHINE and p.get('executable_sha256')==sha(ROOT/'protected/athena')
        if not compatible:
            record=gate(state['latest']['path']);record.update(machine=MACHINE,executable_sha256=sha(ROOT/'protected/athena'));atomic(prior,record)
        from runtime_forecast import refresh as forecast
        forecast()
        assert not (ROOT/'evidence/RESOURCE_STOP.json').exists(),'Resolve resource stop before evolution'
        start=float(os.environ['ST8X_START_EPOCH']);limit=int(os.environ['ST8X_WALL_SECONDS'])
        while not state.get('complete'):
            remaining=limit-(time.time()-start)-180
            if remaining<150:break
            state=run(c,target=c['target'],wall_seconds=int(min(300,remaining)),suppress_segment_3d=True)
            forecast()
            if state['segments'][-1].get('stop_reason') not in [None,'ah_tail_complete']:break
            if (ROOT/'evidence/RESOURCE_STOP.json').exists():break
    state=load(work/'state.json')
    if state.get('complete'):
        if not (ROOT/'review_packet/MANIFEST.json').exists():
            remaining=int(os.environ['ST8X_WALL_SECONDS'])-(time.time()-float(os.environ['ST8X_START_EPOCH']))
            from closeout_reserve import estimate
            timing=estimate(c,work,state,int(os.environ['SLURM_NTASKS']))
            atomic(ROOT/'evidence/jobs'/job()/'CLOSEOUT_RESERVE.json',timing)
            full_snapshots=timing['full_particle_snapshots']
            reserve=timing['seconds']
            if remaining<reserve:
                record=dict(utc=now(),job=job(),case=c,checkpoint=state['latest'],
                    remaining_seconds=remaining,full_particle_snapshots=full_snapshots,
                    required_closeout_reserve_seconds=reserve,physical_endpoint_saved=True,
                    numerical_acceptance_complete=False,
                    reason='Preserve the physical endpoint and close it in a fresh allocation; no further evolution is requested.')
                atomic(ROOT/'evidence/jobs'/job()/'CLOSEOUT_DEFERRED.json',record)
                atomic(ROOT/'evidence/jobs'/job()/'CASE_RESULT.json',record)
                return
            from reduce import reduce_case
            reduce_case(c)
            from final_fields import export
            export(c['id'])
            from case_review import review
            review(c)
            from review_packet import export as packet
            packet(c)
        from endpoint_manifest import finish
        finish(c)
    atomic(ROOT/'evidence/jobs'/job()/'CASE_RESULT.json',dict(utc=now(),job=job(),case=c,state=state['latest'],complete=state.get('complete',False)))
if __name__=='__main__':main()
