"""Restart-only1500->2500 continuation using preserved native executable/mesh."""
import math,time,shutil
from common import *
from checkpoint import validate,parameters
from setup_case import section
from restart_identity import gate
from extension_resources import admission,storage
from run_segment import run

def prepare():
    job();admission();work=ROOT/'runs/yc0500'
    marker=EXT/'evidence/CONTINUATION_PREPARATION_OK.json'
    if marker.exists():
        assert (work/'state.json').exists();return work
    source=PARENT/'yc0500_control/runs/yc0500/rst/yc0500.00083.rst'
    d=validate(source);assert d['sha256']==ANCHOR_SHA and d['time']==1500 and d['cycle']==19530 and d['particle_count']==2000000 and not any(d['removal_counts']) and d['ah_onset']<0
    native_diff=subprocess.check_output(['git','-C',str(EXT/'src/athenak'),'diff','2d1f2525','--','src','cmake','CMakeLists.txt'],text=True);assert not native_diff,'Native source change forbidden'
    for folder in ['protected','inputs','samples','evidence','runs/yc0500/rst','runs/yc0500/checkpoint_receipts']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    for dest,src,expected in [(ROOT/'protected/athena',PARENT/'yc0500_control/protected/athena',OLD_EXE_SHA),(ROOT/'samples/yc0500.bin',PARENT/'protected/yc0500.bin','7c031edd772dd42b3823a0fb26f9566d35732c939c169e7b20271b91f96c4339')]:
        assert sha(src)==expected
        if dest.exists():assert dest.resolve()==src.resolve()
        else:dest.symlink_to(src.resolve())
    text=(PARENT/'yc0500_control/inputs/yc0500.athinput').read_text();before=parameters(text);text=section(text,'time',dict(tlim=2500));text=section(text,'problem',dict(cluster_sample_file=ROOT/'samples/yc0500.bin',cluster_t0_dump=work/'initial_particles.bin'))
    exact=ROOT/'inputs/yc0500.athinput';exact.write_text(text)
    for name in ['mesh','meshblock','mesh_refinement','z4c','z4c_amr']+[n for n in before if n.startswith('refined_region')]:
        assert parameters(text)[name]==before[name],name
    target=work/'rst/yc0500.00083.rst';assert not target.exists();shutil.copy2(source,target);copied=validate(target);assert copied['sha256']==ANCHOR_SHA;copied.pop('parameters')
    atomic(work/'checkpoints.json',[copied]);atomic(work/'checkpoint_receipts/yc0500.00083.rst.json',copied)
    atomic(work/'state.json',dict(segments=[],latest=copied,complete=False,inherited_time=1500.))
    atomic(work/'INITIAL_OK.json',dict(utc=now(),job=job(),inherited=True,original_t0_not_regenerated=True,count=2000000,positions_ids_weights_exact=True,source_checkpoint_sha256=ANCHOR_SHA))
    prefix=EXT/'reference/control_0_1500';prefix.mkdir(parents=True,exist_ok=True);sources=[]
    import csv
    for name in ['shells','history','mass','ah']:
        paths=[PARENT/'reference/control_0_700'/(name+'.csv'),PARENT/'yc0500_control/reduced/yc0500'/(name+'.csv')];rows={}
        for path in paths:
            sources.append(receipt(path))
            with path.open() as f:
                for row in csv.DictReader(f):
                    t=float(row['time']);assert t<=1500+1e-7
                    if t in rows:
                        for k in row:assert abs(float(row[k])-float(rows[t][k]))<=1e-8*max(1,abs(float(row[k]))),(name,t,k)
                    rows[t]=row
        ordered=[rows[t] for t in sorted(rows)];assert abs(float(ordered[-1]['time'])-1500)<1e-7
        with (prefix/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(ordered[0]));w.writeheader();w.writerows(ordered)
    for name in ['CASE_NUMERICAL_ACCEPTANCE.json','CASE_VISUAL_ACCEPTANCE.json']:shutil.copy2(PARENT/'yc0500_control/evidence'/name,prefix/name)
    atomic(prefix/'SOURCE_MANIFEST.json',dict(utc=now(),job=job(),sources=sources,copied=[receipt(p) for p in sorted(prefix.iterdir()) if p.is_file()],scope='Accepted unchanged compact0-1500M history only; no historical evolution rerun.'))
    sizes={name:(PARENT/'yc0500_control/runs/yc0500'/folder).glob(pattern) for name,folder,pattern in [('particle','pvtk','*.vtk'),('cohort','cohort','*.s8c')]};particle=max(p.stat().st_size for p in sizes['particle']);cohort=max(p.stat().st_size for p in sizes['cohort']);bins=PARENT/'yc0500_control/runs/yc0500/bin';slice_bytes=sum(max(p.stat().st_size for p in bins.glob('*.'+name+'.*.bin')) for name in ['z4c2d','con2d','tme2d'])
    disk=storage();future=510*(particle+slice_bytes)+2020*cohort+14*copied['bytes']+3*GIB;projected=disk['own']+future
    assert projected<350*GIB,('Continuation forecast exceeds shared350GiB',projected/GIB)
    record=dict(utc=now(),job=job(),case=CASE,original_source=receipt(source),copied_checkpoint=copied,original_executable=receipt(ROOT/'protected/athena'),sample=receipt(ROOT/'samples/yc0500.bin'),exact_input=receipt(exact),native_source_unchanged=True,prior_allocated_node_seconds=PRIOR_ALLOCATED_NODE_SECONDS,storage=disk,conservative_final_combined_storage_bytes=projected,forecast_scope='510full/slice outputs,2020cohorts,14additional checkpoint-sized files including identity/recoveries,3GiB analysis; actual guards stay active. No AMR growth guarantee.',expected_native_seconds_per_M=30.2,expected_additional_native_node_hours=1000*30.2/3600)
    atomic(marker,record);return work

def main():
    job();work=prepare();state=load(work/'state.json')
    if state['complete']:return state
    prior=ROOT/'evidence/PRODUCTION_ADMISSION.json'
    if not prior.exists():
        identity=gate(state['latest']['path']);identity.update(executable_sha256=sha(ROOT/'protected/athena'),machine='amd');atomic(prior,identity)
    from recover import recover
    state=recover(CASE,work)
    from performance_repair import admit
    state=admit(state,run)
    start=float(os.environ['ST8X_START_EPOCH']);limit=int(os.environ['ST8X_WALL_SECONDS'])
    while not state['complete'] and state['latest']['time']<2500-1e-8:
        remaining=limit-(time.time()-start)-240
        if remaining<180:break
        target=min(2500,100*(math.floor((state['latest']['time']+1e-7)/100)+1))
        state=run(CASE,target=target,wall_seconds=int(min(1200,remaining)),suppress_segment_3d=True)
        assert not (ROOT/'evidence/RESOURCE_STOP.json').exists()
        reason=state['segments'][-1].get('stop_reason')
        if reason not in [None,'ah_tail_complete','qualified_AH_checkpoint']:break
    if state['complete']:
        from final_fields import export
        export('yc0500')
    atomic(ROOT/'evidence/jobs'/job()/'CASE_RESULT.json',dict(utc=now(),job=job(),case=CASE,latest=state['latest'],complete=state['complete']))
    return state
