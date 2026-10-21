"""One production segment, verified checkpoint promotion, bounded recovery rotation."""
import argparse
import re
import time
import shutil
import numpy as np
from common import *
from checkpoint import validate

def cohort(path):
    with Path(path).open('rb') as f:
        h=f.readline().decode().split();assert h[0]=='S8COHORT1'
        t=float(h[1]);cyc,n,stride=map(int,h[2:])
        mask=np.fromfile(f,dtype='<i4',count=n);data=np.fromfile(f,dtype='<f8').reshape(n,8)
    assert np.isin(mask,[0,1]).all() and np.isfinite(data).all()
    assert np.array_equal(data[mask==1,0],np.arange(n)[mask==1]*stride)
    return t,cyc,mask,data

def admission():
    if os.environ.get('S8_MACHINE')=='perseus':
        from perseus_resources import admission as local_admission
        return local_admission()
    assert not (ROOT/'evidence/RESOURCE_STOP.json').exists(),'Unresolved resource stop'
    whole=int(subprocess.check_output(['du','-s','-B1',str(ROOT.parent)],text=True,timeout=180).split()[0])
    own=int(subprocess.check_output(['du','-s','-B1',str(ROOT)],text=True,timeout=180).split()[0])
    # Reserve the entire remaining S8 allowance and 700 GiB of further Jeans growth.
    forecast=whole+max(0,350*GIB-own)+700*GIB
    d=dict(utc=now(),job=job(),whole=whole,session=own,forecast=forecast,jeans_reserve=700*GIB)
    atomic(ROOT/'evidence/jobs'/job()/'admission.json',d)
    assert whole<1.7*TIB and own<350*GIB and forecast<1.9*TIB,d

def promote(work):
    index=work/'checkpoints.json'
    previous=load(index) if index.exists() else []
    records={x['path']:x for x in previous}
    for p in sorted((work/'rst').glob('*.rst')):
        old=records.get(str(p))
        if old and p.stat().st_size==old['bytes']:continue
        d=validate(p);d.pop('parameters');records[str(p)]=d
        atomic(work/'checkpoint_receipts'/f'{p.name}.json',d)
    ordered=sorted(records.values(),key=lambda x:(x['cycle'],x['path']))
    present=[x for x in ordered if Path(x['path']).exists()]
    assert present
    latest=present[-1]
    # Re-read/hash the proposed restart even when its old receipt exists.
    checked=validate(latest['path']);assert checked['sha256']==latest['sha256']
    protected={x['path'] for x in present[-2:]}
    onset=[x for x in ordered if x['ah_onset']>=0]
    if onset:protected.add(onset[0]['path'])
    finals=[x for x in ordered if x.get('final')]
    protected.update(x['path'] for x in finals)
    rotations=[]
    for d in present[:-2]:
        if d['path'] in protected:continue
        # Only this trajectory's obsolete recovery files; never other outputs.
        p=Path(d['path']);assert p.parent==work/'rst'
        assert receipt(p)['sha256']==d['sha256']
        rotations.append(dict(removed=d,replacement=latest,utc=now(),job=job()))
    if rotations:
        inv=work/'rotation'/f'{job()}_{time.time_ns()}.json';atomic(inv,rotations)
        for d in rotations:Path(d['removed']['path']).unlink()
    atomic(index,ordered)
    return latest

def initial_check(work,c):
    with (ROOT/'samples'/f"{c['id']}.bin").open('rb') as f:
        f.readline();sample=np.fromfile(f,dtype='<f8').reshape(-1,8)
    with (work/'initial_particles.bin').open('rb') as f:
        h=f.readline().decode();native=np.fromfile(f,dtype='<f8').reshape(-1,9)
    assert 'STMIGP05' in h and sample.shape==(2000000,8)
    assert np.array_equal(native[:,0],sample[:,0]) and np.all(native[:,1]==0)
    assert np.array_equal(native[:,2:],sample[:,1:]),'Initial sample changed in native loader'
    from reduce_run import read_pvtk
    p=sorted((work/'pvtk').glob('*.vtk'))[0]
    t,cy,n,f=read_pvtk(str(p));order=np.argsort(f['ptag'])
    assert t==0 and cy==0 and n==len(sample)
    assert np.array_equal(f['ptag'][order],sample[:,0])
    for k,col in [('x',1),('y',2),('z',3),('ux',4),('uy',5),('uz',6),('prtcl_mass',7)]:
        assert np.array_equal(f[k][order],sample[:,col].astype(np.float32)),k
    t,cy,mask,co=cohort(sorted((work/'cohort').glob('*.s8c'))[0])
    assert t==0 and cy==0 and mask.sum()==20000 and np.array_equal(co,sample[::100])
    rows=np.genfromtxt(work/f"{c['id']}.mass.csv",delimiter=',',names=True)
    rows=np.atleast_1d(rows);assert np.all(rows['nblocks']==344)
    assert np.all(rows['dx_min']==(.25 if c['unstable'] else .5))
    atomic(work/'INITIAL_OK.json',dict(utc=now(),job=job(),count=len(sample),cohort=20000,
           native_sample_exact=True,vtk_float32_exact=True,blocks=344,dx=float(rows['dx_min'][0])))

def run(c,initialize=False,target=300,wall_seconds=12600):
    job();admission();work=ROOT/'runs'/c['id'];work.mkdir(parents=True,exist_ok=True)
    statefile=work/'state.json';state=load(statefile) if statefile.exists() else dict(segments=[])
    if initialize and state['segments']:
        assert (work/'INITIAL_OK.json').exists() and state['latest']['time']==0
        return state
    assert not state.get('complete'), 'Already complete'
    active=dict(job=job(),work=str(work),case=c['id'],phase='initialization' if initialize else 'evolution')
    atomic(ROOT/'evidence/active.json',active)
    if initialize and list((work/'rst').glob('*.rst')):
        # Resume validation of a completed zero-step initialization, never redraw it.
        latest=promote(work);assert latest['time']==0 and latest['cycle']==0
        original=[load(p) for p in sorted(work.glob('*.result.json'))]
        assert len(original)==1 and original[0]['returncode']==0
        destination=work/'initial_particles.bin'
        if not destination.exists():
            legacy=ROOT/'src/athenak/initial_particles.bin'
            with legacy.open('rb') as f:header=f.readline().decode()
            assert 'npart=2000000' in header and abs(float(re.search(r'yc=([^ ]+)',header)[1])-c['yc'])<1e-12
            before=receipt(legacy);shutil.copy2(legacy,destination);after=receipt(destination)
            assert before['sha256']==after['sha256']
            atomic(work/'INITIAL_DUMP_RECOVERY.json',dict(source=before,destination=after,utc=now(),job=job()))
        initial_check(work,c)
        record=original[0];record.update(checkpoint=latest,complete=False,initial_check_job=job())
        state.update(segments=[record],latest=latest,complete=False);atomic(statefile,state)
        return state
    log=work/f'{job()}.log';assert not log.exists()
    launch=native_launcher()+[str(ROOT/'protected/athena'),'--kokkos-map-device-id-by=mpi_rank','-d',str(work)]
    start_time=0.0
    if initialize:
        assert not state['segments'] and not list((work/'rst').glob('*.rst'))
        launch+=['-i',str(ROOT/'inputs'/f"{c['id']}.athinput"),'time/nlim=0',
                 'problem/cluster_t0_dump='+str(work/'initial_particles.bin')]
    else:
        assert (work/'INITIAL_OK.json').exists()
        latest=promote(work);assert latest['time']<target
        start_time=latest['time']
        if (work/'REQUEST_STOP').exists():
            reason=(work/'REQUEST_STOP').read_text().strip()
            assert reason.startswith('amr_capacity') and int(os.environ['SLURM_NNODES'])==2,reason
            atomic(work/f'{job()}.cleared_capacity_stop.json',dict(reason=reason,utc=now(),checkpoint=latest))
            (work/'REQUEST_STOP').unlink()
        launch+=['-r',latest['path'],'time/nlim=100000000',f'time/tlim={target}']
    if os.environ.get('S8_MACHINE')=='perseus':
        assert (ROOT/'evidence/MIGRATION_RESTART_OK.json').exists()
        launch+=['mesh_refinement/max_nmb_per_rank=192',
                 'problem/cluster_sample_file='+str(ROOT/'samples'/f"{c['id']}.bin"),
                 'problem/cluster_t0_dump='+str(work/'initial_particles.bin')]
    launch+=['-t',f'{wall_seconds//3600:02d}:{wall_seconds//60%60:02d}:{wall_seconds%60:02d}']
    record=dict(**active,utc=now(),launch=launch,executable=receipt(ROOT/'protected/athena'))
    atomic(work/f'{job()}.launch.json',record)
    start=time.monotonic()
    with log.open('w') as out:result=subprocess.run(launch,stdout=out,stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode,elapsed=time.monotonic()-start,end=now())
    atomic(work/f'{job()}.result.json',record)
    assert result.returncode==0, f'Native failure: {log}'
    txt=log.read_text();assert '### FATAL' not in txt and not re.search(r'search_fail=[1-9]',txt)
    latest=promote(work)
    if initialize:initial_check(work,c)
    history=np.atleast_2d(np.loadtxt(work/f"{c['id']}.user.hst"))
    assert history.shape[1]==19 and np.isfinite(history).all(),'Nonfinite native user history'
    native_mass=np.atleast_1d(np.genfromtxt(work/f"{c['id']}.mass.csv",delimiter=',',names=True))
    assert np.isfinite(native_mass['closure']).all() and np.max(abs(native_mass['closure']))<1e-10
    final=latest['time']>=300-1e-8 or (latest['ah_onset']>=0 and latest['time']>=latest['ah_onset']+20-1e-8)
    stop=re.findall(r'^CLEAN_STOP: (.*)$',txt,re.M)
    advanced=latest['time']-start_time
    record.update(checkpoint=latest,complete=final,stop_reason=stop[-1] if stop else None,
                  simulation_advanced=advanced,
                  native_node_hours_per_M=record['elapsed']*int(os.environ['SLURM_NNODES'])/3600/advanced if advanced>0 else None)
    state['segments'].append(record);state.update(latest=latest,complete=final)
    atomic(statefile,state)
    if final:
        index=load(work/'checkpoints.json')
        for d in index:
            if d['path']==latest['path']:d['final']=True
        atomic(work/'checkpoints.json',index)
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--initialize',action='store_true')
    p.add_argument('--target',type=float,default=300);p.add_argument('--wall-seconds',type=int,default=12600)
    a=p.parse_args();c=next(c for c in CASES if c['id']==a.case)
    print(json.dumps(run(c,a.initialize,a.target,a.wall_seconds),indent=2))
