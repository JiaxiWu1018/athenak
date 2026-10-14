"""Default regression, sampler ingestion, initial diagnostics and weighted restart gates."""
import json
import os
import re
import time
from pathlib import Path
import numpy as np
import reduce_run as R
from common import ROOT, CASES, N, atomic, sample, sha, require_job
from run_case import launch, checkpoint, newest

def compare(pa,pb,tolerance):
    da=checkpoint(pa); db=checkpoint(pb)
    def particles(d):
        p=np.memmap(d['path'],dtype='<f8',mode='r',offset=d['particle_data_offset'],shape=(10,d['particle_count']))
        return p[:8,np.argsort(p[9])]
    aa,bb=particles(da),particles(db);pd=float(np.max(np.abs(aa-bb)))
    if not np.allclose(aa,bb,atol=tolerance,rtol=tolerance):raise ValueError('Particle restart differs')
    gd=0.;size=da['data_size_per_mb']*da['nmb']
    if size!=db['data_size_per_mb']*db['nmb']:raise ValueError('Grid layouts differ')
    with open(pa,'rb') as fa,open(pb,'rb') as fb:
        fa.seek(da['ascii_bytes']+252+da['nmb']*20+16);fb.seek(db['ascii_bytes']+252+db['nmb']*20+16)
        remain=size
        while remain:
            take=min(remain,8<<20);a=np.frombuffer(fa.read(take),'<f8');b=np.frombuffer(fb.read(take),'<f8')
            gd=max(gd,float(np.max(np.abs(a-b))))
            if not np.allclose(a,b,rtol=tolerance,atol=tolerance):raise ValueError('Grid restart differs')
            remain-=take
    return {'particle_max_abs':pd,'grid_max_abs':gd,'rtol':tolerance,'atol':tolerance}

def main():
    require_job()
    if not (ROOT/'evidence/BUILD_OK').exists() or not (ROOT/'evidence/SAMPLES_OK').exists():raise ValueError('Build/sample gates absent')
    from reference import ensure
    ensure(dict(np.load(ROOT/'samples/model.npz')))
    root=ROOT/'preflight';root.mkdir(exist_ok=True)
    deck=ROOT/'inputs/default_gate.athinput'
    baseline,_=launch(root/'baseline',deck,2,executable='athena.baseline')
    ordinary,_=launch(root/'default',deck,2)
    # The complete seeded DP initialization must remain byte-identical.
    if sha(root/'baseline/initial_particles.bin')!=sha(root/'default/initial_particles.bin'):
        raise ValueError('Historical initialization changed')
    ta,ca,na,a=R.read_pvtk(str(sorted((baseline/'pvtk').glob('*vtk'))[-1]))
    tb,cb,nb,b=R.read_pvtk(str(sorted((ordinary/'pvtk').glob('*vtk'))[-1]))
    if na!=4096 or nb!=4096:raise ValueError('Default gate count')
    for key in a:
        if not np.allclose(a[key],b[key],rtol=1e-7,atol=1e-7):raise ValueError('Default evolution changed '+key)
    records=[]
    for case,rad,method,seed in CASES:
        if seed!=1:continue
        out,wall=launch(root/case,ROOT/'inputs'/(case+'.athinput'),10.)
        initial=ROOT/'samples'/(case+'.bin')
        first=min((out/'rst').glob('*.rst'),key=lambda p:checkpoint(p,initial=initial)['time'])
        d0=checkpoint(first,0.,initial);end=checkpoint(newest(out),10.,initial)
        _,native=sample(initial)
        dump=root/case/'initial_particles.bin'
        with open(dump,'rb') as f:
            header=f.readline();accepted=np.fromfile(f,'<f8').reshape(N,9)
        if not header.startswith(b'STMIGP05') or not np.array_equal(accepted[:,2:8],native[:,1:7]) or not np.array_equal(accepted[:,8],native[:,7]):
            raise ValueError('Pgen did not ingest exact sample')
        _,_,_,pv=R.read_pvtk(str(sorted((out/'pvtk').glob('*vtk'))[0]))
        energy=pv['prtcl_energy']; target=json.loads((ROOT/'evidence/model.json').read_text())['alpha_s']
        if np.max(np.abs(energy-target))>3e-6:raise ValueError('Initial monoenergy mismatch')
        hist=np.loadtxt(out/(case+'.user.hst'))
        if not np.all(hist[0,[3,4,9,10]]>0):raise ValueError('Placeholder initial constraints')
        particles=list((out/'pvtk').glob('*.part.vtk'))
        cubes=list((out/'bin').glob('*.tme3d.*.bin'))+list((out/'bin').glob('*.con3d.*.bin'))
        slices=[p for p in (out/'bin').glob('*.bin') if p not in cubes]
        checkpoints=list((out/'rst').glob('*.rst'))
        frame_bytes=max(p.stat().st_size for p in particles)+sum(p.stat().st_size for p in slices)/len(particles)
        state_bytes=sum(p.stat().st_size for p in cubes+checkpoints)/len(checkpoints)
        memory=[json.loads(line) for line in (root/case/'memory_samples.jsonl').read_text().splitlines()]
        if not any(r['athena_ranks']==4 and len(r['vram_used_bytes'])>=4 for r in memory):
            raise ValueError('Full four-GPU memory telemetry is missing')
        records.append({'case':case,'wall_seconds':wall,'initial':d0,'final':end,
           'sampled_max_host_rss_bytes':max(r['aggregate_host_rss_bytes'] for r in memory),
           'sampled_max_gpu_vram_bytes':max(v for r in memory for v in r['vram_used_bytes'].values()),
           'initial_energy_max_abs_error':float(np.max(np.abs(energy-target))),
           'frame_bytes_estimate':frame_bytes,'full_state_bytes_estimate':state_bytes,
           'raw_bytes':sum(p.stat().st_size for p in (root/case).rglob('*') if p.is_file())})
    case='proper_cell_s1';initial=ROOT/'samples'/(case+'.bin');deck=ROOT/'inputs'/(case+'.athinput')
    # Restart from an INTERIOR checkpoint of the uninterrupted reference. A fresh
    # 0..10M stop clips its last dt and cannot share the 0..12M timestep sequence.
    direct,_=launch(root/'weighted_direct',deck,12.,extra=('output8/dt=2.0',))
    states=[checkpoint(p,initial=initial) for p in sorted((direct/'rst').glob('*.rst'))]
    split=min((s for s in states if 0<s['time']<11.5),key=lambda s:abs(s['time']-10))
    continued,_=launch(root/'weighted_restart',deck,12.,restart=split['path'])
    checkpoint(newest(continued),12.,initial);checkpoint(newest(direct),12.,initial)
    restart=compare(newest(continued),newest(direct),1e-7)
    restart.update(restart_time_M=split['time'],restart_cycle=split['cycle'],
                   method='Interior checkpoint of uninterrupted reference; common adaptive timestep sequence')
    # Toggle only derived initial constraint evaluation; the t=0 primary state is exact.
    disabled,_=launch(root/'diagnostic_disabled',deck,2.,extra=('time/evaluate_initial_constraints=false',))
    dfirst=min((direct/'rst').glob('*.rst'),key=lambda p:checkpoint(p)['time'])
    offirst=min((disabled/'rst').glob('*.rst'),key=lambda p:checkpoint(p)['time'])
    diagnostic=compare(dfirst,offirst,0.)
    # Exercise the full reduction against genuine native GPU data before the suite.
    from analyze import analyze
    begin=time.monotonic()
    analyze(case,run_root=root,dest_root=root/'reduced_qa')
    qa_analysis_seconds=time.monotonic()-begin
    frames=len(list((root/case/'out/pvtk').glob('*.part.vtk')))
    projected=24*max(r['wall_seconds'] for r in records)*30/3600
    # Include time variation, verification/reduction and larger endpoint I/O in the forecast.
    def forecast(end):
        # Interim 0..100M is reduced again during the run; charge that extra work.
        reduction=24*qa_analysis_seconds*((end//2+1)+51)/frames/3600
        return 1.25*projected*end/300+reduction
    end=300 if forecast(300)<=90 else 200
    with_margin=forecast(end)
    if with_margin>90:raise ValueError('Fixed suite exceeds simulation budget')
    case_walltime=int(np.ceil((with_margin/24*3600+600)/60)*60)
    projected_bytes=24*max((end//2+1)*r['frame_bytes_estimate']+(end//100+1)*r['full_state_bytes_estimate']+N*9*8 for r in records)
    atomic(ROOT/'evidence/PREFLIGHT_OK',{'job':os.environ['SLURM_JOB_ID'],'historical_t0_exact':True,
           'cases':records,'weighted_restart':restart,'diagnostic_primary_state':diagnostic,
           'reduction_qa_seconds':qa_analysis_seconds,'reduction_qa_frames':frames,
           'projected_solver_node_hours_300':projected,'projected_production_node_hours_with_margin_and_reduction':with_margin,
           'case_walltime_seconds':case_walltime,'projected_suite_raw_bytes':projected_bytes,'endpoint_M':end})
    print('PREFLIGHT_OK',projected,end,flush=True)

if __name__=='__main__':main()
