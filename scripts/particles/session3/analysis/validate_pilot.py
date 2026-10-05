"""Allocated-node pilot, timestep, restart and causal/diffusive geometry gates.

Screens are recorded explicitly; passing them is not a stability result. Scientific
targets are evaluated over full production coverage by the reporting pipeline.
"""
import json,math
from pathlib import Path
import sys,numpy as np
from health import require_window,read_rows
from pvtk_reader import read_pvtk
PREF=461.99570043659014
def unique(run,suffix,key):
    rows=read_rows(next((run/'out').glob('*.plummer_'+suffix+'.csv')));d={}
    for row in rows:
        row={k:float(v) for k,v in row.items()};d[tuple(row[k] for k in key)]=row
    return sorted(d.values(),key=lambda row:tuple(row[k] for k in key))
def moments(run):
    result={}
    for row in unique(run,'physical',('cycle','bin')):
        total=result.setdefault(row['time'],np.zeros(5))
        total+=np.array([row[k] for k in ('M0','E','Sr','Stheta','Sphi')])
    return result
def particle_end(run,limit):
    # Actual cycle establishes ordering; floating time in the VTK title is rounded.
    candidates=[]
    for p in (run/'out/pvtk').glob('*.part.vtk'):
        with p.open('rb') as f:
            f.readline();title=f.readline().decode()
        import re
        match=re.search(r'cycle\s*=\s*(\d+)',title)
        if match:candidates.append((int(match[1]),p))
    if not candidates:raise RuntimeError('pilot final particle frame unavailable')
    d=read_pvtk(max(candidates)[1],{'pos','ptag'})
    assert d['time']<=limit+1.e-3
    tags=d['ptag'].astype(np.int64);assert np.array_equal(np.sort(tags),np.arange(d['n']))
    return d['pos'][np.argsort(tags)].astype(np.float64)
def main(root):
    checks=[];health={};refs=[]
    def check(name,value,limit):checks.append(dict(name=name,value=float(value),limit=float(limit),passed=bool(np.isfinite(value) and value<=limit)))
    for arm in ('frozen','live'):
        for prefix,periods in [('pilot',.30),('split',.30),('short',.025),('halfdt',.025)]:
            run=root/'runs'/f'{prefix}_{arm}';h=require_window(run,allow_stopped=False);health[run.name]=h
            check(run.name+'_duration_shortfall',max(0,periods*PREF-h['last_healthy_time']),1.e-9)
        continuous=root/'runs'/('pilot_'+arm);split=root/'runs'/('split_'+arm)
        c=moments(continuous);s=moments(split);tc=max(c);ts=max(s)
        check('restart_'+arm+'_final_time_difference',abs(tc-ts),1.e-9)
        # Frozen is independent of deposition/particle ordering. Live continuation
        # permits 1e-4 bulk differences, two orders below the science tolerance.
        tol=1.e-7 if arm=='frozen' else 1.e-4
        check('restart_'+arm+'_physical_moments',np.max(np.abs(s[ts]/c[tc]-1)),tol)
        x=particle_end(continuous,tc);y=particle_end(split,ts)
        check('restart_'+arm+'_position_rms_over_a',np.sqrt(np.mean(np.sum((x-y)**2,axis=1)))/10,tol)
        b=moments(root/'runs'/('short_'+arm));half=moments(root/'runs'/('halfdt_'+arm))
        check('halfdt_'+arm+'_physical_moments',np.max(np.abs(b[max(b)]/half[max(half)]-1)),1.e-3)
        initial=c[min(c)];drift=max(np.max(np.abs(value/initial-1)) for value in c.values())
        check('pilot_'+arm+'_bulk_moment_drift',drift,.01)
        rad=unique(continuous,'radii',('cycle','quantile'))
        for q in (.1,.25,.5,.75,.9):
            group=[r for r in rad if abs(r['quantile']-q)<1.e-10]
            first=group[0]
            # Bounds disclose the finite radial histogram resolution rather than
            # interpreting a bin-centre jump as exact physical drift.
            drift=max(max(abs(r['r_lower']/first['r_upper']-1),abs(r['r_upper']/first['r_lower']-1)) for r in group)
            check(f'pilot_{arm}_radius_q{q}',drift,.01)
        if arm=='frozen':
            inv=unique(continuous,'invariants',('cycle',))
            for key,limit in [('energy_relative_rms',1.e-3),('energy_relative_max',1.e-2),('angular_vector_rms_normalized',1.e-3)]:
                check('frozen_'+key,max(r[key] for r in inv),limit)
        else:
            rows=unique(continuous,'health',('cycle',))
            refs=[max(r['H_core_L2'],r['M_core_L2']) for r in rows if r['time']>=.25*PREF]
            if len(refs)<8:raise RuntimeError('post-transient reference requires at least eight history samples')
    reference=float(np.median(refs))
    if not math.isfinite(reference) or reference<=0:raise RuntimeError('invalid live constraint reference')
    speed=max(float(r['coordinate_characteristic_bound']) for label in ('pilot_frozen','pilot_live')
              for r in read_rows(next((root/'runs'/label/'out').glob('*.plummer_health.csv'))))
    conservative=max(2.,1.25*speed);T=5*PREF
    geometry=dict(boundary_half_width=20480.,sampling_radius_max=2000.,duration=T,
      measured_pilot_characteristic_bound=speed,assumed_future_bound=conservative,
      boundary_to_sample_travel_time=(20480-2000)/conservative,
      boundary_to_outer_extraction_travel_time=(20480-10000)/conservative,
      conservative_sample_front=2000+conservative*T,extraction_radii=[9000.,10000.],
      hdamp_cH=.02,psi_bound=1.05,diffusion_rms_one_axis=math.sqrt(2*.02*1.05*T),
      diffusion_rms_3D=math.sqrt(6*.02*1.05*T),
      extraction_level_dx=160.,point_level_interval=[5120*math.sqrt(3),10240],
      stencil_note='Extraction points share level 1. Eight-point derivative stencils can meet AMR seams; positive-control ADM mass agreement is checked independently. Monitor future characteristic bounds.')
    check('boundary_to_matter_duration_ratio',T/geometry['boundary_to_sample_travel_time'],1)
    check('boundary_to_extraction_duration_ratio',T/geometry['boundary_to_outer_extraction_travel_time'],1)
    check('sample_front_below_inner_extraction',geometry['conservative_sample_front']/9000,1)
    initial=unique(root/'runs/pilot_live','admmass',('cycle','R'))
    positive=[r for r in initial if r['cycle']==0]
    if len(positive)<2:raise RuntimeError('missing initial ADM mass positive controls')
    for r in positive:check('ADM_positive_control_'+str(r['R']),abs(r['M_adm']/r['analytic_initial_M_adm']-1),1.e-7)
    gate=dict(passed=all(c['passed'] for c in checks),checks=checks,health=health,
       constraint_reference=reference,reference_definition='median max(H_core_L2,M_core_L2) over pilot [0.25,0.30] P_ref',
       geometry=geometry,reference_period=PREF,restart_note='Comparison at matched final times; VTK FP32 quantization limits particle-position agreement to approximately 1e-7.',
       independent_profile=root.joinpath('evidence/profile_agreement.json').as_posix())
    for name in ('profile_agreement','sampling_cpu_frozen','sampling_cpu_live','ledger_t0'):
        evidence=json.loads((root/'evidence'/f'{name}.json').read_text())
        if not evidence['passed']:gate['passed']=False
    (root/'evidence/pilot_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    (root/'state/production_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps(gate,indent=2));return 0 if gate['passed'] else 2
if __name__=='__main__':sys.exit(main(Path(sys.argv[1])))
