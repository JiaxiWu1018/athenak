"""Independent endpoint, coverage, quantile, constraint and AH science checks."""
import csv
import re
import numpy as np
from common import *
from reduce import csv_rows
from reduce_run import read_pvtk

def review(c):
    job();work=ROOT/'runs'/c['id'];red=ROOT/'reduced'/c['id']
    state=load(work/'state.json');s=load(red/'summary.json')
    assert state['complete'] and s['complete']
    assert all(load(red/'REDUCTION_OK.json')[k] for k in ['population_ids','removal_ids','mass_ledger','cohort_vs_full_snapshots'])
    endpoint=state['latest']['time'];onset=state['latest']['ah_onset'];expected=min(c['target'],onset+20) if onset>=0 else c['target']
    tolerance=state['latest']['dt']+1e-7 if onset>=0 and expected<c['target'] else 1e-7
    assert expected-1e-7<=endpoint<=min(c['target']+1e-7,expected+tolerance)
    final=load(work/'FINAL_FIELDS_OK.json');assert final['checkpoint']['sha256']==state['latest']['sha256'] and final['no_evolution']
    history=csv_rows(red/'history.csv');mass=csv_rows(red/'mass.csv');shell=csv_rows(red/'shells.csv');ah=csv_rows(red/'ah.csv');snaps=load(red/'snapshots.json')
    with (red/'cohort_index.csv').open() as f:cohorts=list(csv.DictReader(f))
    maxdt=max(float(r[k]) for r in history for k in ['dt','dt_completed']);start=c.get('inherited_time',0)
    streams={}
    for name,values,nominal in [('history',history,.25),('rest_mass',mass,.25),('cohort',cohorts,.5),('full_particles',snaps,2),('AH',ah,.5)]:
        times=np.unique([float(r['time']) for r in values]);assert len(times)>1 and np.isfinite(times).all()
        assert start-1e-7<=times[0]<=start+nominal+maxdt+1e-7
        assert endpoint-times[-1]<=nominal+maxdt+1e-7 and times[-1]<=endpoint+1e-7
        gap=float(np.max(np.diff(times)));assert gap<=nominal+maxdt+2e-7,(name,gap,nominal,maxdt)
        if name!='AH':assert abs(times[-1]-endpoint)<1e-7,(name,'endpoint missing',times[-1],endpoint)
        streams[name]=dict(first_M=float(times[0]),last_M=float(times[-1]),unique_times=len(times),maximum_gap_M=gap,nominal_cadence_M=nominal)
    assert max(abs(r['closure']) for r in mass)<1e-10
    assert all(abs(r['m_alive']/r['m_initial']+sum(r['n_'+k] for k in ['exit','sphere','lapse','horizon'])/2000000-1-r['closure'])<1e-12 for r in mass)
    radii=np.array([[r[f'r{q}'] for q in range(10,100,10)] for r in shell]);assert np.isfinite(radii).all() and (radii>0).all() and (np.diff(radii,axis=1)>=0).all()
    if s['first_removal'] is not None:assert shell[-1]['time']<s['first_removal']
    eligible=[v for v in snaps if s['first_removal'] is None or v['time']<s['first_removal']]
    checks=[]
    for snap in [eligible[0],eligible[len(eligible)//2],eligible[-1]]:
        t,cycle,n,fields=read_pvtk(snap['file']);assert n==2000000
        pos=np.column_stack([fields[k] for k in 'xyz']);rr=np.sort(np.linalg.norm(pos-pos.mean(axis=0),axis=1))
        row=min(shell,key=lambda v:abs(v['time']-t));assert abs(row['time']-t)<1e-7
        for q in range(10,100,10):
            rank=int(np.ceil(n*q/100))-1
            assert rr[max(0,rank-1)]-1e-10<=row[f'r{q}']<=rr[min(n-1,rank+1)]+1e-10,(t,q)
        checks.append(dict(time=t,source=receipt(Path(snap['file'])),all_nine_equal_mass_COM_quantiles='within one independent particle rank'))
    global_path=work/f"{c['id']}.z4c.user.hst";g=np.atleast_2d(np.loadtxt(global_path));assert g.shape[1]==11 and np.isfinite(g).all() and (g[:,10]>0).all() and (g[:,3:5]>=0).all()
    assert abs(g[-1,0]-endpoint)<1e-7
    global_norms=dict(source=receipt(global_path),definition='sqrt(integral constraint squared proper dV / proper unexcised volume); exterior dilution prevents a local-error interpretation',maximum_H_RMS=float(np.sqrt(g[:,3]/g[:,10]).max()),maximum_M_RMS=float(np.sqrt(g[:,4]/g[:,10]).max()))
    good=[r for r in ah if r['accepted']==1];converged=[r for r in ah if r['converged']==1]
    for r in good:
        assert r['streak']>=3 and r['hrel']<=.05 and r['missing']==0 and r['rmin']>=4*r['dx']
        assert r['mass']>0 and np.isclose(r['mirr'],np.sqrt(r['area']/(16*np.pi)),rtol=1e-12)
    assert (onset<0 and not good) or (good and abs(good[0]['time']-onset)<1e-7)
    geometry=None
    if good and sum(s['removals'].values()):
        from removal_geometry import audit
        geometry=audit(ROOT,ah,maxdt,sum(s['removals'].values()),case=c['id'])
    logs=[];fallbacks=0
    for segment in state['segments']:
        if not segment.get('log'):continue
        path=Path(segment['log']);text=path.read_text(errors='replace')
        assert '### FATAL' not in text and not re.search(r'search_fail=[1-9]',text),str(path)
        fallbacks+=sum(int(v) for v in re.findall(r'cycle\s+\d+:\s+(\d+) of \d+ particles fell back',text))
        logs.append(receipt(path))
    initial=None
    if not c.get('inherited_time'):
        initial=load(ROOT/'evidence/INITIAL_CONSTRAINT_ADMISSION.json')
        assert max(initial['constraint_norm_ratios'].values())<=1.005
        native=load(work/'INITIAL_OK.json');assert native['positions_ids_weights_exact'] and native['count']==2000000
    record=dict(utc=now(),job=job(),case=c,checkpoint=state['latest'],endpoint_M=endpoint,expected_endpoint_M=expected,
      streams=streams,maximum_step_M=maxdt,independent_full_population_quantile_checks=checks,
      global_constraints=global_norms,accepted_AH_samples=len(good),converged_AH_samples=len(converged),
      first_accepted_AH_M=good[0]['time'] if good else None,first_converged_AH_M=converged[0]['time'] if converged else None,
      first_removal_M=s['first_removal'],AH_endpoint_window=s['ah_endpoint_window'],removal_geometry=geometry,
      maximum_rest_mass_closure=max(abs(r['closure']) for r in mass),final_population=state['latest']['particle_count'],
      accepted_native_logs=logs,fallback_particle_steps=fallbacks,initial_constraint_admission=initial,
      final_native_fields=receipt(work/'FINAL_FIELDS_OK.json'),numerical_checks_passed=True,actual_visual_review=False,
      scope='Completed trajectory segment only. Accepted AH confirms black-hole formation; survival/coordinate motion does not prove stable migration. Rest-mass closure is bookkeeping, not ADM-energy conservation. Inherited source-time coupling unchanged; no new convergence claim.')
    if c.get('inherited_time'):
        from boundary import tail
        record['outer_annular_lapse_diagnostics']=tail(c)
    atomic(ROOT/'evidence/CASE_NUMERICAL_ACCEPTANCE.json',record)
    return record

if __name__=='__main__':print(json.dumps(review(CASE),indent=2))
