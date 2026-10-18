"""Allocated reporting readout of already-audited reduced histories.

No native reductions, scientific thresholds, fits to convergence order, or
changes to the approved window definitions. Preserve this as-run script hash.
"""
import csv
import json
import math
import os
from pathlib import Path
import numpy as np
from cfg import ROOT, BASE, P, now, sha, source, require_job
from audit_results import interp, integral

NAMES=['ordinary_1p5m_s1','paired_3m_s1','ordinary_3m_s1','ordinary_6m_s1']

def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))

def location(name):
    return BASE/'reduced' if name==NAMES[0] else ROOT/'reduced'/name

def window_mean(rows,key,lo,hi):
    times=[float(r['time']) for r in rows]
    values=[float(r[key]) for r in rows]
    assert times[0]<=lo and times[-1]>=hi
    t=[lo]+[v for v in times if lo<v<hi]+[hi]
    return integral(t,[interp(v,times,values) for v in t])/(hi-lo)

def entry(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def main():
    require_job()
    target=ROOT/'comparison/SCIENCE_READOUT.json'
    assert not target.exists(),'Preserve previous reporting readout before a new attempt'
    audit=json.loads((ROOT/'evidence/RESULTS_ARITHMETIC_QA.json').read_text())
    assert audit['passed'] and sum(audit['window_rows'].values())==612
    cases={};inputs=[]
    for name in NAMES:
        root=location(name);shells=read(root/'shells.csv');windows=read(root/'windows.csv');health=read(root/'health.csv')
        assert len(shells)==501 and float(shells[0]['time'])==0 and float(shells[-1]['time'])==1000
        summary=json.loads((root/'summary.json').read_text());qvalues={}
        for q in range(10,100,10):
            selected={}
            for tag,start in [('early',P),('late',4*P)]:
                selected[tag]=next(r for r in windows if r['family']=='P' and int(r['quantile'])==q and abs(float(r['start'])-start)<1e-8)
            metrics={tag:{k:float(v[k]) for k in ['half_range_fraction','detrended_half_range_fraction','mean_shift_fraction','slope_fraction_per_M']}for tag,v in selected.items()}
            metrics['raw_growth']=metrics['late']['half_range_fraction']/metrics['early']['half_range_fraction']
            metrics['detrended_growth']=metrics['late']['detrended_half_range_fraction']/metrics['early']['detrended_half_range_fraction']
            initial=float(shells[0][f'r{q}']);metrics['initial_radius_M']=initial;metrics['final_radius_M']=float(shells[-1][f'r{q}'])
            metrics['max_abs_origin_minus_COM_fraction_initial']=max(abs(float(r[f'r{q}_origin'])-float(r[f'r{q}'])) for r in shells)/initial
            qvalues[str(q)]=metrics
        com=np.array([[float(r['com_'+axis])for axis in 'xyz']for r in shells]);distance=np.array([float(r['com_mag'])for r in shells])
        with np.load(root/'profiles.npz') as profiles:
            central=profiles['mass_bins'][:,0];central_ratio=central/central[0]
            central_peak=int(np.argmax(central_ratio));central_stats=dict(peak_over_initial=float(central_ratio[central_peak]),peak_time_M=float(profiles['time'][central_peak]),final_over_initial=float(central_ratio[-1]))
        usable=[r for r in health if r['constraint_usable'].lower()=='true']
        context=dict(COM_initial_M=float(distance[0]),COM_final_M=float(distance[-1]),COM_max_M=float(distance.max()),COM_net_displacement_M=float(np.linalg.norm(com[-1]-com[0])),core_lapse_min=min(float(r['core_lapse'])for r in health),core_lapse_max=max(float(r['core_lapse'])for r in health),central_coordinate_density=central_stats,equilibrium_E_L2_initial=float(summary['field_checks'][0]['E_relative_L2_equilibrium']),equilibrium_E_L2_final=float(summary['field_checks'][-1]['E_relative_L2_equilibrium']),equilibrium_E_L2_max=max(float(r['E_relative_L2_equilibrium'])for r in summary['field_checks']))
        for key in ['H_rms_r10','M_rms_r10','H_rms_r16','M_rms_r16']:
            early=window_mean(usable,key,P,2*P);late=window_mean(usable,key,4*P,5*P)
            context[key]=dict(initial=float(health[0][key]),early=early,late=late,late_over_early=late/early,maximum_usable=max(float(r[key])for r in usable))
        global_path=root/'global_constraints.csv'
        if global_path.exists():
            global_rows=[r for r in read(global_path)if r['constraint_usable'].lower()=='true'];context['global_constraints']={}
            for key in ['H_rms','M_rms']:
                early=window_mean(global_rows,key,P,2*P);late=window_mean(global_rows,key,4*P,5*P)
                context['global_constraints'][key]=dict(early=early,late=late,late_over_early=late/early,maximum_usable=max(float(r[key])for r in global_rows))
            inputs.append(global_path)
        central_energy=root/'central_energy.csv'
        if central_energy.exists():
            rows=read(central_energy);key=next(k for k in rows[0]if k.startswith('E_central'))
            t0=float(rows[0]['time']);v0=float(rows[0][key])
            context['central_E_slice']=dict(first_saved_time_M=t0,last_saved_time_M=float(rows[-1]['time']),initial_normalized_peak_if_t0=max(float(r[key])for r in rows)/v0 if t0==0 else None,definition='Deposited Eulerian energy on z=0,r<.75M; distinct from coordinate rest-mass density. No initial normalization if first saved time is not0.')
            inputs.append(central_energy)
        cases[name]=dict(shells=qvalues,health=context)
        pair_path=root/'pair_symmetry.csv'
        if pair_path.exists():
            pairs=read(pair_path);cases[name]['pair_symmetry']={key:dict(initial=float(pairs[0][key]),final=float(pairs[-1][key]),maximum=max(float(r[key])for r in pairs))for key in ['position_pair_sum_rms_M','position_pair_sum_max_M','covariant_momentum_pair_sum_rms']};inputs.append(pair_path)
        inputs.extend(root/f for f in ['shells.csv','windows.csv','health.csv','profiles.npz','summary.json'])
    differences=read(ROOT/'comparison/shell_differences.csv');ratios=read(ROOT/'comparison/N_difference_ratios.csv')
    selected_differences=[r for r in differences if int(r['quantile'])in[10,50,90]and r['family']in['full','early_P','late_P']]
    selected_ratios=[r for r in ratios if int(r['quantile'])in[10,50,90]and r['family']in['full','early_P','late_P']]
    inputs.extend(ROOT/'comparison'/f for f in ['shell_differences.csv','N_difference_ratios.csv','amplitude_windows.csv','METHOD.json'])
    value=dict(utc=now(),job=os.environ['SLURM_JOB_ID'],execution_source=source(),script=entry(Path(__file__)),period_M=P,early_window_M=[P,2*P],late_window_M=[4*P,5*P],cases=cases,selected_N_differences=selected_differences,selected_N_ratios=selected_ratios,all_window_table='comparison/amplitude_windows.csv',all_N_differences='comparison/shell_differences.csv',all_N_ratios='comparison/N_difference_ratios.csv',inputs=[entry(p)for p in inputs],scope='Reporting readout of actual independently audited full histories; no physical stability threshold or continuum-convergence fit. Native quantities and definitions unchanged.')
    target.write_text(json.dumps(value,indent=2)+'\n')
    print('SCIENCE_READOUT_OK',flush=True)
    print(json.dumps(dict(ordinary6M=cases[NAMES[-1]],N_ratios=selected_ratios),indent=2),flush=True)

if __name__=='__main__':main()
