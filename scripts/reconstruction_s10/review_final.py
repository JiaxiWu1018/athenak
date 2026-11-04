"""Independent scalar checks of the final reductions; no new simulation or DP reduction."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from campaign import ROOT
from state_io import require_job, write_json


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def average(path, key, lo, hi):
    data=[r for r in rows(path) if r.get('constraint_usable','True').lower() in ('true','1')]
    points=[(float(r['time']),float(r[key])) for r in data]
    assert points[0][0]<=lo+1e-8 and points[-1][0]>=hi-1e-8
    assert all(b[0]>a[0] for a,b in zip(points,points[1:]))
    areas=[]
    for (a,x),(b,y) in zip(points,points[1:]):
        left=max(a,lo);right=min(b,hi)
        if right<=left:continue
        vl=x+(y-x)*(left-a)/(b-a);vr=x+(y-x)*(right-a)/(b-a)
        areas.append((right-left)*(vl+vr)/2)
    return math.fsum(areas)/(hi-lo)


def main():
    require_job()
    parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);a=parser.parse_args()
    data=a.packet/'reduced';orders=rows(data/'temporal_order.csv')
    valid=0;undefined=0
    for r in orders:
        d1,d2,scale=(float(r[k]) for k in ('diff_h_h2','diff_h2_h4','reference_rms'))
        floor=100*2.220446049250313e-16*max(scale,1e-16)
        above=d1>floor and d2>floor
        assert above==(r['above_roundoff_floor']=='True')
        if above:
            assert abs(float(r['observed_p'])-math.log(d1/d2,2))<5e-14
            valid+=1
        else:
            assert not r['observed_p'];undefined+=1
    windows=rows(data/'constraint_windows.csv')
    for r in windows:
        region=r['region'];name='global_constraints.csv' if region=='global' else 'health.csv'
        key=r['constraint']+'_rms'+('' if region=='global' else '_'+region)
        n=int(r['N'])//1000000;lo=float(r['start']);hi=float(r['end'])
        old=average(ROOT/f'reference/session7/reduced/mc_{n}m_s1'/name,key,lo,hi)
        new=average(data/f'fixed_fast_v1_{n}m_h010'/name,key,lo,hi)
        for calculated,stored in ((old,float(r['archived_mean'])),(new,float(r['predictor_mean'])),(new/old,float(r['ratio_new_old']))):
            assert math.isclose(calculated,stored,rel_tol=1e-10,abs_tol=1e-20),(r,calculated,stored)
    scalings=rows(data/'particle_number_scaling.csv')
    for r in scalings:
        group=sorted([w for w in windows if all(w[k]==r[k] for k in ('region','constraint','start','end'))],key=lambda w:int(w['N']))
        assert len(group)==4
        key='archived_mean' if r['scheme']=='archived' else 'predictor_mean'
        x=[math.log(int(w['N'])) for w in group];y=[math.log(float(w[key])) for w in group]
        xm=math.fsum(x)/4;ym=math.fsum(y)/4
        alpha=-math.fsum((a-xm)*(b-ym) for a,b in zip(x,y))/math.fsum((a-xm)**2 for a in x)
        assert abs(alpha-float(r['alpha_N']))<1e-12
    summary=json.loads((a.packet/'FINAL_SUMMARY.json').read_text())
    matched={}
    for metric in ('H','M'):
        key=metric+'_rms_r10'
        matched[metric]=average(data/'fixed_fast_v1_1m_h010/health.csv',key,20,100)/average(data/'legacy_1m_h010/health.csv',key,20,100)
        assert math.isclose(matched[metric],summary['matched_h010_core_ratios'][metric],rel_tol=1e-12)
    comparisons={}
    for field in ('position','covariant_momentum','energy'):
        old=next(r for r in orders if r['scheme']=='legacy' and r['field']==field and float(r['time'])==100)
        new=next(r for r in orders if r['scheme']=='fixed_fast_v1' and r['field']==field and float(r['time'])==100)
        comparisons[field]=float(new['diff_h_h2'])/float(old['diff_h2_h4'])
    figs=json.loads((a.packet/'figures/FIGURES.json').read_text())['files'];assert len(figs)==12
    for r in figs:
        p=a.packet/'figures'/r['path'];assert p.stat().st_size==r['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    write_json(a.packet/'SCALAR_REVIEW.json',dict(passed=True,valid_temporal_rows=valid,undefined_temporal_rows=undefined,
        constraint_windows_verified=len(windows),scaling_rows_verified=len(scalings),figure_hashes_verified=len(figs),
        matched_h010_core_ratios=matched,matched_h010_h005_self_difference_new_over_legacy_at100M=comparisons,
        method='Independent stdlib logarithms, clipped piecewise-linear trapezoids, and closed-form least-squares slopes; does not repeat large matched-state reductions.',
        interpretation='Differences are temporal self-differences, not errors against an exact ST solution. Particle orders about1.0-1.3 do not establish second order; one nested realization and restart transients limit interpretation.',
        manual_visual_review_complete=False))


if __name__=='__main__':main()
