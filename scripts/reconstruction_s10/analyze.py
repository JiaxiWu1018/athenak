"""Native-history reductions and matched double-precision temporal self-convergence."""
import argparse
import csv
import json
from pathlib import Path
import math
import numpy as np
from campaign import ROOT, CASES
from state_io import require_job, read_rst, interior, write_json, sha

FIELDS=['chi','gxx','gxy','gxz','gyy','gyz','gzz','Khat','Axx','Axy','Axz','Ayy','Ayz','Azz',
        'Gamma_x','Gamma_y','Gamma_z','Theta','alpha','beta_x','beta_y','beta_z','B_x','B_y','B_z']


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert rows,str(path)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def completed(case):
    return sorted((json.loads(p.read_text()) for p in (ROOT/'runs'/case).glob('*/COMPLETE.json')),key=lambda d:d['target'])


def reduce_case(case,dest):
    segments=completed(case)
    if not segments:return None
    health={};global_rows={};sources=[]
    for segment in segments:
        work=Path(segment['work'])
        hp=work/(case+'.user.hst');gp=work/(case+'.z4c.user.hst')
        h=np.atleast_2d(np.loadtxt(hp));g=np.atleast_2d(np.loadtxt(gp))
        assert h.shape[1]==19 and g.shape[1]==11
        assert np.all(np.isfinite(h)) and np.all(np.isfinite(g))
        assert np.all(h[:,[2,8,14,16]]>0) and np.all(g[:,10]>0)
        assert np.all(h[:,[3,4,9,10]]>=0) and np.all(g[:,[3,4]]>=0)
        assert np.allclose(h[:,0],g[:,0],rtol=0,atol=1e-10)
        for v,w in zip(h,g):
            key=round(float(v[0]),9)
            row=dict(time=float(v[0]),dt=float(v[1]),dt_completed=float(v[18]),constraint_usable=True,
                     core_lapse=float(v[15]/v[14]),outer_lapse=float(v[17]/v[16]))
            for rad,o in ((10,2),(16,8)):
                row.update({f'H_rms_r{rad}':float(np.sqrt(v[o+1]/v[o])),
                    f'M_rms_r{rad}':float(np.sqrt(v[o+2]/v[o])),f'E_mean_r{rad}':float(v[o+3]/v[o])})
            # These constraints are synchronized after deposition, including alignment steps.
            health[key]=row
            global_rows[key]=dict(time=float(w[0]),dt_completed=float(v[18]),constraint_usable=True,
                H_rms=float(np.sqrt(w[3]/w[10])),M_rms=float(np.sqrt(w[4]/w[10])))
        sources.extend([dict(path=str(p),sha256=sha(p)) for p in (hp,gp,work/'COMPLETE.json')])
    write_csv(dest/case/'health.csv',[health[k] for k in sorted(health)])
    write_csv(dest/case/'global_constraints.csv',[global_rows[k] for k in sorted(global_rows)])
    summary=dict(case=case,endpoint=segments[-1]['target'],sources=sources,
        core_measure='coordinate-volume RMS within origin-centered coordinate r<10M and r<16M',
        global_measure='native proper-volume RMS over all active mesh cells',
        timing='metric and particles at the same final endpoint; includes final alignment steps')
    write_json(dest/case/'REDUCTION.json',summary)
    return summary


def coordinates(state,mb):
    # LogicalLocation order lx1,lx2,lx3,level; RegionSize stores the three minima followed by the three maxima.
    lx,ly,lz,level=state['locations'][mb]
    ng,nx,ny,nz=state['block'][:4]
    root_n=np.array(state['root'][1:4],dtype=float)
    n=np.array([nx,ny,nz]);box=state['box']
    lo=np.array(box[:3])
    hi=np.array(box[3:6])
    width=(hi-lo)/(root_n/n)*2.**(state['root_level']-int(level))
    center=[lo[a]+(int(loc)+(np.arange(n[a])+.5)/n[a])*width[a] for a,loc in enumerate((lx,ly,lz))]
    r2=center[2][:,None,None]**2+center[1][None,:,None]**2+center[0][None,None,:]**2
    return r2,float(np.prod(width/n))


def differences(states):
    a,b,c=states
    assert a['count']==b['count']==c['count']
    assert abs(a['time']-b['time'])<1e-9 and abs(a['time']-c['time'])<1e-9
    assert np.array_equal(a['locations'],b['locations']) and np.array_equal(a['locations'],c['locations'])
    result=[]
    accum={region:np.zeros((3,25)) for region in ('global','r10','r16')}
    volumes={region:0. for region in accum}
    for m,blocks in enumerate(zip(*(interior(s) for s in states))):
        ga,gb,gc=blocks
        assert all(np.all(np.isfinite(g)) for g in blocks)
        r2,dv=coordinates(a,m)
        for region,radius in [('global',math.inf),('r10',10),('r16',16)]:
            mask=r2<radius**2
            if not np.any(mask):continue
            volumes[region]+=int(mask.sum())*dv
            for k,delta in enumerate((ga-gb,gb-gc,gc)):
                accum[region][k]+=np.sum(delta[:,mask]**2,axis=1)*dv
    box=a['box'];boxvol=(box[3]-box[0])*(box[4]-box[1])*(box[5]-box[2])
    assert abs(volumes['global']/boxvol-1)<1e-12,(volumes,boxvol)
    for region,values in accum.items():
        assert volumes[region]>0
        norm=np.sqrt(values/volumes[region])
        for k,field in enumerate(FIELDS):
            result.append(order_row('grid',field,region,*norm[:,k]))
    pa,pb,pc=[s['particles'][:,s['order']] for s in states]
    assert np.array_equal(pa[9],pb[9]) and np.array_equal(pa[9],pc[9])
    assert np.array_equal(pa[0],pb[0]) and np.array_equal(pa[0],pc[0])
    weight=pa[0]/np.sum(pa[0])
    for name,indices in [('position',[2,4,6]),('covariant_momentum',[3,5,7]),('energy',[1])]:
        qa,qb,qc=[p[indices] for p in (pa,pb,pc)]
        norms=[float(np.sqrt(np.sum(weight*np.sum(q*q,axis=0)))) for q in (qa-qb,qb-qc,qc)]
        result.append(order_row('particles',name,'all',*norms))
    return result


def order_row(kind,field,region,d1,d2,scale):
    floor=100*np.finfo(float).eps*max(float(scale),1e-16)
    valid=d1>floor and d2>floor
    return dict(kind=kind,field=field,region=region,diff_h_h2=float(d1),diff_h2_h4=float(d2),
        reference_rms=float(scale),ratio=float(d1/d2) if d2>0 else None,
        observed_p=float(np.log2(d1/d2)) if valid else None,
        above_roundoff_floor=bool(valid))


def temporal(dest):
    rows=[];sources=[]
    for scheme in ('legacy','predictor'):
        cases=[f'{scheme}_1m_h{h:03d}' for h in (20,10,5)]
        for t in (20,50,100):
            records=[next((r for r in completed(case) if r['target']==t),None) for case in cases]
            if any(r is None for r in records):continue
            states=[read_rst(r['checkpoint']['path']) for r in records]
            # All members must have the same leaf grid, original IDs and masses.
            for row in differences(states):rows.append(dict(scheme=scheme,time=t,**row))
            sources.extend(r['checkpoint'] for r in records)
            del states
    if rows:write_csv(dest/'temporal_order.csv',rows)
    write_json(dest/'TEMPORAL_SOURCES.json',dict(sources=sources,
        formula='p = log2(||Q_h-Q_h/2|| / ||Q_h/2-Q_h/4||), h=.2M',
        grid_norm='coordinate-volume weighted RMS, active cells only, double precision',
        particle_norm='rest-mass weighted vector RMS, matched original particle IDs, double precision',
        caveat='Measured self-convergence of one realization; a constraint noise floor or non-smooth deposition may prevent asymptotic order.'))
    return rows


def window_average(rows,key,lo,hi):
    use=[r for r in rows if str(r.get('constraint_usable','True')).lower() in ('true','1')]
    t=np.array([float(r['time']) for r in use]);v=np.array([float(r[key]) for r in use])
    assert len(t)>1 and np.all(np.diff(t)>0)
    if t[0]>lo+1e-8 or t[-1]<hi-1e-8:return None
    tt=np.r_[lo,t[(t>lo)&(t<hi)],hi]
    return float(np.trapz(np.interp(tt,t,v),tt)/(hi-lo))


def comparison(dest):
    rows=[];scaling=[]
    for n in (1,2,4,8):
        case=f'predictor_{n}m_h020'
        for region,name in [('r10','health.csv'),('r16','health.csv'),('global','global_constraints.csv')]:
            newpath=dest/case/name
            if not newpath.exists():continue
            old=read_csv(ROOT/f'reference/session7/reduced/mc_{n}m_s1'/name);new=read_csv(newpath)
            for metric in ('H','M'):
                key=f'{metric}_rms'+('' if region=='global' else '_'+region)
                for lo,hi in ((0,20),(20,100),(100,300),(300,500)):
                    oldmean=window_average(old,key,lo,hi);newmean=window_average(new,key,lo,hi)
                    if oldmean is None or newmean is None:continue
                    rows.append(dict(N=n*1000000,region=region,constraint=metric,start=lo,end=hi,
                        archived_mean=oldmean,predictor_mean=newmean,ratio_new_old=newmean/oldmean))
    if rows:write_csv(dest/'constraint_windows.csv',rows)
    for region in ('r10','r16','global'):
        for metric in ('H','M'):
            for lo,hi in ((0,20),(20,100),(100,300),(300,500)):
                group=sorted([r for r in rows if (r['region'],r['constraint'],r['start'],r['end'])==(region,metric,lo,hi)],key=lambda r:r['N'])
                if len(group)!=4:continue
                for scheme,key in [('archived','archived_mean'),('predictor','predictor_mean')]:
                    alpha=-float(np.polyfit(np.log([r['N'] for r in group]),np.log([r[key] for r in group]),1)[0])
                    scaling.append(dict(scheme=scheme,region=region,constraint=metric,start=lo,end=hi,alpha_N=alpha,
                        interpretation='Particle-number scaling for one nested seed; not temporal order'))
    if scaling:write_csv(dest/'particle_number_scaling.csv',scaling)
    return rows,scaling


def main():
    require_job();ap=argparse.ArgumentParser();ap.add_argument('--temporal',action='store_true');ap.add_argument('--dest',required=True)
    a=ap.parse_args();dest=Path(a.dest);dest.mkdir(parents=True,exist_ok=False)
    summaries=[s for case in CASES if (s:=reduce_case(case,dest)) is not None]
    assert summaries,'No completed cases to reduce'
    comparison(dest)
    if a.temporal:temporal(dest)
    write_json(dest/'ANALYSIS.json',dict(cases=summaries,script_sha256=sha(__file__)))

if __name__=='__main__':main()
