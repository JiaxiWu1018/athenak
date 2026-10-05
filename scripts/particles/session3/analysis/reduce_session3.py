"""Allocated-node reduction. Health-bounded ledgers are the physical-moment source.

Particle dumps provide exact tags, coordinate radii, angular vectors and fixed movie
subsets. Never replace live physical stress by coordinate-velocity dispersion.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from health import require_window,read_rows
from pvtk_reader import read_pvtk
from sph_real import ylm_all
PREF=461.99570043659014

def table(path,limit,keys):
    unique={}
    for row in read_rows(path):
        row={k:float(v) for k,v in row.items()}
        if row['time']<=limit+1.e-9:unique[tuple(row[k] for k in keys)]=row
    return sorted(unique.values(),key=lambda row:tuple(row[k] for k in keys))

def save_csv(path,rows):
    if not rows:raise RuntimeError(f'no valid rows for {path}')
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--until',type=float);a=ap.parse_args()
    window=require_window(a.run);limit=window['last_healthy_time']
    if a.until is not None:limit=min(limit,a.until)
    a.out.mkdir(parents=True,exist_ok=True)
    physical=table(next((a.run/'out').glob('*.plummer_physical.csv')),limit,('cycle','bin'))
    radii=table(next((a.run/'out').glob('*.plummer_radii.csv')),limit,('cycle','quantile'))
    health=table(next((a.run/'out').glob('*.plummer_health.csv')),limit,('cycle',))
    grouped={}
    for r in physical:grouped.setdefault(r['cycle'],[]).append(r)
    totals=[];profiles=[]
    for cycle,rows in grouped.items():
        total=dict(time=rows[0]['time'],t_over_P=rows[0]['time']/PREF,cycle=cycle)
        for key in ('M0','E','Sr','Stheta','Sphi'):total[key]=sum(r[key] for r in rows)
        total['anisotropy']=1-(total['Stheta']+total['Sphi'])/(2*total['Sr'])
        totals.append(total)
        for r in rows:
            row=dict(r);V=r['proper_volume'];N=r['N']
            row['rho0']=r['M0']/V if V else 0
            for key in ('E','Sr','Stheta','Sphi'):
                row[key+'_density']=r[key]/V if V else 0
                # Conservative spatial-pair uncertainty. Temporal uncertainty is
                # assessed independently from frozen correlation blocks.
                row[key+'_sem']=math.sqrt(max(0,2*(r[key+'_square']-r[key]**2/max(N,1))))
            row['resolved']=float(r['rhi']-r['rlo']>=4*r['dx_volume_mean'] and N>=200)
            profiles.append(row)
    save_csv(a.out/'integrated.csv',totals);save_csv(a.out/'profiles.csv',profiles)
    save_csv(a.out/'radii.csv',radii);save_csv(a.out/'health.csv',health)
    for suffix in ('invariants','admmass','admmom'):
        found=list((a.run/'out').glob(f'*.plummer_{suffix}.csv'))
        if found:
            rows=table(found[0],limit,('cycle','R') if suffix.startswith('adm') else ('cycle',))
            if rows:save_csv(a.out/f'{suffix}.csv',rows)
    # Shell Ylm ledgers have full history cadence and retain signed coefficients.
    shells=table(next((a.run/'out').glob('*.plummer_shells.csv')),limit,('cycle','bin'))
    save_csv(a.out/'shell_modes.csv',shells)
    mode_rows=[];particle_rows=[];movie=[];selected=None;bands=None
    healthy_cycles={int(r['cycle']):r['time'] for r in health}
    last_cycle=max(healthy_cycles)
    seen_cycles=set()
    for file in sorted((a.run/'out/pvtk').glob('*.part.vtk')):
        d=read_pvtk(file,{'pos','ptag','prtcl_vel','prtcl_energy'})
        if d['cycle']>last_cycle:continue
        if d['cycle'] in seen_cycles:continue
        seen_cycles.add(d['cycle'])
        t=healthy_cycles.get(d['cycle'],d['time'])
        if t>limit+1.e-4:continue  # VTK's displayed time is rounded; cycle bound is authoritative.
        tags=d['ptag'].astype(np.int64)
        if len(np.unique(tags))!=d['n'] or not np.array_equal(np.sort(tags),np.arange(d['n'])):
            raise RuntimeError(f'particle tag accounting failed in {file}')
        pos=d['pos'].astype(np.float64)
        if not np.isfinite(pos).all():raise RuntimeError(f'non-finite particle dump {file}')
        com=pos.mean(axis=0);r=np.linalg.norm(pos,axis=1);centred=pos-com
        cr=np.linalg.norm(centred,axis=1)
        if bands is None:bands=np.quantile(r,[.25,.5,.75])
        particle_rows.append(dict(time=t,t_over_P=t/PREF,N=d['n'],com_x=com[0],com_y=com[1],com_z=com[2],
             **{f'r_iso_q{int(q*100)}':float(np.quantile(r,q)) for q in (.1,.25,.5,.75,.9)}))
        for origin,points,rr in [('com',centred,cr),('origin',pos,r)]:
            direction=points/np.maximum(rr[:,None],1.e-100)
            ids=np.searchsorted(bands,cr)
            for l in range(1,5):
                Y=ylm_all(direction[:,0],direction[:,1],direction[:,2],l)[l]
                for band in range(-1,4):
                    keep=np.ones(d['n'],dtype=bool) if band<0 else ids==band
                    N=int(keep.sum())
                    if not N:continue
                    coeff=Y[:,keep].mean(axis=1)
                    amplitude=float(np.sqrt(4*np.pi/(2*l+1)*np.sum(coeff**2)))
                    row=dict(time=t,t_over_P=t/PREF,origin=origin,band='all' if band<0 else f'band{band}',
                             l=l,N=N,amplitude=amplitude)
                    # Uniform column layout accommodates every l while preserving
                    # signed coefficients and the physical dipole vector.
                    for m in range(-4,5):row[f'coef_m{m}']=float(coeff[m+l]) if abs(m)<=l else 0.
                    vec=direction[keep].mean(axis=0) if l==1 else np.full(3,np.nan)
                    row.update(dipole_x=vec[0],dipole_y=vec[1],dipole_z=vec[2]);mode_rows.append(row)
        if selected is None:
            # Fixed tag subset, evenly spaced in the stratified initial radial
            # quantile. Its identities survive migration and restart exactly.
            selected=np.unique(np.linspace(0,d['n']//2-1,8192,dtype=np.int64)*2)
        index=np.empty(d['n'],dtype=np.int64);index[tags]=np.arange(d['n'])
        movie.append(dict(time=t,pos=pos[index[selected]].astype(np.float32)))
    if particle_rows:
        save_csv(a.out/'particles.csv',particle_rows);save_csv(a.out/'modes.csv',mode_rows)
        np.savez_compressed(a.out/'movie_particles.npz',time=np.array([r['time'] for r in movie]),
                            pos=np.array([r['pos'] for r in movie]),tags=selected)
    report=dict(run=str(a.run),health=window,tmax=limit,reference_period=PREF,
                particle_frames=len(particle_rows),mode_bands='fixed initial coordinate-radius quartile boundaries; COM directions',
                proper_stress_source='in-code orthonormal frame and evolved ADM metric',
                radius_definition='evolved-metric tangential areal-radius proxy, dense histogram with retained bounds',
                initial_pair_noise='N/2; later noise measured from matched frozen time series')
    (a.out/'reduction.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
