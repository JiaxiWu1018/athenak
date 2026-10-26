"""Preserved annular lapse convention; document outer-domain behavior to700M."""
import numpy as np
from common import *

def tail(c):
    job();assert c['L']==512
    from st_migration_lib import read_binary,bin_cell_coords
    edges=np.geomspace(16,512,65);rows={};sources=[]
    for path in sorted((ROOT/'runs'/c['id']/'bin').glob('*.z4c2d.*.bin')):
        field=read_binary(str(path),vis_python=str(Path(os.environ['ST8X_BASE'])/'src/athenak/vis/python'))
        t=float(field['time']);sums=np.zeros(64);areas=np.zeros(64)
        for block,values in enumerate(field['mb_data']['z4c_alpha']):
            assert np.isfinite(values).all()
            x,y,z,dx,dy,dz=bin_cell_coords(field,block);xx,yy=np.meshgrid(x,y);radius=np.sqrt(xx*xx+yy*yy).ravel();v=np.asarray(values).reshape(-1);assert len(radius)==len(v)
            index=np.searchsorted(edges,radius,side='right')-1;valid=(index>=0)&(index<64)
            sums+=np.bincount(index[valid],weights=v[valid]*dx*dy,minlength=64)
            areas+=np.bincount(index[valid],weights=np.full(int(valid.sum()),dx*dy),minlength=64)
        rows[round(t,8)]=np.divide(sums,areas,out=np.full(64,np.nan),where=areas>0);sources.append(dict(time=t,source=receipt(path)))
    times=np.array(sorted(rows));assert len(times)>1 and times[0]>=c.get('inherited_time',0)-1e-7
    destination=ROOT/'reduced'/c['id']/'boundary_tail.npz'
    np.savez_compressed(destination,time=times,radius_edges=edges,radius_centers=np.sqrt(edges[1:]*edges[:-1]),lapse=np.stack([rows[t] for t in times]))
    record=dict(utc=now(),job=job(),case=c,annular_arrays=receipt(destination),sources=sources,scope='Coordinate-area weighted mean of equatorial lapse within each of64 logarithmic annuli16<=r<512M; missing bins masked. Diagnostic only; no larger-box or causal-isolation proof.')
    atomic(destination.with_suffix('.json'),record)
    return record

def figure():
    job()
    import matplotlib.pyplot as plt
    from publication_style import style,save
    style();legacy=EXT/'imported/legacy500_packet';packet=EXT/'imported/yc0500_long700'
    if not packet.exists():packet=EXT/'yc0500_long700/review_packet'
    a=np.load(legacy/'boundary_lapse.npz');b=np.load(packet/'reduced/boundary_tail.npz')
    assert np.array_equal(a['radius_edges'],b['radius_edges'])
    rows={round(float(t),8):v for arrays in [a,b] for t,v in zip(arrays['time'],arrays['lapse'])}
    times=np.array(sorted(rows));values=np.stack([rows[t] for t in times]);centers=a['radius_centers']
    summary=load(ROOT/'reduced/yc0500_long700/summary.json');assert times[0]==0 and abs(times[-1]-summary['endpoint'])<1e-7
    delta=values-values[0];finite=np.isfinite(delta);scale=float(np.max(abs(delta[finite])))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained',gridspec_kw={'width_ratios':[1.7,1]})
    image=axes[0].pcolormesh(centers,times,np.ma.masked_invalid(delta),shading='nearest',cmap='RdBu_r',vmin=-max(scale,1e-15),vmax=max(scale,1e-15))
    axes[0].set(xscale='log',xlabel=r'Coordinate radius $r/M_{\rm ref}$',ylabel=r'$t/M_{\rm ref}$',title='Equatorial annular lapse change from t=0')
    fig.colorbar(image,ax=axes[0],label=r'$\langle\alpha\rangle(r,t)-\langle\alpha\rangle(r,0)$')
    statistics=[];at500=int(np.argmin(abs(times-500)));assert abs(times[at500]-500)<1e-7
    for lo,hi,color in [(96,120,'#275d7f'),(256,512,'#b35c26')]:
        use=(centers>=lo)&(centers<hi);series=np.nanmean(values[:,use],axis=1);assert np.isfinite(series).all()
        axes[1].plot(times,series,c=color,label=f'{lo}–{hi}M')
        statistics.append(dict(radius_min_M=lo,radius_max_M=hi,initial_lapse=float(series[0]),lapse_at500M=float(series[at500]),endpoint_lapse=float(series[-1]),maximum_absolute_change_after500_from500=float(np.max(abs(series[at500:]-series[at500])))))
    for time in [300,500]:
        axes[0].axhline(time,c='.3',ls=':',lw=.8);axes[1].axvline(time,c='.3',ls=':',lw=.8)
    axes[1].set(xlabel=r'$t/M_{\rm ref}$',ylabel='Mean annular lapse',title='Outer equatorial diagnostic');axes[1].legend(frameon=False)
    save(fig,'s08_extension_boundary_lapse')
    np.savez_compressed(ROOT/'reduced/boundary_lapse.npz',time=times,radius_centers=centers,radius_edges=a['radius_edges'],lapse=values,delta=delta)
    atomic(ROOT/'evidence/BOUNDARY_DIAGNOSTICS.json',dict(utc=now(),job=job(),endpoint_M=float(times[-1]),domain_halfwidth_M=512,prior_manifest=receipt(legacy/'MANIFEST.json'),new_slice_sources=load(packet/'reduced/boundary_tail.json'),outer_bands=statistics,maximum_absolute_annular_change=scale,scope='Outer curves equally average annular means, not full-band area weighting. No larger-domain control or proof of causal isolation through the extended interval.'))

if __name__=='__main__':figure()
