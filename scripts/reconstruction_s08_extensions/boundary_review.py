"""Retained-slice boundary/gauge diagnostics; no claim of causal isolation."""
import numpy as np
import matplotlib.pyplot as plt
from common import *
from extension_resources import EXT
from publication_style import style,save
from st_migration_lib import read_binary,bin_cell_coords
BASE=Path(os.environ['ST8X_BASE'])

def main():
    job();style();edges=np.geomspace(16,512,65);centers=np.sqrt(edges[1:]*edges[:-1]);rows={};bindings=[]
    paths=sorted((BASE/'runs/yc0500/bin').glob('*.z4c2d.*.bin'))+sorted((EXT/'long/runs/yc0500/bin').glob('*.z4c2d.*.bin'))
    for path in paths:
        fd=read_binary(str(path),vis_python=str(BASE/'src/athenak/vis/python'));t=float(fd['time']);assert t<=500+1e-7
        assert 'z4c_alpha' in fd['mb_data'];sums=np.zeros(64);areas=np.zeros(64)
        for mb,values in enumerate(fd['mb_data']['z4c_alpha']):
            assert np.isfinite(values).all()
            x,y,z,dx,dy,dz=bin_cell_coords(fd,mb);xx,yy=np.meshgrid(x,y);r=np.sqrt(xx*xx+yy*yy).ravel();v=np.asarray(values).reshape(-1)
            assert len(r)==len(v)
            k=np.searchsorted(edges,r,side='right')-1;valid=(k>=0)&(k<64)
            sums+=np.bincount(k[valid],weights=v[valid]*dx*dy,minlength=64)
            areas+=np.bincount(k[valid],weights=np.full(int(valid.sum()),dx*dy),minlength=64)
        mean=np.divide(sums,areas,out=np.full(64,np.nan),where=areas>0)
        rows[round(t,8)]=mean;bindings.append(dict(time=t,source=receipt(path)))
    times=np.array(sorted(rows));values=np.stack([rows[t] for t in times]);assert times[0]==0 and abs(times[-1]-load(EXT/'long/runs/yc0500/state.json')['latest']['time'])<1e-7
    delta=values-values[0];finite=np.isfinite(delta);assert finite.any();scale=float(np.max(abs(delta[finite])))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained',gridspec_kw={'width_ratios':[1.7,1]})
    image=axes[0].pcolormesh(centers,times,np.ma.masked_invalid(delta),shading='nearest',cmap='RdBu_r',vmin=-scale,vmax=scale)
    axes[0].set(xscale='log',xlabel=r'Coordinate radius $r/M_{\rm ref}$',ylabel=r'$t/M_{\rm ref}$',title=r'Equatorial annular lapse change from $t=0$');axes[0].axhline(300,c='.2',ls='-.',lw=.8)
    fig.colorbar(image,ax=axes[0],label=r'$\langle\alpha\rangle(r,t)-\langle\alpha\rangle(r,0)$')
    for lo,hi,c,label in [(96,120,'#275d7f','96–120M'),(256,512,'#b35c26','256–512M')]:
        use=(centers>=lo)&(centers<hi);series=np.nanmean(values[:,use],axis=1);axes[1].plot(times,series,c=c,label=label)
    axes[1].set(xlabel=r'$t/M_{\rm ref}$',ylabel='Mean annular lapse',title='Outer equatorial diagnostic');axes[1].legend(frameon=False);axes[1].axvline(300,c='.4',ls='-.',lw=.8);axes[1].grid(alpha=.15,lw=.5)
    save(fig,'s08_extension_boundary_lapse')
    np.savez_compressed(ROOT/'reduced/boundary_lapse.npz',time=times,radius_edges=edges,radius_centers=centers,lapse=values,delta=delta)
    atomic(ROOT/'evidence/BOUNDARY_DIAGNOSTICS.json',dict(job=job(),utc=now(),domain_halfwidth_M=512,sources=bindings,maximum_absolute_annular_lapse_change=scale,minimum_bin_coverage=float(np.mean(finite,axis=1).min()),finite_bin_fraction=float(np.mean(finite)),scope='Equatorial coordinate-area annular averages,16<=r<512M; white bins have no sampled cells. Outer lines average reviewed annular means equally. Gauge/constraint effects cannot be assigned to the boundary from these diagnostics alone; no larger-domain control or proof of causal isolation.'))
if __name__=='__main__':main()
