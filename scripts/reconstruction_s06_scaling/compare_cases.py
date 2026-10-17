"""Time-weighted, fixed-grid N comparisons and migration-style shell figures."""
import csv
import numpy as np
from cfg import *
import analyze as A

NAMES=['ordinary_1p5m_s1','paired_3m_s1','ordinary_3m_s1','ordinary_6m_s1']
LABELS=['Ordinary 1.5M','Antipodal 3M','Ordinary 3M','Ordinary 6M']
COLORS=['#656e76','#ba7132','#247caa','#97802e']
STYLES=['-','--','-.',':']

def read(path):
    with open(path)as stream:return list(csv.DictReader(stream))

def number(rows,key):return np.array([float(r[key])for r in rows])

def reduced(name):return BASE/'reduced' if name==NAMES[0] else ROOT/'reduced'/name

def common_times(a,b,lo,hi):
    assert a[0]<=lo and b[0]<=lo and a[-1]>=hi and b[-1]>=hi
    return np.unique(np.r_[lo,a[(a>lo)&(a<hi)],b[(b>lo)&(b<hi)],hi])

def compare():
    require_job();dest=ROOT/'comparison';dest.mkdir(exist_ok=False)
    data={name:read(reduced(name)/'shells.csv')for name in NAMES}
    model=dict(np.load(ROOT/'samples/model.npz'))
    analytic=np.interp(np.arange(.1,1,.1),model['proper'],model['riso_over_m'])
    assert np.all(analytic>0)
    pairs=[(NAMES[0],NAMES[2]),(NAMES[2],NAMES[3]),(NAMES[1],NAMES[2])]
    rows=[];traces=[]
    windows=[(0,1000,'full'),(P,2*P,'early_P'),(4*P,5*P,'late_P')]
    windows.extend((j*100,(j+1)*100,'100M')for j in range(10))
    for first,second in pairs:
        a,b=data[first],data[second];ta,tb=number(a,'time'),number(b,'time')
        for q,rref in zip(range(10,100,10),analytic):
            for lo,hi,family in windows:
                t=common_times(ta,tb,lo,hi)
                delta=(np.interp(t,tb,number(b,f'r{q}'))-np.interp(t,ta,number(a,f'r{q}')))/rref
                mean=float(np.trapz(delta,t)/(hi-lo));rms=float(np.sqrt(np.trapz(delta**2,t)/(hi-lo)))
                rows.append(dict(first=first,second=second,quantile=q,start=lo,end=hi,family=family,
                    common_equilibrium_radius_M=float(rref),signed_mean_difference_fraction=mean,
                    RMS_difference_fraction=rms,max_abs_difference_fraction=float(np.abs(delta).max())))
                if family=='full':
                    traces.extend(dict(first=first,second=second,quantile=q,time=float(tt),difference_fraction=float(dd))for tt,dd in zip(t,delta))
    A.write_csv(dest/'shell_differences.csv',rows);A.write_csv(dest/'difference_histories.csv',traces)
    amplitude=[]
    for name in NAMES:
        for row in read(reduced(name)/'windows.csv'):
            amplitude.append(dict(case=name,**row))
    A.write_csv(dest/'amplitude_windows.csv',amplitude)
    ratios=[]
    for q in range(10,100,10):
        for lo,hi,family in windows:
            selected=[next(r for r in rows if r['first']==a and r['second']==b and r['quantile']==q
                and r['start']==lo and r['end']==hi and r['family']==family)for a,b in pairs[:2]]
            coarse,fine=[r['RMS_difference_fraction']for r in selected]
            ratios.append(dict(quantile=q,start=lo,end=hi,family=family,coarse_RMS_fraction=coarse,
                fine_RMS_fraction=fine,fine_over_coarse_RMS=fine/coarse if coarse>0 else None))
    A.write_csv(dest/'N_difference_ratios.csv',ratios)
    atomic(dest/'METHOD.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),
        period_M=P,common_equilibrium_radii_M=analytic.tolist(),
        history_times='Native times retained. Union of actual saved times plus exact window endpoints used only for differences.',
        RMS_definition='sqrt(integral[(r_second-r_first)/r_equilibrium]^2 dt / window length)',
        difference_ratio='RMS(3M,6M)/RMS(1.5M,3M); descriptive one-seed fixed-grid diagnostic, not a fitted order',
        baseline_pre300='Exact native nine-decile summaries, not quantiles inferred from 20000 displayed particles',
        inputs=[dict(path=str(reduced(n)/'shells.csv'),sha256=sha(reduced(n)/'shells.csv'))for n in NAMES]))
    plot(data,dest)

def plot(data,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':240})
    figures=ROOT/'plots';figures.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),bbox_inches='tight');fig.savefig(figures/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    for center,suffix in [('COM',''),('origin','_origin')]:
        fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,layout='constrained')
        for i,q in enumerate((10,50,90)):
            for name,label,color,style in zip(NAMES,LABELS,COLORS,STYLES):
                d=data[name];t=number(d,'time');r=number(d,f'r{q}'+suffix)
                axes[i,0].plot(t,r,color=color,ls=style,lw=1.35,label=label)
                axes[i,1].plot(t,100*(r/r[0]-1),color=color,ls=style,lw=1.35)
            axes[i,0].set_ylabel(f'$r_{{{q}}}/M$');axes[i,1].set_ylabel(f'$100(r_{{{q}}}/r_{{{q},0}}-1)$ [%]')
            axes[i,1].axhline(0,color='#444444',lw=.5)
        axes[0,0].legend(fontsize=10);axes[0,0].set_title('Enclosed rest-mass shell radius')
        axes[0,1].set_title('Change relative to each case’s initial radius')
        for ax in axes.ravel():ax.set_xlim(0,1000)
        for ax in axes[-1]:ax.set_xlabel('$t/M$')
        for ax in axes[0]:ax.secondary_xaxis('top',functions=(lambda t:t/P,lambda p:p*P)).set_xlabel('$t/P_{surface}$')
        fig.suptitle(f'ST $y_c=0.819$ · radii about {center} · fixed mesh, stratified seed 1')
        save(fig,'shells_'+center.lower())
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,layout='constrained')
    for name,label,color,style in zip(NAMES,LABELS,COLORS,STYLES):
        d=data[name];h=read(reduced(name)/'health.csv');t=number(h,'time')
        axes[0,0].plot(number(d,'time'),number(d,'com_mag'),label=label,color=color,ls=style)
        axes[0,1].plot(t,number(h,'core_lapse'),color=color,ls=style)
        # Retained baseline central-E slices start only at302M. The complete
        # COM-centred coordinate-mass profile provides an honest t=0 density
        # normalization for every case; never label E(302M) as E(t=0).
        with np.load(reduced(name)/'profiles.npz')as profiles:
            v=profiles['mass_bins'][:,0]
            axes[1,0].plot(profiles['time'],v/v[0],color=color,ls=style)
        mask=np.array([str(r['constraint_usable']).lower()=='true'for r in h])
        axes[1,1].plot(t[mask],number(h,'H_rms_r10')[mask],color=color,ls=style)
        axes[2,0].plot(t[mask],number(h,'M_rms_r10')[mask],color=color,ls=style)
        checks=load(reduced(name)/'summary.json')['field_checks']
        axes[2,1].plot([r['time']for r in checks],[r['E_relative_L2_equilibrium']for r in checks],color=color,ls=style,marker='o',ms=3)
    labels=['COM distance / M','Core lapse','Central coordinate density / initial','Hamiltonian RMS, r<10M',
            'Momentum RMS, r<10M','Relative deposited-E L2 departure']
    for ax,label in zip(axes.ravel(),labels):ax.set_ylabel(label);ax.set_xlim(0,1000)
    for ax in (axes[1,1],axes[2,0]):ax.set_yscale('log')
    axes[0,0].legend(fontsize=10)
    for ax in axes[-1]:ax.set_xlabel('$t/M$')
    save(fig,'health_comparison')
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True,layout='constrained')
    amplitudes=read(dest/'amplitude_windows.csv')
    for i,q in enumerate((10,50,90)):
        for name,label,color,style in zip(NAMES,LABELS,COLORS,STYLES):
            rr=[r for r in amplitudes if r['case']==name and r['family']=='P' and int(r['quantile'])==q]
            t=(number(rr,'start')+number(rr,'end'))/2
            axes[i,0].plot(t,100*number(rr,'half_range_fraction'),label=label,color=color,ls=style,marker='o',ms=4)
            axes[i,1].plot(t,100*number(rr,'detrended_half_range_fraction'),color=color,ls=style,marker='o',ms=4)
        axes[i,0].set_ylabel(f'r{q} half-range [% initial]');axes[i,1].set_ylabel(f'r{q} detrended half-range [%]')
    axes[0,0].legend(fontsize=10);axes[0,0].set_title('Raw amplitude, one reference period per window')
    axes[0,1].set_title('Amplitude after subtracting linear drift')
    for ax in axes[-1]:ax.set_xlabel('Window midpoint $t/M$')
    save(fig,'amplitude_comparison')
    atomic(dest/'PLOTS_RENDERED.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),
        actual_visual_review_pending=True,files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(figures.glob('*'))]))

if __name__=='__main__':compare()
