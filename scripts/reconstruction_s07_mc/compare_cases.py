"""Four-count shell/lapse residuals, drift and window comparisons at fixed mesh."""
import csv
import numpy as np
from cfg import *
NAMES=[c['id']for c in CASES]
LABELS=['MC 1 million','MC 2 million','MC 4 million','MC 8 million']
COLORS=['#0072b2','#d55e00','#009e73','#cc79a7']
STYLES=['-','--','-.',':']
METRICS=[f'r{q}'for q in range(10,100,10)]+['core_lapse']
PRIMARY=['r10','r50','r90','core_lapse']

def read(path):
    with Path(path).open()as f:return list(csv.DictReader(f))
def number(rows,key):return np.array([float(r[key])for r in rows])
def write(path,rows):
    import analyze as A
    A.write_csv(path,rows)
def bounds():
    return [(j*w,(j+1)*w,f)for w,f in [(100.,'100M'),(P,'P'),(2*P,'2P')]for j in range(int(1000/w))]
def comparison_bounds():
    return [(0.,1000.,'full'),(P,2*P,'early_P'),(4*P,5*P,'late_P')]+[(j*100.,(j+1)*100.,'100M')for j in range(10)]
def common_times(ta,tb,lo,hi):
    assert ta[0]<=lo and tb[0]<=lo and min(ta[-1],tb[-1])>=hi-1e-7
    return np.unique(np.r_[lo,ta[(ta>lo)&(ta<hi)],tb[(tb>lo)&(tb<hi)],hi])
def residual(ta,a,tb,b,lo,hi,scale=1.,changes=False):
    t=common_times(ta,tb,lo,hi)
    if changes:a=a-a[0];b=b-b[0]
    d=(np.interp(t,tb,b)-np.interp(t,ta,a))/scale
    return t,d,dict(RMS=float(np.sqrt(np.trapz(d*d,t)/(hi-lo))),signed_mean=float(np.trapz(d,t)/(hi-lo)),max_abs=float(np.abs(d).max()))
def window(t,y,initial,lo,hi,scale):
    tt=np.r_[lo,t[(t>lo)&(t<hi)],hi];v=np.interp(tt,t,y)
    w=np.r_[np.diff(tt)[0]/2,(tt[2:]-tt[:-2])/2,np.diff(tt)[-1]/2]
    mean=float(np.trapz(v,tt)/(hi-lo));x=tt-(lo+hi)/2
    slope=float(np.sum(w*x*(v-mean))/np.sum(w*x*x));d=v-mean-x*slope
    return dict(mean=mean/scale,mean_shift=(mean-initial)/scale,
        half_range=float(np.ptp(v)/(2*scale)),detrended_half_range=float(np.ptp(d)/(2*scale)),slope_per_M=slope/scale)
def ratio(fine,coarse):return fine/coarse if coarse>0 else None

def main():
    require_job();dest=ROOT/'comparison';dest.mkdir(exist_ok=False)
    model=dict(np.load(ROOT/'samples/model.npz'))
    scales={f'r{q}':float(np.interp(q/100,model['proper'],model['riso_over_m']))for q in range(10,100,10)};scales['core_lapse']=1.
    data={};health={}
    for name in NAMES:
        shells=read(ROOT/'reduced'/name/'shells.csv');h=read(ROOT/'reduced'/name/'health.csv');health[name]=h
        data[name]={m:(number(h if m=='core_lapse'else shells,'time'),number(h if m=='core_lapse'else shells,m))for m in METRICS}
    differences=[];traces=[];windows=[];ratios=[];window_differences=[]
    for name in NAMES:
        for m,(t,y)in data[name].items():
            for lo,hi,family in bounds():windows.append(dict(case=name,metric=m,start=lo,end=hi,family=family,scale=scales[m],**window(t,y,y[0],lo,hi,scales[m])))
    for level,(first,second)in enumerate(zip(NAMES[:-1],NAMES[1:])):
        for m in METRICS:
            ta,a=data[first][m];tb,b=data[second][m]
            for mode in ['absolute','change_from_initial']:
                for lo,hi,family in comparison_bounds():
                    t,d,stats=residual(ta,a,tb,b,lo,hi,scales[m],mode!='absolute')
                    differences.append(dict(level=level,first=first,second=second,metric=m,mode=mode,start=lo,end=hi,family=family,scale=scales[m],**stats))
                    if family=='full':traces.extend(dict(level=level,metric=m,mode=mode,time=float(tt),difference=float(dd))for tt,dd in zip(t,d))
            for lo,hi,family in bounds():
                wa=window(ta,a,a[0],lo,hi,scales[m]);wb=window(tb,b,b[0],lo,hi,scales[m])
                window_differences.append(dict(level=level,first=first,second=second,metric=m,start=lo,end=hi,family=family,**{k:wb[k]-wa[k]for k in wa}))
    for m in METRICS:
        for mode in ['absolute','change_from_initial']:
            for lo,hi,family in comparison_bounds():
                selected=[next(r for r in differences if r['metric']==m and r['mode']==mode and r['start']==lo and r['end']==hi and r['family']==family and r['level']==j)for j in range(3)]
                for j in (0,1):ratios.append(dict(metric=m,mode=mode,start=lo,end=hi,family=family,coarse_level=j,fine_level=j+1,coarse_RMS=selected[j]['RMS'],fine_RMS=selected[j+1]['RMS'],fine_over_coarse=ratio(selected[j+1]['RMS'],selected[j]['RMS'])))
    for name,rows in [('residual_norms',differences),('residual_histories',traces),('metric_windows',windows),('window_differences',window_differences),('residual_ratios',ratios)]:write(dest/(name+'.csv'),rows)
    atomic(dest/'METHOD.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),period_M=P,scales=scales,
        difference='finer minus coarser; three adjacent particle-count pairs',modes=['absolute','change_from_initial'],
        weighting='Trapezoidal actual-time integration on union of native times and exact window edges',
        interpretation='One nested MC seed at fixed mesh; shrinking differences can coexist with amplification; no fitted order, phase alignment or statistical significance.',
        sample_policy='Public seed1,PCG64(3001),nested coordinates and covariant momenta; equal masses M0/N',native_data_host='AMD only'))
    plot(data,health,windows,differences,traces)
    print('COMPARISON_OK',len(differences),len(ratios),flush=True)

def plot(data,health,windows,differences,traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pypdfium2 as pdfium
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':220})
    dest=ROOT/'plots';dest.mkdir(exist_ok=True);preview=ROOT/'review/figures';preview.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),bbox_inches='tight');fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
        document=pdfium.PdfDocument(str(dest/(name+'.pdf')));page=document[0]
        page.render(scale=1.5).to_pil().save(preview/(name+'_pdf.png'));document.close()
    for center in ['com','origin']:
        fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,layout='constrained')
        for j,m in enumerate(['r10','r50','r90']):
            for name,label,color,style in zip(NAMES,LABELS,COLORS,STYLES):
                rows=read(ROOT/'reduced'/name/'shells.csv');t=number(rows,'time');v=number(rows,m+('_origin'if center=='origin'else''))
                axes[j,0].plot(t,v,label=label,color=color,ls=style);axes[j,1].plot(t,100*(v/v[0]-1),color=color,ls=style)
            axes[j,0].set_ylabel(m+' / M');axes[j,1].set_ylabel('Change from initial [%]')
        axes[0,0].legend();axes[0,0].set_title('Enclosed rest-mass radius');axes[0,1].set_title('Fractional shell change')
        for ax in axes.ravel():ax.set_xlim(0,1000)
        for ax in axes[-1]:ax.set_xlabel('t / M')
        for ax in axes[0]:ax.secondary_xaxis('top',functions=(lambda t:t/P,lambda p:p*P)).set_xlabel('t / surface period')
        fig.suptitle('ST yc=0.819 | proper-profile MC seed1 | '+center.upper()+' coordinate radii');save(fig,'shells_'+center)
    fig,axes=plt.subplots(4,2,figsize=(12,13),sharex=True,layout='constrained')
    for name,label,color,style in zip(NAMES,LABELS,COLORS,STYLES):
        sh=read(ROOT/'reduced'/name/'shells.csv');h=health[name];t=number(h,'time');mask=np.array([r['constraint_usable'].lower()=='true'for r in h])
        axes[0,0].plot(number(sh,'time'),number(sh,'com_mag'),label=label,color=color,ls=style);axes[0,1].plot(t,number(h,'core_lapse'),color=color,ls=style)
        with np.load(ROOT/'reduced'/name/'profiles.npz')as p:axes[1,0].plot(p['time'],p['mass_bins'][:,0]/p['mass_bins'][0,0],color=color,ls=style)
        central=read(ROOT/'reduced'/name/'central_energy.csv');v=number(central,'E_central_slice_r075');axes[1,1].plot(number(central,'time'),v/v[0],color=color,ls=style)
        axes[2,0].plot(t[mask],number(h,'H_rms_r10')[mask],color=color,ls=style);axes[2,1].plot(t[mask],number(h,'M_rms_r10')[mask],color=color,ls=style)
        g=read(ROOT/'reduced'/name/'global_constraints.csv');gt=number(g,'time');gm=np.array([r['constraint_usable'].lower()=='true'for r in g])
        axes[3,0].plot(gt[gm],number(g,'H_rms')[gm],color=color,ls=style)
        checks=load(ROOT/'reduced'/name/'summary.json')['field_checks'];axes[3,1].plot([r['time']for r in checks],[r['E_relative_L2_equilibrium']for r in checks],color=color,ls=style,marker='o',ms=3)
    labels=['COM distance / M','Core lapse, r<0.75M','Central rest-mass density / initial','Central slice E / initial','Core Hamiltonian RMS','Core momentum RMS','Global proper-volume Hamiltonian RMS','E departure from initial equilibrium']
    for ax,label in zip(axes.ravel(),labels):ax.set_ylabel(label);ax.set_xlim(0,1000)
    axes[0,0].legend()
    for ax in axes[-1]:ax.set_xlabel('t / M')
    fig.suptitle('ST MC particle-count comparison | numerical health and redistribution');save(fig,'health_comparison')
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
    for j,m in enumerate(['r10','r50','r90']):
        for name,label,color,style in zip(NAMES,LABELS,COLORS,STYLES):
            rows=[r for r in windows if r['case']==name and r['metric']==m and r['family']=='P'];t=[(r['start']+r['end'])/2 for r in rows]
            for ax,key in zip(axes[j],['half_range','detrended_half_range','mean_shift']):ax.plot(t,[100*r[key]for r in rows],color=color,ls=style,marker='o',label=label)
        axes[j,0].set_ylabel(m+' / equilibrium radius [%]')
    for ax,title in zip(axes[0],['Raw half-range','Detrended half-range','Time-weighted mean shift']):ax.set_title(title)
    axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Complete period-window midpoint / M')
    save(fig,'amplitude_and_drift')
    fig,axes=plt.subplots(4,2,figsize=(13,12),sharex=True,layout='constrained')
    for j,m in enumerate(PRIMARY):
        for k,mode in enumerate(['absolute','change_from_initial']):
            for level,label,color in zip(range(3),['2m minus 1m','4m minus 2m','8m minus 4m'],COLORS):
                rows=[r for r in traces if r['metric']==m and r['mode']==mode and r['level']==level]
                axes[j,k].plot([r['time']for r in rows],[r['difference']for r in rows],label=label,color=color,lw=1)
            axes[j,k].axhline(0,color='#777777',lw=.5);axes[j,k].set_ylabel(m+' residual'+(' / equilibrium radius'if m!='core_lapse'else''))
    axes[0,0].set_title('Absolute-history differences');axes[0,1].set_title('Differences of change from initial');axes[0,0].legend(fontsize=9)
    for ax in axes[-1]:ax.set_xlabel('t / M')
    save(fig,'adjacent_residuals')
    fig,axes=plt.subplots(4,3,figsize=(13,12),layout='constrained')
    for j,m in enumerate(PRIMARY):
        for k,family in enumerate(['full','early_P','late_P']):
            for mode,style in [('absolute','-'),('change_from_initial','--')]:
                rows=[r for r in differences if r['metric']==m and r['mode']==mode and r['family']==family]
                axes[j,k].plot(range(3),[r['RMS']for r in rows],style,marker='o',label=mode.replace('_',' '))
            axes[j,k].set_xticks(range(3),['1m–2m','2m–4m','4m–8m']);axes[j,k].set_ylabel(m+' residual RMS')
            if j==0:axes[j,k].set_title(family)
    axes[0,0].legend(fontsize=8);save(fig,'residual_norms')
if __name__=='__main__':main()
