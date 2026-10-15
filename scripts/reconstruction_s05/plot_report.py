"""Five evidence-backed PNG/PDF figure families and a self-contained campaign report."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, CASES, N, atomic, sha, require_job

METHODS=('mc','stratified','cell','ablation')
LABELS={'mc':'Monte Carlo','stratified':'Stratified','cell':'Cell-aware','ablation':'Random-velocity ablation'}
COLORS={'mc':'#333333','stratified':'#0072B2','cell':'#D55E00','ablation':'#009E73'}
STYLES={'mc':'-','stratified':'--','cell':'-.','ablation':':'}

def read(path):
    with open(path) as f: rows=list(csv.DictReader(f))
    return {key:np.array([float(r[key]) if r[key] not in ('','True','False') else
             (np.nan if not r[key] else float(r[key]=='True')) for r in rows]) for key in rows[0]}

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=220);fig.savefig(out/(name+'.pdf'));plt.close(fig)

def band(ax,x,ys,method,**kwargs):
    a=np.asarray(ys);color=COLORS[method]
    kwargs.setdefault('linestyle',STYLES[method])
    for y in a:ax.plot(x,y,color=color,lw=.55,alpha=.3,linestyle=kwargs['linestyle'])
    ax.plot(x,a.mean(0),color=color,label=LABELS[method],**kwargs)
    ax.fill_between(x,a.min(0),a.max(0),color=color,alpha=.13)

def aligned(rows,key,grid,mask=False):
    """Align actual output times for display; retain invalid-constraint gaps."""
    ys=[]
    for d in rows:
        y=d[key].copy()
        if mask:y[d['constraint_usable']==0]=np.nan
        if np.any(np.diff(d['time'])<0):raise ValueError('Unordered output times')
        if grid[0]<d['time'][0]-.03 or grid[-1]>d['time'][-1]+.03:
            raise ValueError('Time comparison exceeds saved coverage')
        ys.append(np.interp(grid,d['time'],y))
    return ys

def at_slot(d,slot,key):
    use=np.rint(d['time']/100).astype(int)==slot
    if not use.any():raise ValueError('Missing occupancy time slot')
    return d[key][use]

def ratio(value,den):
    # A near-zero Monte Carlo drift has no defensible multiplicative comparison.
    return np.divide(value,den,out=np.full_like(value,np.nan),where=np.abs(den)>1e-12)

def main():
    require_job();ap=argparse.ArgumentParser();ap.add_argument('--interim',action='store_true');args=ap.parse_args()
    data=ROOT/('reduced_interim' if args.interim else 'reduced');out=ROOT/'plots'/('interim100' if args.interim else 'final');out.mkdir(parents=True,exist_ok=True)
    states={};receipt=[]
    expected_endpoint=100 if args.interim else json.loads((ROOT/'evidence/PREFLIGHT_OK').read_text())['endpoint_M']
    for case,rad,method,seed in CASES:
        base=data/case
        if not (base/'summary.json').exists():raise ValueError('Missing validated case '+case)
        d={name:read(base/(name+'.csv')) for name in ('shells','cohorts','cohort_mixing','health','occupancy','field_errors','initial_mass_errors','windows','global_constraints')}
        d['weights']=json.loads((base/'weights.json').read_text())
        d['profiles']=dict(np.load(base/'profiles.npz'));d['summary']=json.loads((base/'summary.json').read_text())
        if d['shells']['nominal_time'][-1]!=expected_endpoint:raise ValueError('Unequal/incomplete common endpoint: '+case)
        if d['summary']['initial_sample_sha256']!=json.loads((ROOT/'evidence/samples'/(case+'.json')).read_text())['sha256']:
            raise ValueError('Reduction refers to a different initial sample: '+case)
        gate=ROOT/'runs'/case/('INTERIM_100_OK' if args.interim else 'RUN_OK')
        if not gate.exists():raise ValueError('Run validation receipt absent: '+case)
        if not args.interim:
            completion=ROOT/'runs'/case/'completion.json'
            d['completion']=json.loads(completion.read_text())
            if d['completion']['endpoint_M']!=expected_endpoint or 'boris_fallbacks' not in d['completion']:
                raise ValueError('Missing complete pusher diagnostic: '+case)
            receipt.append({'path':str(completion.relative_to(ROOT)),'bytes':completion.stat().st_size,'sha256':sha(completion)})
        states[(rad,method,seed)]=d
        for path in sorted(base.iterdir()):
            if path.is_file():receipt.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
    p=dict(np.load(ROOT/'samples/model.npz'))
    endpoint=expected_endpoint
    grid=np.arange(0,endpoint+1,2)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.grid':True,'grid.alpha':.2,'axes.spines.top':False,'axes.spines.right':False})
    # Initial errors: radial CDF, projected E, momentum and stress.
    fig,ax=plt.subplots(2,4,figsize=(16,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for method in METHODS:
            ds=[states[(rad,method,s)] for s in (1,2,3)]
            profiles=[d['profiles'] for d in ds];edges=profiles[0]['edges']
            target=np.cumsum(profiles[0]['target_mass'])/p['M0_over_M']
            if method=='mc':
                eq=np.cumsum(profiles[0]['equilibrium_mass'])/p['M0_over_M']
                ax[row,0].plot(edges[1:],100*(eq-target),'--',color='#888888',label='Equilibrium relative to chosen profile')
            ys=[100*(np.cumsum(d['mass_bins'][0])/p['M0_over_M']-target) for d in profiles]
            band(ax[row,0],edges[1:],ys,method)
            for col,key in ((1,'E_target_error_rms'),(2,'J_coordinate_rms'),(3,'stress_coordinate_anisotropy_rms')):
                ff=[d['field_errors'] for d in ds];use=ff[0]['time']==0
                band(ax[row,col],ff[0]['r'][use],[d[key][d['time']==0] for d in ff],method)
                ax[row,col].set_yscale('log');ax[row,col].set_xlim(0,10)
        for col,title in enumerate(('Cumulative mass error','CIC-projected density error','Coordinate momentum noise','Coordinate stress anisotropy')):
            ax[row,col].set_title(rad+' profile: '+title);ax[row,col].set_xlabel('Coordinate radius / M')
            ax[row,col].set_ylabel('Mass fraction error (%)' if col==0 else 'RMS / $M^{-2}$')
            ax[row,col].set_xlim(0,10)
    ax[0,0].legend(fontsize=8);fig.suptitle('Initial error relative to each chosen radial measure; shading spans three realizations')
    save(fig,out,'f1_initial_errors')
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,baseline in enumerate(('target','equilibrium')):
            for method in METHODS:
                ds=[states[(rad,method,s)]['initial_mass_errors'] for s in (1,2,3)]
                band(ax[row,col],ds[0]['r'],[100*d['CIC_error_'+baseline] for d in ds],method)
                point=np.mean([100*d['point_error_'+baseline] for d in ds],axis=0)
                ax[row,col].plot(ds[0]['r'],point,':',lw=.65,color=COLORS[method])
            ax[row,col].set_title(rad+': '+('chosen radial measure' if baseline=='target' else 'proper-rest-mass equilibrium'))
            ax[row,col].set_xlabel('Origin coordinate radius / M');ax[row,col].set_ylabel('Cumulative rest-mass error (%)');ax[row,col].set_xlim(0,10)
    ax[0,0].legend(fontsize=8);fig.suptitle('Initial cumulative mass: thick CIC-projected comparison, thin dotted particle-centre companion')
    save(fig,out,'f1_initial_mass_projection')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for method in METHODS:
            ff=[states[(rad,method,s)]['field_errors'] for s in (1,2,3)]
            for col,key in enumerate(('E_radial_mean_error','E_target_angular_rms','stress_target_error_rms')):
                use=ff[0]['time']==0
                band(ax[row,col],ff[0]['r'][use],[f[key][f['time']==0] for f in ff],method)
                if col>0:ax[row,col].set_yscale('log')
        for col,title in enumerate(('Density radial-mean error','Density angular error about radial mean','Full stress error versus CIC continuum')):
            ax[row,col].set_title(rad+': '+title);ax[row,col].set_xlabel('Coordinate radius / M');ax[row,col].set_ylabel('$M^{-2}$');ax[row,col].set_xlim(0,10)
    ax[0,0].legend(fontsize=8);save(fig,out,'f1_radial_and_angular_errors')
    # Separate panels at every common checkpoint; select each seed by nominal slot.
    slots=np.arange(0,int(endpoint)//100+1)
    for key,stem,label in (('count_mean','f2_occupancy_profiles','Particle count per cell'),
                           ('neff_mean','f2_effective_count_profiles','CIC effective count per cell')):
        fig,ax=plt.subplots(2,len(slots),figsize=(4*len(slots),7),layout='constrained',squeeze=False)
        for row,rad in enumerate(('legacy','proper')):
            for col,slot in enumerate(slots):
                for method in METHODS:
                    os=[states[(rad,method,s)]['occupancy'] for s in (1,2,3)]
                    band(ax[row,col],at_slot(os[0],slot,'r'),[at_slot(o,slot,key) for o in os],method)
                ax[row,col].set_title(f'{rad}: {100*slot}M');ax[row,col].set_xlabel('Coordinate radius / M');ax[row,col].set_ylabel(label);ax[row,col].set_xlim(0,16)
        ax[0,0].legend(fontsize=8);fig.suptitle(label+' at common saved-state slots; actual timestamps retained in reductions')
        save(fig,out,stem)
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for method in METHODS:
            os=[states[(rad,method,s)]['occupancy'] for s in (1,2,3)]
            empty=[]
            for o in os:
                arr=[]
                for slot in slots:
                    sel=(np.rint(o['time']/100)==slot)&(o['initial_matter_cells']>0)
                    arr.append(np.average(o['initial_matter_empty_fraction'][sel],weights=o['initial_matter_cells'][sel]))
                empty.append(arr)
            band(ax[row,0],100*slots,np.asarray(empty)*100,method,marker='o')
            ws=[states[(rad,method,s)]['weights'] for s in (1,2,3)]
            xx=np.asarray(ws[0]['log10_mass_edges']);xx=(xx[:-1]+xx[1:])/2
            band(ax[row,1],xx,[np.asarray(w['particle_count'])/N for w in ws],method,drawstyle='steps-mid')
        ax[row,0].set_title(rad+': Initially intersecting cells now empty');ax[row,0].set_xlabel('Time / M');ax[row,0].set_ylabel('Empty fraction (%)')
        ax[row,1].set_title(rad+': Fixed particle-weight distribution');ax[row,1].set_xlabel('$\\log_{10}(m_i/M)$');ax[row,1].set_ylabel('Particle fraction per bin');ax[row,1].set_xlim(-14,-4)
    ax[0,0].legend(fontsize=8);save(fig,out,'f2_empty_cells_and_weights')
    # Shell histories, with dotted origin-based companions.
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,q in enumerate((10,50,90)):
            for method in METHODS:
                ds=[states[(rad,method,s)]['shells'] for s in (1,2,3)]
                band(ax[row,col],grid,[100*(v/d[f'r{q}'][0]-1) for v,d in zip(aligned(ds,f'r{q}',grid),ds)],method)
                origin=np.mean([100*(v/d[f'r{q}_origin'][0]-1) for v,d in zip(aligned(ds,f'r{q}_origin',grid),ds)],axis=0)
                ax[row,col].plot(grid,origin,':',color=COLORS[method],lw=.7)
            ax[row,col].set_title(f'{rad}: radius enclosing {q}% rest mass');ax[row,col].set_xlabel('Time / M');ax[row,col].set_ylabel('Change from own start (%)')
    ax[0,0].legend(fontsize=8);fig.suptitle('Shell response: COM solid, origin dotted; seed ranges are not time-sample confidence intervals')
    save(fig,out,'f3_shell_histories')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,q in enumerate((10,50,90)):
            reference=np.interp(q/100,p['proper'],p['riso_over_m'])
            for method in METHODS:
                ds=[states[(rad,method,s)]['shells'] for s in (1,2,3)]
                band(ax[row,col],grid,[100*(v/reference-1) for v in aligned(ds,f'r{q}',grid)],method)
            ax[row,col].axhline(0,color='#888888',ls='--',lw=.7)
            ax[row,col].set_title(f'{rad}: r{q} versus intended equilibrium');ax[row,col].set_xlabel('Time / M');ax[row,col].set_ylabel('Departure from equilibrium (%)')
    ax[0,0].legend(fontsize=8);save(fig,out,'f3_equilibrium_departures')
    fig,ax=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    all_maps=[]
    for rad in ('legacy','proper'):
        for method in METHODS:
            ds=[states[(rad,method,s)]['shells'] for s in (1,2,3)]
            all_maps.append(np.stack([np.mean([100*(v/d[f'r{q}'][0]-1) for v,d in zip(aligned(ds,f'r{q}',grid),ds)],axis=0) for q in range(10,100,10)]))
    vmax=max(float(np.max(np.abs(a))) for a in all_maps)
    for axes,a,(rad,method) in zip(ax.ravel(),all_maps,[(r,m) for r in ('legacy','proper') for m in METHODS]):
        im=axes.pcolormesh(np.r_[grid-1,grid[-1]+1],np.arange(5,100,10),a,shading='flat',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        axes.set_xlim(0,endpoint)
        axes.set_title(rad+': '+LABELS[method]);axes.set_xlabel('Time / M');axes.set_ylabel('Enclosed rest mass (%)');axes.grid(False)
    fig.colorbar(im,ax=ax.ravel().tolist(),label='Mean shell-radius change from own start (%)')
    save(fig,out,'f3_mass_fraction_heatmaps')
    fig,ax=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,method in enumerate(METHODS):
            ds=[states[(rad,method,s)]['cohorts'] for s in (1,2,3)]
            for cohort in (1,3,5,7,9):
                ys=aligned(ds,f'cohort{cohort}',grid)
                mean=np.mean(ys,axis=0);axes=ax[row,col]
                line=axes.plot(grid,mean,label=f'Cohort {cohort}')[0]
                for y in ys:axes.plot(grid,y,color=line.get_color(),lw=.55,alpha=.3)
                axes.fill_between(grid,np.min(ys,axis=0),np.max(ys,axis=0),color=line.get_color(),alpha=.12)
            axes.set_title(rad+': '+LABELS[method]);axes.set_xlabel('Time / M');axes.set_ylabel('Mean COM radius / M')
    ax[0,0].legend(fontsize=8);fig.suptitle('Initial rest-mass cohorts: curves follow the same original particle IDs')
    save(fig,out,'f3_initial_cohorts')
    fig,ax=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,method in enumerate(METHODS):
            arrays=[]
            for seed in (1,2,3):
                d=states[(rad,method,seed)]['cohort_mixing'];sel=np.rint(d['time']/100)==int(endpoint)//100
                a=d['mass_fraction'][sel].reshape(10,10)
                if not np.allclose(a.sum(1),1,atol=1e-10):raise ValueError('Cohort transfer does not conserve mass')
                arrays.append(a)
            im=ax[row,col].imshow(np.mean(arrays,axis=0),origin='lower',extent=[.5,10.5,.5,10.5],vmin=0,vmax=1,cmap='Blues')
            ax[row,col].set_title(rad+': '+LABELS[method]);ax[row,col].set_xlabel('Current rest-mass decile');ax[row,col].set_ylabel('Initial rest-mass cohort');ax[row,col].grid(False)
    fig.colorbar(im,ax=ax.ravel().tolist(),label='Fraction of initial cohort rest mass (seed mean)')
    fig.suptitle(f'Radial mixing at {endpoint:g}M; cohort rows each sum to one')
    save(fig,out,'f3_cohort_mixing')
    # Core constraints, lapse and centre motion.
    fig,ax=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for method in METHODS:
            ds=[states[(rad,method,s)] for s in (1,2,3)];hs=[d['health'] for d in ds]
            for col,key in ((0,'H_rms_r10'),(1,'M_rms_r10'),(2,'core_lapse')):
                ys=aligned(hs,key,grid,mask=col<2)
                band(ax[row,col],grid,ys,method)
                if col<2:
                    key16=key.replace('r10','r16')
                    ys16=aligned(hs,key16,grid,mask=True)
                    ax[row,col].plot(grid,np.mean(ys16,axis=0),'--',color=COLORS[method],lw=.7)
                    ax[row,col].set_yscale('log')
            ss=[d['shells'] for d in ds];band(ax[row,3],grid,aligned(ss,'com_mag',grid),method)
        for col,title in enumerate(('Hamiltonian RMS','Momentum-constraint RMS','Fixed-origin core lapse','COM displacement')):
            ax[row,col].set_title(rad+': '+title);ax[row,col].set_xlabel('Time / M')
            ax[row,col].set_ylabel('$M^{-2}$' if col<2 else ('Lapse' if col==2 else 'M'))
    ax[0,0].legend(fontsize=8);fig.suptitle('Coordinate-volume core constraints: r<10M thick, r<16M thin dashed; tiny-step H/M masked')
    save(fig,out,'f4_constraints_lapse_com')
    fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,key in enumerate(('H_rms','M_rms')):
            for method in METHODS:
                ds=[states[(rad,method,s)]['global_constraints'] for s in (1,2,3)]
                band(ax[row,col],grid,aligned(ds,key,grid,mask=True),method)
            ax[row,col].set_yscale('log');ax[row,col].set_title(rad+': native global '+key);ax[row,col].set_xlabel('Time / M');ax[row,col].set_ylabel('$M^{-2}$')
    ax[0,0].legend(fontsize=8);fig.suptitle('Native global norms retain their proper-volume convention')
    save(fig,out,'f4_native_global_constraints')
    # Window amplitudes versus same-profile MC, and mean shell shifts.
    comparison=[]
    for metric,stem in (('half_range_fraction','f5_amplitude_ratios'),('detrended_half_range_fraction','f5_detrended_amplitude_ratios'),
                        ('mean_shift_fraction','f5_mean_shell_shifts'),('absolute_drift_ratio','f5_drift_ratios')):
        fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for row,rad in enumerate(('legacy','proper')):
            for col,q in enumerate((10,50,90)):
                for method in METHODS:
                    ys=[]
                    for seed in (1,2,3):
                        d=states[(rad,method,seed)]['windows'];sel=d['quantile']==q
                        value=d['mean_shift_fraction' if metric=='absolute_drift_ratio' else metric][sel]
                        starts=d['start'][sel]
                        if metric in ('half_range_fraction','detrended_half_range_fraction'):
                            mc=states[(rad,'mc',seed)]['windows'];den=mc[metric][mc['quantile']==q]
                            value=ratio(value,den)
                            shift=d['mean_shift_fraction'][sel];md=mc['mean_shift_fraction'][mc['quantile']==q]
                            drift=ratio(shift,md)
                            for low,v,dr,sv,mv in zip(starts,value,drift,shift,md):
                                comparison.append({'radial':rad,'method':method,'seed':seed,'quantile':q,'window_start':low,
                                   'amplitude_definition':metric,
                                   'amplitude_ratio_to_mc':float(v) if np.isfinite(v) else None,
                                   'signed_mean_shift_ratio_to_mc':float(dr) if np.isfinite(dr) else None,
                                   'absolute_mean_shift_ratio_to_mc':float(abs(dr)) if np.isfinite(dr) else None,
                                   'mean_shift_fraction':float(sv),'mc_mean_shift_fraction':float(mv)})
                        elif metric=='absolute_drift_ratio':
                            mc=states[(rad,'mc',seed)]['windows'];den=mc['mean_shift_fraction'][mc['quantile']==q]
                            value=np.abs(ratio(value,den))
                        else:value=100*value
                        ys.append(value)
                    band(ax[row,col],starts+50,ys,method,marker='o')
                ax[row,col].set_title(f'{rad}: r{q}');ax[row,col].set_xlabel('Window midpoint / M')
                ax[row,col].set_ylabel('Amplitude / Monte Carlo' if metric=='half_range_fraction' else ('Detrended amplitude / Monte Carlo' if metric=='detrended_half_range_fraction' else ('Absolute mean shift / Monte Carlo' if metric=='absolute_drift_ratio' else 'Mean shift from own start (%)')))
        ax[0,0].legend(fontsize=8);save(fig,out,stem)
    atomic(out/'amplitude_and_drift_ratios.json',comparison)
    # Durable report: describe observations and bounds without promoting physical instability.
    lines=['# Reconstruction Session 5: sampling effects on ST shell response','',
           f'All 24 configurations have validated reductions through approximately {endpoint:.3f}M. Each uses N={N:,}, live Z4c, dx=0.5M and a ±128M box.',
           '', 'The historical and corrected radial measures are analyzed separately. Historical sampling and the equilibrium proper-rest-mass profile differ systematically; this is distinct from finite-particle noise.',
           '', '| Radial measure | Sampler | Initial projected-E relative L2, mean (range) | Final-window shell amplitude / MC, mean (range) | Final-window shell mean-shift RMS / MC, mean (range) |',
           '|---|---|---:|---:|---:|']
    findings=[]
    for rad in ('legacy','proper'):
        for method in METHODS:
            initial=np.array([states[(rad,method,s)]['summary']['initial_error']['E_relative_L2_target'] for s in (1,2,3)])
            ratios=[];drifts=[]
            for seed in (1,2,3):
                w=states[(rad,method,seed)]['windows'];mc=states[(rad,'mc',seed)]['windows'];last=w['end'].max()
                amp=np.sqrt(np.mean(w['half_range_fraction'][w['end']==last]**2));base=np.sqrt(np.mean(mc['half_range_fraction'][mc['end']==last]**2));ratios.append(amp/base)
                drift=np.sqrt(np.mean(w['mean_shift_fraction'][w['end']==last]**2));mb=np.sqrt(np.mean(mc['mean_shift_fraction'][mc['end']==last]**2))
                drifts.append(drift/mb if mb>1e-12 else np.nan)
            ratios=np.array(ratios);drifts=np.array(drifts)
            findings.append({'radial':rad,'method':method,'initial_E_error':initial.tolist(),'final_window_rms_shell_amplitude_ratio':ratios.tolist(),
                'final_window_rms_shell_mean_shift_ratio':[float(v) if np.isfinite(v) else None for v in drifts],
                'all_three_amplitudes_lower_than_mc':bool(np.all(ratios<1)),
                'all_three_drift_magnitudes_lower_than_mc':bool(np.all(drifts<1))})
            drift_text=f'{drifts.mean():.4g} ({drifts.min():.4g}–{drifts.max():.4g})' if np.isfinite(drifts).all() else 'Undefined (near-zero MC drift)'
            lines.append(f'| {rad} | {LABELS[method]} | {initial.mean():.5g} ({initial.min():.5g}–{initial.max():.5g}) | {ratios.mean():.4g} ({ratios.min():.4g}–{ratios.max():.4g}) | {drift_text} |')
    lines+=['','The table uses half the saved peak-to-peak range divided by each run’s own initial radius, and the RMS across r10…r90 in the final complete 100M window. This excursion measure includes secular motion. A companion plot removes each window’s time-weighted least-squares linear trend before measuring the half-range; detrending may also remove part of a slow oscillation. Neither measure estimates the amplitude of an unsampled complete cycle. Window boundaries are exactly 0/100/200/300M; boundary values are interpolated between adjacent saved states. Window means use actual timestamps and trapezoidal time weighting. Displayed ensemble curves are interpolated onto common 2M times; actual timestamps are retained in CSV files. These are short-time comparisons, not tests of permanent stability.',
            '', 'The velocity ablation has exactly the same positions and masses as its stratified partner. Cell-aware weights remain fixed to each original ID. Cohort histories are retained to inspect mixing; they are not a stationarity criterion.',
            '', 'Thin curves show individual realizations; thick curves and shading show their mean and range. Saved time samples are not independent trials. Angular density scatter subtracts the radial mean at each exact Cartesian-cell radius before binning. Momentum/stress noise uses explicitly labelled coordinate covariant-component norms. Signed and absolute drift ratios versus same-profile Monte Carlo are retained alongside amplitude ratios; an MC drift magnitude below 1e-12 makes that ratio undefined, rather than zero.',
            '', 'All runs share the same analytic metric, gauge, pusher, CIC deposition and removal settings. No constraint re-solve, frozen arm, larger domain, reweighting, or imposed perturbation was used. Core constraint norms use coordinate volume; native global norms retain their original proper-volume convention.',
            '', 'The ±128M boundary is within causal reach during this interval and was not varied. Consequently these results cannot establish physical instability, numerical convergence, or long-term Plummer stability. A preferred sampler requires improvements that persist after motion and are consistent across seeds; opposing diagnostics remain a tradeoff.',
            '', 'Figures:']
    if not args.interim:
        # Full-run fallback counts belong to the final interval, not the100M report.
        lines.pop()  # Move the Figures label below the pusher diagnostic table.
        lines+=['The fixed gr_boris pusher can fall back to forward Euler when its fixed-point iteration fails. Counts below are summed from native per-rank/cycle summaries over the full common run. They are diagnostic events, not additional trials. Compare their incidence alongside shell and noise measurements.',
                '', '| Radial measure | Sampler | Fallback events, mean (range) | Cycles with fallback, mean (range) |',
                '|---|---|---:|---:|']
        pusher=[]
        for rad in ('legacy','proper'):
            for method in METHODS:
                rows=[states[(rad,method,seed)]['completion'] for seed in (1,2,3)]
                counts=np.array([r['boris_fallbacks'] for r in rows]);cycles=np.array([r['boris_fallback_cycles'] for r in rows])
                lines.append(f'| {rad} | {LABELS[method]} | {counts.mean():.4g} ({counts.min()}–{counts.max()}) | {cycles.mean():.4g} ({cycles.min()}–{cycles.max()}) |')
                pusher.extend({'case':r['case'],'events':r['boris_fallbacks'],'cycles':r['boris_fallback_cycles'],'endpoint_M':r['endpoint_M']} for r in rows)
        atomic(out/'pusher_fallbacks.json',pusher)
        lines+=['','Figures:']
    for name in ('f1_initial_errors','f1_initial_mass_projection','f1_radial_and_angular_errors','f2_occupancy_profiles','f2_effective_count_profiles','f2_empty_cells_and_weights','f3_shell_histories','f3_equilibrium_departures','f3_mass_fraction_heatmaps','f3_initial_cohorts','f3_cohort_mixing','f4_constraints_lapse_com','f4_native_global_constraints','f5_amplitude_ratios','f5_detrended_amplitude_ratios','f5_mean_shell_shifts','f5_drift_ratios'):
        lines.append(f'- [{name} PNG](../plots/{out.name}/{name}.png) / [PDF](../plots/{out.name}/{name}.pdf)')
    lines+=['','Exact source, submodule, toolchain, inputs, initialization samples, executable hashes, launch commands, checkpoints and raw manifests are retained in the campaign. All raw data remain on AMD under `/work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014`; no deletion or Mac mirror was performed. See the storage receipt for exact retained sizes and allocated job accounting.']
    report=ROOT/'reports'/('REPORT_interim100.md' if args.interim else 'REPORT_session5.md');atomic(report,'\n'.join(lines)+'\n')
    atomic(out/'findings.json',findings);atomic(out/'SOURCE_MANIFEST.json',receipt)
    print('REPORT_AND_FIGURES',report,flush=True)

if __name__=='__main__':main()
