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

def read(path):
    with open(path) as f: rows=list(csv.DictReader(f))
    return {key:np.array([float(r[key]) if r[key] not in ('','True','False') else
             (np.nan if not r[key] else float(r[key]=='True')) for r in rows]) for key in rows[0]}

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=220);fig.savefig(out/(name+'.pdf'));plt.close(fig)

def band(ax,x,ys,method,**kwargs):
    a=np.asarray(ys);color=COLORS[method]
    ax.plot(x,a.mean(0),color=color,label=LABELS[method],**kwargs)
    ax.fill_between(x,a.min(0),a.max(0),color=color,alpha=.13)

def main():
    require_job();ap=argparse.ArgumentParser();ap.add_argument('--interim',action='store_true');args=ap.parse_args()
    data=ROOT/('reduced_interim' if args.interim else 'reduced');out=ROOT/'plots'/('interim100' if args.interim else 'final');out.mkdir(parents=True,exist_ok=True)
    states={};receipt=[]
    for case,rad,method,seed in CASES:
        base=data/case
        if not (base/'summary.json').exists():raise ValueError('Missing validated case '+case)
        d={name:read(base/(name+'.csv')) for name in ('shells','health','occupancy','field_errors','windows','global_constraints')}
        d['profiles']=dict(np.load(base/'profiles.npz'));d['summary']=json.loads((base/'summary.json').read_text())
        states[(rad,method,seed)]=d
        for path in sorted(base.iterdir()):
            if path.is_file():receipt.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
    p=dict(np.load(ROOT/'samples/model.npz'))
    endpoint=min(d['shells']['time'][-1] for d in states.values())
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
    # Occupancy and effective count: both initial and endpoint, with time evolution of empty fraction.
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for method in METHODS:
            os=[states[(rad,method,s)]['occupancy'] for s in (1,2,3)]
            times=np.unique(os[0]['time']);last=times[-1]
            for t,style in ((0,'--'),(last,'-')):
                sel=os[0]['time']==t;x=os[0]['r'][sel]
                for col,key in ((0,'count_mean'),(1,'neff_mean')):
                    ys=[o[key][o['time']==t] for o in os]
                    band(ax[row,col],x,ys,method,linestyle=style)
            empty=[]
            for o in os:
                arr=[]
                for t in np.unique(o['time']):
                    sel=(o['time']==t)&(o['initial_matter_cells']>0)
                    arr.append(np.average(o['initial_matter_empty_fraction'][sel],weights=o['initial_matter_cells'][sel]))
                empty.append(arr)
            band(ax[row,2],times,np.asarray(empty)*100,method,marker='o')
        for col,title in enumerate(('Particle count per cell','CIC effective count per cell','Initially intersecting cells now empty')):
            ax[row,col].set_title(rad+': '+title);ax[row,col].set_xlabel('Time / M' if col==2 else 'Coordinate radius / M')
            ax[row,col].set_ylabel('Empty fraction (%)' if col==2 else 'Count')
            if col<2:ax[row,col].set_xlim(0,16)
    ax[0,2].legend(fontsize=8);fig.suptitle('Sampling after motion: dashed initial, solid endpoint; effective count uses fixed mass × CIC weight')
    save(fig,out,'f2_sampling_after_motion')
    # Shell histories, with dotted origin-based companions.
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,q in enumerate((10,50,90)):
            for method in METHODS:
                ds=[states[(rad,method,s)]['shells'] for s in (1,2,3)];tt=ds[0]['nominal_time']
                if any(not np.array_equal(d['nominal_time'],tt) for d in ds):raise ValueError('Unequal shell cadence')
                band(ax[row,col],tt,[100*(d[f'r{q}']/d[f'r{q}'][0]-1) for d in ds],method)
                origin=np.mean([100*(d[f'r{q}_origin']/d[f'r{q}_origin'][0]-1) for d in ds],axis=0)
                ax[row,col].plot(tt,origin,':',color=COLORS[method],lw=.7)
            ax[row,col].set_title(f'{rad}: radius enclosing {q}% rest mass');ax[row,col].set_xlabel('Time / M');ax[row,col].set_ylabel('Change from own start (%)')
    ax[0,0].legend(fontsize=8);fig.suptitle('Shell response: COM solid, origin dotted; seed ranges are not time-sample confidence intervals')
    save(fig,out,'f3_shell_histories')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,q in enumerate((10,50,90)):
            reference=np.interp(q/100,p['proper'],p['riso_over_m'])
            for method in METHODS:
                ds=[states[(rad,method,s)]['shells'] for s in (1,2,3)]
                band(ax[row,col],ds[0]['nominal_time'],[100*(d[f'r{q}']/reference-1) for d in ds],method)
            ax[row,col].axhline(0,color='#888888',ls='--',lw=.7)
            ax[row,col].set_title(f'{rad}: r{q} versus intended equilibrium');ax[row,col].set_xlabel('Time / M');ax[row,col].set_ylabel('Departure from equilibrium (%)')
    ax[0,0].legend(fontsize=8);save(fig,out,'f3_equilibrium_departures')
    fig,ax=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    all_maps=[]
    for rad in ('legacy','proper'):
        for method in METHODS:
            ds=[states[(rad,method,s)]['shells'] for s in (1,2,3)]
            all_maps.append(np.mean([np.stack([100*(d[f'r{q}']/d[f'r{q}'][0]-1) for q in range(10,100,10)]) for d in ds],axis=0))
    vmax=max(float(np.max(np.abs(a))) for a in all_maps)
    for axes,a,(rad,method) in zip(ax.ravel(),all_maps,[(r,m) for r in ('legacy','proper') for m in METHODS]):
        im=axes.imshow(a,origin='lower',aspect='auto',extent=[0,endpoint,5,95],cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        axes.set_title(rad+': '+LABELS[method]);axes.set_xlabel('Time / M');axes.set_ylabel('Enclosed rest mass (%)');axes.grid(False)
    fig.colorbar(im,ax=ax.ravel().tolist(),label='Mean shell-radius change from own start (%)')
    save(fig,out,'f3_mass_fraction_heatmaps')
    # Core constraints, lapse and centre motion.
    fig,ax=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for method in METHODS:
            ds=[states[(rad,method,s)] for s in (1,2,3)];hs=[d['health'] for d in ds]
            for col,key in ((0,'H_rms_r10'),(1,'M_rms_r10'),(2,'core_lapse')):
                tt=hs[0]['time'];ys=[]
                for h in hs:
                    y=h[key].copy()
                    if col<2:y[h['constraint_usable']==0]=np.nan
                    ys.append(y)
                band(ax[row,col],tt,ys,method)
                if col<2:
                    key16=key.replace('r10','r16')
                    ys16=[]
                    for h in hs:
                        y=h[key16].copy();y[h['constraint_usable']==0]=np.nan;ys16.append(y)
                    ax[row,col].plot(tt,np.mean(ys16,axis=0),'--',color=COLORS[method],lw=.7)
                    ax[row,col].set_yscale('log')
            ss=[d['shells'] for d in ds];band(ax[row,3],ss[0]['nominal_time'],[s['com_mag'] for s in ss],method)
        for col,title in enumerate(('Hamiltonian RMS','Momentum-constraint RMS','Fixed-origin core lapse','COM displacement')):
            ax[row,col].set_title(rad+': '+title);ax[row,col].set_xlabel('Time / M')
            ax[row,col].set_ylabel('$M^{-2}$' if col<2 else ('Lapse' if col==2 else 'M'))
    ax[0,0].legend(fontsize=8);fig.suptitle('Coordinate-volume core constraints: r<10M solid, r<16M dashed; tiny-step H/M masked')
    save(fig,out,'f4_constraints_lapse_com')
    fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,rad in enumerate(('legacy','proper')):
        for col,key in enumerate(('H_rms','M_rms')):
            for method in METHODS:
                ds=[states[(rad,method,s)]['global_constraints'] for s in (1,2,3)]
                ys=[]
                for d in ds:
                    y=d[key].copy();y[d['constraint_usable']==0]=np.nan;ys.append(y)
                band(ax[row,col],ds[0]['time'],ys,method)
            ax[row,col].set_yscale('log');ax[row,col].set_title(rad+': native global '+key);ax[row,col].set_xlabel('Time / M');ax[row,col].set_ylabel('$M^{-2}$')
    ax[0,0].legend(fontsize=8);fig.suptitle('Native global norms retain their proper-volume convention')
    save(fig,out,'f4_native_global_constraints')
    # Window amplitudes versus same-profile MC, and mean shell shifts.
    comparison=[]
    for metric,stem in (('half_range_fraction','f5_amplitude_ratios'),('mean_shift_fraction','f5_mean_shell_shifts')):
        fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for row,rad in enumerate(('legacy','proper')):
            for col,q in enumerate((10,50,90)):
                for method in METHODS:
                    ys=[]
                    for seed in (1,2,3):
                        d=states[(rad,method,seed)]['windows'];sel=d['quantile']==q;value=d[metric][sel]
                        starts=d['start'][sel]
                        if metric=='half_range_fraction':
                            mc=states[(rad,'mc',seed)]['windows'];den=mc[metric][mc['quantile']==q]
                            value=value/den
                            for low,v in zip(starts,value):comparison.append({'radial':rad,'method':method,'seed':seed,'quantile':q,'window_start':low,'amplitude_ratio_to_mc':float(v)})
                        else:value=100*value
                        ys.append(value)
                    band(ax[row,col],starts+50,ys,method,marker='o')
                ax[row,col].set_title(f'{rad}: r{q}');ax[row,col].set_xlabel('Window midpoint / M')
                ax[row,col].set_ylabel('Amplitude / Monte Carlo' if metric=='half_range_fraction' else 'Mean shift from own start (%)')
        ax[0,0].legend(fontsize=8);save(fig,out,stem)
    atomic(out/'amplitude_ratios.json',comparison)
    # Durable report: describe observations and bounds without promoting physical instability.
    lines=['# Reconstruction Session 5: sampling effects on ST shell response','',
           f'All 24 configurations have validated reductions through approximately {endpoint:.3f}M. Each uses N={N:,}, live Z4c, dx=0.5M and a ±128M box.',
           '', 'The historical and corrected radial measures are analyzed separately. Historical sampling and the equilibrium proper-rest-mass profile differ systematically; this is distinct from finite-particle noise.',
           '', '| Radial measure | Sampler | Initial projected-E relative L2, mean (range) | Final-window shell amplitude / MC, mean (range) |',
           '|---|---|---:|---:|']
    findings=[]
    for rad in ('legacy','proper'):
        for method in METHODS:
            initial=np.array([states[(rad,method,s)]['summary']['initial_error']['E_relative_L2_target'] for s in (1,2,3)])
            ratios=[]
            for seed in (1,2,3):
                w=states[(rad,method,seed)]['windows'];mc=states[(rad,'mc',seed)]['windows'];last=w['end'].max()
                amp=np.sqrt(np.mean(w['half_range_fraction'][w['end']==last]**2));base=np.sqrt(np.mean(mc['half_range_fraction'][mc['end']==last]**2));ratios.append(amp/base)
            ratios=np.array(ratios);findings.append({'radial':rad,'method':method,'initial_E_error':initial.tolist(),'final_window_rms_shell_amplitude_ratio':ratios.tolist(),'all_three_amplitudes_lower_than_mc':bool(np.all(ratios<1))})
            lines.append(f'| {rad} | {LABELS[method]} | {initial.mean():.5g} ({initial.min():.5g}–{initial.max():.5g}) | {ratios.mean():.4g} ({ratios.min():.4g}–{ratios.max():.4g}) |')
    lines+=['','Amplitude is half the saved peak-to-peak range divided by each run’s own initial radius. The aggregate uses the RMS across r10…r90 in the final complete 100M window. Window means use actual timestamps and trapezoidal time weighting. These are short-time comparisons, not tests of permanent stability.',
            '', 'The velocity ablation has exactly the same positions and masses as its stratified partner. Cell-aware weights remain fixed to each original ID. Cohort histories are retained to inspect mixing; they are not a stationarity criterion.',
            '', 'Shading shows the range of three independent realizations. It does not treat saved time samples as independent trials. Within-bin angular scatter includes radial gradients over the finite bin width. Momentum/stress noise uses explicitly labelled coordinate covariant-component norms.',
            '', 'All runs share the same analytic metric, gauge, pusher, CIC deposition and removal settings. No constraint re-solve, frozen arm, larger domain, reweighting, or imposed perturbation was used. Core constraint norms use coordinate volume; native global norms retain their original proper-volume convention.',
            '', 'The ±128M boundary is within causal reach during this interval and was not varied. Consequently these results cannot establish physical instability, numerical convergence, or long-term Plummer stability. A preferred sampler requires improvements that persist after motion and are consistent across seeds; opposing diagnostics remain a tradeoff.',
            '', 'Figures:']
    for name in ('f1_initial_errors','f2_sampling_after_motion','f3_shell_histories','f3_equilibrium_departures','f3_mass_fraction_heatmaps','f4_constraints_lapse_com','f4_native_global_constraints','f5_amplitude_ratios','f5_mean_shell_shifts'):
        lines.append(f'- [{name} PNG](../plots/{out.name}/{name}.png) / [PDF](../plots/{out.name}/{name}.pdf)')
    lines+=['','Exact source, submodule, toolchain, inputs, initialization samples, executable hashes, launch commands, checkpoints and raw manifests are retained in the campaign. All raw data remain on AMD under `/work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014`; no deletion or Mac mirror was performed. See the storage receipt for exact retained sizes and allocated job accounting.']
    report=ROOT/'reports'/('REPORT_interim100.md' if args.interim else 'REPORT_session5.md');atomic(report,'\n'.join(lines)+'\n')
    atomic(out/'findings.json',findings);atomic(out/'SOURCE_MANIFEST.json',receipt)
    print('REPORT_AND_FIGURES',report,flush=True)

if __name__=='__main__':main()
