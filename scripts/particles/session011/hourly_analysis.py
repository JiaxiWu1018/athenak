"""Additional hourly scientific comparisons from existing reduced production data."""
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def source_record(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def finish(fig,axes,out,name):
    for ax in np.asarray(axes).ravel():
        ax.grid(alpha=.18)
        lo,hi=ax.get_xlim()
        ticks=[float(t) for t in MaxNLocator(nbins=6).tick_values(lo,hi) if lo<t<hi]
        ax.set_xticks([lo]+ticks+[hi])
        ax.set_xticklabels([f'{t:.2f}'.rstrip('0').rstrip('.') for t in [lo]+ticks+[hi]])
    fig.tight_layout()
    fig.savefig(out/(name+'.png'),dpi=160)
    fig.savefig(out/(name+'.pdf'))
    plt.close(fig)


def analyze(out):
    out=Path(out)
    metrics=json.loads((out/'metrics.json').read_text())
    files=['metric_shells.csv','envelope_wake_moments.csv','component_ledgers.csv',
           'envelope_support_proxy.csv','orbit.csv','metrics.json','accepted_horizons.csv']
    provenance=[source_record(out/name) for name in files]
    accepted=pd.read_csv(out/'accepted_horizons.csv')
    individuals=accepted[accepted.horizon.isin([0,1])]
    paired=individuals.groupby('state_cycle').horizon.nunique()
    cycles=paired[paired==2].index
    pair=float(individuals[individuals.state_cycle.isin(cycles)].state_time.min()) if len(cycles) else None
    grid=pd.read_csv(out/'metric_shells.csv')
    grouped=grid.groupby(['cycle','time','mask']).sum(numeric_only=True,min_count=1).reset_index()
    grouped['H_RMS']=np.sqrt(grouped.H2_volume/grouped.proper_volume)
    grouped['M_RMS']=np.sqrt(grouped.M2_volume/grouped.proper_volume)
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True)
    last={}
    for mask,label,color,style in ((0,'Whole grid','#606060','-'),(1,'Accepted-AH exterior; changing domain','#246BB2','--')):
        rows=grouped[grouped['mask']==mask].sort_values('cycle')
        for ax,name in zip(axs,('H_RMS','M_RMS')):
            ax.semilogy(rows.time,rows[name],style,color=color,label=label)
        if len(rows):last[str(mask)]={k:finite(rows.iloc[-1][k]) for k in ('time','H_RMS','M_RMS')}
    for ax,name in zip(axs,('H RMS','M RMS')):
        if pair is not None:ax.axvline(pair,color='#888888',ls=':',lw=1,label='First individual pair' if ax is axs[0] else None)
        ax.set_ylabel(name);ax.set_xlim(float(grouped.time.min()),float(grouped.time.max()))
    axs[0].legend(fontsize=8);axs[0].set_title('Constraint norms before and after horizon exclusion')
    axs[1].set_xlabel('State time / Mref')
    finish(fig,axs,out,'constraint_domain_comparison')

    wake=pd.read_csv(out/'envelope_wake_moments.csv').sort_values('cycle')
    ledger=pd.read_csv(out/'component_ledgers.csv')
    exterior=ledger[(ledger.component==0)&(ledger['mask']==1)].sort_values('cycle')
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axs[0].plot(wake.time,wake.wake_quadrupole_real,color='#8E5AB5',label='Real')
    axs[0].plot(wake.time,wake.wake_quadrupole_imag,color='#8E5AB5',ls='--',label='Imaginary')
    axs[0].set_ylabel('Orbital-frame quadrupole / Mref^3');axs[0].legend()
    axs[0].set_title('Exterior envelope moments in the measured orbital frame')
    axs[1].plot(exterior.time,exterior.Jz,color='#8E5AB5')
    axs[1].set_ylabel('Exterior-envelope Jz(origin) / Mref^2');axs[1].set_xlabel('State time / Mref')
    for ax in axs:ax.set_xlim(float(wake.time.min()),float(wake.time.max()))
    finish(fig,axs,out,'envelope_wake_and_Jz')

    support=pd.read_csv(out/'envelope_support_proxy.csv')
    orbit=pd.read_csv(out/'orbit.csv')
    support=support.merge(orbit[['cycle','tangential_speed_resolved']],on='cycle',how='left',validate='one_to_one')
    resolved=support.tangential_speed_resolved.astype(str).str.lower().eq('true')
    observed=support[resolved & (support.measured_omega2>0)]
    fig,ax=plt.subplots(figsize=(9,4))
    for column,label,color,style in (
        ('evolving_env_omega2_proxy','Evolving envelope proxy','#8E5AB5','-'),
        ('frozen_env_omega2_proxy','Frozen envelope proxy','#606060','--')):
        values=support[column].where(support[column]>0)
        ax.semilogy(support.time,values,style,color=color,label=label)
    ax.semilogy(observed.time,observed.measured_omega2,'.',color='#202020',ms=2,label='Measured Omega^2; resolved samples')
    ax.set(xlabel='State time / Mref',ylabel='Frequency squared / Mref^-2',
           title='Spherical Newtonian envelope-support proxy')
    ax.set_xlim(float(support.time.min()),float(support.time.max()));ax.legend(fontsize=8)
    finish(fig,[ax],out,'envelope_support_comparison')
    summary=dict(diagnostic_time=metrics['endpoint'],native_audited_time=metrics['endpoint_checkpoint']['time'],
        first_pair_time=pair,constraint_domains=last,
        latest_wake={k:finite(wake.iloc[-1][k]) for k in ('time','wake_quadrupole_real','wake_quadrupole_imag')} if len(wake) else None,
        support_samples=len(support),resolved_measured_support_samples=len(observed),
        source_data=provenance,production_modified=False,
        limits=['Constraint exterior integrates over a changing accepted-AH domain; unknown coverage stays NaN.',
                'Envelope quadrupole and Jz use coordinates and a changing exterior mask; no causal torque fraction is inferred.',
                'Menv(<d/2)/(d/2)^3 is a spherical Newtonian rest-mass proxy, not a relativistic force decomposition.',
                'Unresolved angular-velocity measurements are omitted; resolved derivatives can still contain mesh-quantization noise.',
                'One trajectory supplies no evolution-grid, timestep, seed or physics-control comparison.'])
    (out/'hourly_analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    def show(x):return 'unknown' if x is None else f'{x:.6g}'
    all_h=last.get('0',{}).get('H_RMS');ex_h=last.get('1',{}).get('H_RMS')
    text=f'''# Additional hourly scientific comparisons

Data through t={metrics['endpoint']:.12g}; last audited native state t={metrics['endpoint_checkpoint']['time']:.12g}.
The latest whole-grid H RMS is {show(all_h)} and the accepted-AH exterior H RMS is {show(ex_h)}.
These use different integration domains. A drop when horizons qualify does not
establish whole-grid numerical improvement. Whole-grid behavior remains
visible alongside exterior measurements; missing AH coverage remains unknown.

![Whole-grid and exterior constraint norms](constraint_domain_comparison.png)

The exterior-envelope quadrupole is rotated by the measured orbital phase.
Its real and imaginary parts and the origin-based envelope Jz are coordinate
diagnostics. AH surface motion changes exterior membership, so these curves alone
are not a flux-corrected torque or a causal attribution of orbital evolution.
Unknown phase and horizon intervals remain gaps.

![Envelope wake and angular momentum](envelope_wake_and_Jz.png)

The support comparison uses Menv(<d/2)/(d/2)^3 from spherical coordinate shells,
with evolving and initial envelope rest-mass profiles. It is a Newtonian proxy:
non-spherical and relativistic forces are not included. Measured Omega^2 is shown
only for resolved tangential samples ({len(observed)} of {len(support)} support rows).
Mesh-quantized tracker positions can still make those derivatives noisy.

![Envelope support proxy](envelope_support_comparison.png)

Exact source paths, hashes, observations and qualifications are in
`hourly_analysis.json`. This is a partial analysis of one continuing trajectory;
it does not establish convergence or a unique share for any of the five hypotheses.
'''
    (out/'ANALYSIS.md').write_text(text)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    analyze(p.parse_args().out)
