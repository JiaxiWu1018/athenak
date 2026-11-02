"""Source-bound interim figures/report for completed legacy controls only."""
import json
import os
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from plot import plt, arrays
from analyze import read_csv
from campaign import ROOT, resource_snapshot
from state_io import require_job, sha, write_json


def main():
    require_job()
    data=ROOT/'analysis/job_495334/reduced'
    audit_path=ROOT/'analysis/control_audit_495421/AUDIT.json'
    audit=json.loads(audit_path.read_text())
    out=ROOT/'analysis'/('control_report_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    rows=read_csv(data/'temporal_order.csv')
    assert len(rows)==234 and {r['scheme'] for r in rows}=={'legacy'}
    sources=[audit_path,data/'temporal_order.csv',data/'TEMPORAL_SOURCES.json',data/'ANALYSIS.json']
    colors=['#0072B2','#D55E00','#009E73']
    times=[20,50,100]
    particle_fields=['position','covariant_momentum','energy']
    particle_labels=['Position','Covariant momentum','Energy']
    summary=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for field,label,color,marker in zip(particle_fields,particle_labels,colors,['o','s','^']):
        rr=[next(r for r in rows if r['kind']=='particles' and r['field']==field and int(r['time'])==t) for t in times]
        assert all(r['above_roundoff_floor']=='True' for r in rr)
        axes[0].plot(times,[float(r['observed_p']) for r in rr],marker=marker,color=color,label=label)
    for t in times:
        rr=[r for r in rows if r['kind']=='grid' and r['region']=='r10' and int(r['time'])==t and r['observed_p']]
        p=np.array([float(r['observed_p']) for r in rr]);assert len(p)==22
        axes[1].scatter(t+np.linspace(-2,2,len(p)),p,s=19,color='#0072B2',alpha=.65)
        axes[1].plot([t-3,t+3],[np.median(p)]*2,color='black',lw=2)
        summary.append(dict(time=t,core_min=float(p.min()),core_median=float(np.median(p)),core_max=float(p.max()),
            **{field:float(next(r['observed_p'] for r in rows if r['kind']=='particles' and r['field']==field and int(r['time'])==t)) for field in particle_fields}))
    axes[0].set_title('Particle state: rest-mass weighted RMS')
    axes[1].set_title('22 nonzero grid fields within r < 10M')
    axes[0].legend(frameon=False,loc='center right')
    axes[1].legend(handles=[Line2D([],[],marker='o',ls='',color='#0072B2',label='Individual field'),Line2D([],[],color='black',label='Median across fields')],frameon=False,loc='upper right',bbox_to_anchor=(1,.89))
    for ax in axes:
        ax.axhline(1,color='.4',ls=':',lw=1);ax.axhline(2,color='.4',ls='--',lw=1)
        ax.set_ylim(.8,2.15);ax.set_xlim(10,110);ax.set_xticks(times);ax.set_xlabel('Matched checkpoint time [M]')
        ax.set_ylabel(r'$p=\log_2(\|Q_h-Q_{h/2}\|/\|Q_{h/2}-Q_{h/4}\|)$');ax.grid(alpha=.15)
    fig.suptitle('Legacy RK4 control: temporal self-convergence at fixed 1M particles')
    fig.text(.5,.01,'Nominal h = 0.2, 0.1, 0.05M; restart step-growth transients and legacy Euler fallbacks are retained.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    save(fig,out,'legacy_temporal_order')
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for col,(reg,label,name) in enumerate([('r10','r < 10M','health.csv'),('r16','r < 16M','health.csv'),('global','Full domain','global_constraints.csv')]):
        for row,metric in enumerate(('H','M')):
            key=f'{metric}_rms'+('' if reg=='global' else '_'+reg);ax=axes[row,col]
            for h,color,ls in zip((20,10,5),colors,('-','--','-.')):
                source=data/f'legacy_1m_h{h:03d}'/name;sources.append(source)
                t,v=arrays(source,key,100);assert t[-1]>=100-1e-8
                ax.semilogy(t,v,color=color,ls=ls,lw=1.4)
            old=ROOT/'reference/session7/reduced/mc_1m_s1'/name;sources.append(old)
            t,v=arrays(old,key,100);ax.semilogy(t,v,color='.25',ls=':',lw=1.1)
            ax.set_title(('Hamiltonian' if metric=='H' else 'Momentum')+' · '+label)
            ax.set_ylabel(r'Constraint RMS [$M^{-2}$]');ax.set_xlim(0,100);ax.grid(alpha=.15)
            if row==1:ax.set_xlabel('Time [M]')
    handles=[Line2D([],[],color=c,ls=l,label=f'RK4 h = {h}M, synchronized') for h,c,l in zip(('.2','.1','.05'),colors,('-','--','-.'))]
    handles.append(Line2D([],[],color='.25',ls=':',label='Archived 1M RK4, lagged source'))
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.955),fontsize=9)
    fig.suptitle('ST yc = 0.819 · 1M particles · legacy constraint controls to 100M',y=.99)
    fig.text(.5,.012,'Core: coordinate-volume RMS; full domain: proper-volume RMS. Panels use independent y scales. Archived diagnostic timing differs.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.89));save(fig,out,'legacy_constraint_controls')
    totals={}
    for rec in audit['records']:
        for count in rec['counts'].values():assert count['summaries_consistent']
        totals[rec['case']]=totals.get(rec['case'],0)+rec['counts']['euler_fallback']['parsed_events']
    lines=['# Session10 interim legacy-control results','',
        'The legacy particle trajectories show temporal self-convergence close to first order (p=1.03–1.08) for the exact 1M seed1 realization. This is an empirical result at nominal h=.2/.1/.05M, not a predictor result or a proof of asymptotic order. The four predictor 500M runs remain outstanding.','',
        '| Time [M] | Position p | Momentum p | Energy p | Core grid p: min / median / max |','|---|---|---|---|---|']
    for s in summary:lines.append(f"| {s['time']} | {s['position']:.4f} | {s['covariant_momentum']:.4f} | {s['energy']:.4f} | {s['core_min']:.3f} / {s['core_median']:.3f} / {s['core_max']:.3f} |")
    lines+=['','![Temporal order](legacy_temporal_order.png)','',
        'Order is computed from matched double-precision checkpoint differences at 20/50/100M. Grid norms weight active-cell coordinate volume; particle vector norms weight rest mass and match original IDs. The core spread describes 22 different nonzero fields, not statistical uncertainty; three identically zero B fields have undefined order and are excluded. All plotted differences passed the roundoff-floor check. Global and r<16M per-field values remain in the source CSV.','',
        'Restart files can save a tiny final alignment timestep. Native timestep growth is capped at twice the preceding step, producing short startup sequences after restart. These are nominal maximum-step refinements, not exactly scaled uniform time grids.','',
        '| Control | Recorded forward-Euler fallback events |','|---|---|']
    for case,count in totals.items():lines.append(f'| {case} | {count} |')
    lines+=['','Counts use native per-cycle summaries, cross-checked against rank cumulative counters. They count events, not unique particles. All completed RK4 controls retained all 1M particles and original masses with finite checkpoints and histories; no rejected updates were recorded. Fallback counts alone do not establish what causes the measured order.','',
        '![Constraints](legacy_constraint_controls.png)','',
        'The three new legacy-control curves use synchronized metric/particle endpoint constraints. The archived 1M curve uses lagged particle sources. Therefore their offsets include diagnostic timing changes. Constraint RMS evolution and temporal self-convergence are different measurements; these figures do not measure particle-number scaling or spatial convergence.','',
        'Frozen-source RK3 at h=.2M failed at 18.4M after nonfinite outer lapse and global constraint data appeared in the 18M history. Its initial particle sample matches exactly. Its recorded fallback count may omit final aborted MPI output. It has no valid 20M endpoint, and is excluded from successful-control plots. The cause of instability remains unresolved.','',
        'The predictor pilots still fail the strict implicit solve at a stencil boundary. A choice between fixing the predicted-midpoint stencil and redesigning continuous-force interpolation remains pending. No predictor improvement or order claim is supported yet.','',
        f'Sources: `{data}`, `{audit_path}`; analysis job 495334, audit 495421. Figure generation job'+os.environ['SLURM_JOB_ID']+'. See SOURCES.json for hashes. All raw data remain on AMD; no deletion authorized.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'SUMMARY.json',dict(orders=summary,fallback_events=totals,resources=resource_snapshot()))
    write_json(out/'SOURCES.json',dict(script_sha256=sha(__file__),files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(sources))],outputs=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir())]))
    print(out)


def save(fig,out,name):
    for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),bbox_inches='tight',facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
