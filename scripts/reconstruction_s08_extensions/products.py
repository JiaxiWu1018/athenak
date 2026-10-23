"""Source-bound extension figures, numeric checks and review assembly; Slurm only."""
import csv,shutil,re
import numpy as np
from common import *
from extension_resources import EXT
from reduce import csv_rows,write_csv
from publication_style import style,save,array,shell_limits
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
BASE=Path(os.environ['ST8X_BASE'])

def merge(rows):
    out={}
    for row in rows:out[round(row['time'],8)]=row
    return [out[t] for t in sorted(out)]

def assemble():
    (ROOT/'reduced').mkdir(parents=True,exist_ok=True)
    for name in ['reports','plots','movies','inputs','evidence/samples','runs']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    modes=['long','inward0020']
    first=load(EXT/'inward0020/runs/yc0500/state.json')
    assert first['complete']
    if first['latest']['ah_onset']<0 and first['latest']['time']>=300-1e-8:modes.append('inward0050')
    data=[];sources=[]
    for mode in modes:
        source=EXT/mode;state=load(source/'runs/yc0500/state.json');assert state['complete']
        red=source/'reduced/yc0500';assert load(red/'REDUCTION_OK.json')['population_ids']
        summary=load(red/'summary.json');assert summary['complete'] and summary['endpoint']==state['latest']['time']
        target=500 if mode=='long' else 300;onset=state['latest']['ah_onset'];expected=min(target,onset+20) if onset>=0 else target
        tolerance=max(1e-7,float(state['latest']['dt'])+1e-8) if onset>=0 and expected<target else 1e-7
        assert expected-1e-7<=state['latest']['time']<=min(target+1e-7,expected+tolerance),(mode,expected,state['latest']['time'])
        ident='yc0500_'+mode;dest=ROOT/'reduced'/ident;dest.mkdir(exist_ok=True)
        title='Unperturbed, extended' if mode=='long' else ('Inward $U=-0.02$' if mode=='inward0020' else 'Inward $U=-0.05$')
        case=dict(id=ident,yc=.5,unstable=True,title=title,mode=mode)
        table={}
        for name in ['shells','mass','history','ah','cohort_index']:
            if name=='cohort_index':continue
            values=csv_rows(red/(name+'.csv'))
            if mode=='long':values=csv_rows(BASE/'reduced/yc0500'/(name+'.csv'))+values
            table[name]=merge(values)
            # cohort file paths are strings; use the standard CSV reader instead.
            if name!='cohort_index':write_csv(dest/(name+'.csv'),table[name])
        # Re-read the cohort index with its absolute path column preserved.
        indices=[]
        paths=[BASE/'reduced/yc0500/cohort_index.csv',red/'cohort_index.csv'] if mode=='long' else [red/'cohort_index.csv']
        for path in paths:
            with path.open() as f:
                for row in csv.DictReader(f):
                    row['time']=float(row['time']);row['cycle']=int(row['cycle']);row['alive']=int(row['alive']);assert Path(row['file']).exists();indices.append(row)
        table['cohort_index']=merge(indices);write_csv(dest/'cohort_index.csv',table['cohort_index'])
        summary.update(case=case);atomic(dest/'summary.json',summary)
        shutil.copy2(red/'REDUCTION_OK.json',dest/'REDUCTION_OK.json')
        profile=load(BASE/'evidence/samples/yc0500.json');atomic(ROOT/'evidence/samples'/(ident+'.json'),profile)
        work=ROOT/'runs'/ident;work.mkdir(exist_ok=True)
        shape=source/'runs/yc0500/yc0500.horizon_shape_0.txt'
        if shape.exists() and not (work/(ident+'.horizon_shape_0.txt')).exists():(work/(ident+'.horizon_shape_0.txt')).symlink_to(shape)
        shutil.copy2(source/'inputs/yc0500.athinput',ROOT/'inputs'/(ident+'.athinput'))
        table.update(summary=summary,case=case,state=state,source=source);data.append(table)
        sources.append(dict(mode=mode,raw_root=str(source/'runs/yc0500'),state=receipt(source/'runs/yc0500/state.json'),reduction=receipt(red/'REDUCTION_OK.json'),checkpoint=state['latest'],baseline_prefix=str(BASE/'runs/yc0500') if mode=='long' else None))
    atomic(ROOT/'evidence/SOURCE_MAP.json',dict(job=job(),utc=now(),cases=sources,baseline_prefix_reused=True,raw_moved=False))
    return data

def numeric(data):
    from reduce_run import read_pvtk
    rows=[];spotchecks=[];coverage=[]
    for d in data:
        s=d['summary'];m=d['mass'];h=d['history'];shell=d['shells'];mode=d['case']['mode']
        assert shell and h and m and shell[0]['time']==0 and h[0]['time']==0
        assert abs(m[-1]['time']-s['endpoint'])<1e-7 and abs(h[-1]['time']-s['endpoint'])<1e-7
        assert all(abs(r['m_alive']/r['m_initial']+sum(r['n_'+k] for k in ['exit','sphere','lapse','horizon'])/2000000-1-r['closure'])<1e-12 for r in m)
        assert all(abs(r['closure'])<1e-10 for r in m)
        radii=np.array([[r[f'r{q}'] for q in range(10,100,10)] for r in shell]);assert np.isfinite(radii).all() and (radii>0).all() and (np.diff(radii,axis=1)>=0).all()
        if s['first_removal'] is not None:assert max(r['time'] for r in shell)<s['first_removal']
        good=[r for r in d['ah'] if r['accepted']==1]
        for r in good:
            assert r['streak']>=3 and r['hrel']<=.05 and r['missing']==0 and r['rmin']>=4*r['dx']
            assert r['mirr']>0 and r['mass']>0 and np.isclose(r['mirr'],np.sqrt(r['area']/(16*np.pi)),rtol=1e-12)
        snaps=load(d['source']/'reduced/yc0500/snapshots.json')
        if mode=='long':snaps=load(BASE/'reduced/yc0500/snapshots.json')+snaps
        max_dt=max(float(r[k]) for r in h for k in ['dt','dt_completed'])
        streams={}
        for name,values,nominal in [('history',h,.25),('mass',m,.25),
                                   ('cohort',d['cohort_index'],.5),('full_particles',snaps,1.),
                                   ('AH_searches',d['ah'],.5)]:
            times=np.unique(np.round([r['time'] for r in values],8))
            assert len(times)>1 and np.isfinite(times).all(),(mode,name)
            assert times[0]>=-1e-7 and times[-1]<=s['endpoint']+1e-7,(mode,name)
            gap=float(np.diff(times).max())
            assert gap<=nominal+max_dt+2e-7,(mode,name,'missing outputs',gap,nominal,max_dt)
            if name=='AH_searches':
                assert times[0]<=nominal+max_dt+1e-7 and s['endpoint']-times[-1]<=nominal+max_dt+1e-7
            else:
                assert abs(times[0])<1e-7 and abs(times[-1]-s['endpoint'])<1e-7,(mode,name,'endpoint coverage')
            streams[name]=dict(unique_times=len(times),first_M=float(times[0]),last_M=float(times[-1]),
                               maximum_gap_M=gap,nominal_cadence_M=nominal,maximum_completed_or_reported_step_M=max_dt)
        global_rows=[];global_sources=[]
        for source in ([BASE,d['source']] if mode=='long' else [d['source']]):
            path=source/'runs/yc0500/yc0500.z4c.user.hst'
            values=np.atleast_2d(np.loadtxt(path));assert values.shape[1]==11 and np.isfinite(values).all()
            assert (values[:,10]>0).all() and (values[:,3:5]>=0).all()
            global_rows.extend(dict(time=r[0],H_integral_squared=r[3],M_integral_squared=r[4],
                                    proper_unexcised_volume=r[10],H_RMS=float(np.sqrt(r[3]/r[10])),
                                    M_RMS=float(np.sqrt(r[4]/r[10]))) for r in values)
            global_sources.append(receipt(path))
        global_rows=merge(global_rows)
        assert abs(global_rows[0]['time'])<1e-7 and abs(global_rows[-1]['time']-s['endpoint'])<1e-7
        write_csv(ROOT/'reduced'/d['case']['id']/'global_constraints.csv',global_rows)
        coverage.append(dict(mode=mode,streams=streams,global_constraint_sources=global_sources,
                             global_constraint_definition='sqrt(integral constraint squared dV / integral dV), proper volume over native chi-unexcised cells. Large exterior volume dilutes whole-box RMS; not a cluster-local error bound.'))
        eligible=[v for v in snaps if s['first_removal'] is None or v['time']<s['first_removal']]
        for snap in [eligible[0],eligible[len(eligible)//2],eligible[-1]]:
            t,cy,n,f=read_pvtk(snap['file']);assert n==2000000
            pos=np.column_stack([f[k] for k in 'xyz']);center=pos.mean(axis=0);rr=np.sort(np.linalg.norm(pos-center,axis=1))
            row=min(shell,key=lambda v:abs(v['time']-t));assert abs(row['time']-t)<1e-7
            for q in range(10,100,10):
                k=int(np.ceil(n*q/100))-1
                assert rr[max(0,k-1)]-1e-10<=row[f'r{q}']<=rr[min(n-1,k+1)]+1e-10,(mode,t,q)
            spotchecks.append(dict(mode=mode,time=t,source=receipt(Path(snap['file'])),equal_weight_quantiles='all9 agree within one particle rank; COM independently computed'))
        tail=[r for r in good if r['time']>=s['endpoint']-5]
        record=dict(mode=mode,endpoint_M=s['endpoint'],AH_accepted_records=len(good),first_AH_M=good[0]['time'] if good else None,first_removal_M=s['first_removal'],r50_initial=shell[0]['r50'],r50_last_pre_removal=shell[-1]['r50'],last_shell_time_M=shell[-1]['time'],core_lapse_final=h[-1]['core_lapse'],core_lapse_min=min(r['core_lapse'] for r in h),max_rest_mass_closure=max(abs(r['closure']) for r in m),final_particles=d['state']['latest']['particle_count'],final_Mirr_median=np.median([r['mirr'] for r in tail]) if tail else None,final_MCH_median=np.median([r['mass'] for r in tail]) if tail else None)
        healthy=[r for r in h if s['first_removal'] is None or r['time']<s['first_removal']]
        record.update(max_H_rms_r10_pre_removal=max(r['H_rms_r10'] for r in healthy),max_M_rms_r10_pre_removal=max(r['M_rms_r10'] for r in healthy),max_sampled_blocks=max(r['nblocks'] for r in m),minimum_sampled_dx_M=min(r['dx_min'] for r in m),minimum_completed_dt_M=min(r['dt_completed'] for r in h if r['dt_completed']>0))
        global_healthy=[r for r in global_rows if s['first_removal'] is None or r['time']<s['first_removal']]
        record.update(max_H_rms_global_pre_removal=max(r['H_RMS'] for r in global_healthy),
                      max_M_rms_global_pre_removal=max(r['M_RMS'] for r in global_healthy),
                      final_H_rms_global=global_rows[-1]['H_RMS'],final_M_rms_global=global_rows[-1]['M_RMS'])
        states=[(d['source'],d['state'])]
        if mode=='long':states.insert(0,(BASE,load(BASE/'runs/yc0500/state.json')))
        logs=set()
        for source,st in states:
            for segment in st['segments']:
                found=list((source/'runs/yc0500').glob(str(segment['job'])+'*.log'))
                assert found,('Missing accepted native log',segment['job'])
                logs.update(found)
        fallbacks=0
        for path in logs:
            text=path.read_text(errors='replace')
            assert '### FATAL' not in text and not re.search('search_fail=[1-9]',text),str(path)
            fallbacks+=sum(int(v) for v in re.findall(r'cycle\s+\d+:\s+(\d+) of \d+ particles fell back',text))
        record['fallback_particle_steps']=fallbacks
        if mode!='long':
            initial=load(d['source']/'runs/yc0500/INITIAL_OK.json')
            assert initial['actual_full_particle_output_checked'] and initial['actual_fixed_ID_cohort_checked']

            record.update(initial_relative_mean_abs_L_change=initial['relative_mean_abs_L_change'],initial_particle_Killing_energy_change=initial['particle_Killing_energy_perturbed']-initial['particle_Killing_energy_initial'])
            reference=load(EXT/'evidence/INDEPENDENT_CONSTRAINT_REFERENCE_OK.json')
            U=-.02 if mode=='inward0020' else -.05
            record['initial_continuum_delta_ADM_mass']=next(v['independent_delta_ADM_mass'] for v in reference['results'] if v['U']==U)
        if mode=='long':
            late=[r for r in shell if r['time']>=max(300,shell[-1]['time']-50)]
            record['late_r50_slope']=float(np.polyfit([r['time'] for r in late],[r['r50'] for r in late],1)[0]) if len(late)>1 else None
            peak=max(shell,key=lambda r:r['r50']);record['r50_max']=peak['r50'];record['r50_max_time_M']=peak['time']
        rows.append(record)
    atomic(ROOT/'evidence/INDEPENDENT_NUMERIC_QA.json',dict(job=job(),utc=now(),cases=rows,full_population_audits_reused=True,shell_spotchecks=spotchecks,scope='All reduced ledgers/AH rows/shell ordering; independent full-population shell calculations at3 times per case. Not temporal/mesh convergence.'))
    atomic(ROOT/'evidence/COVERAGE_AND_GLOBAL_CONSTRAINTS.json',dict(job=job(),utc=now(),cases=coverage,
           cadence_rule='Every stream spans the actual endpoint (AH searches within their cadence); maximum gap cannot exceed nominal cadence plus one maximum completed/reported timestep. Duplicate source times are collapsed.'))
    # Dict keys differ for the long-run extra diagnostics; make explicit empty columns.
    keys=list(dict.fromkeys(k for row in rows for k in row));write_csv(ROOT/'reports/extension_summary.csv',[{k:r.get(k) for k in keys} for r in rows])
    return rows

def events(axes,s):
    for ax in axes:
        if s['first_removal'] is not None:ax.axvline(s['first_removal'],ls=':',c='.45',lw=.9)
        if s['ah_first'] is not None:ax.axvline(s['ah_first'],ls='--',c='.15',lw=.9)
        ax.grid(alpha=.15,lw=.5)

def figures(data):
    style();lim=shell_limits(*(d['shells'] for d in data));cm=plt.get_cmap('cividis');norm=Normalize(.1,.9)
    fig,ax=plt.subplots(2,len(data),figsize=(5*len(data),6),squeeze=False,layout='constrained',sharey='row')
    for j,d in enumerate(data):
        s=d['summary'];t=array(d['shells'],'time')
        for q in range(10,100,10):ax[0,j].plot(t,array(d['shells'],f'r{q}'),c=cm(norm(q/100)),lw=1)
        ax[0,j].set(yscale='log',ylim=lim,title=d['case']['title']);ax[1,j].plot(array(d['history'],'time'),array(d['history'],'core_lapse'),c='#275d7f',lw=1.2)
        ax[1,j].set(ylim=(0,1),xlabel=r'$t/M_{\rm ref}$')
        for a in ax[:,j]:a.set_xlim(0,s['endpoint'])
        if d['case']['mode']=='long':
            for a in ax[:,j]:a.axvline(300,c='.55',ls='-.',lw=.8)
        events(ax[:,j],s)
    ax[0,0].set_ylabel(r'COM rest-mass shell radius $r_q/M_{\rm ref}$');ax[1,0].set_ylabel(r'Core lapse $\langle\alpha\rangle_{r<0.75M}$')
    fig.colorbar(ScalarMappable(norm=norm,cmap=cm),ax=ax[0,:],fraction=.025,pad=.02,ticks=[.1,.5,.9],label='Enclosed rest-mass fraction')
    save(fig,'s08_extension_shells_lapse')
    fig,ax=plt.subplots(2,len(data),figsize=(5*len(data),5.5),squeeze=False,layout='constrained',sharey='row')
    extent=max(1e-15,1.2*max(abs(r['closure']) for d in data for r in d['mass']))
    for j,d in enumerate(data):
        m=d['mass'];t=array(m,'time');m0=m[0]['m_initial']
        ax[0,j].plot(t,array(m,'m_alive')/m0,c='#275d7f',label='Alive')
        ax[0,j].plot(t,sum(array(m,'n_'+k) for k in ['sphere','lapse','horizon'])/2000000,c='#b35c26',label='Removed')
        ax[0,j].plot(t,array(m,'n_exit')/2000000,c='.45',ls=':',label='Escaped')
        ax[0,j].plot(t,1+array(m,'closure'),c='.1',ls='--',lw=.9,label='Accounted total')
        ax[0,j].set(title=d['case']['title'],ylim=(-.03,1.04));ax[1,j].plot(t,array(m,'closure'),c='#275d7f',lw=.9)
        ax[1,j].set(ylim=(-extent,extent),xlabel=r'$t/M_{\rm ref}$');ax[1,j].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        for a in ax[:,j]:a.set_xlim(0,d['summary']['endpoint'])
        events(ax[:,j],d['summary'])
    ax[0,0].set_ylabel(r'Rest mass / initial $M_0$');ax[1,0].set_ylabel(r'$M_{0,{\rm accounted}}/M_0(0)-1$');ax[0,-1].legend(frameon=False,fontsize=8,loc='center')
    save(fig,'s08_extension_mass_conservation')
    fig,axes=plt.subplots(1,len(data),figsize=(5*len(data),3.6),squeeze=False,layout='constrained',sharey=True)
    accepted_masses=[r[k] for d in data for r in d['ah'] if r['accepted']==1 for k in ['mirr','mass']]
    ah_upper=max(1.15,1.08*max(accepted_masses)) if accepted_masses else 1.15
    for ax,d in zip(axes[0],data):
        ah=d['ah'];good=array(ah,'accepted')==1
        for key,c,ls,label in [('mirr','#275d7f','-',r'$M_{\rm irr}$'),('mass','#b35c26','--',r'$M_{\rm CH}$')]:
            values=array(ah,key);values[~good]=np.nan;ax.plot(array(ah,'time'),values,c=c,ls=ls,label=label)
        if not good.any():ax.text(.5,.5,'No accepted horizon',ha='center',transform=ax.transAxes)
        ax.set(title=d['case']['title'],xlabel=r'$t/M_{\rm ref}$',xlim=(0,d['summary']['endpoint']),ylim=(0,ah_upper));events([ax],d['summary'])
    axes[0,0].set_ylabel(r'AH mass / $M_{\rm ref}$');axes[0,-1].legend(frameon=False)
    save(fig,'s08_extension_AH_masses')
    atomic(ROOT/'evidence/PLOTS_RENDERED.json',dict(job=job(),utc=now(),products=[receipt(p) for p in sorted((ROOT/'plots').glob('*.pdf'))],visual_review='pending actual inspection',x_axes='Each column uses its actual endpoint; time units common; do not compare column widths as durations'))

def movies(data):
    from movie_helpers import load_case,panel,encode
    products=[];states=[];spatial=[]
    for d in data:
        case=load_case(d['case']);fig=plt.figure(figsize=(10,6),layout='constrained');g=fig.add_gridspec(2,2,width_ratios=[1.1,1]);axes=[fig.add_subplot(g[:,0]),fig.add_subplot(g[0,1]),fig.add_subplot(g[1,1])]
        p=panel(fig,axes,case);axes[0].set_title(d['case']['title']);fig.suptitle(r'$y_c=0.500$ • fixed particle IDs • accepted AH contours',fontsize=11)
        fig.colorbar(p['scatter'],ax=axes[0],fraction=.035,pad=.025,label=r'Initial radius $r_0/R_{\rm iso}(0)$')
        products.append(encode(d['case']['id'],fig,[(case,p)],case['times']))
    fig=plt.figure(figsize=(5*len(data),8),layout='constrained');g=fig.add_gridspec(3,len(data),height_ratios=[2,1,1])
    for j,d in enumerate(data):
        case=load_case(d['case']);axes=[fig.add_subplot(g[k,j]) for k in range(3)];p=panel(fig,axes,case);axes[0].set_title(d['case']['title'])
        for a in axes[1:]:a.set_xlim(0,500)
        states.append((case,p));spatial.append(axes[0])
    fig.colorbar(states[0][1]['scatter'],ax=spatial,fraction=.02,pad=.02,label=r'Initial radius $r_0/R_{\rm iso}(0)$')
    end=max(d['summary']['endpoint'] for d in data);times=np.arange(0,end+1e-8,.5)
    if times[-1]<end-1e-8:times=np.r_[times,end]
    products.append(encode('s08_extension_comparison',fig,states,times,True))
    atomic(ROOT/'evidence/MOVIES_RENDERED.json',dict(job=job(),utc=now(),products=products,visual_review='pending encoded-frame inspection'))

def main():
    job();data=assemble();results=numeric(data);figures(data);movies(data)
    lines=['# Session 8 extensions — scientific results','', 'All cases start from the same proper-volume stratified seed-1 realization with2,000,000 equal-rest-mass particles. M_ref=1 is the unperturbed equilibrium gravitational mass; perturbed cases intentionally change the initial energy and angular momentum slightly.','', '| Case | Endpoint [M_ref] | First accepted AH [M_ref] | Final surviving particles | Max rest-mass closure |','|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['mode']} | {r['endpoint_M']:.3f} | {r['first_AH_M'] if r['first_AH_M'] is not None else 'none accepted'} | {r['final_particles']} | {r['max_rest_mass_closure']:.3e} |")
    lines+=['','| Case | Final-five-M median M_irr / M_ref | Final-five-M median M_CH / M_ref |','|---|---:|---:|']
    for r in results:
        mi='unavailable' if r['final_Mirr_median'] is None else f"{r['final_Mirr_median']:.3f}"
        mc='unavailable' if r['final_MCH_median'] is None else f"{r['final_MCH_median']:.3f}"
        lines.append(f"| {r['mode']} | {mi} | {mc} |")
    lines+=['','[Shells and lapse](../plots/s08_extension_shells_lapse.pdf), [particle rest-mass ledger](../plots/s08_extension_mass_conservation.pdf), [AH masses](../plots/s08_extension_AH_masses.pdf), and [outer-domain diagnostics](../plots/s08_extension_boundary_lapse.pdf).','', 'The accompanying extension_summary.csv records pre-removal constraint norms, sampled AMR resolution/timesteps, fallback particle-step counts, and deliberate initial energy/angular-momentum changes. Completed timestep minima include final endpoint adjustments; the full sampled dt/AMR history is retained. No search failure or fatal native diagnostic is accepted.', '', 'Actual output coverage is checked for histories, mass ledgers, fixed-ID cohorts, full particle snapshots and AH searches. The recorded maximum gap cannot exceed nominal cadence plus one maximum completed or reported timestep. Global constraint RMS uses native proper-volume integrals over cells outside the chi-based diagnostic excision. Large exterior volume dilutes this whole-box norm; it is not an error bound for the cluster. Central norms use their separate fixed coordinate regions. The complete time series and coverage receipt remain on the cluster.']
    lines+=['','Shells enclose10–90% of the rest mass about the instantaneous COM and are coordinate radii. Curves stop before the first particle removal. Each static column uses its own actual endpoint; synchronized movies share time and explicitly clear ended cases. The unperturbed300M boundary is marked. Alive+removed+escaped is a rest-mass bookkeeping check; AH mass is not added, and this is not ADM-energy conservation.','', 'AH masses use the unchanged on-grid/residual/resolution/three-search persistence gates. Missing or rejected surfaces are gaps. Final-five-M accepted-sample summaries are qualitative and do not assert settling.','', 'An accepted AH confirms black-hole formation. Without one, contraction/excision is not promoted to an AH-confirmed black hole. Finite-time survival/expansion does not establish stable migration. The known frozen particle-source coupling remains unchanged; the cause of the unperturbed expansion remains unresolved. These are controlled perturbation/long-time extensions, not a convergence study.','', 'The exact continuation retains the +/-512M box. Boundary diagnostics are assessed separately; a larger-domain control was not performed, so causal isolation of the full500M interval is not asserted.','', 'Native raw data remain under '+str(EXT)+'. Original baseline data remain under '+str(BASE)+'. All current raw data are retained; only verified superseded extension recovery checkpoints may have been rotated.']
    (ROOT/'reports/SCIENCE_REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'reports/FIGURES.md').write_text('''# Extension figure conventions

All times/radii/AH masses use the unperturbed equilibrium gravitational mass M_ref=1. The inward initial constraint solve deliberately changes ADM mass slightly; no renormalization conceals that change. Each static column shows its own actual endpoint; compare the labeled time axes. Shells are rest-mass deciles about the instantaneous COM in coordinate radius, cut strictly before first removal. The lapse is the coordinate-volume mean within r<0.75M_ref, whose discrete integration changes with AMR. Dashed vertical lines mark first accepted AH; dotted lines mark first particle removal; the extended unperturbed panel marks300M.

The mass figure shows alive+removed+escaped particle rest mass normalized by initial M0. Its residual is bookkeeping, not ADM-energy conservation; AH gravitational mass is never added. AH curves contain only finite, on-grid, resolved, residual-qualified, three-search-persistent surfaces. Rejected searches are gaps; absence is labeled explicitly. Final-five-M accepted-sample medians do not assert settling.

The boundary/lapse figure uses equatorial annular coordinate-area means from retained native slices. No larger-domain control or causal-isolation proof is supplied. Source-time-coupling and finite-grid/particle limitations remain.
''')
    (ROOT/'reports/MOVIES.md').write_text('''# Extension movies

Individual and synchronized15fps MP4s use the same20,000 initial IDs (tags divisible by100). Colors remain fixed at initial coordinate radius/r_iso_surface(0), with scale0–1. Only |z|<0.05 r_iso_surface(0) is shown. Each spatial window stays fixed over its whole movie and includes all displayed cohort particles with a5% margin; physical scales may differ and are labeled in M_ref. The dotted circle is the initial coordinate surface. Black contours are contemporaneous accepted AH equators from native harmonic coefficients; rejected or stale searches are not drawn.

Frames use completed cohort samples and print actual sample time. Synchronized movies clear and label ended runs. The nominal comparison cadence is0.5M_ref, with the actual final endpoint appended when fractional. No interpolation invents particles or evolution beyond an endpoint. Every encoded frame is decoded, frame counts checked, and representative encoded frames are retained for actual visual review. Raw data and lossless PNG frames stay on the cluster.
''')
    atomic(ROOT/'evidence/PRODUCTS_READY.json',dict(job=job(),utc=now(),science=results,numeric=True,figures=True,movies_full_decode=True,actual_visual_acceptance=False,accounting_and_thin_mirror_pending=True))
if __name__=='__main__':main()
