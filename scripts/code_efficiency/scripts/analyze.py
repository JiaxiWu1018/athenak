#!/usr/bin/env python3
import os
import fcntl
"""Generate source-backed performance tables, figures, reports, and live status in Slurm."""
import csv,json,sys,math,datetime,textwrap,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(os.environ.get('EFFICIENCY_SESSION',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT.parents[0]/'scripts'))
from paper_style import configure
configure()
report_lock=(ROOT/'results/.report.lock').open('a')
fcntl.flock(report_lock,fcntl.LOCK_EX)
COLORS={'st':'#0072B2','flat':'#D55E00','mesh':'#595959','push':'#0072B2','energy':'#56B4E9','migration':'#CC79A7','records':'#E69F00','sort':'#D55E00','transfer':'#009E73','atomic':'#F0E442','other':'#999999'}
PHASE_LABELS={'mesh':'Spacetime','push':'Particle push','energy':'Particle energy','migration':'Communication / ownership','records':'Deposition records','sort':'Deposition host sort','transfer':'Deposition host/device copies','atomic':'Deposition atomic kernel','other':'Other'}
FILES=[]
NOW=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
def load(kind):
    return [json.loads(p.read_text()) for p in sorted((ROOT/'results'/kind).glob('*/receipt.json'))]
def elapsed(r):return r['timings'][0]['elapsed_max']
def spm(r):return r['timings'][0]['seconds_per_M']
def step(r):return 1000*elapsed(r)/r['timings'][0]['cycles']
def save(fig,name,caption):
    if len(valid)<60:
        fig.text(.99,.006,f'Partial campaign: {len(valid)}/60 throughput cases complete',ha='right',fontsize=7,color='.35',usetex=False)
        caption='Partial campaign; only completed measurements are shown. '+caption
    fig.tight_layout(pad=1.3,rect=(0,.035,1,1))
    for ext in ['pdf','png']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=550)
    FILES.append((name,caption));plt.close(fig)
def phase_groups(r):
    t=max(r['timings'],key=lambda x:x['elapsed_local']);p=t['phases'];dep=p.get('deposition_total',0)
    get=lambda key:p.get(key,0)
    result=dict(mesh=get('spacetime'),push=get('particle_push'),energy=get('particle_energy'),migration=sum(v for k,v in p.items() if k.startswith('migration_') or k=='ownership'),records=get('deposition_zero_count_records'),sort=get('deposition_host_sort_validate'),transfer=get('deposition_device_to_host')+get('deposition_host_to_device'),atomic=get('deposition_atomic_kernel'))
    children=sum(result[k] for k in ['records','sort','transfer','atomic'])+get('deposition_image_exchange')
    result['migration']+=get('deposition_image_exchange')
    result['other']=max(0,dep-children)+get('driver_timestep')+get('particle_timestep')
    result['other']+=max(0,t['elapsed_local']-sum(result.values()))
    return {k:1000*v/t['cycles'] for k,v in result.items()}
devices=[r for r in load('device') if r['valid'] and not r['case']['name'].startswith('numeric_')];device_byname={r['case']['name']:r for r in devices}
allthrough=load('throughput');through=[r for r in allthrough if not r.get('superseded_reason')];valid=[r for r in through if r['valid']];profiles=[r for r in load('profile') if r['valid']];allcases=json.loads((ROOT/'matrix.json').read_text());byname={r['case']['name']:r for r in valid}
rows=[]
for r in allthrough:
    c=r['case'];t=r['timings'][0] if r['timings'] else {};row={**c,'valid':r['valid'],'superseded_reason':r.get('superseded_reason',''),'device_timing_compiled':t.get('device_timing_compiled'),'job_id':r['job_id'],'endpoint':r['endpoint'],'actual_endpoint':t.get('final_time'),'target_reached':r.get('target_reached',True),'elapsed_seconds':t.get('elapsed_max'),'cycles':t.get('cycles'),'particle_updates_per_second':t.get('particles_per_second'),'cell_cycles_per_second':t.get('cells_per_second'),'seconds_per_M':t.get('seconds_per_M'),'M_per_second':1/t['seconds_per_M'] if t.get('seconds_per_M',0)>0 else None,'step_ms':1000*t['elapsed_max']/t['cycles'] if t.get('cycles',0)>0 else None,'cell_rate_kind':'mesh-normalized' if c['frozen'] else 'physical-cell evolution','boris_fallbacks':sum(t.get('boris_fallbacks',0) for t in r['timings']),'boris_rejections':sum(t.get('boris_rejections',0) for t in r['timings'])}
    rows.append(row)
if rows:
    with (ROOT/'results/throughput.csv').open('w') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
profile_rows=[]
for r in profiles:
    groups=phase_groups(r);device=device_byname.get(r['case']['name']);cycles=np.sum([t['device_cycles'] for t in device['timings']],axis=0) if device else np.zeros(5)
    profile_rows.append(dict(case=r['case']['name'],job_id=r['job_id'],**groups,interpolation_eval_cycle_fraction=float(cycles[0]/cycles[1]) if cycles[1] else None,sampled_particle_steps=int(cycles[4]),interpolation_calls=int(cycles[2]),geodesic_evaluations=int(cycles[3])))
if profile_rows:
    with (ROOT/'results/profile_phases.csv').open('w') as out:w=csv.DictWriter(out,fieldnames=list(profile_rows[0]));w.writeheader();w.writerows(profile_rows)
# Nsight's kernel percentages sum GPU execution duration, not elapsed wall time.
kernel_rows=[]
for r in profiles:
 folder=ROOT/r['attempt_path']
 for table in folder.glob('nsys*cuda_gpu_kern_sum*.csv'):
  with table.open() as f:
   reader=csv.DictReader(f)
   for row in reader:
    if not row.get('Name'):continue
    kernel_rows.append(dict(case=r['case']['name'],job_id=r['job_id'],kernel=row['Name'],gpu_time_percent=row.get('Time (%)'),total_nanoseconds=row.get('Total Time (ns)'),instances=row.get('Instances'),source_csv=str(table.relative_to(ROOT))))
if kernel_rows:
 with (ROOT/'results/gpu_kernels.csv').open('w') as out:w=csv.DictWriter(out,fieldnames=list(kernel_rows[0]));w.writeheader();w.writerows(kernel_rows)
# 1: count sweep. Independent y scales keep all four comparisons readable.
count=[r for r in valid if r['case']['test']==1]
if count:
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.3),sharex=True)
    for row,feedback in enumerate([False,True]):
        for metric in ['st','flat']:
            data=sorted([r for r in count if r['case']['feedback']==feedback and r['case']['metric']==metric],key=lambda r:r['case']['n'])
            if not data:continue
            x=[r['case']['n'] for r in data]
            axes[row,0].plot(x,[r['timings'][0]['particles_per_second']/1e6 for r in data],'-o',color=COLORS[metric],label=metric.upper(),ms=4)
            axes[row,1].plot(x,[spm(r) for r in data],'-o',color=COLORS[metric],label=metric.upper(),ms=4)
        axes[row,0].set_ylabel('Million particle updates / s');axes[row,1].set_ylabel('Wall seconds / $M$');axes[row,1].set_yscale('log')
        for ax in axes[row]:ax.set_xscale('log');ax.grid(alpha=.18);ax.text(.04,.94,'Deposition '+('on' if feedback else 'off'),transform=ax.transAxes,va='top');ax.legend(loc='best')
    for ax in axes[1]:ax.set_xlabel('Particles, $N$')
    save(fig,'01_particle_count','Frozen periodic +/-4M box, one rank/GPU, one 32^3 block, four ghost zones, dt=0.05M. Main rates use completed evolution after 0.5M; single passes. Flat and ST use the same random draws and local momentum magnitudes.')
# 2: layout comparison and actual stored-cell overhead.
blocks=[r for r in valid if r['case']['test']==2]
if blocks:
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0))
    labels=[r'$1\times64^3$',r'$8\times32^3$',r'$64\times16^3$']
    for j,feedback in enumerate([False,True]):
        for k,metric in enumerate(['st','flat']):
            y=[step(byname[f'blocks_{metric}_dep{int(feedback)}_B{b}']) if f'blocks_{metric}_dep{int(feedback)}_B{b}' in byname else np.nan for b in [64,32,16]]
            axes[j].bar(np.arange(3)+(k-.5)*.32,y,width=.30,color=COLORS[metric],label=metric.upper())
        axes[j].set_xticks(range(3),labels);axes[j].set_ylabel('Wall ms / full timestep');axes[j].set_ylim(bottom=0);axes[j].legend();
        if not any(r['case']['feedback']==feedback for r in blocks):
            axes[j].text(.5,.5,'Awaiting measurements',ha='center',va='center',transform=axes[j].transAxes);axes[j].set_yticks([])
        axes[j].text(.04,.94,'Deposition '+('on' if feedback else 'off'),transform=axes[j].transAxes,va='top');axes[j].grid(axis='y',alpha=.18)
    save(fig,'02_meshblock_layout','Same one-million-particle realization, periodic +/-4M box and global 64^3 grid, one rank/GPU, dt=0.05M. With four ghosts, stored/physical cell ratios are 1.424, 1.953 and 3.375 for the three layouts.')
# 3: scaling. Baseline is the eight-block case, not one 64^3 block.
ranksets=[r for r in valid if r['case']['test']==3]
if ranksets:
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0))
    for metric in ['st','flat']:
      for feedback in [False,True]:
        baseline=byname.get(f'blocks_{metric}_dep{int(feedback)}_B32')
        if not baseline:continue
        data=[(1,baseline)]+[(p,byname[f'ranks_{metric}_dep{int(feedback)}_R{p}']) for p in [2,4] if f'ranks_{metric}_dep{int(feedback)}_R{p}' in byname]
        x=[p for p,r in data];s=[spm(baseline)/spm(r) for p,r in data];label=f'{metric.upper()}, dep '+('on' if feedback else 'off');style='-' if feedback else '--'
        axes[0].plot(x,s,style+'o',color=COLORS[metric],label=label,ms=4);axes[1].plot(x,[100*a/p for a,p in zip(s,x)],style+'o',color=COLORS[metric],label=label,ms=4)
    axes[0].plot([1,2,4],[1,2,4],':',color='.4',label='Ideal');axes[0].set_ylabel('Speedup over one GPU');axes[1].axhline(100,color='.4',ls=':');axes[1].set_ylabel('Parallel efficiency (%)',usetex=False)
    for ax in axes:ax.set_xticks([1,2,4]);ax.set_xlabel('MPI ranks / A100 GPUs');ax.grid(alpha=.18)
    axes[0].legend(fontsize=7)
    save(fig,'03_strong_scaling','Strong scaling of N=1 million with eight 32^3 blocks on a fixed global 64^3 grid. One A100 per MPI rank; speedup uses wall seconds/M against the matching one-rank eight-block case. Node sharing is recorded in provenance.')
# 4: dynamic PPC rates and diagnostic phase decomposition.
dyn=[r for r in valid if r['case']['name'].startswith('dynamic_ppc')]
pdyn=[r for r in profiles if r['case']['name'].startswith('dynamic_ppc')]
if dyn:
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.1))
    d=sorted(dyn,key=lambda r:r['case']['ppc']);axes[0].plot([r['case']['ppc'] for r in d],[step(r) for r in d],'-o',color=COLORS['st'],label='Live spacetime')
    baseline=byname.get('dynamic_mesh_only');frozen=byname.get('dynamic_matched_frozen_ppc24')
    if baseline:axes[0].axhline(step(baseline),color='.4',ls=':',label='Mesh-only control')
    if frozen:axes[0].scatter([24],[step(frozen)],marker='s',facecolors='none',edgecolors=COLORS['sort'],label='Matched frozen')
    axes[0].set_xscale('log',base=2);axes[0].set_xticks([6,24,96,384],[6,24,96,384]);axes[0].set_xlabel('Particles / physical cell');axes[0].set_ylabel('Wall ms / full timestep');axes[0].legend(fontsize=7);axes[0].grid(alpha=.18)
    p=sorted(pdyn,key=lambda r:r['case']['ppc']);bottom=np.zeros(len(p))
    for key in ['mesh','push','energy','migration','records','sort','transfer','atomic','other']:
      y=np.array([phase_groups(r)[key] for r in p]);axes[1].bar(range(len(p)),y,bottom=bottom,color=COLORS[key],label=PHASE_LABELS[key]);bottom+=y
    axes[1].set_xticks(range(len(p)),[r['case']['ppc'] for r in p]);axes[1].set_xlabel('Particles / physical cell');axes[1].set_ylabel('Profiled ms / full timestep');axes[1].legend(fontsize=6,ncol=1,loc='upper left',bbox_to_anchor=(1.01,1));
    if not p:axes[1].text(.5,.5,'Awaiting phase profiles',ha='center',va='center',transform=axes[1].transAxes);axes[1].set_yticks([])
    axes[1].grid(axis='y',alpha=.18)
    save(fig,'04_dynamic_particle_mesh_cost','ST yc=0.819, global 64^3 grid in +/-12M, eight 32^3 blocks, one GPU, dt=0.05M. Throughput and profiled bars come from separate runs; phase fences affect profiled timings; device-clock instrumentation runs in separate diagnostic cases. Source replicas preserve initial stress-energy; mesh-only holds that initial source fixed.')
# 5: sampling costs and initial rank imbalance.
sampling={('uniform',1):byname['dynamic_ppc24']} if 'dynamic_ppc24' in byname else {}
for r in valid:
    if r['case']['test']==5:sampling[(r['case']['distribution'],r['case']['ranks'])]=r
if sampling:
    distributions=['uniform','central4','central1','block','point'];names=['Uniform','Central 4M','Central 1M','One block','One site'];fig,axes=plt.subplots(1,2,figsize=(7.1,3.5))
    ranks=[1,2,4];palette=['#0072B2','#E69F00','#CC79A7'];imb=np.full((3,5),np.nan)
    for k,p in enumerate(ranks):
      y=[]
      for j,d in enumerate(distributions):
        r=sampling.get((d,p));y.append(spm(r) if r else np.nan)
        if r:counts=np.array([x['local_count'] for x in r['initial']]);imb[k,j]=counts.max()/counts.mean()
      axes[0].bar(np.arange(5)+(k-1)*.24,y,width=.23,color=palette[k],label=f'{p} GPU'+('s' if p>1 else ''))
    axes[0].set_xticks(range(5),names,rotation=30,ha='right');axes[0].set_ylabel('Wall seconds / $M$');axes[0].legend();axes[0].grid(axis='y',alpha=.18)
    image=axes[1].imshow(np.ma.masked_invalid(imb),vmin=1,vmax=4,cmap='cividis',aspect='auto');axes[1].set_xticks(range(5),names,rotation=30,ha='right');axes[1].set_yticks(range(3),['1 GPU','2 GPUs','4 GPUs']);axes[1].set_ylabel('Initial particle-count imbalance')
    for i in range(3):
      for j in range(5):
        if np.isfinite(imb[i,j]):axes[1].text(j,i,f'{imb[i,j]:.2f}',ha='center',va='center',color='white' if imb[i,j]<2.3 else 'black',fontsize=8)
    fig.colorbar(image,ax=axes[1],label='Maximum / mean rank count',fraction=.05,pad=.03)
    save(fig,'05_sampling_and_rank_imbalance','All cases have N=6,291,456, the same mesh, timestep and initial physical source. Extra weighted replicas change number distribution. Central widths are Gaussian sigma values. Hotspots concentrate extra replicas in one block or one source site. Heatmap shows initial count imbalance; final counts and profiling evidence are retained separately.')
# 6: expensive phase ranking and interpolation fraction.
if profiles:
    chosen=[]
    for name in ['dynamic_ppc6','dynamic_ppc24','dynamic_ppc96','dynamic_ppc384']:
      q=next((r for r in profiles if r['case']['name']==name),None)
      if q:chosen.append(q)
    if not chosen:chosen=profiles[:2]
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2));r=chosen[-1];groups=phase_groups(r);order=sorted(groups,key=groups.get);total=sum(groups.values())
    axes[0].barh([PHASE_LABELS[k] for k in order],[100*groups[k]/total for k in order],color=[COLORS[k] for k in order]);axes[0].set_xlabel('Profiled timestep cost (%)',usetex=False);axes[0].grid(axis='x',alpha=.18);axes[0].text(.97,.04,r['case']['name'],ha='right',transform=axes[0].transAxes,fontsize=8,usetex=False)
    labels=[];fractions=[]
    for q in chosen:
      device=device_byname.get(q['case']['name']);dc=np.sum([t['device_cycles'] for t in device['timings']],axis=0) if device else np.zeros(5)
      if dc[1]:labels.append(str(q['case']['ppc']) if q['case']['name'].startswith('dynamic_ppc') else q['case']['metric'].upper()+' frozen');fractions.append(100*dc[0]/dc[1])
    axes[1].bar(labels,fractions,color=COLORS['push']);
    if not fractions:axes[1].text(.5,.5,'Awaiting device-clock samples',ha='center',va='center',transform=axes[1].transAxes)
    axes[1].set_ylim(0,100);axes[1].set_xlabel('Particles / physical cell' if all(q['case']['name'].startswith('dynamic_ppc') for q in chosen) else 'Frozen diagnostic');axes[1].set_ylabel('Interpolation / geodesic evaluation cycles (%)',usetex=False);axes[1].grid(axis='y',alpha=.18)
    save(fig,'06_kernel_cost_and_interpolation','Cost ranking uses the slowest rank of a diagnostic run, without double-counting nested deposition timers. Interpolation fractions use sampled device clock cycles inside geodesic evaluations; they are estimates in an instrumented kernel, not percentages of total GPU wall time. Hardware performance counters are administrator-restricted.')
# Evidence checks and readable reports.
checks=[]
for r in through+load('gate'):
    if not r['valid']:checks.append(f"{r['case']['name']}: failed/incomplete; exit {r['exit_code']}")
source_candidates=[r for r in load('gate')+valid if r['valid'] and r['case']['sampling']=='replicas' and not r['case']['mesh_only']]
source_ok=True;source_max=0
if source_candidates:
    ref=np.array(source_candidates[0]['initial'][0].get('source_moments',[]))
    for r in source_candidates:
      v=np.array(r['initial'][0].get('source_moments',[]))
      if len(v)==len(ref) and len(v):
        peak=max(float(np.max(np.abs(ref))),1e-30);scale=np.where(np.abs(ref)>peak*1e-12,np.abs(ref),peak);difference=float(np.max(np.abs(v-ref)/scale));source_max=max(source_max,difference);source_ok &= difference<1e-8
      else:source_ok=False
source_cell_errors=[x.get('source_reference_max_relative_error') for r in source_candidates for x in r['initial'] if 'source_reference_max_relative_error' in x]
superseded_records=[json.loads(p.read_text()) for p in (ROOT/'results/throughput').glob('*/previous_receipt_*.json')]+allthrough
superseded_count=len({(r['job_id'],r.get('attempt_path','')) for r in superseded_records if r.get('superseded_reason')})
interruptions=json.loads((ROOT/'provenance/interruptions.json').read_text()) if (ROOT/'provenance/interruptions.json').exists() else []
summary=dict(interrupted_attempts=len(interruptions),device_diagnostics=len(devices),planned_profiles=27,planned_device_diagnostics=6,superseded_attempts=superseded_count,source_cell_max_relative_error=max(source_cell_errors) if source_cell_errors else None,updated_utc=NOW,planned=60,completed=len(valid),failed=len(through)-len(valid),profiled=len(profiles),source_moments_match=bool(source_ok) if source_candidates else None,source_moments_max_scaled_difference=source_max,figures=[name for name,caption in FILES],failures=checks)
numerical=json.loads((ROOT/'provenance/numerical_gate.json').read_text()) if (ROOT/'provenance/numerical_gate.json').exists() else {}
validation=dict(throughput_coverage=len(valid)==60,profile_coverage=len(profiles)==27,device_coverage=len(devices)==6,numerical_gate=numerical.get('passed',False),source_moments_match=bool(source_ok),compiled_modes=all(t.get('device_timing_compiled')==(r['kind']=='device') for r in valid+profiles+devices for t in r['timings']))
validation['complete']=all(validation.values())
(ROOT/'results/VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
(ROOT/'results/SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
findings=[]
if count:
  for metric in ['st','flat']:
    large=byname.get(f'count_{metric}_dep1_N100000000');off=byname.get(f'count_{metric}_dep0_N100000000')
    if large and off:findings.append(f'At 100 million particles in the {metric.upper()} fixed background, the complete particle loop costs {spm(large):.3g} s/M versus {spm(off):.3g} s/M with deposition off ({spm(large)/spm(off):.2f}x).')
if pdyn:
  for r in sorted(pdyn,key=lambda r:r['case']['ppc']):
    g=phase_groups(r);key=max(g,key=g.get);findings.append(f'At {r["case"]["ppc"]} particles/cell, {PHASE_LABELS[key].lower()} is the largest measured phase ({100*g[key]/sum(g.values()):.1f}% of the diagnostic timestep).')
for metric in ['st','flat']:
  for feedback in [False,True]:
    layout=[byname.get(f'blocks_{metric}_dep{int(feedback)}_B{b}') for b in [64,32,16]]
    if all(layout):findings.append(f'At fixed global 64³ resolution in {metric.upper()} with deposition {"on" if feedback else "off"}, 1×64³, 8×32³ and 64×16³ blocks cost '+', '.join(f'{step(r):.3g}' for r in layout)+' ms/step, respectively.')
    baseline=byname.get(f'blocks_{metric}_dep{int(feedback)}_B32')
    multi=byname.get(f'ranks_{metric}_dep{int(feedback)}_R4')
    if baseline and multi:findings.append(f'Four GPUs give {spm(baseline)/spm(multi):.2f}× speedup ({25*spm(baseline)/spm(multi):.1f}% parallel efficiency) for the {metric.upper()} million-particle eight-block case with deposition {"on" if feedback else "off"}.')
for r in sorted(devices,key=lambda r:r['case']['n']):
  dc=np.sum([t['device_cycles'] for t in r['timings']],axis=0)
  if dc[1]:findings.append(f'For {r["case"]["name"]}, interpolation accounts for {100*dc[0]/dc[1]:.1f}% of sampled geodesic-evaluation cycles ({int(dc[4]):,} sampled particle steps); this is an instrumented device-cycle estimate.')
if sampling:
  for p in [1,2,4]:
    base=sampling.get(('uniform',p));point=sampling.get(('point',p))
    if base and point:findings.append(f'On {p} GPU(s), the same-site sampling case costs {spm(point)/spm(base):.2f}x the uniform case at the same particle count and initial source.')
for name in sorted(set(x['case'] for x in kernel_rows)):
  ranked=[x for x in kernel_rows if x['case']==name and x['gpu_time_percent']]
  if ranked:
    top=max(ranked,key=lambda x:float(x['gpu_time_percent']));findings.append(f'Nsight GPU kernel evidence for {name}: the leading kernel is in {top['kernel'].split('/',1)[0][:100]} and contributes {float(top["gpu_time_percent"]):.1f}% of summed traced GPU kernel duration; exact kernel names and NVTX phase labels are in results/gpu_kernels.csv. GPU duration excludes host sorting and MPI waiting.')
if not findings:findings=['The campaign is active; completed cases are tabulated below. Superseded build-mode attempts are preserved and excluded from promoted measurements. Bottleneck conclusions will be added when the corresponding comparisons finish.']
report=f'# Paper Session 02 — code efficiency\n\nUpdated {NOW}. **{len(valid)}/60 throughput cases complete**, {len(profiles)} diagnostic profiles complete; {len(through)-len(valid)} failed or incomplete cases.\n\n'+ '\n\n'.join(findings)+'\n\n'
if interruptions:report+=f'{len(interruptions)} scheduler-interrupted attempt(s) are preserved separately and retried; details are in [the interruption record](../provenance/interruptions.json).\n\n'
report+='All throughput measurements are single passes targeting 5M, with a 0.5M warm-up excluded. A solver wall-clock guard finishes cleanly before the Slurm limit; shorter runs are labeled by their actual endpoint and need at least ten measured cycles to supply a rate. Reported particle rates count complete cycle updates; physical-cell rates exclude ghosts. Frozen cell rates are mesh-size normalizations and represent no spacetime evolution. Wall seconds/M uses the measured physical-time interval.\n\n'
report+='## Figures\n\n'
for name,caption in FILES:report+=f'![{name}](../figures/{name}.png)\n\n{caption} [PDF](../figures/{name}.pdf).\n\n'
report+='## Exact performance table\n\n| Case | GPUs | N | Particle updates/s | Cell cycles/s | Wall s/M | Step ms | Actual end (M) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for r in valid:
 c=r['case'];t=r['timings'][0];report+=f'| {c["name"]} | {c["ranks"]} | {c["n"]} | {t["particles_per_second"]:.6g} | {t["cells_per_second"]:.6g} | {spm(r):.6g} | {step(r):.6g} | {t['final_time']:.6g} |\n'
report+='\n## Interpretation and limits\n\nThese are numerical workload measurements, not ST stability or convergence results. Single passes have no replicate-based confidence intervals. Profiling fences, sampled device clocks, and tracing affect diagnostic timings; throughput runs disable them. The interpolation ratio concerns geodesic-evaluation cycles in the instrumented pusher, not a directly separable production kernel. Mesh-only evolves with the initial source held fixed. Weighted replicas keep initial deposited stress-energy fixed within accumulation roundoff, while exact trajectories and accumulation order can diverge during evolution.\n\n'
report+=f'Initial source-moment matching: {("PASS" if source_ok else "FAIL/INCOMPLETE") if source_candidates else "not yet measured"}; maximum scaled difference {source_max:.3g}. Cell-by-cell source checks use the common six-direction template; maximum relative error {max(source_cell_errors) if source_cell_errors else 'not yet measured'}. Every promoted case passed its compiled-mode check and initial/final count, ID-checksum, finite-state, ownership, zero-destruction and frozen-geometry checks. Boris fallback/rejection counts are included in the [CSV](../results/throughput.csv).\n\n'
if checks:report+='## Failed or incomplete cases\n\n'+'\n'.join('- '+x for x in checks)+'\n\n'
report+='[Compact summary](../results/SUMMARY.json), [throughput CSV](../results/throughput.csv), [profile phase CSV](../results/profile_phases.csv), [traced GPU kernels](../results/gpu_kernels.csv). Exact inputs, executable hashes, source state, launches, scheduler jobs and sampled resource observations are retained with each receipt under `results/`.\n'
(ROOT/'reports/REPORT_code_efficiency.md').write_text(report)
agent=f'''# Implementation and provenance report

Updated {NOW}. The parent is origin/NRPIC at ca0f5298e3752363e7e0b105e4004d0631aeca91. ST ODE/metric construction is selectively reused from project/ST-cluster at b51efb27; complete freeze logic is ported from 320ba2b0. No donor pusher, alternative interpolation, or deposition optimization is imported. The dedicated sampler uses global-ID counter draws and source-preserving weighted replicas.

Builds use CUDA12.9, OpenMPI5.0.7, MPI, Kokkos AMPERE80, double precision and Release. A separate build enables sampled device clocks. All builds, simulations, profiling, reductions and rendering run in Slurm. Perseus enforces two CPUs per allocated GPU. Main runs use one GPU/rank and no output/restart blocks; zero-mass vacuum particles remain in the workload. Nsight Compute hardware counters are administrator-restricted; Nsight Systems timelines and sampled device cycles supply diagnostic evidence.

Status: {len(valid)} successful throughput cases; {len(through)-len(valid)} unsuccessful cases; {len(profiles)} profiles. `provenance/jobs.jsonl` is the submitted job ledger; each receipt records the exact launch, source/dirty patch, input and executable hash. Build environments and flags are in `provenance/build_*` and CMake caches.

Reproduction: build with `jobs/build.sbatch`; generate inputs with `scripts/generate_matrix.py`; run focused gates with `scripts/submit.py --kind gate`; submit the main matrix with `--kind throughput`; profiles with `--kind profile`; sampled clocks with `--kind device`; regenerate all artifacts in Slurm using `scripts/analyze.py`. Never run simulation or plotting on the login node.

Superseded compile-mode attempts are excluded from performance results and preserved under their case directories. No scientific snapshots or restarts were generated. Compact initialization count maps, timing/audit JSON, resource samples, logs and selected profiler traces remain in this session. Storage size and running jobs are recorded in SESSION_STATE.md. Historical source snapshots and unrelated jobs remain preserved. No external messages or Mac mirror were requested.
'''
(ROOT/'reports/REPORT_AGENT.md').write_text(agent)
with PdfPages(ROOT/'reports/REPORT_code_efficiency.pending.pdf') as pdf:
 text=f'{len(valid)}/60 throughput cases complete; {len(profiles)} profiles.\nUpdated {NOW}\n\n'+'\n\n'.join(findings)+'\n\nSingle-pass evolution timings exclude initialization and a 0.5M warm-up. Diagnostic timings are measured separately. Phase profiles use the release binary; device-clock samples use a separate instrumented binary. Exact values, kernel names, definitions and provenance are in the Markdown report and CSV tables.'
 lines='\n'.join(textwrap.fill(line,88) if line else '' for line in text.split('\n')).splitlines()
 for page,offset in enumerate(range(0,len(lines),48)):
  fig=plt.figure(figsize=(8.3,11.7));fig.text(.08,.94,'NRPIC code efficiency — Paper Session 02'+(' (continued)' if page else ''),fontsize=14,va='top',usetex=False);fig.text(.08,.88,'\n'.join(lines[offset:offset+48]),fontsize=10,va='top',linespacing=1.4,usetex=False);pdf.savefig(fig);plt.close(fig)
 for name,caption in FILES:
  fig=plt.figure(figsize=(8.3,11.7));ax=fig.add_axes([.06,.32,.88,.6]);ax.imshow(plt.imread(ROOT/'figures'/f'{name}.png'));ax.axis('off');fig.text(.08,.28,'\n'.join(textwrap.wrap(caption,100)),fontsize=9,va='top',usetex=False);pdf.savefig(fig,dpi=220);plt.close(fig)
os.replace(ROOT/'reports/REPORT_code_efficiency.pending.pdf',ROOT/'reports/REPORT_code_efficiency.pdf')
# Keep live state truthful; no completion based only on reaching the last scheduler job.
queue=subprocess.run(['squeue','-u','jiaxiwu','-h','-o','%i %j %t %M %R'],text=True,stdout=subprocess.PIPE,check=False).stdout
jobs=[line for line in queue.splitlines() if 'eff-' in line]
size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts)
state=f'# Session state\n\nUpdated {NOW}. Completed throughput cases: {len(valid)}/60. Completed profiles: {len(profiles)}. Failed/incomplete throughput cases: {len(through)-len(valid)}.\n\nActive campaign jobs:\n\n'+ ('\n'.join('- '+line for line in jobs) or '- None')+f'\n\nSession path: {ROOT}\nRetained bytes excluding Git objects: {size:,}. No scientific snapshots/restarts; compact benchmark evidence and build/source artifacts retained.\nNext: '+('inspect failed cases and repair/retry without overwriting evidence.' if checks else ('complete queued runs and profiling, then verify final reports/figures and push closure.' if len(valid)<60 else 'verify figures, source gates, profiling coverage and final Git push before closure.'))+'\n'
(ROOT/'SESSION_STATE.md').write_text(state)
print(json.dumps(summary),flush=True)
