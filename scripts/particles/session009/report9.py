"""Durable milestone reports using verified science and scheduler records."""
import json,os,subprocess,time
from pathlib import Path
from runtime import atomic

def accounting(path):
 actual=0.;rows=[]
 if path.exists():
  for line in path.read_text().splitlines():
   r=line.split('|')
   if len(r)>=5 and r[0].isdigit():
    cost=float(r[3])*int(r[4])/3600;actual+=cost;rows.append(dict(job=int(r[0]),state=r[1],exit=r[2],elapsed_seconds=int(r[3]),nodes=int(r[4]),raw_node_hours=cost))
 return actual,rows

def write_reports(root,out,metrics,milestone):
 state=json.loads((root/'evidence/amd_state.json').read_text());config=json.loads((root/'evidence/config.json').read_text())
 nodeh,jobs=accounting(root/'evidence/amd_accounting.psv')
 storage=int(subprocess.check_output(['du','-sx','--block-size=1',str(root)],text=True).split()[0])
 ledger_path=root/'evidence/active_archive_jobs.json';anta_jobs=json.loads(ledger_path.read_text())['jobs'] if ledger_path.exists() else []
 anta_actual=None;anta_accounting=''
 if anta_jobs:
  ids=','.join(str(j['id']) for j in anta_jobs)
  anta_accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',ids,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes'],text=True)
  (out/'anta_accounting.psv').write_text(anta_accounting)
  anta_actual=sum(float(r[3])*int(r[4])/3600 for r in (l.split('|') for l in anta_accounting.splitlines()) if len(r)>=5 and r[0].isdigit())
 record=dict(milestone=milestone,generated_utc=time.time(),state=state,AMD_raw_node_hours=nodeh,AMD_jobs=jobs,Anta_node_hours=anta_actual,Anta_GPU_hours_equal_node_hours=True,Anta_reserved_jobs=len(anta_jobs),Anta_maximum_reserved_node_hours=4*len(anta_jobs),archive_bytes=storage,monetary_cost=None,tariff='No account monetary tariff provided; no invented dollar cost.',compiled_commit=config.get('compiled_source_commit'),input_sha256=config.get('input_sha256'))
 # Recent measured production cost, excluding gates and analysis; estimates stay labeled.
 segment_names={j['id']:j['name'] for j in state['jobs'] if j['name'].startswith('segment')}
 production_nodeh=sum(j['raw_node_hours'] for j in jobs if j['job'] in segment_names)
 covered=float(state.get('time',0));rate=production_nodeh/covered if covered>0 and production_nodeh else None
 snapshots=[]
 for run in (root/'runs').glob('*'):
  p=run/'storage_status.json'
  if p.exists():snapshots.append(json.loads(p.read_text()))
 record['AMD_last_allocated_storage_sample']=max(snapshots,key=lambda r:r['utc']) if snapshots else None
 record['mean_production_node_hours_per_M']=rate;record['estimated_remaining_node_hours_to_t400']=rate*(400-covered) if rate else None
 atomic(out/'resources.json',record)
 checkpoint_text=f"t={state['checkpoint']['time']:.8g} M_ref" if state.get('checkpoint') else 'not available yet'
 report=f'''# Jeans-in-cluster Session 009 — update {milestone}

Status: **{state['status']}**. Last production checkpoint accepted by the controller: **{checkpoint_text}**. Analysis uses checksum-verified Anta data and may lag AMD. Generated UTC epoch: {record['generated_utc']}.

Fresh t=0; local Lorentz boost0.133215, left−y/right+y. Approximate companion support is not exact GR equilibrium. Five million initial particles; source masses .76/.12/.12, envelope arealR30, clump centers±3, sigma.70, thermal spread.02 and seed4001. K_ij≈0 leaves the local momentum constraint unsolved. M_ref=1 is the inherited unit, not a measured horizon/rest/ADM mass. Sampling residuals are retained.

Domain±1024, root256³ dx8, blocks32³, physical refinement0..11 with finestdx1/256. Continuous radial wave floor dx≤.25 to46; outer dx.5/1/2/4 to72/104/168/296. Actual complete startup mesh is in evidence/mesh_audit.json. RK4/CFL.4, inherited gauge/deposition/pusher; remove alpha<.05, AH removal OFF, tracker_floor=false.

Single coordinate extractionr40 saves complex rPsi4, ell2..8 every.025. Particles every.25, planes.1, full3D/checkpoints10 plus final stops. Intended f≤.4 cycles/M_ref depends on the linear propagation gate; cells/wavelength is not convergence evidence. One radius cannot establish radius dependence or extrapolate to infinity. t-r is an approximate coordinate retarded time.

## Available findings

Both individual horizons formed: {metrics.get('both_individual_horizons_formed')}. Shared formation time: {metrics.get('both_formation_time')}. Measured post-formation coordinate revolutions: {metrics.get('post_formation_revolutions')}; live intervals: {metrics.get('post_formation_live_intervals')}. Strict accepted common rows proven to enclose both live tracked centers: {metrics.get('accepted_enclosing_common_rows',0)}. See ringdown_review.json for usable signal and tail tests. A shrinking separation alone is not a radiation-driven inspiral claim. Surviving-particle centers can change under particle removal; coordinate distances and phase are not gauge-invariant orbital elements.

## Resources and limits

AMD measured raw node-hours: **{nodeh:.6f}**, including recorded builds, gates, failures and inspectors. Anta measured allocated node-hours/GPU-hours: **{anta_actual}**; {len(anta_jobs)} of96 archive/analysis jobs submitted, each at most4h. Monetary cost is unavailable without the actual account tariff. Anta archive size: **{storage/1024**4:.4f} TiB** (cap16TiB). AMD campaign cap1.25TiB plus whole-user warn1.5/stop1.7/projected1.9TiB. Latest three verified AMD checkpoints remain; older copies are removed only after verified Anta archival.

Recent production estimate: {rate} raw nodeh/M_ref; estimated remaining to400: {record['estimated_remaining_node_hours_to_t400']} nodeh. These are measured-history extrapolations, not a guaranteed completion date. Hard termination: t400,10000 raw AMD nodeh or45calendar days. Earlier success needs strict accepted enclosing common horizon, usable supported-band ringdown and at least100 saved units after its outgoing peak; inspect the receipt before interpreting completion.

## Review products

'''
 if not metrics:
  start=report.index('## Available findings')
  end=report.index('## Resources and limits')
  report=report[:start]+'## Available findings\n\nNo verified full-particle evolution data are available yet. Separation, orbital motion, horizons and gravitational-wave findings are unknown. No scientific plots or movies have been produced. Startup mesh, memory and restart measurements remain pending; empty evidence files are placeholders.\n\n'+report[end:]
 initial_path=root/'evidence/initial_validation.json'
 if initial_path.exists():
  initial=json.loads(initial_path.read_text())
  if initial.get('logged'):
   report+='## Measured initialization\n\n| Component | Count | Sampled rest mass | Sum(mW) | Pcov,y | Jcov,z |\n|---|---:|---:|---:|---:|---:|\n'
   for name,r in zip(('envelope','left','right'),initial['logged']):report+=f"| {name} | {r['count']} | {r['rest_mass']:.10g} | {r['sum_mW']:.10g} | {r['P_cov'][1]:.9g} | {r['J_origin_cov'][2]:.9g} |\n"
   report+='\nTotal covariant momentum residual: '+str(initial['total_P_cov'])+'; cancellation residual fraction: '+str(initial.get('cancelling_linear_momentum_residual_fraction'))+'. No symmetry or recoil was imposed. Source/model masses, sampled rest masses, horizon masses and an ADM estimate are distinct; no ADM estimate is assumed.\n\n'
 report+='Movie colors: gray envelope, blue left, orange right, green accepted common horizon. Six display frames/sec; particle sampling every0.25M plus restart/final extras, with actual time on each frame. Binary phase/revolutions stop at a proven accepted common horizon or unresolved coincidence; separation of surviving tagged matter remains a distinct diagnostic. Internal mesh-transition reflections are a possible waveform limitation even though the outer boundary is causally buffered; the linear gate is not nonlinear convergence evidence.\n\n'
 captions=[('orbit.png','Coordinate trajectories, separation and phase within contiguous valid intervals.'),('separation/separation_vs_time.png','Full surviving-particle centers, live trackers and simultaneous accepted AH centers versus time; gaps remain visible.'),('radial_tangential.png','Radial/tangential relative motion; small-denominator and diagnostic gaps are masked.'),('accepted_horizons.png','Strict accepted masses/spins; rejected or stale surfaces are excluded.'),('removal_history.png','Component removal counts and covariant removed-matter Jz, displayed separately from horizons.'),('particle_components.png','Component counts and covariant matter Jz; no removed-particle-plus-horizon conserved sum.'),('constraints.png','Inherited proper-volume chi mask, not an AH exterior; norms over differing empty volumes are not comparable.'),('raw_waveform.png','Raw (2,±2) at r40; early transients remain visible.'),('wave_phase_frequency.png','Wave phase/frequency and optional modes; no phase is unwrapped across timestamp gaps.'),('strain_cutoff_sensitivity.png','Conditional finite-radius fixed-frequency strain; cutoff sensitivity and conventions in strain_method.json.'),('central.mp4','Tagged central projection with tracks and accepted rmin illustrations.'),('context.mp4','Envelope/context tagged projection.'),('density.mp4','Full-particle rest weights on fixed Cartesian bins, avoiding native-AMR seams.')]
 for file,caption in captions:
  if (out/file).exists():report+=f'- `{out/file}` — {caption}\n'
 report+='\nMovies use a deterministic rendering cohort only; simulation count is unchanged. Density uses every surviving particle, |z|<.7 slab,256² bins and fixed color limits; it is coordinate rest-mass surface density, not proper energy density. Accepted rmin circles are illustrations, not complete horizon surfaces. Representative plot/frame visual review remains pending unless separately recorded. Raw complex multipoles are primary evidence. No unsupported merger/inspiral/ringdown conclusion is inferred.\n'
 (out/'REPORT_Jeans9.md').write_text(report);(root/'REPORT_Jeans9.md').write_text(report)
 technical='# Session 009 technical update\n\nExact compiled revision/configuration, tested receipts and accounting follow. Source/submodule/build/toolchain/executable hashes are in evidence/build_provenance.txt and evidence/*.sha256. Checkpoint/run manifests are under runs/. Numerical failures halt for review.\n\n```json\n'+json.dumps(record,indent=2)+'\n```\n\nReproduce on an Anta allocated compute node:\n\n```sh\n/home/jiaxiwu/miniconda3/bin/python '+str(root/'scripts/analyze_s9.py')+' --root '+str(root)+' --milestone '+str(milestone)+'\n```\n'
 (out/'REPORT_AGENT.md').write_text(technical);(root/'REPORT_AGENT.md').write_text(technical)
 (out/'CAPTIONS.md').write_text('\n'.join(f'{file}: {caption}' for file,caption in captions)+'\n')
 atomic(root/'analysis/latest.json',dict(path=str(out),milestone=milestone,generated_utc=time.time()))
