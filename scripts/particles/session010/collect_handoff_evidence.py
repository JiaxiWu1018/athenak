#!/usr/bin/env python3
"""Collect small verification evidence and inventory raw preparation data in Slurm."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from checkpoint_io import atomic,digest

def main():
 assert os.environ.get('SLURM_JOB_ID')
 s=Path(__file__).resolve().parents[4];repo=s/'code';p=repo/'scripts/particles/session010/vista_package';e=p/'evidence'
 e.mkdir(exist_ok=True)
 def copy(f,target):
  f=Path(f);target=Path(target);assert f.is_file() and f.stat().st_size<16*1024**2
  assert f.suffix not in ('.rst','.vtk','.bin','.h5','.o','.a','.pyc')
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
 for name in ['build_receipt.json','focused_receipt.json','operations_receipt.json','compact_receipt.json','memory_receipt.json','mesh_options_receipt.json','gpu_amr_receipt.json','kokkos_history_receipt.json','historical_7093f488.patch','adapted_buffer_reuse.patch']:
  copy(s/'evidence'/name,e/name)
 complete=Path(json.loads((s/'evidence/compact_receipt.json').read_text())['root'])
 assert json.loads((complete/'COMPLETE.json').read_text())['passed']
 for f in complete.rglob('*'):
  if f.is_file() and (f.name in ['run.log','launch.json','result.json','memory.csv','COMPLETE.json','resumed_evidence.json','initial_validation.json','restart_comparison.json','matched_native_particle_comparison.json','memory_growth.json','matched_memory.png','matched_memory.pdf']):copy(f,e/'compact'/f.relative_to(complete))
 prior=Path(json.loads((complete/'resumed_evidence.json').read_text())['prior_root'])
 for case in ['reference','seed','restart_3rank']:
  for name in ['run.log','launch.json','result.json']:copy(prior/case/name,e/'prior_compact'/case/name)
 for name in ['baseline.configure.log','patched.configure.log','provenance.txt','baseline.source','patched.source','baseline.sha256','patched.sha256']:
  copy(s/'evidence/build_12753'/name,e/'build'/name)
 for f in (s/'evidence/build_repair_12803').iterdir():
  if f.name!='athena':copy(f,e/'build_repair'/f.name)
 for rel in ['focused_12766/crossing.tests.log','focused_12766/deposition.tests.log','feedback_12770/tests.log']:
  copy(s/'evidence'/rel,e/'focused'/Path(rel).name)
 copy(s/'evidence/operations.12811.log',e/'operations.log')
 amr=Path(json.loads((s/'evidence/gpu_amr_receipt.json').read_text())['root'])
 for f in amr.rglob('*'):
  if f.is_file() and f.name in ['run.log','launch.json','result.json','RESULT.json','memory.csv']:copy(f,e/'gpu_amr'/f.relative_to(amr))
 mesh=Path(json.loads((s/'evidence/mesh_options_receipt.json').read_text())['root'])
 copy(mesh/'RESULT.json',e/'mesh_options/RESULT.json')
 for f in mesh.rglob('*'):
  if f.is_file() and f.name in ['mesh.log','mesh_only.athinput']:copy(f,e/'mesh_options'/f.relative_to(mesh))
 for f in (s/'scripts').iterdir():
  if f.is_file():copy(f,e/'perseus_scripts'/f.name)
 copy(s/'inputs/VALIDATION_ONLY_compact.athinput',e/'VALIDATION_ONLY_compact.athinput')
 for n,rel in [('SESSION007_REPORT.md','session_007_part1_vista_handoff_20260908/REPORT.md'),('SESSION009_REPORT_AGENT.md','session_009_orbit_gw_production_20261005/REPORT_AGENT.md'),('SESSION009_REPORT_Jeans9.md','session_009_orbit_gw_production_20261005/REPORT_Jeans9.md'),('SESSION009_memory_remedy_review.json','session_009_orbit_gw_production_20261005/evidence/memory_remedy_review_20261012.json')]:copy(s.parent/rel,p/'reference_reports'/n)
 for name in ['TT_CONVENTION.md','wave_gate.json']:
  copy(s.parent/'session_009_orbit_gw_production_20261005/evidence'/name,p/'reference_reports'/('SESSION009_'+name))
 for f in (s/'handoff/reference_session007').rglob('*'):
  if f.is_file() and f.suffix in ['.py','.sh','.sbatch','.cmake']:copy(f,p/'reference_session007'/f.relative_to(s/'handoff/reference_session007'))
 for name,old,new in [('final_vs_baseline.patch','7c4f8b68','HEAD'),('final_vs_retained_vista.patch','263dcf21','HEAD')]:
  with (e/name).open('w') as out:subprocess.run(['git','-C',str(repo),'diff',old,new,'--','src','CMakeLists.txt'],stdout=out,check=True)
 # Exact counts/sizes and hashes of verification metadata, without reading raw field payloads.
 groups={};inventory=[]
 for top in ['evidence','inputs','scripts']:
  for f in (s/top).rglob('*'):
   if not f.is_file() or f.is_symlink():continue
   st=f.stat();relative=f.relative_to(s);group=str(relative.parts[1])
   g=groups.setdefault(top+'/'+group,dict(files=0,bytes=0));g['files']+=1;g['bytes']+=st.st_size
   inventory.append(dict(path=str(relative),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
 atomic(s/'evidence/RAW_INVENTORY.json',dict(root=str(s),job=os.environ['SLURM_JOB_ID'],generated_epoch=time.time(),groups=groups,files=inventory))
 atomic(e/'RAW_STORAGE_SUMMARY.json',dict(root=str(s),groups=groups,total_bytes=sum(r['bytes'] for r in inventory),status='Retained on Perseus; no deletion or new large-data transfer. Raw/checkpoints omitted from handoff.'))
 build=json.loads((s/'evidence/build_receipt.json').read_text())
 atomic(e/'COMPILED_PROVENANCE.json',dict(baseline_commit='7c4f8b68a0afbac013e880f425183dbc98ef35b0',patched_cpp_commit='80ef74255e0b94892b8ff976a160105132767ab1',kokkos_commit='6739bc623081648af9e752b616d9671527922cbf',baseline_executable=build['baseline'],baseline_sha256=digest(build['baseline']),patched_executable=build['patched'],patched_sha256=digest(build['patched']),host=os.uname().nodename))
 growth=json.loads((complete/'memory_growth.json').read_text());lines=['# Matched memory results','','Same native baseline checkpoint; 792 fixed MeshBlocks, five million particles, two MPI ranks, cap 400/rank on two allocated A100 80 GiB GPUs. Each arm advances 128 steps (dt=0.00625); fit advances 32–126, excluding startup/final output. Sampling every two seconds. Fits use pairwise median slopes and endpoints, rather than a sawtooth-sensitive ordinary least-squares fit.','','| Arm / device | Median growth MiB/min | Endpoint MiB/min | Fit minutes | Begin / end / peak MiB |','|---|---:|---:|---:|---:|']
 for arm,groups in growth['arms'].items():
  for device,v in groups.items():lines.append(f"| {arm} / {device} | {v['theil_sen_MiB_min']:.3f} | {v['endpoint_MiB_min']:.3f} | {v['fit_minutes']:.2f} | {v['begin_MiB']:.0f} / {v['end_MiB']:.0f} / {v['peak_MiB']:.0f} |")
 lines+=['','## Observed outcome','','Both baseline and patched GPU traces are flat at NVML MiB granularity over the matched ~20-minute fit windows. This CUDA test did not reproduce Session009’s HIP/MI210 GPU growth, so it cannot quantify a GPU-growth reduction or prove that all causes of that growth are removed. The patched host RSS retains a small positive drift: 0.0123 MiB/min median, 0.0321 MiB/min endpoint (~0.63 MiB over the window). The baseline host drift is similarly small; a percentage reduction of such a small drift is not strong causal evidence. The repair retains about 66/18 MiB more steady GPU allocation and ~55 MiB more host RSS here, consistent with high-water capacity. It is an implemented and numerically validated repair with unproved full-size/site memory feasibility.','','Device UUIDs and every sample are in the CSVs. A positive slope is residual growth, not a bounded-memory proof; a flat or negative short-window slope does not prove zero longer-term growth. High-water buffers can retain more steady-state memory while reducing allocation churn. This CUDA/UCX fixed-mesh result does not measure MI210/HIP or Grace/H200 behavior, nor future AMR/collapse allocations. Historical Singularity growth reductions were 54% GPU and 88.4% host; do not substitute them for these measurements.','','Both final native checkpoints retain all five million unique tags, exact GIDs/rest weights and the same mesh; tag-sorted real-state comparison uses rtol=1e-10/atol=1e-12. The exact outcome is in matched_native_particle_comparison.json.','','## Safe segmentation','','No numerical segment length is certified for full-size Vista by this compact test. Measure each GPU restart baseline, host RSS, active/reserved blocks, AMR maxima and residual growth with the final native executable. Select a finite short first segment after discussion, and reserve the measured checkpoint/manifest wall time and allocation headroom. For a conservative positive growth estimate g in GiB/min and remaining headroom H to the agreed memory guard, use T <= min(agreed wall budget minus finalization reserve, 0.5 H/g); also enforce node-hour/segment/calendar/scientific-time ceilings. Use the worst device and add AMR and checkpoint reserves. If g is unresolved or changes during collapse, use a short measured bound rather than an unlimited extrapolation. Stop cleanly before the guard, verify actual native checkpoint progress/hash/accounting and reassess on restart.','','The preparation uses an 85% guard only as a bounded test safeguard; Vista must explicitly agree its own limits. The 512-step draft was shortened to 128 matched advances after measuring compact timing to keep each arm within one hour. No full historical validation campaign was repeated.']
 (p/'reports/MEMORY_RESULTS.md').write_text('\n'.join(lines)+'\n')
 total=sum(r['bytes'] for r in inventory);(p/'reports/ARCHIVE_STATUS.md').write_text(f'''# Perseus preparation storage and closeout\n\nRaw validation data and native checkpoints remain under `{s}/evidence/`. Inventory: `{s}/evidence/RAW_INVENTORY.json`; grouped sizes are in evidence/RAW_STORAGE_SUMMARY.json. This snapshot totals {total:,} bytes ({total/2**30:.2f} GiB) across evidence/inputs/scripts, including retained build objects and failed attempts. Source clones are separate source snapshots. No raw-data deletion or new large-data transfer was performed, and earlier sessions are preserved.\n\nPerseus /data had approximately 11 TiB free at the compact allocation, 91% filesystem use. It is an authoritative archive, not a second-site backup; retention/cleanup remains explicit. The transfer contains only small test evidence, scripts, source bundles, exact inputs and context, without any restart/VTK/field payload.\n\nPreparation jobs include builds 12753/12803, focused 12766/12770, final operations 12811, exact geometry comparison 12809, compact 12805 and reduction 12807. The packaging audit records its own allocation in PACKAGE_RECEIPT.json. The final root closeout records any remaining queued/running jobs; no production or Vista job is launched here. Perseus Slurm accounting is not configured for sacct historical usage, so no invented usage total is reported; retained allocation/job records and application times provide the preparation provenance.\n''')
 atomic(s/'evidence/collection_receipt.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],raw_bytes=total))
 print('SMALL EVIDENCE COLLECTED',total)
if __name__=='__main__':main()
