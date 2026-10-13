#!/usr/bin/env python3
"""Bounded validation-only 5M-particle before/after, restart and memory checks."""
import argparse,csv,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np
from checkpoint_io import checkpoint as native_checkpoint,digest,atomic

def checkpoint(path):
 return native_checkpoint(path,expected_root=1)
import compare_restart_s7 as compare
import validate_initial_s7 as initial

def run(exe,deck,root,label,ranks,args,limit=3600):
 d=root/label;d.mkdir();log=d/'run.log'; rows=[]
 # NVML indices are cgroup-local on some allocations and global on others.
 # Identify allocated devices once, then sample their immutable UUIDs.
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
 visible=[tuple(z.strip() for z in row.split(',')) for row in raw.splitlines()]
 allocation=os.environ.get('SLURM_JOB_GPUS','');count=int(os.environ['SLURM_GPUS_ON_NODE'])
 if len(visible)==count:
  selected=visible
 else:
  allowed=set(allocation.split(','));selected=[row for row in visible if row[0] in allowed]
 if len(selected)!=count:raise RuntimeError(('Cannot identify allocated NVML GPUs',visible,allocation,count))
 gpuids=','.join(row[1] for row in selected)
 cmd=['mpirun','--bind-to','none','--map-by','slot','-np',str(ranks),str(exe),'--kokkos-map-device-id-by=mpi_rank','-i',str(deck),'-d',str(d),'-t','00:55:00']+args
 atomic(d/'launch.json',dict(command=cmd,host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],exe_sha256=digest(exe),input_sha256=digest(deck),sample_global_gpu_ids=gpuids,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
 start=time.time()
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=d,stdout=f,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   now=time.time()
   with log.open('rb') as samplelog:
    samplelog.seek(max(0,log.stat().st_size-32768));tail=samplelog.read().decode(errors='replace')
   cs=re.findall(r'cycle\s*=\s*(\d+)',tail);cycle=int(cs[-1]) if cs else -1
   out=subprocess.check_output(['nvidia-smi','--id='+gpuids,'--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
   for line in out.splitlines():
    ix,uuid,used,total=[z.strip() for z in line.split(',')]
    rows.append(dict(epoch=now,elapsed=now-start,gpu=ix,uuid=uuid,used_MiB=float(used),total_MiB=float(total),kind='gpu',cycle=cycle))
    if float(used)/float(total)>=.85:(d/'REQUEST_STOP').touch()
   ps=subprocess.check_output(['ps','-eo','pid,ppid,rss,comm'],text=True)
   entries=[x.split(None,3) for x in ps.splitlines()[1:]];desc={proc.pid}
   for _ in range(12):
    nxt={int(x[0]) for x in entries if int(x[1]) in desc};old=len(desc);desc|=nxt
    if len(desc)==old:break
   rss=sum(int(x[2])/1024 for x in entries if int(x[0]) in desc and x[3]=='athena')
   rows.append(dict(epoch=now,elapsed=now-start,gpu='total',uuid='',used_MiB=rss,total_MiB=0,kind='host',cycle=cycle))
   if now-start>limit:(d/'REQUEST_STOP').touch()
   if now-start>limit+180:proc.terminate();raise RuntimeError('validation process exceeded finalization reserve')
   time.sleep(2)
 rc=proc.returncode
 with (d/'memory.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['epoch','elapsed','gpu','uuid','used_MiB','total_MiB','kind','cycle']);w.writeheader();w.writerows(rows)
 text=log.read_text(errors='replace');cps=sorted((d/'rst').glob('*.rst'))
 if rc!=0 or '### FATAL ERROR' in text or '[conservation OK]' not in text:raise RuntimeError('failed '+label)
 cp=checkpoint(cps[-1]) if cps else None
 atomic(d/'result.json',dict(exit_code=rc,elapsed_seconds=time.time()-start,checkpoint=cp,log_sha256=digest(log)))
 return d,cp

def compare_runs(a,b,source,target):
 sys.path.insert(0,str(source/'vis/python'));import bin_convert
 report={}
 for kind in ('z4c_xy','con_xy','tmunu_xy'):
  x=bin_convert.read_binary(str(compare.latest(a,'*.'+kind+'.*.bin')))
  y=bin_convert.read_binary(str(compare.latest(b,'*.'+kind+'.*.bin')))
  report[kind]=compare.field_comparison(x,y,1e-5,1e-8)
  del x,y
 x=compare.read_particles(compare.latest(a,'*.part.vtk'));y=compare.read_particles(compare.latest(b,'*.part.vtk'))
 ox=np.argsort(x[3]);oy=np.argsort(y[3]);assert x[0]==y[0]
 assert np.array_equal(x[3][ox],np.arange(5_000_000)) and np.array_equal(x[3][ox],y[3][oy])
 report['particles']={name:float(np.max(np.abs(x[i][ox]-y[i][oy]))) for i,name in ((1,'position'),(2,'momentum'),(4,'mass'))}
 assert all(v==0 for v in report['particles'].values()),report['particles']
 report['trackers']={}
 for i in (0,1):
  u,v=compare.last_tracker(a,i),compare.last_tracker(b,i)
  assert u[0]==v[0] and abs(u[1]-v[1])<1e-12
  cols=list(range(2,13))+[16]
  delta=max(abs(u[z]-v[z]) for z in cols)
  assert delta<5e-7,(i,delta)
  assert all(np.isfinite(v[z]) for z in range(2,17)) and v[13]>0 and v[14]>0
  report['trackers'][str(i)]=dict(persistent_max_abs=delta,live_reacquired=True,cycle=int(v[0]),time=v[1])
 atomic(target,dict(passed=True,comparison=report))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--session',type=Path,required=True);ap.add_argument('--resume-from',type=Path);a=ap.parse_args();s=a.session.resolve()
 builds=json.loads((s/'evidence/build_receipt.json').read_text());base,patch=Path(builds['baseline']),Path(builds['patched'])
 root=s/'evidence'/('compact_'+os.environ['SLURM_JOB_ID']);root.mkdir()
 deck=s/'inputs/VALIDATION_ONLY_compact.athinput'
 common=['mesh_refinement/max_nmb_per_rank=400','time/stop_check_every=1','z4c/co_0_out_every=1','z4c/co_1_out_every=1','z4c/tracker_walk_every=1']
 if a.resume_from:
  prior=a.resume_from.resolve()
  ref=prior/'reference';seed=prior/'seed'
  refcp=json.loads((ref/'result.json').read_text())['checkpoint']
  seedcp=json.loads((seed/'result.json').read_text())['checkpoint']
  assert seedcp['header_bytes']>40960 and refcp['cycle']==2
  assert 'Total number of MeshBlocks = 4936' in (prior/'production_mesh/mesh.log').read_text()
  atomic(root/'resumed_evidence.json',dict(prior_root=str(prior),reference=refcp,baseline_seed=seedcp,reason='Repaired SetReal precision; reuse accepted field/particle/reference evidence.'))
 else:
  md=root/'production_mesh';md.mkdir()
  subprocess.run([str(patch),'-i',str(s/'inputs/gi_cluster_s10.athinput'),'-m','mesh_refinement/max_nmb_per_rank=6000'],cwd=md,stdout=(md/'mesh.log').open('w'),stderr=subprocess.STDOUT,check=True)
  assert 'Total number of MeshBlocks = 4936' in (md/'mesh.log').read_text()
  ref,refcp=run(patch,deck,root,'reference',2,common+['time/nlim=2','time/stop_file='+str(root/'reference/REQUEST_STOP')])
  seed,seedcp=run(base,deck,root,'seed',2,common+['time/nlim=1','time/stop_file='+str(root/'seed/REQUEST_STOP')])
 # A new seed is necessary: precision lost in historical headers is unrecoverable.
 fullseed,fullcp=run(patch,deck,root,'seed_full_precision',2,common+['time/nlim=1','time/stop_file='+str(root/'seed_full_precision/REQUEST_STOP')])
 assert fullcp['header_bytes']>40960
 restart,restartcp=run(patch,deck,root,'restart_3rank',3,common+['-r',fullcp['path'],'time/nlim=2','time/stop_file='+str(root/'restart_3rank/REQUEST_STOP')])
 compare_runs(ref,restart,s/'code',root/'restart_comparison.json')
 # Stop marker is checked after a bounded advance; the native checkpoint records actual progress.
 # run() exclusively creates the directory, so retain the prewritten marker in a separate path.
 marker=root/'CLEAN_STOP_REQUEST';marker.touch()
 clean,cleancp=run(patch,deck,root,'clean_stop_run',3,common+['-r',restartcp['path'],'time/nlim=4','time/stop_file='+str(marker)])
 assert 'Terminating on clean stop-file request' in (clean/'run.log').read_text()
 logged=initial.parse_ledger(ref/'run.log');written,ledger=initial.written_particle_stats(ref,profile_name='gi_profile_M076_two_clump_s10.txt',domain_half_extent=64)
 agreement,_=initial.ledger_agreement(logged,ledger)
 checks=dict(inside_domain=written['inside_domain'],centers=all(np.max(np.abs(np.asarray(x['position_mean'])-initial.CENTERS[x['component']]))<.03 for x in written['components'][1:]),widths=all(all(.60<z<.80 for z in x['position_std']) for x in written['components'][1:]),thermal=all(all(.019<z<.021 for z in x['rest_frame_thermal_std']) for x in written['components'][1:]),counts=[x['count'] for x in logged]==[3_000_000,1_000_000,1_000_000],tags=written['exact_unique_tag_range'],finite=written['finite'],positive_weights=written['positive_rest_weights'],ledger=agreement,boost_signs=logged[1]['P_hat'][1]<0<logged[2]['P_hat'][1],positive_Jz=all(x['J_origin_cov'][2]>0 for x in logged[1:]))
 assert all(checks.values()),checks
 atomic(root/'initial_validation.json',dict(passed=True,checks=checks,written=written,logged=logged))
 # Both arms start from the same baseline seed, suppress bulk output and freeze AMR.
 disable=['output'+str(i)+'/dt=-1' for i in range(1,22) if i not in (1,5,7)]
 memargs=common+disable+['-r',seedcp['path'],'time/nlim='+str(seedcp['cycle']+128),'mesh_refinement/refinement=static','z4c/nrad_wave_extraction=0','fastflow/start_time_0=400','fastflow/start_time_1=400','fastflow/start_time_2=400']
 arms={}
 for name,exe in (('memory_baseline',base),('memory_patched',patch)):
  d,cp=run(exe,deck,root,name,2,memargs+['time/stop_file='+str(root/name/'REQUEST_STOP')])
  assert cp['cycle']==seedcp['cycle']+128,(name,cp)
  assert cp['particles']==5_000_000 and cp['blocks']==seedcp['blocks']
  arms[name]=dict(run=str(d),checkpoint=cp)
 # Compare native records by immutable tag; GPU atomic fill order is not physical identity.
 cps=[Path(arms[k]['checkpoint']['path']) for k in ('memory_baseline','memory_patched')]
 n=5_000_000;native=[]
 for cp in cps:
  data=np.memmap(cp,dtype='<f8',mode='r',offset=cp.stat().st_size-n*10*8,shape=(10,n))
  order=np.argsort(data[9]);assert np.array_equal(data[9,order],np.arange(n))
  native.append((data,order))
 x,ox=native[0];y,oy=native[1];errors={}
 assert np.array_equal(x[8,ox],y[8,oy]),'native GID mismatch'
 assert np.array_equal(x[0,ox],y[0,oy]),'native rest-weight mismatch'
 for col in range(1,8):
  u=np.asarray(x[col,ox]);v=np.asarray(y[col,oy]);errors[str(col)]=float(np.max(np.abs(u-v)))
  assert np.isfinite(u).all() and np.isfinite(v).all()
  assert np.allclose(u,v,rtol=1e-10,atol=1e-12),('native state',col,errors[str(col)])
 atomic(root/'matched_native_particle_comparison.json',dict(passed=True,tag_sorted=True,exact_tags_GIDs_weights=True,max_abs_by_real_slot=errors,rtol=1e-10,atol=1e-12))
 atomic(root/'COMPLETE.json',dict(passed=True,root=str(root),initial=checks,seed_checkpoint=seedcp,arms=arms,production_blocks=4936,memory_advances=128,memory_warmup_advances=32,full_production_GPU_gate_deferred_to_Vista=True))
 atomic(s/'evidence/compact_receipt.json',dict(passed=True,root=str(root)))
 print('SESSION010 COMPACT CHECKS PASSED',flush=True)
if __name__=='__main__':main()
