#!/usr/bin/env python3
"""Session007-style standalone 3D movie from immutable Session010 caches, t<=98."""
from __future__ import annotations
import argparse,csv,datetime as dt,hashlib,importlib.util,json,os,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def save(p,obj):
 Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def tol(t):return .500001*10**(np.floor(np.log10(abs(t)))-5) if t else 1e-12

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--session',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 args=ap.parse_args()
 assert os.environ.get('SLURM_JOB_ID'),'Slurm allocation required'
 s=args.session.resolve();out=args.out.resolve();ev=out/'evidence';ev.mkdir(parents=True,exist_ok=True)
 assert not (ev/'COMPLETE.json').exists(),'Preserve completed release'
 summary=s/'session010_summary_20261019';data=summary/'workspace/runtime_s10/analysis/through_segment002_20261018'
 native=summary/'workspace/runtime_s10';inputmeta=json.loads((data/'INPUTS.json').read_text())
 reductions=json.loads((data/'REDUCTIONS.json').read_text());joins=json.loads((data/'horizon_joins.json').read_text())
 strict=json.loads((data/'common_enclosure.json').read_text());common=float(strict['first_time'])
 assert abs(common-64.36874999995982)<1e-10
 manifest={}
 for line in (summary/'SHA256SUMS.txt').read_text().splitlines():
  digest,name=line.split(maxsplit=1);manifest[name.lstrip('*')]=digest
 inputs=[]
 def bind(path):
  digest=sha(path);rel=str(path.relative_to(summary))
  assert digest==manifest.get(rel),('Original source SHA mismatch',rel)
  inputs.append(dict(path=rel,bytes=path.stat().st_size,sha256=digest));return digest
 for name in ('INPUTS.json','REDUCTIONS.json','horizon_joins.json','common_enclosure.json','mesh.npy','cohort_tags.npy'):
  bind(data/name)
 # Recover native source case by persistent VTK file number. Validate this mapping
 # against cache count and native cycle intervals before using any AH overlay.
 vtk={};case_ends=[];case_seals=[]
 for case,root in enumerate(inputmeta['ordered_outputs']):
  local=native/Path(root.split('/runtime_s10/',1)[1]);seal=local/'SCIENCE_VERIFIED.json'
  bind(seal);sealobj=json.loads(seal.read_text());end=int(sealobj['native_checkpoint']['cycle'])
  case_ends.append(end);case_seals.append(str(seal.relative_to(summary)))
  for row in sealobj['files']:
   if row['path'].endswith('.part.vtk'):
    name=Path(row['path']).name;vtk[name]=dict(case=case,source=root+'/'+row['path'],native_sha256=row['sha256'])
 sources=[vtk[k] for k in sorted(vtk,key=lambda k:int(re.search(r'\.(\d+)\.part\.vtk$',k)[1]))]
 assert len(sources)==len(reductions['particle_times']),('VTK/cached source-count mismatch',len(sources),len(reductions['particle_times']))
 assert all(a<=b for a,b in zip(case_ends,case_ends[1:]))
 print('Validated',len(sources),'native VTK source identities',flush=True)
 mesh=np.load(data/'mesh.npy',allow_pickle=False)
 cohort=np.load(data/'cohort_tags.npy',allow_pickle=False).astype(np.int64)
 assert len(cohort)==90000 and len(np.unique(cohort))==90000
 assert [int(np.sum((cohort>=lo)&(cohort<hi))) for lo,hi in ((0,3000000),(3000000,4000000),(4000000,5000000))]==[40000,25000,25000]
 horizons=[]
 for hid in range(3):
  p=data/f'accepted_horizon_{hid}.csv';bind(p)
  rows=np.genfromtxt(p,delimiter=',',names=True,ndmin=1)
  assert all(not stats['unmatched'] and not stats['ambiguous'] for key,stats in joins.items() if key.endswith(':'+str(hid)))
  horizons.append(rows)
 oldpath=out/'reference/cca_render_gi_3d_movie.py'
 spec=importlib.util.spec_from_file_location('session007_style',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 for sub in ('frames','sequence','qa'): (out/sub).mkdir(exist_ok=True)
 records=[];previous=set(cohort.tolist())
 for i,printed in enumerate(reductions['particle_times']):
  if printed>98:continue
  path=data/'cache/particles'/f'{i:04d}.npz';digest=bind(path)
  with np.load(path,allow_pickle=False) as z:
   tm=float(z['time']);xyz=z['xyz'].copy();tags=z['tag'].astype(np.int64);groups=z['group'].copy()
  assert tm==printed and 0<=tm<=98 and np.isfinite(xyz).all()
  current=set(tags.tolist());assert len(current)==len(tags) and current<=previous
  assert np.array_equal(groups,np.where(tags<3000000,0,np.where(tags<4000000,1,2)))
  previous=current
  mi=np.flatnonzero(abs(mesh[:,0]-tm)<=tol(tm)+1e-10);assert len(mi)==1,(tm,mi)
  cycle=int(mesh[mi[0],1]);case=sources[i]['case']
  assert cycle<=case_ends[case] and (case==0 or cycle>=case_ends[case-1]),(tm,cycle,case,case_ends)
  ids=(0,1) if tm<common else (2,)
  spheres=[];missing=[]
  for hid in ids:
   rows=horizons[hid];match=rows[(rows['cycle']==cycle)&(rows['case']==case)]
   assert len(match)<=1,(hid,cycle,case)
   if not len(match):missing.append(dict(horizon=hid,reason='No accepted same-case/cycle surface'));continue
   row=match[0];assert abs(float(row['time'])-tm)<=tol(tm)+1e-10
   center=[float(row[k]) for k in ('center_x','center_y','center_z')];radius=float(row['meanradius'])
   assert np.isfinite(center).all() and np.isfinite(radius) and radius>0
   spheres.append(dict(horizon=hid,case=case,cycle=cycle,time=float(row['time']),center=center,mean_coordinate_radius=radius))
  buried=np.zeros(len(xyz),dtype=bool)
  for h in spheres:buried|=np.sum((xyz-np.asarray(h['center']))**2,axis=1)<(1.002*h['mean_coordinate_radius'])**2
  keep=(~buried)&(abs(xyz)<=old.VIEW_LIMIT).all(axis=1)
  fig=plt.figure(figsize=(16,9),dpi=120,facecolor=old.BACKGROUND)
  ax=fig.add_axes([0,0,1,1],projection='3d',facecolor=old.BACKGROUND)
  ax.set_proj_type('persp',focal_length=old.CAMERA['focal_length']);ax.view_init(elev=old.CAMERA['elev'],azim=old.CAMERA['azim'],roll=old.CAMERA['roll'])
  ax.set(xlim=(-24,24),ylim=(-24,24),zlim=(-24,24));ax.set_box_aspect((1,1,1));ax.set_axis_off()
  for group in (0,1,2):
   q=keep&(groups==group);style=old.GROUP_STYLES[group]
   ax.scatter(*xyz[q].T,s=style['size'],alpha=style['alpha'],color=old.GROUP_COLORS[group],edgecolors='none',linewidths=0,depthshade=False,rasterized=True)
  for h in spheres:
   ax.plot_surface(*old.make_sphere(tuple(h['center']),h['mean_coordinate_radius']),color='#000000',edgecolor='none',linewidth=0,antialiased=True,shade=False,alpha=1)
  frame=out/'frames'/f'state_{i:04d}.png'
  assert not frame.exists(),'Choose a new versioned output for a rerender'
  fig.savefig(frame,facecolor=old.BACKGROUND,edgecolor=old.BACKGROUND,pad_inches=0);plt.close(fig)
  records.append(dict(index=i,time=tm,cycle=cycle,case=case,cache=str(path.relative_to(s)),cache_sha256=digest,
    native_source=sources[i],frame=str(frame.relative_to(out)),frame_sha256=sha(frame),cohort_alive=len(tags),
    visible_counts=[int(np.sum(keep&(groups==g))) for g in range(3)],buried=int(buried.sum()),horizons=spheres,missing_surfaces=missing))
  print('FRAME',i,'t',tm,'case',case,'horizons',[h['horizon'] for h in spheres],flush=True)
 assert len(records)==50 and records[0]['time']==0 and records[-1]['time']==98
 # Exactly reproduce the old 0.25-Mref-per-frame pacing (12 fps = 3 Mref/s).
 # Zero-order hold on that clock; very short initial states may receive zero frames.
 times=np.asarray([r['time'] for r in records]);counts=np.zeros(len(records),dtype=int);schedule=[]
 for k in range(393):
  physical=k/4;j=int(np.searchsorted(times,physical,side='right')-1);rec=records[j];counts[j]+=1
  link=out/'sequence'/f'frame_{k:05d}.png';assert not link.exists();link.symlink_to(Path('..')/rec['frame'])
  schedule.append(dict(frame=k,playback_seconds=k/12,hold_clock_Mref=physical,actual_saved_time=rec['time'],source_index=rec['index'],cycle=rec['cycle'],case=rec['case']))
 for r,n in zip(records,counts):r['encoded_frames']=int(n)
 assert schedule[-1]['actual_saved_time']==98 and all(r['actual_saved_time']<=r['hold_clock_Mref']<=98 for r in schedule)
 with (out/'TIMING.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(schedule[0]));w.writeheader();w.writerows(schedule)
 save(out/'FRAMES.json',records);save(out/'SOURCE_MANIFEST.json',inputs)
 movie=out/'jeans_session010_particles3d_t0_to98.mp4';ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe');assert ffmpeg and ffprobe
 command=[ffmpeg,'-n','-v','error','-framerate','12','-i',str(out/'sequence/frame_%05d.png'),'-frames:v','393','-c:v','libx264','-threads','2','-preset','slow','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
 subprocess.run(command,check=True)
 with (ev/'full_decode.log').open('w') as f:
  subprocess.run([ffmpeg,'-v','error','-i',str(movie),'-map','0:v:0','-f','null','-'],stdout=f,stderr=subprocess.STDOUT,check=True)
 probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames,duration','-of','json',str(movie)],text=True))
 st=probe['streams'][0];assert st['codec_name']=='h264' and st['pix_fmt']=='yuv420p' and (st['width'],st['height'])==(1920,1080)
 assert st['r_frame_rate']=='12/1' and int(st['nb_read_frames'])==393 and abs(float(st['duration'])-32.75)<1e-5
 save(ev/'FFPROBE.json',probe)
 qa_indices=[]
 for target in (0,8,10,62,63.6312,66,98):
  j=int(np.argmin(abs(times-target)));assert abs(times[j]-target)<1e-5
  encoded=np.flatnonzero([r['source_index']==records[j]['index'] for r in schedule]);assert len(encoded)
  k=int(encoded[-1] if target==98 else encoded[0]);qa_indices.append((j,k))
  subprocess.run([ffmpeg,'-n','-v','error','-i',str(movie),'-vf',f'select=eq(n\\,{k})','-frames:v','1',str(out/'qa'/f'decoded_t{target:g}.png')],check=True)
 # Review the exact old standalone reference using the same local compute job.
 for k in (0,128,256):
  subprocess.run([ffmpeg,'-n','-v','error','-i',str(out/'reference/slide27_jeans_particles_3d.mp4'),'-vf',f'select=eq(n\\,{k})','-frames:v','1',str(out/'qa'/f'session007_{k:03d}.png')],check=True)
 contact=Image.new('RGB',(1280,4*390),(7,26,43));draw=ImageDraw.Draw(contact)
 for cell,(j,k) in enumerate(qa_indices):
  r=records[j];img=Image.open(out/'frames'/f"state_{r['index']:04d}.png").convert('RGB');img.thumbnail((640,360))
  x=(cell%2)*640;y=(cell//2)*390;contact.paste(img,(x,y+25));draw.text((x+8,y+5),f"saved t={r['time']:g}; AH {[h['horizon'] for h in r['horizons']]}; cycle {r['cycle']}",fill='white')
 contact.save(out/'qa/representative_frames.jpg',quality=95)
 ref=Image.new('RGB',(1280,720),(7,26,43))
 for n,path in enumerate((out/'qa/session007_000.png',out/'frames/state_0000.png')):
  img=Image.open(path).convert('RGB');img.thumbnail((640,360));ref.paste(img,(640*n,100))
 dr=ImageDraw.Draw(ref);dr.text((10,70),'Session007 exact reference',fill='white');dr.text((650,70),'Session010 matched appearance',fill='white');ref.save(out/'qa/reference_comparison.jpg',quality=95)
 source_last=np.asarray(Image.open(out/records[-1]['frame']).convert('RGB'),dtype=float)
 decoded_last=np.asarray(Image.open(out/'qa/decoded_t98.png').convert('RGB'),dtype=float)
 mae=float(np.mean(abs(source_last-decoded_last)));assert mae<3,mae
 metadata=dict(status='machine checks passed; visual review pending',job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
 source_commit=inputmeta['source_commit'],analysis_parent_commit=subprocess.check_output(['git','-C',str(s/'code'),'rev-parse','HEAD'],text=True).strip(),
 movie=movie.name,sha256=sha(movie),bytes=movie.stat().st_size,native_simulation_modified=False,ah_repair_implemented=False,
 data_time_range=[0,98],saved_states=50,encoded_frames=393,fps=12,duration_seconds=32.75,resolution=[1920,1080],
 pacing='3 Mref/playback second; CFR 0.25-Mref grid, hold most recent actual saved state. Final t=98 included for one frame. No interpolation.',
 zero_encoded_duration_states=[r['time'] for r in records if not r['encoded_frames']],
 geometry='Actual xyz, equal fixed +/-24 cube, standalone, axes/text hidden',camera=old.CAMERA,background=old.BACKGROUND,group_colors=old.GROUP_COLORS,group_styles=old.GROUP_STYLES,
 particles='Existing fixed 90k-tag cohort only; nested surviving tag sets checked; no replacement/resampling',
 ah_selection='Strict accepted consumer-matched CSV row with exact native cycle/case and printed-time tolerance; measured coordinate center and meanradius. Individuals before first strict common enclosure, common only after it; missing/rejected surfaces omitted without interpolation/extrapolation.',
 common_enclosure_time=common,first_rendered_common_time=min(r['time'] for r in records if any(h['horizon']==2 for h in r['horizons'])),
 sphere='Opaque black sphere schematic, not reconstructed AH shape; 0.2% interior suppression buffer matches Session007 renderer',
 cache_case_recovery='Persistent native VTK output file number, deduplicated later-case winner, validated 112-cache count and per-frame native checkpoint cycle bounds; cycle from matching mesh.npy row',
 references=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in (out/'reference').iterdir()],
 software=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,ffmpeg=ffmpeg,ffmpeg_sha256=sha(Path(ffmpeg))),
 encode_command=command,last_frame_source_time=records[-1]['time'],decoded_last_frame_mean_absolute_RGB_error=mae,
 full_decode_passed=True,no_later_cache_loaded=True,source_sha256_manifest=sha(summary/'SHA256SUMS.txt'))
 save(ev/'COMPLETE.json',metadata)
 (out/(movie.name+'.sha256')).write_text(metadata['sha256']+'  '+movie.name+'\n')
 for name in ('TIMING.csv','FRAMES.json','SOURCE_MANIFEST.json'):print(name,sha(out/name),flush=True)
 print('MOVIE MACHINE CHECKS PASSED',movie,flush=True)

if __name__=='__main__':main()
