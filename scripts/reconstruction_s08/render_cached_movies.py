"""Bounded, atomic final movie rendering from a verified five-case review packet."""
import argparse
from PIL import Image
from common import *
import movies as m

p=argparse.ArgumentParser();p.add_argument('mode',choices=['configure','render','encode','close']);p.add_argument('--name');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=100);a=p.parse_args();job()
e=ROOT/'evidence/movie_cache';e.mkdir(parents=True,exist_ok=True)
assert load(ROOT/'evidence/ASSEMBLY_READY.json')['all_original_endpoints']==300
assert not (ROOT/'evidence/RESOURCE_STOP.json').exists()
source=[receipt(Path(m.__file__)),receipt(Path(__file__)),receipt(Path(m.style.__code__.co_filename))]
inputs=[ROOT/'evidence/SOURCE_MAP.json',ROOT/'evidence/ASSEMBLY_READY.json']
for c in CASES:
 d=ROOT/'reduced'/c['id'];inputs += [d/x for x in ['summary.json','REDUCTION_OK.json','cohort_index.csv','history.csv','shells.csv']]
 inputs += [ROOT/'evidence/samples'/f"{c['id']}.json"]
 if c['unstable']:inputs += [d/'ah.csv']
bindings=[receipt(x) for x in inputs]
if a.mode=='configure':
 assert not (e/'CONFIG.json').exists()
 ds=[m.load_case(c) for c in CASES[::-1]];limits=m.shell_limits(*(d['shell'] for d in ds))
 cfg=dict(utc=now(),job=job(),inputs=bindings,source=source,shell_limits=limits,cases={d['case']['id']:dict(times=d['times'].tolist(),spatial_window=m.spatial_window(d)) for d in ds},synchronized_times=m.np.arange(0,300.0000001,.5).tolist(),fps=15,scope='Finalfivecasefixed-IDcentral-slicemovies withsharedshelllimits; actualencodedartifactacceptanceseparate')
 atomic(e/'CONFIG.json',cfg);print('Configured finalmovie windows and sharedlimits',limits);exit()
cfg=load(e/'CONFIG.json');assert cfg['inputs']==bindings and cfg['source']==source
if a.mode=='close':
 products=[load(e/f'{name}_ENCODED.json') for name in [c['id'] for c in CASES[::-1]]+['s08_comparison']]
 assert len(products)==6 and all(d['full_decode'] for d in products)
 for d in products:assert receipt(Path(d['path']))=={k:d[k] for k in ['path','bytes','sha256']}
 atomic(ROOT/'evidence/MOVIES_RENDERED.json',dict(utc=now(),job=job(),products=products,config=receipt(e/'CONFIG.json'),visual_review='pending actualencodedinitial/middle/finalframes',source=source))
 (ROOT/'reports/MOVIES.md').write_text('''# Session 8 movies

Five individual movies and one synchronized comparison,15fps. Fixed20,000 original particle IDs (tags divisibleby100), immutable initial-radius colors r0/Riso(0), common labeled0–1color scale. Only the instantaneous central slice |z|<0.05Riso(0) is displayed; alive count refers to the wholecohort. The dotted circle is the initial isotropic surface. Spatial axes remain fixed per model across time and include all displayed particles with a5%margin; spatial scales differ between models and carry M units. Shellpanels share the common allfive radius range. Synchronized panels use the latestcompleted cohort sample, print its actualtime, and clear/mark a panel beyond its runendpoint. No interpolation or uncomputed evolution is shown. Black contours wouldshow only contemporaneous accepted AH equators; nonewere accepted in the threeenabled runs and the twostablefinders were disabled. Rejected/stale surfacesare absent. Every frame fullydecoded; encoded initial/middle/final frames and source/encoder comparisons are retained. Actual visual acceptance is recorded separately.
''');print('Closed sixmovie encoding receipts');exit()
assert a.name in [c['id'] for c in CASES]+['s08_comparison'];name=a.name
folder=ROOT/'frames'/name;folder.mkdir(parents=True,exist_ok=True)
comparison=name=='s08_comparison';times=cfg['synchronized_times'] if comparison else cfg['cases'][name]['times'];n=len(times)
if a.mode=='render':
 m.style();ds=[m.load_case(c) for c in CASES[::-1] if comparison or c['id']==name]
 for d in ds:d['spatial_window']=cfg['cases'][d['case']['id']]['spatial_window'];d['shell_limits']=cfg['shell_limits']
 if comparison:
  fig=m.plt.figure(figsize=(16,7.5),layout='constrained');grid=fig.add_gridspec(3,5,height_ratios=[2,1,1]);states=[];spatial=[]
  for j,d in enumerate(ds):
   axes=[fig.add_subplot(grid[k,j]) for k in range(3)];states.append((d,m.panel(fig,axes,d)));spatial.append(axes[0])
   if j:
    for ax in axes:ax.set_ylabel('')
  cb=fig.colorbar(states[0][1]['scatter'],ax=spatial,fraction=.015,pad=.01,ticks=[0,.5,1])
 else:
  d=ds[0];fig=m.plt.figure(figsize=(10,6),layout='constrained');grid=fig.add_gridspec(2,2,width_ratios=[1.1,1]);axes=[fig.add_subplot(grid[:,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,1])];state=m.panel(fig,axes,d);states=[(d,state)];cb=fig.colorbar(state['scatter'],ax=axes[0],fraction=.035,pad=.025,ticks=[0,.5,1]);fig.suptitle('Fixed particle IDs • initial-radius colors • central slice',fontsize=11)
 cb.set_label(r'Initial radius $r_0/R_{\rm iso}(0)$');fig.canvas.draw();fig.canvas.draw();rows=[]
 for i in range(a.start,min(a.stop,n)):
  assert not (ROOT/'evidence/RESOURCE_STOP.json').exists()
  target=folder/f'{i:06d}.png'
  if target.exists():
   with Image.open(target) as image:image.verify()
  else:
   for d,panel in states:m.update(d,panel,float(times[i]),comparison)
   if comparison:fig.suptitle(f'Session 8 • synchronized time t/M = {times[i]:.1f}',fontsize=12)
   temp=folder/f'{i:06d}.{job()}.tmp.png';fig.savefig(temp,dpi=120,facecolor='white');temp.replace(target)
  rows.append(dict(frame=i,time=float(times[i]),**receipt(target)))
 m.plt.close(fig);atomic(e/f'{name}_BATCH_{a.start:06d}_{min(a.stop,n):06d}.json',dict(utc=now(),job=job(),config=receipt(e/'CONFIG.json'),files=rows,atomic_frame_writes=True));print(name,a.start,min(a.stop,n));exit()
files={x['frame']:x for f in sorted(e.glob(name+'_BATCH_*.json')) for x in load(f)['files']};assert set(files)==set(range(n))
for i,item in files.items():assert receipt(Path(item['path']))=={k:item[k] for k in ['path','bytes','sha256']}
path=ROOT/'movies'/f'{name}.mp4';assert not path.exists();temp=ROOT/'movies'/f'{name}.{job()}.tmp.mp4';exe=m.ffmpeg_exe()
cmd([exe,'-y','-v','error','-xerror','-framerate','15','-i',folder/'%06d.png','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2',temp])
v=subprocess.run([exe,'-v','error','-xerror','-threads','1','-i',str(temp),'-f','null','-'],capture_output=True,text=True);assert v.returncode==0 and not v.stderr,v.stderr
import imageio_ffmpeg
count,duration=imageio_ffmpeg.count_frames_and_secs(str(temp));assert count==n and abs(duration-n/15)<.15
temp.replace(path);previews=[]
for i in sorted({0,n//2,n-1}):
 out=ROOT/'plots'/f'{name}_decoded_{i:04d}.png';cmd([exe,'-v','error','-xerror','-threads','1','-i',path,'-vf',f'select=eq(n{chr(92)},{i})','-frames:v','1','-threads','1',out])
 raw=m.np.asarray(Image.open(folder/f'{i:06d}.png').convert('RGB'),dtype=float);decoded=m.np.asarray(Image.open(out).convert('RGB'),dtype=float);assert raw.shape==decoded.shape
 mse=float(m.np.mean((raw-decoded)**2));psnr=float('inf') if mse==0 else 10*m.math.log10(255**2/mse);mae=float(m.np.mean(abs(raw-decoded)));assert psnr>=28 and mae<3,(i,psnr,mae)
 previews.append(dict(frame=i,time=float(times[i]),PSNR_dB=psnr,mean_absolute_channel_error=mae,**receipt(out)))
atomic(e/f'{name}_ENCODED.json',dict(utc=now(),job=job(),**receipt(path),frames=count,duration=duration,fps=15,full_decode=True,decoded_previews=previews,frames_directory=str(folder),config=receipt(e/'CONFIG.json'),shell_radius_limits_M=cfg['shell_limits'],visual_review='pending actualencodedframes'))
print(name,'encoded/fullydecoded',count)
